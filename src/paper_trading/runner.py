"""통합 페이퍼 트레이딩 실행 모듈.

SP500(main.py) + NASDAQ/NYSE(run_full_scan.py) 두 풀의 ranked_df를
parquet 스냅샷으로 저장한 뒤, 합쳐서 단 한 번 paper trading을 실행한다.

흐름:
  main.py        → save_ranked_snapshot(df, "sp500")
  run_full_scan  → save_ranked_snapshot(df, "nasdaq")
  run_paper_trading.py → run_unified_paper_trading()
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from paper_trading.golden_cross import extract_golden_cross as _extract_golden_cross_imminent

logger = logging.getLogger(__name__)

# ── 스냅샷 경로 ──────────────────────────────────────────────
_SNAPSHOT_DIR = Path("data/paper_trading")
_SP500_SNAPSHOT = _SNAPSHOT_DIR / "sp500_ranked.parquet"
_NASDAQ_SNAPSHOT = _SNAPSHOT_DIR / "nasdaq_ranked.parquet"

# 스크리너가 쓴 마지막 일봉 날짜 (main.py / run_full_scan.py에서 기록)
BAR_DATE_COL = "_bar_date"

# PT-1 후보 유니버스: "all" = S&P 500 + 나스닥/NYSE 전체 스캔 (기존), "sp500" = S&P 500 스냅샷만,
# "ex_sp500" = all 에서 S&P 500 구성종목을 뺀 것 (지수 행 SPY·QQQ·IWM 은 남는다 — 레짐 판단이 SPY 행을 쓴다).
# 연구·백테스트로 검증한 범위는 S&P 500뿐이다. 환경변수 PT1_UNIVERSE 로 바꿀 수 있다.
PT1_UNIVERSE = os.environ.get("PT1_UNIVERSE", "all")

# PT-1 규칙 계좌들 (소형주가 값을 하는지 실거래로 비교하는 병행 계좌):
#   pt1s = PT-1과 같은 규칙, 후보만 S&P 500 / pt1n = 같은 규칙, 후보만 S&P 500 밖 나스닥/NYSE 종목
#   → PT-1 후보 = PT-1S 후보 ∪ PT-1N 후보 (서로 겹치지 않음)
PT1_ACCOUNTS: dict[str, dict[str, Any]] = {
    "pt1": {"label": "PT-1", "subdir": "", "universe": None, "email": True,
            "tabs": None,  # None = config 기본 탭 (페이퍼_거래로그 등)
            "dollar_tab": "페이퍼_계좌"},  # $5,000 계좌 (pt1_account)
    "pt1s": {"label": "PT-1S (S&P 500만)", "subdir": "pt1s", "universe": "sp500", "email": False,
             "tabs": {"log": "페이퍼S_거래로그", "positions": "페이퍼S_포지션현황", "summary": "페이퍼S_성과요약"},
             "dollar_tab": "페이퍼S_계좌"},
    "pt1n": {"label": "PT-1N (S&P 500 밖)", "subdir": "pt1n", "universe": "ex_sp500", "email": False,
             "tabs": {"log": "페이퍼N_거래로그", "positions": "페이퍼N_포지션현황", "summary": "페이퍼N_성과요약"},
             "dollar_tab": "페이퍼N_계좌"},
}


def snapshot_bar_date(df: pd.DataFrame | None) -> str | None:
    """스냅샷의 일봉 날짜(YYYY-MM-DD). 컬럼이 없으면 None."""
    if df is None or df.empty or BAR_DATE_COL not in df.columns:
        return None
    values = df[BAR_DATE_COL].dropna().astype(str)
    values = values[values.str.len() == 10]
    return str(values.max()) if not values.empty else None


# ── 저장 ─────────────────────────────────────────────────────


def save_ranked_snapshot(df: pd.DataFrame, source: str) -> bool:
    """ranked_df (prepare_export_dataframe 이전 원본)를 parquet으로 저장.

    Args:
        df: _last_ranked DataFrame — 모든 내부 컬럼 포함
        source: "sp500" 또는 "nasdaq"

    Returns:
        저장 성공 여부
    """
    try:
        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = _SP500_SNAPSHOT if source == "sp500" else _NASDAQ_SNAPSHOT

        df_save = df.copy()

        # 회사 건전성 점수 + 시가총액 컬럼 추가 (display-only, score/CCS에 영향 없음)
        try:
            from screener.fundamentals import calculate_health_score, format_market_cap
            health = df_save.apply(calculate_health_score, axis=1, result_type="expand")
            df_save["health_score"] = health[0]
            df_save["[헬스체크]"] = health[1]
            df_save["[시총]"] = df_save.apply(format_market_cap, axis=1)
        except Exception as exc:
            logger.warning(f"[Runner] 헬스체크/시총 계산 실패 (무시): {exc}")

        # SEC 공시 경고 (display-only) — [공시] = 🔴·🟠 한 줄 (메일·시트), sec_flags = 전체 JSON (웹)
        try:
            from screener.sec_filings import fetch_flags, flags_json, format_flags
            if "티커" in df_save.columns:
                flags = fetch_flags(df_save["티커"].astype(str).tolist())
                if flags:
                    keys = df_save["티커"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
                    df_save["[공시]"] = [format_flags(flags.get(k, [])) for k in keys]
                    df_save["sec_flags"] = [flags_json(flags[k]) if k in flags else "" for k in keys]
        except Exception as exc:
            logger.warning(f"[Runner] SEC 공시 경고 실패 (무시): {exc}")

        # 공매도 (display-only) — FINRA 잔고·되사기 일수 + yfinance 유통주식 → [공매도] 한 줄, short_high = 주황 표시
        try:
            from screener.short_squeeze import is_high, short_data, short_line
            sd = short_data()
            if sd is not None and "티커" in df_save.columns:
                pcts, dtcs, lines, highs, changes = [], [], [], [], []
                for _, r in df_save.iterrows():
                    got = sd.get(str(r["티커"]))
                    fl = pd.to_numeric(r.get("fund_float_shares"), errors="coerce")
                    pct = got["si"] / fl if got and fl and fl > 0 else pd.to_numeric(r.get("fund_short_pct_float"), errors="coerce")
                    pct = float(pct) if pd.notna(pct) and pct < 3 else None
                    dtc = got["dtc"] if got else None
                    pcts.append(pct); dtcs.append(dtc); changes.append(got["si_change"] if got else None)
                    lines.append(short_line(pct, dtc, sd.settle if got else None))
                    highs.append(is_high(pct, dtc))
                df_save["short_pct"], df_save["short_dtc"], df_save["short_change"] = pcts, dtcs, changes
                df_save["short_settle"] = sd.settle
                df_save["[공매도]"], df_save["short_high"] = lines, highs
        except Exception as exc:
            logger.warning(f"[Runner] 공매도 표시 실패 (무시): {exc}")

        # parquet은 object dtype 컬럼에 혼합 타입이 있으면 오류가 날 수 있으므로
        # 저장 전에 object 컬럼을 string으로 변환
        for col in df_save.select_dtypes(include="object").columns:
            df_save[col] = df_save[col].astype(str)

        df_save.to_parquet(path, index=True, engine="pyarrow")
        logger.info(
            f"[Runner] {source.upper()} ranked snapshot 저장: "
            f"{len(df_save)}개 종목 → {path}"
        )
        print(
            f"[Paper Trading] {source.upper()} ranked_df 스냅샷 저장 완료 "
            f"({len(df_save)}개 종목)"
        )
        return True
    except Exception as exc:
        logger.error(f"[Runner] {source} ranked snapshot 저장 실패: {exc}")
        print(f"[Paper Trading] {source.upper()} 스냅샷 저장 실패: {exc}")
        return False


# ── 로드 & 합치기 ─────────────────────────────────────────────


def load_and_merge_snapshots(universe: str | None = None) -> pd.DataFrame | None:
    """SP500 + NASDAQ ranked_df 스냅샷을 로드하고 합침.

    규칙:
    - 같은 티커가 두 풀에 있으면 바닥반등_적합도 + 모멘텀_적합도 합산이 높은 쪽 유지
    - 어느 한 파일만 있어도 동작
    - 둘 다 없으면 None 반환
    - universe="sp500"(또는 PT1_UNIVERSE)이면 NASDAQ/NYSE 스냅샷은 읽지 않는다
    - universe="ex_sp500"이면 둘 다 합친 뒤 S&P 500 구성종목(data.sp500_tickers, 매월 갱신)을 뺀다
    """
    universe = (universe or PT1_UNIVERSE).lower()
    combined = _load_and_merge(universe)
    if combined is None or universe != "ex_sp500" or "티커" not in combined.columns:
        return combined
    from data.sp500_tickers import SP500_TICKERS

    sp500 = set(SP500_TICKERS)
    kept = combined[~combined["티커"].astype(str).isin(sp500)].reset_index(drop=True)
    print(f"[Unified PT] PT1_UNIVERSE=ex_sp500 — S&P 500 구성종목 {len(combined) - len(kept)}개 제외 → {len(kept)}개")
    return kept


def _load_and_merge(universe: str) -> pd.DataFrame | None:
    """스냅샷 로드·합치기 (load_and_merge_snapshots 본체). universe 는 소문자."""
    sources = [(_SP500_SNAPSHOT, "SP500")]
    if universe != "sp500":
        sources.append((_NASDAQ_SNAPSHOT, "NASDAQ"))
    else:
        print("[Unified PT] PT1_UNIVERSE=sp500 — S&P 500 스냅샷만 사용")
    dfs: list[pd.DataFrame] = []

    for path, label in sources:
        if path.exists():
            try:
                df = pd.read_parquet(path, engine="pyarrow")
                logger.info(f"[Runner] {label} snapshot 로드: {len(df)}개 종목")
                print(f"[Unified PT] {label} snapshot 로드: {len(df)}개 종목")
                dfs.append(df)
            except Exception as exc:
                logger.error(f"[Runner] {label} snapshot 로드 실패: {exc}")
                print(f"[Unified PT] {label} snapshot 로드 실패: {exc}")
        else:
            logger.warning(f"[Runner] {label} snapshot 없음: {path}")
            print(f"[Unified PT] {label} snapshot 없음 ({path}) — 건너뜀")

    if not dfs:
        logger.error("[Runner] 로드된 snapshot이 없습니다.")
        return None

    # 한쪽 스크리너가 실패하면 이전 날짜 스냅샷이 남는다 → 최신 일봉 날짜 스냅샷만 사용
    dated = [snapshot_bar_date(d) for d in dfs]
    known = [d for d in dated if d]
    if known:
        newest = max(known)
        if any(d != newest for d in dated):
            print(f"[Unified PT] 일봉 날짜가 다른 스냅샷 제외: {dated} → {newest}만 사용")
            dfs = [d for d, bd in zip(dfs, dated) if bd == newest]

    if len(dfs) == 1:
        return dfs[0]

    # ── 두 풀 합치기 ──
    combined = pd.concat(dfs, ignore_index=True)

    if "티커" not in combined.columns:
        return combined

    # 점수 계산 (string → float 안전 변환)
    def _score(row: pd.Series) -> float:
        def _to_float(val) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        b = _to_float(row.get("바닥반등_적합도", 0))
        m = _to_float(row.get("모멘텀_적합도", 0))
        return b + m

    combined["_merge_score"] = combined.apply(_score, axis=1)
    combined = (
        combined
        .sort_values("_merge_score", ascending=False)
        .drop_duplicates(subset=["티커"], keep="first")
        .drop(columns=["_merge_score"])
    )

    # 매수적합도 기준 정렬 복원
    if "매수적합도" in combined.columns:
        combined = combined.sort_values("매수적합도", ascending=False)

    print(
        f"[Unified PT] 합산 완료: {len(combined)}개 종목 "
        f"(SP500 {len(dfs[0])} + NASDAQ {len(dfs[1])}개, 중복 제거 후)"
    )
    return combined.reset_index(drop=True)


# ── 통합 실행 ─────────────────────────────────────────────────


def run_unified_paper_trading(dry_run: bool = False, as_of: str | None = None, account: str = "pt1") -> None:
    """통합 paper trading 실행 진입점 (PT-1 규칙 계좌: pt1, pt1s).

    GitHub Actions: main.py → run_full_scan.py → run_paper_trading.py 순으로 실행.

    Args:
        dry_run: True면 임시 폴더에서만 매매하고 시트·이메일·상태 기록을 하지 않는다.
        as_of: 거래일(YYYY-MM-DD) 강제 지정. 없으면 스냅샷의 일봉 날짜를 쓴다.
        account: PT1_ACCOUNTS 키. pt1s는 S&P 500 후보만, 별도 폴더·시트 탭, 메일 없음.
    """
    acct = PT1_ACCOUNTS[account]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print(f"[Unified Paper Trading] {acct['label']} 실행 시작" + (" (DRY-RUN)" if dry_run else ""))
    print("=" * 60)

    merged_df = load_and_merge_snapshots(acct["universe"])
    if merged_df is None or merged_df.empty:
        print("[Unified PT] merged ranked_df가 없어 paper trading을 건너뜁니다.")
        return

    from paper_trading.market_date import check_run_guard, mark_processed, market_today
    from screener.config import PAPER_TRADING_DATA_DIR

    data_dir = Path(PAPER_TRADING_DATA_DIR) / acct["subdir"] if acct["subdir"] else Path(PAPER_TRADING_DATA_DIR)
    bar_date = as_of or snapshot_bar_date(merged_df)
    report_only = False  # True면 매매·상태 기록·시트 동기화 없이 현재 보유 현황 리포트만 발송
    if dry_run:
        if not bar_date:
            bar_date = str(market_today())
            print(f"[Unified PT] 스냅샷에 일봉 날짜가 없어 {bar_date}로 가정 (DRY-RUN)")
    else:
        ok, reason = check_run_guard(bar_date, data_dir)
        if not ok:
            print(f"[Unified PT] 건너뜀: {reason} — 매매 없이 일일 리포트만 발송")
            report_only = True
    print(f"[Unified PT] 거래일(일봉 날짜): {bar_date}")

    golden_cross = _extract_golden_cross_imminent(merged_df)
    print(f"[Unified PT] 골든크로스임박 종목: {len(golden_cross)}개")

    from paper_trading.engine import run_daily_trading
    from paper_trading.portfolio import load_positions, load_trades
    from paper_trading.sheet_sync import sync_all
    from screener.exporter import send_paper_trading_email
    from data.fetch import fetch_latest_prices

    work_dir = data_dir
    if dry_run:
        work_dir = Path(tempfile.mkdtemp(prefix="pt1_dryrun_"))
        for name in ("positions.json", "trades.json"):
            if (data_dir / name).exists():
                shutil.copy2(data_dir / name, work_dir / name)

    if report_only:
        pt_result = {
            "date": bar_date or str(market_today()), "sells": [], "buys": [],
            "skipped": "신규일봉없음", "holdings": len(load_positions(data_dir)),
        }
    else:
        pt_result = run_daily_trading(merged_df, data_dir=work_dir, today=bar_date)
        if not dry_run:
            mark_processed(data_dir, bar_date, account=account)

    sells: list[dict[str, Any]] = pt_result.get("sells", [])
    buys: list[dict[str, Any]] = pt_result.get("buys", [])
    holdings: int = pt_result.get("holdings", 0)

    print(
        f"[Unified PT] 매도={len(sells)} 매수={len(buys)} 보유={holdings}종목"
    )

    # 선정 디버그 출력
    debug = pt_result.get("selection_debug", {})
    if debug:
        regime = debug.get("regime", "?")
        rejections = debug.get("rejections", {})
        top5 = debug.get("top5", [])
        skipped = pt_result.get("skipped")
        print(f"[Unified PT] 레짐={regime} | 필터={rejections}")
        if skipped:
            print(f"[Unified PT] 스킵 사유: {skipped}")
        if top5:
            print("[Unified PT] Top5 후보:")
            for s in top5:
                print(
                    f"  {s['ticker']:6s} CCS={s['ccs']:.4f} "
                    f"전략={s['strategy'][:8]} 섹터={s['sector']}"
                )

    if dry_run:
        print(f"[Unified PT] DRY-RUN 종료 — 결과 파일: {work_dir} (시트·이메일·상태 기록 안 함)")
        return

    # ── 구글 시트 동기화 ──
    positions = load_positions(data_dir)
    trades = load_trades(data_dir)
    held_tickers = [p["ticker"] for p in positions]
    prices = fetch_latest_prices(held_tickers) if held_tickers else {}
    if not report_only:
        sync_all(pt_result, positions, trades, prices, tabs=acct["tabs"])
        print("[Unified PT] 구글 시트 동기화 완료")

    # ── $5,000 계좌: 같은 거래를 가상 자본으로 다시 계산 (매매와 무관, 표시만) ──
    dollar_html = ""
    dollar_holdings: list[dict] = []
    account_positions: dict[str, list[dict]] = {acct["label"]: positions}  # 계좌 이름 → 보유 position dict
    try:
        from paper_trading.pt1_account import build_accounts, held_rows, section_html, sync_sheet

        books = {acct["label"]: (trades, positions)}
        if acct["email"]:  # PT-1 메일에는 PT-1S·PT-1N 계좌도 함께 보여 준다
            for other_key in ("pt1s", "pt1n"):
                other = PT1_ACCOUNTS[other_key]
                other_dir = Path(PAPER_TRADING_DATA_DIR) / other["subdir"]
                account_positions[other["label"]] = load_positions(other_dir)
                books[other["label"]] = (load_trades(other_dir), account_positions[other["label"]])
        dollars = build_accounts(books)
        if not report_only:
            sync_sheet(dollars[acct["label"]], acct["label"], acct["dollar_tab"])
            print(f"[Unified PT] $5,000 계좌 탭 '{acct['dollar_tab']}' 동기화 완료")
        dollar_html = section_html(list(dollars.items()))
        dollar_holdings = held_rows(list(dollars.items()))
    except Exception as exc:  # noqa: BLE001 — 매매·메일은 계속
        print(f"[Unified PT] $5,000 계좌 계산 실패 (무시): {exc}")

    if not acct["email"]:
        print(f"[Unified PT] {acct['label']}: 메일 없음 (PT-1 메일의 SPY 대비 표에 함께 표시)")
        return

    # ── 이메일 알림 ──
    # 재평가중 포지션도 있으므로 prices 전달 (현재가 컬럼 + 수익률 계산용)
    all_held_tickers = [p["ticker"] for p in positions]
    if all_held_tickers:
        prices = fetch_latest_prices(all_held_tickers)

    # ── 내부자 거래 90일 요약 (openinsider, 표시용) — 후보 / 매수 / 매도 / 보유 / 골든크로스 ──
    top5 = pt_result.get("selection_debug", {}).get("top5", [])
    # PT-1S 보유 종목도 합친 보유 표에 나오므로 내부자·거래량을 함께 채운다
    other_held = [p for label, ps in account_positions.items() if label != acct["label"] for p in ps]
    shown = top5 + buys + sells + golden_cross + positions + other_held
    try:
        from screener.insider import fetch_insider_snapshots

        insider_tickers = list(dict.fromkeys(i.get("ticker", "") for i in shown if i.get("ticker")))
        insider_map: dict[str, str] = {}
        if insider_tickers:
            insider_df = fetch_insider_snapshots(insider_tickers)
            if not insider_df.empty:
                insider_map = {
                    str(r["티커"]).upper().strip(): r["내부자(90일)"]
                    for _, r in insider_df.iterrows()
                }
        for item in shown:
            item["insider"] = insider_map.get(str(item.get("ticker", "")).upper().strip(), "—")
    except Exception as exc:
        print(f"[내부자] 조회 실패 (이메일은 계속 발송): {exc}")

    # 골든크로스 티커의 CCS: candidate selector가 계산한 점수 dict에서 lookup
    all_scores = pt_result.get("selection_debug", {}).get("all_scores", {})
    for g in golden_cross:
        score = all_scores.get(g.get("ticker", ""))
        g["ccs"] = score["ccs"] if score else None

    # ── 보유 종목 거래량 배수 주입 (merged_df 평면 snapshot에서; 후보/매수와 동일 출처) ──
    # merged_df는 티커가 행(row)인 평면 DataFrame이므로 "티커" 컬럼으로 필터한다.
    if merged_df is not None and "티커" in merged_df.columns:
        def _vol_float(val) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        for p in positions + other_held:
            rows = merged_df[merged_df["티커"] == p.get("ticker", "")]
            if not rows.empty:
                r = rows.iloc[0]
                p["vol_ratio"] = _vol_float(r.get("거래량돌파배수"))
                p["vol_ma20"] = _vol_float(r.get("volume_ma20"))

    # 합친 보유 표: $5,000 계좌 줄마다 그 계좌 position 의 전략·섹터·고점·내부자·거래량을 붙인다
    for row in dollar_holdings:
        pos = next((p for p in account_positions.get(row["label"], [])
                    if p.get("ticker") == row["ticker"] and str(p.get("entry_date")) == row["entry_date"]), {})
        for key in ("strategy", "sector", "highest_price", "trailing_stop_override", "defer_count", "last_defer_date",
                    "insider", "vol_ratio", "vol_ma20"):
            if key in pos:
                row[key] = pos[key]

    send_paper_trading_email(
        pt_result, positions, prices=prices,
        golden_cross=golden_cross, extra_html=dollar_html,
        dollar_holdings=dollar_holdings,
    )
    print("=" * 60)
    print("[Unified Paper Trading] 완료")
    print("=" * 60)
