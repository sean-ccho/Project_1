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
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── 스냅샷 경로 ──────────────────────────────────────────────
_SNAPSHOT_DIR = Path("data/paper_trading")
_SP500_SNAPSHOT = _SNAPSHOT_DIR / "sp500_ranked.parquet"
_NASDAQ_SNAPSHOT = _SNAPSHOT_DIR / "nasdaq_ranked.parquet"


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


def load_and_merge_snapshots() -> pd.DataFrame | None:
    """SP500 + NASDAQ ranked_df 스냅샷을 로드하고 합침.

    규칙:
    - 같은 티커가 두 풀에 있으면 바닥반등_적합도 + 모멘텀_적합도 합산이 높은 쪽 유지
    - 어느 한 파일만 있어도 동작
    - 둘 다 없으면 None 반환
    """
    dfs: list[pd.DataFrame] = []

    for path, label in [(_SP500_SNAPSHOT, "SP500"), (_NASDAQ_SNAPSHOT, "NASDAQ")]:
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


def _extract_golden_cross_imminent(df: pd.DataFrame | None) -> list[dict]:
    """merged_df에서 골든크로스 임박/직후 종목을 추출.

    각 TF(일봉/주봉/월봉) 패턴 컬럼에서:
      - "골든크로스임박" → 음수 갭 (단기MA < 장기MA, 5% 이내)
      - "골든크로스" → 양수 갭 (방금 교차, 0~5% 이내만 포함)
    부호로 임박/직후를 구분한다 (표시 레이어가 +/- 그대로 출력).

    정렬:
      1) 다중 TF 우선 (tf_count 내림차순)
      2) 일봉 포함 우선
      3) 갭 0에 가장 가까운 순 (교차 시점 근접도)
    """
    if df is None or df.empty or "티커" not in df.columns:
        return []

    # 패턴 컬럼 ↔ MA갭 컬럼 매핑.
    # gap_col이 dataframe에 없으면 conf_col로 폴백 (임박 conf만 역산 가능, 직후는 미감지).
    # 폴백은 features.py 업데이트 전의 기존 parquet에서 회귀를 막기 위한 안전망.
    pattern_cols = [
        ("일봉패턴", "일봉", "ema_gap_20_50", "일봉_골든크로스_신뢰도"),
        ("주봉패턴", "주봉", "주봉_MA갭", "주봉_골든크로스_신뢰도"),
        ("월봉패턴", "월봉", "월봉_MA갭", "월봉_골든크로스_신뢰도"),
    ]
    result: list[dict] = []

    def _safe_float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _conf_to_neg_gap(conf: float) -> float:
        # patterns.py 임박 공식: conf = 0.7 + (0.05 - |gap|) / 0.05 * 0.25 → |gap| 역산.
        # 결과는 임박 가정으로 항상 음수 반환.
        if conf <= 0:
            return 0.0
        return -max(0.0, 0.05 - 0.2 * (conf - 0.7))

    for _, row in df.iterrows():
        ticker = str(row.get("티커", "")).strip()
        if not ticker:
            continue
        hit_tfs: list[str] = []
        tf_gaps: list[tuple[str, float]] = []
        for col, label, gap_col, conf_col in pattern_cols:
            if col not in df.columns:
                continue
            tokens = [t.strip() for t in str(row.get(col, "")).split(",")]
            is_imminent = "골든크로스임박" in tokens
            is_passed = "골든크로스" in tokens
            if not (is_imminent or is_passed):
                continue
            # 새 갭 컬럼 우선, 없으면 conf 역산 폴백 (임박만)
            if gap_col in df.columns:
                gap = _safe_float(row.get(gap_col))
            elif is_imminent:
                gap = _conf_to_neg_gap(_safe_float(row.get(conf_col)))
            else:
                # 직후인데 갭 컬럼 없음 → 표시 불가, 스킵
                continue
            if is_imminent:
                # 임박은 음수 갭(단기<장기) — 5% 이내 안전 필터
                if gap > 0 or abs(gap) > 0.05:
                    continue
            else:
                # 직후는 양수 갭(단기>장기) — 0~5% 이내만 (HCWB +37% 같은 과이자 제외)
                if gap < 0 or gap > 0.05:
                    continue
            hit_tfs.append(label)
            tf_gaps.append((label, gap))
        if hit_tfs:
            result.append({
                "ticker": ticker,
                "sector": str(row.get("섹터", "")),
                "current_price": _safe_float(row.get("현재가격", row.get("close"))),
                "timeframes": hit_tfs,
                "tf_gaps": tf_gaps,
                "strategy": str(row.get("전략구분", "")),
                "star": str(row.get("매수적합도_표시", "")),
                "vol_ratio": _safe_float(row.get("거래량돌파배수")),
                "vol_ma20": _safe_float(row.get("volume_ma20")),
                "tf_count": len(hit_tfs),
                "has_daily": "일봉" in hit_tfs,
            })

    # 정렬: 다중 TF → 일봉 포함 → 갭 0에 가까운 순
    result.sort(
        key=lambda g: (-g["tf_count"], not g["has_daily"], min(abs(x) for _, x in g["tf_gaps"]))
    )
    return result


# ── 통합 실행 ─────────────────────────────────────────────────


def run_unified_paper_trading() -> None:
    """통합 paper trading 실행 진입점.

    GitHub Actions: main.py → run_full_scan.py → run_paper_trading.py 순으로 실행.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print("[Unified Paper Trading] SP500 + NASDAQ/NYSE 통합 실행 시작")
    print("=" * 60)

    merged_df = load_and_merge_snapshots()
    if merged_df is None or merged_df.empty:
        print("[Unified PT] merged ranked_df가 없어 paper trading을 건너뜁니다.")
        return

    golden_cross = _extract_golden_cross_imminent(merged_df)
    print(f"[Unified PT] 골든크로스임박 종목: {len(golden_cross)}개")

    from paper_trading.engine import run_daily_trading
    from paper_trading.portfolio import load_positions, load_trades
    from paper_trading.sheet_sync import sync_all
    from screener.exporter import send_paper_trading_email
    from data.fetch import fetch_latest_prices

    pt_result = run_daily_trading(merged_df)

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

    # ── 구글 시트 동기화 ──
    positions = load_positions()
    trades = load_trades()
    held_tickers = [p["ticker"] for p in positions]
    prices = fetch_latest_prices(held_tickers) if held_tickers else {}
    sync_all(pt_result, positions, trades, prices)
    print("[Unified PT] 구글 시트 동기화 완료")

    # ── PDF 리포트 생성 ──
    pdf_bytes = None
    if pt_result.get("buys") or pt_result.get("sells"):
        try:
            from paper_trading.report_generator import generate_trading_report
            trades = load_trades()
            pdf_bytes = generate_trading_report(
                result=pt_result,
                positions=positions,
                merged_df=merged_df,
                prices=prices,
                trades=trades,
            )
            if pdf_bytes:
                print(f"[Unified PT] PDF 리포트 생성 완료 ({len(pdf_bytes):,} bytes)")
            else:
                print("[Unified PT] PDF 리포트 생성 실패 — 이메일은 계속 발송")
        except Exception as exc:
            print(f"[Unified PT] PDF 생성 오류 (이메일은 계속 발송): {exc}")

    # ── 이메일 알림 ──
    # 재평가중 포지션도 있으므로 prices 전달 (현재가 컬럼 + 수익률 계산용)
    all_held_tickers = [p["ticker"] for p in positions]
    if all_held_tickers:
        prices = fetch_latest_prices(all_held_tickers)

    # ── 시장 분석 데이터 enrichment (후보 / 매수 / 보유) ──
    try:
        from data.fetch import fetch_analyst_data, fetch_latest_news as _fetch_news

        top5 = pt_result.get("selection_debug", {}).get("top5", [])
        buys = pt_result.get("buys", [])

        # 분석 대상 티커 수집 (후보 + 매수 + 보유 + 매도 + 골든크로스임박)
        analysis_tickers: list[str] = []
        for item in top5 + buys + sells + golden_cross:
            t = item.get("ticker", "")
            if t and t not in analysis_tickers:
                analysis_tickers.append(t)
        for p in positions:
            t = p.get("ticker", "")
            if t and t not in analysis_tickers:
                analysis_tickers.append(t)

        if analysis_tickers:
            print(f"[시장분석] {len(analysis_tickers)}개 종목 데이터 수집 중...")
            analyst_data = fetch_analyst_data(analysis_tickers)
            news_data = _fetch_news(analysis_tickers, max_items=3)

            # 내부자 거래 90일 요약 (openinsider, 표시용)
            insider_map: dict[str, str] = {}
            try:
                from screener.insider import fetch_insider_snapshots

                insider_df = fetch_insider_snapshots(analysis_tickers)
                if not insider_df.empty:
                    insider_map = {
                        str(r["티커"]).upper().strip(): r["내부자(90일)"]
                        for _, r in insider_df.iterrows()
                    }
            except Exception as _exc:
                print(f"[시장분석] 내부자 정보 조회 실패 (무시): {_exc}")

            # 기술적 지표 룩업 (merged_df에서 추출)
            tech_cols = {
                "rsi": ["RSI", "rsi"],
                "adx": ["ADX", "adx"],
                "bb_pband": ["bb_pband", "bollinger_pband", "BB_pband"],
                "vol_z": ["vol_z_20", "volume_z", "vol_z"],
                "pos_52w": ["pos_52w", "position_52w", "52w_pos"],
            }

            def _get_tech(ticker: str) -> dict:
                tech: dict = {}
                if merged_df is None or ticker not in merged_df.columns.get_level_values(0):
                    return tech
                try:
                    row = merged_df[ticker].iloc[-1]
                    for key, candidates in tech_cols.items():
                        for col in candidates:
                            if col in row.index and pd.notna(row[col]):
                                tech[key] = float(row[col])
                                break
                except Exception:
                    pass
                return tech

            # 각 dict에 market_analysis 키 주입
            def _enrich(item: dict) -> None:
                ticker = item.get("ticker", "")
                item["market_analysis"] = {
                    "analyst": analyst_data.get(ticker),
                    "tech": _get_tech(ticker),
                    "news_html": news_data.get(ticker, ""),
                    "insider": insider_map.get(str(ticker).upper().strip(), "—"),
                }

            # 표시용 insider 한 줄 요약만 필요한 항목 (매도/골든크로스)
            def _attach_insider_only(item: dict) -> None:
                ticker = item.get("ticker", "")
                item["insider"] = insider_map.get(str(ticker).upper().strip(), "—")

            # 골든크로스 티커의 CCS: candidate selector가 계산한 점수 dict에서 lookup
            all_scores = pt_result.get("selection_debug", {}).get("all_scores", {})

            for item in top5:
                _enrich(item)
            for item in buys:
                _enrich(item)
            for p in positions:
                _enrich(p)
            for s in sells:
                _attach_insider_only(s)
            for g in golden_cross:
                _attach_insider_only(g)
                score = all_scores.get(g.get("ticker", ""))
                g["ccs"] = score["ccs"] if score else None

            if "selection_debug" in pt_result:
                pt_result["selection_debug"]["top5"] = top5
            pt_result["buys"] = buys
            pt_result["sells"] = sells
            print("[시장분석] enrichment 완료")
    except Exception as exc:
        print(f"[시장분석] 데이터 수집 실패 (이메일은 계속 발송): {exc}")

    # ── 보유 종목 거래량 배수 주입 (merged_df 평면 snapshot에서; 후보/매수와 동일 출처) ──
    # merged_df는 티커가 행(row)인 평면 DataFrame이므로 "티커" 컬럼으로 필터한다.
    if merged_df is not None and "티커" in merged_df.columns:
        def _vol_float(val) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        for p in positions:
            rows = merged_df[merged_df["티커"] == p.get("ticker", "")]
            if not rows.empty:
                r = rows.iloc[0]
                p["vol_ratio"] = _vol_float(r.get("거래량돌파배수"))
                p["vol_ma20"] = _vol_float(r.get("volume_ma20"))

    send_paper_trading_email(
        pt_result, positions, pdf_attachment=pdf_bytes, prices=prices,
        golden_cross=golden_cross,
    )
    print("=" * 60)
    print("[Unified Paper Trading] 완료")
    print("=" * 60)
