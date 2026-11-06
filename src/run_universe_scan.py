#!/usr/bin/env python3
"""전체 분석 — 필터 없이 시장 전체를 분석만 한다 (표시 전용).

분석 탭([SP500 분석]·[NASDAQ/NYSE 분석])은 유동성 컷·사전 필터로 수백 종목만 남는다. 여기서는 거르지 않는다:
- [SP500 전체]       S&P 500 전 종목 + SPY·QQQ·IWM (SPY 는 시장 레짐 판단용, 웹 목록에는 안 보임)
- [NASDAQ/NYSE 전체] 나스닥·NYSE 상장 보통주 전부에서 S&P 500 을 뺀 것 (두 탭이 시장을 나눈다)
- [신규상장]          일봉 120개 미만이라 점수를 못 내는 종목 — 간단 지표만
- 번외 (웹만)        두 시장 밖이지만 늘 보고 싶은 종목 (EXTRA_TICKERS) — 같은 계산을 따로 돌려 extra.json 에만 싣는다
- [캐나다 전체]       TSX · TSX Venture 상장 보통주 전부 (--market ca, 별도 워크플로 canada-scan.yml → canada-data 브랜치 ·
                     2026-11-05). 미국 두 시장과 따로 돌고 ca.json 에 싣는다 — 아래 canada_market 참고
계산은 분석 탭과 같은 함수(피처 → 중립화 → 신호·적합도)이고 유동성 필터만 뺀다. 다만 중립화 z-score·IC 가중치가
비교 종목 집합에 따라 달라서, 같은 종목도 분석 탭과 점수가 조금 다를 수 있다.
극점 모델(저점·고점확률)은 돌리지 않는다 — 분석 탭에서도 점수를 매긴 뒤에 붙어 점수·시트·웹에 안 쓰이고, 종목마다 과거 2년을
3일 간격으로 피처를 다시 계산해서 5천 종목이면 하루가 넘게 걸린다 (2026-10-20 로컬: S&P 500 만 10분 넘게).
SEC 공시·공매도·내부자·ISB·SQZ·골든크로스는 웹 전체차트와 같은 정보를 붙인다.

매매·이메일·차트 캡처·PT 스냅샷(data/paper_trading)·main 커밋은 건드리지 않는다. CI 는 별도 워크플로
(.github/workflows/universe-scan.yml) 로 돌린다 — 일일 실행이 이미 6시간 한도에 가까워서.

사용:
    PYTHONPATH=.:src python src/run_universe_scan.py --out-dir "$RUNNER_TEMP/universe"
    # 로컬 (시트 자격증명 없음, 캐시 안 씀): CI=1 PYTHONPATH=.:src python src/run_universe_scan.py --no-sheet --limit 200 --out-dir /tmp/u
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.fetch import fetch_chart_prices, fetch_ohlcv
from data.sp500_tickers import SP500_TICKERS
from data.ticker_fetcher import fetch_listings
from screener.config import (
    COMPANY_NAME_MAP,
    INDICES,
    SECTOR_ETFS,
    SECTOR_ROTATION_ENABLED,
    TRADINGVIEW_TICKER_MAP,
)
from screener.features import compute_features_snapshot
from screener.fundamentals_guard import SNAPSHOT_DIR, fetch_with_retry, fill_from_previous_snapshot
from screener.processing import apply_neutralization
from screener.signals import attach_signals_and_sort
from web_export.market_summary import price_stats

SHEET_SP500 = "[SP500 전체]"
SHEET_NASDAQ = "[NASDAQ/NYSE 전체]"
SHEET_NEW = "[신규상장]"
MIN_BARS = 120  # features.compute_features_for_ticker 가 계산을 건너뛰는 기준과 같다
SEC_BUDGET_SEC = 1500  # 5천 종목 × 0.13초 ≈ 11분 + 여유 (실전 스냅샷은 240초)
# Yahoo 종목 정보가 막혔을 때 쉬는 초 — 실전(누적 약 18분)보다 길게 (누적 약 88분, 잡 한도 300분 안).
# 2026-10-20 로컬: 약 1,500건마다 막히고 5~10분 쉬면 풀렸다 → 5천 종목이면 서너 번 막힌다 (한 번 쓴 대기는 다시 안 쓴다)
INFO_RETRY_WAITS = (60, 120, 300, 600, 600, 900, 900, 900, 900)
# fundamentals_guard 가 읽는 이름 (SNAPSHOT_FILES)
SNAPSHOT_NAMES = {"sp500": "sp500_ranked.parquet", "nasdaq": "nasdaq_ranked.parquet", "ca": "ca_ranked.parquet"}
SNAPSHOT_EXTRA = ("티커", "섹터", "next_earnings_date")
NEW_COLUMNS = [  # [신규상장] 시트 칸
    "티커", "회사", "시장", "현재가격", "첫거래일", "상장후거래일", "상장후수익률", "5일수익률", "20일수익률", "RSI",
    "거래량돌파배수", "상장후위치", "최근20일평균거래대금", "섹터", "[시총]", "[헬스체크]",
    "내부자(90일)", "[공시]", "[공매도]", "SQZ",
]
NEW_PERCENT = ("상장후수익률", "5일수익률", "20일수익률", "상장후위치")
EXTRA_SHEET_COLUMNS = ("[공시]", "[공매도]", "ISB", "SQZ")
STYLE_SHEET_COLUMNS = ("경기 성격", "매출 지역", "미국 매출 비중")   # 맨 끝 (2026-10-29 — data.industry_style)
EARN_SHEET_COLUMN = "실적 움직임"   # 그 뒤 맨 끝 (2026-10-29 — screener.earnings_moves, 최근 8번 평균·최악·보통 시간)
# 번외 — S&P 500 · 나스닥·NYSE 밖이지만 늘 보고 싶은 종목 (main.py [SP500 차트분석] 고정 종목과 같다 · 2026-11-05).
# 웹 전체분석 '번외' 탭 · 전체 탭에만 싣는다 (extra.json · stocks/). 시트 · 오늘 탭 시장 숫자(market.json) · 종목 상세 카드 ·
# 다음 날 보충 스냅샷에는 넣지 않는다. 이미 두 시장에 있는 티커는 거기서 보이니 건너뛴다.
EXTRA_TICKERS = ("NBM.V",)
EXTRA_KEY, EXTRA_LABEL = "extra", "번외"
# 캐나다 상장 접미사 → TradingView 거래소 (실시간 차트 심볼 — 접두어 없이 'ABC' 만이면 다른 거래소 종목이 열릴 수 있다)
CA_TV_EXCHANGE = {".V": "TSXV", ".TO": "TSX", ".CN": "CSE", ".NE": "NEO"}
# 캐나다 — TSX · TSXV 상장 보통주 전부 (data.ca_listings, 2026-11-05). 웹 ca.json 의 종목 market 은 'tsx' · 'tsxv'
CA_KEY, CA_LABEL, SHEET_CA = "ca", "캐나다", "[캐나다 전체]"
# 캐나다 시트에서 빼는 칸 — 미국 제도 · 미국 자료라 늘 비어 있다 (SEC 공시 · SEC 10-K 미국 매출 비중 · SEC 8-K 실적 반응)
CA_SHEET_DROP = ("[공시]", "미국 매출 비중", EARN_SHEET_COLUMN)
CA_DTC_DAYS = 20  # 되사기 일수 = 공매도 잔고 ÷ 최근 이만큼 거래일 평균 거래량


def _norm(t: Any) -> str:
    return str(t).strip().upper().replace(".", "-")


def _log(msg: str, start: float) -> None:
    print(f"[전체분석] {msg} ({(time.perf_counter() - start) / 60:.1f}분)", flush=True)


# ── 종목 ─────────────────────────────────────────────────────


def split_universe(listed: list[str], sp500: list[str], indices: list[str]) -> tuple[list[str], list[str]]:
    """(S&P 500 + 지수 ETF, 상장 보통주 − S&P 500). 티커는 야후 형식(BRK-B), 순서·중복 정리."""
    sp = list(dict.fromkeys(_norm(t) for t in [*sp500, *indices]))
    sp_set = set(sp)
    rest = [t for t in dict.fromkeys(_norm(t) for t in listed) if t not in sp_set]
    return sp, rest


def tv_of(ticker: str, tv_map: dict[str, str]) -> str:
    """TradingView 심볼 — FTP 거래소 접두어가 있으면 그것 (NYSE:BRK.B), 없으면 캡처와 같은 규칙 (BRK-B → BRK.B)."""
    from screener.tv_symbol import tv_symbol

    return tv_map.get(ticker) or tv_symbol(ticker, TRADINGVIEW_TICKER_MAP)


def is_foreign(ticker: str) -> bool:
    """미국 밖 상장 (캐나다 접미사 .V·.TO·.CN·.NE) — SEC 공시 · openinsider 내부자 · FINRA 공매도가 없는 종목."""
    return str(ticker).strip().upper().endswith(tuple(CA_TV_EXCHANGE))


def extra_tv(ticker: str, tv_map: dict[str, str]) -> str:
    """번외 종목 TradingView 심볼 — 캐나다 상장은 거래소를 붙인다 (ABC.V → TSXV:ABC, 이름이 다르면 TRADINGVIEW_TICKER_MAP)."""
    t = str(ticker).strip().upper()
    for suffix, exch in CA_TV_EXCHANGE.items():
        if t.endswith(suffix):
            base = TRADINGVIEW_TICKER_MAP.get(t) or t[: -len(suffix)]
            return base if ":" in base else f"{exch}:{base.replace('-', '.')}"
    return tv_of(t, tv_map)


# ── 분석 ─────────────────────────────────────────────────────


def strong_sectors() -> set[str] | None:
    """강한 섹터 (main.py·run_full_scan.py 와 같은 규칙). 끄여 있거나 못 받으면 None — 표시만 빠진다."""
    if not SECTOR_ROTATION_ENABLED:
        return None
    from screener.sector_rotation import get_strong_sectors

    tickers = list(SECTOR_ETFS.values()) + ["SPY"]
    raw = fetch_ohlcv(tickers, period="1y")
    if raw.empty:
        return None
    etf_data = {t: raw[t].dropna(how="all") for t in raw.columns.levels[0] if t in tickers}
    spy = etf_data.get("SPY", pd.DataFrame())
    if spy.empty:
        return None
    strong = set(get_strong_sectors(etf_data, spy))
    print(f"[전체분석] 강한 섹터: {sorted(strong) if strong else '없음'}")
    return strong


def tag_sectors(df: pd.DataFrame, strong: set[str] | None) -> pd.DataFrame:
    if strong is None or "섹터" not in df.columns:
        return df
    from screener.sector_rotation import is_strong_or_unknown

    out = df.copy()
    out["in_strong_sector"] = out["섹터"].apply(lambda s: is_strong_or_unknown(s, strong))
    out["섹터강도"] = out["in_strong_sector"].map(lambda ok: "✅ 강함" if ok else "❌ 약함")
    return out


def add_health(df: pd.DataFrame) -> pd.DataFrame:
    """[헬스체크]·[시총] (PT 스냅샷 저장과 같은 계산). 실패하면 그대로."""
    try:
        from screener.fundamentals import calculate_health_score, format_market_cap

        out = df.copy()
        health = out.apply(calculate_health_score, axis=1, result_type="expand")
        out["health_score"], out["[헬스체크]"] = health[0], health[1]
        out["[시총]"] = out.apply(format_market_cap, axis=1)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[전체분석] 헬스체크 계산 실패 (건너뜀): {exc}")
        return df


def analyze_market(tickers: list[str], label: str, names: dict[str, str], strong: set[str] | None,
                   snapshot_dir: Path) -> dict[str, Any]:
    """한 시장 분석 → {ranked, young(신규 상장 일봉), young_fund, bar_date, no_data}. 유동성 필터 없음."""
    from paper_trading.market_date import is_bar_complete

    t0 = time.perf_counter()
    df = fetch_ohlcv(tickers, period="5y")
    bar_date = str(pd.Timestamp(df.index.max()).date()) if not df.empty else ""
    if bar_date and not is_bar_complete(bar_date):
        print(f"[{label}] {bar_date} 일봉은 장중 데이터")
        bar_date = ""
    price_map = {t: df[t].dropna(how="all") for t in df.columns.levels[0]} if not df.empty else {}
    price_map = {t: f for t, f in price_map.items() if not f.empty and f["Close"].notna().any()}
    no_data = [t for t in tickers if t not in price_map]

    # 종목 정보는 신규 상장까지 한 번에 받는다 (compute_all_features_guarded 와 같은 재시도·전날 스냅샷 보충)
    t1 = time.perf_counter()
    fund = fetch_with_retry(list(price_map), f"[{label}]", waits=INFO_RETRY_WAITS)
    fund = fill_from_previous_snapshot(fund, f"[{label}]", snapshot_dir=snapshot_dir)
    t2 = time.perf_counter()
    # 패턴 · 교차 라벨은 TradingView 차트와 같은 가격(분할만 반영 · 상장 이후 전체)으로 본다 (2026-10-27)
    chart_map = fetch_chart_prices(list(price_map))
    features = compute_features_snapshot(price_map, fundamentals_df=fund, chart_map=chart_map)
    t3 = time.perf_counter()
    print(f"[{label}] 가격 {(t1 - t0) / 60:.1f}분 · 종목 정보 {(t2 - t1) / 60:.1f}분 · 피처 {(t3 - t2) / 60:.1f}분")
    analyzed = set(features["티커"]) if not features.empty else set()
    young = {t: f for t, f in price_map.items() if t not in analyzed and len(f) < MIN_BARS}
    print(f"[{label}] 분석 {len(analyzed)} · 신규 상장(일봉<{MIN_BARS}) {len(young)} · 가격 없음 {len(no_data)}")

    ranked = pd.DataFrame()
    if not features.empty:
        ranked = attach_signals_and_sort(tag_sectors(apply_neutralization(features), strong))
        if {"모멘텀_적합도", "바닥반등_적합도"}.issubset(ranked.columns):
            ranked = ranked.sort_values(["모멘텀_적합도", "바닥반등_적합도"], ascending=[False, False])
        last_close = {t: float(f["Close"].dropna().iloc[-1]) for t, f in price_map.items() if t in analyzed}
        loc = ranked.columns.get_loc("티커")
        ranked.insert(loc, "회사", ranked["티커"].map(lambda t: names.get(t) or COMPANY_NAME_MAP.get(t) or _cached_name(t)))
        ranked.insert(loc + 1, "현재가격", ranked["티커"].map(last_close))
        # 3달 · 1년 수익률 — 웹 섹터 지도 칸 중앙값용 (표시 전용, 시트 칸 아님 · 2026-10-30)
        for k, col in ((63, "63일수익률"), (252, "252일수익률")):
            ranked[col] = ranked["티커"].map(lambda t, k=k: _ret_k(price_map[t]["Close"].dropna(), k))
        # 하루 수익률 · 52주 최저/최고(차트 가격) · 50/200일선 · 신고가 — 웹 '투자자 한눈에' (표시 전용, 시트 칸 아님 · 2026-11-03)
        stats = {t: price_stats(price_map[t], chart_map.get(t)) for t in analyzed}
        add_web_stats(ranked, stats)
        ranked["_bar_date"] = bar_date
        ranked = add_health(ranked).reset_index(drop=True)
    else:
        stats = {}

    young_fund = fund[fund["티커"].isin(list(young))] if not fund.empty else pd.DataFrame()
    return {"ranked": ranked, "young": young, "young_fund": young_fund, "bar_date": bar_date, "no_data": no_data,
            "fund": fund, "stats": stats}


def add_web_stats(ranked: pd.DataFrame, stats: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """price_stats 결과 → 웹 표시 칸 (표시 전용, 시트 칸 아님 — universe.item_info 가 읽는다). ranked 를 고쳐서 돌려준다.

    1일수익률 · 52주최저/최고 (2026-11-03), 52주신고가/신저가 (2026-11-04 — market.json 숫자와 같은 표시라
    오늘 탭 숫자를 누르면 사이트가 이 표시로 그 종목들을 거른다).
    """
    tick = ranked["티커"]
    ranked["1일수익률"] = tick.map(lambda t: stats[t]["ret_1d"])
    ranked["52주최저"] = tick.map(lambda t: (stats[t]["range_52w"] or {}).get("low"))
    ranked["52주최고"] = tick.map(lambda t: (stats[t]["range_52w"] or {}).get("high"))
    ranked["52주신고가"] = tick.map(lambda t: stats[t]["new_high"])
    ranked["52주신저가"] = tick.map(lambda t: stats[t]["new_low"])
    return ranked


def _ret_k(close: pd.Series, k: int) -> float | None:
    """k 거래일 수익률 (features 의 pct_change(k) 와 같은 계산). 일봉이 k+1 개 미만이면 None."""
    if len(close) <= k or not close.iloc[-1 - k] > 0:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - k] - 1)


_NAME_CACHE: dict[str, str] | None = None


def _cached_name(ticker: str) -> str:
    """src/data/company_names_cache.json 을 읽기만 한다 (없으면 '')."""
    global _NAME_CACHE
    if _NAME_CACHE is None:
        from data.fetch import COMPANY_NAME_CACHE

        _NAME_CACHE = dict(COMPANY_NAME_CACHE)
    return _NAME_CACHE.get(ticker, "")


# ── 신규 상장 ───────────────────────────────────────────────


def _rsi(close: pd.Series, window: int = 14) -> float | None:
    if len(close) <= window:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    g, lo = gain.iloc[-1], loss.iloc[-1]
    if pd.isna(g) or pd.isna(lo):
        return None
    return 100.0 if lo == 0 else float(100 - 100 / (1 + g / lo))


def young_row(ticker: str, frame: pd.DataFrame) -> dict[str, Any]:
    """일봉 120개 미만 종목의 간단 지표 (점수 없음)."""
    f = frame.dropna(subset=["Close"])
    close, vol = f["Close"], f["Volume"].fillna(0)
    last = float(close.iloc[-1])
    n = len(close)

    def ret(k: int) -> float | None:
        return float(last / close.iloc[-1 - k] - 1) if n > k and close.iloc[-1 - k] > 0 else None

    first_close = float(close.iloc[0])
    hi, lo = float(f["High"].max()), float(f["Low"].min())
    vol_avg = float(vol.iloc[-21:-1].mean()) if n > 1 else float("nan")
    dollar = float((close * vol).iloc[-20:].mean())
    return {
        "티커": ticker,
        "현재가격": last,
        "첫거래일": str(pd.Timestamp(close.index[0]).date()),
        "상장후거래일": n,
        "상장후수익률": last / first_close - 1 if first_close > 0 else None,
        "5일수익률": ret(5),
        "20일수익률": ret(20),
        "RSI": _rsi(close),
        "거래량돌파배수": float(vol.iloc[-1] / vol_avg) if vol_avg and vol_avg > 0 else None,
        "상장후위치": (last - lo) / (hi - lo) if hi > lo else None,
        "52주포지션": (last - lo) / (hi - lo) if hi > lo else None,  # 웹 52주 위치 칸 = 상장 후 고저 안 위치
        "최근20일평균거래대금": dollar,
    }


def young_listings(young: dict[str, pd.DataFrame], fund: pd.DataFrame, names: dict[str, str],
                   market_of: dict[str, str]) -> pd.DataFrame:
    """신규 상장 표 — 최근 상장 순. 재무·섹터·실적일은 같은 실행에서 받은 종목 정보."""
    if not young:
        return pd.DataFrame(columns=["티커"])
    rows = pd.DataFrame([young_row(t, f) for t, f in young.items()])
    if not fund.empty:
        rows = rows.merge(fund, on="티커", how="left")
        if "fund_sector" in rows.columns:
            rows["섹터"] = rows["fund_sector"].fillna("Unknown")
    rows.insert(1, "회사", rows["티커"].map(lambda t: names.get(t) or _cached_name(t)))
    rows["market"] = rows["티커"].map(market_of)
    rows["시장"] = rows["market"].map({"sp500": "S&P 500", "nasdaq": "NASDAQ/NYSE"})
    rows = add_health(rows)
    return rows.sort_values(["첫거래일", "티커"], ascending=[False, True]).reset_index(drop=True)


# ── 덧붙이는 정보 (웹 전체차트와 같은 것) ─────────────────────


def add_sec(df: pd.DataFrame, flags: dict[str, list]) -> pd.DataFrame:
    """[공시] 한 줄 (🔴·🟠) + sec_flags JSON — paper_trading/runner.py save_ranked_snapshot 과 같다."""
    from screener.sec_filings import flags_json, format_flags

    out = df.copy()
    keys = out["티커"].map(_norm)
    out["[공시]"] = [format_flags(flags.get(k, [])) for k in keys]
    out["sec_flags"] = [flags_json(flags[k]) if k in flags else "" for k in keys]
    return out


def add_short(df: pd.DataFrame, sd: Any) -> pd.DataFrame:
    """공매도 칸 — paper_trading/runner.py save_ranked_snapshot 과 같은 계산 (FINRA 잔고 ÷ 유통주식, 없으면 yfinance 비중)."""
    from screener.short_squeeze import is_high, short_line

    out = df.copy()
    if sd is None:
        out["[공매도]"] = ""
        return out
    pcts, dtcs, lines, highs, changes = [], [], [], [], []
    for _, r in out.iterrows():
        got = sd.get(str(r["티커"]))
        fl = pd.to_numeric(r.get("fund_float_shares"), errors="coerce")
        pct = got["si"] / fl if got and fl and fl > 0 else pd.to_numeric(r.get("fund_short_pct_float"), errors="coerce")
        pct = float(pct) if pd.notna(pct) and pct < 3 else None
        dtc = got["dtc"] if got else None
        pcts.append(pct)
        dtcs.append(dtc)
        changes.append(got["si_change"] if got else None)
        lines.append(short_line(pct, dtc, sd.settle if got else None))
        highs.append(is_high(pct, dtc))
    out["short_pct"], out["short_dtc"], out["short_change"] = pcts, dtcs, changes
    out["short_settle"] = sd.settle
    out["[공매도]"], out["short_high"] = lines, highs
    return out


def add_yahoo_short(df: pd.DataFrame, prices: Mapping[str, pd.DataFrame] | None) -> pd.DataFrame:
    """미국 밖 상장(캐나다) 공매도 칸 — FINRA 잔고가 없어서 Yahoo 종목 정보로 (add_short 와 같은 칸).

    비중 = Yahoo 공매도 잔고(fund_short_interest — 캐나다는 CIRO 보름 집계) ÷ 유통주식 (없으면 Yahoo 비중).
    되사기 = 잔고 ÷ 최근 20거래일 평균 거래량 (ISB · SQZ 와 같은 일봉 — Yahoo 는 상장 거래소 거래량만이라 다른 거래 시장까지
    합친 FINRA 되사기보다 길게 나올 수 있다). Yahoo 는 집계 기준일을 종목 정보에 안 줘서 기준일은 비운다.
    """
    from screener.short_squeeze import is_high, short_line
    from web_export.isb_signals import yf_symbol

    out = df.copy()
    pcts, dtcs, lines, highs = [], [], [], []
    for _, r in out.iterrows():
        si = pd.to_numeric(r.get("fund_short_interest"), errors="coerce")
        fl = pd.to_numeric(r.get("fund_float_shares"), errors="coerce")
        pct = si / fl if pd.notna(si) and pd.notna(fl) and fl > 0 else pd.to_numeric(r.get("fund_short_pct_float"),
                                                                                     errors="coerce")
        pct = float(pct) if pd.notna(pct) and 0 <= pct < 3 else None
        frame = (prices or {}).get(yf_symbol(str(r["티커"])))
        vol = float(frame["Volume"].tail(CA_DTC_DAYS).mean()) if frame is not None and len(frame) else float("nan")
        dtc = float(si / vol) if pd.notna(si) and np.isfinite(vol) and vol > 0 else None
        pcts.append(pct)
        dtcs.append(dtc)
        lines.append(short_line(pct, dtc, None))
        highs.append(is_high(pct, dtc))
    out["short_pct"], out["short_dtc"] = pcts, dtcs
    out["short_change"], out["short_settle"] = None, None
    out["[공매도]"], out["short_high"] = lines, highs
    return out


def add_insider(df: pd.DataFrame, insider: pd.DataFrame) -> pd.DataFrame:
    """내부자(90일) 한 줄. 시장 전체 조회에 거래가 없던 종목은 '—'."""
    out = df.copy()
    text = dict(zip(insider["티커"], insider["내부자(90일)"])) if not insider.empty else {}
    out["내부자(90일)"] = out["티커"].map(lambda t: text.get(_norm(t), "—"))
    return out


def golden_list(ranked: pd.DataFrame) -> list[dict[str, Any]]:
    """PT 골든크로스 목록과 같은 규칙·순서 (web_export.dashboard.golden_cross_lists 한 시장분)."""
    from paper_trading.golden_cross import extract_golden_cross
    from web_export.dashboard import _TF_KEYS, _num

    if ranked.empty:
        return []
    return [{"ticker": g["ticker"],
             "timeframes": [{"tf": _TF_KEYS.get(lab, lab), "gap": _num(gap, 4)} for lab, gap in g["tf_gaps"]]}
            for g in extract_golden_cross(ranked)]


# ── 출력 ─────────────────────────────────────────────────────


def sheet_frame(ranked: pd.DataFrame, isb: dict, sqz: dict, earn: dict | None = None) -> pd.DataFrame:
    """분석 탭과 같은 칸 (prepare_export_dataframe) + [공시]·[공매도]·ISB·SQZ 를 [헬스체크] 뒤에,
    경기 성격 · 매출 지역 · 미국 매출 비중 · 실적 움직임을 맨 끝에 (사용자 수식이 앞 칸을 가리켜도 안 밀리게)."""
    from data.industry_style import style_for, style_text
    from screener.exporter import prepare_export_dataframe
    from web_export.dashboard import _industry
    from web_export.universe import isb_text, sqz_text

    out = prepare_export_dataframe(ranked)
    tick = ranked["티커"].astype(str).tolist()
    extra = {
        "[공시]": ranked.get("[공시]", pd.Series("", index=ranked.index)).tolist(),
        "[공매도]": ranked.get("[공매도]", pd.Series("", index=ranked.index)).tolist(),
        "ISB": [isb_text(isb.get(t)) for t in tick],
        "SQZ": [sqz_text(sqz.get(t)) for t in tick],
    }
    at = out.columns.get_loc("[헬스체크]") + 1 if "[헬스체크]" in out.columns else len(out.columns)
    for i, (col, vals) in enumerate(extra.items()):
        out.insert(at + i, col, vals)
    inds = ranked.get("fund_industry", pd.Series(None, index=ranked.index)).tolist()
    texts = [style_text(style_for(t, _industry(ind))) for t, ind in zip(tick, inds)]
    for i, col in enumerate(STYLE_SHEET_COLUMNS):
        out[col] = [x[i] for x in texts]
    out[EARN_SHEET_COLUMN] = [earn_text((earn or {}).get(t)) for t in tick]
    return out


def earn_text(st: dict | None) -> str:
    """'±8.8% · 최악 -17.8% · 장 마감 후 (8번)' — 실적 반응 요약 한 칸. 모르면 빈칸."""
    from screener.earnings_moves import TIMING_KO

    if not st:
        return ""
    when = TIMING_KO.get(st.get("timing") or "", "")
    return f"±{st['avg']:.1%} · 최악 {st['worst']:+.1%}" + (f" · {when}" if when else "") + f" ({st['n']}번)"


def new_sheet_frame(young: pd.DataFrame, sqz: dict) -> pd.DataFrame:
    from web_export.universe import sqz_text

    out = young.copy()
    out["SQZ"] = [sqz_text(sqz.get(t)) for t in out["티커"].astype(str)]
    for col in NEW_PERCENT:
        if col in out.columns:
            out[col] = (pd.to_numeric(out[col], errors="coerce") * 100).round(1)
    if "최근20일평균거래대금" in out.columns:
        out["최근20일평균거래대금"] = (out["최근20일평균거래대금"] / 1_000_000).round(1)
    for col in ("현재가격", "RSI", "거래량돌파배수"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    out = out.reindex(columns=NEW_COLUMNS)
    return out.rename(columns={"최근20일평균거래대금": "최근20일평균거래대금(M)", "상장후위치": "상장후위치(%)",
                               "상장후수익률": "상장후수익률(%)", "5일수익률": "5일수익률(%)", "20일수익률": "20일수익률(%)"})


def company_about(fund: pd.DataFrame) -> dict[str, str]:
    """종목 정보 표 → {티커: 회사 소개(영어, yfinance)}. 못 받은 종목은 전날 스냅샷 값 (fill_from_previous_snapshot)."""
    if fund is None or fund.empty or "fund_summary" not in fund.columns:
        return {}
    pairs = zip(fund["티커"].astype(str), fund["fund_summary"])
    return {t: str(v).strip() for t, v in pairs if not pd.isna(v) and str(v).strip() not in ("", "nan", "None")}


def snapshot_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """다음 날 종목 정보가 막혔을 때 쓸 값 — 티커·섹터·실적일·fund_* 만 (fundamentals_guard 가 읽는 칸)."""
    parts = [f[[c for c in f.columns if c in SNAPSHOT_EXTRA or str(c).startswith("fund_")]] for f in frames if not f.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).drop_duplicates("티커", keep="first")
    for col in out.columns:  # parquet 은 글자·숫자가 섞인 칸을 못 쓴다 (runner.save_ranked_snapshot 과 같은 처리)
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: None if pd.isna(v) else str(v))  # 빈 값은 'nan' 글자가 아니라 빈 칸으로
    return out


DETAIL_BUDGET_SEC = 1500  # 분기 실적 · 의견 수 받는 시간 한도 (첫 밤 약 2천 종목 중 일부, 나머지는 다음 밤)


def web_extras(ranked_frames: list[pd.DataFrame], items: dict[str, dict[str, Any]],
               stats: dict[str, dict[str, Any]], sp_tickers: list[str], bar_date: str | None,
               cache_dir: Path | None, out_dir: Path,
               budget_sec: float = DETAIL_BUDGET_SEC) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """웹 '투자자 한눈에' (2026-11-03, 표시 전용) → (종목 상세 {티커: detail}, market.json 내용).

    밸류에이션 업종 중앙값 · 목표주가는 이미 받은 종목 정보로, 분기 실적 · 의견 수는 S&P 500 + 시총 $2B 이상만
    stock_detail 캐시(cache_dir/detail.json.gz → out_dir/snapshot/)로 시간 한도 안에서 조금씩 받는다.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from web_export import stock_detail
    from web_export.dashboard import _industry
    from web_export.market_summary import (
        MOVER_MIN_CAP, analyst_block, industry_medians, market_summary, valuation_block, valuation_values,
    )

    frames = [f for f in ranked_frames if f is not None and not f.empty]
    recs = pd.concat(frames, ignore_index=True).drop_duplicates("티커").to_dict("records") if frames else []
    med = industry_medians({"industry": _industry(r.get("fund_industry")), "cap": r.get("fund_market_cap"),
                            **valuation_values(r)} for r in recs)

    now = datetime.now(ZoneInfo("America/New_York"))
    sp_set = set(sp_tickers)
    scope = [t for t, v in items.items()
             if t in sp_set or ((v.get("fundamentals") or {}).get("market_cap") or 0) >= MOVER_MIN_CAP]
    cache = stock_detail.load_cache(cache_dir / stock_detail.CACHE_NAME if cache_dir else None)
    last_report = {t: (((items[t].get("earnings_move") or {}).get("last") or [[None]])[-1] or [None])[0] for t in scope}
    stock_detail.refresh(scope, cache, now.date(), last_report, budget_sec=budget_sec)
    stock_detail.save_cache(out_dir / "snapshot" / stock_detail.CACHE_NAME, cache, keep=set(scope))

    details: dict[str, dict[str, Any]] = {}
    for r in recs:
        t = str(r["티커"])
        if t not in items:
            continue
        ent = cache.get(t)
        d = {"as_of": bar_date,
             "valuation": valuation_block(r, _industry(r.get("fund_industry")), med),
             "analyst": analyst_block(r, (ent or {}).get("rec")),
             "quarters": stock_detail.quarters_block(ent, items[t].get("earnings_date"))}
        if d["valuation"] or d["analyst"] or d["quarters"]:
            details[t] = d
    market = market_summary(items, sp_tickers, stats, bar_date, (stats.get("SPY") or {}).get("ret_1d"),
                            now.isoformat(timespec="seconds"))
    return details, market


def extra_market(tickers: list[str], names: dict[str, str], tv_map: dict[str, str], strong: set[str] | None,
                 prev_dir: Path, sd: Any, insider_map: Mapping[str, dict[str, Any]],
                 no_trades: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """번외 종목 → (시장 JSON 'extra', 회사 소개). 두 시장과 같은 함수로 이 종목들만 따로 분석한다 (EXTRA_TICKERS).

    중립화 비교 집단이 이 종목들뿐이라 점수는 [SP500 차트분석] 시트의 고정 종목처럼 참고용이다.
    미국 밖 상장(.V 등)은 SEC 공시 · openinsider 내부자 · FINRA 공매도가 없다 — 공시는 안 찾고, 내부자는 '정보 없음'(None,
    '거래 없음'이 아니라 — 웹은 ceo.ca 링크를 보여 준다), 공매도는 Yahoo 잔고(add_yahoo_short — 캐나다 탭과 같다).
    분석된 종목이 없으면 (None, {}).
    """
    from screener.sec_filings import fetch_flags
    from web_export.dashboard import daily_prices, earnings_moves_block, isb_block, sqz_block
    from web_export.universe import item_info, market_payload

    res = analyze_market(list(tickers), f"[{EXTRA_LABEL}]", names, strong, prev_dir)
    ranked = res["ranked"]
    young = young_listings(res["young"], res["young_fund"], names, {})
    all_t = [str(t) for df in (ranked, young) if not df.empty for t in df["티커"]]
    if not all_t:
        print(f"[{EXTRA_LABEL}] 분석된 종목 없음 (가격 없음 {res['no_data']})")
        return None, {}
    us = [t for t in all_t if not is_foreign(t)]
    accepts: dict[str, list[str]] = {}
    flags = fetch_flags(us, budget_sec=120, earnings=accepts) if us else {}
    prices = daily_prices(all_t)
    isb = isb_block(all_t, prices) if prices else {}
    sqz = sqz_block(all_t, prices) if prices else {}
    earn = earnings_moves_block({t: accepts[_norm(t)] for t in us if accepts.get(_norm(t))}, prices)

    def info(row: pd.Series, gold: dict[str, Any] | None = None) -> dict[str, Any]:
        t = str(row["티커"])
        insider = None if is_foreign(t) else insider_map.get(_norm(t), no_trades)
        return item_info(row, tv=extra_tv(t, tv_map), insider=insider, isb=isb.get(t), sqz=sqz.get(t), golden=gold,
                         earnings_move=earn.get(t))

    def extras(df: pd.DataFrame) -> pd.DataFrame:
        """공시 · 공매도 칸. 미국 밖 상장은 Yahoo 잔고로 (FINRA 잔고가 없어 add_short 는 야후 값에 FINRA 날짜를 붙인다)."""
        out = add_sec(df.assign(market=EXTRA_KEY), flags)
        foreign = out["티커"].map(is_foreign).astype(bool)
        parts = [add_short(out[~foreign], sd)] if (~foreign).any() else []
        parts += [add_yahoo_short(out[foreign], prices)] if foreign.any() else []
        return pd.concat(parts).loc[out.index]

    items: dict[str, dict[str, Any]] = {}
    golden: list[dict[str, Any]] = []
    if not ranked.empty:
        ranked = extras(ranked)
        golden = golden_list(ranked)
        rank_of = {g["ticker"]: (i, g["timeframes"]) for i, g in enumerate(golden)}
        for _, row in ranked.iterrows():
            t = str(row["티커"])
            items[t] = info(row, {"rank": rank_of[t][0], "timeframes": rank_of[t][1]} if t in rank_of else None)
    if not young.empty:
        young = extras(young)
        for _, row in young.iterrows():
            items[str(row["티커"])] = info(row)
    counts = {"listed": len(tickers), "analyzed": len(ranked), "young": len(young), "no_data": len(res["no_data"])}
    print(f"[{EXTRA_LABEL}] {', '.join(items)} — 분석 {len(ranked)} · 신규 상장 {len(young)} · 가격 없음 {len(res['no_data'])}")
    payload = market_payload(EXTRA_KEY, EXTRA_LABEL, items, bar_date=res["bar_date"] or None, counts=counts,
                             golden=golden)
    return payload, company_about(res["fund"])


def local_details(ranked: pd.DataFrame, items: Mapping[str, dict[str, Any]],
                  bar_date: str | None) -> dict[str, dict[str, Any]]:
    """종목 상세 '밸류에이션' · '애널리스트' 카드 — 이미 받은 종목 정보만으로 (web_extras 의 앞 두 칸과 같은 계산).

    업종 중앙값은 이 시장 안에서 낸다 (캐나다 종목을 미국 업종과 섞지 않게). 분기 실적 · 이번 달 의견 수(stock_detail 캐시)는
    따로 받아야 해서 없다 — 사이트는 그 카드만 안 보여 준다.
    """
    from web_export.dashboard import _industry
    from web_export.market_summary import analyst_block, industry_medians, valuation_block, valuation_values

    if ranked.empty:
        return {}
    recs = ranked.to_dict("records")
    med = industry_medians({"industry": _industry(r.get("fund_industry")), "cap": r.get("fund_market_cap"),
                            **valuation_values(r)} for r in recs)
    out: dict[str, dict[str, Any]] = {}
    for r in recs:
        t = str(r["티커"])
        if t not in items:
            continue
        d = {"as_of": bar_date, "valuation": valuation_block(r, _industry(r.get("fund_industry")), med),
             "analyst": analyst_block(r, None), "quarters": None}
        if d["valuation"] or d["analyst"]:
            out[t] = d
    return out


def ca_sheet_frame(ranked: pd.DataFrame, isb: dict, sqz: dict) -> pd.DataFrame:
    """[캐나다 전체] 시트 — 미국 전체 탭과 같은 칸 + '거래소'(TSX · TSXV)를 티커 뒤에, 늘 비는 미국 자료 칸(CA_SHEET_DROP)은 뺀다."""
    out = sheet_frame(ranked, isb, sqz).drop(columns=list(CA_SHEET_DROP), errors="ignore")
    exch = ranked["market"].astype(str).str.upper().tolist() if "market" in ranked.columns else [""] * len(ranked)
    at = out.columns.get_loc("티커") + 1 if "티커" in out.columns else 0
    out.insert(at, "거래소", exch)
    return out


def canada_market(listings: pd.DataFrame, prev_dir: Path) -> dict[str, Any] | None:
    """캐나다 상장 보통주(TSX · TSXV) 전부 → {payload(ca.json), about, details, sheet, snapshot}. 분석된 종목이 없으면 None.

    미국 두 시장과 같은 함수(analyze_market — 피처 → 중립화 → 신호 · 적합도)를 캐나다 종목끼리 돌린다. 다른 점:
    - 강한 섹터 표시는 없다 — 미국 섹터 ETF 로 정하는 규칙이라 캐나다 시장에는 맞지 않는다 (섹터강도 칸 없음 · 점수 가산 없음).
    - SEC 공시 · 실적 반응(SEC 8-K) 없음 — 캐나다는 SEDAR+ 라서.
    - 내부자: 칸을 비운다 (None) — Yahoo 의 SEDI 자료가 TSXV 소형주에서 빠지는 거래가 있다 (2026-11-05 사용자 확인 —
      ceo.ca 에 Yahoo 보다 거래가 많았다). 틀린 '거래 없음' 대신 웹은 그 종목 ceo.ca 페이지 링크를 보여 준다.
    - 공매도: Yahoo 잔고 (add_yahoo_short).
    - 신규 상장(일봉 120개 미만)도 같은 파일에 (번외처럼 — 미국 [신규상장] 탭 · 시트와 섞지 않는다).
    가격 · 시총은 C$ (Yahoo 가 캐나다 종목을 CAD 로 준다 — 웹은 티커 접미사로 C$ 를 붙인다).
    """
    from web_export.dashboard import daily_prices, isb_block, sqz_block
    from web_export.universe import item_info, market_payload

    tickers = listings["ticker"].astype(str).tolist()
    names = dict(zip(listings["ticker"], listings["name"]))
    tv_map = dict(zip(listings["ticker"], listings["tv"]))
    board = {t: str(x).lower() for t, x in zip(listings["ticker"], listings["exchange"])}   # 'tsx' · 'tsxv'

    res = analyze_market(tickers, SHEET_CA, names, None, prev_dir)
    ranked = res["ranked"]
    young = young_listings(res["young"], res["young_fund"], names, {})
    all_t = [str(t) for df in (ranked, young) if not df.empty for t in df["티커"]]
    if not all_t:
        print(f"[{CA_LABEL}] 분석된 종목 없음 (가격 없음 {len(res['no_data'])})")
        return None
    prices = daily_prices(all_t)
    isb = isb_block(all_t, prices) if prices else {}
    sqz = sqz_block(all_t, prices) if prices else {}

    def tag(df: pd.DataFrame) -> pd.DataFrame:
        return add_yahoo_short(df.assign(market=df["티커"].map(board)), prices)

    def info(row: pd.Series, gold: dict[str, Any] | None = None) -> dict[str, Any]:
        t = str(row["티커"])
        return item_info(row, tv=tv_map.get(t) or extra_tv(t, {}), insider=None, isb=isb.get(t), sqz=sqz.get(t),
                         golden=gold)

    items: dict[str, dict[str, Any]] = {}
    golden: list[dict[str, Any]] = []
    if not ranked.empty:
        ranked = tag(ranked)
        golden = golden_list(ranked)
        rank_of = {g["ticker"]: (i, g["timeframes"]) for i, g in enumerate(golden)}
        for _, row in ranked.iterrows():
            t = str(row["티커"])
            items[t] = info(row, {"rank": rank_of[t][0], "timeframes": rank_of[t][1]} if t in rank_of else None)
    if not young.empty:
        young = tag(young)
        for _, row in young.iterrows():
            items[str(row["티커"])] = info(row)
    del prices
    counts = {"listed": len(tickers), "analyzed": len(ranked), "young": len(young), "no_data": len(res["no_data"])}
    n_short = sum(1 for v in items.values() if v.get("short"))
    print(f"[{CA_LABEL}] 분석 {len(ranked)} · 신규 상장 {len(young)} · 가격 없음 {len(res['no_data'])} · "
          f"골든 {len(golden)} · 공매도 {n_short}")
    bar_date = res["bar_date"] or None
    return {"payload": market_payload(CA_KEY, CA_LABEL, items, bar_date=bar_date, counts=counts, golden=golden),
            "about": company_about(res["fund"]),
            "details": local_details(ranked, items, bar_date),
            "sheet": ca_sheet_frame(ranked, isb, sqz) if not ranked.empty else None,
            "snapshot": snapshot_frame([ranked, res["young_fund"]])}


def _prev_snapshot_dir(prev_dir: Path | None) -> Path:
    """전날 스냅샷 폴더 — 아는 이름(SNAPSHOT_NAMES)의 파일이 하나라도 있으면 그 폴더, 없으면 실전 스냅샷(data/paper_trading)."""
    if prev_dir and any((prev_dir / n).exists() for n in SNAPSHOT_NAMES.values()):
        return prev_dir
    return SNAPSHOT_DIR


def canada_main(args: argparse.Namespace) -> int:
    """--market ca — 캐나다만 (별도 워크플로 canada-scan.yml). 분석된 종목이 없으면 1 (CI 가 빈 브랜치를 올리지 않게)."""
    from data.ca_listings import fetch_ca_listings
    from web_export.universe import write_payloads

    start = time.perf_counter()
    listings = fetch_ca_listings()
    if args.limit:
        listings = listings.groupby("exchange", sort=False).head(args.limit).reset_index(drop=True)
    prev_dir = _prev_snapshot_dir(args.prev_dir)
    print(f"[{CA_LABEL}] 종목 정보 보충 스냅샷: {prev_dir}")
    res = canada_market(listings, prev_dir)
    _log(f"{CA_LABEL} 분석 끝", start)
    if res is None:
        return 1
    sizes = write_payloads(args.out_dir, [res["payload"]], res["about"], res["details"], None)
    print(f"[{CA_LABEL}] 회사 소개 {len(res['about'])}종목 · 종목 상세 {len(res['details'])}종목 · 웹 파일: {sizes}")
    if not res["snapshot"].empty:
        snap_dir = args.out_dir / "snapshot"
        snap_dir.mkdir(parents=True, exist_ok=True)
        res["snapshot"].to_parquet(snap_dir / SNAPSHOT_NAMES[CA_KEY], index=False)
    if args.no_sheet:
        print("[전체분석] --no-sheet: 시트는 건너뜀")
    elif res["sheet"] is not None:
        from screener.exporter import export_to_google_sheet

        ok = export_to_google_sheet(res["sheet"], SHEET_CA)
        print(f"[{SHEET_CA}] 구글 시트 {'업데이트 완료' if ok else '업데이트 실패'} ({len(res['sheet'])}행)")
    _log("완료", start)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="전체 분석 (필터 없음, 표시 전용)")
    ap.add_argument("--out-dir", type=Path, required=True, help="웹 JSON·스냅샷을 쓸 폴더")
    ap.add_argument("--prev-dir", type=Path, default=None,
                    help="전날 전체분석 스냅샷 폴더 (종목 정보가 막혔을 때 보충). 없으면 data/paper_trading")
    ap.add_argument("--no-sheet", action="store_true", help="구글 시트에 쓰지 않는다 (로컬 확인용)")
    ap.add_argument("--limit", type=int, default=0, help="시장마다 앞에서 N종목만 (빠른 확인)")
    ap.add_argument("--market", choices=("all", "sp500", "nasdaq", EXTRA_KEY, CA_KEY), default="all",
                    help=f"all = 미국 두 시장 + 번외 · {EXTRA_KEY} = 번외 종목만 (로컬 확인용 — 시트 · market.json 없이 "
                         f"{EXTRA_KEY}.json 만) · {CA_KEY} = 캐나다만 ({CA_KEY}.json · [캐나다 전체], 별도 워크플로)")
    ap.add_argument("--detail-budget", type=float, default=DETAIL_BUDGET_SEC,
                    help="분기 실적 · 의견 수를 받는 초 (0 = 안 받고 캐시만)")
    args = ap.parse_args(argv)
    if args.market == CA_KEY:
        return canada_main(args)

    from screener.sec_filings import fetch_flags
    from screener.short_squeeze import short_data
    from screener.insider import fetch_insider_market
    from web_export.dashboard import daily_prices, earnings_moves_block, insider_records, isb_block, sqz_block
    from web_export.universe import item_info, market_payload, write_payloads

    start = time.perf_counter()
    listings = fetch_listings()
    names = dict(zip(listings["ticker"], listings["name"]))
    tv_map = dict(zip(listings["ticker"], listings["tv"]))
    sp_tickers, rest = split_universe(listings["ticker"].tolist(), SP500_TICKERS, INDICES)
    if args.limit:
        sp_tickers = list(dict.fromkeys([*sp_tickers[:args.limit], "SPY"]))
        rest = rest[:args.limit]
    markets = [m for m in (("sp500", "S&P 500", SHEET_SP500, sp_tickers), ("nasdaq", "나스닥·NYSE", SHEET_NASDAQ, rest))
               if args.market in ("all", m[0])]
    print(f"[전체분석] S&P 500 {len(sp_tickers)} · NASDAQ/NYSE(S&P 제외) {len(rest)}")
    prev_dir = _prev_snapshot_dir(args.prev_dir)
    print(f"[전체분석] 종목 정보 보충 스냅샷: {prev_dir}")

    strong = strong_sectors()
    insider_df = fetch_insider_market()
    insider_map = insider_records(insider_df)
    # 시장 전체 조회가 됐으면 목록에 없는 종목 = 90일간 거래 없음 (웹 '거래 없음'). 조회 실패면 None (웹 '정보 없음')
    no_trades = {"buys": 0, "sells": 0, "net_value": 0.0, "last_date": None, "signal": None} if insider_map else None
    sd = short_data()
    _log("준비 끝", start)

    payloads: list[dict[str, Any]] = []
    sheets: list[tuple[str, pd.DataFrame]] = []
    young_parts: list[pd.DataFrame] = []
    young_items: dict[str, dict[str, Any]] = {}
    young_sqz: dict[str, Any] = {}
    snapshots: dict[str, list[pd.DataFrame]] = {}
    bar_dates: list[str] = []
    market_of = {t: "sp500" for t in sp_tickers} | {t: "nasdaq" for t in rest}
    about: dict[str, str] = {}
    sp_items: dict[str, dict[str, Any]] = {}       # [섹터 대표] 시트 — S&P 500 웹 정보
    web_items: dict[str, dict[str, Any]] = {}      # 두 시장 웹 정보 (지수 빼고) — market.json · 종목 상세
    web_stats: dict[str, dict[str, Any]] = {}      # price_stats (SPY 포함) — 시장 폭

    # 번외 먼저 — 몇 종목뿐이라 금방이고, 5천 종목 뒤에는 Yahoo 종목 정보가 막혀 있을 수 있다. 표시 전용이라 실패해도 계속
    extra_payload: dict[str, Any] | None = None
    if args.market in ("all", EXTRA_KEY):
        todo = [t for t in EXTRA_TICKERS if _norm(t) not in market_of]
        try:
            if todo:
                extra_payload, extra_about = extra_market(todo, names, tv_map, strong, prev_dir, sd, insider_map,
                                                          no_trades)
                about.update(extra_about)
        except Exception as exc:  # noqa: BLE001
            print(f"[{EXTRA_LABEL}] 실패 (건너뜀): {exc}")
            extra_payload = None
        _log(f"{EXTRA_LABEL} 끝", start)

    for key, label, sheet, tickers in markets:
        res = analyze_market(tickers, sheet, names, strong, prev_dir)
        _log(f"{sheet} 분석 끝", start)
        about.update(company_about(res["fund"]))
        ranked = res["ranked"]
        young = young_listings(res["young"], res["young_fund"], names, market_of)
        all_t = ranked["티커"].astype(str).tolist() if not ranked.empty else []
        all_t += young["티커"].astype(str).tolist() if not young.empty else []

        accepts: dict[str, list[str]] = {}
        flags = fetch_flags(all_t, budget_sec=SEC_BUDGET_SEC, earnings=accepts)
        prices = daily_prices(all_t)
        isb = isb_block(all_t, prices) if prices else {}
        sqz = sqz_block(all_t, prices) if prices else {}
        earn = earnings_moves_block({t: accepts[_norm(t)] for t in all_t if accepts.get(_norm(t))}, prices)
        print(f"[{sheet}] 실적 움직임 {len(earn)}종목 (실적 8-K {len(accepts)}종목)")
        del prices
        _log(f"{sheet} SEC·ISB·SQZ 끝", start)

        golden = []
        if not ranked.empty:
            ranked["market"] = key
            ranked = add_insider(add_short(add_sec(ranked, flags), sd), insider_df)
            golden = golden_list(ranked)
            rank_of = {g["ticker"]: (i, g["timeframes"]) for i, g in enumerate(golden)}
            items = {}
            for _, row in ranked.iterrows():
                t = str(row["티커"])
                gold = {"rank": rank_of[t][0], "timeframes": rank_of[t][1]} if t in rank_of else None
                items[t] = item_info(row, tv=tv_of(t, tv_map), insider=insider_map.get(_norm(t), no_trades),
                                     isb=isb.get(t), sqz=sqz.get(t), golden=gold, earnings_move=earn.get(t))
            counts = {"listed": len(tickers), "analyzed": len(ranked), "young": len(young), "no_data": len(res["no_data"])}
            payloads.append(market_payload(key, label, items, bar_date=res["bar_date"] or None, counts=counts,
                                           golden=golden))
            for t, v in payloads[-1]["items"].items():   # 지수(SPY 등)는 빠진 목록
                web_items.setdefault(t, v)
            web_stats.update(res["stats"])
            sheets.append((sheet, sheet_frame(ranked, isb, sqz, earn)))
            bar_dates.append(res["bar_date"])
            if key == "sp500":
                sp_items = items
        if not young.empty:
            young = add_insider(add_short(add_sec(young, flags), sd), insider_df)
            young_parts.append(young)
            for _, row in young.iterrows():
                t = str(row["티커"])
                young_items[t] = item_info(row, tv=tv_of(t, tv_map), insider=insider_map.get(_norm(t), no_trades),
                                           isb=isb.get(t), sqz=sqz.get(t), earnings_move=earn.get(t))
                young_sqz[t] = sqz.get(t)
        snapshots[key] = [ranked, res["young_fund"]]
        del res

    if young_parts:
        young_all = pd.concat(young_parts, ignore_index=True).sort_values(["첫거래일", "티커"], ascending=[False, True])
        items = {str(t): young_items[str(t)] for t in young_all["티커"]}
        payloads.append(market_payload("new", "신규 상장", items, bar_date=max(bar_dates) if bar_dates else None,
                                       counts={"listed": len(young_all)}))
        sheets.append((SHEET_NEW, new_sheet_frame(young_all, young_sqz)))
    if sp_items:
        from screener.sector_reps import SHEET_TAB as REPS_TAB, sector_reps_frame

        reps = sector_reps_frame(sp_items)
        print(f"[전체분석] 섹터 대표 {len(reps)}행 (글로벌 {int((reps['구분'] == '글로벌 매출 중심').sum())})")
        if len(reps):
            sheets.append((REPS_TAB, reps))

    if extra_payload:
        payloads.append(extra_payload)

    real_dates = [d for d in bar_dates if d]
    details: dict[str, dict[str, Any]] = {}
    market: dict[str, Any] | None = None
    if snapshots:  # 두 시장 중 하나라도 돌았을 때만 (--market extra 는 번외 파일만)
        details, market = web_extras([f[0] for f in snapshots.values()], web_items, web_stats, sp_tickers,
                                     max(real_dates) if real_dates else None, args.prev_dir, args.out_dir,
                                     budget_sec=args.detail_budget)
        _log("투자자 한눈에 끝", start)
    sizes = write_payloads(args.out_dir, payloads, about, details, market)
    print(f"[전체분석] 회사 소개 {len(about)}종목 · 종목 상세 {len(details)}종목 · market.json {'있음' if market else '없음'}")
    snap_dir = args.out_dir / "snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for key, frames in snapshots.items():
        snap = snapshot_frame(frames)
        if not snap.empty:
            snap.to_parquet(snap_dir / SNAPSHOT_NAMES[key], index=False)
    print(f"[전체분석] 웹 파일: {sizes}")

    if args.no_sheet:
        print("[전체분석] --no-sheet: 시트는 건너뜀")
    else:
        from screener.exporter import export_to_google_sheet

        for sheet, frame in sheets:
            ok = export_to_google_sheet(frame, sheet)
            print(f"[{sheet}] 구글 시트 {'업데이트 완료' if ok else '업데이트 실패'} ({len(frame)}행)")
    _log("완료", start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
