"""웹 정렬용 ISB 라벨 — TradingView 차트의 BUY·REV · ◇ BOTTOM · ◆ TOP · EXIT 가 최근 언제 나왔는지.

엔진은 연구용 포팅 `scripts/isb_engine.py` (pinescript/ISB.pine v9.4, 입력값 기본) 를 그대로 쓴다.
2026-10-12 ABT 등 TradingView 차트와 라벨 날짜가 일치한 그 엔진이다.

- 가격: TradingView 기본 차트(ADJ 꺼짐 = 분할만 반영)와 같게 yfinance auto_adjust=False 의 OHLC.
  배당 조정 가격을 넣으면 배당이 큰 종목은 라벨 날짜가 달라진다 (isb_engine 머리말). 소수 4자리로 반올림한다 (_frame).
- 기록 7년: 엔진이 과거 매매 상태를 이어 가는 구조라 기록이 짧으면 라벨이 달라질 수 있다.
  2026-10-19 120종목 확인 — 5년 이상이면 최근 1년 라벨이 12년치 계산과 일봉·주봉 모두 같았다 (3년은 주봉 7종목 다름).
- 끝나지 않은 봉의 라벨은 안 낸다 (Pine 라벨은 barstate.isconfirmed — 장중 일봉·이번 주 주봉·이번 달 월봉에는 안 찍힌다).
- 월봉 (2026-11-13): 엔진 상태가 수십 년을 이어 가서 7년 일봉으로는 모자라 야후 1mo 전체 기간(1985년~)을 따로 받는다
  (MONTHLY_HISTORY — 겹치는 84달은 일봉을 모은 월봉과 OHLCV 가 같았다). 라벨 날짜는 그 달 마지막 거래일 (일봉 기준).
- 벤치마크(상대강도 · REV 시장 투매 · 추세 시장 필터)는 Pine benchSym Auto 처럼 상장 거래소로 고른다:
  TSX(.TO) → TSX:TSX = 야후 ^GSPTSE, TSXV(.V) → TSX:JX = ^SPCDNX, 그 밖 → SPY. TSXV 는 매도 방식도 Pine 처럼
  주기와 상관없이 Swing (v9.3) · S2 끔 · 시장 필터 끔 (IsbParams.junior). 2026-11-12 전에는 캐나다도 SPY · 미국 규칙이라
  표본 150종목씩 TSX 3% · TSXV 23% 의 라벨이 Pine 규칙대로 계산한 것과 달랐다 (미국 종목은 그대로).
- 한계: 좀비 필터(재무 부실이면 매수 대신 ◇ BOTTOM)는 엔진에 없다 (재무 없으면 Pine 도 통과).
  적자·현금흐름 마이너스 소형주는 TradingView 에 ◇ BOTTOM 인데 여기서는 BUY·REV 로 나올 수 있다.

표시 전용 — 매매 판단·상태 파일과 무관하다.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper_trading.market_date import BAR_COMPLETE_TIME, is_bar_complete, market_now

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
LABELS = ("rev", "bottom", "top", "exit")  # BUY·REV · ◇ BOTTOM · ◆ TOP · EXIT
HISTORY = "7y"
MONTHLY_HISTORY = "max"  # 월봉은 야후 1mo 전체 (일봉 7년 = 84달로는 엔진 상태 · 100봉 지표가 TradingView 와 달라진다)
# 마지막 완성 봉에서 이 봉 수 안의 가장 최근 라벨만 (일봉 ≈ 3달, 주봉 ≈ 6달, 월봉 1년)
WINDOW = {"D": 63, "W": 26, "M": 12}
MACRO = {"vix": "^VIX", "dxy": "DX-Y.NYB"}  # ISB 매크로 리스크 (isb_engine.load_macro 와 같은 심볼)
BENCH = "SPY"
# ISB.pine benchSym Auto: TSXV → TSX:JX (S&P/TSX 벤처 종합) · TSX → TSX:TSX (S&P/TSX 종합) · 그 밖 → AMEX:SPY.
# 야후 심볼 = 같은 지수 (2026-11-12 TradingView 기호 검색으로 TSX:JX · TSX:TSX 이름 확인).
CA_BENCH = {".V": "^SPCDNX", ".TO": "^GSPTSE"}
CHUNK = 100
PRICE_COLS = ["Open", "High", "Low", "Close"]
PRICE_DECIMALS = 4


def _engine() -> Any:
    """scripts/isb_engine.py 모듈 (scripts 는 패키지가 아니라 경로로 불러온다)."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import isb_engine  # noqa: PLC0415

    return isb_engine


_CAD_SUFFIX = (".V", ".TO", ".CN", ".NE")  # 캐나다 상장 접미사 — 야후도 점 그대로 (ABC.V 등, 전체분석 번외)


def yf_symbol(ticker: str) -> str:
    """yfinance 표기로 (BRK.B → BRK-B). 캐나다 접미사(.V·.TO·.CN·.NE)는 점을 그대로 둔다 (account_watch._yf_symbol 과 같다)."""
    t = ticker.strip().upper()
    return t if t.endswith(_CAD_SUFFIX) else t.replace(".", "-")


def bench_symbol(ticker: str) -> str:
    """ISB 벤치마크 야후 심볼 — Pine benchSym Auto 처럼 상장 거래소로 (TSXV ^SPCDNX · TSX ^GSPTSE · 그 밖 SPY).

    CSE(.CN) · NEO(.NE) 는 Pine 에서도 TSX · TSXV 가 아니라 SPY 다.
    """
    t = ticker.strip().upper()
    return next((sym for suffix, sym in CA_BENCH.items() if t.endswith(suffix)), BENCH)


def is_junior(ticker: str) -> bool:
    """TSXV 상장인가 (Pine isJunior) — 매도 방식 Auto 가 Swing · S2 끔 · 시장 필터 끔."""
    return ticker.strip().upper().endswith(".V")


def _frame(raw: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    """yf.download 결과에서 한 종목 OHLCV (없으면 None). 인덱스 = 시간대 없는 날짜."""
    if isinstance(raw.columns, pd.MultiIndex):
        if sym not in raw.columns.get_level_values(0):
            return None
        df = raw[sym]
    else:
        df = raw
    cols = PRICE_COLS + ["Volume"]
    if any(c not in df.columns for c in cols):
        return None
    df = df[cols].dropna(subset=["Close"])
    if df.empty:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    # yfinance 가격은 25.459999… 같은 부동소수점 잡음이 있다. 엔진은 같은 값끼리 비교(피벗 동점 등)를 TradingView 처럼
    # 하므로 잡음이 라벨을 바꾼다 — 2026-10-19 100종목: 반올림 전 일봉 라벨 5개가 연구 캐시와 달랐고 반올림 후 0개.
    df[PRICE_COLS] = df[PRICE_COLS].round(PRICE_DECIMALS)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download(symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    """분할만 반영한 OHLCV (yfinance auto_adjust=False) — {심볼: DataFrame}. 못 받은 심볼은 빠진다."""
    import yfinance as yf  # noqa: PLC0415

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        raw = yf.download(chunk, period=period, interval=interval, auto_adjust=False, actions=False,
                          group_by="ticker", progress=False, threads=True)
        if raw is None or raw.empty:
            continue
        for s in chunk:
            df = _frame(raw, s)
            if df is not None:
                out[s] = df
    return out


def download_daily(symbols: list[str], period: str = HISTORY) -> dict[str, pd.DataFrame]:
    """분할만 반영한 일봉 OHLCV (yfinance auto_adjust=False) — {심볼: DataFrame}. 못 받은 심볼은 빠진다."""
    return _download(symbols, period, "1d")


def download_monthly(symbols: list[str], period: str = MONTHLY_HISTORY) -> dict[str, pd.DataFrame]:
    """분할만 반영한 월봉 (yfinance 1mo — 그 달 1일 날짜 한 줄, 이번 달은 진행 중인 값) — {심볼: DataFrame}."""
    return _download(symbols, period, "1mo")


def drop_unfinished_day(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """장중에 받은 오늘 일봉은 뺀다 (TradingView 에서도 아직 라벨이 안 찍힌다)."""
    if len(df) and not is_bar_complete(df.index[-1].strftime("%Y-%m-%d"), now):
        return df.iloc[:-1]
    return df


def week_complete(bar_dates: pd.DatetimeIndex, now: datetime) -> np.ndarray:
    """주봉이 끝났는지 — 그 주 금요일이 지났거나, 오늘이 금요일이고 장 마감(16:30) 뒤."""
    friday = bar_dates.to_period("W-FRI").end_time.normalize()
    today = pd.Timestamp(now.date())
    return np.asarray((friday < today) | ((friday == today) & (now.time() >= BAR_COMPLETE_TIME)))


def _month_last_weekday(bar_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """그 달 마지막 평일 (공휴일은 모른다)."""
    return pd.DatetimeIndex([pd.offsets.BMonthEnd().rollforward(d) for d in pd.DatetimeIndex(bar_dates).normalize()])


def month_complete(bar_dates: pd.DatetimeIndex, now: datetime) -> np.ndarray:
    """월봉이 끝났는지 — 그 달 마지막 평일이 지났거나, 오늘이 그날이고 장 마감(16:30) 뒤.

    월말 평일이 공휴일이면 하루 늦게 끝난 것으로 본다 (주봉의 금요일 공휴일과 같은 근사).
    """
    last = _month_last_weekday(bar_dates)
    today = pd.Timestamp(now.date())
    return np.asarray((last < today) | ((last == today) & (now.time() >= BAR_COMPLETE_TIME)))


def month_last_days(bar_dates: pd.DatetimeIndex, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """월봉 날짜(야후 1mo = 그 달 1일) → 그 달 마지막 거래일 (일봉에 있는 달) · 일봉에 없는 달은 마지막 평일.

    라벨 날짜를 주봉(그 주 마지막 거래일)과 같은 뜻으로 맞춘다.
    """
    days = pd.DatetimeIndex(days)
    last = pd.Series(days, index=days.to_period("M")).groupby(level=0).max() if len(days) else pd.Series(dtype="datetime64[ns]")
    fallback = _month_last_weekday(bar_dates)
    return pd.DatetimeIndex([last.get(p, f) for p, f in zip(pd.DatetimeIndex(bar_dates).to_period("M"), fallback)])


def label_flags(eng: pd.DataFrame) -> dict[str, np.ndarray]:
    """엔진 봉별 표 → 라벨별 봉 표시 (Pine 라벨 그리는 조건과 같다).

    rev    : BUY·REV (매수 신호 중 바닥 반전형)
    bottom : ◇ BOTTOM (반전 패턴은 나왔지만 시장 조건 때문에 매수 보류)
    top    : ◆ TOP (보유 없을 때 과열 고점 — 보유 중이면 차트에 TRIM·TOP 으로 찍혀 여기서는 빠진다)
    exit   : EXIT (보유 청산 — 손절·트레일·추세 붕괴)
    """
    return {
        "rev": (eng["ev_buy"] & (eng["sig_type"] == "REV")).to_numpy(dtype=bool),
        "bottom": eng["btm"].to_numpy(dtype=bool),
        "top": eng["ev_topwarn"].to_numpy(dtype=bool),
        "exit": eng["ev_exit"].to_numpy(dtype=bool),
    }


def recent_labels(eng: pd.DataFrame, complete: np.ndarray, window: int) -> dict[str, dict[str, Any]]:
    """끝난 봉만 보고, 마지막 끝난 봉에서 window 봉 안에 나온 가장 최근 라벨 → {라벨: {date, ago}}.

    date = 그 봉의 마지막 거래일 (주봉은 그 주 마지막 거래일), ago = 마지막 끝난 봉에서 몇 봉 전 (0 = 그 봉).
    """
    done = np.flatnonzero(complete)
    if done.size == 0:
        return {}
    last = int(done[-1])
    out: dict[str, dict[str, Any]] = {}
    for key, flag in label_flags(eng).items():
        hits = np.flatnonzero(flag[:last + 1])
        hits = hits[hits > last - window]
        if hits.size:
            j = int(hits[-1])
            out[key] = {"date": eng.index[j].strftime("%Y-%m-%d"), "ago": last - j}
    return out


def ticker_labels(daily: pd.DataFrame, markets: dict[str, Any], now: datetime,
                  junior: bool = False, monthly: pd.DataFrame | None = None) -> dict[str, dict[str, Any]]:
    """한 종목 일봉 (+ 월봉) → {"D": {...}, "W": {...}, "M": {...}} (라벨이 하나도 없는 주기는 빠진다).

    markets: 그 종목 벤치마크로 만든 {주기: Market} — "M" 이 없거나 monthly 가 None 이면 월봉은 건너뛴다.
    monthly: 야후 1mo 월봉 전체 기간 (download_monthly). junior: TSXV 상장 (is_junior).
    """
    E = _engine()
    daily = drop_unfinished_day(E.clean_ohlcv(daily), now)
    out: dict[str, dict[str, Any]] = {}
    if len(daily) < 60:
        return out
    for tf in ("D", "W"):
        p = E.IsbParams(tf, junior=junior)
        bars, inp = E.prepare_bars(daily, markets[tf], p)
        eng, _ = E.run_engine(bars, inp, p)
        complete = np.ones(len(eng), dtype=bool) if tf == "D" else week_complete(eng.index, now)
        rec = recent_labels(eng, complete, WINDOW[tf])
        if rec:
            out[tf] = rec
    if monthly is not None and "M" in markets:
        m = E.clean_ohlcv(monthly)
        if len(m):
            p = E.IsbParams("M", junior=junior)
            bars, inp = E.prepare_bars(m, markets["M"], p)
            eng, _ = E.run_engine(bars, inp, p)
            complete = month_complete(eng.index, now)
            eng.index = month_last_days(eng.index, daily.index)
            rec = recent_labels(eng, complete, WINDOW["M"])
            if rec:
                out["M"] = rec
    return out


def isb_labels(tickers: list[str], now: datetime | None = None,
               prices: dict[str, pd.DataFrame] | None = None) -> dict[str, dict[str, Any]]:
    """종목들 ISB 최근 라벨 — {티커: {"D": {"rev": {"date", "ago"}, ...}, "W": {...}, "M": {...}}}.

    라벨이 없는 종목도 {} 로 넣는다 (계산은 됐다는 뜻 — 웹은 '최근 없음'). 가격을 못 받은 종목은 빠진다 (웹은 '정보 없음').
    VIX·DXY 를 못 받으면 빈 dict. 벤치마크(bench_symbol — SPY · ^GSPTSE · ^SPCDNX)를 못 받으면 그 벤치마크 종목만 빠진다
    (다른 지수로 대신 계산하면 TradingView 와 다른 라벨이 나오므로).
    prices: 이미 받은 download_daily 결과 — 주면 다시 받지 않는다 (SQZ 모멘텀과 같은 일봉을 같이 쓴다). 거기 없는 벤치마크만 따로 받는다.
    월봉은 야후 1mo 전체 기간을 따로 받는다 — 월봉 VIX·DXY · 벤치마크 · 종목 월봉이 없으면 그 종목은 월봉만 빠진다.
    """
    now = (now or market_now())
    E = _engine()
    syms = {t: yf_symbol(t) for t in tickers}
    bench_of = {t: bench_symbol(t) for t in tickers}
    benches = sorted(set(bench_of.values()))
    if prices is None:
        prices = download_daily(sorted(set(syms.values()) | set(benches)))
    elif any(b not in prices for b in benches):
        prices = {**prices, **download_daily([b for b in benches if b not in prices])}
    macro_px = download_daily(list(MACRO.values()))
    if any(s not in macro_px for s in MACRO.values()):
        print(f"[web_export] ISB: VIX·DXY 가격 없음 (건너뜀) — 받은 것 {sorted(macro_px)}")
        return {}
    macro = pd.DataFrame({k: macro_px[s]["Close"] for k, s in MACRO.items()})
    monthly = download_monthly(sorted(set(syms.values()) | set(benches) | set(MACRO.values())))
    macro_m = None
    if all(s in monthly for s in MACRO.values()):
        macro_m = pd.DataFrame({k: monthly[s]["Close"] for k, s in MACRO.items()})
    else:
        print(f"[web_export] ISB: 월봉 VIX·DXY 없음 — 월봉 라벨 건너뜀 (받은 월봉 {len(monthly)}개)")
    markets: dict[str, dict[str, Any]] = {}
    for b in benches:
        if b in prices:
            bench = drop_unfinished_day(E.clean_ohlcv(prices[b]), now)
            markets[b] = {tf: E.build_market(bench, macro, E.IsbParams(tf)) for tf in ("D", "W")}
            if macro_m is not None and b in monthly:
                markets[b]["M"] = E.build_market(E.clean_ohlcv(monthly[b]), macro_m, E.IsbParams("M"))
    lost = [b for b in benches if b not in markets]
    if lost:
        n_lost = sum(1 for b in bench_of.values() if b in lost)
        print(f"[web_export] ISB: 벤치마크 가격 없음 {lost} — 그 벤치마크 {n_lost}종목 건너뜀")
    no_m = sum(1 for t, s in syms.items() if s in prices and bench_of[t] in markets
               and (s not in monthly or "M" not in markets[bench_of[t]]))
    if no_m:
        print(f"[web_export] ISB: 월봉 없음 {no_m}종목 — 일봉 · 주봉만")

    out: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for t, s in syms.items():
        mk = markets.get(bench_of[t])
        if s not in prices or mk is None:
            continue
        try:
            out[t] = ticker_labels(prices[s], mk, now, junior=is_junior(t), monthly=monthly.get(s))
        except Exception as exc:  # noqa: BLE001 — 한 종목 실패로 나머지를 버리지 않는다
            failed.append(f"{t}({type(exc).__name__})")
    if failed:
        print(f"[web_export] ISB 계산 실패 {len(failed)}종목: {', '.join(failed[:10])}")
    return out
