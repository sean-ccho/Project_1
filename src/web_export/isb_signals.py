"""웹 정렬용 ISB 라벨 — TradingView 차트의 BUY·REV · ◇ BOTTOM · ◆ TOP · EXIT 가 최근 언제 나왔는지.

엔진은 연구용 포팅 `scripts/isb_engine.py` (pinescript/ISB.pine v9.4, 입력값 기본) 를 그대로 쓴다.
2026-10-12 ABT 등 TradingView 차트와 라벨 날짜가 일치한 그 엔진이다.

- 가격: TradingView 기본 차트(ADJ 꺼짐 = 분할만 반영)와 같게 yfinance auto_adjust=False 의 OHLC.
  배당 조정 가격을 넣으면 배당이 큰 종목은 라벨 날짜가 달라진다 (isb_engine 머리말). 소수 4자리로 반올림한다 (_frame).
- 기록 7년: 엔진이 과거 매매 상태를 이어 가는 구조라 기록이 짧으면 라벨이 달라질 수 있다.
  2026-10-19 120종목 확인 — 5년 이상이면 최근 1년 라벨이 12년치 계산과 일봉·주봉 모두 같았다 (3년은 주봉 7종목 다름).
- 끝나지 않은 봉의 라벨은 안 낸다 (Pine 라벨은 barstate.isconfirmed — 장중 일봉·이번 주 주봉에는 안 찍힌다).
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
# 마지막 완성 봉에서 이 봉 수 안의 가장 최근 라벨만 (일봉 ≈ 3달, 주봉 ≈ 6달)
WINDOW = {"D": 63, "W": 26}
MACRO = {"vix": "^VIX", "dxy": "DX-Y.NYB"}  # ISB 매크로 리스크 (isb_engine.load_macro 와 같은 심볼)
BENCH = "SPY"
CHUNK = 100
PRICE_COLS = ["Open", "High", "Low", "Close"]
PRICE_DECIMALS = 4


def _engine() -> Any:
    """scripts/isb_engine.py 모듈 (scripts 는 패키지가 아니라 경로로 불러온다)."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import isb_engine  # noqa: PLC0415

    return isb_engine


def yf_symbol(ticker: str) -> str:
    """yfinance 표기로 (BRK.B → BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


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


def download_daily(symbols: list[str], period: str = HISTORY) -> dict[str, pd.DataFrame]:
    """분할만 반영한 일봉 OHLCV (yfinance auto_adjust=False) — {심볼: DataFrame}. 못 받은 심볼은 빠진다."""
    import yfinance as yf  # noqa: PLC0415

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        raw = yf.download(chunk, period=period, interval="1d", auto_adjust=False, actions=False,
                          group_by="ticker", progress=False, threads=True)
        if raw is None or raw.empty:
            continue
        for s in chunk:
            df = _frame(raw, s)
            if df is not None:
                out[s] = df
    return out


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


def ticker_labels(daily: pd.DataFrame, markets: dict[str, Any], now: datetime) -> dict[str, dict[str, Any]]:
    """한 종목 일봉 → {"D": {...}, "W": {...}} (라벨이 하나도 없는 주기는 빠진다)."""
    E = _engine()
    daily = drop_unfinished_day(E.clean_ohlcv(daily), now)
    out: dict[str, dict[str, Any]] = {}
    if len(daily) < 60:
        return out
    for tf in ("D", "W"):
        p = E.IsbParams(tf)
        bars, inp = E.prepare_bars(daily, markets[tf], p)
        eng, _ = E.run_engine(bars, inp, p)
        complete = np.ones(len(eng), dtype=bool) if tf == "D" else week_complete(eng.index, now)
        rec = recent_labels(eng, complete, WINDOW[tf])
        if rec:
            out[tf] = rec
    return out


def isb_labels(tickers: list[str], now: datetime | None = None,
               prices: dict[str, pd.DataFrame] | None = None) -> dict[str, dict[str, Any]]:
    """종목들 ISB 최근 라벨 — {티커: {"D": {"rev": {"date", "ago"}, ...}, "W": {...}}}.

    라벨이 없는 종목도 {} 로 넣는다 (계산은 됐다는 뜻 — 웹은 '최근 없음'). 가격을 못 받은 종목은 빠진다 (웹은 '정보 없음').
    SPY·VIX·DXY 를 못 받으면 빈 dict.
    prices: 이미 받은 download_daily 결과 (SPY 포함) — 주면 다시 받지 않는다 (SQZ 모멘텀과 같은 일봉을 같이 쓴다).
    """
    now = (now or market_now())
    E = _engine()
    syms = {t: yf_symbol(t) for t in tickers}
    if prices is None:
        prices = download_daily(sorted(set(syms.values()) | {BENCH}))
    macro_px = download_daily(list(MACRO.values()))
    if BENCH not in prices or any(s not in macro_px for s in MACRO.values()):
        print(f"[web_export] ISB: SPY·VIX·DXY 가격 없음 (건너뜀) — 받은 것 {sorted(macro_px)}")
        return {}
    macro = pd.DataFrame({k: macro_px[s]["Close"] for k, s in MACRO.items()})
    spy = drop_unfinished_day(E.clean_ohlcv(prices[BENCH]), now)
    markets = {tf: E.build_market(spy, macro, E.IsbParams(tf)) for tf in ("D", "W")}

    out: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for t, s in syms.items():
        if s not in prices:
            continue
        try:
            out[t] = ticker_labels(prices[s], markets, now)
        except Exception as exc:  # noqa: BLE001 — 한 종목 실패로 나머지를 버리지 않는다
            failed.append(f"{t}({type(exc).__name__})")
    if failed:
        print(f"[web_export] ISB 계산 실패 {len(failed)}종목: {', '.join(failed[:10])}")
    return out
