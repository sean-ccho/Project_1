"""웹 목록·정렬용 SQZ 모멘텀 상태 — TradingView 'Squeeze Momentum Indicator [LazyBear]' (기본값) 와 같은 값.

계산은 screener.squeeze_momentum (Pine 그대로, 2026-10-19 사용자 차트와 AAPL·MSFT·KO 일·주·월 값 일치 확인).
가격은 ISB 라벨과 같은 일봉(yfinance auto_adjust=False = 분할만 반영, TradingView ADJ 꺼짐)을 받아 쓴다.

봉마다 내보내는 것 (끝난 봉 기준):
- dot   : 마지막 끝난 봉 십자 (black = 스퀴즈 중, gray = 풀림, blue = 둘 다 아님)
- run   : 검은 십자가 그 봉까지 몇 봉째 이어지는지 (검정이 아니면 0)
- color : 막대 색 (lime · green · red · maroon) · val = 막대 값 · val_pct = val ÷ 종가 × 100 (종목끼리 비교용)
- fire  : 최근 FIRE_BARS 봉 안에 검은 십자 → 회색으로 풀린 봉 {ago(0 = 마지막 끝난 봉), run(풀리기 전 눌린 봉 수), color}
          막대 색과 상관없이 낸다 (차트에서도 똑같이 회색 십자로 보인다 — 방향은 color 로 구분)
- lime  : 최근 FIRE_BARS 봉 안에 '눌린 채로 막대가 연두로 바뀐' 봉 {ago} (screener.squeeze_momentum 의 lime_sqz)
- live  : 이번 주·이번 달처럼 아직 안 끝난 봉의 지금 값 {date, dot, color, val} — TradingView 화면과 비교용 (일봉은 없음)
fire · lime · live 는 없으면 키 자체가 빠진다.

봉이 끝났는지: 일봉 = 그날 16:30(ET) 뒤, 주봉 = 그 주 금요일 16:30 뒤, 월봉 = 그 달 마지막 평일 16:30 뒤.
마지막 날이 휴장이면 하루 더 '진행 중'으로 둔다 (보수적).

표시 전용 — 2026-10-19 10년 백테스트(tier3/SQZ_REPORT.md)에서 매수 신호로 효과가 없었다. 점수·CCS·메일에 넣지 않는다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from paper_trading.market_date import BAR_COMPLETE_TIME, is_bar_complete, market_now
from screener.squeeze_momentum import squeeze_frame

TFS = ("D", "W", "M")
FIRE_BARS = 3  # 풀림·연두 전환을 보여 주는 봉 수 (주봉 3주 · 월봉 3달)


def period_complete(bar_dates: pd.DatetimeIndex, tf: str, now: datetime) -> np.ndarray:
    """주·월봉이 끝났는지 — 그 기간 마지막 평일 16:30(ET)이 지났으면 True."""
    end = bar_dates.to_period("W-FRI" if tf == "W" else "M").end_time.normalize()
    end = end - pd.to_timedelta(np.maximum(0, end.dayofweek - 4), unit="D")
    today = pd.Timestamp(now.date())
    return np.asarray((end < today) | ((end == today) & (now.time() >= BAR_COMPLETE_TIME)))


def clean_daily(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """가격 없는 날 빼고, 아직 안 끝난 오늘 일봉도 뺀다."""
    px = df[["Open", "High", "Low", "Close"]].astype(float)
    px = px[px.notna().all(axis=1) & (px > 0).all(axis=1)].sort_index()
    if len(px) and not is_bar_complete(px.index[-1].strftime("%Y-%m-%d"), now):
        px = px.iloc[:-1]
    return px


def _num(v: float, nd: int) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def tf_state(daily: pd.DataFrame, tf: str, now: datetime) -> dict[str, Any] | None:
    """한 종목·한 봉의 상태 (값이 아직 없으면 None). daily = clean_daily 를 거친 일봉."""
    if daily.empty:
        return None
    f = squeeze_frame(daily, tf)
    if f.empty:
        return None
    complete = np.ones(len(f), dtype=bool) if tf == "D" else period_complete(f.index, tf, now)
    live = None
    if not complete[-1]:
        r = f.iloc[-1]
        if np.isfinite(r["val"]):
            live = {"date": f.index[-1].strftime("%Y-%m-%d"), "dot": str(r["dot"]), "color": str(r["color"]),
                    "val": _num(r["val"], 4)}
    done = f[complete]
    if done.empty or not np.isfinite(done["val"].iloc[-1]):
        return None
    on = done["sqz_on"].to_numpy(dtype=bool)
    valid = done["valid"].to_numpy(dtype=bool)
    last = len(done) - 1
    r = done.iloc[last]

    def black_run(end: int) -> int:
        k = 0
        while end - k >= 0 and on[end - k]:
            k += 1
        return k

    fire = None
    lime = None
    for ago in range(min(FIRE_BARS, last + 1)):
        i = last - ago
        if fire is None and i >= 1 and valid[i] and on[i - 1] and not on[i]:
            fire = {"ago": ago, "run": black_run(i - 1), "color": str(done["color"].iloc[i])}
        if lime is None and bool(done["lime_sqz"].iloc[i]):
            lime = {"ago": ago}
    close = float(r["close"])
    out = {"date": done.index[last].strftime("%Y-%m-%d"), "dot": str(r["dot"]), "run": black_run(last),
           "color": str(r["color"]), "val": _num(r["val"], 4),
           "val_pct": _num(r["val"] / close * 100 if close > 0 else np.nan, 2)}
    # 없는 것은 키를 빼서 dashboard.json 을 줄인다 (352종목 × 3봉에서 약 30 KB)
    out.update({k: v for k, v in (("fire", fire), ("lime", lime), ("live", live)) if v is not None})
    return out


def ticker_sqz(daily: pd.DataFrame, now: datetime | None = None) -> dict[str, dict[str, Any]]:
    """한 종목 일봉(분할만 반영 OHLC) → {"D": {...}, "W": {...}, "M": {...}} (값이 없는 봉은 빠진다)."""
    now = now or market_now()
    d = clean_daily(daily, now)
    out: dict[str, dict[str, Any]] = {}
    for tf in TFS:
        st = tf_state(d, tf, now)
        if st is not None:
            out[tf] = st
    return out


def sqz_states(prices: dict[str, pd.DataFrame], now: datetime | None = None) -> dict[str, dict[str, Any]]:
    """{티커: 일봉} → {티커: ticker_sqz}. 한 종목 실패로 나머지를 버리지 않는다."""
    now = now or market_now()
    out: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for t, df in prices.items():
        try:
            out[t] = ticker_sqz(df, now)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{t}({type(exc).__name__})")
    if failed:
        print(f"[web_export] SQZ 모멘텀 계산 실패 {len(failed)}종목: {', '.join(failed[:10])}")
    return out
