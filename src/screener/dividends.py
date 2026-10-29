"""지금 받는 배당 (2026-10-28) — 증권사 화면처럼 '최근 정기 배당 1번 × 1년에 주는 횟수'. 표시 전용, 점수에 안 쓴다.

예전 칸(features 의 dividend_yield · annual_dividend)은 최근 1년에 실제로 준 합계라서
특별배당·청산 배당이 있으면 수백 %가 되고 (AIV 167% · NLOP 205%), 배당을 줄이면 옛 배당이 1년 동안 남는다
(TELUS 0.418 → 0.188: 합계로는 13.9%, 지금 배당으로는 6.3%).
특별배당만 주는 종목은 일봉 배당 기록만으로는 정기 배당과 구별이 잘 안 돼서, Yahoo 의 정기 배당(dividendRate)이
없으면 '정기 배당 없음'으로 본다 — 합치는 곳은 web_export.dashboard.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

FREQ_STEPS = ((45, 12), (75, 6), (135, 4), (270, 2))  # 배당 사이 날짜(중앙값)가 이 이하면 1년에 이 횟수, 넘으면 1번
LOOKBACK_DAYS = 3 * 365  # 이만큼의 배당 기록으로 횟수·정기 금액을 본다
SPECIAL_MULT = 2.0  # 직전 정기 배당 중앙값의 이 배수를 넘고 다음 배당이 같은 금액이 아니면 특별배당


def _per_year(gap_days: float) -> int:
    for limit, n in FREQ_STEPS:
        if gap_days <= limit:
            return n
    return 1


def forward_dividend(div: pd.Series, asof: Optional[pd.Timestamp] = None) -> float:
    """일봉 Dividends 칸 → 1주당 1년 배당(달러) = 최근 정기 배당 × 1년 횟수. 배당이 없거나 끊겼으면 0.

    - 횟수: 최근 정기 배당 사이 날짜의 중앙값 (한 달 12 · 분기 4 · 반년 2 · 1년 1)
    - 특별배당: 직전 정기 배당들 중앙값의 2배를 넘는데 다음 배당이 같은 금액(±10%)이 아니면 빼고 본다
      (배당을 크게 올린 첫 번은 다음 배당이 나올 때까지 옛 금액으로 보인다)
    - 끊김: 마지막 정기 배당 뒤로 평소 간격의 1.5배 + 30일이 지나도록 없으면 0
    """
    if div is None or div.empty:
        return 0.0
    asof = pd.Timestamp(asof if asof is not None else div.index[-1])
    idx = pd.DatetimeIndex(div.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
        asof = asof.tz_localize(None) if asof.tzinfo is not None else asof
    s = pd.Series(np.asarray(div, dtype=float), index=idx)
    s = s[(s > 0) & (s.index > asof - pd.Timedelta(days=LOOKBACK_DAYS)) & (s.index <= asof)]
    if s.empty:
        return 0.0

    amounts = s.to_numpy()
    regular: list[int] = []
    for i, a in enumerate(amounts):
        prev = [amounts[j] for j in regular[-4:]]
        nxt = amounts[i + 1] if i + 1 < len(amounts) else None
        repeats = nxt is not None and abs(nxt - a) <= 0.1 * a
        if prev and a > SPECIAL_MULT * float(np.median(prev)) and not repeats:
            continue
        regular.append(i)
    if not regular:
        return 0.0

    dates = s.index[regular]
    gaps = pd.Series(dates).diff().dt.total_seconds().dropna().to_numpy() / 86_400  # 날짜 단위(ns·us·s)와 상관없이
    gap = float(np.median(gaps[-5:])) if len(gaps) else 365.0
    if (asof - dates[-1]).days > 1.5 * gap + 30:
        return 0.0
    return float(amounts[regular[-1]]) * _per_year(gap)


MAX_YIELD = 1.0  # 이보다 크면 데이터 오류로 본다 (GMEX: Yahoo 배당 기록이 1주당 $453 — 수익률 5만 %)


def _num(v: object) -> Optional[float]:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def dividend_now(rate: object, yahoo_rate: object, price: object) -> tuple[Optional[float], Optional[float]]:
    """웹·시트에 보일 배당 → (수익률, 1년 1주당 달러). 모르면 (None, None).

    rate: forward_dividend (일봉 기록) · yahoo_rate: Yahoo dividendRate (0 = 정기 배당 없음, NaN = 정보 없음).
    기록으로는 배당이 있어도 Yahoo 가 정기 배당이 없다고 하면 특별배당·청산 배당만 준 것으로 보고 0.
    """
    r, y, p = _num(rate), _num(yahoo_rate), _num(price)
    if r is None:
        r = y  # 기록 계산이 없는 옛 스냅샷
    elif r > 0 and y == 0:
        r = 0.0
    if r is None:
        return None, None
    if r == 0:
        return 0.0, 0.0
    if p is None or p <= 0:
        return None, round(r, 4)
    if r / p > MAX_YIELD:
        return None, None
    return round(r / p, 4), round(r, 4)
