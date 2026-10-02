"""일봉 OHLC 리스트로 계산하는 기술지표 (pandas 없이 동작).

PT-2/PT-3 청산 판단에 쓴다. 실거래(account_engine)와 백테스트(account_backtest)가
같은 함수를 써서 청산 결과가 일치한다.
EMA는 ta 라이브러리와 같은 방식(adjust=False), ATR·RSI·ADX는 Wilder 평활.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import Any, Mapping, Sequence

Bar = dict[str, Any]
NAN = float("nan")


def is_num(x: Any) -> bool:
    """유한한 숫자인지 (None/NaN/inf 제외)."""
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def num(row: Mapping[str, Any] | None, key: str, default: float = NAN) -> float:
    """스냅샷 행에서 숫자 값을 꺼낸다. 없거나 숫자가 아니면 default."""
    if row is None:
        return default
    val = row.get(key)
    return float(val) if is_num(val) else default


class PercentileRanker:
    """그날 스냅샷 전체(유니버스) 대비 백분위 (0~1, 같은 값은 중간 순위). 값이 없으면 0.5."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]):
        self._sorted = {
            col: sorted(v for v in (num(r, col) for r in rows) if is_num(v)) for col in columns
        }

    def rank(self, column: str, row: Mapping[str, Any]) -> float:
        values = self._sorted.get(column) or []
        v = num(row, column)
        if not values or not is_num(v):
            return 0.5
        lo, hi = bisect_left(values, v), bisect_right(values, v)
        return (lo + hi) / 2 / len(values)


def ema_series(values: Sequence[float], period: int) -> list[float]:
    """지수이동평균. 처음 period-1개는 NaN."""
    alpha = 2.0 / (period + 1)
    out: list[float] = []
    prev = NAN
    for i, v in enumerate(values):
        prev = v if i == 0 else alpha * v + (1 - alpha) * prev
        out.append(prev if i >= period - 1 else NAN)
    return out


def _true_ranges(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[float]:
    trs = [highs[0] - lows[0]] if highs else []
    for i in range(1, len(closes)):
        pc = closes[i - 1]
        trs.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
    return trs


def atr_series(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> list[float]:
    """ATR (Wilder). 처음 period-1개는 NaN."""
    trs = _true_ranges(highs, lows, closes)
    out = [NAN] * len(trs)
    if len(trs) < period:
        return out
    atr = sum(trs[:period]) / period
    out[period - 1] = atr
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        out[i] = atr
    return out


def rsi_series(closes: Sequence[float], period: int = 14) -> list[float]:
    """RSI (Wilder, ta와 같은 EWM alpha=1/period). 처음 period-1개는 NaN."""
    alpha = 1.0 / period
    out: list[float] = []
    avg_up = avg_dn = 0.0
    for i in range(len(closes)):
        diff = closes[i] - closes[i - 1] if i > 0 else 0.0
        up, dn = max(diff, 0.0), max(-diff, 0.0)
        if i == 0:
            avg_up, avg_dn = up, dn
        else:
            avg_up = alpha * up + (1 - alpha) * avg_up
            avg_dn = alpha * dn + (1 - alpha) * avg_dn
        if i < period - 1:
            out.append(NAN)
        elif avg_dn == 0:
            out.append(100.0 if avg_up > 0 else 50.0)
        else:
            out.append(100.0 - 100.0 / (1.0 + avg_up / avg_dn))
    return out


def adx_series(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> list[float]:
    """ADX (Wilder). 2*period개 이상 필요, 그 전은 NaN."""
    n = len(closes)
    out = [NAN] * n
    if n < 2 * period + 1:
        return out
    trs, pdms, ndms = [0.0], [0.0], [0.0]
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        pdms.append(up if up > down and up > 0 else 0.0)
        ndms.append(down if down > up and down > 0 else 0.0)
        pc = closes[i - 1]
        trs.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))

    s_tr = sum(trs[1:period + 1])
    s_p = sum(pdms[1:period + 1])
    s_n = sum(ndms[1:period + 1])
    dxs: list[float] = []
    for i in range(period, n):
        if i > period:
            s_tr = s_tr - s_tr / period + trs[i]
            s_p = s_p - s_p / period + pdms[i]
            s_n = s_n - s_n / period + ndms[i]
        pdi = 100.0 * s_p / s_tr if s_tr else 0.0
        ndi = 100.0 * s_n / s_tr if s_tr else 0.0
        dxs.append(100.0 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) else 0.0)

    adx = sum(dxs[:period]) / period
    out[2 * period - 1] = adx
    for j in range(period, len(dxs)):
        adx = (adx * (period - 1) + dxs[j]) / period
        out[period + j] = adx
    return out


def bollinger_pband(closes: Sequence[float], window: int = 20, dev: float = 2.0) -> float:
    """마지막 봉의 볼린저 %B (ta와 같은 모집단 표준편차)."""
    if len(closes) < window:
        return NAN
    w = closes[-window:]
    mean = sum(w) / window
    std = math.sqrt(sum((x - mean) ** 2 for x in w) / window)
    if std == 0:
        return 0.5
    lower = mean - dev * std
    return (closes[-1] - lower) / (2 * dev * std)


def compute_indicators(bars: Sequence[Bar]) -> dict[str, float]:
    """마지막 봉 기준 청산용 지표. 데이터가 모자라면 해당 값은 NaN."""
    if not bars:
        return {}
    closes = [float(b["close"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]

    ema20 = ema_series(closes, 20)
    ema50 = ema_series(closes, 50)
    streak = 0
    for c, e in zip(reversed(closes), reversed(ema50)):
        if is_num(e) and c < e:
            streak += 1
        else:
            break

    return {
        "close": closes[-1],
        "ema20": ema20[-1],
        "ema50": ema50[-1],
        "atr": atr_series(highs, lows, closes, 14)[-1],
        "adx": adx_series(highs, lows, closes, 14)[-1],
        "rsi": rsi_series(closes, 14)[-1],
        "bb_pband": bollinger_pband(closes, 20, 2.0),
        "ret_20d": closes[-1] / closes[-21] - 1.0 if len(closes) > 20 and closes[-21] else NAN,
        "below_ema50_streak": float(streak),
    }
