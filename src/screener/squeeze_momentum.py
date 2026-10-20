"""TradingView 'Squeeze Momentum Indicator [LazyBear]' (SQZMOM_LB, 기본값) 와 같은 계산.

TradingView 에서 가장 많이 쓰는 스퀴즈 모멘텀 (John Carter TTM Squeeze 를 LazyBear 가 옮긴 것). 원본 Pine (기본값):

    length = 20, mult = 2.0, lengthKC = 20, multKC = 1.5, useTrueRange = true
    basis = sma(close, length)
    dev = multKC * stdev(close, length)          ← 원본 그대로: BB 폭에 mult(2.0)가 아니라 multKC(1.5)를 쓴다
    upperBB = basis + dev ; lowerBB = basis - dev
    ma = sma(close, lengthKC) ; range = tr ; rangema = sma(range, lengthKC)
    upperKC = ma + rangema * multKC ; lowerKC = ma - rangema * multKC
    sqzOn  = lowerBB > lowerKC and upperBB < upperKC      → 검은 십자 (스퀴즈 중)
    sqzOff = lowerBB < lowerKC and upperBB > upperKC      → 회색 십자 (스퀴즈 풀림)
    noSqz  = 둘 다 아님                                  → 파란 십자
    val = linreg(close - avg(avg(highest(high, lengthKC), lowest(low, lengthKC)), sma(close, lengthKC)), lengthKC, 0)
    막대 색 = val > 0 ? (val > nz(val[1]) ? lime : green) : (val < nz(val[1]) ? red : maroon)

Pine 세부:
- stdev = 모집단 표준편차 (n 으로 나눔). tr = ta.tr(false): 첫 봉(직전 종가 없음)은 값 없음.
- linreg(src, n, 0) = 최근 n봉 최소제곱 직선의 이번 봉 값 (= 3·WMA − 2·SMA).
- 가격·봉은 ma_cross 와 같다: 분할만 반영한 가격 (TradingView 기본 차트, ADJ 꺼짐 = yfinance auto_adjust=False),
  주봉 = 그 주(월~금) · 월봉 = 달력 월, 날짜 = 그 봉의 마지막 거래일 (다음 거래일 시가 진입이 미래를 안 보게).

매매 사건 (끝난 봉 기준, 연구용 — scripts/sqz_research.py):
- FIRE  = 스퀴즈 풀림 (직전 봉 검은 십자 → 이번 봉 아님) + 연두 막대 (val > 0 이고 오름). LazyBear 설명의 Carter 규칙.
- MOM0  = 모멘텀 0선 상향 돌파 (Pine ta.crossover(val, 0)).
- TURN  = 빨강 → 갈색 (0 아래에서 떨어지던 모멘텀이 처음 멈추거나 오름).
- LIME_SQZ = 검은 십자(직전 봉 스퀴즈 중)에 눌려 있다가 막대가 연두로 바뀐 봉 (직전 봉은 연두 아님). 2026-10-19 사용자 규칙
  "스퀴즈로 눌려있다가 연두색으로 변할 때". 이번 봉 십자는 검정·회색 모두 된다 (FIRE 와 겹칠 수 있음).
- 청산 FADE = 모멘텀이 떨어진 첫 봉 (val < val[1] — 연두→초록, 갈색→빨강). "색이 바뀌면 나온다" (LazyBear 설명).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from screener.ma_cross import bar_complete

Timeframe = Literal["D", "W", "M"]


@dataclass(frozen=True)
class SqzParams:
    """SQZMOM_LB 입력값 (TradingView 기본값)."""

    length: int = 20       # BB Length
    mult: float = 2.0      # BB MultFactor — 원본 코드는 이 값을 쓰지 않는다 (dev 에 multKC 사용)
    length_kc: int = 20    # KC Length
    mult_kc: float = 1.5   # KC MultFactor
    use_true_range: bool = True


DEFAULT = SqzParams()


def to_ohlc_bars(df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    """일봉 OHLC → 그 봉의 OHLC. 날짜 = 봉의 마지막 거래일. 종가가 없는 날(거래 없음)은 먼저 뺀다."""
    d = df[["Open", "High", "Low", "Close"]].astype("float64")
    d = d[d["Close"].notna()]
    if tf == "D" or d.empty:
        return d
    key = d.index.to_period("W-FRI" if tf == "W" else "M")
    g = d.groupby(key)
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last()})
    out.index = pd.DatetimeIndex(d.index.to_series().groupby(key).max().to_numpy())
    return out


def pine_linreg(x: np.ndarray, n: int) -> np.ndarray:
    """Pine linreg(x, n, 0): 최근 n봉(오래된 봉 x=0 … 이번 봉 x=n−1) 최소제곱 직선의 이번 봉 값. 창에 NaN 이 있으면 NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.size, np.nan)
    if x.size < n:
        return out
    win = np.lib.stride_tricks.sliding_window_view(x, n)
    t = np.arange(n, dtype=float)
    tm = t.mean()
    ym = win.mean(axis=1)
    slope = ((win - ym[:, None]) @ (t - tm)) / float(((t - tm) ** 2).sum())
    out[n - 1:] = ym + slope * (n - 1 - tm)
    return out


def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Pine tr (= ta.tr(false)): 직전 종가가 없으면 NaN."""
    pc = np.r_[np.nan, c[:-1]]
    return np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc))) * np.where(np.isnan(pc), np.nan, 1.0)


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _shift(x: np.ndarray, k: int, fill: object) -> np.ndarray:
    """Pine x[k] — k 봉 전 값 (앞쪽은 fill)."""
    out = np.empty_like(x)
    out[:k] = fill
    out[k:] = x[: max(0, x.size - k)]
    return out


def squeeze_frame(daily: pd.DataFrame, tf: Timeframe, as_of: pd.Timestamp | None = None,
                  p: SqzParams = DEFAULT) -> pd.DataFrame:
    """봉마다 스퀴즈 모멘텀 값·색·사건.

    Args:
        daily: 일봉 OHLC (Open·High·Low·Close, 분할만 반영, 날짜 인덱스).
        tf: "D" / "W" / "M".
        as_of: 마지막 봉이 끝났는지 판단할 날짜 (없으면 마지막 거래일).
        p: 입력값 (기본 = TradingView 기본값).

    Returns:
        인덱스 = 봉의 마지막 거래일. 컬럼:
        close · val · sqz_on · sqz_off · no_sqz · dot(black/gray/blue) · color(lime/green/red/maroon)
        · rising(val > val[1]) · falling(val < val[1]) · fire · mom0 · turn · lime_sqz
        · valid(val·val[1]·val[2]·sqz_on[1] 있음)
        · n_bar(첫 봉 = 0) · complete(봉이 끝났는지).
    """
    bars = to_ohlc_bars(daily, tf)
    out = pd.DataFrame(index=bars.index)
    if bars.empty:
        return out.assign(close=[], val=[], complete=[])
    h, l, c = (bars[k].to_numpy() for k in ("High", "Low", "Close"))
    basis = _sma(c, p.length)
    dev = p.mult_kc * pd.Series(c).rolling(p.length, min_periods=p.length).std(ddof=0).to_numpy()
    upper_bb, lower_bb = basis + dev, basis - dev
    ma = _sma(c, p.length_kc)
    rng = _true_range(h, l, c) if p.use_true_range else h - l
    rangema = _sma(rng, p.length_kc)
    upper_kc, lower_kc = ma + rangema * p.mult_kc, ma - rangema * p.mult_kc
    with np.errstate(invalid="ignore"):
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_sqz = ~sqz_on & ~sqz_off
    hh = pd.Series(h).rolling(p.length_kc, min_periods=p.length_kc).max().to_numpy()
    ll = pd.Series(l).rolling(p.length_kc, min_periods=p.length_kc).min().to_numpy()
    val = pine_linreg(c - ((hh + ll) / 2 + ma) / 2, p.length_kc)

    prev, prev2 = _shift(val, 1, np.nan), _shift(val, 2, np.nan)
    prev_nz = np.nan_to_num(prev)
    with np.errstate(invalid="ignore"):
        up = val > 0
        color = np.where(up, np.where(val > prev_nz, "lime", "green"), np.where(val < prev_nz, "red", "maroon"))
        rising, falling = val > prev, val < prev
    color = np.where(np.isnan(val), "", color)
    prev_on = _shift(sqz_on, 1, False)
    prev_color = _shift(color, 1, "")
    has_prev_on = _shift(~np.isnan(rangema) & ~np.isnan(dev), 1, False)
    valid = ~np.isnan(val) & ~np.isnan(prev) & ~np.isnan(prev2) & has_prev_on

    out["close"] = c
    out["val"] = val
    out["sqz_on"], out["sqz_off"], out["no_sqz"] = sqz_on, sqz_off, no_sqz
    out["dot"] = np.where(no_sqz, "blue", np.where(sqz_on, "black", "gray"))
    out["color"] = color
    out["rising"], out["falling"] = rising & valid, falling & valid
    with np.errstate(invalid="ignore"):
        out["fire"] = valid & prev_on & ~sqz_on & (color == "lime")
        out["mom0"] = valid & (val > 0) & (prev <= 0)
    out["turn"] = valid & (color == "maroon") & (prev_color == "red")
    out["lime_sqz"] = valid & prev_on & (color == "lime") & (prev_color != "lime")
    out["valid"] = valid
    out["n_bar"] = np.arange(len(out))
    as_of = bars.index[-1] if as_of is None else as_of
    out["complete"] = bar_complete(out.index, tf, pd.Timestamp(as_of))
    return out


ENTRIES = ("fire", "mom0", "turn")


def system_position(f: pd.DataFrame, entry: str) -> np.ndarray:
    """매매 규칙의 봉별 보유 상태 (그 봉 종가에서 결정 → 다음 거래일 시가부터 적용).

    미보유 + 진입 사건 → 보유. 보유 + 모멘텀이 떨어진 봉(FADE) → 미보유. 진입 봉은 떨어지는 봉이 아니라 같은 봉 청산은 없다.
    끝난 봉만 넣는다.
    """
    ev = f[entry].to_numpy(dtype=bool)
    fall = f["falling"].to_numpy(dtype=bool)
    pos = np.zeros(len(f), dtype=bool)
    held = False
    for i in range(len(f)):
        if held and fall[i]:
            held = False
        elif not held and ev[i]:
            held = True
        pos[i] = held
    return pos
