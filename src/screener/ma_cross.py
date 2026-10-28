"""TradingView 와 같은 이동평균 크로스 계산 (골든·데드크로스, 골든크로스 임박).

차트와 똑같이 보이게 하는 것이 목적이다. 2026-10-16 HL·T·UNH 일/주/월 SeanEMA 범례와 1센트까지 맞춰 봤다.

- 가격: 분할만 반영한 종가 (TradingView 기본 차트, ADJ 꺼짐) = yfinance auto_adjust=False 의 Close.
  배당 조정 가격(auto_adjust=True)을 넣으면 배당이 큰 종목은 선이 몇 % 어긋난다.
- EMA = Pine ta.ema: 처음 n−1 봉은 값 없음(TV 범례 '∅'), n번째 봉 = 처음 n봉 단순평균, 그다음부터
  alpha = 2/(n+1) 로 이어간다. 2026-10-16 상장 몇 년 안 된 종목(ABNB·APP·BBAI 주봉 EMA200)으로 확인 —
  첫 봉 값에서 시작하면 수 % 어긋난다. 이어지는 계산이라 긴 선일수록 오래된 기록이 필요하다 (월봉 EMA200 은 상장 첫날부터).
- SMA = Pine ta.sma: 최근 n봉 평균 (n봉이 차기 전에는 없음).
- 봉: 일봉 그대로 · 주봉 = 그 주(월~금) · 월봉 = 달력 월. 값은 patterns.get_weekly_data / get_monthly_data 와 같고,
  날짜는 그 봉의 **마지막 거래일**로 붙인다 (그다음 거래일 시가에 진입하는 백테스트가 미래를 안 보게).
  거래가 없는 날(상장 전·거래 정지)은 봉이 없는 것으로 친다 — TradingView 와 같다.
- 교차: Pine ta.crossover — 이번 봉 단기 > 장기, 직전 봉 단기 ≤ 장기 (데드크로스는 반대).
- 임박 (GI): 지금 앱 규칙(patterns.detect_golden_cross 의 임박 부분)을 봉마다 그대로 계산한다.
  GI+ 는 GI 에 '종가 > 단기선'을 더한 것 (가격은 떨어지는데 장기선이 내려와서 생기는 가짜 임박을 막는다).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Timeframe = Literal["D", "W", "M"]
MaKind = Literal["ema", "sma"]

IMMINENT_GAP = 0.05  # patterns.detect_golden_cross: 갭 5% 이내


@dataclass(frozen=True)
class TfRule:
    """봉별 앱 규칙 (patterns.detect_all_patterns / detect_weekly_patterns / detect_monthly_patterns 와 같은 값)."""

    recent_window: int        # 교차 라벨이 남아 있는 봉 수 (이 안에 교차가 있으면 임박이 아니다)
    slope_lookback: int       # 갭이 좁혀지는지 비교하는 봉 (iloc[-slope_lookback] → slope_lookback-1 봉 전)
    short_slope_confirm: bool  # 단기선이 slope_lookback-1 봉 전보다 올라야 함
    close_filter_ema: int     # >0 이면 종가 > EMA(이 기간) 이어야 임박


TF_RULES: dict[str, TfRule] = {
    "D": TfRule(recent_window=5, slope_lookback=10, short_slope_confirm=False, close_filter_ema=10),
    "W": TfRule(recent_window=5, slope_lookback=5, short_slope_confirm=True, close_filter_ema=0),
    "M": TfRule(recent_window=3, slope_lookback=3, short_slope_confirm=True, close_filter_ema=0),
}


def to_bars(close: pd.Series, tf: Timeframe) -> pd.Series:
    """일봉 종가 → 그 봉의 종가 시리즈. 날짜 = 봉의 마지막 거래일. 빈 값(거래 없는 날)은 먼저 뺀다."""
    s = close.dropna().astype("float64")
    if tf == "D" or s.empty:
        return s
    freq = "W-FRI" if tf == "W" else "M"
    key = s.index.to_period(freq)
    last_close = s.groupby(key).last()
    last_date = s.index.to_series().groupby(key).max()
    return pd.Series(last_close.to_numpy(), index=pd.DatetimeIndex(last_date.to_numpy()), name=close.name)


def last_weekday(period_end: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """그 기간의 마지막 평일 — 토·일요일에 끝나는 달은 그 전 금요일 (주봉 기간은 원래 금요일에 끝난다)."""
    back = np.maximum(np.asarray(period_end.dayofweek) - 4, 0)
    return period_end - pd.to_timedelta(back, unit="D")


def bar_complete(bar_dates: pd.DatetimeIndex, tf: Timeframe, as_of: pd.Timestamp) -> np.ndarray:
    """봉이 끝났는지. 일봉은 항상 True. 주·월봉은 그 기간의 마지막 평일(주봉 금요일 · 월봉 말일, 주말이면 그 전 금요일)이
    as_of 이전이면 True — TradingView 도 그날 장이 끝나면 그 봉을 끝난 봉으로 그린다.

    2026-10-27: 예전엔 달력상 말일이라 토·일요일에 끝나는 달은 금요일 장 뒤 밤 실행에서 아직 '진행 중'이었다.
    공휴일(성금요일 등)인 마지막 평일은 모르므로 그 전 거래일에는 끝난 봉도 False (보수적으로 '진행 중').
    """
    if tf == "D":
        return np.ones(len(bar_dates), dtype=bool)
    freq = "W-FRI" if tf == "W" else "M"
    period_end = last_weekday(bar_dates.to_period(freq).end_time.normalize())
    return np.asarray(period_end <= pd.Timestamp(as_of).normalize())


def moving_average(s: pd.Series, n: int, kind: MaKind) -> pd.Series:
    """Pine ta.ema / ta.sma 와 같은 이동평균 (빈 값 없는 봉 시리즈를 넣는다)."""
    if kind == "ema":
        return pine_ema(s, n)
    if kind == "sma":
        return s.rolling(n).mean()
    raise ValueError(f"kind 는 'ema' 또는 'sma': {kind}")


def pine_ema(s: pd.Series, n: int) -> pd.Series:
    """Pine ta.ema: n−1 봉까지 NaN, n번째 봉 = 처음 n봉 평균, 이후 alpha = 2/(n+1)."""
    x = s.astype("float64").copy()
    if len(x) < n:
        return pd.Series(np.nan, index=s.index, name=s.name)
    seed = float(x.iloc[:n].mean())
    x.iloc[: n - 1] = np.nan
    x.iloc[n - 1] = seed
    return x.ewm(span=n, adjust=False).mean()


def cross_frame(close: pd.Series, short: int, long: int, kind: MaKind, tf: Timeframe,
                as_of: pd.Timestamp | None = None, *, bars: bool = False, tail: int | None = None) -> pd.DataFrame:
    """봉마다 크로스 상태와 사건.

    Args:
        close: 일봉 종가 (분할만 반영, 날짜 인덱스).
        short, long: 단기·장기 이동평균 기간.
        kind: "ema" 또는 "sma".
        tf: "D" / "W" / "M".
        as_of: 마지막 봉이 끝났는지 판단할 날짜 (없으면 마지막 거래일).
        bars: close 가 이미 그 봉의 종가(봉마다 한 값, 날짜 = 마지막 거래일)면 True — 일봉 → 봉 변환을 건너뛴다.
        tail: 주면 선(이동평균 · 종가 필터 EMA)은 전체 기록으로 계산하고, 교차 · 임박 · 끝난 봉은 마지막 tail 봉에서만
            계산한다 (2026-10-27, 백테스트 속도). 마지막 봉 판정이 보는 창(최대 약 25봉)보다 길면 그 봉들의 값은 전체로
            계산한 것과 같다. 이때 bars_since 는 그 안에서만 센다 (교차가 그 전이면 −1).

    Returns:
        인덱스 = 봉의 마지막 거래일. 컬럼:
        close · ma_s · ma_l · gap((단기−장기)/장기) · golden(단기 > 장기) · gc(골든크로스) · dc(데드크로스)
        · bars_since(마지막 교차 후 봉 수, 교차 봉 = 0, 교차 전 = −1) · n_bar(첫 봉 = 0)
        · gi(임박, 앱 규칙) · gi_plus(임박 + 종가 > 단기선) · complete(봉이 끝났는지)
        · close_ema(tail 을 주고 종가 필터가 있는 봉만 — 전체 기록으로 계산한 필터 EMA).
    """
    if short >= long:
        raise ValueError(f"short({short}) < long({long}) 이어야 한다")
    rule = TF_RULES[tf]
    c = close.dropna().astype("float64") if bars else to_bars(close, tf)
    ms = moving_average(c, short, kind)
    ml = moving_average(c, long, kind)
    ce = pine_ema(c, rule.close_filter_ema) if rule.close_filter_ema > 0 else None
    offset = 0
    if tail is not None and len(c) > tail:
        offset = len(c) - tail
        c, ms, ml = c.iloc[offset:], ms.iloc[offset:], ml.iloc[offset:]
        ce = ce.iloc[offset:] if ce is not None else None
    gap = (ms - ml) / ml
    valid = ms.notna() & ml.notna()
    above = (ms > ml) & valid
    prev_valid = valid.shift(1, fill_value=False)
    prev_le = (ms.shift(1) <= ml.shift(1)) & prev_valid
    prev_ge = (ms.shift(1) >= ml.shift(1)) & prev_valid
    gc = valid & (ms > ml) & prev_le
    dc = valid & (ms < ml) & prev_ge

    cross = (gc | dc).to_numpy()
    pos = np.arange(len(c))
    last = np.maximum.accumulate(np.where(cross, pos, -1)) if len(c) else pos
    gi = _imminent(c, ms, ml, gap, valid, cross, short, rule, close_ema=ce)
    as_of = c.index[-1] if as_of is None or c.empty else as_of
    cols: dict[str, object] = {
        "close": c, "ma_s": ms, "ma_l": ml, "gap": gap, "golden": above, "gc": gc, "dc": dc,
        "bars_since": np.where(last >= 0, pos - last, -1),
        "n_bar": pos + offset,  # 첫 봉 = 0 (상장 직후 봉을 빼고 싶을 때 쓴다)
        "gi": gi, "gi_plus": gi & (c > ms),
        "complete": bar_complete(c.index, tf, as_of) if not c.empty else np.zeros(0, dtype=bool),
    }
    if tail is not None and ce is not None:
        cols["close_ema"] = ce
    return pd.DataFrame(cols, index=c.index)


def _imminent(c: pd.Series, ms: pd.Series, ml: pd.Series, gap: pd.Series, valid: pd.Series,
              cross: np.ndarray, short: int, rule: TfRule, close_ema: pd.Series | None = None) -> pd.Series:
    """patterns.detect_golden_cross 의 '골든크로스 임박'을 봉마다 (마지막 봉이 그 봉이라고 보고) 계산.

    close_ema: 종가 필터 EMA 를 미리 (전체 기록으로) 계산해 넘길 때 — 없으면 c 로 계산한다.
    """
    lag = rule.slope_lookback - 1
    prev_gap = gap.shift(lag)
    # 최근 recent_window 봉 안에 교차(골든·데드)가 있으면 앱은 교차 라벨을 낸다
    recent_cross = pd.Series(cross, index=c.index).astype(float).rolling(rule.recent_window, min_periods=1).max() > 0
    # 단기선이 short 봉 동안 계속 장기선 아래 (둘 다 있는 봉만 — 앱은 NaN 봉을 뺀다)
    below = ((ms < ml) | ~valid).astype(float).rolling(short, min_periods=1).min() > 0
    any_valid = valid.astype(float).rolling(short, min_periods=1).max() > 0
    sustained_below = below & any_valid
    narrowing = gap.abs() < prev_gap.abs()
    ok = valid & ~recent_cross & (gap < 0) & (gap.abs() < IMMINENT_GAP) & sustained_below & narrowing
    if rule.short_slope_confirm:
        ok &= ms > ms.shift(lag)
    if rule.close_filter_ema > 0:
        ok &= c > (close_ema if close_ema is not None else pine_ema(c, rule.close_filter_ema))
    return ok.fillna(False).astype(bool)


def latest_state(close: pd.Series, short: int, long: int, kind: MaKind, tf: Timeframe,
                 as_of: pd.Timestamp | None = None) -> dict[str, object] | None:
    """마지막 봉의 상태 한 줄 (표시용). 선이 아직 없으면 None."""
    f = cross_frame(close, short, long, kind, tf, as_of)
    if f.empty or pd.isna(f["ma_l"].iloc[-1]):
        return None
    r = f.iloc[-1]
    return {"date": f.index[-1], "close": float(r["close"]), "ma_s": float(r["ma_s"]), "ma_l": float(r["ma_l"]),
            "gap": float(r["gap"]), "golden": bool(r["golden"]), "bars_since": int(r["bars_since"]),
            "gi": bool(r["gi"]), "gi_plus": bool(r["gi_plus"]), "complete": bool(r["complete"])}
