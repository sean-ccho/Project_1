"""Simplified signal evaluation using a restricted indicator set."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ADX_BUY_MIN,
    ADX_SELL_MAX,
    JUDGEMENT_DISPLAY,
    RECOMMENDATION_DISPLAY,
    RSI_BUY_MAX,
    RSI_SELL_MIN,
    SIGNAL_PRIORITY,
    VOLUME_BREAKOUT_MULTIPLIER,
)


def _safe_bool(series: pd.Series) -> pd.Series:
    """Return a boolean series with NaN treated as False."""

    return series.fillna(False).astype(bool)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """Gracefully retrieve a column, filling with NaN if absent."""

    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def _compute_buy_signal(df: pd.DataFrame) -> pd.Series:
    """EMA/MACD/RSI/Volume/ADX/OBV/ATR filters for entry."""

    ema20 = _column(df, "ema20")
    ema50 = _column(df, "ema50")
    macd_hist = _column(df, "macd_hist")

    core = (ema20 > ema50) & (macd_hist > 0)

    rsi = _column(df, "RSI")
    volume = _column(df, "volume")
    volume_ma = _column(df, "volume_ma20")
    adx = _column(df, "adx")
    obv = _column(df, "obv")
    obv_ma = _column(df, "obv_ma20")
    obv_mom = _column(df, "obv_mom_5")
    atr_pct = _column(df, "atr_pct")
    atr_buy_max = _column(df, "atr_buy_max")

    support_rsi = rsi < RSI_BUY_MAX
    support_volume = volume > volume_ma * VOLUME_BREAKOUT_MULTIPLIER
    support_adx = adx > ADX_BUY_MIN
    support_obv = (obv > obv_ma) | (obv_mom > 0)
    support_atr = atr_pct < atr_buy_max

    support_stack = pd.concat(
        [support_rsi, support_volume, support_adx, support_obv, support_atr], axis=1
    )
    support_count = support_stack.fillna(False).sum(axis=1)

    buy = core & (support_count >= 3)
    return _safe_bool(buy)


def _compute_sell_signal(df: pd.DataFrame) -> pd.Series:
    """Protective exit filters mirroring the simplified indicator set."""

    ema20 = _column(df, "ema20")
    ema50 = _column(df, "ema50")
    macd_hist = _column(df, "macd_hist")
    adx = _column(df, "adx")
    obv = _column(df, "obv")
    obv_ma = _column(df, "obv_ma20")
    obv_mom = _column(df, "obv_mom_5")
    atr_pct = _column(df, "atr_pct")
    atr_buy_max = _column(df, "atr_buy_max")

    core = (ema20 < ema50) | (macd_hist < 0)

    support_adx = adx <= ADX_BUY_MIN
    support_obv = (obv <= obv_ma) & (obv_mom <= 0)
    support_atr = atr_pct >= atr_buy_max

    support_stack = pd.concat(
        [support_adx, support_obv, support_atr], axis=1
    )
    support_count = support_stack.fillna(False).sum(axis=1)

    sell = core & (support_count >= 2)
    return _safe_bool(sell)


def _evaluate_judgement(buy: pd.Series, sell: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Determine judgement/recommendation strings from the signal booleans."""

    buy_only = buy & ~sell
    sell_only = sell & ~buy
    both = buy & sell

    judgement = np.select(
        [buy_only, sell_only, both],
        ["매수 후보", "관망 약세", "관심 관찰"],
        default="관심 관찰",
    )
    recommendation = np.select(
        [buy_only, sell_only],
        ["적극 매수", "관망/보유"],
        default="추가 관찰",
    )

    return (
        pd.Series(judgement, index=buy.index, name="_판단원본"),
        pd.Series(recommendation, index=buy.index, name="_추천원본"),
    )


def attach_signals_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Attach simplified buy/sell flags and return the sorted result set."""

    out = df.copy()

    out["buy_signal"] = _compute_buy_signal(out)
    out["sell_signal"] = _compute_sell_signal(out)

    judgement, recommendation = _evaluate_judgement(
        out["buy_signal"], out["sell_signal"]
    )

    out[judgement.name] = judgement
    out["판단"] = judgement.map(lambda value: JUDGEMENT_DISPLAY.get(value, value))
    out[recommendation.name] = recommendation
    out["추천"] = recommendation.map(
        lambda value: RECOMMENDATION_DISPLAY.get(value, value)
    )

    out["우선순위"] = (
        judgement.map(SIGNAL_PRIORITY).fillna(99).astype(int)
    )

    sorted_out = out.sort_values(
        ["우선순위", "트렌드점수_최종"], ascending=[True, False]
    )
    return sorted_out.drop(columns=[judgement.name, recommendation.name])
