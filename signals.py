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
    SIGNAL_PRIORITY,
    VOLUME_BREAKOUT_MULTIPLIER,
    MARKET_FILTER_ENABLED,
    STRATEGY_MODE,
)


def _safe_bool(series: pd.Series) -> pd.Series:
    """Return a boolean series with NaN treated as False."""

    return series.fillna(False).astype(bool)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """Gracefully retrieve a column, filling with NaN if absent."""

    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def _compute_buy_signal(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
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

    # Strategy Mode Adjustments
    rsi_threshold = RSI_BUY_MAX
    adx_threshold = ADX_BUY_MIN

    if STRATEGY_MODE == "AGGRESSIVE":
        rsi_threshold = 60  # Relax RSI requirement (allow higher momentum)
        adx_threshold = 15  # Relax Trend Strength requirement

    support_rsi = rsi < rsi_threshold
    support_volume = volume > volume_ma * VOLUME_BREAKOUT_MULTIPLIER
    support_adx = adx > adx_threshold
    support_obv = (obv > obv_ma) | (obv_mom > 0)
    support_atr = atr_pct < atr_buy_max

    support_stack = pd.concat(
        [support_rsi, support_volume, support_adx, support_obv, support_atr], axis=1
    )
    support_count = support_stack.fillna(False).sum(axis=1)

    buy = core & (support_count >= 3)
    return _safe_bool(buy), support_count


def _compute_sell_signal(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
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
    return _safe_bool(sell), support_count


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

    buy_signal, buy_counts = _compute_buy_signal(out)
    sell_signal, sell_counts = _compute_sell_signal(out)

    # --- Market Regime Filter ---
    # If enabled, disable NEW buy signals when SPY is below EMA200 (Bear Market).
    if MARKET_FILTER_ENABLED:
        spy_row = out[out["티커"] == "SPY"]
        if not spy_row.empty:
            spy_close = spy_row["close"].values[0]
            spy_ema200 = spy_row["ema200"].values[0]
            
            # Check if SPY is valid and below EMA200
            if pd.notna(spy_close) and pd.notna(spy_ema200) and spy_close < spy_ema200:
                # Market is Bearish -> Force Buy Signal to False
                buy_signal[:] = False
                # Optional: We could leave "low_prob" (Bottom Fishing) active, 
                # but for safety, we suppress standard momentum buys.
    # ----------------------------

    buy_counts = buy_counts.fillna(0).astype(int)
    sell_counts = sell_counts.fillna(0).astype(int)

    out["buy_signal"] = buy_signal
    out["sell_signal"] = sell_signal
    out["buy_support_count"] = buy_counts
    out["sell_support_count"] = sell_counts
    out["support_count"] = buy_counts

    def _format(flag: bool, count: int) -> str:
        label = "TRUE" if bool(flag) else "FALSE"
        return f"{label}({count})"

    out["buy_signal_text"] = [
        _format(flag, count) for flag, count in zip(buy_signal.tolist(), buy_counts.tolist())
    ]
    out["sell_signal_text"] = [
        _format(flag, count) for flag, count in zip(sell_signal.tolist(), sell_counts.tolist())
    ]

    judgement, recommendation = _evaluate_judgement(
        out["buy_signal"], out["sell_signal"]
    )

    low_prob = _column(out, "저점확률")
    high_prob = _column(out, "고점확률")
    judgement = judgement.copy()
    recommendation = recommendation.copy()

    if not low_prob.isna().all() or not high_prob.isna().all():
        high_flag = high_prob >= 0.65
        low_flag = (low_prob >= 0.65) & ~high_flag

        judgement.loc[high_flag] = "관망 과열"
        recommendation.loc[high_flag] = "차익 실현 고려"

        judgement.loc[low_flag] = "저점 관찰"
        recommendation.loc[low_flag] = "저점 분할 매수"

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
