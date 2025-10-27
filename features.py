"""특징(Feature) 계산과 트렌드 점수 산출 로직.

원시 OHLCV 데이터를 받아 유동성, 모멘텀, 변동성 등의 지표를 계산하고 이를 단일 점수로
압축하는 과정을 담당한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from ta.momentum import (
    ROCIndicator,
    StochasticOscillator,
)
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    OnBalanceVolumeIndicator,
)
from numpy.lib.stride_tricks import sliding_window_view

from config import (
    ATR_BUY_THRESHOLD_MULTIPLIER,
    ATR_MEDIAN_LOOKBACK,
    ATR_POSITION_MULTIPLE,
    ATR_SELL_THRESHOLD_MULTIPLIER,
    DEFAULT_EQUITY,
    HAMMER_LOWER_SHADOW_MIN,
    HAMMER_UPPER_SHADOW_MAX,
    EARNINGS_SOON_DAYS,
    EXTREME_HIGH_LOOKBACK,
    EXTREME_LOW_LOOKBACK,
    LONG_TERM_SLOPE_LOOKBACK,
    OBV_MOMENTUM_LOOKBACK,
    OBV_ROLLING_WINDOW,
    REL_STRENGTH_LOOKBACK,
    RISK_PER_TRADE,
    SECTOR_MAP,
    VOLATILITY_PENALTY_END,
    VOLATILITY_PENALTY_START,
    VOLUME_ROLLING_WINDOW,
    WEIGHTS,
)
from fundamentals import fetch_fundamental_snapshots


def _sliding_window(series: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window < 1:
        raise ValueError("window must be positive")
    values = series.to_numpy(dtype=float)
    n = len(values)
    if n < window:
        return values, np.empty((0, window), dtype=float), np.array([], dtype=bool)
    windowed = sliding_window_view(values, window_shape=window)
    invalid = np.isnan(windowed).any(axis=1)
    return values, windowed, invalid


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    values, windowed, invalid = _sliding_window(series, window)
    result = np.full(len(values), np.nan, dtype=float)
    if windowed.size == 0:
        return pd.Series(result, index=series.index, dtype=float)
    means = windowed.mean(axis=1)
    means[invalid] = np.nan
    result[window - 1 :] = means
    return pd.Series(result, index=series.index, dtype=float)


def _rolling_highest(series: pd.Series, window: int) -> pd.Series:
    values, windowed, invalid = _sliding_window(series, window)
    result = np.full(len(values), np.nan, dtype=float)
    if windowed.size == 0:
        return pd.Series(result, index=series.index, dtype=float)
    highest = windowed.max(axis=1)
    highest[invalid] = np.nan
    result[window - 1 :] = highest
    return pd.Series(result, index=series.index, dtype=float)


def _rolling_lowest(series: pd.Series, window: int) -> pd.Series:
    values, windowed, invalid = _sliding_window(series, window)
    result = np.full(len(values), np.nan, dtype=float)
    if windowed.size == 0:
        return pd.Series(result, index=series.index, dtype=float)
    lowest = windowed.min(axis=1)
    lowest[invalid] = np.nan
    result[window - 1 :] = lowest
    return pd.Series(result, index=series.index, dtype=float)


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    values, windowed, invalid = _sliding_window(series, window)
    result = np.full(len(values), np.nan, dtype=float)
    if windowed.size == 0:
        return pd.Series(result, index=series.index, dtype=float)
    means = windowed.mean(axis=1, keepdims=True)
    variance = ((windowed - means) ** 2).mean(axis=1)
    std = np.sqrt(variance)
    std[invalid] = np.nan
    result[window - 1 :] = std
    return pd.Series(result, index=series.index, dtype=float)


def _rolling_linear_regression(series: pd.Series, window: int) -> pd.Series:
    if window < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)

    values, windowed, invalid = _sliding_window(series, window)
    result = np.full(len(values), np.nan, dtype=float)
    if windowed.size == 0:
        return pd.Series(result, index=series.index, dtype=float)

    x = np.arange(window, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denominator = window * sum_x2 - sum_x**2
    if denominator == 0:
        return pd.Series(result, index=series.index, dtype=float)

    sum_y = windowed.sum(axis=1)
    sum_xy = (windowed * x).sum(axis=1)
    slope = (window * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / window
    linreg_values = intercept + slope * (window - 1)
    linreg_values[invalid] = np.nan
    result[window - 1 :] = linreg_values
    return pd.Series(result, index=series.index, dtype=float)


def _rma(values: pd.Series, length: int) -> pd.Series:
    """TradingView rma와 일치하는 Wilder 이동평균을 계산한다."""

    if length <= 0:
        raise ValueError("length must be positive")
    if values.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)

    result = np.full(len(values), np.nan, dtype=float)
    data = values.to_numpy(dtype=float)
    alpha = 1.0 / float(length)
    sum_init = 0.0
    count = 0
    prev = np.nan
    for idx, v in enumerate(data):
        if np.isnan(v):
            result[idx] = np.nan
            continue
        if count < length:
            sum_init += v
            count += 1
            if count == length:
                prev = sum_init / length
                result[idx] = prev
            else:
                result[idx] = np.nan
            continue
        if np.isnan(prev):
            prev = v
        else:
            prev = (1.0 - alpha) * prev + alpha * v
        result[idx] = prev
    return pd.Series(result, index=values.index, dtype=float)


def _wilders_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """LazyBear / TradingView CM Ultimate RSI와 동일한 Wilder 기반 RSI."""

    if length < 1 or series.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    zero_gain_mask = (avg_loss != 0.0) & (avg_gain == 0.0)
    rsi = rsi.where(~zero_gain_mask, 0.0)
    return rsi


def _compute_squeeze_fields(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    bb_length: int = 20,
    bb_multiplier: float = 2.0,
    kc_length: int = 20,
    kc_multiplier: float = 1.5,
    use_true_range: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """LazyBear Squeeze Momentum의 상태·모멘텀 및 스퀴즈 해제 신호를 계산한다."""

    if close.empty or high.empty or low.empty:
        index = close.index if not close.empty else high.index if not high.empty else low.index
        empty_bool = pd.Series(False, index=index, dtype=bool)
        empty_float = pd.Series(np.nan, index=index, dtype=float)
        return empty_bool, empty_bool, empty_float, empty_bool

    effective_length = min(len(close.dropna()), len(high.dropna()), len(low.dropna()))
    min_required = max(bb_length, kc_length)
    if effective_length < min_required:
        empty_bool = pd.Series(False, index=close.index, dtype=bool)
        empty_float = pd.Series(np.nan, index=close.index, dtype=float)
        return empty_bool, empty_bool, empty_float, empty_bool

    bb_basis = _rolling_mean(close, bb_length)
    bb_std = _rolling_std(close, bb_length)
    upper_bb = bb_basis + bb_multiplier * bb_std
    lower_bb = bb_basis - bb_multiplier * bb_std

    kc_ma = _rolling_mean(close, kc_length)
    if use_true_range:
        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
    else:
        true_range = (high - low).abs()

    range_ma = _rolling_mean(true_range, kc_length)
    upper_kc = kc_ma + kc_multiplier * range_ma
    lower_kc = kc_ma - kc_multiplier * range_ma

    squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    squeeze_outside = (lower_bb < lower_kc) & (upper_bb > upper_kc)

    squeeze_on = squeeze_on.fillna(False)
    squeeze_outside = squeeze_outside.fillna(False)

    highest_high = _rolling_highest(high, kc_length)
    lowest_low = _rolling_lowest(low, kc_length)
    avg_range = (highest_high + lowest_low) / 2.0
    sma_close = _rolling_mean(close, kc_length)
    composite_mid = (avg_range + sma_close) / 2.0
    diff_series = close - composite_mid
    squeeze_momentum = _rolling_linear_regression(diff_series, kc_length)

    positive_momentum = (squeeze_momentum >= 0).fillna(False)
    prev_positive = positive_momentum.shift(1, fill_value=False)
    prev_squeeze_on = squeeze_on.shift(1, fill_value=False)
    prev_momentum = squeeze_momentum.shift(1)
    crossed_from_negative = prev_momentum < 0

    release_signal = (
        (~squeeze_on)
        & positive_momentum
        & (crossed_from_negative.fillna(False) | prev_squeeze_on)
    )

    return squeeze_on, release_signal, squeeze_momentum, squeeze_outside


def _aggregate_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """일봉 데이터를 지정 주기(rule)로 리샘플해 OHLCV를 생성한다."""

    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    agg_map: dict[str, str] = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    if "Dividends" in df.columns:
        agg_map["Dividends"] = "sum"
    if "Adj Close" in df.columns:
        agg_map["Adj Close"] = "last"

    trimmed_rule = "ME" if rule == "M" else rule
    try:
        resampled = df.resample(trimmed_rule, label="right", closed="right").agg(agg_map)
    except ValueError:
        return pd.DataFrame()

    resampled = resampled.dropna(how="all")
    if resampled.empty:
        return pd.DataFrame()

    return resampled


def _latest_timeframe_metrics(df: pd.DataFrame, rule: str) -> tuple[bool, bool, float, float]:
    """주어진 일봉 데이터를 기반으로 특정 주기 스퀴즈/RSI 최신값을 반환한다."""

    aggregated = _aggregate_ohlcv(df, rule)
    if aggregated.empty:
        return False, False, float("nan"), float("nan")

    squeeze_on, squeeze_release, squeeze_momentum, squeeze_outside = _compute_squeeze_fields(
        aggregated["Close"],
        aggregated["High"],
        aggregated["Low"],
    )
    aggregated = aggregated.copy()
    aggregated["squeeze_on"] = squeeze_on
    aggregated["squeeze_off"] = squeeze_release
    aggregated["squeeze_outside"] = squeeze_outside
    aggregated["squeeze_momentum"] = squeeze_momentum

    if aggregated["Close"].dropna().size < 14:
        aggregated["rsi"] = np.nan
    else:
        aggregated["rsi"] = _wilders_rsi(aggregated["Close"], length=14)

    aggregated = aggregated.dropna(how="all")
    if aggregated.empty:
        return False, False, float("nan"), float("nan")

    latest = aggregated.iloc[-1]
    squeeze_on_value = bool(latest["squeeze_on"]) if not pd.isna(latest["squeeze_on"]) else False
    squeeze_off_value = bool(latest["squeeze_off"]) if not pd.isna(latest["squeeze_off"]) else False
    squeeze_momentum_value = (
        float(latest["squeeze_momentum"]) if not np.isnan(latest["squeeze_momentum"]) else float("nan")
    )
    rsi_value = float(latest["rsi"]) if not np.isnan(latest["rsi"]) else float("nan")

    return squeeze_on_value, squeeze_off_value, squeeze_momentum_value, rsi_value


def to_market(ticker: str) -> str:
    """티커 접미사를 이용해 미국/캐나다 시장을 구분한다."""

    return "CA" if ticker.endswith(".TO") else "US"


def smooth_tanh(x: float, scale: float = 1.0) -> float:
    """극단값을 완화하기 위해 하이퍼볼릭 탄젠트를 사용한 스케일링."""

    return float(np.tanh(x / scale))


def rsi_smooth_score(rsi: float) -> float:
    """RSI(상대강도지수)를 -1~1 범위로 부드럽게 변환한다."""

    if np.isnan(rsi):
        return 0.0
    return float(np.tanh((rsi - 55.0) / 10.0))


@dataclass
class FeatureSet:
    """단일 종목의 핵심 지표를 담는 컨테이너."""

    trend_score: float
    ret_5d: float
    ret_20d: float
    vol_z20: float
    pos_52w: float
    atr_pct: float
    rsi: float
    macd_hist: float
    stoch_k: float
    roc_10: float
    adx: float
    ema_gap_20_50: float
    ema_gap_50_200: float
    ema200_slope_20: float
    close_to_ema200_pct: float
    bollinger_pband: float
    obv_z20: float
    cmf_20: float
    accdist_slope_5: float
    gap_down_pct: float
    avg_dollar_vol_20d: float
    volume_stability_ratio: float
    hammer_candle: bool
    intraday_recovery: float
    distance_from_10d_low: float
    distance_from_10d_high: float
    volume_breakout_ratio: float
    volatility_contraction: float
    squeeze_on: bool
    squeeze_off: bool
    squeeze_momentum: float
    weekly_squeeze_on: bool
    weekly_squeeze_off: bool
    weekly_squeeze_momentum: float
    weekly_rsi: float
    monthly_squeeze_on: bool
    monthly_squeeze_off: bool
    monthly_squeeze_momentum: float
    monthly_rsi: float
    dividend_yield: float
    annual_dividend: float
    ema20: float
    ema50: float
    volume: float
    volume_ma20: float
    obv: float
    obv_ma20: float
    obv_mom_5: float
    obv_mom_ratio: float
    atr_med_252: float
    atr_buy_max: float
    atr_sell_max: float
    close: float
    atr_value: float
    stop_dist: float
    position_size: float


def compute_features_for_ticker(p: pd.DataFrame) -> Optional[FeatureSet]:
    """종목별 히스토리 데이터에서 계산 가능한 모든 특징을 생성한다."""

    if len(p.dropna(how="all")) < 120:  # 데이터가 너무 짧으면 신뢰도가 떨어지므로 계산을 생략한다.
        return None

    p = p.copy()
    p = p.dropna(how="all")
    if "Dividends" in p.columns:
        p["Dividends"] = p["Dividends"].fillna(0.0)
    else:
        p["Dividends"] = 0.0

    p["ret_5d"] = p["Close"].pct_change(5)
    p["ret_20d"] = p["Close"].pct_change(REL_STRENGTH_LOOKBACK)

    # 거래량 기반 Z-score: 최근 거래량이 얼마나 평소와 다른지 확인한다.
    p["vol_ma20"] = p["Volume"].rolling(VOLUME_ROLLING_WINDOW).mean()
    p["vol_std20"] = p["Volume"].rolling(VOLUME_ROLLING_WINDOW).std(ddof=0)
    p["vol_z20"] = (p["Volume"] - p["vol_ma20"]) / (p["vol_std20"] + 1e-9)

    # ATR%: 절대적 가격 수준과 무관한 변동성 지표로 사용.
    atr = AverageTrueRange(p["High"], p["Low"], p["Close"], window=14).average_true_range()
    p["atr_pct"] = atr / p["Close"]

    # RSI: CM Ultimate RSI와 동일한 Wilder 방식을 적용.
    p["rsi"] = _wilders_rsi(p["Close"], length=14)

    # MACD 지표(추세 방향 및 모멘텀).
    macd_indicator = MACD(p["Close"], window_slow=26, window_fast=12, window_sign=9)
    p["macd_hist"] = macd_indicator.macd_diff()

    # Stochastic Oscillator: 과매수/과매도 빠르게 포착.
    stoch = StochasticOscillator(p["High"], p["Low"], p["Close"], window=14, smooth_window=3)
    p["stoch_k"] = stoch.stoch()

    # 10일 ROC(가격 변화율).
    p["roc_10"] = ROCIndicator(p["Close"], window=10).roc()

    # ADX(추세 강도) 및 +DI/-DI.
    adx_indicator = ADXIndicator(p["High"], p["Low"], p["Close"], window=14)
    p["adx"] = adx_indicator.adx()

    # EMA 간격(추세 정배열/역배열 확인).
    ema20 = EMAIndicator(p["Close"], window=20).ema_indicator()
    ema50 = EMAIndicator(p["Close"], window=50).ema_indicator()
    ema200 = EMAIndicator(p["Close"], window=200).ema_indicator()
    p["ema_gap_20_50"] = (ema20 - ema50) / (ema50 + 1e-9)
    p["ema_gap_50_200"] = (ema50 - ema200) / (ema200 + 1e-9)

    ema200_latest = ema200.iloc[-1]
    ema200_reference = np.nan
    if len(ema200.dropna()) > LONG_TERM_SLOPE_LOOKBACK:
        ema200_reference = ema200.iloc[-(LONG_TERM_SLOPE_LOOKBACK + 1)]
    ema200_slope = np.nan
    if not np.isnan(ema200_latest) and not np.isnan(ema200_reference) and ema200_reference != 0:
        ema200_slope = (ema200_latest / ema200_reference) - 1.0
    close_to_ema200 = np.nan
    if not np.isnan(ema200_latest) and ema200_latest != 0:
        close_to_ema200 = (p["Close"].iloc[-1] - ema200_latest) / ema200_latest

    # Bollinger Bands: 밴드 위치와 폭.
    bb = BollingerBands(p["Close"], window=20, window_dev=2)
    p["bollinger_pband"] = bb.bollinger_pband()

    # OBV 기반 수급 흐름: 20일 Z-score로 정규화.
    obv = OnBalanceVolumeIndicator(p["Close"], p["Volume"]).on_balance_volume()
    obv_ma = obv.rolling(OBV_ROLLING_WINDOW).mean()
    obv_std = obv.rolling(OBV_ROLLING_WINDOW).std(ddof=0)
    p["obv_z20"] = (obv - obv_ma) / (obv_std + 1e-9)

    # Chaikin Money Flow: 20일 자금 흐름지수.
    p["cmf_20"] = ChaikinMoneyFlowIndicator(
        p["High"], p["Low"], p["Close"], p["Volume"], window=20
    ).chaikin_money_flow()

    # Acc/Dist Index의 5일 기울기(수급 변화 추세).
    acc = AccDistIndexIndicator(p["High"], p["Low"], p["Close"], p["Volume"]).acc_dist_index()
    p["accdist_slope_5"] = acc.diff(5) / (abs(acc.shift(5)) + 1e-9)

    # 52주 범위 대비 위치(0~1) 계산.
    p["roll_max_252"] = p["Close"].rolling(252, min_periods=63).max()
    p["roll_min_252"] = p["Close"].rolling(252, min_periods=63).min()
    p["range_52w"] = (p["roll_max_252"] - p["roll_min_252"]).replace(0, np.nan)
    p["pos_52w"] = (p["Close"] - p["roll_min_252"]) / p["range_52w"]

    (
        squeeze_on_series,
        squeeze_release_series,
        squeeze_momentum_series,
        squeeze_outside_series,
    ) = _compute_squeeze_fields(
        p["Close"],
        p["High"],
        p["Low"],
    )
    p["squeeze_on"] = squeeze_on_series.astype(bool)
    p["squeeze_off"] = squeeze_release_series.astype(bool)
    p["squeeze_outside"] = squeeze_outside_series.astype(bool)
    p["squeeze_momentum"] = squeeze_momentum_series

    weekly_on, weekly_off, weekly_momentum, weekly_rsi = _latest_timeframe_metrics(p, "W-FRI")
    monthly_on, monthly_off, monthly_momentum, monthly_rsi = _latest_timeframe_metrics(p, "M")

    latest = p.iloc[-1]  # 최신 시점의 지표만 활용해 현재 상태를 평가한다.

    # 가중 점수 계산을 위한 전처리.
    s_ret5 = latest["ret_5d"]
    s_vol = smooth_tanh(latest["vol_z20"], scale=3.0)
    s_break = float(np.clip(latest["pos_52w"], 0, 1))
    s_vola = float(np.clip(latest["atr_pct"], 0, 0.1) / 0.1)
    s_rsi = rsi_smooth_score(latest["rsi"])
    accel_raw = latest["ret_5d"] - latest["ret_20d"]
    s_accel = smooth_tanh(accel_raw, scale=0.05)

    penalty_span = max(VOLATILITY_PENALTY_END - VOLATILITY_PENALTY_START, 1e-6)
    vola_penalty = np.clip(
        (latest["atr_pct"] - VOLATILITY_PENALTY_START) / penalty_span,
        0.0,
        1.0,
    )

    trend_score = (
        WEIGHTS.get("ret5", 0.0) * s_ret5
        + WEIGHTS.get("vol", 0.0) * s_vol
        + WEIGHTS.get("break", 0.0) * s_break
        + WEIGHTS.get("vola", 0.0) * s_vola
        + WEIGHTS.get("rsi", 0.0) * s_rsi
        + WEIGHTS.get("accel", 0.0) * s_accel
        + WEIGHTS.get("vola_penalty", 0.0) * vola_penalty
    )

    avg_dollar_vol = (p["Close"].iloc[-20:] * p["Volume"].iloc[-20:]).mean()
    avg_dollar_vol_60d = np.nan
    if len(p) >= 60:
        avg_dollar_vol_60d = (p["Close"].iloc[-60:] * p["Volume"].iloc[-60:]).mean()
    volume_stability_ratio = np.nan
    if not np.isnan(avg_dollar_vol_60d) and avg_dollar_vol_60d != 0:
        volume_stability_ratio = avg_dollar_vol / avg_dollar_vol_60d

    gap_down_pct = np.nan
    if len(p) >= 2:
        prev_close = p["Close"].iloc[-2]
        latest_open = p["Open"].iloc[-1]
        if prev_close and prev_close != 0:
            gap_down_pct = (latest_open - prev_close) / prev_close

    range_total = latest["High"] - latest["Low"]
    hammer_candle = False
    if range_total and range_total > 0:
        body = abs(latest["Close"] - latest["Open"])
        lower_shadow = min(latest["Open"], latest["Close"]) - latest["Low"]
        upper_shadow = latest["High"] - max(latest["Open"], latest["Close"])
        if lower_shadow < 0:
            lower_shadow = 0.0
        if upper_shadow < 0:
            upper_shadow = 0.0
        lower_ratio = lower_shadow / range_total
        upper_ratio = upper_shadow / range_total
        body_ratio = body / range_total
        hammer_candle = (
            latest["Close"] > latest["Open"]
            and lower_ratio >= HAMMER_LOWER_SHADOW_MIN
            and upper_ratio <= HAMMER_UPPER_SHADOW_MAX
            and body_ratio <= 0.4
        )

    intraday_recovery = np.nan
    if latest["Low"] and latest["Low"] > 0:
        intraday_recovery = (latest["Close"] / latest["Low"]) - 1.0

    distance_from_10d_low = np.nan
    if len(p) >= EXTREME_LOW_LOOKBACK:
        recent_low = p["Close"].iloc[-EXTREME_LOW_LOOKBACK:].min()
        if recent_low and recent_low > 0:
            distance_from_10d_low = (latest["Close"] / recent_low) - 1.0

    distance_from_10d_high = np.nan
    if len(p) >= EXTREME_HIGH_LOOKBACK:
        recent_high = p["Close"].iloc[-EXTREME_HIGH_LOOKBACK:].max()
        if recent_high and recent_high > 0:
            distance_from_10d_high = (latest["Close"] / recent_high) - 1.0

    ema20_latest = float(ema20.iloc[-1]) if not np.isnan(ema20.iloc[-1]) else float("nan")
    ema50_latest = float(ema50.iloc[-1]) if not np.isnan(ema50.iloc[-1]) else float("nan")

    volume_latest = float(p["Volume"].iloc[-1]) if not np.isnan(p["Volume"].iloc[-1]) else float("nan")
    volume_ma_latest = (
        float(p["vol_ma20"].iloc[-1]) if not np.isnan(p["vol_ma20"].iloc[-1]) else float("nan")
    )
    volume_breakout_ratio = np.nan
    if not np.isnan(volume_latest) and not np.isnan(volume_ma_latest) and volume_ma_latest != 0:
        volume_breakout_ratio = volume_latest / volume_ma_latest

    obv_latest = float(obv.iloc[-1]) if not np.isnan(obv.iloc[-1]) else float("nan")
    obv_ma_latest = float(obv_ma.iloc[-1]) if not np.isnan(obv_ma.iloc[-1]) else float("nan")
    obv_mom_series = obv.diff(OBV_MOMENTUM_LOOKBACK)
    obv_mom_latest = (
        float(obv_mom_series.iloc[-1]) if not np.isnan(obv_mom_series.iloc[-1]) else float("nan")
    )

    atr_median_series = p["atr_pct"].rolling(ATR_MEDIAN_LOOKBACK).median()
    atr_median_latest = (
        float(atr_median_series.iloc[-1])
        if not np.isnan(atr_median_series.iloc[-1])
        else float("nan")
    )

    volatility_contraction = np.nan
    if (
        not np.isnan(latest["atr_pct"])
        and not np.isnan(atr_median_latest)
        and atr_median_latest > 0
        and latest["atr_pct"] > 0
    ):
        volatility_contraction = latest["atr_pct"] / atr_median_latest

    atr_buy_max = (
        atr_median_latest * ATR_BUY_THRESHOLD_MULTIPLIER
        if not np.isnan(atr_median_latest)
        else float("nan")
    )
    atr_sell_max = (
        atr_median_latest * ATR_SELL_THRESHOLD_MULTIPLIER
        if not np.isnan(atr_median_latest)
        else float("nan")
    )

    total_div_1y = p["Dividends"].iloc[-252:].sum()
    latest_close = p["Close"].iloc[-1]
    dividend_yield = np.nan
    if not np.isnan(latest_close) and latest_close > 0:
        dividend_yield = total_div_1y / latest_close

    atr_value = np.nan
    if not np.isnan(latest_close) and not np.isnan(latest["atr_pct"]):
        atr_value = latest["atr_pct"] * latest_close

    stop_dist = np.nan
    if not np.isnan(atr_value):
        stop_dist = atr_value * ATR_POSITION_MULTIPLE

    base_risk = DEFAULT_EQUITY * RISK_PER_TRADE

    trend_weight = 0.0
    if not np.isnan(latest["adx"]):
        trend_weight = float(np.clip((latest["adx"] - 20.0) / 20.0, 0.0, 1.0))

    obv_momentum = np.nan
    if not np.isnan(obv_latest) and not np.isnan(obv_ma_latest) and obv_ma_latest != 0:
        obv_momentum = (obv_latest - obv_ma_latest) / obv_ma_latest

    vol_adj_size = 1.0
    if (
        not np.isnan(latest["atr_pct"])
        and not np.isnan(atr_median_latest)
        and atr_median_latest > 0
        and latest["atr_pct"] > 0
    ):
        vol_adj_size = float(atr_median_latest / latest["atr_pct"])

    position_size = base_risk * trend_weight * float(np.clip(vol_adj_size, 0.5, 2.0))

    return FeatureSet(
        trend_score=float(trend_score),
        ret_5d=float(latest["ret_5d"]),
        ret_20d=float(latest["ret_20d"]),
        vol_z20=float(latest["vol_z20"]),
        pos_52w=float(latest["pos_52w"]),
        atr_pct=float(latest["atr_pct"]),
        rsi=float(latest["rsi"]),
        macd_hist=float(latest["macd_hist"]),
        stoch_k=float(latest["stoch_k"]),
        roc_10=float(latest["roc_10"]),
        adx=float(latest["adx"]),
        ema_gap_20_50=float(latest["ema_gap_20_50"]),
        ema_gap_50_200=float(latest["ema_gap_50_200"]),
        ema200_slope_20=float(ema200_slope) if not np.isnan(ema200_slope) else float("nan"),
        close_to_ema200_pct=float(close_to_ema200)
        if not np.isnan(close_to_ema200)
        else float("nan"),
        bollinger_pband=float(latest["bollinger_pband"]),
        obv_z20=float(latest["obv_z20"]),
        cmf_20=float(latest["cmf_20"]),
        accdist_slope_5=float(latest["accdist_slope_5"]),
        gap_down_pct=float(gap_down_pct) if not np.isnan(gap_down_pct) else float("nan"),
        avg_dollar_vol_20d=float(avg_dollar_vol),
        volume_stability_ratio=float(volume_stability_ratio)
        if not np.isnan(volume_stability_ratio)
        else float("nan"),
        hammer_candle=bool(hammer_candle),
        intraday_recovery=float(intraday_recovery)
        if not np.isnan(intraday_recovery)
        else float("nan"),
        distance_from_10d_low=float(distance_from_10d_low)
        if not np.isnan(distance_from_10d_low)
        else float("nan"),
        distance_from_10d_high=float(distance_from_10d_high)
        if not np.isnan(distance_from_10d_high)
        else float("nan"),
        volume_breakout_ratio=float(volume_breakout_ratio)
        if not np.isnan(volume_breakout_ratio)
        else float("nan"),
        volatility_contraction=float(volatility_contraction)
        if not np.isnan(volatility_contraction)
        else float("nan"),
        squeeze_on=bool(latest["squeeze_on"]) if not pd.isna(latest["squeeze_on"]) else False,
        squeeze_off=bool(latest["squeeze_off"]) if not pd.isna(latest["squeeze_off"]) else False,
        squeeze_momentum=float(latest["squeeze_momentum"])
        if not np.isnan(latest["squeeze_momentum"])
        else float("nan"),
        weekly_squeeze_on=weekly_on,
        weekly_squeeze_off=weekly_off,
        weekly_squeeze_momentum=weekly_momentum,
        weekly_rsi=float(weekly_rsi) if not np.isnan(weekly_rsi) else float("nan"),
        monthly_squeeze_on=monthly_on,
        monthly_squeeze_off=monthly_off,
        monthly_squeeze_momentum=monthly_momentum,
        monthly_rsi=float(monthly_rsi) if not np.isnan(monthly_rsi) else float("nan"),
        dividend_yield=float(dividend_yield) if not np.isnan(dividend_yield) else np.nan,
        annual_dividend=float(total_div_1y),
        ema20=ema20_latest,
        ema50=ema50_latest,
        volume=volume_latest,
        volume_ma20=volume_ma_latest,
        obv=obv_latest,
        obv_ma20=obv_ma_latest,
        obv_mom_5=obv_mom_latest,
        obv_mom_ratio=float(obv_momentum) if not np.isnan(obv_momentum) else float("nan"),
        atr_med_252=atr_median_latest,
        atr_buy_max=atr_buy_max,
        atr_sell_max=atr_sell_max,
        close=float(latest_close) if not np.isnan(latest_close) else float("nan"),
        atr_value=float(atr_value) if not np.isnan(atr_value) else float("nan"),
        stop_dist=float(stop_dist) if not np.isnan(stop_dist) else float("nan"),
        position_size=float(position_size),
    )


def feature_row_from_set(ticker: str, feature_set: FeatureSet) -> dict:
    return {
        "티커": ticker,
        "트렌드점수": feature_set.trend_score,
        "5일수익률": feature_set.ret_5d,
        "20일수익률": feature_set.ret_20d,
        "거래량Z(20)": feature_set.vol_z20,
        "52주포지션": feature_set.pos_52w,
        "ATR%": feature_set.atr_pct,
        "RSI": feature_set.rsi,
        "macd_hist": feature_set.macd_hist,
        "stoch_k": feature_set.stoch_k,
        "roc_10": feature_set.roc_10,
        "adx": feature_set.adx,
        "ema_gap_20_50": feature_set.ema_gap_20_50,
        "ema_gap_50_200": feature_set.ema_gap_50_200,
        "ema200_slope_20": feature_set.ema200_slope_20,
        "close_to_ema200_pct": feature_set.close_to_ema200_pct,
        "bollinger_pband": feature_set.bollinger_pband,
        "obv_z20": feature_set.obv_z20,
        "cmf_20": feature_set.cmf_20,
        "accdist_slope_5": feature_set.accdist_slope_5,
        "갭하락률": feature_set.gap_down_pct,
        "최근20일평균거래대금": feature_set.avg_dollar_vol_20d,
        "거래대금안정비": feature_set.volume_stability_ratio,
        "저점반전캔들": int(feature_set.hammer_candle),
        "장중반등률": feature_set.intraday_recovery,
        "10일저점괴리": feature_set.distance_from_10d_low,
        "10일고점괴리": feature_set.distance_from_10d_high,
        "거래량돌파배수": feature_set.volume_breakout_ratio,
        "변동성압축": feature_set.volatility_contraction,
        "squeeze_on": feature_set.squeeze_on,
        "squeeze_off": feature_set.squeeze_off,
        "squeeze_momentum": feature_set.squeeze_momentum,
        "squeeze_on_weekly": feature_set.weekly_squeeze_on,
        "squeeze_off_weekly": feature_set.weekly_squeeze_off,
        "squeeze_momentum_weekly": feature_set.weekly_squeeze_momentum,
        "RSI_weekly": feature_set.weekly_rsi,
        "squeeze_on_monthly": feature_set.monthly_squeeze_on,
        "squeeze_off_monthly": feature_set.monthly_squeeze_off,
        "squeeze_momentum_monthly": feature_set.monthly_squeeze_momentum,
        "RSI_monthly": feature_set.monthly_rsi,
        "dividend_yield": feature_set.dividend_yield,
        "annual_dividend": feature_set.annual_dividend,
        "시장": to_market(ticker),
        "섹터": SECTOR_MAP.get(ticker, "Unknown"),
        "ema20": feature_set.ema20,
        "ema50": feature_set.ema50,
        "volume": feature_set.volume,
        "volume_ma20": feature_set.volume_ma20,
        "obv": feature_set.obv,
        "obv_ma20": feature_set.obv_ma20,
        "obv_mom_5": feature_set.obv_mom_5,
        "obv_mom_ratio": feature_set.obv_mom_ratio,
        "atr_med_252": feature_set.atr_med_252,
        "atr_buy_max": feature_set.atr_buy_max,
        "atr_sell_max": feature_set.atr_sell_max,
        "close": feature_set.close,
        "atr_value": feature_set.atr_value,
        "stop_dist": feature_set.stop_dist,
        "position_size": feature_set.position_size,
    }


def compute_features_snapshot(
    price_map: Mapping[str, pd.DataFrame],
    *,
    include_fundamentals: bool = True,
) -> pd.DataFrame:
    """주어진 종목별 시계열 스냅샷에서 특징 테이블을 생성한다."""

    rows = []
    for ticker, frame in price_map.items():
        features = compute_features_for_ticker(frame.dropna(how="all"))
        if features is None:
            continue
        rows.append(feature_row_from_set(ticker, features))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if "20일수익률" in out.columns:
        out["시장상대강도"] = out["20일수익률"] - out.groupby("시장")["20일수익률"].transform(
            "mean"
        )
        if (out["섹터"] != "Unknown").any():
            out["섹터상대강도"] = out["20일수익률"] - out.groupby("섹터")[
                "20일수익률"
            ].transform("mean")
        else:
            out["섹터상대강도"] = np.nan
    else:
        out["시장상대강도"] = np.nan
        out["섹터상대강도"] = np.nan

    if include_fundamentals:
        fundamentals = fetch_fundamental_snapshots(out["티커"].tolist())
        if not fundamentals.empty:
            out = out.merge(fundamentals, on="티커", how="left")

    if "days_to_next_earnings" in out.columns:
        earnings_window = out["days_to_next_earnings"].between(
            0, EARNINGS_SOON_DAYS, inclusive="both"
        )
        out["event_earnings_within_window"] = earnings_window.fillna(False)
        out["event_earnings_within_window"] = out[
            "event_earnings_within_window"
        ].astype(bool)

    return out


def compute_all_features(
    df: pd.DataFrame, *, include_fundamentals: bool = True
) -> pd.DataFrame:
    """모든 종목에 대해 특징을 계산하고 테이블 형태로 반환한다."""

    price_map: Dict[str, pd.DataFrame] = {
        ticker: df[ticker].dropna(how="all") for ticker in df.columns.levels[0]
    }
    return compute_features_snapshot(
        price_map, include_fundamentals=include_fundamentals
    )
