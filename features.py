"""특징(Feature) 계산과 트렌드 점수 산출 로직.

원시 OHLCV 데이터를 받아 유동성, 모멘텀, 변동성 등의 지표를 계산하고 이를 단일 점수로
압축하는 과정을 담당한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from ta.momentum import (
    RSIIndicator,
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

from config import (
    ALPHA_DEFAULT_WEIGHTS,
    ALPHA_FORWARD_RETURN_DAYS,
    ALPHA_IC_LOOKBACK,
    ALPHA_IC_MIN_SAMPLES,
    ALPHA_MODEL_ENABLED,
    ATR_BUY_THRESHOLD_MULTIPLIER,
    ATR_MEDIAN_LOOKBACK,
    ATR_POSITION_MULTIPLE,
    ATR_SELL_THRESHOLD_MULTIPLIER,
    BOTTOM_MACD_DIV_LOOKBACK,
    BOTTOM_RSI_LOOKBACK,
    BOTTOM_RSI_OVERSOLD,
    BOTTOM_RSI_RECOVERY,
    BOTTOM_SUPPORT_TOLERANCE,
    BOTTOM_VOLUME_SURGE_MULT,
    DEFAULT_EQUITY,
    HAMMER_LOWER_SHADOW_MIN,
    HAMMER_UPPER_SHADOW_MAX,
    EARNINGS_SOON_DAYS,
    EXTREME_HIGH_LOOKBACK,
    EXTREME_LOW_LOOKBACK,
    LONG_TERM_SLOPE_LOOKBACK,
    OBV_MOMENTUM_LOOKBACK,
    OBV_MOMENTUM_LOOKBACK,
    OBV_ROLLING_WINDOW,
    PATTERN_LOOKBACK_DAYS,
    REL_STRENGTH_LOOKBACK,
    RISK_PER_TRADE,
    SECTOR_MAP,
    VOLATILITY_PENALTY_END,
    VOLATILITY_PENALTY_START,
    VOLUME_ROLLING_WINDOW,
    WEIGHTS,
)
from alpha_model import compute_factor_scores, compute_factor_scores_from_indicators, compute_alpha_score, compute_ic_weights
from fundamentals import fetch_fundamental_snapshots
from patterns import detect_all_patterns, detect_weekly_patterns, AllPatterns


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
    alpha_score: float
    factor_momentum: float
    factor_trend: float
    factor_volume: float
    factor_volatility: float
    factor_mean_reversion: float
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
    dividend_yield: float
    annual_dividend: float
    ema20: float
    ema50: float
    ema200: float
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
    # 차트 패턴
    pattern_triangle: str
    pattern_wedge: str
    pattern_double: str
    pattern_head_shoulders: str
    pattern_cup_handle: bool
    pattern_candlestick: str
    pattern_golden_cross: str
    weekly_patterns: str  # 주봉 패턴 (콤마로 구분된 문자열)
    # 저점 반등 지표
    rsi_reversal: bool           # RSI 과매도 후 반등
    bollinger_bounce: bool       # 볼린저 하단 터치 후 복귀
    volume_surge_bottom: bool    # 저점 부근 거래량 급증
    support_test: bool           # 지지선 2회 테스트
    macd_divergence: bool        # MACD 다이버전스
    bottom_reversal_score: float # 반등 점수 (최대 10점)
    weekly_patterns_list: List = field(default_factory=list)


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

    p["ret_5d"] = p["Close"].pct_change(5, fill_method=None)
    p["ret_20d"] = p["Close"].pct_change(REL_STRENGTH_LOOKBACK, fill_method=None)
    p["ret_63d"] = p["Close"].pct_change(63, fill_method=None)
    p["ret_126d"] = p["Close"].pct_change(126, fill_method=None)

    # 거래량 기반 Z-score: 최근 거래량이 얼마나 평소와 다른지 확인한다.
    p["vol_ma20"] = p["Volume"].rolling(VOLUME_ROLLING_WINDOW).mean()
    p["vol_std20"] = p["Volume"].rolling(VOLUME_ROLLING_WINDOW).std(ddof=0)
    p["vol_z20"] = (p["Volume"] - p["vol_ma20"]) / (p["vol_std20"] + 1e-9)

    # ATR%: 절대적 가격 수준과 무관한 변동성 지표로 사용.
    atr = AverageTrueRange(p["High"], p["Low"], p["Close"], window=14).average_true_range()
    p["atr_pct"] = atr / p["Close"]

    # RSI: 과매수/과매도 판단용.
    p["rsi"] = RSIIndicator(p["Close"], window=14).rsi()

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

    legacy_trend_score = (
        WEIGHTS.get("ret5", 0.0) * s_ret5
        + WEIGHTS.get("vol", 0.0) * s_vol
        + WEIGHTS.get("break", 0.0) * s_break
        + WEIGHTS.get("vola", 0.0) * s_vola
        + WEIGHTS.get("rsi", 0.0) * s_rsi
        + WEIGHTS.get("accel", 0.0) * s_accel
        + WEIGHTS.get("vola_penalty", 0.0) * vola_penalty
    )

    # 알파 모델: 다중 팩터 점수 계산 (이미 계산된 지표 재활용)
    _factor_scores = None
    if ALPHA_MODEL_ENABLED and len(p) >= 130:
        _atr_med = p["atr_pct"].rolling(ATR_MEDIAN_LOOKBACK, min_periods=60).median().iloc[-1]
        _factor_scores = compute_factor_scores_from_indicators(
            ret_20d=float(latest["ret_20d"]) if not np.isnan(latest["ret_20d"]) else 0.0,
            ret_63d=float(latest["ret_63d"]) if not np.isnan(latest.get("ret_63d", np.nan)) else 0.0,
            ret_126d=float(latest["ret_126d"]) if not np.isnan(latest.get("ret_126d", np.nan)) else 0.0,
            ema_gap_20_50=float(latest["ema_gap_20_50"]) if not np.isnan(latest["ema_gap_20_50"]) else 0.0,
            ema_gap_50_200=float(latest["ema_gap_50_200"]) if not np.isnan(latest["ema_gap_50_200"]) else 0.0,
            adx=float(latest["adx"]) if not np.isnan(latest["adx"]) else np.nan,
            macd_hist=float(latest["macd_hist"]) if not np.isnan(latest["macd_hist"]) else 0.0,
            obv_z20=float(latest["obv_z20"]) if not np.isnan(latest["obv_z20"]) else 0.0,
            cmf_20=float(latest["cmf_20"]) if not np.isnan(latest["cmf_20"]) else 0.0,
            vol_z20=float(latest["vol_z20"]) if not np.isnan(latest["vol_z20"]) else 0.0,
            atr_pct=float(latest["atr_pct"]) if not np.isnan(latest["atr_pct"]) else np.nan,
            atr_med_252=float(_atr_med) if not np.isnan(_atr_med) else np.nan,
            rsi=float(latest["rsi"]) if not np.isnan(latest["rsi"]) else np.nan,
            bollinger_pband=float(latest["bollinger_pband"]) if not np.isnan(latest["bollinger_pband"]) else np.nan,
        )
    if _factor_scores is not None:
        _alpha = compute_alpha_score(_factor_scores, ALPHA_DEFAULT_WEIGHTS)
        trend_score = _alpha
        factor_momentum = _factor_scores.momentum
        factor_trend = _factor_scores.trend
        factor_volume = _factor_scores.volume
        factor_volatility = _factor_scores.volatility
        factor_mean_reversion = _factor_scores.mean_reversion
        alpha_score = _alpha
    else:
        trend_score = legacy_trend_score
        factor_momentum = 0.0
        factor_trend = 0.0
        factor_volume = 0.0
        factor_volatility = 0.0
        factor_mean_reversion = 0.0
        alpha_score = float(legacy_trend_score)

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

    # =========================================================================
    # 저점 반등 지표 계산
    # =========================================================================
    
    # 1. RSI 반등: 최근 N일 내 RSI가 과매도 수준에서 반등했는지
    rsi_reversal = False
    if len(p) >= BOTTOM_RSI_LOOKBACK + 1:
        rsi_recent = p["rsi"].iloc[-(BOTTOM_RSI_LOOKBACK + 1):].dropna()
        if len(rsi_recent) >= 2:
            rsi_min = rsi_recent.min()
            rsi_current = rsi_recent.iloc[-1]
            # 최근에 RSI 30 이하를 터치했고, 현재 35 이상으로 반등
            if rsi_min <= BOTTOM_RSI_OVERSOLD and rsi_current >= BOTTOM_RSI_RECOVERY:
                rsi_reversal = True
    
    # 2. 볼린저 바운스: 저점이 하단밴드 터치 후 현재 밴드 안쪽
    bollinger_bounce = False
    if len(p) >= 10:
        bb_lower = bb.bollinger_lband()
        recent_low = p["Low"].iloc[-10:]
        recent_bb_lower = bb_lower.iloc[-10:]
        # 최근 10일 내 저점이 하단밴드 터치
        touched_lower = (recent_low <= recent_bb_lower * 1.01).any()
        # 현재 종가가 하단밴드 위
        current_above_lower = latest["Close"] > bb_lower.iloc[-1]
        if touched_lower and current_above_lower:
            bollinger_bounce = True
    
    # 3. 거래량 급증 (저점 부근): 최근 저점일에 거래량이 급증
    volume_surge_bottom = False
    if len(p) >= 10:
        recent_closes = p["Close"].iloc[-10:]
        recent_volumes = p["Volume"].iloc[-10:]
        recent_vol_ma = p["vol_ma20"].iloc[-10:]
        low_idx = recent_closes.idxmin()
        if low_idx in recent_volumes.index and low_idx in recent_vol_ma.index:
            vol_at_low = recent_volumes.loc[low_idx]
            vol_ma_at_low = recent_vol_ma.loc[low_idx]
            if not np.isnan(vol_at_low) and not np.isnan(vol_ma_at_low) and vol_ma_at_low > 0:
                if vol_at_low > vol_ma_at_low * BOTTOM_VOLUME_SURGE_MULT:
                    volume_surge_bottom = True
    
    # 4. 지지선 2회 테스트 (Double Bottom 유사)
    support_test = False
    if len(p) >= 30:
        recent_lows = p["Low"].iloc[-30:]
        sorted_lows = recent_lows.nsmallest(5)
        if len(sorted_lows) >= 2:
            low1 = sorted_lows.iloc[0]
            low2 = sorted_lows.iloc[1]
            # 두 저점이 유사한 가격대 (허용 오차 내)
            if abs(low1 - low2) / low1 < BOTTOM_SUPPORT_TOLERANCE:
                # 현재 가격이 저점보다 높음 (반등 확인)
                if latest["Close"] > max(low1, low2) * 1.02:
                    support_test = True
    
    # 5. MACD 다이버전스: 가격은 하락하는데 MACD는 상승
    macd_divergence = False
    if len(p) >= BOTTOM_MACD_DIV_LOOKBACK + 1:
        macd_recent = p["macd_hist"].iloc[-(BOTTOM_MACD_DIV_LOOKBACK + 1):].dropna()
        close_recent = p["Close"].iloc[-(BOTTOM_MACD_DIV_LOOKBACK + 1):]
        if len(macd_recent) >= BOTTOM_MACD_DIV_LOOKBACK and len(close_recent) >= BOTTOM_MACD_DIV_LOOKBACK:
            # 가격: 초반 저점 vs 최근 저점 (가격 하락)
            price_early = close_recent.iloc[:BOTTOM_MACD_DIV_LOOKBACK // 2].min()
            price_late = close_recent.iloc[-(BOTTOM_MACD_DIV_LOOKBACK // 2):].min()
            # MACD: 초반 저점 vs 최근 저점 (MACD 상승)
            macd_early = macd_recent.iloc[:BOTTOM_MACD_DIV_LOOKBACK // 2].min()
            macd_late = macd_recent.iloc[-(BOTTOM_MACD_DIV_LOOKBACK // 2):].min()
            
            if price_late < price_early * 0.99 and macd_late > macd_early:
                macd_divergence = True
    
    # 반등 스코어 계산 (최대 10점)
    bottom_reversal_score = (
        (2.0 if rsi_reversal else 0.0) +
        (1.5 if bollinger_bounce else 0.0) +
        (2.0 if volume_surge_bottom else 0.0) +
        (2.5 if support_test else 0.0) +
        (2.0 if macd_divergence else 0.0)
    )

    # 차트 패턴 감지
    # 차트 패턴: 일봉
    patterns = detect_all_patterns(p[-PATTERN_LOOKBACK_DAYS * 2 :])  # 충분한 데이터 전달

    # 차트 패턴: 주봉 (Weekly)
    # 전체 데이터를 넘기면 내부에서 리샘플링 후 패턴 감지
    w_patterns_list = detect_weekly_patterns(p)
    # 신뢰도가 높은 순으로 정렬 또는 중요도 순으로 정렬 가능하나, 여기선 감지된 순서대로 나열
    # 엑셀 출력을 위해 문자열로 변환 (예: "Golden Cross, Double Bottom")
    # 이름 매핑을 좀 더 예쁘게 할 수도 있음 (영어 그대로 사용)
    weekly_patterns_str = ""
    if w_patterns_list:
        # 영어 -> 한글 매핑
        pattern_map = {
            "golden_cross": "골든크로스",
            "double_bottom": "이중바닥",
            "inverse_head_and_shoulders": "역헤드앤숄더",
            "cup_with_handle": "컵앤핸들",
            "ascending_triangle": "상승삼각형",
            "falling_wedge": "하락쐐기",
            "bullish_engulfing": "강세잉걸핑",
            "morning_star": "모닝스타",
            "weekly_uptrend": "상승추세(정배열)",
        }
        weekly_patterns_str = ", ".join([pattern_map.get(name, name) for name, conf in w_patterns_list])

    return FeatureSet(
        trend_score=trend_score,
        alpha_score=alpha_score,
        factor_momentum=factor_momentum,
        factor_trend=factor_trend,
        factor_volume=factor_volume,
        factor_volatility=factor_volatility,
        factor_mean_reversion=factor_mean_reversion,
        ret_5d=float(latest["ret_5d"]) if not np.isnan(latest["ret_5d"]) else 0.0,
        ret_20d=float(latest["ret_20d"]) if not np.isnan(latest["ret_20d"]) else 0.0,
        vol_z20=float(latest["vol_z20"]) if not np.isnan(latest["vol_z20"]) else 0.0,
        pos_52w=float(latest["pos_52w"]) if not np.isnan(latest["pos_52w"]) and not np.isinf(latest["pos_52w"]) else 0.5,
        atr_pct=float(latest["atr_pct"]) if not np.isnan(latest["atr_pct"]) else 0.0,
        rsi=float(latest["rsi"]) if not np.isnan(latest["rsi"]) else 50.0,
        macd_hist=float(latest["macd_hist"]) if not np.isnan(latest["macd_hist"]) else 0.0,
        stoch_k=float(latest["stoch_k"]) if not np.isnan(latest["stoch_k"]) else 50.0,
        roc_10=float(latest["roc_10"]) if not np.isnan(latest["roc_10"]) else 0.0,
        adx=float(latest["adx"]) if not np.isnan(latest["adx"]) else 0.0,
        ema_gap_20_50=float(latest["ema_gap_20_50"]) if not np.isnan(latest["ema_gap_20_50"]) else 0.0,
        ema_gap_50_200=float(latest["ema_gap_50_200"]) if not np.isnan(latest["ema_gap_50_200"]) else 0.0,
        ema200_slope_20=float(ema200_slope) if not np.isnan(ema200_slope) else 0.0,
        close_to_ema200_pct=float(close_to_ema200) if not np.isnan(close_to_ema200) else 0.0,
        bollinger_pband=float(latest["bollinger_pband"]) if not np.isnan(latest["bollinger_pband"]) else 0.5,
        obv_z20=float(latest["obv_z20"]) if not np.isnan(latest["obv_z20"]) else 0.0,
        cmf_20=float(latest["cmf_20"]) if not np.isnan(latest["cmf_20"]) else 0.0,
        accdist_slope_5=float(latest["accdist_slope_5"]) if not np.isnan(latest["accdist_slope_5"]) else 0.0,
        gap_down_pct=float(gap_down_pct),
        avg_dollar_vol_20d=float(avg_dollar_vol),
        volume_stability_ratio=float(volume_stability_ratio),
        hammer_candle=bool(hammer_candle),
        intraday_recovery=float(intraday_recovery),
        distance_from_10d_low=float(distance_from_10d_low),
        distance_from_10d_high=float(distance_from_10d_high),
        volume_breakout_ratio=float(volume_breakout_ratio),
        volatility_contraction=float(volatility_contraction),
        dividend_yield=float(dividend_yield),
        annual_dividend=float(total_div_1y),
        ema20=float(ema20_latest) if not np.isnan(ema20_latest) else 0.0,
        ema50=float(ema50_latest) if not np.isnan(ema50_latest) else 0.0,
        ema200=float(ema200_latest) if not np.isnan(ema200_latest) else 0.0,
        volume=float(volume_latest) if not np.isnan(volume_latest) else 0.0,
        volume_ma20=float(volume_ma_latest) if not np.isnan(volume_ma_latest) else 0.0,
        obv=float(obv_latest) if not np.isnan(obv_latest) else 0.0,
        obv_ma20=float(obv_ma_latest) if not np.isnan(obv_ma_latest) else 0.0,
        obv_mom_5=float(obv_mom_latest),
        obv_mom_ratio=float(obv_momentum),
        atr_med_252=atr_median_latest,
        atr_buy_max=atr_buy_max,
        atr_sell_max=atr_sell_max,
        close=float(latest_close),
        atr_value=float(atr_value),
        stop_dist=float(stop_dist),
        position_size=float(position_size),
        # 차트 패턴
        pattern_triangle=patterns.triangle,
        pattern_wedge=patterns.wedge,
        pattern_double=patterns.double,
        pattern_head_shoulders=patterns.head_shoulders,
        pattern_cup_handle=patterns.cup_handle,
        pattern_candlestick=patterns.candlestick,
        pattern_golden_cross=patterns.golden_cross,
        weekly_patterns=weekly_patterns_str,
        weekly_patterns_list=w_patterns_list,
        # 저점 반등 지표
        rsi_reversal=rsi_reversal,
        bollinger_bounce=bollinger_bounce,
        volume_surge_bottom=volume_surge_bottom,
        support_test=support_test,
        macd_divergence=macd_divergence,
        bottom_reversal_score=float(bottom_reversal_score),
    )


def feature_row_from_set(ticker: str, feature_set: FeatureSet) -> dict:
    row = {
        "티커": ticker,
        "트렌드점수": feature_set.trend_score,
        "알파점수": feature_set.alpha_score,
        "팩터_모멘텀": feature_set.factor_momentum,
        "팩터_추세": feature_set.factor_trend,
        "팩터_거래량": feature_set.factor_volume,
        "팩터_변동성": feature_set.factor_volatility,
        "팩터_평균회귀": feature_set.factor_mean_reversion,
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
        "dividend_yield": feature_set.dividend_yield,
        "annual_dividend": feature_set.annual_dividend,
        "시장": to_market(ticker),
        "섹터": SECTOR_MAP.get(ticker, "Unknown"),
        "ema20": feature_set.ema20,
        "ema50": feature_set.ema50,
        "ema200": feature_set.ema200,
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
        # 차트 패턴
        "패턴_삼각형": feature_set.pattern_triangle,
        "패턴_쐐기": feature_set.pattern_wedge,
        "패턴_더블": feature_set.pattern_double,
        "패턴_헤드숄더": feature_set.pattern_head_shoulders,
        "패턴_컵핸들": feature_set.pattern_cup_handle,
        "패턴_캔들": feature_set.pattern_candlestick,
        "패턴_골든크로스": feature_set.pattern_golden_cross,
        "주봉패턴": feature_set.weekly_patterns,
        # 저점 반등 지표
        "RSI반등": int(feature_set.rsi_reversal),
        "볼린저바운스": int(feature_set.bollinger_bounce),
        "저점거래량급증": int(feature_set.volume_surge_bottom),
        "지지선테스트": int(feature_set.support_test),
        "MACD다이버전스": int(feature_set.macd_divergence),
        "반등스코어": feature_set.bottom_reversal_score,
    }

    # 주봉 패턴 개별 컬럼 추가
    weekly_cols = {
        "주봉_삼각형": "",
        "주봉_쐐기": "",
        "주봉_더블": "",
        "주봉_헤드숄더": "",
        "주봉_컵핸들": "",
        "주봉_골든크로스": "",
        "주봉_상승추세": "",
    }
    
    # 매핑 (영어 key -> (컬럼명, 표시값))
    w_map = {
        "ascending_triangle": ("주봉_삼각형", "상승삼각형"),
        "falling_wedge": ("주봉_쐐기", "하락쐐기"),
        "double_bottom": ("주봉_더블", "이중바닥"),
        "inverse_head_and_shoulders": ("주봉_헤드숄더", "역헤드앤숄더"),
        "cup_with_handle": ("주봉_컵핸들", "컵앤핸들"),
        "golden_cross": ("주봉_골든크로스", "골든크로스"),
        "weekly_uptrend": ("주봉_상승추세", "상승추세"),
        "bullish_engulfing": ("주봉_캔들", "강세잉걸핑"), 
        "morning_star": ("주봉_캔들", "모닝스타"),
    }
    
    # w_patterns_list는 [(name, conf), ...] 형태
    for name, conf in feature_set.weekly_patterns_list:
        if name in w_map:
            col, val = w_map[name]
            if col in weekly_cols:
                # 이미 값이 있으면 콤마로 연결
                if weekly_cols[col]:
                    weekly_cols[col] += f", {val}"
                else:
                    weekly_cols[col] = val
    
    row.update(weekly_cols)
    return row


def compute_features_snapshot(
    price_map: Mapping[str, pd.DataFrame],
    *,
    include_fundamentals: bool = True,
) -> pd.DataFrame:
    """주어진 종목별 시계열 스냅샷에서 특징 테이블을 생성한다."""

    # 알파 모델: IC 기반 동적 가중치 계산 (전 종목 공통)
    alpha_weights = ALPHA_DEFAULT_WEIGHTS.copy()
    if ALPHA_MODEL_ENABLED:
        try:
            alpha_weights = compute_ic_weights(
                price_map,
                lookback=ALPHA_IC_LOOKBACK,
                forward_days=ALPHA_FORWARD_RETURN_DAYS,
                min_samples=ALPHA_IC_MIN_SAMPLES,
                default_weights=ALPHA_DEFAULT_WEIGHTS,
            )
        except Exception:
            alpha_weights = ALPHA_DEFAULT_WEIGHTS.copy()

    rows = []
    for ticker, frame in price_map.items():
        features = compute_features_for_ticker(frame.dropna(how="all"))
        if features is None:
            continue

        # IC 가중치로 alpha_score 재계산 (기본 가중치 대신 동적 가중치 적용)
        if ALPHA_MODEL_ENABLED and features.factor_momentum != 0.0:
            from alpha_model import FactorScores
            factor_scores = FactorScores(
                momentum=features.factor_momentum,
                trend=features.factor_trend,
                volume=features.factor_volume,
                volatility=features.factor_volatility,
                mean_reversion=features.factor_mean_reversion,
            )
            recalc_alpha = compute_alpha_score(factor_scores, alpha_weights)
            features = FeatureSet(
                **{**features.__dict__, "alpha_score": recalc_alpha, "trend_score": recalc_alpha}
            )

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
            # yfinance에서 가져온 섹터 정보로 업데이트
            if "fund_sector" in out.columns:
                out["섹터"] = out["fund_sector"].fillna(out["섹터"])

    if "최근20일평균거래대금" in out.columns:
        # fund_market_cap 우선 사용, 없으면 유통주식수 * 현재가로 대체.
        market_cap = out["fund_market_cap"] if "fund_market_cap" in out.columns else np.nan
        float_cap = np.nan
        if "fund_float_shares" in out.columns and "close" in out.columns:
            float_cap = out["fund_float_shares"] * out["close"]
        effective_cap = market_cap
        if isinstance(market_cap, pd.Series):
            effective_cap = market_cap.copy()
            if isinstance(float_cap, pd.Series):
                effective_cap = effective_cap.fillna(float_cap)

        turnover = out["최근20일평균거래대금"] / effective_cap
        turnover = turnover.where(np.isfinite(turnover))
        out["거래대금회전율"] = turnover

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
