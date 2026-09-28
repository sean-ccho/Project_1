"""다중 팩터 알파 모델.

5개 팩터 카테고리(Momentum, Trend, Volume, Volatility, Mean-Reversion)별 점수를
계산하고, IC(Information Coefficient) 기반 동적 가중치를 적용하여 종합 alpha
score를 산출한다.

기존 WEIGHTS 딕셔너리 기반의 고정 가중치 trend_score를 대체한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 팩터 점수 계산
# =============================================================================

@dataclass
class FactorScores:
    """5개 팩터 카테고리의 점수를 담는 컨테이너.

    모든 점수는 -1 ~ +1 범위로 정규화된다.
    """

    momentum: float
    trend: float
    volume: float
    volatility: float
    mean_reversion: float

    def as_dict(self) -> dict[str, float]:
        return {
            "momentum": self.momentum,
            "trend": self.trend,
            "volume": self.volume,
            "volatility": self.volatility,
            "mean_reversion": self.mean_reversion,
        }


def _safe_tanh(x: float, scale: float = 1.0) -> float:
    """NaN-safe hyperbolic tangent scaling to [-1, +1]."""
    if np.isnan(x):
        return 0.0
    return float(np.tanh(x / scale))


def compute_factor_scores(p: pd.DataFrame) -> Optional[FactorScores]:
    """종목의 OHLCV 히스토리에서 5개 팩터 카테고리 점수를 계산한다.

    Args:
        p: 단일 종목의 OHLCV 데이터프레임 (최소 130일 이상)

    Returns:
        FactorScores 또는 데이터 부족 시 None
    """
    if len(p) < 130:
        return None

    close = p["Close"].dropna()
    if len(close) < 130:
        return None

    # -------------------------------------------------------------------------
    # 1. Momentum 팩터: 다중 기간 모멘텀
    # -------------------------------------------------------------------------
    ret_21d = close.pct_change(21).iloc[-1]   # 1개월
    ret_63d = close.pct_change(63).iloc[-1]   # 3개월
    ret_126d = close.pct_change(126).iloc[-1]  # 6개월

    # 각 기간 수익률을 tanh로 [-1, 1] 스케일링 후 가중 평균
    mom_1m = _safe_tanh(ret_21d, scale=0.10)
    mom_3m = _safe_tanh(ret_63d, scale=0.20)
    mom_6m = _safe_tanh(ret_126d, scale=0.30)

    # 최근 모멘텀에 더 높은 가중치
    momentum_score = 0.45 * mom_1m + 0.35 * mom_3m + 0.20 * mom_6m

    # -------------------------------------------------------------------------
    # 2. Trend 팩터: EMA 정배열 + ADX + MACD
    # -------------------------------------------------------------------------
    from ta.trend import ADXIndicator, EMAIndicator, MACD

    ema20 = EMAIndicator(close, window=20).ema_indicator()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()

    # EMA 정배열 강도: (EMA20-EMA50)/EMA50 + (EMA50-EMA200)/EMA200
    ema_gap_short = (ema20.iloc[-1] - ema50.iloc[-1]) / (ema50.iloc[-1] + 1e-9)
    ema_gap_long = (ema50.iloc[-1] - ema200.iloc[-1]) / (ema200.iloc[-1] + 1e-9)
    ema_alignment = _safe_tanh(ema_gap_short + ema_gap_long, scale=0.06)

    # ADX 추세 강도
    adx_indicator = ADXIndicator(p["High"], p["Low"], close, window=14)
    adx_val = adx_indicator.adx().iloc[-1]
    adx_score = _safe_tanh((adx_val - 25.0), scale=15.0) if not np.isnan(adx_val) else 0.0

    # MACD 방향
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = macd.macd_diff().iloc[-1]
    macd_score = _safe_tanh(macd_hist, scale=1.0)

    trend_score = 0.40 * ema_alignment + 0.35 * adx_score + 0.25 * macd_score

    # -------------------------------------------------------------------------
    # 3. Volume 팩터: OBV + CMF + 거래량 Z-score
    # -------------------------------------------------------------------------
    from ta.volume import (
        ChaikinMoneyFlowIndicator,
        OnBalanceVolumeIndicator,
    )

    volume = p["Volume"]

    # OBV 모멘텀 (20일 Z-score)
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_ma20 = obv.rolling(20).mean()
    obv_std20 = obv.rolling(20).std(ddof=0)
    obv_z = ((obv - obv_ma20) / (obv_std20 + 1e-9)).iloc[-1]
    obv_score = _safe_tanh(obv_z, scale=2.0)

    # CMF (20일 Chaikin Money Flow)
    cmf = ChaikinMoneyFlowIndicator(
        p["High"], p["Low"], close, volume, window=20
    ).chaikin_money_flow().iloc[-1]
    cmf_score = _safe_tanh(cmf, scale=0.15)

    # 거래량 Z-score
    vol_ma20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std(ddof=0)
    vol_z = ((volume - vol_ma20) / (vol_std20 + 1e-9)).iloc[-1]
    vol_z_score = _safe_tanh(vol_z, scale=3.0)

    volume_score = 0.40 * obv_score + 0.35 * cmf_score + 0.25 * vol_z_score

    # -------------------------------------------------------------------------
    # 4. Volatility 팩터: 낮은 변동성 선호 + 변동성 압축
    # -------------------------------------------------------------------------
    from ta.volatility import AverageTrueRange

    atr = AverageTrueRange(p["High"], p["Low"], close, window=14).average_true_range()
    atr_pct = (atr / close).iloc[-1]

    # ATR% 역수: 변동성이 낮을수록 점수가 높음
    # 중앙값(252일) 대비 현재 ATR%의 비율
    atr_pct_series = atr / close
    atr_median = atr_pct_series.rolling(min(252, len(atr_pct_series)), min_periods=60).median().iloc[-1]

    if not np.isnan(atr_pct) and not np.isnan(atr_median) and atr_median > 0:
        vol_ratio = atr_pct / atr_median
        # 비율 < 1이면 변동성 압축(좋음), > 1이면 변동성 확대(나쁨)
        low_vol_score = _safe_tanh(1.0 - vol_ratio, scale=0.3)
    else:
        low_vol_score = 0.0

    # 변동성 압축: 최근 변동성이 중앙값보다 낮으면 양수
    contraction = 0.0
    if not np.isnan(atr_pct) and not np.isnan(atr_median) and atr_median > 0:
        contraction = _safe_tanh((atr_median - atr_pct) / atr_median, scale=0.3)

    volatility_score = 0.55 * low_vol_score + 0.45 * contraction

    # -------------------------------------------------------------------------
    # 5. Mean-Reversion 팩터: RSI 평균회귀 + 볼린저 위치
    # -------------------------------------------------------------------------
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands

    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]

    # RSI 평균회귀: 50 근처가 중립, 극단에서 반대 방향 점수 부여
    # RSI 30 이하 → 강한 매수(+1), RSI 70 이상 → 강한 매도(-1)
    if not np.isnan(rsi):
        rsi_mr_score = _safe_tanh((50.0 - rsi) / 10.0, scale=1.5)
    else:
        rsi_mr_score = 0.0

    # 볼린저 밴드 위치: 0.5가 중앙, 0에 가까울수록 매수, 1에 가까울수록 매도
    bb = BollingerBands(close, window=20, window_dev=2)
    pband = bb.bollinger_pband().iloc[-1]
    if not np.isnan(pband):
        bb_mr_score = _safe_tanh((0.5 - pband), scale=0.3)
    else:
        bb_mr_score = 0.0

    mean_reversion_score = 0.55 * rsi_mr_score + 0.45 * bb_mr_score

    return FactorScores(
        momentum=float(np.clip(momentum_score, -1, 1)),
        trend=float(np.clip(trend_score, -1, 1)),
        volume=float(np.clip(volume_score, -1, 1)),
        volatility=float(np.clip(volatility_score, -1, 1)),
        mean_reversion=float(np.clip(mean_reversion_score, -1, 1)),
    )


def compute_factor_scores_from_indicators(
    *,
    ret_20d: float,
    ret_63d: float,
    ret_126d: float,
    ema_gap_20_50: float,
    ema_gap_50_200: float,
    adx: float,
    macd_hist: float,
    obv_z20: float,
    cmf_20: float,
    vol_z20: float,
    atr_pct: float,
    atr_med_252: float,
    rsi: float,
    bollinger_pband: float,
) -> Optional[FactorScores]:
    """이미 계산된 지표값으로 팩터 점수를 산출한다 (중복 계산 없음).

    features.py에서 이미 계산한 지표를 재활용하여 ta 라이브러리 호출을 생략한다.
    IC 계산에는 기존 compute_factor_scores(p)를 사용한다.
    """
    # 1. Momentum: 다중 기간 수익률
    mom_1m = _safe_tanh(ret_20d, scale=0.10)
    mom_3m = _safe_tanh(ret_63d, scale=0.20)
    mom_6m = _safe_tanh(ret_126d, scale=0.30)
    momentum_score = 0.45 * mom_1m + 0.35 * mom_3m + 0.20 * mom_6m

    # 2. Trend: EMA 정배열 + ADX + MACD
    ema_alignment = _safe_tanh(ema_gap_20_50 + ema_gap_50_200, scale=0.06)
    adx_score = _safe_tanh((adx - 25.0), scale=15.0) if not np.isnan(adx) else 0.0
    macd_score = _safe_tanh(macd_hist, scale=1.0)
    trend_score = 0.40 * ema_alignment + 0.35 * adx_score + 0.25 * macd_score

    # 3. Volume: OBV + CMF + 거래량 Z-score
    obv_score = _safe_tanh(obv_z20, scale=2.0)
    cmf_score = _safe_tanh(cmf_20, scale=0.15)
    vol_z_score = _safe_tanh(vol_z20, scale=3.0)
    volume_score = 0.40 * obv_score + 0.35 * cmf_score + 0.25 * vol_z_score

    # 4. Volatility: ATR% 역수 + 변동성 압축
    if not np.isnan(atr_pct) and not np.isnan(atr_med_252) and atr_med_252 > 0:
        vol_ratio = atr_pct / atr_med_252
        low_vol_score = _safe_tanh(1.0 - vol_ratio, scale=0.3)
        contraction = _safe_tanh((atr_med_252 - atr_pct) / atr_med_252, scale=0.3)
    else:
        low_vol_score = 0.0
        contraction = 0.0
    volatility_score = 0.55 * low_vol_score + 0.45 * contraction

    # 5. Mean-Reversion: RSI 평균회귀 + 볼린저 위치
    rsi_mr_score = _safe_tanh((50.0 - rsi) / 10.0, scale=1.5) if not np.isnan(rsi) else 0.0
    bb_mr_score = _safe_tanh((0.5 - bollinger_pband), scale=0.3) if not np.isnan(bollinger_pband) else 0.0
    mean_reversion_score = 0.55 * rsi_mr_score + 0.45 * bb_mr_score

    return FactorScores(
        momentum=float(np.clip(momentum_score, -1, 1)),
        trend=float(np.clip(trend_score, -1, 1)),
        volume=float(np.clip(volume_score, -1, 1)),
        volatility=float(np.clip(volatility_score, -1, 1)),
        mean_reversion=float(np.clip(mean_reversion_score, -1, 1)),
    )


# =============================================================================
# IC 기반 동적 가중치 계산
# =============================================================================

def compute_ic_weights(
    price_map: Mapping[str, pd.DataFrame],
    *,
    lookback: int = 60,
    forward_days: int = 5,
    min_samples: int = 30,
    default_weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """최근 lookback일 동안 각 팩터의 IC를 계산하여 동적 가중치를 반환한다.

    IC = rank_correlation(factor_score, forward_return)
    IC가 높을수록 해당 팩터의 예측력이 좋으므로 가중치를 높인다.

    Args:
        price_map: {ticker: OHLCV DataFrame} 매핑
        lookback: IC 계산 기간 (거래일, 역사적 데이터에서)
        forward_days: 미래 수익률 기간 (IC 계산용)
        min_samples: IC 계산에 필요한 최소 종목 수
        default_weights: IC 계산 불가 시 폴백 가중치

    Returns:
        {factor_name: weight} — 합이 1.0인 가중치 딕셔너리
    """
    if default_weights is None:
        default_weights = {
            "momentum": 0.30,
            "trend": 0.25,
            "volume": 0.15,
            "volatility": 0.15,
            "mean_reversion": 0.15,
        }

    factor_names = list(default_weights.keys())

    # 각 과거 시점에서 팩터 점수와 미래 수익률을 수집
    # 롤링 IC를 계산하기 위해 여러 시점의 cross-sectional 데이터 필요
    ic_accum: dict[str, list[float]] = {f: [] for f in factor_names}

    # 공통 날짜 인덱스 추출
    all_dates: set[pd.Timestamp] = set()
    for frame in price_map.values():
        if "Close" in frame.columns and len(frame) > 0:
            all_dates.update(frame.index.tolist())

    if not all_dates:
        return default_weights.copy()

    sorted_dates = sorted(all_dates)
    if len(sorted_dates) < lookback + forward_days + 130:
        return default_weights.copy()

    # lookback 기간 내에서 주간(5일) 단위로 IC 측정점 샘플링
    eval_end = len(sorted_dates) - forward_days - 1
    eval_start = max(130, eval_end - lookback)

    sample_indices = list(range(eval_start, eval_end, 5))  # 매 5일마다
    if not sample_indices:
        return default_weights.copy()

    for idx in sample_indices:
        cutoff = sorted_dates[idx]
        forward_date = sorted_dates[min(idx + forward_days, len(sorted_dates) - 1)]

        factor_values: dict[str, list[float]] = {f: [] for f in factor_names}
        forward_returns: list[float] = []

        for ticker, frame in price_map.items():
            if ticker in ("SPY", "QQQ", "IWM"):
                continue  # 벤치마크 제외

            history = frame.loc[:cutoff]
            if len(history) < 130:
                continue

            # 미래 수익률 계산
            future = frame.loc[cutoff:forward_date]
            if len(future) < 2:
                continue

            close_at_cutoff = future["Close"].iloc[0]
            close_at_forward = future["Close"].iloc[-1]
            if np.isnan(close_at_cutoff) or np.isnan(close_at_forward) or close_at_cutoff == 0:
                continue

            fwd_ret = (close_at_forward / close_at_cutoff) - 1.0

            # 팩터 점수 계산
            scores = compute_factor_scores(history)
            if scores is None:
                continue

            score_dict = scores.as_dict()
            for f in factor_names:
                factor_values[f].append(score_dict[f])
            forward_returns.append(fwd_ret)

        # Cross-sectional IC 계산 (Spearman rank correlation)
        if len(forward_returns) >= min_samples:
            for f in factor_names:
                try:
                    corr, _ = stats.spearmanr(factor_values[f], forward_returns)
                    if not np.isnan(corr):
                        ic_accum[f].append(corr)
                except Exception:
                    pass

    # 평균 IC 계산 → 가중치 변환
    avg_ics: dict[str, float] = {}
    for f in factor_names:
        if ic_accum[f]:
            avg_ic = np.mean(ic_accum[f])
            avg_ics[f] = max(avg_ic, 0.0)  # 음수 IC는 0으로 처리
        else:
            avg_ics[f] = default_weights[f]  # 데이터 부족 시 기본값

    total_ic = sum(avg_ics.values())
    if total_ic <= 0:
        return default_weights.copy()

    # 정규화하여 합이 1.0이 되도록
    weights = {f: ic / total_ic for f, ic in avg_ics.items()}

    return weights


# =============================================================================
# 종합 Alpha Score 계산
# =============================================================================

def compute_alpha_score(
    factor_scores: FactorScores,
    weights: dict[str, float],
) -> float:
    """팩터 점수와 가중치를 결합하여 종합 alpha score를 산출한다.

    Args:
        factor_scores: 5개 팩터 카테고리 점수
        weights: 팩터별 가중치 (합 = 1.0)

    Returns:
        종합 alpha score (대략 -1 ~ +1 범위)
    """
    scores = factor_scores.as_dict()
    alpha = sum(scores[f] * weights.get(f, 0.0) for f in scores)
    return float(alpha)
