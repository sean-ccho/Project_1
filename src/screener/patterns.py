"""차트 패턴 인식 모듈.

기술적 분석에서 사용하는 주요 차트 패턴을 자동으로 감지한다.

[가격 패턴 — 8종]
- 대칭 삼각형 (Symmetrical Triangle)
- 상승 삼각형 (Ascending Triangle)
- 하락 삼각형 (Descending Triangle)
- 상승 쐐기 (Rising Wedge)
- 하락 쐐기 (Falling Wedge)
- 더블 바텀 (Double Bottom)
- 더블 탑 (Double Top)
- 컵위드핸들 (Cup with Handle)

[반전 패턴 — 2종]
- 헤드앤숄더 (Head and Shoulders)
- 역헤드앤숄더 (Inverse Head and Shoulders)

[캔들스틱 패턴 — 5종]
- 도지 (Doji)
- 강세 잉걸핑 (Bullish Engulfing)
- 약세 잉걸핑 (Bearish Engulfing)
- 모닝스타 (Morning Star)
- 이브닝스타 (Evening Star)

[이동평균 패턴 — 4종]
- 골든크로스 (Golden Cross)
- 데드크로스 (Death Cross)
- 골든크로스 임박 (Golden Cross Imminent)
- 데드크로스 임박 (Death Cross Imminent)

[추세 상태 지표 — 6종] (주봉/월봉 전용, 각 3단계)
- 주봉 정배열 완전 (Weekly Uptrend Full)     — Price > EMA10 > EMA20 > EMA50
- 주봉 정배열 눌림목 (Weekly Uptrend Pullback) — EMA10 > EMA20 > EMA50, Price > EMA20
- 주봉 정배열 형성중 (Weekly Uptrend Forming) — EMA10 > EMA20, Price > EMA10
- 월봉 정배열 완전 (Monthly Uptrend Full)    — Price > EMA3 > EMA6 > EMA12
- 월봉 정배열 눌림목 (Monthly Uptrend Pullback) — EMA3 > EMA6 > EMA12, Price > EMA6
- 월봉 정배열 형성중 (Monthly Uptrend Forming) — EMA3 > EMA6, Price > EMA3

v2: 거래량 확인, 트렌드 컨텍스트 보정, 시간 대칭 검증 등
    월스트리트 투자자급 정확도로 개선
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress

from screener.config import (
    PATTERN_LOOKBACK_DAYS,
    PATTERN_MIN_TOUCHES,
    PATTERN_CONVERGENCE_THRESHOLD,
    PEAK_PROMINENCE,
    PATTERN_TREND_LOOKBACK,
    PATTERN_MIN_R2,
    PATTERN_FLAT_TOLERANCE,
    PATTERN_BREAKOUT_CONFIRM,
    PATTERN_VOLUME_SURGE,
    PATTERN_MIN_GAP_DAYS,
    PATTERN_DOUBLE_TOLERANCE,
    PATTERN_SHOULDER_TIME_RATIO,
)


@dataclass
class PatternResult:
    """패턴 감지 결과를 담는 컨테이너."""
    detected: bool
    pattern_type: str
    confidence: float
    breakout_level: Optional[float] = None


# =============================================================================
# 유틸리티 함수
# =============================================================================

def find_peaks_and_troughs(
    prices: pd.Series,
    order: int = 7,
    prominence_pct: float = 0.03
) -> Tuple[np.ndarray, np.ndarray]:
    """가격 시계열에서 고점과 저점 인덱스를 찾는다.

    Args:
        prices: 종가 시계열
        order: 양쪽으로 비교할 포인트 수 (5→7: 노이즈 감소)
        prominence_pct: 최소 돌출도 (가격의 %)

    Returns:
        (고점 인덱스 배열, 저점 인덱스 배열)
    """
    values = prices.values

    # scipy argrelextrema로 극값 찾기
    peaks = argrelextrema(values, np.greater_equal, order=order)[0]
    troughs = argrelextrema(values, np.less_equal, order=order)[0]

    # 돌출도 기준으로 필터링
    if len(peaks) > 0 and prominence_pct > 0:
        avg_price = np.mean(values)
        min_prominence = avg_price * prominence_pct

        filtered_peaks = []
        for p in peaks:
            left_min = np.min(values[max(0, p - order):p]) if p > 0 else values[p]
            right_min = np.min(values[p + 1:min(len(values), p + order + 1)]) if p < len(values) - 1 else values[p]
            prominence = values[p] - max(left_min, right_min)
            if prominence >= min_prominence:
                filtered_peaks.append(p)
        peaks = np.array(filtered_peaks)

    if len(troughs) > 0 and prominence_pct > 0:
        avg_price = np.mean(values)
        min_prominence = avg_price * prominence_pct

        filtered_troughs = []
        for t in troughs:
            left_max = np.max(values[max(0, t - order):t]) if t > 0 else values[t]
            right_max = np.max(values[t + 1:min(len(values), t + order + 1)]) if t < len(values) - 1 else values[t]
            prominence = min(left_max, right_max) - values[t]
            if prominence >= min_prominence:
                filtered_troughs.append(t)
        troughs = np.array(filtered_troughs)

    return peaks, troughs


def fit_trendline(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """추세선을 피팅하고 기울기, 절편, R² 값을 반환한다.

    Returns:
        (slope, intercept, r_squared)
    """
    if len(x) < 2:
        return 0.0, 0.0, 0.0

    # x값이 모두 동일하면 선형회귀 불가
    if np.all(x == x[0]):
        return 0.0, float(np.mean(y)), 0.0

    try:
        result = linregress(x, y)
        r_squared = result.rvalue ** 2
        return result.slope, result.intercept, r_squared
    except Exception:
        return 0.0, 0.0, 0.0


def line_flatness(prices: np.ndarray) -> float:
    """수평 추세선의 평탄도를 측정한다.

    상승/하락 삼각형의 수평 저항선·지지선은 기울기가 0에 가까워 R²가
    구조적으로 0에 수렴한다 (R²는 기울기 있는 선의 적합도 지표이며,
    수평선은 x와의 상관계수가 0이라 R²≈0). 따라서 수평선은 R² 대신
    "터치 가격들이 하나의 수평 레벨에 얼마나 모여있는가"로 검증한다.

    Args:
        prices: 추세선을 이루는 터치(고점 또는 저점) 가격 배열

    Returns:
        (최대 - 최소) / 평균. 0에 가까울수록 완벽한 수평선.
    """
    if len(prices) == 0:
        return 1.0
    mean = float(np.mean(prices))
    if mean <= 0:
        return 1.0
    return (float(np.max(prices)) - float(np.min(prices))) / mean


def segment_trend(prices: np.ndarray, threshold: float = 0.07) -> str:
    """가격 구간의 추세 방향을 시작→끝 변화율로 판단한다.

    get_trend_context가 "최근(트레일링)" 추세를 보는 것과 달리, 패턴 "진입"
    구간처럼 임의의 과거 구간 추세를 판단할 때 사용한다 (예: H&S 좌어깨 직전).

    Returns:
        "uptrend" / "downtrend" / "sideways"
    """
    if len(prices) < 3:
        return "sideways"
    start = float(prices[0])
    end = float(prices[-1])
    if start <= 0:
        return "sideways"
    change = (end - start) / start
    if change > threshold:
        return "uptrend"
    if change < -threshold:
        return "downtrend"
    return "sideways"


# =============================================================================
# 거래량 확인 유틸리티 (v2 추가)
# =============================================================================

def _get_volume_ma(df: pd.DataFrame, period: int = 20) -> Optional[pd.Series]:
    """거래량 이동 평균을 계산한다. Volume 컬럼이 없으면 None 반환."""
    if "Volume" not in df.columns:
        return None
    vol = df["Volume"].values.flatten()
    vol_series = pd.Series(vol)
    if vol_series.sum() == 0:
        return None
    return vol_series.rolling(period, min_periods=1).mean()


def check_volume_surge(
    df: pd.DataFrame,
    idx: int,
    ma_period: int = 20,
    threshold: float = None,
) -> bool:
    """특정 시점의 거래량이 이동 평균 대비 급증했는지 확인한다.

    Args:
        df: OHLCV 데이터
        idx: 검사할 인덱스 (df 내 위치)
        ma_period: 이동 평균 기간
        threshold: 배수 기준 (None이면 config 값 사용)

    Returns:
        급증 여부 (Volume 없으면 True 반환 — 거래량 데이터 부족 시 블로킹 방지)
    """
    if threshold is None:
        threshold = PATTERN_VOLUME_SURGE

    if "Volume" not in df.columns:
        return True  # 거래량 데이터 없으면 패턴 자체는 통과

    vol = df["Volume"].values.flatten()
    if idx >= len(vol) or idx < ma_period:
        return True

    vol_at_idx = float(vol[idx])
    vol_ma = float(np.mean(vol[max(0, idx - ma_period):idx]))
    if vol_ma <= 0:
        return True

    return vol_at_idx >= vol_ma * threshold


def check_volume_declining(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> bool:
    """구간 내 거래량이 감소 추세인지 확인한다.

    Args:
        df: OHLCV 데이터
        start_idx, end_idx: 검사 구간 (df 내 인덱스)

    Returns:
        감소 추세 여부 (Volume 없으면 True 반환)
    """
    if "Volume" not in df.columns:
        return True

    vol = df["Volume"].values.flatten()
    if end_idx >= len(vol):
        end_idx = len(vol) - 1
    if start_idx >= end_idx:
        return True

    segment = vol[start_idx:end_idx + 1].astype(float)
    if len(segment) < 5:
        return True

    # 전반부 평균 vs 후반부 평균
    mid = len(segment) // 2
    first_half_avg = np.mean(segment[:mid])
    second_half_avg = np.mean(segment[mid:])

    if first_half_avg <= 0:
        return True

    return second_half_avg < first_half_avg


# =============================================================================
# 패턴 감지 함수
# =============================================================================

def get_trend_context(df: pd.DataFrame, lookback: int = None) -> str:
    """최근 트렌드 방향을 판단한다.

    Args:
        df: OHLC 데이터프레임
        lookback: 트렌드 판단 기간 (기본값: PATTERN_TREND_LOOKBACK)

    Returns:
        "uptrend" / "downtrend" / "sideways"
    """
    if lookback is None:
        lookback = PATTERN_TREND_LOOKBACK

    if len(df) < lookback:
        return "sideways"

    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    start_price = float(close_vals[0])
    end_price = float(close_vals[-1])

    if start_price <= 0:
        return "sideways"

    change_pct = (end_price - start_price) / start_price

    # EMA 기반 추가 판단
    ema_slope = change_pct  # 기본값

    # 20일 EMA 계산
    if len(df) >= 20:
        close_series = pd.Series(df["Close"].values.flatten()[-lookback:])
        ema20_recent = close_series.ewm(span=20, adjust=False).mean()
        if len(ema20_recent) >= lookback:
            ema_start = float(ema20_recent.iloc[0])
            ema_end = float(ema20_recent.iloc[-1])
            if ema_start > 0:
                ema_slope = (ema_end - ema_start) / ema_start

    # 상승 추세: 7% 이상 상승 + EMA 상승 (5%→7%로 상향)
    if change_pct > 0.07 and ema_slope > 0:
        return "uptrend"
    # 하락 추세: 7% 이상 하락 + EMA 하락
    elif change_pct < -0.07 and ema_slope < 0:
        return "downtrend"

    return "sideways"


def detect_triangle(df: pd.DataFrame, lookback: int = 60) -> PatternResult:
    """삼각형 수렴 패턴을 감지한다.

    - 대칭 삼각형: 고점 하락 + 저점 상승
    - 상승 삼각형: 고점 수평 + 저점 상승
    - 하락 삼각형: 고점 하락 + 저점 수평

    v2: 거래량 감소 추세 확인, R² 기준 강화
    """
    if len(df) < lookback:
        return PatternResult(False, "none", 0.0)

    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    closes = pd.Series(close_vals, index=range(len(close_vals)))

    peaks, troughs = find_peaks_and_troughs(closes, order=7, prominence_pct=PEAK_PROMINENCE)

    if len(peaks) < PATTERN_MIN_TOUCHES or len(troughs) < PATTERN_MIN_TOUCHES:
        return PatternResult(False, "none", 0.0)

    # 추세선 피팅
    peak_slope, peak_intercept, peak_r2 = fit_trendline(peaks, closes.iloc[peaks].values)
    trough_slope, trough_intercept, trough_r2 = fit_trendline(troughs, closes.iloc[troughs].values)

    # 기울기 정규화 (일일 수익률 기준)
    avg_price = closes.mean()
    peak_slope_pct = peak_slope / avg_price if avg_price > 0 else 0
    trough_slope_pct = trough_slope / avg_price if avg_price > 0 else 0

    # 수평 추세선의 평탄도 (R²로는 수평선을 검증할 수 없으므로 별도 측정)
    peak_flatness = line_flatness(closes.iloc[peaks].values)
    trough_flatness = line_flatness(closes.iloc[troughs].values)

    # 수렴 판정
    slope_threshold = PATTERN_CONVERGENCE_THRESHOLD / lookback

    pattern_type = "none"
    confidence = 0.0

    # 추세선 적합도 검증 규칙:
    #  - 기울어진 추세선 → R² ≥ PATTERN_MIN_R2 (선형 적합도)
    #  - 수평 추세선   → 평탄도 ≤ PATTERN_FLAT_TOLERANCE (R²는 수평선에서 항상 0에 수렴)

    # 대칭 삼각형: 고점↘ 저점↗ (양쪽 모두 기울어짐 → 양쪽 R² 검증)
    if peak_slope_pct < -slope_threshold and trough_slope_pct > slope_threshold:
        if peak_r2 >= PATTERN_MIN_R2 and trough_r2 >= PATTERN_MIN_R2:
            pattern_type = "symmetrical_triangle"
            confidence = min(peak_r2, trough_r2)

    # 상승 삼각형: 고점 수평 + 저점↗ (저항선=수평→평탄도, 지지선=기울기→R²)
    elif abs(peak_slope_pct) < slope_threshold and trough_slope_pct > slope_threshold:
        if trough_r2 >= PATTERN_MIN_R2 and peak_flatness <= PATTERN_FLAT_TOLERANCE:
            pattern_type = "ascending_triangle"
            confidence = trough_r2

    # 하락 삼각형: 고점↘ + 저점 수평 (저항선=기울기→R², 지지선=수평→평탄도)
    elif peak_slope_pct < -slope_threshold and abs(trough_slope_pct) < slope_threshold:
        if peak_r2 >= PATTERN_MIN_R2 and trough_flatness <= PATTERN_FLAT_TOLERANCE:
            pattern_type = "descending_triangle"
            confidence = peak_r2

    if pattern_type != "none" and confidence > 0.5:
        # v2: 거래량 감소 추세 확인 (삼각형 수렴의 핵심 특성)
        df_recent = df.tail(lookback)
        vol_declining = check_volume_declining(df_recent, 0, len(df_recent) - 1)
        if not vol_declining:
            confidence *= 0.7  # 거래량이 안 줄면 신뢰도 감소

        # 예상 돌파 가격 (두 추세선의 교점 근처)
        last_idx = len(closes) - 1
        upper_level = peak_intercept + peak_slope * last_idx
        lower_level = trough_intercept + trough_slope * last_idx
        breakout_level = (upper_level + lower_level) / 2

        if confidence > 0.5:
            return PatternResult(True, pattern_type, confidence, breakout_level)

    return PatternResult(False, "none", 0.0)


def detect_wedge(df: pd.DataFrame, lookback: int = 60) -> PatternResult:
    """쐐기 패턴을 감지한다.

    - 상승 쐐기 (Rising Wedge): 두 선 모두 상승, 하지만 수렴 → 하락 신호
    - 하락 쐐기 (Falling Wedge): 두 선 모두 하락, 하지만 수렴 → 상승 신호

    v2: 수렴 판정 엄격화 (30%→20%), 거래량 감소 확인, 최소 패턴 기간 검증
    """
    if len(df) < lookback:
        return PatternResult(False, "none", 0.0)

    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    closes = pd.Series(close_vals, index=range(len(close_vals)))

    peaks, troughs = find_peaks_and_troughs(closes, order=7, prominence_pct=PEAK_PROMINENCE)

    # 최소 3개의 터치 포인트 필요
    if len(peaks) < PATTERN_MIN_TOUCHES or len(troughs) < PATTERN_MIN_TOUCHES:
        return PatternResult(False, "none", 0.0)

    # v2: 패턴이 최소 20거래일 이상 걸쳐야 함
    peak_span = int(peaks[-1]) - int(peaks[0]) if len(peaks) >= 2 else 0
    trough_span = int(troughs[-1]) - int(troughs[0]) if len(troughs) >= 2 else 0
    if max(peak_span, trough_span) < 20:
        return PatternResult(False, "none", 0.0)

    peak_slope, _, peak_r2 = fit_trendline(peaks, closes.iloc[peaks].values)
    trough_slope, _, trough_r2 = fit_trendline(troughs, closes.iloc[troughs].values)

    pattern_type = "none"
    confidence = 0.0

    # R² 값이 모두 충분히 높아야 함 (명확한 추세선)
    min_r2 = min(peak_r2, trough_r2)
    if min_r2 < PATTERN_MIN_R2:
        return PatternResult(False, "none", 0.0)

    # v2: 수렴 판정 강화: 기울기 차이 비율 계산 (30%→20%)
    max_slope = max(abs(peak_slope), abs(trough_slope), 0.001)
    slope_diff_ratio = abs(peak_slope - trough_slope) / max_slope

    # 수렴 조건: 기울기 차이가 20% 이하
    is_converging = slope_diff_ratio < 0.20

    # 추가: 두 기울기의 부호가 같아야 함 (wedge 특성)
    same_direction = (peak_slope > 0 and trough_slope > 0) or (peak_slope < 0 and trough_slope < 0)

    if not same_direction or not is_converging:
        return PatternResult(False, "none", 0.0)

    # 수렴 = 두 추세선 간격(상단-하단)이 우측으로 갈수록 좁아짐
    #      = d(gap)/dx = peak_slope - trough_slope < 0
    #      = peak_slope < trough_slope  (상승·하락 쐐기 공통 조건)
    # 상승 쐐기: 둘 다 상승하지만 수렴 (저점선이 더 가파르게 상승)
    if peak_slope > 0 and trough_slope > 0 and peak_slope < trough_slope:
        pattern_type = "rising_wedge"
        confidence = min_r2

    # 하락 쐐기: 둘 다 하락하지만 수렴 (고점선이 더 가파르게 하락)
    elif peak_slope < 0 and trough_slope < 0 and peak_slope < trough_slope:
        pattern_type = "falling_wedge"
        confidence = min_r2

    if pattern_type != "none" and confidence > 0.5:
        # v2: 거래량 감소 추세 확인
        df_recent = df.tail(lookback)
        vol_declining = check_volume_declining(df_recent, 0, len(df_recent) - 1)
        if not vol_declining:
            confidence *= 0.7

        if confidence > 0.5:
            return PatternResult(True, pattern_type, confidence)

    return PatternResult(False, "none", 0.0)


def detect_double_bottom_top(df: pd.DataFrame, lookback: int = 60) -> PatternResult:
    """더블 바텀/탑 패턴을 감지한다.

    - 더블 바텀: 비슷한 가격의 두 저점 (W 모양) → 상승 반전
    - 더블 탑: 비슷한 가격의 두 고점 (M 모양) → 하락 반전

    v2 개선:
    - 허용치 3%→2% (PATTERN_DOUBLE_TOLERANCE)
    - 두 극점 간 최소 간격 10거래일 (PATTERN_MIN_GAP_DAYS)
    - 거래량 확인: 두 번째 저점 거래량 < 첫 번째
    - 트렌드 보정 수정: 하락 추세→더블 바텀 신뢰도 증가 (기존은 반대였음)
    """
    if len(df) < lookback:
        return PatternResult(False, "none", 0.0)

    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    closes = pd.Series(close_vals, index=range(len(close_vals)))
    current_price = float(close_vals[-1])

    # 트렌드 컨텍스트 확인
    trend = get_trend_context(df, lookback=PATTERN_TREND_LOOKBACK)

    peaks, troughs = find_peaks_and_troughs(closes, order=7, prominence_pct=PEAK_PROMINENCE)

    pattern_type = "none"
    confidence = 0.0
    breakout_level = None

    # 더블 바텀 체크
    if len(troughs) >= 2:
        last_two_troughs = troughs[-2:]
        trough_prices = closes.iloc[last_two_troughs].values

        # v2: 허용치 config에서 가져옴 (3%→2%)
        price_diff_pct = abs(trough_prices[0] - trough_prices[1]) / trough_prices[0]

        # v2: 두 저점 간 최소 간격 확인
        gap_days = int(last_two_troughs[1]) - int(last_two_troughs[0])

        # 두 저점 사이에 고점이 있는지
        between_peaks = [p for p in peaks if last_two_troughs[0] < p < last_two_troughs[1]]

        if (price_diff_pct < PATTERN_DOUBLE_TOLERANCE
                and gap_days >= PATTERN_MIN_GAP_DAYS
                and len(between_peaks) >= 1):

            # 네크라인: 두 저점 사이 가장 높은 고점 사용
            neckline = float(closes.iloc[between_peaks].max())

            # 돌파 확인: 현재 가격이 네크라인 근처 또는 위
            # v2: 최소 패턴 높이 확인 (3.5% 이상이어야 유효, 너무 작은 패턴 무시)
            pattern_height = (neckline - max(trough_prices)) / max(trough_prices)
            
            breakout_confirmed = current_price > max(trough_prices) * (1 + PATTERN_BREAKOUT_CONFIRM)

            if breakout_confirmed and pattern_height >= 0.035:
                pattern_type = "double_bottom"
                confidence = 1.0 - price_diff_pct * 10
                breakout_level = neckline

                # v2: 거래량 확인 — 두 번째 저점 거래량이 첫 번째보다 낮으면 신뢰도 증가
                df_recent = df.tail(lookback)
                if "Volume" in df_recent.columns:
                    vol = df_recent["Volume"].values.flatten()
                    t1_idx = int(last_two_troughs[0])
                    t2_idx = int(last_two_troughs[1])
                    if t1_idx < len(vol) and t2_idx < len(vol):
                        if float(vol[t2_idx]) < float(vol[t1_idx]):
                            confidence = min(1.0, confidence * 1.1)  # 거래량 감소 확인
                        else:
                            confidence *= 0.85  # 거래량 패턴 불일치

                # v2 돌파 시점 거래량 급증 확인
                if check_volume_surge(df_recent, len(df_recent) - 1):
                    confidence = min(1.0, confidence * 1.1)

    # 더블 탑 체크
    if pattern_type == "none" and len(peaks) >= 2:
        last_two_peaks = peaks[-2:]
        peak_prices = closes.iloc[last_two_peaks].values

        price_diff_pct = abs(peak_prices[0] - peak_prices[1]) / peak_prices[0]

        # v2: 최소 간격 확인
        gap_days = int(last_two_peaks[1]) - int(last_two_peaks[0])

        between_troughs = [t for t in troughs if last_two_peaks[0] < t < last_two_peaks[1]]

        if (price_diff_pct < PATTERN_DOUBLE_TOLERANCE
                and gap_days >= PATTERN_MIN_GAP_DAYS
                and len(between_troughs) >= 1):

            # 네크라인: 두 고점 사이 가장 낮은 저점 사용
            neckline = float(closes.iloc[between_troughs].min())

            # 돌파 확인: 현재 가격이 두 고점보다 낮아야 함
            # v2: 최소 패턴 높이 확인 (3.5% 이상)
            pattern_height = (min(peak_prices) - neckline) / neckline  # neckline이 더 낮음 (Peak > Neckline)
            # 위 식은 음수가 나옴. Peak - Neckline으로 계산해야 함.
            pattern_height = (min(peak_prices) - neckline) / neckline
            
            breakdown_confirmed = current_price < min(peak_prices) * (1 - PATTERN_BREAKOUT_CONFIRM)

            if breakdown_confirmed and pattern_height >= 0.035:
                pattern_type = "double_top"
                confidence = 1.0 - price_diff_pct * 10
                breakout_level = neckline

                # v2: 거래량 확인 — 두 번째 고점 거래량이 첫 번째보다 낮으면 신뢰도 증가
                df_recent = df.tail(lookback)
                if "Volume" in df_recent.columns:
                    vol = df_recent["Volume"].values.flatten()
                    p1_idx = int(last_two_peaks[0])
                    p2_idx = int(last_two_peaks[1])
                    if p1_idx < len(vol) and p2_idx < len(vol):
                        if float(vol[p2_idx]) < float(vol[p1_idx]):
                            confidence = min(1.0, confidence * 1.1)
                        else:
                            confidence *= 0.85

    # v2: 트렌드 보정 수정 — 기존 코드는 반대로 되어 있었음
    # 더블 바텀은 하락 추세에서 나타나야 유효 (반전 패턴)
    if pattern_type == "double_bottom":
        if trend == "downtrend":
            confidence = min(1.0, confidence * 1.15)  # 하락 추세 → 반전 신뢰
        elif trend == "uptrend":
            confidence *= 0.6  # 상승 추세에서 더블 바텀은 의심
    # 더블 탑은 상승 추세에서 나타나야 유효 (반전 패턴)
    if pattern_type == "double_top":
        if trend == "uptrend":
            confidence = min(1.0, confidence * 1.15)  # 상승 추세 → 반전 신뢰
        elif trend == "downtrend":
            confidence *= 0.6  # 하락 추세에서 더블 탑은 의심

    if pattern_type != "none" and confidence > 0.5:
        return PatternResult(True, pattern_type, confidence, breakout_level)

    return PatternResult(False, "none", 0.0)


def detect_head_and_shoulders(df: pd.DataFrame, lookback: int = 60) -> PatternResult:
    """헤드앤숄더 패턴을 감지한다.

    - 헤드앤숄더 (천장): 왼쪽 어깨 < 머리(중앙 고점) > 오른쪽 어깨 → 하락 신호
    - 역헤드앤숄더 (바닥): 왼쪽 어깨 > 머리(중앙 저점) < 오른쪽 어깨 → 상승 신호

    v2 개선:
    - 트렌드 보정 수정: H&S는 상승 추세 끝에서 유효 (기존 반대)
    - 어깨 시간 대칭 검증 (PATTERN_SHOULDER_TIME_RATIO)
    - 네크라인 기울기 제한
    - 머리에서 거래량 감소 확인
    """
    if len(df) < lookback:
        return PatternResult(False, "none", 0.0)

    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    closes = pd.Series(close_vals, index=range(len(close_vals)))
    current_price = float(close_vals[-1])

    # 진입 추세는 각 패턴의 좌어깨 직전 구간에서 측정한다 (아래 분기에서 설정).
    # 트레일링 추세(get_trend_context)를 쓰면 네크라인 돌파/붕괴가 추세를 오염시켜
    # H&S 붕괴 → 트레일링이 항상 하락추세 → 페널티로 사실상 감지 불가가 된다.
    entry_trend = "sideways"
    full_close = df["Close"].values.flatten()

    peaks, troughs = find_peaks_and_troughs(closes, order=7, prominence_pct=PEAK_PROMINENCE)

    pattern_type = "none"
    confidence = 0.0
    breakout_level = None

    min_time_ratio, max_time_ratio = PATTERN_SHOULDER_TIME_RATIO

    # 헤드앤숄더 (천장) 체크
    if len(peaks) >= 3:
        last_three = peaks[-3:]
        prices = closes.iloc[last_three].values

        left_shoulder, head, right_shoulder = prices

        if head > left_shoulder and head > right_shoulder:
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder

            # 머리가 어깨보다 충분히 높은지 (3% 이상)
            head_prominence = (head - max(left_shoulder, right_shoulder)) / head

            if shoulder_diff < 0.05 and head_prominence > 0.03:
                # v2: 시간 대칭 검증
                left_to_head = int(last_three[1]) - int(last_three[0])
                head_to_right = int(last_three[2]) - int(last_three[1])
                if left_to_head > 0 and head_to_right > 0:
                    time_ratio = left_to_head / head_to_right
                    if not (min_time_ratio <= time_ratio <= max_time_ratio):
                        # 시간 대칭이 안 맞으면 거부
                        return PatternResult(False, "none", 0.0)

                between_troughs = [t for t in troughs if last_three[0] < t < last_three[2]]
                if between_troughs:
                    neckline = float(closes.iloc[between_troughs].mean())

                    # v2: 네크라인 기울기 제한 (두 저점 간 기울기가 너무 가파르면 거부)
                    if len(between_troughs) >= 2:
                        neck_prices = closes.iloc[between_troughs].values
                        neck_slope = abs(neck_prices[-1] - neck_prices[0]) / neck_prices[0]
                        if neck_slope > 0.08:  # 8% 이상 기울기면 네크라인이 아님
                            return PatternResult(False, "none", 0.0)

                    # 돌파 확인: 현재 가격이 네크라인 아래
                    if current_price < neckline * (1 - PATTERN_BREAKOUT_CONFIRM):
                        pattern_type = "head_and_shoulders"
                        confidence = 1.0 - shoulder_diff * 5
                        breakout_level = neckline

                        # 진입 추세: 좌어깨 직전 구간 (상승추세 끝의 천장 패턴이어야 유효)
                        ls_abs = len(df) - len(closes) + int(last_three[0])
                        entry_trend = segment_trend(
                            full_close[max(0, ls_abs - PATTERN_TREND_LOOKBACK):ls_abs + 1]
                        )

                        # v2: 머리에서 거래량이 어깨보다 낮으면 신뢰도 증가
                        df_recent = df.tail(lookback)
                        if "Volume" in df_recent.columns:
                            vol = df_recent["Volume"].values.flatten()
                            head_idx = int(last_three[1])
                            left_idx = int(last_three[0])
                            if head_idx < len(vol) and left_idx < len(vol):
                                if float(vol[head_idx]) < float(vol[left_idx]):
                                    confidence = min(1.0, confidence * 1.1)

    # 역헤드앤숄더 (바닥) 체크
    if pattern_type == "none" and len(troughs) >= 3:
        last_three = troughs[-3:]
        prices = closes.iloc[last_three].values

        left_shoulder, head, right_shoulder = prices

        if head < left_shoulder and head < right_shoulder:
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder

            # 머리가 어깨보다 충분히 낮은지 (3% 이상)
            head_depth = (min(left_shoulder, right_shoulder) - head) / min(left_shoulder, right_shoulder)

            if shoulder_diff < 0.05 and head_depth > 0.03:
                # v2: 시간 대칭 검증
                left_to_head = int(last_three[1]) - int(last_three[0])
                head_to_right = int(last_three[2]) - int(last_three[1])
                if left_to_head > 0 and head_to_right > 0:
                    time_ratio = left_to_head / head_to_right
                    if not (min_time_ratio <= time_ratio <= max_time_ratio):
                        return PatternResult(False, "none", 0.0)

                between_peaks = [p for p in peaks if last_three[0] < p < last_three[2]]
                if between_peaks:
                    neckline = float(closes.iloc[between_peaks].mean())

                    # v2: 네크라인 기울기 제한
                    if len(between_peaks) >= 2:
                        neck_prices = closes.iloc[between_peaks].values
                        neck_slope = abs(neck_prices[-1] - neck_prices[0]) / neck_prices[0]
                        if neck_slope > 0.08:
                            return PatternResult(False, "none", 0.0)

                    # 돌파 확인: 현재 가격이 네크라인 위
                    if current_price > neckline * (1 + PATTERN_BREAKOUT_CONFIRM):
                        pattern_type = "inverse_head_and_shoulders"
                        confidence = 1.0 - shoulder_diff * 5
                        breakout_level = neckline

                        # 진입 추세: 좌어깨 직전 구간 (하락추세 끝의 바닥 패턴이어야 유효)
                        ls_abs = len(df) - len(closes) + int(last_three[0])
                        entry_trend = segment_trend(
                            full_close[max(0, ls_abs - PATTERN_TREND_LOOKBACK):ls_abs + 1]
                        )

                        # v2: 머리에서 거래량이 어깨보다 낮으면 신뢰도 증가
                        df_recent = df.tail(lookback)
                        if "Volume" in df_recent.columns:
                            vol = df_recent["Volume"].values.flatten()
                            head_idx = int(last_three[1])
                            left_idx = int(last_three[0])
                            if head_idx < len(vol) and left_idx < len(vol):
                                if float(vol[head_idx]) < float(vol[left_idx]):
                                    confidence = min(1.0, confidence * 1.1)

    # 진입 추세(좌어깨 직전) 기반 보정 — 측정 구간을 트레일링이 아닌 '진입'으로
    # 바꿔 정확도를 높였다. H&S는 상승 추세의 천장, 역H&S는 하락 추세의 바닥에서 유효.
    if pattern_type == "head_and_shoulders":
        if entry_trend == "uptrend":
            confidence = min(1.0, confidence * 1.15)  # 상승 끝 → 천장 패턴 유효
        elif entry_trend == "downtrend":
            confidence *= 0.5  # 하락 중 천장은 의심
    # 역H&S는 하락 추세의 바닥에서 나타나는 패턴 → 하락 추세에서 신뢰도 증가
    if pattern_type == "inverse_head_and_shoulders":
        if entry_trend == "downtrend":
            confidence = min(1.0, confidence * 1.15)  # 하락 끝 → 바닥 패턴 유효
        elif entry_trend == "uptrend":
            confidence *= 0.5  # 상승 중 바닥은 의심

    if pattern_type != "none" and confidence > 0.6:
        return PatternResult(True, pattern_type, confidence, breakout_level)

    return PatternResult(False, "none", 0.0)


def detect_cup_with_handle(df: pd.DataFrame, lookback: int = 90) -> PatternResult:
    """컵위드핸들 패턴을 감지한다.

    U자형 바닥 (컵) + 작은 되돌림 (핸들) → 상승 지속 신호

    v2 개선:
    - V자형 급락 방지: 바닥 구간 최소 기간 확인
    - 핸들 깊이를 컵 깊이의 1/3 이하로 동적 조정
    - 거래량 패턴 확인: 바닥에서 낮은 거래량, 오른쪽 상승에서 증가
    """
    if len(df) < lookback:
        return PatternResult(False, "none", 0.0)

    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    closes = pd.Series(close_vals, index=range(len(close_vals)))

    # 컵 형태 체크: 시작 고점 → 바닥 → 복귀
    first_third = closes.iloc[:lookback // 3]
    middle_third = closes.iloc[lookback // 3: 2 * lookback // 3]
    last_third = closes.iloc[2 * lookback // 3:]

    # 스칼라 값으로 변환
    left_high = float(first_third.max())
    cup_low = float(middle_third.min())
    right_high = float(last_third.max())
    current = float(close_vals[-1])

    # 컵 깊이 (10~35% 되돌림이 이상적)
    cup_depth = (left_high - cup_low) / left_high if left_high > 0 else 0

    # 오른쪽이 왼쪽 고점의 90% 이상까지 복귀했는지
    recovery_ratio = right_high / left_high if left_high > 0 else 0

    if 0.10 < cup_depth < 0.35 and recovery_ratio > 0.90:
        # v2: V자형 급락 방지 — 바닥 구간 최소 기간 확인
        # 컵 바닥 근처(±5%)에 머문 기간이 전체의 1/4 이상이어야 U자형
        bottom_zone = cup_low * 1.05
        bottom_bars = sum(1 for v in middle_third.values if float(v) <= bottom_zone)
        bottom_ratio = bottom_bars / len(middle_third) if len(middle_third) > 0 else 0

        if bottom_ratio < 0.25:
            # 바닥 구간이 짧으면 V자형으로 판단 → 거부
            return PatternResult(False, "none", 0.0)

        # 핸들 체크: 최근 하락 후 되돌림
        handle_lookback = min(15, len(last_third) // 2)
        if handle_lookback > 5:
            handle = closes.iloc[-handle_lookback:]
            handle_high = float(handle.max())
            handle_low = float(handle.min())
            handle_depth = (handle_high - handle_low) / handle_high if handle_high > 0 else 0

            # v2: 핸들 깊이를 컵 깊이의 1/3 이하로 동적 조정
            max_handle_depth = cup_depth / 3
            if 0.03 < handle_depth < max_handle_depth:
                confidence = min(0.9, recovery_ratio)
                breakout_level = max(left_high, right_high)

                # v2: 거래량 패턴 확인
                df_recent = df.tail(lookback)
                if "Volume" in df_recent.columns:
                    vol = df_recent["Volume"].values.flatten()
                    # 바닥 구간 거래량이 왼쪽 하락 구간보다 낮으면 신뢰도 증가
                    left_vol = float(np.mean(vol[:lookback // 3]))
                    mid_vol = float(np.mean(vol[lookback // 3: 2 * lookback // 3]))
                    right_vol = float(np.mean(vol[2 * lookback // 3:]))
                    if left_vol > 0 and mid_vol < left_vol and right_vol > mid_vol:
                        confidence = min(1.0, confidence * 1.1)
                    elif left_vol > 0:
                        confidence *= 0.85

                return PatternResult(True, "cup_with_handle", confidence, breakout_level)

    return PatternResult(False, "none", 0.0)


def detect_candlestick_patterns(df: pd.DataFrame) -> PatternResult:
    """주요 캔들스틱 패턴을 감지한다.

    - 도지 (Doji): 시가 ≈ 종가
    - 강세 잉걸핑 (Bullish Engulfing): 음봉 후 큰 양봉
    - 약세 잉걸핑 (Bearish Engulfing): 양봉 후 큰 음봉
    - 모닝스타 (Morning Star): 하락 후 반전 3봉 패턴
    - 이브닝스타 (Evening Star): 상승 후 반전 3봉 패턴

    v2 개선:
    - 추세 컨텍스트 추가: 강세 패턴은 하락 추세 끝, 약세 패턴은 상승 추세 끝
    - 모닝스타/이브닝스타 갭 조건 버그 수정 (미사용 변수 반영)
    - 거래량 확인 추가
    - 신뢰도 동적 조정
    """
    if len(df) < 3:
        return PatternResult(False, "none", 0.0)

    detected_patterns = []

    # v2: 추세 컨텍스트
    trend = get_trend_context(df, lookback=PATTERN_TREND_LOOKBACK)

    # 최근 3봉 데이터
    last_3 = df.tail(3)
    o = last_3["Open"].values.flatten()
    h = last_3["High"].values.flatten()
    l = last_3["Low"].values.flatten()
    c = last_3["Close"].values.flatten()

    # 최신 봉 기준
    latest_range = float(h[-1]) - float(l[-1])
    latest_body = abs(float(c[-1]) - float(o[-1]))

    # 1. 도지 (최신 봉) — 추세 무관하게 탐지
    if latest_range > 0 and latest_body / latest_range < 0.1:
        detected_patterns.append("doji")

    # 2. 강세 잉걸핑 (마지막 2봉) — v2: 하락 추세에서만 유효
    if len(df) >= 2:
        prev_bearish = float(c[-2]) < float(o[-2])
        curr_bullish = float(c[-1]) > float(o[-1])
        engulfs = float(o[-1]) < float(c[-2]) and float(c[-1]) > float(o[-2])
        if prev_bearish and curr_bullish and engulfs:
            if trend in ("downtrend", "sideways"):  # v2: 상승 추세에서는 무시
                detected_patterns.append("bullish_engulfing")

    # 3. 약세 잉걸핑 — v2: 상승 추세에서만 유효
    if len(df) >= 2:
        prev_bullish = float(c[-2]) > float(o[-2])
        curr_bearish = float(c[-1]) < float(o[-1])
        engulfs = float(o[-1]) > float(c[-2]) and float(c[-1]) < float(o[-2])
        if prev_bullish and curr_bearish and engulfs:
            if trend in ("uptrend", "sideways"):  # v2: 하락 추세에서는 무시
                detected_patterns.append("bearish_engulfing")

    # 4. 모닝스타 (3봉) — v2: 갭 조건 실제 반영
    if len(df) >= 3:
        first_bearish = float(c[-3]) < float(o[-3]) and abs(float(c[-3]) - float(o[-3])) > (float(h[-3]) - float(l[-3])) * 0.5
        middle_small = abs(float(c[-2]) - float(o[-2])) < (float(h[-2]) - float(l[-2])) * 0.3
        middle_gap_down = max(float(o[-2]), float(c[-2])) < float(c[-3])  # v2: 실제 조건에 사용
        third_bullish = float(c[-1]) > float(o[-1]) and float(c[-1]) > (float(o[-3]) + float(c[-3])) / 2

        # v2: middle_gap_down 조건 실제 반영 + 하락 추세에서만 유효
        if first_bearish and middle_small and middle_gap_down and third_bullish:
            if trend in ("downtrend", "sideways"):
                detected_patterns.append("morning_star")

    # 5. 이브닝스타 (3봉) — v2: 갭 조건 실제 반영
    if len(df) >= 3:
        first_bullish = float(c[-3]) > float(o[-3]) and abs(float(c[-3]) - float(o[-3])) > (float(h[-3]) - float(l[-3])) * 0.5
        middle_small = abs(float(c[-2]) - float(o[-2])) < (float(h[-2]) - float(l[-2])) * 0.3
        middle_gap_up = min(float(o[-2]), float(c[-2])) > float(c[-3])  # v2: 실제 조건에 사용
        third_bearish = float(c[-1]) < float(o[-1]) and float(c[-1]) < (float(o[-3]) + float(c[-3])) / 2

        # v2: middle_gap_up 조건 실제 반영 + 상승 추세에서만 유효
        if first_bullish and middle_small and middle_gap_up and third_bearish:
            if trend in ("uptrend", "sideways"):
                detected_patterns.append("evening_star")

    if detected_patterns:
        pattern_str = ",".join(detected_patterns)

        # v2: 신뢰도 동적 조정
        base_confidence = 0.7
        # 거래량 확인: 최신 봉의 거래량이 평균 이상이면 신뢰도 증가
        if check_volume_surge(df, len(df) - 1, threshold=1.2):
            base_confidence = 0.85

        return PatternResult(True, pattern_str, base_confidence)

    return PatternResult(False, "none", 0.0)


def detect_golden_cross(
    df: pd.DataFrame,
    short_period: int = 50,
    long_period: int = 200,
    recent_window: int = 5,
    slope_lookback: int = 10,
    ma_type: str = "sma",
    require_short_slope_confirm: bool = True,
    imminent_close_filter_ema: int = 0,
) -> PatternResult:
    """골든크로스 / 데드크로스 패턴을 감지한다.

    - 골든크로스: 단기 이동평균이 장기 이동평균을 상향 돌파 → 강력한 상승 신호
    - 데드크로스: 단기 이동평균이 장기 이동평균을 하향 돌파 → 강력한 하락 신호

    타임프레임별 MA 설정 (현재):
    - 일봉: EMA20 / EMA50 (TradingView SeanEMA 차트와 일치, 빠른 신호)
    - 주봉: SMA10 / SMA40 (동일한 캘린더 시간 범위)
    - 월봉: SMA3  / SMA10

    Parameters
    ----------
    ma_type : "sma" 또는 "ema". 기본값 "sma".
    require_short_slope_confirm : 임박 판정 시 "단기 MA가 slope_lookback 봉 전보다
        상승(데드는 하락) 중"인지 확인할지 여부. EMA는 반응이 빨라 lookback 기간
        동안 평균값 회복까지 시간차가 크므로, 일봉 EMA 호출은 False 권장.
        기본값 True는 SMA 시대 안전장치 유지.
    imminent_close_filter_ema : 0보다 크면 임박 판정에 "close > EMA(이 기간)"
        (데드는 "close < EMA") 추가 필터. 가격이 단기 EMA를 차지(=실질적 반등 시작)
        해야 임박으로 인정. 횡보·하락 종목 오탐 차단. 0이면 필터 끔(기본).

    추가 확인:
    - 최근 recent_window 봉 이내에 교차가 발생했는지 (실시간성)
    - 교차 시점 거래량 급증 여부
    - 두 이동평균의 기울기 방향
    """
    if len(df) < long_period:
        return PatternResult(False, "none", 0.0)

    close = df["Close"].values.flatten()
    close_series = pd.Series(close)

    if ma_type == "ema":
        ma_short = close_series.ewm(span=short_period, adjust=False).mean()
        ma_long = close_series.ewm(span=long_period, adjust=False).mean()
    else:
        ma_short = close_series.rolling(short_period).mean()
        ma_long = close_series.rolling(long_period).mean()

    # NaN 제거 후 최근 데이터 확인
    if ma_short.isna().iloc[-1] or ma_long.isna().iloc[-1]:
        return PatternResult(False, "none", 0.0)

    cross_type = "none"
    cross_day = -1

    for i in range(1, min(recent_window + 1, len(ma_short))):
        idx = len(ma_short) - i
        prev_idx = idx - 1
        if prev_idx < 0:
            break

        curr_s = float(ma_short.iloc[idx])
        curr_l = float(ma_long.iloc[idx])
        prev_s = float(ma_short.iloc[prev_idx])
        prev_l = float(ma_long.iloc[prev_idx])

        if pd.isna(curr_s) or pd.isna(curr_l) or pd.isna(prev_s) or pd.isna(prev_l):
            continue

        # 골든크로스: 전봉 short < long → 당봉 short >= long
        if prev_s < prev_l and curr_s >= curr_l:
            cross_type = "golden_cross"
            cross_day = i
            break

        # 데드크로스: 전봉 short > long → 당봉 short <= long
        if prev_s > prev_l and curr_s <= curr_l:
            cross_type = "death_cross"
            cross_day = i
            break

    # 교차가 아직 발생하지 않은 경우 → 임박 여부 체크
    if cross_type == "none":
        curr_s = float(ma_short.iloc[-1])
        curr_l = float(ma_long.iloc[-1])
        gap_pct = (curr_s - curr_l) / curr_l  # 양수면 short>long, 음수면 short<long

        # slope_lookback 봉 전 갭과 비교하여 수렴 중인지 확인
        prev_s = float(ma_short.iloc[-slope_lookback]) if len(ma_short) >= slope_lookback else float(ma_short.iloc[0])
        prev_l = float(ma_long.iloc[-slope_lookback]) if len(ma_long) >= slope_lookback else float(ma_long.iloc[0])
        prev_gap_pct = (prev_s - prev_l) / prev_l

        # 단기 이동평균 기울기
        short_slope = curr_s - prev_s

        # "임박"은 단기 MA가 장기 MA 아래(데드크로스 임박은 위)에 confirm_window봉 이상
        # 지속된 경우에만 인정한다. 이미 교차를 통과한 종목이 일시적 눌림목에서 잠깐
        # 단기 MA가 장기 MA 밑으로 내려온 것을 신규 교차 직전으로 오판하지 않기 위함.
        confirm_window = short_period
        recent_short = ma_short.iloc[-confirm_window:]
        recent_long = ma_long.iloc[-confirm_window:]
        valid = recent_short.notna() & recent_long.notna()
        sustained_below = bool(valid.any()) and bool((recent_short[valid] < recent_long[valid]).all())
        sustained_above = bool(valid.any()) and bool((recent_short[valid] > recent_long[valid]).all())

        # 임박 추가 필터: close > EMA(imminent_close_filter_ema) — 가격이 단기 EMA 차지
        # 횡보/하락 종목 오탐 차단 (예: 일봉 EMA10 필터). 0이면 필터 끔.
        filter_ema_last = None
        if imminent_close_filter_ema > 0:
            filter_ema_last = float(
                close_series.ewm(span=imminent_close_filter_ema, adjust=False).mean().iloc[-1]
            )
        close_last = float(close_series.iloc[-1])

        # 골든크로스 임박: short < long, 갭 5% 이내, 갭 좁혀지는 중
        #                  + 단기 MA가 장기 MA 아래에 충분히 오래 머무름(미교차 확정)
        #                  + (선택) 단기 MA 상승 중 — SMA 시대 안전장치, EMA 일봉에선 끔
        #                  + (선택) close > EMA(filter) — 실질적 반등 시작 확인
        if gap_pct < 0 and abs(gap_pct) < 0.05 and sustained_below:
            slope_ok = (short_slope > 0) if require_short_slope_confirm else True
            close_filter_ok = True if filter_ema_last is None else (close_last > filter_ema_last)
            if abs(gap_pct) < abs(prev_gap_pct) and slope_ok and close_filter_ok:
                cross_type = "golden_cross_imminent"
                confidence = 0.7 + (0.05 - abs(gap_pct)) / 0.05 * 0.25  # 0.7 ~ 0.95
                breakout_level = float(ma_long.iloc[-1])
                return PatternResult(True, cross_type, confidence, breakout_level)

        # 데드크로스 임박: short > long, 갭 5% 이내, 갭 좁혀지는 중
        #                  + 단기 MA가 장기 MA 위에 충분히 오래 머무름(미교차 확정)
        #                  + (선택) 단기 MA 하락 중 — SMA 시대 안전장치
        #                  + (선택) close < EMA(filter) — 실질적 약세 시작 확인
        elif gap_pct > 0 and gap_pct < 0.05 and sustained_above:
            slope_ok = (short_slope < 0) if require_short_slope_confirm else True
            close_filter_ok = True if filter_ema_last is None else (close_last < filter_ema_last)
            if gap_pct < prev_gap_pct and slope_ok and close_filter_ok:
                cross_type = "death_cross_imminent"
                confidence = 0.7 + (0.05 - gap_pct) / 0.05 * 0.25
                breakout_level = float(ma_long.iloc[-1])
                return PatternResult(True, cross_type, confidence, breakout_level)

        return PatternResult(False, "none", 0.0)

    # 신뢰도 계산
    confidence = 0.8

    # 최근일수록 신뢰도 높음
    if cross_day <= 3:
        confidence = 0.9
    elif cross_day <= 5:
        confidence = 0.85
    else:
        confidence = 0.7

    # 이동평균 기울기가 교차 방향과 일치하는지 확인
    slope_len = min(recent_window, len(ma_short))
    short_slope = float(ma_short.iloc[-1]) - float(ma_short.iloc[-slope_len]) if len(ma_short) >= slope_len else 0
    long_slope = float(ma_long.iloc[-1]) - float(ma_long.iloc[-slope_len]) if len(ma_long) >= slope_len else 0

    if cross_type == "golden_cross":
        if short_slope > 0 and long_slope >= 0:
            confidence = min(1.0, confidence * 1.1)
        elif short_slope <= 0:
            confidence *= 0.7
    elif cross_type == "death_cross":
        if short_slope < 0 and long_slope <= 0:
            confidence = min(1.0, confidence * 1.1)
        elif short_slope >= 0:
            confidence *= 0.7

    # 거래량 확인: 교차 시점 거래량 급증
    cross_idx = len(df) - cross_day
    if check_volume_surge(df, cross_idx, threshold=1.3):
        confidence = min(1.0, confidence * 1.1)

    # 돌파 레벨: 현재 장기 이동평균 가격
    breakout_level = float(ma_long.iloc[-1])

    return PatternResult(True, cross_type, confidence, breakout_level)


# =============================================================================
# 통합 함수
# =============================================================================

@dataclass
class AllPatterns:
    """모든 패턴 감지 결과를 담는 컨테이너."""
    triangle: str           # "none" / "symmetrical_triangle" / "ascending_triangle" / "descending_triangle"
    wedge: str              # "none" / "rising_wedge" / "falling_wedge"
    double: str             # "none" / "double_bottom" / "double_top"
    head_shoulders: str     # "none" / "head_and_shoulders" / "inverse_head_and_shoulders"
    cup_handle: bool        # 컵위드핸들 감지 여부
    candlestick: str        # 캔들스틱 패턴명 (쉼표 구분) 또는 "none"
    golden_cross: str       # "none" / "golden_cross" / "death_cross"

    # 신뢰도 및 돌파 레벨
    triangle_confidence: float
    wedge_confidence: float
    double_confidence: float
    head_shoulders_confidence: float
    cup_handle_confidence: float
    golden_cross_confidence: float

    triangle_breakout: Optional[float]
    double_breakout: Optional[float]
    head_shoulders_breakout: Optional[float]
    cup_handle_breakout: Optional[float]
    golden_cross_breakout: Optional[float]


def detect_all_patterns(df: pd.DataFrame, lookback: int = None) -> AllPatterns:
    """모든 차트 패턴을 감지하고 결과를 반환한다."""
    if lookback is None:
        lookback = PATTERN_LOOKBACK_DAYS

    # 각 패턴 감지
    triangle = detect_triangle(df, lookback)
    wedge = detect_wedge(df, lookback)
    double = detect_double_bottom_top(df, lookback)
    head_shoulders = detect_head_and_shoulders(df, lookback)
    cup_handle = detect_cup_with_handle(df, max(lookback, 90))
    candlestick = detect_candlestick_patterns(df)
    # 일봉: EMA 20/50 (TradingView SeanEMA 차트와 일치, 빠른 임박/교차 감지)
    # require_short_slope_confirm=False: EMA는 반응이 빨라 10일 slope 게이트가 과보수적.
    # imminent_close_filter_ema=10: close > EMA10 필터 — 가격이 단기 EMA를 차지(실질적
    #   반등 시작)해야 임박으로 인정. SCHW/ADBE/AEHL 같은 횡보·하락 종목 오탐 차단.
    golden_cross = detect_golden_cross(
        df, short_period=20, long_period=50, ma_type="ema",
        require_short_slope_confirm=False,
        imminent_close_filter_ema=10,
    )

    return AllPatterns(
        triangle=triangle.pattern_type,
        wedge=wedge.pattern_type,
        double=double.pattern_type,
        head_shoulders=head_shoulders.pattern_type,
        cup_handle=cup_handle.detected,
        candlestick=candlestick.pattern_type,
        golden_cross=golden_cross.pattern_type,

        triangle_confidence=triangle.confidence,
        wedge_confidence=wedge.confidence,
        double_confidence=double.confidence,
        head_shoulders_confidence=head_shoulders.confidence,
        cup_handle_confidence=cup_handle.confidence,
        golden_cross_confidence=golden_cross.confidence,

        triangle_breakout=triangle.breakout_level,
        double_breakout=double.breakout_level,
        head_shoulders_breakout=head_shoulders.breakout_level,
        cup_handle_breakout=cup_handle.breakout_level,
        golden_cross_breakout=golden_cross.breakout_level,
    )


def get_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """일봉 데이터를 주봉 데이터로 변환한다."""
    if df.empty:
        return df

    # 일봉 데이터를 주봉으로 리샘플링 (금요일 기준)
    logic = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }
    # 컬럼 존재 여부 확인 후 로직 적용
    agg_logic = {k: v for k, v in logic.items() if k in df.columns}
    
    weekly = df.resample("W-FRI").agg(agg_logic)
    weekly = weekly.dropna()
    
    return weekly


def get_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """일봉 데이터를 월봉 데이터로 변환한다."""
    if df.empty:
        return df

    # 일봉 데이터를 월봉으로 리샘플링 (월말 기준)
    logic = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }
    agg_logic = {k: v for k, v in logic.items() if k in df.columns}
    
    monthly = df.resample("ME").agg(agg_logic)
    monthly = monthly.dropna()
    
    return monthly


def detect_monthly_patterns(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """월봉 데이터를 생성하여 주요 패턴을 감지한다."""
    if len(df) < 120:  # 최소 데이터 요구량 (약 6개월)
        return []

    monthly = get_monthly_data(df)
    
    # 월봉 데이터가 너무 적으면 스킵 (최소 12개월)
    if len(monthly) < 12:
        return []

    patterns_found = []

    # 1. 골든크로스 / 골든크로스 임박 (월봉) - 초장기 대세 상승
    gc = detect_golden_cross(monthly, short_period=3, long_period=10, recent_window=3, slope_lookback=3)
    if gc.detected and gc.pattern_type in ("golden_cross", "golden_cross_imminent"):
        patterns_found.append((gc.pattern_type, gc.confidence))

    # 2. 더블 바텀 (월봉) - 역사적 바닥
    double = detect_double_bottom_top(monthly, lookback=24) # 2년치
    if double.detected and double.pattern_type == "double_bottom":
        patterns_found.append(("double_bottom", double.confidence))

    # 3. 역헤드앤숄더 (월봉) - 역사적 반전
    hs = detect_head_and_shoulders(monthly, lookback=24)
    if hs.detected and hs.pattern_type == "inverse_head_and_shoulders":
        patterns_found.append(("inverse_head_and_shoulders", hs.confidence))

    # 4. 컵앤핸들 (월봉) - 수년간의 매집
    cup = detect_cup_with_handle(monthly, lookback=36) # 3년치
    if cup.detected:
        patterns_found.append(("cup_with_handle", cup.confidence))

    # 5. 상승 삼각형 (월봉)
    tri = detect_triangle(monthly, lookback=24)
    if tri.detected and tri.pattern_type == "ascending_triangle":
        patterns_found.append(("ascending_triangle", tri.confidence))
    
    # 6. 하락 쐐기 (월봉)
    wdg = detect_wedge(monthly, lookback=24)
    if wdg.detected and wdg.pattern_type == "falling_wedge":
        patterns_found.append(("falling_wedge", wdg.confidence))
        
    # 7. 캔들스틱 (월봉) - 모닝스타, 강세 잉걸핑
    candle = detect_candlestick_patterns(monthly)
    if candle.detected:
        for p in candle.pattern_type.split(","):
            if p in ("bullish_engulfing", "morning_star"):
                patterns_found.append((p, candle.confidence))

    # 8. 월봉 정배열 (3단계) — EMA3/6/12 기반
    # 완전: Price > EMA3 > EMA6 > EMA12
    # 눌림목: EMA3 > EMA6 > EMA12, EMA3 >= Price > EMA6 (단기 눌림 허용)
    # 형성중: EMA3 > EMA6, EMA6 <= EMA12, Price > EMA3 (초기 신호)
    if len(monthly) >= 12:
        close_m = monthly["Close"]
        ema3 = close_m.ewm(span=3, adjust=False).mean().iloc[-1]
        ema6 = close_m.ewm(span=6, adjust=False).mean().iloc[-1]
        ema12 = close_m.ewm(span=12, adjust=False).mean().iloc[-1]
        price_m = close_m.iloc[-1]
        if price_m > ema3 > ema6 > ema12:
            patterns_found.append(("monthly_uptrend_full", 1.0))
        elif ema3 > ema6 > ema12 and ema3 >= price_m > ema6:
            patterns_found.append(("monthly_uptrend_pullback", 1.0))
        elif ema3 > ema6 and ema6 <= ema12 and price_m > ema3:
            patterns_found.append(("monthly_uptrend_forming", 1.0))

    # 정배열(완전/눌림목)이면 이미 골든크로스를 통과한 상태 → 골든크로스 계열 태그 제거.
    # (정배열형성중은 교차 형성 단계라 제외하지 않는다.)
    pattern_names = {name for name, _ in patterns_found}
    if "monthly_uptrend_full" in pattern_names or "monthly_uptrend_pullback" in pattern_names:
        patterns_found = [
            (name, conf) for name, conf in patterns_found
            if name not in ("golden_cross", "golden_cross_imminent")
        ]

    return patterns_found


def detect_weekly_patterns(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """주봉 데이터를 생성하 주요 패턴을 감지한다.
    
    Returns:
        [(패턴이름, 신뢰도), ...] 리스트
    """
    if len(df) < 60:  # 최소 데이터 요구량 (일봉 기준)
        return []

    weekly = get_weekly_data(df)
    
    # 주봉 데이터가 너무 적으면 스킵 (최소 20주 = 약 5개월)
    if len(weekly) < 20:
        return []

    patterns_found = []

    # 1. 골든크로스 / 골든크로스 임박 (주봉) - 대세 상승
    gc = detect_golden_cross(weekly, short_period=10, long_period=40, recent_window=5, slope_lookback=5)
    if gc.detected and gc.pattern_type in ("golden_cross", "golden_cross_imminent"):
        patterns_found.append((gc.pattern_type, gc.confidence))

    # 2. 더블 바텀 (주봉) - 장기 바닥
    double = detect_double_bottom_top(weekly, lookback=52) # 1년치
    if double.detected and double.pattern_type == "double_bottom":
        patterns_found.append(("double_bottom", double.confidence))

    # 3. 역헤드앤숄더 (주봉) - 장기 반전
    hs = detect_head_and_shoulders(weekly, lookback=52)
    if hs.detected and hs.pattern_type == "inverse_head_and_shoulders":
        patterns_found.append(("inverse_head_and_shoulders", hs.confidence))

    # 4. 컵앤핸들 (주봉) - 지속형 상승
    cup = detect_cup_with_handle(weekly, lookback=52)
    if cup.detected:
        patterns_found.append(("cup_with_handle", cup.confidence))

    # 5. 상승 삼각형 (주봉)
    tri = detect_triangle(weekly, lookback=40)
    if tri.detected and tri.pattern_type == "ascending_triangle":
        patterns_found.append(("ascending_triangle", tri.confidence))
    
    # 6. 하락 쐐기 (주봉) - 반전
    wdg = detect_wedge(weekly, lookback=40)
    if wdg.detected and wdg.pattern_type == "falling_wedge":
        patterns_found.append(("falling_wedge", wdg.confidence))
        
    # 7. 캔들스틱 (주봉) - 모닝스타, 강세 잉걸핑
    # 캔들스틱은 최신 봉 기준이므로 lookback 불필요
    candle = detect_candlestick_patterns(weekly)
    if candle.detected:
        # 쉼표로 구분된 문자열일 수 있음
        for p in candle.pattern_type.split(","):
            if p in ("bullish_engulfing", "morning_star"):
                patterns_found.append((p, candle.confidence))

    # 8. 주봉 정배열 (3단계) — EMA10/20/50 기반
    # 완전: Price > EMA10 > EMA20 > EMA50
    # 눌림목: EMA10 > EMA20 > EMA50, EMA10 >= Price > EMA20 (단기 눌림 허용)
    # 형성중: EMA10 > EMA20, EMA20 <= EMA50, Price > EMA10 (초기 신호)
    if len(weekly) >= 50:
        close_w = weekly["Close"]
        ema10 = close_w.ewm(span=10, adjust=False).mean().iloc[-1]
        ema20 = close_w.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close_w.ewm(span=50, adjust=False).mean().iloc[-1]
        price_w = close_w.iloc[-1]
        if price_w > ema10 > ema20 > ema50:
            patterns_found.append(("weekly_uptrend_full", 1.0))
        elif ema10 > ema20 > ema50 and ema10 >= price_w > ema20:
            patterns_found.append(("weekly_uptrend_pullback", 1.0))
        elif ema10 > ema20 and ema20 <= ema50 and price_w > ema10:
            patterns_found.append(("weekly_uptrend_forming", 1.0))

    # 정배열(완전/눌림목)이면 이미 골든크로스를 통과한 상태 → 골든크로스 계열 태그 제거.
    # (정배열형성중은 교차 형성 단계라 제외하지 않는다.)
    pattern_names = {name for name, _ in patterns_found}
    if "weekly_uptrend_full" in pattern_names or "weekly_uptrend_pullback" in pattern_names:
        patterns_found = [
            (name, conf) for name, conf in patterns_found
            if name not in ("golden_cross", "golden_cross_imminent")
        ]

    return patterns_found
