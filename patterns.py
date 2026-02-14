"""차트 패턴 인식 모듈.

기술적 분석에서 사용하는 주요 차트 패턴을 자동으로 감지한다:
- 삼각형 수렴 (Triangle)
- 쐐기 패턴 (Wedge)
- 더블 바텀/탑 (Double Bottom/Top)
- 헤드앤숄더 (Head and Shoulders)
- 컵위드핸들 (Cup with Handle)
- 캔들스틱 패턴 (Candlestick Patterns)

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

from config import (
    PATTERN_LOOKBACK_DAYS,
    PATTERN_MIN_TOUCHES,
    PATTERN_CONVERGENCE_THRESHOLD,
    PEAK_PROMINENCE,
    PATTERN_TREND_LOOKBACK,
    PATTERN_MIN_R2,
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

    # R² 기준 확인 (각 추세선이 의미있는 적합도를 가져야 함)
    if min(peak_r2, trough_r2) < PATTERN_MIN_R2:
        return PatternResult(False, "none", 0.0)

    # 기울기 정규화 (일일 수익률 기준)
    avg_price = closes.mean()
    peak_slope_pct = peak_slope / avg_price if avg_price > 0 else 0
    trough_slope_pct = trough_slope / avg_price if avg_price > 0 else 0

    # 수렴 판정
    slope_threshold = PATTERN_CONVERGENCE_THRESHOLD / lookback

    pattern_type = "none"
    confidence = 0.0

    # 대칭 삼각형: 고점↘ 저점↗
    if peak_slope_pct < -slope_threshold and trough_slope_pct > slope_threshold:
        pattern_type = "symmetrical_triangle"
        confidence = min(peak_r2, trough_r2)

    # 상승 삼각형: 고점 수평, 저점↗
    elif abs(peak_slope_pct) < slope_threshold and trough_slope_pct > slope_threshold:
        pattern_type = "ascending_triangle"
        confidence = trough_r2

    # 하락 삼각형: 고점↘, 저점 수평
    elif peak_slope_pct < -slope_threshold and abs(trough_slope_pct) < slope_threshold:
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

    # 상승 쐐기: 둘 다 상승하지만 수렴 (저점 기울기 > 고점 기울기)
    if peak_slope > 0 and trough_slope > 0 and trough_slope > peak_slope:
        pattern_type = "rising_wedge"
        confidence = min_r2

    # 하락 쐐기: 둘 다 하락하지만 수렴 (고점 기울기 > 저점 기울기)
    elif peak_slope < 0 and trough_slope < 0 and peak_slope > trough_slope:
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
            breakout_confirmed = current_price > max(trough_prices) * (1 + PATTERN_BREAKOUT_CONFIRM)

            if breakout_confirmed:
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
            breakdown_confirmed = current_price < min(peak_prices) * (1 - PATTERN_BREAKOUT_CONFIRM)

            if breakdown_confirmed:
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

    # 트렌드 컨텍스트 확인
    trend = get_trend_context(df, lookback=PATTERN_TREND_LOOKBACK)

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

                        # v2: 머리에서 거래량이 어깨보다 낮으면 신뢰도 증가
                        df_recent = df.tail(lookback)
                        if "Volume" in df_recent.columns:
                            vol = df_recent["Volume"].values.flatten()
                            head_idx = int(last_three[1])
                            left_idx = int(last_three[0])
                            if head_idx < len(vol) and left_idx < len(vol):
                                if float(vol[head_idx]) < float(vol[left_idx]):
                                    confidence = min(1.0, confidence * 1.1)

    # v2: 트렌드 보정 수정 — 기존 코드는 반대로 되어 있었음
    # H&S는 상승 추세의 천장에서 나타나는 패턴 → 상승 추세에서 신뢰도 증가
    if pattern_type == "head_and_shoulders":
        if trend == "uptrend":
            confidence = min(1.0, confidence * 1.15)  # 상승 → 천장 패턴 유효
        elif trend == "downtrend":
            confidence *= 0.5  # 이미 하락 중이면 의심
    # 역H&S는 하락 추세의 바닥에서 나타나는 패턴 → 하락 추세에서 신뢰도 증가
    if pattern_type == "inverse_head_and_shoulders":
        if trend == "downtrend":
            confidence = min(1.0, confidence * 1.15)  # 하락 → 바닥 패턴 유효
        elif trend == "uptrend":
            confidence *= 0.5  # 이미 상승 중이면 의심

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


def detect_golden_cross(df: pd.DataFrame) -> PatternResult:
    """골든크로스 / 데드크로스 패턴을 감지한다.

    - 골든크로스: 50일 이동평균이 200일 이동평균을 상향 돌파 → 강력한 상승 신호
    - 데드크로스: 50일 이동평균이 200일 이동평균을 하향 돌파 → 강력한 하락 신호

    추가 확인:
    - 최근 5거래일 이내에 교차가 발생했는지 (실시간성)
    - 교차 시점 거래량 급증 여부
    - 두 이동평균의 기울기 방향
    """
    if len(df) < 200:
        return PatternResult(False, "none", 0.0)

    close = df["Close"].values.flatten()
    close_series = pd.Series(close)

    # 50일, 200일 단순 이동평균
    sma50 = close_series.rolling(50).mean()
    sma200 = close_series.rolling(200).mean()

    # NaN 제거 후 최근 데이터 확인
    if sma50.isna().iloc[-1] or sma200.isna().iloc[-1]:
        return PatternResult(False, "none", 0.0)

    # 최근 10거래일에서 교차 발생 여부 확인
    recent_window = 10
    cross_type = "none"
    cross_day = -1

    for i in range(1, min(recent_window + 1, len(sma50))):
        idx = len(sma50) - i
        prev_idx = idx - 1
        if prev_idx < 0:
            break

        curr_50 = float(sma50.iloc[idx])
        curr_200 = float(sma200.iloc[idx])
        prev_50 = float(sma50.iloc[prev_idx])
        prev_200 = float(sma200.iloc[prev_idx])

        if pd.isna(curr_50) or pd.isna(curr_200) or pd.isna(prev_50) or pd.isna(prev_200):
            continue

        # 골든크로스: 전일 50 < 200 → 당일 50 >= 200
        if prev_50 < prev_200 and curr_50 >= curr_200:
            cross_type = "golden_cross"
            cross_day = i  # 며칠 전 발생
            break

        # 데드크로스: 전일 50 > 200 → 당일 50 <= 200
        if prev_50 > prev_200 and curr_50 <= curr_200:
            cross_type = "death_cross"
            cross_day = i
            break

    if cross_type == "none":
        return PatternResult(False, "none", 0.0)

    # 신뢰도 계산
    confidence = 0.8

    # 최근일수록 신뢰도 높음 (5일 이내 → 높음, 10일 → 낮음)
    if cross_day <= 3:
        confidence = 0.9
    elif cross_day <= 5:
        confidence = 0.85
    else:
        confidence = 0.7

    # 50일선과 200일선의 기울기가 교차 방향과 일치하는지 확인
    sma50_slope = float(sma50.iloc[-1]) - float(sma50.iloc[-5]) if len(sma50) >= 5 else 0
    sma200_slope = float(sma200.iloc[-1]) - float(sma200.iloc[-5]) if len(sma200) >= 5 else 0

    if cross_type == "golden_cross":
        # 50일선이 상승하고 200일선도 횡보/상승이면 강한 신호
        if sma50_slope > 0 and sma200_slope >= 0:
            confidence = min(1.0, confidence * 1.1)
        elif sma50_slope <= 0:
            confidence *= 0.7  # 50일선이 하락 중이면 약한 골든크로스
    elif cross_type == "death_cross":
        if sma50_slope < 0 and sma200_slope <= 0:
            confidence = min(1.0, confidence * 1.1)
        elif sma50_slope >= 0:
            confidence *= 0.7

    # 거래량 확인: 교차 시점 거래량 급증
    cross_idx = len(df) - cross_day
    if check_volume_surge(df, cross_idx, threshold=1.3):
        confidence = min(1.0, confidence * 1.1)

    # 돌파 레벨: 현재 200일선 가격
    breakout_level = float(sma200.iloc[-1])

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
    golden_cross = detect_golden_cross(df)

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
