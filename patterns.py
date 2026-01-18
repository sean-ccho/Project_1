"""차트 패턴 인식 모듈.

기술적 분석에서 사용하는 주요 차트 패턴을 자동으로 감지한다:
- 삼각형 수렴 (Triangle)
- 쐐기 패턴 (Wedge)
- 더블 바텀/탑 (Double Bottom/Top)
- 헤드앤숄더 (Head and Shoulders)
- 컵위드핸들 (Cup with Handle)
- 캔들스틱 패턴 (Candlestick Patterns)
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
    order: int = 5,
    prominence_pct: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """가격 시계열에서 고점과 저점 인덱스를 찾는다.
    
    Args:
        prices: 종가 시계열
        order: 양쪽으로 비교할 포인트 수
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
    
    # EMA 기반 추가 판단 (사용 가능한 경우)
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
    
    # 상승 추세: 5% 이상 상승 + EMA 상승
    if change_pct > 0.05 and ema_slope > 0:
        return "uptrend"
    # 하락 추세: 5% 이상 하락 + EMA 하락
    elif change_pct < -0.05 and ema_slope < 0:
        return "downtrend"
    
    return "sideways"

def detect_triangle(df: pd.DataFrame, lookback: int = 60) -> PatternResult:
    """삼각형 수렴 패턴을 감지한다.
    
    - 대칭 삼각형: 고점 하락 + 저점 상승
    - 상승 삼각형: 고점 수평 + 저점 상승
    - 하락 삼각형: 고점 하락 + 저점 수평
    """
    if len(df) < lookback:
        return PatternResult(False, "none", 0.0)
    
    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    closes = pd.Series(close_vals, index=range(len(close_vals)))
    
    peaks, troughs = find_peaks_and_troughs(closes, order=5, prominence_pct=PEAK_PROMINENCE)
    
    if len(peaks) < PATTERN_MIN_TOUCHES or len(troughs) < PATTERN_MIN_TOUCHES:
        return PatternResult(False, "none", 0.0)
    
    # 추세선 피팅
    peak_slope, peak_intercept, peak_r2 = fit_trendline(peaks, closes.iloc[peaks].values)
    trough_slope, trough_intercept, trough_r2 = fit_trendline(troughs, closes.iloc[troughs].values)
    
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
        # 예상 돌파 가격 (두 추세선의 교점 근처)
        last_idx = len(closes) - 1
        upper_level = peak_intercept + peak_slope * last_idx
        lower_level = trough_intercept + trough_slope * last_idx
        breakout_level = (upper_level + lower_level) / 2
        
        return PatternResult(True, pattern_type, confidence, breakout_level)
    
    return PatternResult(False, "none", 0.0)


def detect_wedge(df: pd.DataFrame, lookback: int = 60) -> PatternResult:
    """쐐기 패턴을 감지한다.
    
    - 상승 쐐기 (Rising Wedge): 두 선 모두 상승, 하지만 수렴 → 하락 신호
    - 하락 쐐기 (Falling Wedge): 두 선 모두 하락, 하지만 수렴 → 상승 신호
    """
    if len(df) < lookback:
        return PatternResult(False, "none", 0.0)
    
    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    closes = pd.Series(close_vals, index=range(len(close_vals)))
    
    peaks, troughs = find_peaks_and_troughs(closes, order=5, prominence_pct=PEAK_PROMINENCE)
    
    # 최소 3개의 터치 포인트 필요 (wedge는 더 엄격하게)
    if len(peaks) < 3 or len(troughs) < 3:
        return PatternResult(False, "none", 0.0)
    
    peak_slope, _, peak_r2 = fit_trendline(peaks, closes.iloc[peaks].values)
    trough_slope, _, trough_r2 = fit_trendline(troughs, closes.iloc[troughs].values)
    
    pattern_type = "none"
    confidence = 0.0
    
    # R² 값이 모두 충분히 높아야 함 (명확한 추세선)
    min_r2 = min(peak_r2, trough_r2)
    if min_r2 < PATTERN_MIN_R2:
        return PatternResult(False, "none", 0.0)
    
    # 수렴 판정 강화: 기울기 차이 비율 계산
    max_slope = max(abs(peak_slope), abs(trough_slope), 0.001)
    slope_diff_ratio = abs(peak_slope - trough_slope) / max_slope
    
    # 수렴 조건: 기울기 차이가 30% 이하
    is_converging = slope_diff_ratio < 0.3
    
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
        return PatternResult(True, pattern_type, confidence)
    
    return PatternResult(False, "none", 0.0)


def detect_double_bottom_top(df: pd.DataFrame, lookback: int = 60) -> PatternResult:
    """더블 바텀/탑 패턴을 감지한다.
    
    - 더블 바텀: 비슷한 가격의 두 저점 (W 모양) → 상승 반전
    - 더블 탑: 비슷한 가격의 두 고점 (M 모양) → 하락 반전
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
    
    # 더블 바텀 체크 (우선순위: 하락/횡보 추세에서 더 유효)
    if len(troughs) >= 2:
        # 마지막 두 저점
        last_two_troughs = troughs[-2:]
        trough_prices = closes.iloc[last_two_troughs].values
        
        # 두 저점이 비슷한 가격인지 (3% 이내)
        price_diff_pct = abs(trough_prices[0] - trough_prices[1]) / trough_prices[0]
        
        # 두 저점 사이에 고점이 있는지
        between_peaks = [p for p in peaks if last_two_troughs[0] < p < last_two_troughs[1]]
        
        if price_diff_pct < 0.03 and len(between_peaks) >= 1:
            neckline = float(closes.iloc[between_peaks[0]])
            
            # 돌파 확인: 현재 가격이 두 저점보다 높아야 함
            breakout_confirmed = current_price > max(trough_prices) * (1 + PATTERN_BREAKOUT_CONFIRM)
            
            if breakout_confirmed:
                pattern_type = "double_bottom"
                confidence = 1.0 - price_diff_pct * 10
                breakout_level = neckline
                
                # 상승 추세에서 double_bottom 신뢰도 강화
                if trend == "uptrend":
                    confidence = min(1.0, confidence * 1.1)
    
    # 더블 탑 체크 (우선순위: 상승/횡보 추세에서 더 유효)
    if pattern_type == "none" and len(peaks) >= 2:
        last_two_peaks = peaks[-2:]
        peak_prices = closes.iloc[last_two_peaks].values
        
        price_diff_pct = abs(peak_prices[0] - peak_prices[1]) / peak_prices[0]
        
        between_troughs = [t for t in troughs if last_two_peaks[0] < t < last_two_peaks[1]]
        
        if price_diff_pct < 0.03 and len(between_troughs) >= 1:
            neckline = float(closes.iloc[between_troughs[0]])
            
            # 돌파 확인: 현재 가격이 두 고점보다 낮아야 함
            breakdown_confirmed = current_price < min(peak_prices) * (1 - PATTERN_BREAKOUT_CONFIRM)
            
            if breakdown_confirmed:
                pattern_type = "double_top"
                confidence = 1.0 - price_diff_pct * 10
                breakout_level = neckline
                
                # 하락 추세에서 double_top 신뢰도 강화
                if trend == "downtrend":
                    confidence = min(1.0, confidence * 1.1)
    
    # 트렌드에 반하는 패턴은 신뢰도 감소
    if pattern_type == "double_top" and trend == "uptrend":
        confidence *= 0.5  # 상승 추세에서 천장 패턴은 의심
    if pattern_type == "double_bottom" and trend == "downtrend":
        confidence *= 0.7  # 하락 추세에서 바닥 패턴도 신중히
    
    if pattern_type != "none" and confidence > 0.5:
        return PatternResult(True, pattern_type, confidence, breakout_level)
    
    return PatternResult(False, "none", 0.0)


def detect_head_and_shoulders(df: pd.DataFrame, lookback: int = 60) -> PatternResult:
    """헤드앤숄더 패턴을 감지한다.
    
    - 헤드앤숄더 (천장): 왼쪽 어깨 < 머리(중앙 고점) > 오른쪽 어깨 → 하락 신호
    - 역헤드앤숄더 (바닥): 왼쪽 어깨 > 머리(중앙 저점) < 오른쪽 어깨 → 상승 신호
    """
    if len(df) < lookback:
        return PatternResult(False, "none", 0.0)
    
    recent = df.tail(lookback)
    close_vals = recent["Close"].values.flatten()
    closes = pd.Series(close_vals, index=range(len(close_vals)))
    current_price = float(close_vals[-1])
    
    # 트렌드 컨텍스트 확인
    trend = get_trend_context(df, lookback=PATTERN_TREND_LOOKBACK)
    
    peaks, troughs = find_peaks_and_troughs(closes, order=5, prominence_pct=PEAK_PROMINENCE)
    
    pattern_type = "none"
    confidence = 0.0
    breakout_level = None
    
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
                between_troughs = [t for t in troughs if last_three[0] < t < last_three[2]]
                if between_troughs:
                    neckline = float(closes.iloc[between_troughs].mean())
                    
                    # 돌파 확인: 현재 가격이 네크라인 아래
                    if current_price < neckline * (1 - PATTERN_BREAKOUT_CONFIRM):
                        pattern_type = "head_and_shoulders"
                        confidence = 1.0 - shoulder_diff * 5
                        breakout_level = neckline
    
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
                between_peaks = [p for p in peaks if last_three[0] < p < last_three[2]]
                if between_peaks:
                    neckline = float(closes.iloc[between_peaks].mean())
                    
                    # 돌파 확인: 현재 가격이 네크라인 위
                    if current_price > neckline * (1 + PATTERN_BREAKOUT_CONFIRM):
                        pattern_type = "inverse_head_and_shoulders"
                        confidence = 1.0 - shoulder_diff * 5
                        breakout_level = neckline
    
    # 트렌드에 따른 신뢰도 조정
    if pattern_type == "head_and_shoulders" and trend == "uptrend":
        confidence *= 0.6  # 상승 추세에서 천장 패턴은 신뢰도 감소
    if pattern_type == "inverse_head_and_shoulders" and trend == "uptrend":
        confidence = min(1.0, confidence * 1.1)  # 상승 추세에서 바닥 패턴 완성은 신뢰도 증가
    
    if pattern_type != "none" and confidence > 0.6:
        return PatternResult(True, pattern_type, confidence, breakout_level)
    
    return PatternResult(False, "none", 0.0)


def detect_cup_with_handle(df: pd.DataFrame, lookback: int = 90) -> PatternResult:
    """컵위드핸들 패턴을 감지한다.
    
    U자형 바닥 (컵) + 작은 되돌림 (핸들) → 상승 지속 신호
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
        # 핸들 체크: 최근 하락 후 되돌림 (5~15%)
        handle_lookback = min(15, len(last_third) // 2)
        if handle_lookback > 5:
            handle = closes.iloc[-handle_lookback:]
            handle_high = float(handle.max())
            handle_low = float(handle.min())
            handle_depth = (handle_high - handle_low) / handle_high if handle_high > 0 else 0
            
            if 0.05 < handle_depth < 0.15:
                confidence = min(0.9, recovery_ratio)
                breakout_level = max(left_high, right_high)
                return PatternResult(True, "cup_with_handle", confidence, breakout_level)
    
    return PatternResult(False, "none", 0.0)


def detect_candlestick_patterns(df: pd.DataFrame) -> PatternResult:
    """주요 캔들스틱 패턴을 감지한다.
    
    - 도지 (Doji): 시가 ≈ 종가
    - 강세 잉걸핑 (Bullish Engulfing): 음봉 후 큰 양봉
    - 약세 잉걸핑 (Bearish Engulfing): 양봉 후 큰 음봉
    - 모닝스타 (Morning Star): 하락 후 반전 3봉 패턴
    - 이브닝스타 (Evening Star): 상승 후 반전 3봉 패턴
    """
    if len(df) < 3:
        return PatternResult(False, "none", 0.0)
    
    detected_patterns = []
    
    # 최근 3봉 데이터
    last_3 = df.tail(3)
    o = last_3["Open"].values
    h = last_3["High"].values
    l = last_3["Low"].values
    c = last_3["Close"].values
    
    # 최신 봉 기준
    latest_range = h[-1] - l[-1]
    latest_body = abs(c[-1] - o[-1])
    
    # 1. 도지 (최신 봉)
    if latest_range > 0 and latest_body / latest_range < 0.1:
        detected_patterns.append("doji")
    
    # 2. 강세 잉걸핑 (마지막 2봉)
    if len(df) >= 2:
        prev_bearish = c[-2] < o[-2]
        curr_bullish = c[-1] > o[-1]
        engulfs = o[-1] < c[-2] and c[-1] > o[-2]
        if prev_bearish and curr_bullish and engulfs:
            detected_patterns.append("bullish_engulfing")
    
    # 3. 약세 잉걸핑
    if len(df) >= 2:
        prev_bullish = c[-2] > o[-2]
        curr_bearish = c[-1] < o[-1]
        engulfs = o[-1] > c[-2] and c[-1] < o[-2]
        if prev_bullish and curr_bearish and engulfs:
            detected_patterns.append("bearish_engulfing")
    
    # 4. 모닝스타 (3봉)
    if len(df) >= 3:
        first_bearish = c[-3] < o[-3] and abs(c[-3] - o[-3]) > (h[-3] - l[-3]) * 0.5
        middle_small = abs(c[-2] - o[-2]) < (h[-2] - l[-2]) * 0.3
        middle_gap_down = max(o[-2], c[-2]) < c[-3]
        third_bullish = c[-1] > o[-1] and c[-1] > (o[-3] + c[-3]) / 2
        
        if first_bearish and middle_small and third_bullish:
            detected_patterns.append("morning_star")
    
    # 5. 이브닝스타 (3봉)
    if len(df) >= 3:
        first_bullish = c[-3] > o[-3] and abs(c[-3] - o[-3]) > (h[-3] - l[-3]) * 0.5
        middle_small = abs(c[-2] - o[-2]) < (h[-2] - l[-2]) * 0.3
        middle_gap_up = min(o[-2], c[-2]) > c[-3]
        third_bearish = c[-1] < o[-1] and c[-1] < (o[-3] + c[-3]) / 2
        
        if first_bullish and middle_small and third_bearish:
            detected_patterns.append("evening_star")
    
    if detected_patterns:
        pattern_str = ",".join(detected_patterns)
        return PatternResult(True, pattern_str, 0.8)
    
    return PatternResult(False, "none", 0.0)


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
    
    # 신뢰도 및 돌파 레벨
    triangle_confidence: float
    wedge_confidence: float
    double_confidence: float
    head_shoulders_confidence: float
    cup_handle_confidence: float
    
    triangle_breakout: Optional[float]
    double_breakout: Optional[float]
    head_shoulders_breakout: Optional[float]
    cup_handle_breakout: Optional[float]


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
    
    return AllPatterns(
        triangle=triangle.pattern_type,
        wedge=wedge.pattern_type,
        double=double.pattern_type,
        head_shoulders=head_shoulders.pattern_type,
        cup_handle=cup_handle.detected,
        candlestick=candlestick.pattern_type,
        
        triangle_confidence=triangle.confidence,
        wedge_confidence=wedge.confidence,
        double_confidence=double.confidence,
        head_shoulders_confidence=head_shoulders.confidence,
        cup_handle_confidence=cup_handle.confidence,
        
        triangle_breakout=triangle.breakout_level,
        double_breakout=double.breakout_level,
        head_shoulders_breakout=head_shoulders.breakout_level,
        cup_handle_breakout=cup_handle.breakout_level,
    )
