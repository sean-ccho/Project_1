"""Simplified signal evaluation using a restricted indicator set."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ADX_BUY_MIN,
    ADX_SELL_MAX,
    BOTTOM_REVERSAL_THRESHOLD,
    JUDGEMENT_DISPLAY,
    RECOMMENDATION_DISPLAY,
    RSI_BUY_MAX,
    RSI_SELL_MIN,
    SIGNAL_PRIORITY,
    VOLUME_BREAKOUT_MULTIPLIER,
    MARKET_FILTER_ENABLED,
    STRATEGY_MODE,
    SECTOR_ROTATION_ENABLED,
    BACKTEST_ENTRY_SCORE_MIN,
    BACKTEST_LOW_PROB_THRESHOLD,
    BACKTEST_REVERSAL_SCORE_MIN,
    BACKTEST_BUY_SIGNAL_WEIGHT,
    BACKTEST_LOW_PROB_WEIGHT,
    BACKTEST_REVERSAL_WEIGHT,
    BACKTEST_PATTERN_WEIGHT,
    BACKTEST_SECTOR_WEIGHT,
    BACKTEST_TREND_SCORE_WEIGHT,
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
    # ----------------------------

    # --- Sector Rotation Filter ---
    # If enabled, disable buy signals for stocks NOT in strong sectors.
    if SECTOR_ROTATION_ENABLED and "섹터" in out.columns:
        # 강한 섹터 판별을 위해 출력 데이터에 표시 (섹터 강도는 main.py에서 계산)
        # 여기서는 "in_strong_sector" 컴럼이 있으면 그것을 사용
        if "in_strong_sector" in out.columns:
            weak_sector_mask = ~out["in_strong_sector"].fillna(True).astype(bool)
            buy_signal = buy_signal & ~weak_sector_mask
    # ------------------------------

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

    # --- 저점 반등 스코어 판단 ---
    reversal_score = _column(out, "반등스코어")
    if not reversal_score.isna().all():
        reversal_flag = reversal_score >= BOTTOM_REVERSAL_THRESHOLD
        # 반등 스코어가 높으면 '저점 반등'으로 우선 판단
        judgement.loc[reversal_flag] = "저점 반등"
        recommendation.loc[reversal_flag] = "반등 매수 고려"

    out[judgement.name] = judgement
    out["판단"] = judgement.map(lambda value: JUDGEMENT_DISPLAY.get(value, value))
    out[recommendation.name] = recommendation
    out["추천"] = recommendation.map(
        lambda value: RECOMMENDATION_DISPLAY.get(value, value)
    )

    out["우선순위"] = (
        judgement.map(SIGNAL_PRIORITY).fillna(99).astype(int)
    )

    # --- 매수적합도 (Entry Score) 계산 ---
    # 백테스트에서 사용한 Entry Score를 Signals 시트에서 바로 확인할 수 있도록 추가
    def _calc_entry_score(row):
        """
        진입 스코어를 계산한다. (백테스트 로직과 동일)
        
        구성 요소:
        - buy_signal: 기본 매수 신호 (가중치 2.0)
        - 저점확률: ML 모델 기반 저점 예측 (가중치 1.5)
        - 반등스코어: 기술적 반등 지표 (가중치 1.5)
        - 상승 패턴: 차트 패턴 (가중치 1.0)
        - 강한 섹터: 섹터 로테이션 (가중치 0.5)
        - 트렌드점수: 높은 모멘텀 (가중치 1.0) - 대체 조건
        - RSI 기반 가점/감점
        - 급등/급락 감점/가점
        """
        score = 0.0
        
        # buy_signal (시장 필터에 의해 False일 수 있음)
        if row.get("buy_signal", False):
            score += BACKTEST_BUY_SIGNAL_WEIGHT
        
        # 저점확률 (백테스트와 동기화)
        low_prob = row.get("저점확률", 0)
        if pd.notna(low_prob) and low_prob >= BACKTEST_LOW_PROB_THRESHOLD:
            score += BACKTEST_LOW_PROB_WEIGHT
        
        # 반등스코어 (백테스트와 동기화)
        reversal = row.get("반등스코어", 0)
        if pd.notna(reversal) and reversal >= BACKTEST_REVERSAL_SCORE_MIN:
            score += BACKTEST_REVERSAL_WEIGHT
        
        # 상승 패턴 (더블바텀, 역헤드숄더, 컵핸들, 상승삼각형, 하락쐐기)
        bullish_patterns = [
            row.get("패턴_더블", "") == "더블바텀",
            row.get("패턴_헤드숄더", "") == "역헤드앤숄더",
            row.get("패턴_컵핸들", False),
            row.get("패턴_삼각형", "") == "상승삼각형",
            row.get("패턴_쐐기", "") == "하락쐐기",
            str(row.get("패턴_캔들", "")) in ["강세잉걸핑", "모닝스타"],
        ]
        if any(bullish_patterns):
            score += BACKTEST_PATTERN_WEIGHT
        
        # 강한 섹터
        if row.get("in_strong_sector", True):
            score += BACKTEST_SECTOR_WEIGHT
        
        # 트렌드점수 (buy_signal이 비활성화된 경우 대체 조건)
        trend = row.get("트렌드점수_최종", row.get("트렌드점수", 0))
        if pd.notna(trend) and trend > 0.1:  # 상위 모멘텀
            score += BACKTEST_TREND_SCORE_WEIGHT
        
        # RSI 기반 점수 (최적화된 버전)
        rsi = row.get("RSI", 50)
        if pd.notna(rsi):
            if rsi < 30:
                score += 1.5    # 과매도 - 반등 기회
            elif rsi < 40:
                score += 1.0    # 과매도 탈출 구간
            elif rsi <= 45:
                score += 0.5    # 중립 하단
            elif rsi > 80:
                score -= 2.0    # 극심한 과매수 - 위험
            elif rsi > 70:
                score -= 1.0    # 과매수 - 주의 (완화)
        
        # EMA 정배열 확인 (추가 가점)
        ema20 = row.get("ema20", 0)
        ema50 = row.get("ema50", 0)
        if pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50:
            score += 0.5
        
        # 급등 감점 (5일 수익률 > 20%) - 기준 완화
        ret_5d = row.get("5일수익률", 0)
        if pd.notna(ret_5d) and ret_5d > 0.20:
            score -= 1.0    # 급등 후 조정 위험
        
        # 볼린저밴드 상단 돌파 감점
        bollinger_pband = row.get("bollinger_pband", 0.5)
        if pd.notna(bollinger_pband) and bollinger_pband > 0.95:
            score -= 1.0    # 상단 밴드 돌파 - 과열
        
        # 급락 가점 (5일 수익률 < -8%) - 기준 완화
        if pd.notna(ret_5d) and ret_5d < -0.08:
            score += 0.5    # 급락 후 반등 기회
        
        return score
    
    out["매수적합도"] = out.apply(_calc_entry_score, axis=1)
    
    # 매수적합도 텍스트 (★ 표시로 직관적으로)
    def _score_to_stars(score):
        if score >= 5.0:
            return f"★★★★★ ({score:.1f})"
        elif score >= 4.0:
            return f"★★★★☆ ({score:.1f})"
        elif score >= 3.5:
            return f"★★★☆☆ ({score:.1f})"
        elif score >= 3.0:
            return f"★★☆☆☆ ({score:.1f})"
        elif score >= 2.0:
            return f"★☆☆☆☆ ({score:.1f})"
        else:
            return f"☆☆☆☆☆ ({score:.1f})"
    
    out["매수적합도_표시"] = out["매수적합도"].apply(_score_to_stars)
    # -----------------------------------


    sorted_out = out.sort_values(
        ["우선순위", "트렌드점수_최종"], ascending=[True, False]
    )
    # Drop internal working columns
    cols_to_drop = [judgement.name, recommendation.name]
    if "in_strong_sector" in sorted_out.columns:
        cols_to_drop.append("in_strong_sector")
    
    return sorted_out.drop(columns=cols_to_drop)
