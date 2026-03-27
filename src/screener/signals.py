"""Institutional-grade signal evaluation and ranking system.

Cross-sectional percentile ranking, risk-adjusted scoring (Sharpe proxy),
factor-driven alpha integration, earnings risk adjustment, and
correlation-based diversification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.config import (
    ADX_BUY_MIN,
    ADX_SELL_MAX,
    ALPHA_FACTOR_STRONG_THRESHOLD,
    ALPHA_FACTOR_WEAK_THRESHOLD,
    BOTTOM_REVERSAL_THRESHOLD,
    BOTTOM_STRATEGY_WEIGHTS,
    JUDGEMENT_DISPLAY,
    MOMENTUM_STRATEGY_WEIGHTS,
    RECOMMENDATION_DISPLAY,
    RSI_BUY_MAX,
    RSI_SELL_MIN,
    SIGNAL_PRIORITY,
    VOLUME_BREAKOUT_MULTIPLIER,
    WEEKLY_PATTERN_WEIGHTS,
    MONTHLY_PATTERN_WEIGHTS,
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
    OUTPUT_TOP_N,
    OUTPUT_MAX_PER_SECTOR,
    OUTPUT_MIN_SCORE,
    FUND_HEALTH_ROE_MIN,
    FUND_HEALTH_PROFIT_MARGIN_MIN,
    FUND_HEALTH_EARNINGS_GROWTH_MIN,
    FUND_HEALTH_DEBT_TO_EQUITY_MAX,
    CROSS_SECTIONAL_ENABLED,
    CROSS_SECTIONAL_MIN_PERCENTILE,
    RISK_ADJUSTED_ENABLED,
    SHARPE_PROXY_WEIGHT,
    ALPHA_RANKING_WEIGHT,
    STRATEGY_SCORE_WEIGHT,
    EARNINGS_PENALTY_DAYS,
    EARNINGS_PENALTY_MULTIPLIER,
    CORRELATION_FILTER_ENABLED,
    CORRELATION_MAX_THRESHOLD,
    CORRELATION_LOOKBACK,
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


# =============================================================================
# Institutional-Grade Ranking Components
# =============================================================================

def _compute_sharpe_proxy(df: pd.DataFrame) -> pd.Series:
    """Risk-adjusted score: 기대수익 / 변동성 (Sharpe Ratio proxy).

    같은 수익률이면 변동성이 낮은 종목이 우선. 헤지펀드의 핵심 원칙.
    """
    ret_20d = _column(df, "20일수익률").fillna(0)
    atr_pct = _column(df, "ATR%").fillna(0.03)  # ATR% as vol proxy

    # Avoid division by zero; floor volatility at 0.5%
    vol = atr_pct.clip(lower=0.005)
    sharpe = ret_20d / vol

    # Normalize to 0-1 range using percentile rank
    if len(sharpe.dropna()) > 1:
        return sharpe.rank(pct=True)
    return pd.Series(0.5, index=df.index)


def _compute_alpha_percentile(df: pd.DataFrame) -> pd.Series:
    """5-팩터 알파 모델의 크로스섹션 퍼센타일 랭크.

    절대 점수 대신 "이 종목이 전체 유니버스에서 상위 몇 %인지"로 판단.
    """
    alpha = _column(df, "알파점수").fillna(0)
    if len(alpha.dropna()) > 1:
        return alpha.rank(pct=True)
    return pd.Series(0.5, index=df.index)


def _apply_earnings_penalty(df: pd.DataFrame, score: pd.Series) -> pd.Series:
    """실적 발표 N일 전 종목은 점수를 할인한다.

    어닝 직전 진입 = 고변동성 도박. 헤지펀드는 어닝 전 포지션을 줄인다.
    """
    days_col = _column(df, "days_to_next_earnings")
    if days_col.isna().all():
        return score

    penalty_mask = (days_col >= 0) & (days_col <= EARNINGS_PENALTY_DAYS)
    result = score.copy()
    result.loc[penalty_mask] = result.loc[penalty_mask] * EARNINGS_PENALTY_MULTIPLIER
    return result


def _correlation_diversify(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """상관관계 기반 다변화: Top N 종목이 같은 트레이드가 되지 않도록 필터.

    높은 상관관계 종목 중 점수가 낮은 쪽을 제거한다.
    수익률 기반 상관관계를 사용하여 실질적 포지션 겹침을 방지.
    """
    if not CORRELATION_FILTER_ENABLED:
        return df

    if len(df) <= 1:
        return df

    # 수익률 컬럼으로 상관관계 추정 (가용한 데이터로 근사)
    corr_features = ["5일수익률", "20일수익률", "RSI", "52주포지션", "bollinger_pband"]
    available = [c for c in corr_features if c in df.columns]
    if len(available) < 2:
        return df

    # 표준화 후 코사인 유사도를 상관관계 proxy로 사용
    feature_matrix = df[available].fillna(0)
    # Z-score normalization
    std = feature_matrix.std()
    std = std.replace(0, 1)
    normed = (feature_matrix - feature_matrix.mean()) / std

    # Pairwise correlation matrix
    corr_matrix = normed.T.corr()

    # Greedy selection: 점수 높은 순으로 선택, 이미 선택된 종목과 상관 높으면 스킵
    selected_indices = []
    sorted_idx = df[score_col].sort_values(ascending=False).index.tolist()

    for idx in sorted_idx:
        if idx not in corr_matrix.index:
            continue
        is_redundant = False
        for sel_idx in selected_indices:
            if sel_idx in corr_matrix.columns and idx in corr_matrix.index:
                corr_val = corr_matrix.loc[idx, sel_idx]
                if pd.notna(corr_val) and abs(corr_val) > CORRELATION_MAX_THRESHOLD:
                    is_redundant = True
                    break
        if not is_redundant:
            selected_indices.append(idx)

    return df.loc[selected_indices]


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

    # --- 바닥 반등 적합도 계산 ---
    W_B = BOTTOM_STRATEGY_WEIGHTS

    def _calc_bottom_reversal_score(row):
        """바닥 반등 전략 점수: 과매도·반등 신호에 집중한다.

        헤지펀드 수준: Conviction Score 기반 — 여러 팩터가 동시에 같은 방향을
        가리킬 때만 높은 점수를 부여한다.
        """
        score = 0.0
        conviction_signals = 0  # 확신 팩터 카운트

        # 저점확률 (AI 모델)
        low_prob = row.get("저점확률", 0)
        if pd.notna(low_prob) and low_prob >= BACKTEST_LOW_PROB_THRESHOLD:
            # 연속 가중: 확률이 높을수록 더 많은 점수
            score += W_B["저점확률"] * min(low_prob / 0.4, 1.5)
            conviction_signals += 1

        # 반등스코어 (기술적 반등 지표)
        reversal = row.get("반등스코어", 0)
        if pd.notna(reversal) and reversal >= BACKTEST_REVERSAL_SCORE_MIN:
            # 연속 가중: 반등스코어가 높을수록 비례 증가
            score += W_B["반등스코어"] * min(reversal / 5.0, 2.0)
            conviction_signals += 1

        # RSI 구간별 가점/감점
        rsi = row.get("RSI", 50)
        if pd.notna(rsi):
            if rsi < 30:
                score += W_B["RSI_과매도"]
                conviction_signals += 1
            elif rsi < 40:
                score += W_B["RSI_탈출"]
            elif rsi <= 45:
                score += W_B["RSI_중립하단"]
            elif rsi > 80:
                score += W_B["RSI_극과매수감점"]
            elif rsi > 75:
                score += W_B["RSI_과매수감점"]

        # 상승형 차트 패턴
        bullish_patterns = [
            row.get("패턴_더블", "") == "더블바텀",
            row.get("패턴_헤드숄더", "") == "역헤드앤숄더",
            row.get("패턴_컵핸들", False),
            row.get("패턴_삼각형", "") == "상승삼각형",
            row.get("패턴_쐐기", "") == "하락쐐기",
            str(row.get("패턴_캔들", "")) in ["강세잉걸핑", "모닝스타"],
        ]
        if any(bullish_patterns):
            score += W_B["상승패턴"]
            conviction_signals += 1

        # 급락 후 반등 기회 (5일 수익률 < -8%)
        ret_5d = row.get("5일수익률", 0)
        if pd.notna(ret_5d) and ret_5d < -0.08:
            score += W_B["급락반등"]

        # 강한 섹터
        if row.get("in_strong_sector", True):
            score += W_B["강한섹터"]

        # 감점: 급등 (5일 수익률 > 20%)
        if pd.notna(ret_5d) and ret_5d > 0.20:
            score += W_B["급등감점"]

        # 감점: 볼린저 상단 돌파
        bollinger_pband = row.get("bollinger_pband", 0.5)
        if pd.notna(bollinger_pband) and bollinger_pband > 0.95:
            score += W_B["볼린저과열감점"]

        # 알파 모델 팩터: Mean-Reversion (과매도 반등 신호)
        mean_rev = row.get("팩터_평균회귀", 0)
        if pd.notna(mean_rev) and mean_rev > ALPHA_FACTOR_STRONG_THRESHOLD:
            score += W_B["팩터_평균회귀_강"]
            conviction_signals += 1
        elif pd.notna(mean_rev) and mean_rev > ALPHA_FACTOR_WEAK_THRESHOLD:
            score += W_B["팩터_평균회귀_약"]

        # 알파 모델 팩터: Volatility (변동성 압축 → 반등 에너지 축적)
        vol_score = row.get("팩터_변동성", 0)
        if pd.notna(vol_score) and vol_score > ALPHA_FACTOR_STRONG_THRESHOLD:
            score += W_B["팩터_변동성"]
            conviction_signals += 1

        # 펀더멘탈 건강 체크 (Phase 3)
        fund_roe = row.get("fund_roe", np.nan)
        fund_margin = row.get("fund_profit_margin", np.nan)
        fund_growth = row.get("fund_earnings_growth", np.nan)
        fund_debt = row.get("fund_debt_to_equity", np.nan)
        fund_checks = 0
        if pd.notna(fund_roe) and fund_roe >= FUND_HEALTH_ROE_MIN:
            fund_checks += 1
        if pd.notna(fund_margin) and fund_margin >= FUND_HEALTH_PROFIT_MARGIN_MIN:
            fund_checks += 1
        if pd.notna(fund_growth) and fund_growth >= FUND_HEALTH_EARNINGS_GROWTH_MIN:
            fund_checks += 1
        if pd.notna(fund_debt) and fund_debt <= FUND_HEALTH_DEBT_TO_EQUITY_MAX:
            fund_checks += 1
        if fund_checks >= 3:
            score += W_B["펀더멘탈_건강"]
            conviction_signals += 1

        # 기관 보유 비율
        inst_hold = row.get("fund_institutional_holders_pct", np.nan)
        if pd.notna(inst_hold) and inst_hold >= 0.50:
            score += W_B["기관수급"]

        # 주봉 패턴 가산점 (Weekly Patterns)
        weekly_pat_str = str(row.get("주봉패턴", ""))
        if weekly_pat_str and weekly_pat_str.lower() != "nan":
            for pat in weekly_pat_str.split(", "):
                if pat in WEEKLY_PATTERN_WEIGHTS:
                    score += WEEKLY_PATTERN_WEIGHTS[pat]

        # 월봉 패턴 가산점 (Monthly Patterns)
        monthly_pat_str = str(row.get("월봉패턴", ""))
        if monthly_pat_str and monthly_pat_str.lower() != "nan":
            for pat in monthly_pat_str.split(", "):
                if pat in MONTHLY_PATTERN_WEIGHTS:
                    score += MONTHLY_PATTERN_WEIGHTS[pat]

        # Conviction Bonus: 3개 이상 독립 팩터가 동시에 같은 방향
        if conviction_signals >= 3:
            score += W_B["conviction_bonus"]

        return score

    # --- 모멘텀 추격 적합도 계산 ---
    W_M = MOMENTUM_STRATEGY_WEIGHTS

    def _calc_momentum_score(row):
        """모멘텀 추격 전략 점수: 추세·돌파 강도에 집중한다.

        헤지펀드 수준: Conviction Score 기반 — 추세 + 수급 + 강도가
        동시에 확인될 때만 높은 점수를 부여한다.
        """
        score = 0.0
        conviction_signals = 0

        # 매수 신호 (EMA/MACD 정배열 + 보조지표)
        if row.get("buy_signal", False):
            score += W_M["매수신호"]
            conviction_signals += 1

        # 트렌드 점수 (연속 가중: 강할수록 더 많은 점수)
        trend = row.get("트렌드점수_최종", row.get("트렌드점수", 0))
        if pd.notna(trend) and trend > 0.1:
            # 연속 스케일: trend 0.1에서 1.0까지 비례
            trend_mult = min(trend / 0.3, 2.0)
            score += W_M["트렌드점수"] * trend_mult
            conviction_signals += 1

        # EMA 정배열 (EMA20 > EMA50)
        ema20 = row.get("ema20", 0)
        ema50 = row.get("ema50", 0)
        if pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50:
            score += W_M["EMA정배열"]

        # 거래량 돌파 (연속 가중: 배수가 클수록 더 많은 점수)
        volume = row.get("volume", 0)
        volume_ma = row.get("volume_ma20", 0)
        if pd.notna(volume) and pd.notna(volume_ma) and volume_ma > 0:
            vol_ratio = volume / volume_ma
            if vol_ratio > 1.8:
                # 연속 스케일: 1.8x에서 3.0x까지 비례
                vol_mult = min(vol_ratio / 1.8, 2.0)
                score += W_M["거래량돌파"] * vol_mult
                conviction_signals += 1

        # ADX 강세 (추세 강도 > 25, 연속 가중)
        adx = row.get("adx", 0)
        if pd.notna(adx) and adx > 25:
            adx_mult = min(adx / 30.0, 1.5)
            score += W_M["ADX강세"] * adx_mult
            conviction_signals += 1

        # 강한 섹터
        if row.get("in_strong_sector", True):
            score += W_M["강한섹터"]

        # 상승 패턴
        bullish_patterns = [
            row.get("패턴_더블", "") == "더블바텀",
            row.get("패턴_헤드숄더", "") == "역헤드앤숄더",
            row.get("패턴_컵핸들", False),
            row.get("패턴_삼각형", "") == "상승삼각형",
            row.get("패턴_쐐기", "") == "하락쐐기",
            str(row.get("패턴_캔들", "")) in ["강세잉걸핑", "모닝스타"],
        ]
        if any(bullish_patterns):
            score += W_M["상승패턴"]
            conviction_signals += 1

        # 감점: RSI 과매수 (> 75)
        rsi = row.get("RSI", 50)
        if pd.notna(rsi) and rsi > 75:
            score += W_M["RSI_과매수감점"]

        # 감점: 급등 (5일 수익률 > 20%)
        ret_5d = row.get("5일수익률", 0)
        if pd.notna(ret_5d) and ret_5d > 0.20:
            score += W_M["급등감점"]

        # 감점: 볼린저 상단 돌파
        bollinger_pband = row.get("bollinger_pband", 0.5)
        if pd.notna(bollinger_pband) and bollinger_pband > 0.95:
            score += W_M["볼린저과열감점"]

        # 알파 모델 팩터: Momentum (다중 기간 모멘텀)
        mom_factor = row.get("팩터_모멘텀", 0)
        if pd.notna(mom_factor) and mom_factor > ALPHA_FACTOR_STRONG_THRESHOLD:
            score += W_M["팩터_모멘텀_강"]
            conviction_signals += 1
        elif pd.notna(mom_factor) and mom_factor > ALPHA_FACTOR_WEAK_THRESHOLD:
            score += W_M["팩터_모멘텀_약"]

        # 알파 모델 팩터: Trend (강한 추세 지속)
        trend_factor = row.get("팩터_추세", 0)
        if pd.notna(trend_factor) and trend_factor > ALPHA_FACTOR_STRONG_THRESHOLD:
            score += W_M["팩터_추세"]
            conviction_signals += 1

        # 펀더멘탈 건강 체크
        fund_roe = row.get("fund_roe", np.nan)
        fund_margin = row.get("fund_profit_margin", np.nan)
        fund_growth = row.get("fund_earnings_growth", np.nan)
        fund_debt = row.get("fund_debt_to_equity", np.nan)
        fund_checks = 0
        if pd.notna(fund_roe) and fund_roe >= FUND_HEALTH_ROE_MIN:
            fund_checks += 1
        if pd.notna(fund_margin) and fund_margin >= FUND_HEALTH_PROFIT_MARGIN_MIN:
            fund_checks += 1
        if pd.notna(fund_growth) and fund_growth >= FUND_HEALTH_EARNINGS_GROWTH_MIN:
            fund_checks += 1
        if pd.notna(fund_debt) and fund_debt <= FUND_HEALTH_DEBT_TO_EQUITY_MAX:
            fund_checks += 1
        if fund_checks >= 3:
            score += W_M["펀더멘탈_건강"]
            conviction_signals += 1

        # 기관 보유 비율
        inst_hold = row.get("fund_institutional_holders_pct", np.nan)
        if pd.notna(inst_hold) and inst_hold >= 0.50:
            score += W_M["기관수급"]

        # 주봉 패턴 가산점 (Weekly Patterns)
        weekly_pat_str = str(row.get("주봉패턴", ""))
        if weekly_pat_str and weekly_pat_str.lower() != "nan":
            for pat in weekly_pat_str.split(", "):
                if pat in WEEKLY_PATTERN_WEIGHTS:
                    score += WEEKLY_PATTERN_WEIGHTS[pat]

        # 월봉 패턴 가산점 (Monthly Patterns)
        monthly_pat_str = str(row.get("월봉패턴", ""))
        if monthly_pat_str and monthly_pat_str.lower() != "nan":
            for pat in monthly_pat_str.split(", "):
                if pat in MONTHLY_PATTERN_WEIGHTS:
                    score += MONTHLY_PATTERN_WEIGHTS[pat]

        # Conviction Bonus: 3개 이상 독립 팩터 동시 확인
        if conviction_signals >= 3:
            score += W_M["conviction_bonus"]

        return score

    out["바닥반등_적합도"] = out.apply(_calc_bottom_reversal_score, axis=1)
    out["모멘텀_적합도"] = out.apply(_calc_momentum_score, axis=1)

    # =================================================================
    # Institutional-Grade Composite Ranking
    # =================================================================
    # 기존 전략 점수를 퍼센타일로 변환 (절대 → 상대)
    strategy_raw = out[["바닥반등_적합도", "모멘텀_적합도"]].max(axis=1)

    if CROSS_SECTIONAL_ENABLED and len(out) > 1:
        strategy_pctl = strategy_raw.rank(pct=True)
    else:
        # Normalize to 0-1 (max possible ~15)
        strategy_pctl = (strategy_raw / strategy_raw.max()).clip(0, 1) if strategy_raw.max() > 0 else strategy_raw * 0

    # 1. Sharpe Proxy (수익/변동성 비율)
    if RISK_ADJUSTED_ENABLED:
        sharpe_pctl = _compute_sharpe_proxy(out)
    else:
        sharpe_pctl = pd.Series(0.5, index=out.index)

    # 2. Alpha Model Percentile (5-팩터 크로스섹션 순위)
    alpha_pctl = _compute_alpha_percentile(out)

    # 3. Composite Score (가중 합산)
    # ALPHA(35%) + STRATEGY(35%) + SHARPE(30%) = 100%
    composite = (
        ALPHA_RANKING_WEIGHT * alpha_pctl
        + STRATEGY_SCORE_WEIGHT * strategy_pctl
        + SHARPE_PROXY_WEIGHT * sharpe_pctl
    )

    # 4. Earnings Risk Penalty
    composite = _apply_earnings_penalty(out, composite)

    out["composite_score"] = composite

    # Composite를 10점 만점으로 스케일 (표시용)
    if len(out) > 1:
        c_min, c_max = composite.min(), composite.max()
        if c_max > c_min:
            scaled = 10.0 * (composite - c_min) / (c_max - c_min)
        else:
            scaled = composite * 0 + 5.0
    else:
        scaled = composite * 10.0
    out["매수적합도"] = scaled

    # ★ 별점 변환 (두 전략 공용)
    def _score_to_stars(score):
        if score >= 8.0:
            return f"★★★★★ ({score:.1f})"
        elif score >= 6.5:
            return f"★★★★☆ ({score:.1f})"
        elif score >= 5.0:
            return f"★★★☆☆ ({score:.1f})"
        elif score >= 3.5:
            return f"★★☆☆☆ ({score:.1f})"
        elif score >= 2.0:
            return f"★☆☆☆☆ ({score:.1f})"
        else:
            return f"☆☆☆☆☆ ({score:.1f})"

    out["바닥반등_적합도_표시"] = out["바닥반등_적합도"].apply(
        lambda s: _score_to_stars(s)
    )
    out["모멘텀_적합도_표시"] = out["모멘텀_적합도"].apply(
        lambda s: _score_to_stars(s)
    )
    out["매수적합도_표시"] = out["매수적합도"].apply(_score_to_stars)

    # Ranking 보조 컬럼 (Google Sheets 표시용)
    out["Sharpe_순위"] = sharpe_pctl.rank(ascending=False).astype(int) if len(out) > 0 else 0
    out["Alpha_순위"] = alpha_pctl.rank(ascending=False).astype(int) if len(out) > 0 else 0
    # -----------------------------------

    # 정렬: Composite Score 내림차순 (이것이 최종 랭킹)
    sorted_out = out.sort_values("composite_score", ascending=False)

    # --- Phase 4: 출력 제한 (Top N, 섹터 집중 방지, 상관관계) ---

    # 1. 크로스섹션 퍼센타일 컷오프
    if CROSS_SECTIONAL_ENABLED and len(sorted_out) > 1:
        pctl_threshold = CROSS_SECTIONAL_MIN_PERCENTILE / 100.0
        pctl_ranks = sorted_out["composite_score"].rank(pct=True)
        sorted_out = sorted_out[pctl_ranks >= pctl_threshold]

    # 2. 최소 원점수 컷오프 (보험)
    if not sorted_out.empty:
        raw_strategy = sorted_out[["바닥반등_적합도", "모멘텀_적합도"]].max(axis=1)
        sorted_out = sorted_out[raw_strategy >= OUTPUT_MIN_SCORE]

    # 3. 상관관계 기반 다변화 (같은 트레이드 방지)
    if not sorted_out.empty:
        sorted_out = _correlation_diversify(sorted_out, "composite_score")

    if not sorted_out.empty:
        # 4. 섹터별 최대 종목 수 제한
        if "섹터" in sorted_out.columns:
            sector_counts: dict[str, int] = {}
            keep_mask = []
            for _, row in sorted_out.iterrows():
                sector = row.get("섹터", "Unknown")
                count = sector_counts.get(sector, 0)
                if count < OUTPUT_MAX_PER_SECTOR:
                    keep_mask.append(True)
                    sector_counts[sector] = count + 1
                else:
                    keep_mask.append(False)
            sorted_out = sorted_out[keep_mask]

        # 5. Top N 제한
        sorted_out = sorted_out.head(OUTPUT_TOP_N)

    # Drop internal working columns
    cols_to_drop = [judgement.name, recommendation.name, "composite_score"]
    if "in_strong_sector" in sorted_out.columns:
        cols_to_drop.append("in_strong_sector")

    return sorted_out.drop(columns=cols_to_drop)
