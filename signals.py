"""시그널 판정 및 정렬 헬퍼."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    ADX_TREND_THRESHOLD,
    ADX_WEAK_THRESHOLD,
    BOLLINGER_BREAKOUT_PBAND,
    BOLLINGER_OVERBOUGHT_PBAND,
    BOLLINGER_OVERSOLD_PBAND,
    BOTTOM_POS_THRESHOLD,
    BOTTOM_TREND_SCORE,
    BOTTOM_VOLUME_Z,
    EARNINGS_EVENT_WINDOW_DAYS,
    EMA200_DISTANCE_MIN,
    BUY_POS_THRESHOLD,
    BUY_RSI_MAX,
    BUY_RSI_MIN,
    BUY_SCORE_THRESHOLD,
    BUY_NEGATIVE_MAX,
    BUY_POSITIVE_MIN,
    CMF_BUY_THRESHOLD,
    FUND_HEALTH_CURRENT_RATIO_MIN,
    FUND_HEALTH_DEBT_TO_EQUITY_MAX,
    FUND_HEALTH_EARNINGS_GROWTH_MIN,
    FUND_HEALTH_PROFIT_MARGIN_MIN,
    FUND_HEALTH_REVENUE_GROWTH_MIN,
    FUND_HEALTH_ROE_MIN,
    GAP_DOWN_EXTREME,
    INTRADAY_RECOVERY_MIN,
    LONG_TERM_SLOPE_MIN,
    MACD_BOTTOM_THRESHOLD,
    MACD_BUY_HIST_THRESHOLD,
    MACD_SELL_HIST_THRESHOLD,
    OBV_Z_BUY_THRESHOLD,
    OVERBOUGHT_RSI,
    REL_STRENGTH_MARKET_BUFFER,
    REL_STRENGTH_SECTOR_BUFFER,
    RSI_OVERSOLD,
    SECTOR_REL_STRENGTH_MIN,
    SIGNAL_PRIORITY,
    STOCH_MIN_BUY,
    STOCH_OVERSOLD,
    STOCH_OVERBOUGHT,
    VOLUME_STABILITY_RATIO_MIN,
    WATCH_POS_THRESHOLD,
    WATCH_SCORE_THRESHOLD,
    WATCH_NEGATIVE_MAX,
    WATCH_POSITIVE_MIN,
    BOTTOM_NEGATIVE_TOLERANCE,
    BUY_REL_STRENGTH_MIN,
    JUDGEMENT_DISPLAY,
    RECOMMENDATION_DISPLAY,
)


@dataclass
class BottomContext:
    score: float
    grade: str
    signals: list[str]
    cautions: list[str]
    volume_support: bool
    momentum_turn: bool
    trend_ok: bool
    oversold_count: int
    fundamentals_ok: bool
    relative_strength_ok: bool
    earnings_risk: bool


def collect_signal_evidence(row: pd.Series):
    positives: list[str] = []
    negatives: list[str] = []
    overbought: list[str] = []
    oversold: list[str] = []

    def add_positive(reason):
        positives.append(reason)

    def add_negative(reason):
        negatives.append(reason)

    def add_overbought(reason):
        overbought.append(reason)

    def add_oversold(reason):
        oversold.append(reason)

    macd_hist = row.get("macd_hist", np.nan)
    if not np.isnan(macd_hist):
        if macd_hist >= MACD_BUY_HIST_THRESHOLD:
            add_positive("MACD 히스토그램 강세")
        elif macd_hist <= MACD_SELL_HIST_THRESHOLD:
            add_negative("MACD 히스토그램 약세")
        if macd_hist <= MACD_BOTTOM_THRESHOLD:
            add_oversold("MACD 심도 약세")

    adx = row.get("adx", np.nan)
    if not np.isnan(adx):
        if adx >= ADX_TREND_THRESHOLD:
            add_positive("ADX 추세 강함")
        elif adx <= ADX_WEAK_THRESHOLD:
            add_negative("ADX 추세 약함")

    stoch_k = row.get("stoch_k", np.nan)
    if not np.isnan(stoch_k):
        if stoch_k >= STOCH_OVERBOUGHT:
            add_overbought("스토캐스틱 과열")
        elif stoch_k >= STOCH_MIN_BUY:
            add_positive("스토캐스틱 상승")
        elif stoch_k <= 20:
            add_negative("스토캐스틱 침체")
        if stoch_k <= STOCH_OVERSOLD:
            add_oversold("스토캐스틱 과매도")

    rsi = row.get("RSI", np.nan)
    if not np.isnan(rsi):
        if BUY_RSI_MIN <= rsi <= BUY_RSI_MAX:
            add_positive("RSI 건강한 상승")
        elif rsi >= OVERBOUGHT_RSI:
            add_overbought("RSI 과열")
        elif rsi <= 40:
            add_negative("RSI 약세")
        if rsi <= RSI_OVERSOLD:
            add_oversold("RSI 과매도")

    pos_52w = row.get("52주포지션", np.nan)
    if not np.isnan(pos_52w):
        if pos_52w >= BUY_POS_THRESHOLD:
            add_positive("52주 고점 상회")
        elif pos_52w <= 0.3:
            add_negative("52주 저점 부근")
        if pos_52w <= BOTTOM_POS_THRESHOLD:
            add_oversold("52주 저점권")

    trend_score = row.get("트렌드점수_최종", np.nan)
    if not np.isnan(trend_score):
        if trend_score >= BUY_SCORE_THRESHOLD:
            add_positive("트렌드 점수 우위")
        elif trend_score <= WATCH_SCORE_THRESHOLD / 2:
            add_negative("트렌드 점수 약세")

    vol_z = row.get("거래량Z(20)", np.nan)
    if not np.isnan(vol_z):
        if vol_z >= 0:
            add_positive("거래량 증가")
        elif vol_z <= -1.5:
            add_negative("거래량 감소")

    cmf = row.get("cmf_20", np.nan)
    if not np.isnan(cmf):
        if cmf >= CMF_BUY_THRESHOLD:
            add_positive("CMF 자금 유입")
        elif cmf <= -0.05:
            add_negative("CMF 자금 유출")

    obv_z = row.get("obv_z20", np.nan)
    if not np.isnan(obv_z):
        if obv_z >= OBV_Z_BUY_THRESHOLD:
            add_positive("OBV 상승")
        elif obv_z <= -0.5:
            add_negative("OBV 하락")

    boll_p = row.get("bollinger_pband", np.nan)
    if not np.isnan(boll_p):
        if boll_p >= BOLLINGER_BREAKOUT_PBAND:
            add_positive("볼린저 상단 돌파")
        if boll_p >= BOLLINGER_OVERBOUGHT_PBAND:
            add_overbought("볼린저 과열")
        elif boll_p <= 0.2:
            add_negative("볼린저 하단 부근")
        if boll_p <= BOLLINGER_OVERSOLD_PBAND:
            add_oversold("볼린저 하단 이탈")

    ema_gap = row.get("ema_gap_50_200", np.nan)
    if not np.isnan(ema_gap):
        if ema_gap >= 0:
            add_positive("EMA 정배열")
        else:
            add_negative("EMA 역배열")
        if ema_gap <= -0.05:
            add_oversold("EMA 장기 역배열")

    atr_pct = row.get("ATR%", np.nan)
    if not np.isnan(atr_pct):
        if atr_pct <= 0.05:
            add_positive("ATR 안정")
        elif atr_pct >= 0.08:
            add_negative("ATR 변동성 확대")

    roc_10 = row.get("roc_10", np.nan)
    if not np.isnan(roc_10):
        if roc_10 >= 0:
            add_positive("10일 ROC 상승")
        elif roc_10 <= -0.05:
            add_negative("10일 ROC 하락")
        if roc_10 <= -0.06:
            add_oversold("10일 ROC 과매도")

    dividend_yield = row.get("dividend_yield", np.nan)
    if not np.isnan(dividend_yield):
        if dividend_yield >= 0.02:
            add_positive("배당 수익 매력")
        elif dividend_yield <= 0.002:
            add_negative("배당 미미")

    return positives, negatives, overbought, oversold


def evaluate_bottom_context(
    row: pd.Series,
    positives: list[str],
    negatives: list[str],
    oversold: list[str],
) -> BottomContext:
    """다양한 기술·재무 지표를 종합해 저점 강도와 근거를 평가한다."""

    score = len(oversold) * 1.5
    signals: list[str] = []
    cautions: list[str] = []
    volume_support = False
    momentum_turn = False
    trend_ok = False
    oversold_count = len(oversold)
    fundamentals_ok = False
    relative_strength_ok = False
    earnings_risk = False

    def ensure(target: list[str], text: str) -> None:
        if text not in target:
            target.append(text)

    def add_signal(text: str, weight: float) -> None:
        nonlocal score
        ensure(signals, text)
        score += weight

    def add_caution(text: str, weight: float) -> None:
        nonlocal score
        ensure(cautions, text)
        score += weight

    rsi = row.get("RSI", np.nan)
    if not np.isnan(rsi):
        if rsi <= RSI_OVERSOLD:
            add_signal("RSI 과매도", 1.2)
        elif rsi <= 45:
            add_signal("RSI 저점권", 0.4)
        elif rsi >= 65:
            add_caution("RSI 반등 진행", -0.3)

    stoch_k = row.get("stoch_k", np.nan)
    if not np.isnan(stoch_k):
        if stoch_k <= STOCH_OVERSOLD:
            add_signal("스토캐스틱 과매도", 1.1)
        elif stoch_k <= 35:
            add_signal("스토캐스틱 저점권", 0.35)
        elif stoch_k >= 80:
            add_caution("스토캐스틱 반등", -0.3)

    boll_p = row.get("bollinger_pband", np.nan)
    if not np.isnan(boll_p):
        if boll_p <= BOLLINGER_OVERSOLD_PBAND:
            add_signal("볼린저 하단 이탈", 0.9)
        elif boll_p <= 0.25:
            add_signal("볼린저 하단 근접", 0.4)
        elif boll_p >= 0.8:
            add_caution("볼린저 중단 회복", -0.2)

    pos_52w = row.get("52주포지션", np.nan)
    if not np.isnan(pos_52w):
        if pos_52w <= 0.2:
            add_signal("52주 저점 인접", 1.0)
        elif pos_52w <= BOTTOM_POS_THRESHOLD:
            add_signal("52주 저점권", 0.7)
        elif pos_52w >= 0.55:
            add_caution("52주 중단 회복", -0.3)

    ret_20d = row.get("20일수익률", np.nan)
    if not np.isnan(ret_20d):
        if ret_20d <= -0.15:
            add_signal("20일 급락", 0.6)
        elif ret_20d >= 0.05:
            add_caution("20일 상승 지속", -0.4)

    macd_hist = row.get("macd_hist", np.nan)
    if not np.isnan(macd_hist):
        if macd_hist <= MACD_BOTTOM_THRESHOLD:
            add_signal("MACD 저점권", 0.7)
            momentum_turn = True
        elif macd_hist <= 0:
            add_signal("MACD 약세 둔화", 0.35)
            momentum_turn = True
        else:
            add_signal("MACD 반등", 0.3)
            momentum_turn = True

    ema_gap = row.get("ema_gap_50_200", np.nan)
    if not np.isnan(ema_gap) and ema_gap <= -0.05:
        add_signal("EMA 장기 역배열", 0.5)

    ema200_slope = row.get("ema200_slope_20", np.nan)
    if not np.isnan(ema200_slope):
        if ema200_slope >= LONG_TERM_SLOPE_MIN:
            add_signal("장기 추세 유지", 0.6)
            trend_ok = True
        else:
            add_caution("장기 추세 약화", -0.7)

    close_to_ema200 = row.get("close_to_ema200_pct", np.nan)
    if not np.isnan(close_to_ema200):
        if close_to_ema200 <= EMA200_DISTANCE_MIN:
            add_signal("EMA200 과도한 이탈", 0.6)
        elif close_to_ema200 >= 0.1:
            add_caution("EMA200 상단", -0.3)

    vol_z = row.get("거래량Z(20)", np.nan)
    if not np.isnan(vol_z):
        if vol_z >= 0.7:
            add_signal("거래량 반등", 0.6)
            volume_support = True
        elif vol_z >= 0:
            add_signal("거래량 회복", 0.4)
            volume_support = True
        elif vol_z <= -1.0:
            add_caution("거래량 부족", -0.6)

    volume_ratio = row.get("거래대금안정비", np.nan)
    if not np.isnan(volume_ratio):
        if volume_ratio >= VOLUME_STABILITY_RATIO_MIN:
            add_signal("거래대금 유지", 0.45)
            volume_support = True
        elif volume_ratio <= 0.4:
            add_caution("거래대금 위축", -0.4)

    cmf = row.get("cmf_20", np.nan)
    if not np.isnan(cmf):
        if cmf >= 0.05:
            add_signal("자금 유입", 0.35)
        elif cmf <= -0.1:
            add_caution("자금 유출", -0.45)

    obv_z = row.get("obv_z20", np.nan)
    if not np.isnan(obv_z):
        if obv_z >= 0.5:
            add_signal("OBV 회복", 0.3)
        elif obv_z <= -0.5:
            add_caution("OBV 약세", -0.4)

    roc_10 = row.get("roc_10", np.nan)
    if not np.isnan(roc_10):
        if -0.04 <= roc_10 <= 0.04:
            add_signal("단기 낙폭 둔화", 0.35)
            momentum_turn = True
        elif roc_10 <= -0.06:
            add_caution("단기 하락 지속", -0.5)

    atr_pct = row.get("ATR%", np.nan)
    if not np.isnan(atr_pct):
        if atr_pct <= 0.05:
            add_signal("변동성 안정", 0.3)
        elif atr_pct >= 0.1:
            add_caution("변동성 확대", -0.5)

    distance_10d = row.get("10일저점괴리", np.nan)
    if not np.isnan(distance_10d):
        if distance_10d <= 0.03:
            add_signal("10일 저점 인접", 0.45)
        elif distance_10d >= 0.15:
            add_caution("저점 이탈", -0.3)

    gap_down_pct = row.get("갭하락률", np.nan)
    if not np.isnan(gap_down_pct):
        if gap_down_pct <= GAP_DOWN_EXTREME:
            add_signal("갭다운 투매", 0.5)
        elif gap_down_pct >= 0.03:
            add_caution("갭상승", -0.2)

    hammer_flag = row.get("저점반전캔들", 0)
    if hammer_flag:
        add_signal("저점 반전 캔들", 0.4)

    intraday_recovery = row.get("장중반등률", np.nan)
    if not np.isnan(intraday_recovery):
        if intraday_recovery >= INTRADAY_RECOVERY_MIN:
            add_signal("장중 강한 반등", 0.4)
        elif intraday_recovery <= 0:
            add_caution("장중 회복 실패", -0.4)

    sector_rel = row.get("섹터상대강도", np.nan)
    market_rel = row.get("시장상대강도", np.nan)
    rs_hits = 0
    if not np.isnan(sector_rel):
        if sector_rel >= REL_STRENGTH_SECTOR_BUFFER:
            add_signal("섹터 상대강도 방어", 0.35)
            rs_hits += 1
        else:
            add_caution("섹터 상대약세", -0.35)
    if not np.isnan(market_rel):
        if market_rel >= REL_STRENGTH_MARKET_BUFFER:
            add_signal("시장 상대강도 방어", 0.35)
            rs_hits += 1
        else:
            add_caution("시장 상대약세", -0.35)
    if np.isnan(sector_rel) and np.isnan(market_rel):
        relative_strength_ok = True
    else:
        relative_strength_ok = rs_hits >= 1

    trend_score = row.get("트렌드점수_최종", np.nan)
    if not np.isnan(trend_score):
        if trend_score >= BOTTOM_TREND_SCORE + 0.05:
            add_signal("트렌드 점수 개선", 0.35)
            trend_ok = True
        elif trend_score <= BOTTOM_TREND_SCORE - 0.05:
            add_caution("트렌드 점수 부진", -0.5)
        elif trend_score >= BOTTOM_TREND_SCORE:
            trend_ok = True

    fundamental_score = 0.0
    fundamentals_present = False

    fund_roe = row.get("fund_roe", np.nan)
    if not np.isnan(fund_roe):
        fundamentals_present = True
        if fund_roe >= FUND_HEALTH_ROE_MIN:
            add_signal("ROE 양호", 0.45)
            fundamental_score += 1.0
        elif fund_roe <= 0:
            add_caution("ROE 부진", -0.6)

    fund_debt = row.get("fund_debt_to_equity", np.nan)
    if not np.isnan(fund_debt):
        fundamentals_present = True
        if fund_debt <= FUND_HEALTH_DEBT_TO_EQUITY_MAX:
            add_signal("부채 안정", 0.35)
            fundamental_score += 0.5
        elif fund_debt > FUND_HEALTH_DEBT_TO_EQUITY_MAX * 1.5:
            add_caution("부채 과다", -0.7)

    fund_rev = row.get("fund_revenue_growth", np.nan)
    if not np.isnan(fund_rev):
        fundamentals_present = True
        if fund_rev >= FUND_HEALTH_REVENUE_GROWTH_MIN:
            add_signal("매출 성장", 0.35)
            fundamental_score += 0.5
        elif fund_rev <= 0:
            add_caution("매출 감소", -0.5)

    fund_profit = row.get("fund_profit_margin", np.nan)
    if not np.isnan(fund_profit):
        fundamentals_present = True
        if fund_profit >= FUND_HEALTH_PROFIT_MARGIN_MIN:
            add_signal("이익률 견조", 0.35)
            fundamental_score += 0.5
        elif fund_profit <= 0:
            add_caution("이익률 적자", -0.6)

    fund_earnings = row.get("fund_earnings_growth", np.nan)
    if not np.isnan(fund_earnings):
        fundamentals_present = True
        if fund_earnings >= FUND_HEALTH_EARNINGS_GROWTH_MIN:
            add_signal("이익 성장", 0.3)
            fundamental_score += 0.4
        elif fund_earnings <= -0.1:
            add_caution("이익 감소", -0.5)

    fund_current_ratio = row.get("fund_current_ratio", np.nan)
    if not np.isnan(fund_current_ratio):
        fundamentals_present = True
        if fund_current_ratio >= FUND_HEALTH_CURRENT_RATIO_MIN:
            add_signal("유동비율 안정", 0.3)
            fundamental_score += 0.4
        elif fund_current_ratio <= 0.8:
            add_caution("유동비율 부족", -0.4)

    fund_quick_ratio = row.get("fund_quick_ratio", np.nan)
    if not np.isnan(fund_quick_ratio):
        fundamentals_present = True
        if fund_quick_ratio >= 0.8:
            fundamental_score += 0.2
        elif fund_quick_ratio <= 0.5:
            add_caution("당좌비율 취약", -0.3)

    fund_free_cf = row.get("fund_free_cashflow", np.nan)
    if not np.isnan(fund_free_cf):
        fundamentals_present = True
        if fund_free_cf > 0:
            add_signal("자유현금흐름 플러스", 0.3)
            fundamental_score += 0.4
        else:
            add_caution("자유현금흐름 마이너스", -0.4)

    fund_oper_cf = row.get("fund_operating_cashflow", np.nan)
    if not np.isnan(fund_oper_cf):
        fundamentals_present = True
        if fund_oper_cf > 0:
            fundamental_score += 0.2
        else:
            add_caution("영업현금흐름 음수", -0.3)

    if fundamentals_present:
        fundamentals_ok = fundamental_score >= 1.5
    else:
        fundamentals_ok = True

    days_to_next_earnings = row.get("days_to_next_earnings", np.nan)
    if not np.isnan(days_to_next_earnings):
        if -EARNINGS_EVENT_WINDOW_DAYS <= days_to_next_earnings < 0:
            add_signal("실적 이벤트 통과", 0.2)
        elif 0 <= days_to_next_earnings <= EARNINGS_EVENT_WINDOW_DAYS:
            add_caution("실적 발표 임박", -0.6)
            earnings_risk = True

    if len(negatives) > len(positives) + 3:
        add_caution("부정 시그널 우위", -0.7)

    score = max(score, 0.0)
    if score >= 7.0:
        grade = "강한 저점"
    elif score >= 5.0:
        grade = "진입 탐색"
    elif score >= 3.2:
        grade = "관찰 저점"
    elif score >= 1.8:
        grade = "약한 저점"
    else:
        grade = "저점 미흡"

    return BottomContext(
        score=score,
        grade=grade,
        signals=signals,
        cautions=cautions,
        volume_support=volume_support,
        momentum_turn=momentum_turn,
        trend_ok=trend_ok,
        oversold_count=oversold_count,
        fundamentals_ok=fundamentals_ok,
        relative_strength_ok=relative_strength_ok,
        earnings_risk=earnings_risk,
    )


def classify_signal(
    row: pd.Series,
    positives: list[str],
    negatives: list[str],
    overbought: list[str],
    oversold: list[str],
    bottom: BottomContext,
) -> str:
    score = row.get("트렌드점수_최종", np.nan)
    pos_52w = row.get("52주포지션", np.nan)

    if np.isnan(score) or np.isnan(pos_52w):
        return "관망 약세"

    positive_count = len(positives)
    negative_count = len(negatives)
    overbought_count = len(overbought)
    bottom_score = bottom.score
    bottom_grade = bottom.grade
    bottom_volume = bottom.volume_support
    bottom_momentum = bottom.momentum_turn
    bottom_trend = bottom.trend_ok
    bottom_cautions = len(bottom.cautions)
    bottom_fundamentals = bottom.fundamentals_ok
    bottom_relative = bottom.relative_strength_ok
    earnings_risk = bottom.earnings_risk

    market_rel = row.get("시장상대강도", np.nan)
    sector_rel = row.get("섹터상대강도", np.nan)
    if overbought_count >= 3:
        return "관망 과열"

    strong_relative = (
        (np.isnan(market_rel) or market_rel >= BUY_REL_STRENGTH_MIN)
        and (np.isnan(sector_rel) or sector_rel >= SECTOR_REL_STRENGTH_MIN)
        and bottom.relative_strength_ok
    )

    base_strong = (
        score >= BUY_SCORE_THRESHOLD
        and pos_52w >= BUY_POS_THRESHOLD
        and positive_count >= BUY_POSITIVE_MIN
        and negative_count <= BUY_NEGATIVE_MAX
        and strong_relative
        and bottom.fundamentals_ok
        and (bottom.volume_support or bottom.momentum_turn)
        and overbought_count <= 1
    )
    if base_strong:
        return "매수 후보"

    if overbought_count >= 1 and score < BUY_SCORE_THRESHOLD + 0.04:
        return "관망 과열"

    watchable_relative = (
        (np.isnan(market_rel) or market_rel >= REL_STRENGTH_MARKET_BUFFER)
        and (np.isnan(sector_rel) or sector_rel >= REL_STRENGTH_SECTOR_BUFFER)
    )

    watchable = (
        score >= WATCH_SCORE_THRESHOLD
        and pos_52w >= WATCH_POS_THRESHOLD
        and positive_count >= WATCH_POSITIVE_MIN
        and negative_count <= WATCH_NEGATIVE_MAX
        and watchable_relative
        and overbought_count <= 1
    )

    bottom_ready = (
        bottom_score >= 4.0
        and bottom_grade in ("강한 저점", "진입 탐색")
        and pos_52w <= BOTTOM_POS_THRESHOLD + 0.04
        and negative_count <= positive_count + BOTTOM_NEGATIVE_TOLERANCE
        and bottom_volume
        and (bottom_momentum or score >= BOTTOM_TREND_SCORE)
        and bottom_trend
        and bottom_relative
        and bottom_fundamentals
        and overbought_count <= 1
    )
    if bottom_ready and bottom_cautions <= 2:
        if earnings_risk and bottom_score < 6:
            return "관심 관찰"
        return "저점 관찰"

    if watchable:
        return "관심 관찰"

    if overbought_count >= 1:
        return "관망 과열"

    if negative_count >= positive_count + 1:
        return "관망 약세"

    return "관심 관찰"


def recommendation_from_signal(
    row: pd.Series,
    judgement: str,
    positives: list[str],
    negatives: list[str],
    overbought: list[str],
    oversold: list[str],
    bottom: BottomContext,
) -> str:
    positive_count = len(positives)
    negative_count = len(negatives)
    overbought_count = len(overbought)
    bottom_score = bottom.score
    bottom_grade = bottom.grade
    bottom_cautions = len(bottom.cautions)
    bottom_volume = bottom.volume_support
    bottom_momentum = bottom.momentum_turn
    bottom_trend = bottom.trend_ok
    bottom_fundamentals = bottom.fundamentals_ok
    bottom_relative = bottom.relative_strength_ok
    earnings_risk = bottom.earnings_risk

    macd_hist = row.get("macd_hist", np.nan)
    adx = row.get("adx", np.nan)
    dividend_yield = row.get("dividend_yield", np.nan)
    score = row.get("트렌드점수_최종", np.nan)
    vol_z = row.get("거래량Z(20)", np.nan)
    market_rel = row.get("시장상대강도", np.nan)
    sector_rel = row.get("섹터상대강도", np.nan)
    rsi = row.get("RSI", np.nan)
    pos_52w = row.get("52주포지션", np.nan)

    bottom_ready = (
        bottom_score >= 4.0
        and bottom_grade in ("강한 저점", "진입 탐색")
        and not np.isnan(pos_52w)
        and pos_52w <= BOTTOM_POS_THRESHOLD + 0.04
        and bottom_cautions <= 2
        and bottom_volume
        and (bottom_momentum or score >= BOTTOM_TREND_SCORE)
        and bottom_trend
        and bottom_relative
        and bottom_fundamentals
        and overbought_count <= 1
    )
    if judgement == "관망 과열":
        if overbought_count >= 2 or negative_count >= positive_count:
            return "차익 실현 고려"
        return "고평가 관망"

    if judgement == "관망 약세":
        if negative_count >= positive_count + 3:
            return "관망/보유"
        return "추가 관찰"

    if judgement == "매수 후보":
        if (
            overbought_count >= 2
            or (not np.isnan(rsi) and rsi >= OVERBOUGHT_RSI + 2)
            or (not np.isnan(market_rel) and market_rel < BUY_REL_STRENGTH_MIN)
            or (not np.isnan(sector_rel) and sector_rel < SECTOR_REL_STRENGTH_MIN)
            or not bottom_fundamentals
            or not bottom_relative
            or (earnings_risk and bottom_score < 6.5)
        ):
            return "조건 확인 후 매수"

        strong_entry = (
            positive_count >= BUY_POSITIVE_MIN + 2
            or (
                (not np.isnan(macd_hist) and macd_hist >= MACD_BUY_HIST_THRESHOLD * 1.5)
                and (not np.isnan(adx) and adx >= ADX_TREND_THRESHOLD + 10)
            )
            or (bottom_grade == "강한 저점" and bottom_score >= 5.5 and bottom_cautions <= 2)
            or (not np.isnan(score) and score >= BUY_SCORE_THRESHOLD + 0.08)
        )

        if strong_entry:
            return "적극 매수"
        if not np.isnan(dividend_yield) and dividend_yield >= 0.03:
            return "분할 매수"
        if bottom_grade in ("강한 저점", "진입 탐색") and bottom_cautions <= 2:
            return "분할 매수"
        return "조건 확인 후 매수"

    if judgement == "저점 관찰":
        if not bottom_fundamentals:
            return "반등 모니터링"
        if not bottom_relative and bottom_score < 6:
            return "반등 모니터링"
        if earnings_risk and bottom_score < 6:
            return "저점 매수 대기"
        if overbought_count >= 2:
            return "반등 모니터링"
        if (
            bottom_grade == "강한 저점"
            and bottom_score >= 5.5
            and bottom_cautions <= 2
            and (
                bottom_volume
                or (
                    positive_count >= negative_count
                    and (np.isnan(market_rel) or market_rel >= REL_STRENGTH_MARKET_BUFFER)
                )
                or (not np.isnan(vol_z) and vol_z >= 0)
            )
            and (
                bottom_momentum
                or np.isnan(macd_hist)
                or macd_hist >= MACD_BOTTOM_THRESHOLD
            )
        ):
            return "저점 분할 매수"
        if (
            bottom_score >= 3.5
            and bottom_grade in ("강한 저점", "진입 탐색", "관찰 저점")
            and bottom_cautions <= 3
            and (np.isnan(score) or score >= BOTTOM_TREND_SCORE)
            and (
                bottom_volume
                or bottom_momentum
                or (np.isnan(vol_z) or vol_z >= BOTTOM_VOLUME_Z)
            )
        ):
            return "저점 매수 대기"
        return "반등 모니터링"

    if judgement == "관심 관찰":
        if bottom_ready or (
            bottom_score >= 4.5
            and bottom_volume
            and bottom_cautions <= 3
            and bottom_grade != "저점 미흡"
            and bottom_fundamentals
        ):
            return "저점 매수 대기"
        if positive_count >= negative_count + 2:
            return "조건 충족 시 매수"
        return "추가 관찰"

    return "추가 관찰"


def attach_signals_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """판정을 붙이고 우선순위 기준으로 정렬한 결과를 반환한다."""

    records = []
    for _, row in df.iterrows():
        positives, negatives, overbought, oversold = collect_signal_evidence(row)
        bottom_context = evaluate_bottom_context(row, positives, negatives, oversold)
        judgement = classify_signal(
            row,
            positives,
            negatives,
            overbought,
            oversold,
            bottom_context,
        )
        recommendation = recommendation_from_signal(
            row,
            judgement,
            positives,
            negatives,
            overbought,
            oversold,
            bottom_context,
        )

        def summarise(items: list[str]) -> str:
            if not items:
                return ""
            top = items[:3]
            text = ", ".join(top)
            if len(items) > 3:
                text += " 등"
            return text

        positive_text = summarise(positives)
        bottom_reasons = oversold + [r for r in bottom_context.signals if r not in oversold]

        bottom_text = summarise(bottom_reasons)
        bottom_reason_text = summarise(bottom_context.signals)
        warning_text = summarise(negatives + overbought + bottom_context.cautions)
        fundamentals_label = "건강" if bottom_context.fundamentals_ok else "확인 필요"
        relative_label = "방어" if bottom_context.relative_strength_ok else "약세"
        event_label = "실적 임박" if bottom_context.earnings_risk else ""

        enriched = row.copy()
        enriched["_판단원본"] = judgement
        enriched["판단"] = JUDGEMENT_DISPLAY.get(judgement, judgement)
        enriched["_추천원본"] = recommendation
        enriched["추천"] = RECOMMENDATION_DISPLAY.get(recommendation, recommendation)
        enriched["긍정"] = positive_text
        enriched["저점"] = bottom_text
        enriched["저점강도"] = bottom_context.grade
        enriched["저점근거"] = bottom_reason_text
        enriched["저점점수"] = round(bottom_context.score, 2)
        enriched["저점건강"] = fundamentals_label
        enriched["상대강도"] = relative_label
        enriched["이벤트주의"] = event_label
        enriched["경계"] = warning_text
        enriched["_positives"] = len(positives)
        enriched["_negatives"] = len(negatives) + len(overbought)
        enriched["_oversold"] = len(oversold)
        records.append(enriched)

    out = pd.DataFrame(records)
    out["우선순위"] = out["_판단원본"].map(SIGNAL_PRIORITY).astype(int)
    out = out.sort_values(["우선순위", "트렌드점수_최종"], ascending=[True, False])
    return out.drop(columns=["_positives", "_negatives", "_oversold", "_추천원본"])
