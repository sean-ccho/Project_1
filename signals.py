"""시그널 판정 및 정렬 헬퍼."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ADX_TREND_THRESHOLD,
    ADX_WEAK_THRESHOLD,
    BOLLINGER_BREAKOUT_PBAND,
    BOLLINGER_OVERBOUGHT_PBAND,
    BUY_POS_THRESHOLD,
    BUY_RSI_MAX,
    BUY_RSI_MIN,
    BUY_SCORE_THRESHOLD,
    CMF_BUY_THRESHOLD,
    MACD_BUY_HIST_THRESHOLD,
    MACD_SELL_HIST_THRESHOLD,
    OBV_Z_BUY_THRESHOLD,
    OVERBOUGHT_RSI,
    SIGNAL_PRIORITY,
    STOCH_MIN_BUY,
    STOCH_OVERBOUGHT,
    WATCH_POS_THRESHOLD,
    WATCH_SCORE_THRESHOLD,
)


def collect_signal_evidence(row: pd.Series):
    positives: list[str] = []
    negatives: list[str] = []
    overbought: list[str] = []
    oversold: list[str] = []

    def add_positive(reason):
        positives.append(reason)

    def add_negative(reason):
        negatives.append(reason)

    macd_hist = row.get("macd_hist", np.nan)
    if not np.isnan(macd_hist):
        if macd_hist >= MACD_BUY_HIST_THRESHOLD:
            add_positive("MACD 히스토그램 강세")
        elif macd_hist <= MACD_SELL_HIST_THRESHOLD:
            add_negative("MACD 히스토그램 약세")

    adx = row.get("adx", np.nan)
    if not np.isnan(adx):
        if adx >= ADX_TREND_THRESHOLD:
            add_positive("ADX 추세 강함")
        elif adx <= ADX_WEAK_THRESHOLD:
            add_negative("ADX 추세 약함")

    stoch_k = row.get("stoch_k", np.nan)
    if not np.isnan(stoch_k):
        if stoch_k >= STOCH_OVERBOUGHT:
            overbought.append("스토캐스틱 과열")
        elif stoch_k >= STOCH_MIN_BUY:
            add_positive("스토캐스틱 상승")
        elif stoch_k <= 20:
            add_negative("스토캐스틱 침체")

    rsi = row.get("RSI", np.nan)
    if not np.isnan(rsi):
        if BUY_RSI_MIN <= rsi <= BUY_RSI_MAX:
            add_positive("RSI 건강한 상승")
        elif rsi >= OVERBOUGHT_RSI:
            overbought.append("RSI 과열")
        elif rsi <= 40:
            add_negative("RSI 약세")

    pos_52w = row.get("52주포지션", np.nan)
    if not np.isnan(pos_52w):
        if pos_52w >= BUY_POS_THRESHOLD:
            add_positive("52주 고점 상회")
        elif pos_52w <= 0.3:
            add_negative("52주 저점 부근")

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
            overbought.append("볼린저 과열")
        elif boll_p <= 0.2:
            add_negative("볼린저 하단 부근")

    ema_gap = row.get("ema_gap_20_200", np.nan)
    if not np.isnan(ema_gap):
        if ema_gap >= 0:
            add_positive("EMA 정배열")
        else:
            add_negative("EMA 역배열")

    atr_pct = row.get("ATR%", np.nan)
    if not np.isnan(atr_pct):
        if atr_pct <= 0.05:
            add_positive("ATR 안정")
        elif atr_pct >= 0.08:
            add_negative("ATR 변동성 확대")

    keltner_width = row.get("keltner_width", np.nan)
    if not np.isnan(keltner_width):
        if keltner_width >= 0.6:
            add_positive("켈트너 채널 확장")
        elif keltner_width <= 0.2:
            add_negative("켈트너 채널 축소")

    roc_10 = row.get("roc_10", np.nan)
    if not np.isnan(roc_10):
        if roc_10 >= 0:
            add_positive("10일 ROC 상승")
        elif roc_10 <= -0.05:
            add_negative("10일 ROC 하락")

    dividend_yield = row.get("dividend_yield", np.nan)
    if not np.isnan(dividend_yield):
        if dividend_yield >= 0.02:
            add_positive("배당 수익 매력")
        elif dividend_yield <= 0.002:
            add_negative("배당 미미")

    return positives, negatives, overbought, oversold


def classify_signal(
    row: pd.Series,
    positives: list[str],
    negatives: list[str],
    overbought: list[str],
    oversold: list[str],
) -> str:
    score = row.get("트렌드점수_최종", np.nan)
    pos_52w = row.get("52주포지션", np.nan)

    if np.isnan(score) or np.isnan(pos_52w):
        return "관망 약세"

    positive_count = len(positives)
    negative_count = len(negatives)
    overbought_count = len(overbought)

    if overbought_count >= 2:
        return "관망 과열"

    base_strong = (
        score >= BUY_SCORE_THRESHOLD
        and pos_52w >= BUY_POS_THRESHOLD
        and positive_count >= 6
        and negative_count <= 2
    )
    if base_strong:
        return "매수 후보"

    watchable = (
        score >= WATCH_SCORE_THRESHOLD
        and pos_52w >= WATCH_POS_THRESHOLD
        and positive_count >= 4
        and negative_count <= 3
    )
    if watchable:
        return "관심 관찰"

    if negative_count > positive_count + 1:
        return "관망 약세"

    return "관심 관찰"


def recommendation_from_signal(
    row: pd.Series,
    judgement: str,
    positives: list[str],
    negatives: list[str],
    overbought: list[str],
    oversold: list[str],
) -> str:
    positive_count = len(positives)
    negative_count = len(negatives)
    overbought_count = len(overbought)

    macd_hist = row.get("macd_hist", np.nan)
    adx = row.get("adx", np.nan)
    dividend_yield = row.get("dividend_yield", np.nan)

    if judgement == "매수 후보":
        if (
            positive_count >= 9
            or (
                (not np.isnan(macd_hist) and macd_hist >= MACD_BUY_HIST_THRESHOLD * 1.5)
                and (not np.isnan(adx) and adx >= ADX_TREND_THRESHOLD + 10)
            )
        ):
            return "적극 매수"
        if not np.isnan(dividend_yield) and dividend_yield >= 0.03:
            return "분할 매수"
        return "분할 매수"

    if judgement == "관심 관찰":
        if positive_count >= negative_count + 2:
            return "조건 충족 시 매수"
        return "추가 관찰"

    if judgement == "관망 과열":
        if overbought_count >= 2 or negative_count >= positive_count:
            return "차익 실현 고려"
        return "고평가 관망"

    if negative_count >= positive_count + 3:
        return "관망/보유"
    return "추가 관찰"


def attach_signals_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """판정을 붙이고 우선순위 기준으로 정렬한 결과를 반환한다."""

    records = []
    for _, row in df.iterrows():
        positives, negatives, overbought, oversold = collect_signal_evidence(row)
        judgement = classify_signal(row, positives, negatives, overbought, oversold)
        recommendation = recommendation_from_signal(
            row, judgement, positives, negatives, overbought, oversold
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
        warning_text = summarise(negatives + overbought)

        enriched = row.copy()
        enriched["판단"] = judgement
        enriched["추천"] = recommendation
        enriched["긍정"] = positive_text
        enriched["경계"] = warning_text
        enriched["_positives"] = len(positives)
        enriched["_negatives"] = len(negatives) + len(overbought)
        records.append(enriched)

    out = pd.DataFrame(records)
    out["우선순위"] = out["판단"].map(SIGNAL_PRIORITY).astype(int)
    out = out.sort_values(["우선순위", "트렌드점수_최종"], ascending=[True, False])
    return out.drop(columns=["_positives", "_negatives"])
