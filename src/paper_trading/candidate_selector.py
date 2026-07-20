"""109개 → 1개: 다단계 필터 + 복합 스코어링 후보 선정 엔진.

Hard Filter (7개) → Composite Conviction Score (CCS, 5 서브스코어)
→ 시장 레짐별 가중치 조정 → 섹터 분산 페널티 → Top 1 선정.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────

from screener.config import (
    ADX_BUY_MIN,
    CANDIDATE_MIN_STRATEGY_SCORE,
    CANDIDATE_BOTTOM_MAX_52W_POS,
    CANDIDATE_CCS_MIN_NORMAL,
    CANDIDATE_CCS_MIN_BEAR,
    CANDIDATE_RSI_MAX,
    CANDIDATE_BOLLINGER_MAX,
    CANDIDATE_5D_RETURN_MAX,
    CANDIDATE_LIQUIDITY_MIN,
    CANDIDATE_EARNINGS_BUFFER_DAYS,
    CANDIDATE_MAX_SAME_SECTOR,
    CANDIDATE_REGIME_WEIGHTS as REGIME_WEIGHTS,
)

# 판단/추천 컬럼의 최상위 등급 (exporter에서 display 변환된 값)
_TOP_JUDGMENTS = {"1. 매수 후보", "1. 저점 반등"}
_TOP_RECOMMENDATIONS = {"1. 즉시 진입", "1. 반등 매수"}

# 상승 패턴 목록
_BULLISH_PATTERNS = {
    "이중바닥", "역헤드앤숄더", "하락쐐기", "강세잉걸핑", "모닝스타",
    "상승삼각형", "컵앤핸들", "골든크로스", "골든크로스임박",
    "정배열(완전)", "정배열(눌림목)",
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    """NaN/None 안전 변환."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (ValueError, TypeError):
        return default


# ── Phase 1: Hard Filters ────────────────────────────────────


def _apply_hard_filters(
    df: pd.DataFrame,
    current_holdings: list[dict[str, Any]],
    rsi_max: float | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """7개 hard filter 적용. (통과 df, 필터별 탈락 수 dict) 반환."""
    rejections: dict[str, int] = {}
    total = len(df)
    effective_rsi_max = rsi_max if rsi_max is not None else CANDIDATE_RSI_MAX

    # 1. 전략 점수 >= threshold (전략 라벨 부여된 종목만)
    bottom_scores = df.get("바닥반등_적합도", pd.Series(0, index=df.index)).fillna(0)
    momentum_scores = df.get("모멘텀_적합도", pd.Series(0, index=df.index)).fillna(0)
    max_scores = bottom_scores.combine(momentum_scores, max)
    mask = max_scores >= CANDIDATE_MIN_STRATEGY_SCORE
    rejections[f"전략점수<{CANDIDATE_MIN_STRATEGY_SCORE}"] = int((~mask).sum())

    # 전략점수 분포 로그 (필터 튜닝용)
    bins = [0, 4.0, 5.0, 5.5, CANDIDATE_MIN_STRATEGY_SCORE]
    labels = [f"~4.0", "4.0~5.0", f"5.0~5.5", f"5.5~{CANDIDATE_MIN_STRATEGY_SCORE}"]
    dist_parts = []
    for i, label in enumerate(labels):
        lo = bins[i]
        hi = bins[i + 1]
        count = int(((max_scores >= lo) & (max_scores < hi)).sum())
        dist_parts.append(f"{label}: {count}")
    above = int((max_scores >= CANDIDATE_MIN_STRATEGY_SCORE).sum())
    dist_parts.append(f"{CANDIDATE_MIN_STRATEGY_SCORE}+: {above}")
    logger.info(f"[전략점수 분포] {' | '.join(dist_parts)}")

    df = df[mask]

    # 1b. 바닥반등 52주 포지션 필터 (이미 바닥에서 많이 올라온 종목 차단)
    strategy_col = df.get("전략구분", pd.Series("", index=df.index)).astype(str)
    is_bottom = strategy_col.str.contains("바닥반등", na=False)
    pos_52w = df.get("52주포지션", pd.Series(0.5, index=df.index)).fillna(0.5)
    bottom_pos_ok = ~is_bottom | (pos_52w <= CANDIDATE_BOTTOM_MAX_52W_POS)
    rejections["바닥반등_52주과열"] = int((~bottom_pos_ok).sum())
    df = df[bottom_pos_ok]

    # 2. 판단 등급
    judgment_ok = df.get("판단", pd.Series("", index=df.index)).isin(_TOP_JUDGMENTS)
    recommend_ok = df.get("추천", pd.Series("", index=df.index)).isin(_TOP_RECOMMENDATIONS)
    mask = judgment_ok | recommend_ok
    rejections["판단등급_미달"] = int((~mask).sum())
    df = df[mask]

    # 3. 과열 차단
    rsi_ok = df.get("RSI", pd.Series(50, index=df.index)).fillna(50) < effective_rsi_max
    bband_ok = df.get("bollinger_pband", pd.Series(0.5, index=df.index)).fillna(0.5) < CANDIDATE_BOLLINGER_MAX
    ret5_ok = df.get("5일수익률", pd.Series(0, index=df.index)).fillna(0) < CANDIDATE_5D_RETURN_MAX
    mask = rsi_ok & bband_ok & ret5_ok
    rejections["과열_차단"] = int((~mask).sum())
    df = df[mask]

    # 4. 유동성
    liq = df.get("최근20일평균거래대금", pd.Series(0, index=df.index)).fillna(0)
    mask = liq >= CANDIDATE_LIQUIDITY_MIN
    rejections["유동성_부족"] = int((~mask).sum())
    df = df[mask]

    # 5. 어닝 회피
    earnings_days = df.get("days_to_next_earnings", pd.Series(float("nan"), index=df.index))
    mask = earnings_days.isna() | (earnings_days > CANDIDATE_EARNINGS_BUFFER_DAYS)
    rejections["어닝_임박"] = int((~mask).sum())
    df = df[mask]

    # 6. 보유 중복 + 벤치마크(SPY) 제외
    held_tickers = {p["ticker"] for p in current_holdings} | {"SPY"}
    mask = ~df.get("티커", pd.Series("", index=df.index)).isin(held_tickers)
    rejections["보유_중복"] = int((~mask).sum())
    df = df[mask]

    # 7. 모멘텀 전략 ADX 최소 필터 (ADX < 20이면 추세 약함 → 모멘텀 진입 차단)
    strategy_col2 = df.get("전략구분", pd.Series("", index=df.index)).astype(str)
    is_momentum = strategy_col2.str.contains("모멘텀", na=False)
    adx_col = df.get("adx", pd.Series(25, index=df.index)).fillna(25)
    momentum_adx_ok = ~is_momentum | (adx_col >= ADX_BUY_MIN)
    rejections["모멘텀_ADX부족"] = int((~momentum_adx_ok).sum())
    df = df[momentum_adx_ok]

    # 8. 바닥반등 RSI 30-35 데드존 차단 (해당 구간 승률 25% — 분석 결과)
    # df 필터링 후 strategy_col2 인덱스를 맞춤
    strategy_col2 = strategy_col2.reindex(df.index, fill_value="")
    is_bottom = strategy_col2.str.contains("바닥", na=False)
    rsi_col = df.get("RSI", pd.Series(50, index=df.index)).fillna(50)
    bottom_rsi_dead = is_bottom & (rsi_col >= 30) & (rsi_col < 35)
    rejections["바닥_RSI데드존"] = int(bottom_rsi_dead.sum())
    df = df[~bottom_rsi_dead]

    # 9. 섹터 집중 차단
    sector_counts: dict[str, int] = {}
    for p in current_holdings:
        s = p.get("sector", "")
        if s:
            sector_counts[s] = sector_counts.get(s, 0) + 1
    blocked_sectors = {s for s, c in sector_counts.items() if c >= CANDIDATE_MAX_SAME_SECTOR}
    if blocked_sectors:
        mask = ~df.get("섹터", pd.Series("", index=df.index)).isin(blocked_sectors)
        rejections["섹터_집중"] = int((~mask).sum())
        df = df[mask]
    else:
        rejections["섹터_집중"] = 0

    rejections["_통과"] = len(df)
    rejections["_전체"] = total
    return df, rejections


# ── Phase 2: Composite Conviction Score ──────────────────────


def _score_strategy_fit(row: pd.Series) -> float:
    """A. Strategy Fit [0, 1]."""
    bottom = _safe_float(row.get("바닥반등_적합도"))
    momentum = _safe_float(row.get("모멘텀_적합도"))
    score = max(bottom, momentum) / 10.0
    return min(score, 1.0)


def _score_entry_timing(row: pd.Series) -> float:
    """B. Entry Timing [0, 1]."""
    timing = 0.0
    rsi = _safe_float(row.get("RSI"), 50)
    bband = _safe_float(row.get("bollinger_pband"), 0.5)
    vol_z = _safe_float(row.get("거래량Z(20)"))
    macd_hist = _safe_float(row.get("macd_hist"))
    strategy = str(row.get("전략구분", ""))

    # RSI 스윗스팟
    if "바닥반등" in strategy:
        if 25 <= rsi <= 40:
            timing += 0.4
        elif 40 < rsi <= 50:
            timing += 0.2
    elif "모멘텀" in strategy:
        if 45 <= rsi <= 60:
            timing += 0.4
        elif 35 <= rsi < 45:
            timing += 0.2

    # 볼린저 위치
    if bband < 0.2:
        timing += 0.3
    elif bband < 0.4:
        timing += 0.15

    # 거래량 확인
    if vol_z > 1.0:
        timing += 0.15
    elif vol_z > 0.0:
        timing += 0.05

    # MACD 방향
    if "모멘텀" in strategy and macd_hist > 0:
        timing += 0.15
    elif "바닥반등" in strategy and macd_hist > -0.5:
        # 턴업 중 (깊은 마이너스가 아닌 경우)
        timing += 0.1

    return min(timing, 1.0)


def _score_alpha_factor(row: pd.Series) -> float:
    """C. Alpha Factor [0, 1] — 전략별 다른 팩터 가중치. ML_ENABLED 시 블렌딩."""
    mom = _safe_float(row.get("팩터_모멘텀"))
    trend = _safe_float(row.get("팩터_추세"))
    vol = _safe_float(row.get("팩터_거래량"))
    volat = _safe_float(row.get("팩터_변동성"))
    mr = _safe_float(row.get("팩터_평균회귀"))
    strategy = str(row.get("전략구분", ""))

    if "바닥반등" in strategy:
        raw = 0.10 * mom + 0.10 * trend + 0.15 * vol + 0.25 * volat + 0.40 * mr
    else:  # 모멘텀 또는 기타
        # 모멘텀/추세 과대평가 → 고점매수 방지: volume·mean_reversion 비중 강화
        raw = 0.20 * mom + 0.20 * trend + 0.25 * vol + 0.15 * volat + 0.20 * mr

    # [-1, 1] → [0, 1]
    rule_score = max(0.0, min((raw + 1.0) / 2.0, 1.0))

    # Phase B: ML 블렌딩 (ML_ENABLED=False면 즉시 반환)
    from screener.config import ML_ENABLED, ML_BLEND_WEIGHT_ALPHA
    if not ML_ENABLED:
        return rule_score
    try:
        from ml.prediction import predict_entry_quality
        features = {
            "factor_momentum": mom, "factor_trend": trend,
            "factor_volume": vol, "factor_volatility": volat,
            "factor_mean_reversion": mr,
            "alpha_score": _safe_float(row.get("알파점수")),
            "RSI": _safe_float(row.get("RSI")),
            "adx": _safe_float(row.get("adx")),
            "ret_5d": _safe_float(row.get("5일수익률")),
            "ret_20d": _safe_float(row.get("20일수익률")),
            "vol_z20": _safe_float(row.get("거래량Z(20)")),
            "sector": row.get("섹터"),
            "regime": row.get("regime"),
        }
        ml_score = predict_entry_quality(features, target="y_win")
        return ML_BLEND_WEIGHT_ALPHA * ml_score + (1 - ML_BLEND_WEIGHT_ALPHA) * rule_score
    except Exception as e:
        logger.warning(f"ML alpha blend 실패, rule-based 폴백: {e}")
        return rule_score


def _score_risk_quality(row: pd.Series) -> float:
    """D. Risk-Adjusted Quality [0, 1]."""
    score = 0.0
    strategy = str(row.get("전략구분", ""))

    # 변동성 압축
    atr_pct = _safe_float(row.get("ATR%"))
    atr_med = _safe_float(row.get("atr_med_252"))
    if atr_pct > 0 and atr_med > 0:
        vol_ratio = atr_pct / atr_med
        if vol_ratio < 0.8:
            score += 0.3
        elif vol_ratio < 1.0:
            score += 0.15

    # 52주 포지션
    w52 = _safe_float(row.get("52주포지션"), 0.5)
    if "바닥반등" in strategy:
        if 0.10 <= w52 <= 0.40:
            score += 0.2
    elif "모멘텀" in strategy:
        if 0.55 <= w52 <= 0.85:
            score += 0.2

    # 펀더멘탈
    roe = _safe_float(row.get("fund_roe"))
    if roe > 0.10:
        score += 0.15
    debt = _safe_float(row.get("fund_debt_to_equity"), 999)
    if debt < 100:
        score += 0.1

    # 기관 보유
    inst = _safe_float(row.get("fund_institutional_holders_pct"))
    if inst > 0.60:
        score += 0.1

    # 공매도
    short_pct = _safe_float(row.get("fund_short_pct_float"))
    if short_pct < 0.05:
        score += 0.05
    elif short_pct > 0.15:
        score -= 0.05

    # 주봉/월봉 추세 건강도
    weekly_pat = str(row.get("주봉패턴", ""))
    monthly_pat = str(row.get("월봉패턴", ""))

    if "정배열(완전)" in monthly_pat:
        score += 0.10
    elif "정배열(눌림목)" in monthly_pat:
        score += 0.05
    elif "정배열(형성중)" in monthly_pat:
        score += 0.02

    if "정배열(완전)" in weekly_pat:
        score += 0.05
    elif "정배열(눌림목)" in weekly_pat:
        score += 0.025

    return max(0.0, min(score, 1.0))


def _has_bullish_pattern(pattern_str: Any) -> bool:
    """패턴 문자열에서 상승 패턴 존재 여부."""
    if not pattern_str or str(pattern_str).lower() in ("nan", "none", ""):
        return False
    return any(p in str(pattern_str) for p in _BULLISH_PATTERNS)


def _score_confluence(row: pd.Series) -> float:
    """E. Pattern & ML Confluence [0, 1]."""
    score = 0.0
    strategy = str(row.get("전략구분", ""))

    # ML 저점확률
    low_prob = _safe_float(row.get("저점확률"))
    if "바닥반등" in strategy:
        if low_prob >= 0.7:
            score += 0.4
        elif low_prob >= 0.5:
            score += 0.2

    # 반등스코어
    reversal = _safe_float(row.get("반등스코어"))
    score += min(reversal / 10.0, 1.0) * 0.2

    # 멀티타임프레임 패턴
    pattern_count = sum([
        _has_bullish_pattern(row.get("일봉패턴")),
        _has_bullish_pattern(row.get("주봉패턴")),
        _has_bullish_pattern(row.get("월봉패턴")),
    ])
    if pattern_count >= 3:
        score += 0.3
    elif pattern_count == 2:
        score += 0.15
    elif pattern_count == 1:
        score += 0.05

    # EMA 정배열
    ema20 = _safe_float(row.get("ema20"))
    ema50 = _safe_float(row.get("ema50"))
    ema200 = _safe_float(row.get("ema200"))
    if ema20 > 0 and ema50 > 0 and ema200 > 0 and ema20 > ema50 > ema200:
        score += 0.1

    return min(score, 1.0)


# ── Phase 3: Market Regime ───────────────────────────────────


def detect_market_regime(df: pd.DataFrame) -> str:
    """SPY 행이 있으면 EMA 기반 레짐 판단, 없으면 'neutral'."""
    spy_rows = df[df.get("티커", pd.Series(dtype=str)) == "SPY"]
    if spy_rows.empty:
        return "neutral"

    spy = spy_rows.iloc[0]
    close = _safe_float(spy.get("현재가격", spy.get("close")))
    ema50 = _safe_float(spy.get("ema50"))
    ema200 = _safe_float(spy.get("ema200"))

    if close > 0 and ema50 > 0 and ema200 > 0:
        if close > ema50 > ema200:
            return "bull"
        elif close < ema200:
            return "bear"
    return "neutral"


# ── Phase 4: Sector Penalty ──────────────────────────────────


def _sector_penalty(
    candidate_sector: str, current_holdings: list[dict[str, Any]]
) -> float:
    """같은 섹터 보유 시 페널티."""
    same = sum(1 for p in current_holdings if p.get("sector") == candidate_sector and candidate_sector)
    if same >= 2:
        return 0.15  # hard filter에서 이미 차단하지만 안전망
    elif same == 1:
        return 0.05
    return 0.0


# ── Main: Select Best Candidate ──────────────────────────────


def select_best_candidate(
    df: pd.DataFrame,
    current_holdings: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """스크리너 결과에서 최고 후보 1개 선정.

    Returns:
        (best_candidate_dict | None, debug_info)
        debug_info: regime, rejections, top5 등 디버깅 정보.
    """
    debug: dict[str, Any] = {}

    # 레짐 판단
    regime = detect_market_regime(df)
    debug["regime"] = regime
    weights = REGIME_WEIGHTS[regime]

    # 약세장 추가 제한 (글로벌 변수 대신 로컬 변수 사용)
    effective_rsi_max = 70 if regime == "bear" else CANDIDATE_RSI_MAX

    # Hard Filter
    filtered, rejections = _apply_hard_filters(df, current_holdings, effective_rsi_max)
    debug["rejections"] = rejections

    if filtered.empty:
        debug["top5"] = []
        return None, debug

    # 약세장: buy_signal 필수 + 모멘텀 전략 제외
    if regime == "bear":
        buy_sig = filtered.get("buy_signal", pd.Series(True, index=filtered.index))
        bear_filtered = filtered[buy_sig == True]
        if not bear_filtered.empty:
            filtered = bear_filtered
        # 모멘텀 전략 제외 (약세장에서 데드캣 바운스 위험)
        strategy_col = filtered.get("전략구분", pd.Series("", index=filtered.index))
        non_momentum = filtered[~strategy_col.str.contains("모멘텀", na=False)]
        if not non_momentum.empty:
            filtered = non_momentum

    # CCS 계산
    scores = []
    for idx, row in filtered.iterrows():
        a = _score_strategy_fit(row)
        b = _score_entry_timing(row)
        c = _score_alpha_factor(row)
        d = _score_risk_quality(row)
        e = _score_confluence(row)

        sector = str(row.get("섹터", ""))
        penalty = _sector_penalty(sector, current_holdings)

        base_ccs = (
            weights["strategy"] * a
            + weights["timing"] * b
            + weights["alpha"] * c
            + weights["risk"] * d
            + weights["confluence"] * e
            - penalty
        )

        # 레짐-전략 정합성 보너스
        regime_bonus = 0.0
        row_strategy = str(row.get("전략구분", ""))
        dominant_score = max(
            _safe_float(row.get("바닥반등_적합도")),
            _safe_float(row.get("모멘텀_적합도")),
        )
        if regime == "bull" and "모멘텀" in row_strategy:
            regime_bonus = 0.015 * (dominant_score / 10.0)
        elif regime == "bear" and "바닥반등" in row_strategy:
            regime_bonus = 0.015 * (dominant_score / 10.0)

        ccs = base_ccs + regime_bonus

        scores.append({
            "idx": idx,
            "ticker": row.get("티커", ""),
            "ccs": round(ccs, 4),
            "strategy_fit": round(a, 3),
            "timing": round(b, 3),
            "alpha": round(c, 3),
            "risk": round(d, 3),
            "confluence": round(e, 3),
            "penalty": round(penalty, 3),
            "strategy": str(row.get("전략구분", "")),
            "star": str(row.get("매수적합도_표시", "")),
            "sector": str(row.get("섹터", "")),
            "vol_ratio": _safe_float(row.get("거래량돌파배수")),
            "vol_ma20": _safe_float(row.get("volume_ma20")),
            "current_price": _safe_float(row.get("현재가격", row.get("close"))),
        })

    scores.sort(key=lambda x: x["ccs"], reverse=True)

    # 표시용: 문턱값 필터 전 전체 스코어를 ticker→score dict로 보존
    # (골든크로스 임박 등 후보 외 티커의 CCS 조회용)
    debug["all_scores"] = {s["ticker"]: s for s in scores}

    # CCS 최소 문턱값 (약세장: 0.45, 그 외: 0.40)
    min_ccs = CANDIDATE_CCS_MIN_BEAR if regime == "bear" else CANDIDATE_CCS_MIN_NORMAL
    scores = [s for s in scores if s["ccs"] >= min_ccs]
    if not scores:
        debug["top5"] = []
        debug["rejection_reason"] = f"CCS 문턱값 미달 (min={min_ccs})"
        return None, debug

    # 동점 처리 (차이 < 0.02)
    if len(scores) >= 2 and abs(scores[0]["ccs"] - scores[1]["ccs"]) < 0.02:
        top_tied = [s for s in scores if abs(s["ccs"] - scores[0]["ccs"]) < 0.02]

        def tiebreak_key(s):
            row = filtered.loc[s["idx"]]
            strategy = s["strategy"]

            # 1순위: 레짐-전략 정합성
            regime_aligned = 1  # 기본: 비정합
            if regime == "bull" and "모멘텀" in strategy:
                regime_aligned = 0
            elif regime == "bear" and "바닥반등" in strategy:
                regime_aligned = 0
            elif regime == "neutral":
                regime_aligned = 0  # neutral에서는 동등

            # 2순위: 전략 점수 (높을수록 우선)
            raw_score = max(
                _safe_float(row.get("바닥반등_적합도")),
                _safe_float(row.get("모멘텀_적합도")),
            )

            # 3순위: buy_support_count
            support = _safe_float(row.get("buy_support_count"))

            # 4순위: ATR% (낮을수록 안전)
            atr = _safe_float(row.get("ATR%"), 999)

            return (regime_aligned, -raw_score, -support, atr)

        top_tied.sort(key=tiebreak_key)
        best_score = top_tied[0]
    else:
        best_score = scores[0]

    # Top 5 로깅
    debug["top5"] = scores[:5]

    # 선정된 종목의 전체 row를 dict로 반환
    best_row = filtered.loc[best_score["idx"]]
    result = {
        "ticker": best_score["ticker"],
        "entry_price": _safe_float(best_row.get("현재가격", best_row.get("close"))),
        "strategy": best_score["strategy"],
        "star_rating": best_score["star"],
        "ccs_score": best_score["ccs"],
        "sector": best_score["sector"],
        "ccs_breakdown": {
            k: best_score[k]
            for k in ("strategy_fit", "timing", "alpha", "risk", "confluence", "penalty")
        },
        "vol_ratio": _safe_float(best_row.get("거래량돌파배수")),
        "vol_ma20": _safe_float(best_row.get("volume_ma20")),
    }

    logger.info(
        f"[CandidateSelector] Regime={regime} | "
        f"Filtered={rejections['_통과']}/{rejections['_전체']} | "
        f"Selected: {result['ticker']} (CCS={result['ccs_score']:.4f})"
    )

    return result, debug
