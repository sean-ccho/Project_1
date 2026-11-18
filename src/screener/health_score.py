"""회사 건전성 점수(헬스체크) — 표시 전용 (매매 · 점수 · 필터에 안 쓴다).

Yahoo 종목 정보(fundamentals.fetch_fundamental_snapshots 의 fund_* 칸) 8개 항목을 기준과 비교해 0~100 점.

받은 항목만으로 매긴다 (2026-11-18): Yahoo 는 은행에 부채비율 · 유동비율 · 잉여현금흐름을, 자본이 마이너스인 회사
(자사주 매입이 많은 ABBV · LOW 등)에 ROE · 부채비율을 주지 않는다. 예전에는 빠진 항목을 0점으로 세서 받은 항목을
전부 통과한 JPM 이 65(C), RY.TO 가 60(C) 였다. 이제 점수 = 통과한 항목 점수 ÷ 받은 항목 만점 × 100 이고,
받은 항목이 5개 미만이면 점수를 매기지 않는다 ('N/A' — 웹은 '자료 부족'). 8개를 다 받은 종목은 예전 점수와 같다.

fundamentals.py(백테스트 캐시 지문 대상)에서 옮겨 왔다 — 표시만 바꿀 때 10년 백테스트 캐시가 깨지지 않게.
"""

from __future__ import annotations

import math
from typing import Any

from screener.config import (
    FUND_HEALTH_CURRENT_RATIO_MIN,
    FUND_HEALTH_DEBT_TO_EQUITY_MAX,
    FUND_HEALTH_EARNINGS_GROWTH_MIN,
    FUND_HEALTH_PROFIT_MARGIN_MIN,
    FUND_HEALTH_REVENUE_GROWTH_MIN,
    FUND_HEALTH_ROE_MIN,
)
from screener.fundamentals import format_growth_indicators

MIN_ITEMS = 5  # 받은 항목이 이보다 적으면 점수 없음 (4개면 만점 45점짜리로 A+ 가 나올 수 있어 근거가 얇다)

# (칸, 기준, 점수, "gt" = 기준보다 커야 통과 / "lt" = 작아야 통과) — 8개 합 100
ITEMS: tuple[tuple[str, float, int, str], ...] = (
    ("fund_roe", FUND_HEALTH_ROE_MIN, 20, "gt"),
    ("fund_debt_to_equity", FUND_HEALTH_DEBT_TO_EQUITY_MAX, 15, "lt"),  # Yahoo 는 % (200 = 2배)
    ("fund_revenue_growth", FUND_HEALTH_REVENUE_GROWTH_MIN, 15, "gt"),
    ("fund_profit_margin", FUND_HEALTH_PROFIT_MARGIN_MIN, 15, "gt"),
    ("fund_earnings_growth", FUND_HEALTH_EARNINGS_GROWTH_MIN, 10, "gt"),
    ("fund_current_ratio", FUND_HEALTH_CURRENT_RATIO_MIN, 10, "gt"),
    ("fund_free_cashflow", 0, 10, "gt"),
    ("fund_institutional_holders_pct", 0.60, 5, "gt"),
)


def health_grade(score: int) -> str:
    """0~100 → A+ (90↑) · A · B · C · D · F (50 미만)."""
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def _value(row: Any, col: str) -> float | None:
    """칸 값 → 숫자. 없음 · 글자 · NaN 은 None (받지 못한 항목)."""
    v = row.get(col) if hasattr(row, "get") else None
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def calculate_health_score(row: Any) -> tuple[int | None, str]:
    """회사 건전성 점수 (0-100) + 표시 문자열.

    Returns:
        (score, display)
        - score: 0-100, 받은 항목이 5개 미만이면 None
        - display: "100 (A+) · 매출+10% 마진+35% 이익+10% · 5/8개 항목" (8개 다 받으면 뒤 ' · N/8개 항목' 없음) 또는 "N/A"
          (시트는 그대로 한 줄, 메일은 첫 ' · ' 로 나눠 두 줄, 웹은 web_export.dashboard.parse_health)
    """
    earned = possible = got = 0
    for col, threshold, points, op in ITEMS:
        v = _value(row, col)
        if v is None:
            continue
        got += 1
        possible += points
        if (op == "gt" and v > threshold) or (op == "lt" and v < threshold):
            earned += points
    if got < MIN_ITEMS:
        return None, "N/A"
    score = math.floor(earned * 100 / possible + 0.5)  # 반올림 (0.5 는 올림)
    text = f"{score} ({health_grade(score)}) · {format_growth_indicators(row)}"
    if got < len(ITEMS):
        text += f" · {got}/{len(ITEMS)}개 항목"
    return score, text
