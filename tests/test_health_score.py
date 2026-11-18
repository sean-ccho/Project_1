"""헬스체크 점수 (screener/health_score.py) — 받은 항목만으로 환산 · 5개 미만 N/A · 8개 다 있으면 예전 점수와 같음."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screener.health_score import ITEMS, MIN_ITEMS, calculate_health_score, health_grade  # noqa: E402

NAN = float("nan")

# 8개 전부 통과하는 값 (기준: ROE 8% · 부채 200% · 매출 3% · 순이익률 5% · 이익 2% · 유동 1 · FCF 0 · 기관 60%)
GOOD = {"fund_roe": 0.20, "fund_debt_to_equity": 50.0, "fund_revenue_growth": 0.10, "fund_profit_margin": 0.30,
        "fund_earnings_growth": 0.10, "fund_current_ratio": 1.5, "fund_free_cashflow": 1e9,
        "fund_institutional_holders_pct": 0.75}


def _old_score(row: dict) -> int | None:
    """2026-11-17 까지의 방식 — 빠진 항목은 0점, 4개 미만 N/A (8개 다 있을 때 새 방식과 같아야 한다)."""
    total = got = 0
    for col, thr, pts, op in ITEMS:
        v = row.get(col)
        if v is None or v != v:
            continue
        got += 1
        if (op == "gt" and v > thr) or (op == "lt" and v < thr):
            total += pts
    return None if got < 4 else total


def test_all_eight_items_same_as_before() -> None:
    rows = [GOOD,
            {**GOOD, "fund_roe": 0.05, "fund_free_cashflow": -1.0},           # 100 - 20 - 10 = 70
            {**GOOD, "fund_debt_to_equity": 250.0, "fund_institutional_holders_pct": 0.3},
            {k: (-1.0 if k != "fund_debt_to_equity" else 999.0) for k in GOOD}]  # 다 탈락 → 0
    for row in rows:
        score, text = calculate_health_score(pd.Series(row))
        assert score == _old_score(row)
        assert "개 항목" not in text  # 8개 다 받으면 항목 수를 안 붙인다
    assert calculate_health_score(pd.Series(GOOD)) == (100, "100 (A+) · 매출+10% 마진+30% 이익+10%")


def test_bank_like_missing_items_are_not_failures() -> None:
    """JPM 모양 — Yahoo 가 부채비율 · 유동비율 · 잉여현금흐름을 안 준다. 받은 5개 다 통과 → 100 (예전 65 C)."""
    bank = {**GOOD, "fund_debt_to_equity": NAN, "fund_current_ratio": None, "fund_free_cashflow": NAN}
    score, text = calculate_health_score(pd.Series(bank))
    assert score == 100 and text.startswith("100 (A+) · ") and text.endswith(" · 5/8개 항목")
    assert _old_score(bank) == 65


def test_partial_pass_is_rounded_share_of_received_items() -> None:
    """자본 마이너스 회사 모양 (ROE · 부채비율 없음, 만점 65) — 유동비율만 탈락: 55/65 = 84.6 → 85 (A)."""
    row = {**GOOD, "fund_roe": NAN, "fund_debt_to_equity": NAN, "fund_current_ratio": 0.87}
    score, text = calculate_health_score(pd.Series(row))
    assert score == 85 and "(A)" in text and text.endswith("6/8개 항목")


def test_fewer_than_five_items_is_na() -> None:
    four = {k: GOOD[k] for k in list(GOOD)[:4]}
    assert calculate_health_score(pd.Series(four)) == (None, "N/A")
    five = {k: GOOD[k] for k in list(GOOD)[:MIN_ITEMS]}
    assert calculate_health_score(pd.Series(five))[0] == 100
    assert calculate_health_score(pd.Series({"fund_roe": "abc", **{k: NAN for k in list(GOOD)[1:]}})) == (None, "N/A")
    assert calculate_health_score({}) == (None, "N/A")  # dict 도 된다 (시트 · 전체분석은 Series)


def test_grades() -> None:
    assert [health_grade(s) for s in (100, 90, 89, 80, 70, 60, 50, 49, 0)] == ["A+", "A+", "A", "A", "B", "C", "D", "F", "F"]
