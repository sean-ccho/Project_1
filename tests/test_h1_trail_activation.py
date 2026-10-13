"""scripts/h1_trail_activation.py 테스트 — 사전 등록 판정 규칙·청산 이유 묶기·실험 정의 (백테스트 없이)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from h1_trail_activation import P_BONF, exit_mix, judge  # noqa: E402
from run_hypothesis_ab import VARIANTS  # noqa: E402


def _st(lo: float, p: float) -> dict:
    return {"ΔSharpe": lo + 0.1, "ΔSharpe_하한": lo, "ΔSharpe_상한": lo + 0.2, "p_개선아님": p, "구간_개선": 3, "구간_수": 3}


def test_judge_needs_verdict_and_bonferroni_p():
    assert P_BONF == pytest.approx(0.025)
    assert judge(_st(0.05, 0.01), "채택 후보") == "채택 후보"
    assert judge(_st(0.05, 0.03), "채택 후보") == "관찰"      # 하한 > 0 이지만 p 가 0.025 초과
    assert judge(_st(0.05, 0.01), "운과 구분 안 됨") == "관찰"  # 구간·MDD 조건 탈락
    assert judge(_st(-0.10, 0.40), "운과 구분 안 됨") == "탈락"
    assert judge({}, "") == "탈락"


def test_exit_mix_groups_by_reason_prefix():
    trades = [{"exit_reason": "트레일링(-4.1%)", "return_pct": -0.04}, {"exit_reason": "트레일링(장중,-2.0%)", "return_pct": -0.02},
              {"exit_reason": "목표가(+18.2%)", "return_pct": 0.18}, {"exit_reason": "교체→AAA", "return_pct": 0.01},
              {"exit_reason": "기간종료", "return_pct": 0.0}]
    mix = exit_mix(trades)
    assert mix["트레일링"] == (2, pytest.approx(-0.03)) and mix["목표가"][0] == 1
    assert mix["교체"][0] == 1 and mix["기타"][0] == 1


def test_variants_change_only_trail_activation():
    for key, value in (("H1a", 0.03), ("H1b", 0.05)):
        overrides = VARIANTS[key][1]
        assert set(overrides) == {"EXIT_PARAMS"}
        assert overrides["EXIT_PARAMS"] == {"모멘텀": {"trail_activate_pct": value}, "바닥반등": {"trail_activate_pct": value}}
