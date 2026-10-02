"""PT-1 Tier 1.5 스위치(H1~H6) 테스트.

각 스위치마다: 기본값이면 기존 동작 그대로인지, 켜면 의도대로 바뀌는지 확인한다.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading import candidate_selector
from paper_trading.candidate_selector import (
    _apply_hard_filters,
    _score_alpha_factor,
    select_best_candidate,
    select_top_candidates,
)
from paper_trading.config_override import config_overrides
from paper_trading.engine import check_sell_conditions, should_replace


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _pos(highest: float, strategy: str = "모멘텀") -> dict:
    return {"ticker": "AAA", "entry_price": 100.0, "highest_price": highest,
            "entry_date": "2026-09-01", "strategy": strategy}


def _passing_row(ticker: str, strategy: str) -> dict:
    """하드 필터를 모두 통과하는 스크리너 행."""
    fit_col = "바닥반등_적합도" if strategy == "바닥반등" else "모멘텀_적합도"
    return {
        "티커": ticker, "전략구분": strategy, fit_col: 7.0, "52주포지션": 0.3,
        "판단": "1. 매수 후보", "RSI": 55.0, "bollinger_pband": 0.5, "5일수익률": 0.02,
        "최근20일평균거래대금": 50_000_000, "days_to_next_earnings": 30, "adx": 30.0, "섹터": "Tech",
    }


def _spy_row(close: float, ema50: float, ema200: float) -> dict:
    return {"티커": "SPY", "현재가격": close, "ema50": ema50, "ema200": ema200}


# ── H1: 트레일링 활성화 문턱 ──────────────────────────────────────────────────

def test_h1_default_trailing_activates_on_any_gain():
    # 고점 103 → 97.5 (-5.3%) : 모멘텀 트레일링 5% 발동, 손절(-10%)은 아님
    sell, reason = check_sell_conditions(_pos(103.0), 97.5, "2026-09-05")
    assert sell and reason.startswith("트레일링")


def test_h1_trailing_waits_until_activation_gain():
    with config_overrides({"EXIT_PARAMS": {"모멘텀": {"trail_activate_pct": 0.05}}}):
        assert check_sell_conditions(_pos(103.0), 97.5, "2026-09-05") == (False, "")
        # 고점이 +5%를 넘은 뒤에는 다시 작동
        sell, reason = check_sell_conditions(_pos(108.0), 102.0, "2026-09-05")
        assert sell and reason.startswith("트레일링")


# ── H4: 교체 ─────────────────────────────────────────────────────────────────

def test_h4_replace_margin_and_switch():
    assert should_replace(0.60, 0.45)            # 기본: 마진 0.10
    with config_overrides({"CCS_REPLACE_MARGIN": 0.20}):
        assert not should_replace(0.60, 0.45)
    with config_overrides({"PT1_REPLACE_ENABLED": False}):
        assert not should_replace(0.99, 0.10)


# ── H2: 허용 전략 ────────────────────────────────────────────────────────────

def test_h2_default_allows_all_strategies():
    df = pd.DataFrame([_passing_row("AAA", "모멘텀"), _passing_row("BBB", "바닥반등")])
    passed, rejections = _apply_hard_filters(df, [])
    assert set(passed["티커"]) == {"AAA", "BBB"}
    assert "허용전략_외" not in rejections


def test_h2_momentum_only():
    df = pd.DataFrame([_passing_row("AAA", "모멘텀"), _passing_row("BBB", "바닥반등")])
    with config_overrides({"CANDIDATE_ALLOWED_STRATEGIES": ["모멘텀"]}):
        passed, rejections = _apply_hard_filters(df, [])
    assert list(passed["티커"]) == ["AAA"]
    assert rejections["허용전략_외"] == 1


# ── Tier 2: 하루 여러 종목 선정 ───────────────────────────────────────────────

def test_top_candidates_k1_matches_best_and_respects_sector_cap(monkeypatch):
    monkeypatch.setattr(candidate_selector, "CANDIDATE_CCS_MIN_NORMAL", -1.0)
    rows = [_passing_row("AAA", "모멘텀"), _passing_row("BBB", "모멘텀"), _passing_row("CCC", "모멘텀")]
    rows[2]["섹터"] = "Energy"
    df = pd.DataFrame(rows)

    best, _ = select_best_candidate(df, [])
    picks, _ = select_top_candidates(df, [], k=1)
    assert [p["ticker"] for p in picks] == [best["ticker"]]

    # Tech 1개 보유 중 + 섹터 한도 2 → 같은 날 Tech는 1개만 더
    picks, _ = select_top_candidates(df, [{"ticker": "ZZZ", "sector": "Tech"}], k=3)
    assert len(picks) == 2
    assert sum(p["sector"] == "Tech" for p in picks) == 1
    assert "CCC" in {p["ticker"] for p in picks}


# ── H3: 약세장 신규 진입 중단 ─────────────────────────────────────────────────

def test_h3_blocks_only_in_bear_when_enabled():
    bear = pd.DataFrame([_spy_row(380.0, 400.0, 420.0), _passing_row("BBB", "바닥반등")])
    _, debug = select_best_candidate(bear, [])
    assert debug["regime"] == "bear"
    assert "H3" not in debug.get("rejection_reason", "")

    with config_overrides({"CANDIDATE_BEAR_BLOCK_NEW": True}):
        cand, debug = select_best_candidate(bear, [])
        assert cand is None
        assert "H3" in debug["rejection_reason"]

        bull = pd.DataFrame([_spy_row(450.0, 430.0, 400.0), _passing_row("BBB", "바닥반등")])
        _, debug = select_best_candidate(bull, [])
        assert "H3" not in debug.get("rejection_reason", "")


# ── H6: 바닥반등 알파 가중치 ─────────────────────────────────────────────────

def _factor_row(strategy: str) -> pd.Series:
    return pd.Series({"팩터_모멘텀": 0.8, "팩터_추세": 0.6, "팩터_거래량": 0.2,
                      "팩터_변동성": -0.4, "팩터_평균회귀": -0.6, "전략구분": strategy})


def test_h6_default_matches_previous_formula():
    raw_bottom = 0.10 * 0.8 + 0.10 * 0.6 + 0.15 * 0.2 + 0.25 * -0.4 + 0.40 * -0.6
    raw_mom = 0.20 * 0.8 + 0.20 * 0.6 + 0.25 * 0.2 + 0.15 * -0.4 + 0.20 * -0.6
    assert _score_alpha_factor(_factor_row("바닥반등")) == pytest.approx((raw_bottom + 1) / 2)
    assert _score_alpha_factor(_factor_row("모멘텀")) == pytest.approx((raw_mom + 1) / 2)


def test_h6_override_changes_only_bottom_strategy():
    before_mom = _score_alpha_factor(_factor_row("모멘텀"))
    h6 = {"바닥반등": {"mom": 0.25, "trend": 0.25, "vol": 0.20, "volat": 0.10, "mr": 0.20}}
    with config_overrides({"CANDIDATE_ALPHA_WEIGHTS": h6}):
        raw = 0.25 * 0.8 + 0.25 * 0.6 + 0.20 * 0.2 + 0.10 * -0.4 + 0.20 * -0.6
        assert _score_alpha_factor(_factor_row("바닥반등")) == pytest.approx((raw + 1) / 2)
        assert _score_alpha_factor(_factor_row("모멘텀")) == pytest.approx(before_mom)
