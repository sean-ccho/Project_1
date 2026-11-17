"""scripts/published_rules_backtest.py — 공개 규칙 5개의 지표 · 고르기 (2026-11-16)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import published_rules_backtest as pr  # noqa: E402


def _yr(**kw) -> dict:
    base = {"end": "2019-12-31", "rev": 1000.0, "gp": 400.0, "ni": 100.0, "ocf": 150.0, "fcf": 100.0, "ebit": 160.0,
            "ta": 2000.0, "ca": 600.0, "cl": 300.0, "eq": 1000.0, "ppe": 700.0, "shares": 100.0, "cash": 100.0,
            "debt": 400.0, "ltd": 350.0, "cur_debt": 50.0, "div": 30.0, "rep": 50.0, "iss": 10.0,
            "debt_iss": None, "debt_rep": None}
    base.update(kw)
    return base


def _improving() -> dict:
    """세 해 내내 좋아지는 회사 — F = 9."""
    return {"t": _yr(ni=120.0, ocf=180.0, gp=430.0, rev=1050.0, ltd=300.0, ca=650.0, shares=99.0, ta=2000.0),
            "p": _yr(ni=100.0, ocf=150.0, gp=400.0, rev=1000.0, ltd=350.0, ca=600.0, shares=100.0, ta=2000.0),
            "pp": _yr(ta=2000.0)}


def test_piotroski_signals_and_missing_rule():
    f = _improving()
    score, sig = pr.piotroski(f)
    assert score == 9 and set(sig) == set(pr.PIO_SIGNALS)
    f["t"]["gp"] = None                                  # 매출총이익 없음 → 8개 계산, 빠진 신호 0점
    score, sig = pr.piotroski(f)
    assert score == 8 and "dmargin" not in sig
    f["t"]["shares"] = None                              # 둘 빠지면 점수 없음
    assert pr.piotroski(f)[0] is None
    g = _improving()
    g["t"]["ni"], g["t"]["ocf"] = -10.0, 5.0             # 적자 · 발생액은 좋음
    _, s = pr.piotroski(g)
    assert not s["roa"] and s["cfo"] and s["accrual"] and not s["droa"]
    z = _improving()
    for k in ("t", "p", "pp"):
        z[k]["ltd"] = 0.0                                # 장기 차입금이 두 해 모두 0 → 레버리지 신호 1
    assert pr.piotroski(z)[1]["dlever"]


def test_mf_metrics_excess_cash_and_short_debt_out_of_working_capital():
    t = _yr()
    m = pr.mf_metrics(t, mcap_fc=1000.0)
    nwc = (600 - 100) - (300 - 50)
    assert m["roc"] == pytest.approx(160 / (nwc + 700))
    assert m["ey"] == pytest.approx(160 / (1000 + 400 - 100))
    assert pr.mf_metrics(_yr(ca=100.0, cl=500.0), 1000.0)["roc"] == pytest.approx(160 / 700)   # 음수 운전자본 → 0
    assert pr.mf_metrics(_yr(ebit=-5.0), 1000.0) is None
    assert pr.mf_metrics(_yr(cash=5000.0), 1000.0) is None                                     # EV ≤ 0
    assert pr.am_metric(_yr(), 1000.0) == pytest.approx(160 / 1300)


def test_shy_metric_debt_paydown_and_quality_guard():
    f = {"t": _yr(debt_rep=80.0, debt_iss=20.0), "p": _yr()}
    assert pr.shy_metric(f, 1000.0) == pytest.approx((30 + 50 - 10 + 60) / 1000)
    f = {"t": _yr(debt=300.0), "p": _yr(debt=400.0)}                                           # 현금흐름표 없음 → 차입금 감소
    assert pr.shy_metric(f, 1000.0) == pytest.approx((30 + 50 - 10 + 100) / 1000)
    assert pr.shy_metric({"t": _yr(fcf=-1.0), "p": _yr()}, 1000.0) is None
    assert pr.shy_metric({"t": _yr(ni=-1.0), "p": _yr()}, 1000.0) is None


def _rec(t: str, **kw) -> dict:
    r = {"ticker": t, "sector": "Industrials", "industry": "X", "mf": None, "am": None, "bm": None, "f": None,
         "vmq": {k: None for k in ("bp", "ep", "cfp", "mom", "gpoa", "roe", "roa", "cfoa", "gmar", "acc", "lev", "bab")},
         "shy": None}
    r.update(kw)
    return r


def test_select_mf_rank_sum_and_utilities_out(monkeypatch):
    monkeypatch.setattr(pr, "N_PICK", 2)
    recs = [_rec("A", mf={"ey": 0.20, "roc": 0.10}), _rec("B", mf={"ey": 0.10, "roc": 0.90}),
            _rec("C", mf={"ey": 0.15, "roc": 0.95}), _rec("U", sector="Utilities", mf={"ey": 0.9, "roc": 0.9})]
    picks, pool = pr.select_rule("MF", recs)
    assert "U" not in pool and len(pool) == 3
    assert picks == ["C", "A"]          # 순위 합 (1 = 최고): C 2+1=3, A 1+3=4, B 3+2=5
    tie = [_rec("A", mf={"ey": 0.20, "roc": 0.10}), _rec("B", mf={"ey": 0.10, "roc": 0.20})]
    assert pr.select_rule("MF", tie)[0] == ["A", "B"]          # 합이 같으면 이익수익률 높은 쪽
    picks_am, _ = pr.select_rule("AM", [_rec("A", am=0.2), _rec("B", am=0.3), _rec("U", sector="Utilities", am=0.9)])
    assert picks_am == ["B", "A"]


def test_select_pio_high_bm_quintile_and_fill(monkeypatch):
    monkeypatch.setattr(pr, "PIO_MIN", 3)
    recs = [_rec(f"T{i:02d}", bm=1.0 - i * 0.01, f=(9 if i in (0, 3) else 5)) for i in range(25)]
    picks, pool = pr.select_rule("PIO", recs)
    assert pool == [f"T{i:02d}" for i in (0, 3, 1, 2, 4)]    # 상위 20% = 5개, F 순 → B/M 순
    assert picks == ["T00", "T03", "T01"]                     # F ≥ 8 이 2개뿐 → 3개까지 채움
    monkeypatch.setattr(pr, "PIO_MIN", 2)
    assert pr.select_rule("PIO", recs)[0] == ["T00", "T03"]


def test_select_vmq_needs_momentum_and_combines_three_ranks(monkeypatch):
    monkeypatch.setattr(pr, "N_PICK", 1)
    q = dict(gpoa=0.5, roe=0.3, roa=0.1, cfoa=0.1, gmar=0.4, acc=0.0, lev=-0.1, bab=-1.0)
    good = _rec("G", vmq=dict(bp=0.9, ep=0.10, cfp=0.12, mom=0.30, **q))
    bad = _rec("B", vmq=dict(bp=0.1, ep=0.01, cfp=0.02, mom=-0.20, **{k: v - 0.05 for k, v in q.items()}))
    nomom = _rec("N", vmq=dict(bp=2.0, ep=0.5, cfp=0.5, mom=None, **q))
    picks, pool = pr.select_rule("VMQ", [good, bad, nomom])
    assert picks == ["G"] and "N" not in pool


def test_momentum_twelve_minus_one():
    idx = pd.date_range("2018-01-31", "2020-12-31", freq="ME")
    df = pd.DataFrame({"adj": np.arange(1, len(idx) + 1, dtype=float), "close": 1.0, "split": 0.0}, index=idx)
    d0 = pd.Timestamp("2020-05-31")
    i = list(idx).index(d0)
    assert pr.momentum(df, d0) == pytest.approx((i - 1 + 1) / (i - 12 + 1) - 1)


def test_pct_rank_ties_and_none():
    assert pr.pct_rank([1.0, None, 3.0, 3.0]) == [pytest.approx(1 / 3), None, pytest.approx(5 / 6), pytest.approx(5 / 6)]
