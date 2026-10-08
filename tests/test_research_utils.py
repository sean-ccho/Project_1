"""scripts/research_utils.py 테스트 — NW t값, Bonferroni 임계, 시도 기록, 개발 구간 자르기."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import research_utils as ru


def test_nw_lag0_equals_simple_t():
    x = np.random.default_rng(0).normal(0.1, 1.0, 500)
    simple_t = x.mean() / (x.std(ddof=0) / math.sqrt(x.size))
    assert math.isclose(ru.nw_tstat(x, 0), simple_t, rel_tol=1e-12)


def test_nw_shrinks_t_for_overlapping_returns():
    # 겹치는 5일 합 수익률 → 양의 자기상관 → NW t가 단순 t보다 작아야 한다
    d = np.random.default_rng(1).normal(0.02, 1.0, 3000)
    overlap = pd.Series(d).rolling(5).sum().dropna().to_numpy()
    assert abs(ru.nw_tstat(overlap, 4)) < abs(ru.nw_tstat(overlap, 0))


def test_nw_short_sample_nan():
    assert math.isnan(ru.nw_tstat(np.ones(10), 0))


def test_bonferroni_z():
    assert math.isclose(ru.bonferroni_z(1), 1.96, abs_tol=0.01)
    assert math.isclose(ru.bonferroni_z(150), 3.59, abs_tol=0.01)


def test_halves_and_yearly():
    s = pd.Series([1.0, 1.0, 3.0, 3.0], index=pd.to_datetime(["2020-01-02", "2020-06-01", "2021-01-04", "2021-06-01"]))
    assert ru.halves_mean(s) == (1.0, 3.0)
    assert ru.yearly_mean(s).to_dict() == {2020: 1.0, 2021: 3.0}


def test_log_trials_and_total(tmp_path, monkeypatch):
    monkeypatch.setattr(ru, "TRIAL_LOG", tmp_path / "TRIAL_LOG.csv")
    ru.log_trials("s1", "fam_a", 10, [{"test_id": "a1", "판정": "등록"}, {"test_id": "a2", "판정": "등록"}])
    ru.log_trials("s1", "fam_b", 5, [{"test_id": "(합계)"}])
    assert ru.total_trials() == 15


def test_load_ohlcv_clips_holdout(tmp_path):
    idx = pd.date_range("2025-09-25", "2025-10-10", freq="B")
    cols = pd.MultiIndex.from_product([["Open", "Close"], ["AAA"]], names=["Price", "Ticker"])
    pd.DataFrame(1.0, index=idx, columns=cols).to_parquet(tmp_path / "o.parquet")
    out = ru.load_ohlcv(tmp_path / "o.parquet")
    assert out.index.max() <= ru.DEV_END
