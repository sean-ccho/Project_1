"""연구소 G군(순환·방어 전환) 테스트 — 12 산업 파일 읽기, 격자·이웃, 상승·하락 때 비중, 미래 정보 없음."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lab_data as ld  # noqa: E402
import lab_families as lf  # noqa: E402
import lab_sector_rotation as sr  # noqa: E402

TEXT = """This file was created using the 202608 CRSP database.
Missing data are indicated by -99.99 or -999.

  Average Value Weighted Returns -- Daily
,NoDur,Durbl,Manuf,Enrgy,Chems,BusEq,Telcm,Utils,Shops,Hlth,Money,Other
19260701,   0.02,  -0.28,  -0.41,   0.57,   0.52,  -0.25,  -0.02,   0.61,  -0.01,   0.97,   0.58,   0.19
19260702,   0.29,   1.07,   0.83,   0.64,   0.97,   0.13,   0.26,   0.47,   0.01,   0.13, -99.99,   0.11

  Average Equal Weighted Returns -- Daily
,NoDur,Durbl,Manuf,Enrgy,Chems,BusEq,Telcm,Utils,Shops,Hlth,Money,Other
19260701,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00
"""


def test_parse_industries_reads_value_weighted_table_only():
    d = ld.parse_french_industries(TEXT)
    assert list(d.columns) == list(ld.IND12) and len(d) == 2
    assert d.iloc[0]["NoDur"] == pytest.approx(0.0002) and np.isnan(d.iloc[1]["Money"])


def test_grid_and_neighbors():
    g = sr.FAM.grid()
    assert len(g) == 180 and len({sr.FAM.key(c) for c in g}) == 180
    keys = {sr.FAM.key(c) for c in g}
    cfg = {"sig": "sma", "n": 200, "on": "CYC", "off": "DEF", "sched": "W"}
    assert len(sr.FAM.neighbors(cfg, keys)) == 4                              # n 150·250, 주기 M·D10
    assert set(sr.CYC) | set(sr.DEF) == set(ld.IND12) and not set(sr.CYC) & set(sr.DEF)


def _universe(n: int = 400, up: bool = True) -> lf.Universe:
    idx = pd.bdate_range("2000-01-03", periods=n)
    rng = np.random.default_rng(0)
    cols = list(ld.IND12) + ["MKT"]
    ret = pd.DataFrame(rng.normal(0.0, 0.002, (n, len(cols))), index=idx, columns=cols)
    ret["MKT"] += 0.002 if up else -0.002
    return lf.Universe(ret, pd.Series(0.0, index=idx))


def test_weights_follow_regime():
    up, down = _universe(up=True), _universe(up=False)
    on_cfg = {"sig": "sma", "n": 100, "on": "CYC", "off": "DEF", "sched": "M"}
    t = sr.FAM.target(up, on_cfg).iloc[-1]
    assert t[sr.CYC].to_numpy() == pytest.approx(np.full(8, 1 / 8)) and t[sr.DEF].sum() == 0 and t["MKT"] == 0
    t = sr.FAM.target(down, on_cfg).iloc[-1]
    assert t[sr.DEF].to_numpy() == pytest.approx(np.full(4, 1 / 4)) and t[sr.CYC].sum() == 0
    cash = sr.FAM.target(down, {**on_cfg, "off": "CASH"}).iloc[-1]
    assert cash.sum() == 0                                                    # 하락 + 현금 → 아무것도 안 듦
    mkt = sr.FAM.target(up, {**on_cfg, "on": "MKT"}).iloc[-1]
    assert mkt["MKT"] == 1.0 and mkt.drop("MKT").sum() == 0
    assert sr.FAM.target(up, on_cfg).iloc[:252].isna().all().all()            # 기록 253일 전엔 결정 안 함


def test_relative_strength_regime():
    u = _universe()
    u.ret.loc[:, sr.CYC] += 0.001                                              # 순환주가 계속 강함
    u2 = lf.Universe(u.ret, u.rf)
    on = sr.regime(u2, "rel", 63, sr.CYC, sr.DEF, "MKT")
    assert bool(on.iloc[-1]) and not bool(on.iloc[10])                         # 앞쪽은 63일 전 값이 없어 False


@pytest.mark.parametrize("cfg", [{"sig": "sma", "n": 150, "on": "CYC", "off": "DEF", "sched": "M"},
                                 {"sig": "rel", "n": 126, "on": "MKT", "off": "DEF", "sched": "W"},
                                 {"sig": "mom", "n": 252, "on": "CYC", "off": "CASH", "sched": "D5"}])
def test_no_lookahead(cfg):
    u = _universe(600)
    cut = u.ret.index[450]
    full = sr.FAM.target(u, cfg).loc[:cut]
    part = sr.FAM.target(u.truncate(cut), cfg)
    pd.testing.assert_frame_equal(full, part, check_freq=False)
