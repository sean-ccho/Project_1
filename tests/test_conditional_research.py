"""scripts/conditional_research.py 테스트 — 조건 변수·날짜별 분위."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from conditional_research import _bucket, conditioning_vars


def test_conditioning_vars():
    idx = pd.bdate_range("2024-01-01", periods=80)
    close = pd.Series(100.0, index=idx)
    close.iloc[-1] = 110.0
    opens = close.shift(1).fillna(100.0)
    opens.iloc[-1] = 104.0  # 마지막 날 4% 갭
    vol = pd.Series(1000.0, index=idx)
    vol.iloc[-5:] = 3000.0
    data = {("Close", "AAA"): close, ("Open", "AAA"): opens, ("Volume", "AAA"): vol}
    ohlcv = pd.DataFrame(data)
    ohlcv.columns.names = ["Price", "Ticker"]
    last = conditioning_vars(ohlcv).set_index("date").loc[idx[-1]]
    assert math.isclose(last["ret_5d_c"], 0.10)
    assert math.isclose(last["abn_vol"], 3.0)
    assert math.isclose(last["max_gap_5"], 0.04)


def test_bucket_per_date():
    dates = pd.Series(["d1"] * 10 + ["d2"] * 5)
    vals = pd.Series(list(range(10)) + list(range(5)))
    b = _bucket(vals, dates, 5)
    assert b[:10].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert b[10:].tolist() == [1, 2, 3, 4, 5]
