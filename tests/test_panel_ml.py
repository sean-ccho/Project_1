"""scripts/panel_ml.py 테스트 — 누설 피처 제외."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from panel_ml import features


def test_features_exclude_leaky_and_targets():
    panel = pd.DataFrame({"date": [1], "티커": ["A"], "섹터": ["X"], "close": [1.0], "ret_5d": [0.1],
                          "ret_5d_sn": [0.0], "beta_252": [1.0], "fwd_ret_10d": [0.0]})
    assert features(panel) == ["ret_5d", "beta_252"]
