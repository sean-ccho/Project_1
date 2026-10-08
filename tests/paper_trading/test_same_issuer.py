"""같은 회사 다른 클래스주 중복 매수 방지 테스트 (GOOG 보유 중 GOOGL 등)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading.candidate_selector import _issuer_key, _same_issuer_mask


def test_issuer_key():
    assert _issuer_key("Alphabet Inc. - Class C Capital Stock") == "alphabet"
    assert _issuer_key("Alphabet Inc.") == "alphabet"
    assert _issuer_key("Fox Corporation - Class A Common Stock") == "fox"
    assert _issuer_key(None) == "" and _issuer_key(float("nan")) == ""


def test_same_issuer_by_name_and_by_class_map():
    full = pd.DataFrame({"티커": ["GOOG", "GOOGL", "MSFT", "NWSA", "XYZ1"],
                         "회사": ["Alphabet Inc.", "Alphabet Inc.", "Microsoft Corp", None, "Xyz Holdings Inc."]})
    df = full[full["티커"] != "GOOG"]  # 보유 종목 행이 앞 필터에서 빠진 상황
    same = _same_issuer_mask(df, full, {"GOOG"})
    assert same.tolist() == [True, False, False, False]
    # 회사명이 없어도 클래스주 묶음으로 잡는다 (NWS 보유 → NWSA 제외)
    same2 = _same_issuer_mask(df, full, {"NWS"})
    assert same2.tolist() == [False, False, True, False]


def test_arrow_string_columns_and_empty():
    full = pd.DataFrame({"티커": pd.Series(["GOOGL", "MSFT"], dtype="string[pyarrow]"),
                         "회사": pd.Series(["Alphabet Inc.", None], dtype="string[pyarrow]")})
    out = _same_issuer_mask(full, full, {"AAPL"})  # 겹치는 회사 없음 (백테스트에서 터지던 경우)
    assert out.dtype == bool and out.tolist() == [False, False]
    assert _same_issuer_mask(full.iloc[:0], full, {"GOOG"}).tolist() == []
