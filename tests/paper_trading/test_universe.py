"""S&P 500 과거 구성종목(universe) 테스트 — 순수 파이썬."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading.universe import Membership, load_membership

CSV = (
    "date,tickers\n"
    '2022-03-01,"AAA,DDD,EEE"\n'
    '2020-01-02,"AAA,BRK.B,CCC"\n'
    '2021-06-01,"AAA,BRK.B,DDD"\n'
)


def _membership(tmp_path: Path) -> Membership:
    p = tmp_path / "m.csv"
    p.write_text(CSV, encoding="utf-8")
    return Membership(load_membership(p))


def test_load_sorts_and_normalizes(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(CSV, encoding="utf-8")
    rows = load_membership(p)
    assert [d for d, _ in rows] == ["2020-01-02", "2021-06-01", "2022-03-01"]
    assert "BRK-B" in rows[0][1]  # yfinance 표기


def test_members_on_date(tmp_path):
    m = _membership(tmp_path)
    assert m.on("2019-12-31") == frozenset()
    assert m.on("2020-01-02") == {"AAA", "BRK-B", "CCC"}
    assert m.on("2021-12-31") == {"AAA", "BRK-B", "DDD"}  # CCC는 빠진 뒤
    assert m.on("2030-01-01") == {"AAA", "DDD", "EEE"}


def test_union_includes_removed_members(tmp_path):
    m = _membership(tmp_path)
    assert m.union_between("2021-01-01", "2022-12-31") == {"AAA", "BRK-B", "CCC", "DDD", "EEE"}
    assert m.union_between("2021-07-01", "2021-12-31") == {"AAA", "BRK-B", "DDD"}


def test_missing_file_explains_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_sp500_membership"):
        load_membership(tmp_path / "none.csv")
