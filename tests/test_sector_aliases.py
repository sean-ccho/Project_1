"""섹터명 yfinance → GICS 매핑과 강한 섹터 판정 테스트 (실거래 buy_signal 차단 버그)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener.config import SECTOR_ETFS
from screener.sector_rotation import SECTOR_ALIASES, is_strong_or_unknown, to_gics


def test_aliases_map_into_etf_keys():
    assert set(SECTOR_ALIASES.values()) <= set(SECTOR_ETFS)


def test_to_gics():
    assert to_gics("Technology") == "Information Technology"
    assert to_gics("Energy") == "Energy"
    assert to_gics(None) == "Unknown" and to_gics(float("nan")) == "Unknown" and to_gics("") == "Unknown"


def test_is_strong_or_unknown():
    strong = {"Information Technology"}
    assert is_strong_or_unknown("Technology", strong)            # 예전엔 False → buy_signal 차단
    assert is_strong_or_unknown("Information Technology", strong)
    assert is_strong_or_unknown("Unknown", strong)
    assert not is_strong_or_unknown("Healthcare", strong)


def test_fill_unknown_sectors_and_mark_strong():
    import pandas as pd
    from screener.sector_rotation import fill_unknown_sectors, mark_strong_sectors

    sectors = pd.Series(["Unknown", "Technology", None, "Unknown"])
    tickers = pd.Series(["AAPL", "MSFT", "JPM", "ZZZZ_DELISTED"])
    filled = fill_unknown_sectors(sectors, tickers)
    assert filled.tolist() == ["Information Technology", "Information Technology", "Financials", "Unknown"]
    assert mark_strong_sectors(filled, {"Financials"}).tolist() == [False, False, True, True]
