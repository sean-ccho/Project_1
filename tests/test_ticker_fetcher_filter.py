"""ticker_fetcher 비주식 상품(ETN·채권·펀드·우선주·SPAC) 제거 테스트."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.ticker_fetcher import _parse_nasdaq_listed, _parse_other_listed

OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
VXX|iPath Series B S&P 500 VIX Short-Term Futures ETN|Z|VXX|N|100|N|VXX
HTZ|Hertz Global Holdings, Inc - Common Stock|N|HTZ|N|100|N|HTZ
ABXL|Abacus Global Management, Inc. 9.875% Fixed Rate Senior Notes due 2028|N|ABXL|N|100|N|ABXL
DNP|DNP Select Income Fund, Inc. Common Stock|N|DNP|N|100|N|DNP
AAT|American Assets Trust, Inc. Common Stock|N|AAT|N|100|N|AAT
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
File Creation Time: 1|||||||
"""
NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
GOOG|Alphabet Inc. - Class C Capital Stock|Q|N|N|100|N|N
AIIA|AI Infrastructure Acquisition Corp. Class A Ordinary Shares|G|N|N|100|N|N
FNGD|MicroSectors FANG  Index -3X Inverse Leveraged ETNs due January 8, 2038|G|N|N|100|N|N
PFBC|Preferred Bank - Common Stock|Q|N|N|100|N|N
XPFD|Example Corp - 6.5% Series A Preferred Stock|G|N|N|100|N|N
File Creation Time: 1|||||||
"""


def test_other_listed_drops_non_equity():
    assert _parse_other_listed(OTHER)["ticker"].tolist() == ["HTZ", "AAT"]


def test_nasdaq_listed_drops_spac_and_etn():
    # PFBC 는 이름에 Preferred 가 있지만 보통주 → 남긴다. 진짜 우선주는 뺀다
    assert _parse_nasdaq_listed(NASDAQ)["ticker"].tolist() == ["GOOG", "PFBC"]
