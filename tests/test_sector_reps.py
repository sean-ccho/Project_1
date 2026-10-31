"""섹터별 대표 기업 (screener.sector_reps) — 시트 [섹터 대표]."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.industry_style import INDUSTRY_STYLE  # noqa: E402
from screener.sector_reps import SECTOR_CYCLE, SECTORS, sector_reps, sector_reps_frame  # noqa: E402


def _info(sector, region, cap, src="sec", us=0.5, cycle="cyclical", tv="NASDAQ:X", ret=0.01):
    return {"sector": sector, "company": f"{sector} co", "price": 10.0, "tv": tv, "ret_20d": ret,
            "fundamentals": {"market_cap": cap},
            "style": {"cycle": cycle, "region": region, "us_share": us, "src": src, "fy": "2025-12-31"}}


def test_top_n_per_region_and_sector_in_sheet_order():
    items = {f"T{i}": _info("Technology", "global", 100 - i) for i in range(6)}
    items |= {"BANK": _info("Financial Services", "domestic", 50, src="sec_none", us=None, tv="NYSE:BANK"),
              "TEL": _info("Communication Services", "global", 70),
              "NOCAP": _info("Technology", "global", None), "NOSTYLE": {"sector": "Technology", "fundamentals": {"market_cap": 999}},
              "ETF": _info("Unknown", "global", 1e12)}
    rows = sector_reps(items, top=4)
    assert [r["ticker"] for r in rows] == ["TEL", "T0", "T1", "T2", "T3", "BANK"]   # 글로벌 먼저, 섹터는 시트 순서
    sheet = sector_reps_frame(items, top=4)
    bank = sheet.set_index("티커").loc["BANK"]
    assert bank["구분"] == "내수 매출 중심" and bank["섹터"] == "Financials (금융)" and bank["거래소"] == "NYSE"
    assert bank["미국 매출 비중"] == "추정" and bank["경기 성격"] == "경기순환" and bank["1달 수익"] == "+1.0%"
    assert bank["섹터 성격"] == "경기순환 · 손해보험 · 보험중개는 방어"
    assert sheet.set_index("티커").loc["T0", "미국 매출 비중"] == "50%"
    assert sector_reps_frame({}).empty and list(sector_reps_frame({}).columns)[0] == "구분"


def test_sector_cycle_matches_industry_table():
    """섹터 문구의 예외 업종이 업종 표와 맞는지 — 표를 바꾸면 문구도 같이."""
    assert set(SECTOR_CYCLE) == {s for s, _ in SECTORS}
    cyc = {k: v[0] for k, v in INDUSTRY_STYLE.items()}
    for ind in ("Telecom Services", "Restaurants", "Waste Management", "Insurance - Property & Casualty", "Insurance Brokers"):
        assert cyc[ind] == "defensive"
    for ind in ("Internet Content & Information", "Software - Application", "Software - Infrastructure", "Travel Services",
                "Resorts & Casinos", "Insurance - Life", "Banks - Diversified", "Oil & Gas Integrated"):
        assert cyc[ind] == "cyclical"
