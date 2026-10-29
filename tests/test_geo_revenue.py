"""SEC 10-K 미국 매출 비중 (data.geo_revenue) · 종목 4칸 성격 (data.industry_style) — 네트워크 없이 작은 가짜 XBRL 로."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.geo_revenue import GeoRow, is_foreign_region, member_kind, parse_instance, read_csv, write_csv  # noqa: E402
from data.industry_style import INDUSTRY_STYLE, stock_style, style_text  # noqa: E402

END = "2025-12-31"
RFC = "RevenueFromContractWithCustomerExcludingAssessedTax"


def _ctx(cid: str, dims: list[tuple[str, str]] = (), start: str = "2025-01-01", end: str = END) -> str:
    seg = "".join(f'<xbrldi:explicitMember dimension="{a}">{m}</xbrldi:explicitMember>' for a, m in dims)
    seg = f"<xbrli:segment>{seg}</xbrli:segment>" if dims else ""
    return (f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier>{seg}'
            f"</xbrli:entity><xbrli:period><xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}"
            f"</xbrli:endDate></xbrli:period></xbrli:context>")


def _doc(contexts: list[str], facts: list[tuple[str, str, str]]) -> bytes:
    body = "".join(contexts) + "".join(f'<us-gaap:{c} contextRef="{ref}" unitRef="usd">{v}</us-gaap:{c}>'
                                       for c, ref, v in facts)
    return ('<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi" '
            f'xmlns:us-gaap="http://fasb.org/us-gaap/2025">{body}</xbrli:xbrl>').encode()


GEO = "srt:StatementGeographicalAxis"


def test_member_kinds():
    assert member_kind("country:US") == "us" and member_kind("aapl:UnitedStatesMember") == "us"
    assert member_kind("mcd:USMarketMember") == "us" and member_kind("x:DomesticMember") == "us"
    assert member_kind("us-gaap:NonUsMember") == "foreign" and member_kind("c:InternationalMember") == "foreign"
    for m in ("srt:NorthAmericaMember", "x:UnitedStatesAndCanadaMember", "srt:AmericasMember", "country:CN"):
        assert member_kind(m) is None
    assert is_foreign_region("country:CN") and is_foreign_region("srt:EuropeMember")
    assert is_foreign_region("goog:AmericasExcludingUnitedStatesMember") and is_foreign_region("fis:NonNorthAmericaMember")
    for m in ("country:US", "srt:NorthAmericaMember", "dhi:SouthwestMember", "x:UnitedStatesAndCanadaMember"):
        assert not is_foreign_region(m)


def test_us_member_share_and_wrong_period_ignored():
    xml = _doc([_ctx("t"), _ctx("us", [(GEO, "country:US")]), _ctx("cn", [(GEO, "country:CN")]),
                _ctx("old", [(GEO, "country:US")], "2024-01-01", "2024-12-31"),
                _ctx("q4", [(GEO, "country:US")], "2025-10-01", END)],          # 분기 — 1년이 아님
               [(RFC, "t", "416"), (RFC, "us", "152"), (RFC, "cn", "64"), (RFC, "old", "999"), (RFC, "q4", "50")])
    got = parse_instance(xml, END)
    assert got["method"] == "geo_us" and abs(got["us_share"] - 152 / 416) < 1e-12 and got["member"] == "country:US"


def test_foreign_member_other_concept_total_and_concentration_dims():
    citi = _doc([_ctx("t"), _ctx("na", [(GEO, "srt:NorthAmericaMember")]), _ctx("intl", [(GEO, "c:InternationalMember")])],
                [("Revenues", "t", "85"), ("Revenues", "na", "44"), ("Revenues", "intl", "42")])
    got = parse_instance(citi, END)
    assert got["method"] == "geo_foreign" and abs(got["us_share"] - (1 - 42 / 85)) < 1e-12
    googl = _doc([_ctx("t"), _ctx("us", [(GEO, "country:US")])],                # 지역 값은 계약 매출, 합계는 Revenues 만
                 [("Revenues", "t", "403"), (RFC, "us", "194")])
    assert abs(parse_instance(googl, END)["us_share"] - 194 / 403) < 1e-12
    conc = [("us-gaap:ConcentrationRiskByBenchmarkAxis", "us-gaap:RevenueFromContractWithCustomerMember"),
            ("us-gaap:ConcentrationRiskByTypeAxis", "us-gaap:GeographicConcentrationRiskMember")]
    qcom = _doc([_ctx("t"), _ctx("us", [(GEO, "country:US"), *conc])], [("Revenues", "t", "44"), ("Revenues", "us", "10.5")])
    assert parse_instance(qcom, END)["method"] == "geo_us"
    pct = _doc([_ctx("t"), _ctx("p", [(GEO, "country:US"), *conc])],             # 퍼센트만 (TSN)
               [("Revenues", "t", "53"), ("ConcentrationRiskPercentage1", "p", "0.95")])
    got = parse_instance(pct, END)
    assert got["method"] == "geo_pct" and got["us_share"] == 0.95


def test_segment_regions_none_and_bad_ratio():
    seg = "us-gaap:StatementBusinessSegmentsAxis"
    mcd = _doc([_ctx("t"), _ctx("u", [(seg, "mcd:USMarketMember")])], [("Revenues", "t", "26"), ("Revenues", "u", "10.5")])
    assert parse_instance(mcd, END)["method"] == "segment_us"
    klac = _doc([_ctx("t"), *[_ctx(c, [(GEO, f"country:{c}")]) for c in ("CN", "TW", "KR")], _ctx("eu", [(GEO, "srt:EuropeMember")])],
                [(RFC, "t", "100"), (RFC, "CN", "30"), (RFC, "TW", "25"), (RFC, "KR", "13"), (RFC, "eu", "8")])
    got = parse_instance(klac, END)
    assert got["method"] == "geo_regions" and abs(got["foreign_min"] - 0.76) < 1e-12   # 안 겹치면 다 더함
    ctsh = _doc([_ctx("t"), *[_ctx(k, [(GEO, m)]) for k, m in (("na", "srt:NorthAmericaMember"), ("gb", "country:GB"),
                                                                ("eu", "srt:EuropeMember"), ("ex", "x:EuropeExcludingUnitedKingdomMember"))]],
                [(RFC, "t", "100"), (RFC, "na", "74"), (RFC, "gb", "10"), (RFC, "eu", "19"), (RFC, "ex", "9")])
    assert abs(parse_instance(ctsh, END)["foreign_min"] - 0.19) < 1e-12              # 겹침(합 112) → 가장 큰 하나
    conc = [("us-gaap:ConcentrationRiskByBenchmarkAxis", "us-gaap:RevenueFromContractWithCustomerMember"),
            ("us-gaap:ConcentrationRiskByTypeAxis", "us-gaap:GeographicConcentrationRiskMember")]
    crm = _doc([_ctx("t"), _ctx("am", [(GEO, "srt:AmericasMember")]), _ctx("eu", [(GEO, "srt:EuropeMember")]),
                _ctx("p", [(GEO, "country:US"), *conc])],
               [(RFC, "t", "41.5"), (RFC, "am", "27.2"), (RFC, "eu", "14.3"), ("ConcentrationRiskPercentage1", "p", "0.93")])
    assert parse_instance(crm, END)["method"] == "geo_regions"                       # 93% 는 지역 값(해외 34%)과 안 맞음
    mchp = _doc([_ctx("t"), _ctx("p", [(GEO, "us-gaap:NonUsMember"), *conc])],
                [(RFC, "t", "4.4"), ("ConcentrationRiskPercentage1", "p", "0.75")])
    assert abs(parse_instance(mchp, END)["us_share"] - 0.25) < 1e-12
    dhi = _doc([_ctx("t"), _ctx("sw", [(GEO, "dhi:SouthwestMember")])], [("Revenues", "t", "34"), ("Revenues", "sw", "9")])
    assert parse_instance(dhi, END)["method"] == "none"                           # 미국 안 지역뿐 → 지역 공시 없음과 같음
    assert parse_instance(_doc([_ctx("t")], [("Revenues", "t", "5")]), END)["method"] == "none"
    assert parse_instance(_doc([_ctx("t")], []), END)["method"] == "no_revenue"
    bad = _doc([_ctx("t"), _ctx("us", [(GEO, "country:US")])], [(RFC, "t", "10"), (RFC, "us", "50")])
    assert parse_instance(bad, END)["method"] == "none"                           # 미국 > 합계 → 안 받음


def test_csv_round_trip(tmp_path):
    rows = [GeoRow(ticker="AAPL", cik=320193, form="10-K", fy_end="2025-09-27", accession="a", us_share=0.36474,
                   method="geo_us", us_rev=152.0, total_rev=416.0),
            GeoRow(ticker="CL", method="geo_regions", foreign_min=None), GeoRow(ticker="TU", method="foreign_filer")]
    write_csv(tmp_path / "g.csv", rows)
    back = read_csv(tmp_path / "g.csv")
    assert back["AAPL"].us_share == 0.3647 and back["AAPL"].cik == 320193 and back["CL"].foreign_min is None
    assert read_csv(tmp_path / "없음.csv") == {}


def test_stock_style_rules():
    assert all(v in {("cyclical", "domestic"), ("cyclical", "global"), ("defensive", "domestic"), ("defensive", "global")}
               for v in INDUSTRY_STYLE.values()) and len(INDUSTRY_STYLE) == 145
    sec = lambda share, **kw: GeoRow(ticker="X", us_share=share, method="geo_us", fy_end="2025-12-31", **kw)  # noqa: E731
    assert stock_style("Software - Application", sec(0.70))["region"] == "domestic"     # 70% 경계는 내수
    s = stock_style("Software - Application", sec(0.69))
    assert s == {"cycle": "defensive", "region": "global", "us_share": 0.69, "foreign_min": None, "src": "sec",
                 "fy": "2025-12-31"}
    assert stock_style("Banks - Regional", GeoRow(ticker="X", method="none"))["src"] == "sec_none"
    assert stock_style("Tobacco", GeoRow(ticker="X", method="geo_regions", foreign_min=0.42))["region"] == "global"
    assert stock_style("Waste Management", GeoRow(ticker="X", method="geo_regions", foreign_min=0.01))["region"] == "domestic"
    cl = stock_style("Household & Personal Products", GeoRow(ticker="X", method="geo_regions"))   # 크기 모름 → 업종
    assert cl["region"] == "global" and cl["src"] == "industry"
    assert stock_style("REIT - Industrial", GeoRow(ticker="X", method="geo_regions"))["region"] == "domestic"
    assert stock_style("Telecom Services", GeoRow(ticker="X", method="foreign_filer"))["src"] == "foreign"
    est = stock_style("Semiconductors", GeoRow(ticker="X", method="no_revenue"))
    assert est["region"] == "global" and est["src"] == "industry" and est["us_share"] is None
    assert stock_style(None, None) is None and stock_style("Railroads", None)["src"] == "industry"
    assert stock_style(None, sec(0.9)) == {"cycle": None, "region": "domestic", "us_share": 0.9, "foreign_min": None,
                                          "src": "sec", "fy": "2025-12-31"}
    assert style_text(stock_style("Railroads", None)) == ("경기순환", "내수 (추정)", "")
    assert style_text(stock_style("Software - Application", sec(0.51))) == ("경기방어", "글로벌", "51%")
