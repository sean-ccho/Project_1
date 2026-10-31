"""ETF 전체 보유 종목 (data.etf_holdings) — 네트워크 없이 작은 가짜 N-PORT · 가짜 SEC 클라이언트로."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.etf_holdings import NameMatcher, build, coverage, norm_name, parse_nport  # noqa: E402

NPORT = """<edgarSubmission><formData><genInfo><repPdDate>2026-06-30</repPdDate></genInfo><invstOrSecs>
<invstOrSec><name>Coca-Cola Co/The</name><cusip>191216100</cusip><pctVal>2.5</pctVal><assetCat>EC</assetCat>
<invCountry>US</invCountry></invstOrSec>
<invstOrSec><name>NXP Semiconductors NV</name><cusip>N6596X109</cusip><pctVal>1.0</pctVal><assetCat>EC</assetCat>
<invCountry>NL</invCountry></invstOrSec>
<invstOrSec><name>AT&amp;T Inc</name><cusip>00206R102</cusip><pctVal>0.5</pctVal><assetCat>EC</assetCat>
<invCountry>US</invCountry></invstOrSec>
<invstOrSec><name>US Treasury Bill</name><cusip>912797</cusip><pctVal>3.0</pctVal><assetCat>DBT</assetCat></invstOrSec>
<invstOrSec><name>Short Future</name><cusip>N/A</cusip><pctVal>-0.1</pctVal><assetCat>EC</assetCat></invstOrSec>
</invstOrSecs></formData></edgarSubmission>"""

SEC_ROWS = [{"cik_str": 21344, "ticker": "KO", "title": "COCA COLA CO"},
            {"cik_str": 1413447, "ticker": "NXPI", "title": "NXP Semiconductors N.V."},
            {"cik_str": 732717, "ticker": "T", "title": "AT&T INC."},
            {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
            {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."}]


def test_norm_name_and_parse():
    assert norm_name("Coca-Cola Co/The") == norm_name("COCA COLA CO") == "COCA COLA"
    assert norm_name("TJX Cos Inc/The") == norm_name("TJX COMPANIES INC /DE/") == "TJX"
    assert norm_name("NXP Semiconductors NV") == norm_name("NXP Semiconductors N.V.")
    assert norm_name("Alphabet Inc., Class C") == "ALPHABET"
    rep, held = parse_nport(NPORT)
    assert rep == "2026-06-30" and [h.name for h in held] == ["Coca-Cola Co/The", "NXP Semiconductors NV", "AT&T Inc"]
    assert abs(held[0].weight - 0.025) < 1e-12 and held[1].country == "NL"     # 채권 · 음수 비중은 뺀다


def test_matcher_prefers_known_ticker_and_universe_names():
    m = NameMatcher(SEC_ROWS, {"BRK-B": "Berkshire Hathaway Inc."}, known={"KO", "NXPI", "T", "GOOG", "BRK-B"})
    assert m.ticker("Coca-Cola Co/The") == "KO" and m.ticker("AT&T Inc") == "T"
    assert m.ticker("Alphabet Inc") == "GOOG"                       # 전체분석에 있는 종류를 고른다
    assert m.ticker("Berkshire Hathaway Inc") == "BRK-B"             # SEC 목록에 없으면 전체분석 이름으로
    assert m.ticker("Exxon Mobil Corp") == "XOM"                     # 이름이 바뀐 회사는 별칭
    assert m.ticker("Unknown Holdings Plc") is None


class FakeClient:
    def __init__(self, acc: str) -> None:
        self.acc, self.docs = acc, 0

    def get_json(self, url: str):
        return {"fields": ["cik", "seriesId", "classId", "symbol"], "data": [[1, "S000001", "C1", "AAA"]]}

    def get_bytes(self, url: str) -> bytes:
        if "browse-edgar" in url:
            return f"<filing-href>https://www.sec.gov/Archives/edgar/data/1/x/{self.acc}-index.htm</filing-href>".encode()
        if url.endswith("-index.htm"):
            return b'<a href="/Archives/edgar/data/1/x/xslFormNPORT-P_X01/primary_doc.xml"></a>' \
                   b'<a href="/Archives/edgar/data/1/x/primary_doc.xml"></a>'
        self.docs += 1
        assert "xsl" not in url
        return NPORT.encode()


def test_build_new_then_same_accession_reuses_rows():
    m = NameMatcher(SEC_ROWS, {}, known={"KO", "NXPI", "T"})
    c = FakeClient("0000000001-26-000001")
    rows, status = build(["AAA", "BBB"], c, m, prev={})
    assert status == {"AAA": "new", "BBB": "no_series"} and len(rows) == 3 and c.docs == 1
    assert [r["ticker"] for r in rows] == ["KO", "NXPI", "T"] and rows[0]["rep_date"] == "2026-06-30"
    cov = coverage(rows)["AAA"]
    assert abs(cov[0] - 0.04) < 1e-9 and abs(cov[1] - 0.04) < 1e-9
    prev = {"AAA": [{**r, "ticker": ""} for r in rows]}
    rows2, status2 = build(["AAA"], c, m, prev=prev)
    assert status2 == {"AAA": "same"} and c.docs == 1 and [r["ticker"] for r in rows2] == ["KO", "NXPI", "T"]
