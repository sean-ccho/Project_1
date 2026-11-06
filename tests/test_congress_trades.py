"""국회의원 거래 신고 (data.congress_trades, 2026-11-06) — 하원 색인 · PTR 글자 · 의원 정보 · 쌓기 · 출력. 네트워크 없이."""
import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import congress_trades as ct  # noqa: E402

N = "\x00"
# pypdf 가 실제 PTR 에서 뽑는 꼴 (이름표 글자가 NUL, 금액 줄바꿈, 두 번째 페이지 머리줄)
PTR_TEXT = f"""P{N*7} T{N*10} R{N*5}
Clerk of the House of Representatives • Legislative Resource Center • B81 Cannon Building • Washington, DC 20515
Name: Hon. Jonathan Jackson
State/District:IL01
ID Owner Asset Transaction
Type
Date Notification
Date
Amount Cap.
Gains >
$200?
JT Datadog, Inc. - Class A Common
Stock (DDOG) [ST]
P 09/28/202610/01/2026$50,001 -
$100,000
F{N*5} S{N*5}: New
S{N*10} O{N}: Morgan Stanley Trust Account
US Treasury Bill [GS] P 09/28/202610/01/2026$15,001 - $50,000
F{N*5} S{N*5}: New
SP Berkshire Hathaway Inc. New Common Stock (BRK.B) [ST]
S (partial) 09/04/202610/01/2026$1,000,001 - $5,000,000
F{N*5} S{N*5}: New
Filing ID #20035551
ID Owner Asset Transaction
Type
Date Notification
Date
Amount Cap.
Gains >
$200?
Apple Inc. (AAPL) [OP] P 09/30/202610/05/2026Over $50,000,000
D{N*9}: Call options; strike $250; expires 01/15/2027
* For the complete list of asset type abbreviations, please visit https://fd.house.gov/reference/asset-type-codes.aspx.
"""


def _index_zip(members: list[dict]) -> bytes:
    xml = "<FinancialDisclosure>" + "".join(
        "<Member>" + "".join(f"<{k}>{v}</{k}>" for k, v in m.items()) + "</Member>" for m in members) + "</FinancialDisclosure>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2026FD.xml", xml)
        z.writestr("2026FD.txt", "ignored")
    return buf.getvalue()


INDEX = [
    {"Prefix": "Hon.", "Last": "Jackson", "First": "Jonathan", "FilingType": "P", "StateDst": "IL01", "Year": "2026",
     "FilingDate": "10/19/2026", "DocID": "20035551"},
    {"Prefix": "", "Last": "Harshbarger", "First": "Diana", "FilingType": "P", "StateDst": "TN01", "Year": "2026",
     "FilingDate": "11/2/2026", "DocID": "9116389"},                       # 스캔 신고
    {"Prefix": "", "Last": "Old", "First": "Al", "FilingType": "P", "StateDst": "CA12", "Year": "2026",
     "FilingDate": "1/2/2025", "DocID": "20000001"},                       # 1년 넘음
    {"Prefix": "", "Last": "Annual", "First": "Bo", "FilingType": "C", "StateDst": "TX02", "Year": "2026",
     "FilingDate": "5/13/2026", "DocID": "10078016"},                      # PTR 아님
]

LEGIS = json.dumps([
    {"id": {"bioguide": "J000309"}, "name": {"official_full": "Jonathan L. Jackson"},
     "terms": [{"type": "rep", "state": "IL", "district": 1, "party": "Democrat"}],
     "leadership_roles": [{"title": "House Democratic Caucus Vice Chair", "chamber": "house", "start": "2025-01-03"}]},
    {"id": {"bioguide": "H001086"}, "name": {"official_full": "Diana Harshbarger"},
     "terms": [{"type": "rep", "state": "TN", "district": 1, "party": "Republican"}]},
    {"id": {"bioguide": "S000001"}, "name": {"official_full": "Some Senator"},
     "terms": [{"type": "sen", "state": "TN", "party": "Republican"}]},
])


def test_parse_house_index_ptr_only() -> None:
    rows = ct.parse_house_index(_index_zip(INDEX))
    assert [r["doc"] for r in rows] == ["20000001", "20035551", "9116389"]          # PTR 만 · 신고일 순
    j = rows[1]
    assert (j["state"], j["district"], j["filed"], j["scanned"]) == ("IL", 1, "2026-10-19", False)
    assert rows[2]["scanned"] is True


def test_parse_ptr_text_owner_partial_amounts_and_options() -> None:
    tx = ct.parse_ptr_text(PTR_TEXT)
    assert [x["ticker"] for x in tx] == ["DDOG", "BRK-B", "AAPL"]                   # 채권(티커 없음)은 빠짐
    d, b, a = tx
    assert d["owner"] == "joint" and d["type"] == "buy" and (d["low"], d["high"]) == (50_001, 100_000)
    assert d["asset"] == "Datadog, Inc. - Class A Common Stock" and d["date"] == "2026-09-28" and d["notified"] == "2026-10-01"
    assert b["owner"] == "spouse" and b["type"] == "sell" and b["partial"] is True and b["high"] == 5_000_000
    assert b["asset"].startswith("Berkshire Hathaway")                              # 앞 채권 줄이 이름에 안 붙는다
    assert a["atype"] == "OP" and a["owner"] == "self" and (a["low"], a["high"]) == (50_000_001, None)
    assert ct.parse_amount("$1,001 - $15,000") == (1001, 15000) and ct.parse_amount("none") == (None, None)


def test_parse_legislators_house_only_with_leaders() -> None:
    m = ct.parse_legislators(LEGIS)
    assert m[("IL", 1)]["party"] == "D" and m[("IL", 1)]["leader"] == "House Democratic Caucus Vice Chair"
    assert m[("TN", 1)]["leader"] is None and len(m) == 2                           # 상원은 빠짐


def test_update_and_outputs(tmp_path: Path) -> None:
    calls: list[str] = []

    def get(url: str) -> bytes:
        calls.append(url)
        if url.endswith("FD.zip"):
            return _index_zip(INDEX) if "2026" in url else _index_zip([])
        if url.endswith("20035551.pdf"):
            return b"%PDF fake"
        raise AssertionError(url)

    orig = ct.pdf_text
    ct.pdf_text = lambda raw: PTR_TEXT                                               # PDF 대신 글자
    try:
        state: dict = {"docs": {"OLD": {"filed": "2024-01-01", "tx": []}}}
        today = date(2026, 11, 6)
        stats = ct.update(state, today, get, pause=0, log=lambda _m: None)
        assert stats["read"] == 1 and stats["scanned"] == 1 and stats["failed"] == 0 and "OLD" not in state["docs"]
        assert "20000001" not in state["docs"]                                       # 1년 넘은 신고는 안 받는다
        # 다음 밤 — 이미 읽은 신고는 다시 안 받는다
        calls.clear()
        again = ct.update(state, today, get, pause=0, log=lambda _m: None)
        assert again["read"] == 0 and not any(u.endswith(".pdf") for u in calls)
    finally:
        ct.pdf_text = orig

    payload, by_ticker, summary = ct.build_outputs(state, ct.parse_legislators(LEGIS), date(2026, 11, 6))
    assert payload["counts"]["tx_recent"] == 3 and payload["recent"][0]["member"] == "Jonathan L. Jackson"
    r = next(x for x in payload["recent"] if x["ticker"] == "DDOG")
    assert r["lag"] == 21 and r["party"] == "D" and r["leader"] and r["url"].endswith("/2026/20035551.pdf")
    assert payload["scanned"][0]["doc"] == "9116389"
    assert set(by_ticker) == {"DDOG", "BRK-B", "AAPL"}
    assert summary["DDOG"] == {"buys": 1, "sells": 0, "members": 1, "last": "2026-10-19", "leader": True, "big_buy": True}
    assert summary["BRK-B"]["sells"] == 1 and summary["BRK-B"]["big_buy"] is False
    json.dumps(payload, allow_nan=False)

    ct.save_state(tmp_path / "snapshot" / ct.STATE_NAME, state)
    assert ct.load_state(tmp_path / "snapshot" / ct.STATE_NAME)["docs"].keys() == state["docs"].keys()
    (tmp_path / "bad.json.gz").write_bytes(b"broken")
    assert ct.load_state(tmp_path / "bad.json.gz") == {"docs": {}} and ct.load_state(None) == {"docs": {}}


def test_failed_pdf_is_retried_next_night() -> None:
    def get(url: str) -> bytes:
        if url.endswith("FD.zip"):
            return _index_zip(INDEX[:1])
        raise RuntimeError("timeout")

    state: dict = {"docs": {}}
    stats = ct.update(state, date(2026, 11, 6), get, pause=0, log=lambda _m: None)
    assert stats["failed"] == 1 and state["docs"] == {}                               # 못 받은 신고는 남기지 않는다


@pytest.mark.parametrize("raw,owner", [("SP Apple Inc. (AAPL) [ST] P 10/01/202610/02/2026$1,001 - $15,000", "spouse"),
                                       ("DC Tesla, Inc. (TSLA) [ST] S 10/01/202610/02/2026$1,001 - $15,000", "child")])
def test_owner_codes(raw: str, owner: str) -> None:
    assert ct.parse_ptr_text("ID Owner Asset Transaction\n" + raw)[0]["owner"] == owner
