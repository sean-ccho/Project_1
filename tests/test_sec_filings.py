"""SEC 공시 경고 — 규칙·기간·3.01 판별·표시 (네트워크 없이)."""
from datetime import date

import pandas as pd

from screener import sec_filings as sf

TODAY = date(2026, 10, 16)


def _recent(rows):
    """(filingDate, form, items, accession, primaryDocument) → submissions filings.recent 모양."""
    keys = ["filingDate", "form", "items", "accessionNumber", "primaryDocument"]
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


def test_rules_windows_and_order():
    rec = _recent([
        ("2026-10-14", "8-K", "5.02,9.01", "0001-26-000001", "a.htm"),     # 임원 변동 (노랑)
        ("2026-10-10", "424B5", "", "0001-26-000002", "b.htm"),            # 자금 조달
        ("2026-10-01", "424B5", "", "0001-26-000003", "c.htm"),            # 자금 조달 2건째
        ("2026-08-01", "424B5", "", "0001-26-000004", "d.htm"),            # 30일 밖 → 안 셈
        ("2026-09-01", "8-K", "3.01", "0001-26-000005", "e.htm"),          # 상장폐지 경고
        ("2026-06-01", "NT 10-Q", "", "0001-26-000006", "f.htm"),          # 90일 밖 → 없음
        ("2026-10-02", "424B2", "", "0001-26-000007", "g.htm"),            # 은행 구조화채권 — 안 셈
        ("2026-10-12", "144", "", "0001-26-000008", "h.htm"),
    ])
    flags = sf.flags_from_recent(rec, 123, TODAY)
    codes = [f.code for f in flags]
    assert codes == ["delisting", "offering", "officer", "insider_sale_plan"]   # 빨강 → 주황 → 노랑
    off = flags[1]
    assert off.count == 2 and off.date == "2026-10-10" and off.url.endswith("/123/000126000002/")
    assert sf.format_flags(flags) == "🔴 상장폐지 경고 9/1 · 🟠 자금 조달 10/10 (2건)"           # 메일·시트 = 주황까지
    assert "🟡 임원 변동 10/14" in sf.format_flags(flags, "yellow")


def test_transfer_301_is_downgraded():
    rec = _recent([("2026-09-25", "8-K", "3.01", "0002-26-000001", "x.htm")])
    flags = sf.flags_from_recent(rec, 9, TODAY, frozenset({"0002-26-000001"}))
    assert [(f.code, f.severity) for f in flags] == [("listing_transfer", "orange")]


def test_classify_301_ignores_item_title():
    title = "Item 3.01 Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing."
    assert sf.classify_301(f"<p>{title}</p> The Company notified the NYSE of its intention to voluntarily withdraw "
                           "the listing and transfer the listing to Nasdaq.") == "transfer"
    assert sf.classify_301(f"{title} On Oct 1 the Company received a notification letter from Nasdaq that it is "
                           "not in compliance with the minimum bid price requirement.") == "deficiency"
    assert sf.classify_301(f"{title} Other text only.") == "unknown"


def test_fetch_flags_without_user_agent_is_empty(capsys):
    assert sf.fetch_flags(["AAPL"], TODAY, client=sf.SecClient(user_agent="")) == {}
    assert "SEC_USER_AGENT 없음" in capsys.readouterr().out


class _Resp:
    def __init__(self, code, payload=None, text=""):
        self.status_code, self._p, self.text = code, payload, text

    def json(self):
        return self._p


class _Session:
    """티커 목록 · submissions · 3.01 본문만 흉내."""

    def __init__(self):
        self.urls = []

    def get(self, url, headers=None, timeout=None):
        self.urls.append((url, headers["User-Agent"]))
        if url == sf.TICKERS_URL:
            return _Resp(200, {"0": {"ticker": "BRK.B", "cik_str": 77}, "1": {"ticker": "ZZZ", "cik_str": 88}})
        if "CIK0000000077" in url:
            return _Resp(200, {"filings": {"recent": _recent([("2026-10-09", "8-K", "3.01", "0003-26-000001", "d.htm")])}})
        if url.endswith("/d.htm"):
            return _Resp(200, text="received a notification letter ... not in compliance with the minimum bid price")
        return _Resp(404)


def test_fetch_flags_with_fake_sec(monkeypatch):
    monkeypatch.setattr(sf, "MIN_INTERVAL", 0.0)
    s = _Session()
    out = sf.fetch_flags(["BRK.B", "NOPE", "ZZZ"], TODAY, client=sf.SecClient("Me me@example.com", session=s))
    assert set(out) == {"BRK-B"}                      # NOPE 는 SEC 에 없음, ZZZ 는 submissions 404
    assert out["BRK-B"][0].code == "delisting"       # 본문이 요건 미달 → 빨강 유지
    assert all(ua == "Me me@example.com" for _, ua in s.urls)


def test_snapshot_gets_sec_columns(monkeypatch, tmp_path):
    from paper_trading import runner
    monkeypatch.setattr(runner, "_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(runner, "_SP500_SNAPSHOT", tmp_path / "sp500_ranked.parquet")
    flag = sf.Flag("delisting", "상장폐지 경고", "red", "2026-10-09", "8-K", 1, "u")
    monkeypatch.setattr(sf, "fetch_flags", lambda tickers, **kw: {"BRK-B": [flag]})
    df = pd.DataFrame({"티커": ["BRK.B", "AAPL"], "현재가격": [1.0, 2.0]})
    assert runner.save_ranked_snapshot(df, "sp500")
    out = pd.read_parquet(tmp_path / "sp500_ranked.parquet")
    assert out["[공시]"].tolist() == ["🔴 상장폐지 경고 10/9", ""]
    assert '"delisting"' in out["sec_flags"].iloc[0] and out["sec_flags"].iloc[1] == ""


def test_dashboard_sec_flags_parse():
    from web_export.dashboard import _sec_flags
    assert _sec_flags('[{"code": "offering"}]') == [{"code": "offering"}]
    assert _sec_flags("") == [] and _sec_flags("nan") == [] and _sec_flags("{bad") == [] and _sec_flags(None) == []


def test_pt_email_health_cell_shows_sec_line(monkeypatch, tmp_path):
    """PT 메일 보유 표의 헬스체크 칸에 [공시] 줄이 붙는다 (스냅샷에서 읽음)."""
    import smtplib
    from screener import exporter

    (tmp_path / "data" / "paper_trading").mkdir(parents=True)
    pd.DataFrame({"티커": ["AAA"], "[헬스체크]": ["72 (B) · 매출+10% 마진+5% 이익+3%"], "[시총]": ["$1.0B"],
                  "[공시]": ["🔴 상장폐지 경고 10/9"]}).to_parquet(tmp_path / "data/paper_trading/sp500_ranked.parquet")
    monkeypatch.chdir(tmp_path)
    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, msg): sent["html"] = msg.get_payload()[0].get_payload(decode=True).decode()

    monkeypatch.setattr(exporter, "EMAIL_ENABLED", True)
    monkeypatch.setattr(exporter, "EMAIL_PASSWORD", "x", raising=False)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    pos = [{"ticker": "AAA", "entry_price": 10.0, "entry_date": "2026-10-01", "shares": 1, "strategy": "📈 모멘텀"}]
    exporter.send_paper_trading_email({"date": "2026-10-16"}, pos, prices={"AAA": 11.0})
    assert "🔴 상장폐지 경고 10/9" in sent["html"] and "72 (B)" in sent["html"]
