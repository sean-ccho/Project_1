"""내계좌 30분 체크용 종목 헤드라인 (watch_news) 테스트 — 네트워크 없음."""
from datetime import datetime, timedelta, timezone

import pytest
import requests

from screener import watch_news as wn

NOW = datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)


def _item(title, source, hours_ago=2.0, link="https://news.google.com/rss/articles/abc"):
    when = (NOW - timedelta(hours=hours_ago)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    return (f"<item><title>{title} - {source}</title><link>{link}</link><pubDate>{when}</pubDate>"
            f"<source url='https://example.com'>{source}</source></item>")


def _rss(*items):
    return f"<?xml version='1.0'?><rss version='2.0'><channel><title>x</title>{''.join(items)}</channel></rss>"


def _h(title, source="Reuters", hours_ago=2.0):
    return wn.Headline(title, source, "https://news.google.com/rss/articles/x", NOW - timedelta(hours=hours_ago))


# ── RSS ─────────────────────────────────────────────────────────────────────

def test_parse_rss_strips_source_suffix_and_reads_time():
    got = wn.parse_rss(_rss(_item("ServiceNow raises guidance", "Reuters", hours_ago=3)))
    assert len(got) == 1
    assert got[0].title == "ServiceNow raises guidance" and got[0].source == "Reuters"
    assert got[0].time == NOW - timedelta(hours=3)


def test_parse_rss_keeps_only_google_news_links_and_survives_garbage():
    xml = _rss(_item("Good one", "AP", link="https://news.google.com/rss/articles/1"),
               _item("Phishy", "AP", link="https://evil.example/x"))
    assert [h.title for h in wn.parse_rss(xml)] == ["Good one"]
    assert wn.parse_rss("<html>consent page</html>") == []
    assert wn.parse_rss("not xml at all") == []


def test_parse_rss_decodes_entities_and_tags():
    got = wn.parse_rss(_rss("<item><title>Foo &amp;amp; Bar &lt;b&gt;beats&lt;/b&gt; - CNBC</title>"
                            "<link>https://news.google.com/rss/articles/1</link><source>CNBC</source></item>"))
    assert got[0].title == "Foo & Bar beats"


# ── 이름·검색어 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("ServiceNow, Inc.", "ServiceNow"),
    ("The Hershey Company", "Hershey"),
    ("Celsius Holdings, Inc.", "Celsius Holdings"),
    ("Alphabet Inc. Class A", "Alphabet"),
    ("PepsiCo, Inc. (PEP)", "PepsiCo"),
    ("", None), (None, None),
])
def test_clean_name(raw, expected):
    assert wn.clean_name(raw) == expected


def test_search_name_and_keyword():
    assert wn.search_name("Neo Battery Materials") == "Neo Battery"
    assert wn.search_name("Bank of America") == "Bank of America"
    assert wn.search_name("HubSpot") == "HubSpot"
    assert wn.keyword("Old Dominion Freight") == "old dominion"
    assert wn.keyword("ServiceNow") == "servicenow"
    assert wn.keyword(None) is None


def test_build_query():
    assert wn.build_query("ServiceNow, Inc.", "NOW") == '"ServiceNow"'
    assert wn.build_query(None, "BRK.B") == "BRK stock"
    assert wn.build_query("", "NOW") == "NOW stock"
    assert wn.build_query("ON Semiconductor", "ON") == '"ON Semiconductor"'
    assert wn.build_query("ABC Corp", "ABC") == '"ABC Corp"'   # 이름이 티커처럼 짧으면 전체 이름


# ── 고르기 ──────────────────────────────────────────────────────────────────

def test_pick_filters_noise_and_unrelated():
    items = [
        _h("ServiceNow stock jumps after AI deal"),
        _h("Acme Capital Management LLC Purchases New Stake in ServiceNow, Inc.", "Some Blog"),
        _h("ServiceNow shares bought by Foo Partners", "Some Blog"),
        _h("ServiceNow hit by class action lawsuit, investors should contact Rosen Law", "GlobeNewswire"),
        _h("ServiceNow insider alert", "MarketBeat"),
        _h("Completely unrelated company soars"),
    ]
    got = wn.pick(items, "NOW", "ServiceNow, Inc.", now=NOW)
    assert [h.title for h in got] == ["ServiceNow stock jumps after AI deal"]


def test_pick_matches_ticker_without_company_name():
    got = wn.pick([_h("NOW rallies on software rotation"), _h("Nowhere to hide this week")], "NOW", None, now=NOW)
    assert [h.title for h in got] == ["NOW rallies on software rotation"]


def test_pick_merges_same_event_and_orders_fresh_then_tier():
    items = [
        _h("ServiceNow raises full year outlook after strong quarter", "Benzinga", hours_ago=2),
        _h("ServiceNow raises full year outlook after strong quarterly results", "Reuters", hours_ago=3),
        _h("ServiceNow names new chief financial officer", "Small Site", hours_ago=1),
        _h("ServiceNow old story from last week", "Reuters", hours_ago=100),
    ]
    got = wn.pick(items, "NOW", "ServiceNow", n=3, now=NOW)
    assert [h.title for h in got] == [
        "ServiceNow raises full year outlook after strong quarterly results",   # 같은 사건 → 높은 매체가 대표
        "ServiceNow names new chief financial officer",
        "ServiceNow old story from last week",                                  # 하루 넘은 것은 뒤로
    ]
    assert len(wn.pick(items, "NOW", "ServiceNow", n=1, now=NOW)) == 1


def test_age_text():
    assert _h("x", hours_ago=0.2).age_text(NOW) == "방금"
    assert _h("x", hours_ago=5).age_text(NOW) == "5시간 전"
    assert _h("x", hours_ago=50).age_text(NOW) == "2일 전"
    assert wn.Headline("x", "s", "u", None).age_text(NOW) == ""


# ── 받기 ────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, text=""):
        self.status_code, self.text = status, text


class _Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.urls = resp, exc, []

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        if self.exc:
            raise self.exc
        return self.resp


def test_fetch_rss_ok_and_edition():
    s = _Session(_Resp(200, _rss(_item("ServiceNow up", "CNBC"))))
    assert len(wn.fetch_rss('"ServiceNow" when:3d', cad=False, session=s)) == 1
    assert "ceid=US:en" in s.urls[0] and "ServiceNow" in s.urls[0]
    wn.fetch_rss("x", cad=True, session=s)
    assert "ceid=CA:en" in s.urls[1]


@pytest.mark.parametrize("sess", [
    _Session(_Resp(429, "slow down")),
    _Session(_Resp(200, "<html>Before you continue to Google</html>")),
    _Session(exc=requests.Timeout("boom")),
])
def test_fetch_rss_failures_give_empty_list(sess):
    assert wn.fetch_rss("x", session=sess) == []


def test_headlines_never_throws():
    s = _Session(exc=RuntimeError("weird"))
    assert wn.headlines("ServiceNow", "NOW", session=s) == []
    ok = _Session(_Resp(200, _rss(_item("ServiceNow up on AI demand", "CNBC", hours_ago=1))))
    got = wn.headlines("ServiceNow, Inc.", "NOW", n=2, session=ok)
    assert [h.title for h in got] == ["ServiceNow up on AI demand"]
    assert "when%3A3d" in ok.urls[0] or "when:3d" in ok.urls[0]
