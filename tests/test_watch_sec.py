"""내계좌 30분 체크용 새 SEC 공시 (watch_sec) 테스트 — 네트워크 없음."""
from datetime import date

import pytest

from screener import watch_sec as ws

TODAY = date(2026, 10, 21)


def _recent(*rows):
    """rows: (filingDate, form, items, accession)"""
    return {"filingDate": [r[0] for r in rows], "form": [r[1] for r in rows],
            "items": [r[2] for r in rows], "accessionNumber": [r[3] for r in rows]}


# ── 분류 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("form, items, expected", [
    ("8-K", "2.02,9.01", (("2.02",), "실적 발표", "orange")),
    ("8-K", "9.01", None),
    ("8-K", "5.07", None),
    ("8-K", "7.01,8.01", None),
    ("8-K", "", None),
    ("8-K/A", "4.02", (("4.02",), "재무제표 재작성", "red")),
    ("8-K", "5.02", (("5.02",), "임원 변동", "yellow")),
    ("8-K", "2.02,5.02", (("2.02", "5.02"), "실적 발표 · 임원 변동", "orange")),   # 센 쪽이 심각도
    ("8-K", "1.01,3.01", (("1.01", "3.01"), "중요 계약 체결 · 상장 요건 통보", "red")),
    ("424B5", "", ((), "증자(공모)", "orange")),
    ("NT 10-K", "", ((), "보고서 지연", "red")),
    ("NT 10-Q/A", "", ((), "보고서 지연", "red")),
    ("10-Q", "", None),
    ("S-3", "", None),            # 선반 등록은 큰 회사에 늘 있어서 안 알린다
    ("4", "", None),
])
def test_classify(form, items, expected):
    assert ws.classify(form, items) == expected


# ── 새 공시 고르기 ──────────────────────────────────────────────────────────

def test_filings_from_recent_window_seen_and_order():
    recent = _recent(
        ("2026-10-21", "8-K", "5.02", "0001-26-000005"),            # 오늘, 노랑
        ("2026-10-20", "8-K", "2.02,9.01", "0001-26-000004"),       # 어제 마감 후 실적, 주황
        ("2026-10-20", "8-K", "5.07", "0001-26-000003"),            # 안 알리는 항목
        ("2026-10-19", "8-K", "4.02", "0001-26-000002"),            # 너무 오래됨 (어제 이전)
        ("2026-10-21", "424B5", "", "0001-26-000006"),              # 증자
        ("2026-10-21", "8-K", "2.02", "0001-26-000007"),            # 이미 알림
        ("not-a-date", "8-K", "2.02", "0001-26-000008"),
        ("2026-10-22", "8-K", "2.02", "0001-26-000009"),            # 미래 (시계 오차) 무시
    )
    got = ws.filings_from_recent("NOW", recent, 1373715, TODAY, seen={"0001-26-000007"})
    # 심각도 → 최근 순: 같은 주황이면 오늘 낸 증자가 어제 낸 실적보다 앞
    assert [(f.acc, f.severity) for f in got] == [
        ("0001-26-000006", "orange"), ("0001-26-000004", "orange"), ("0001-26-000005", "yellow")]
    f = got[1]
    assert f.is_earnings and f.key == "sec:NOW:0001-26-000004" and f.filed == date(2026, 10, 20)
    assert f.url == "https://www.sec.gov/Archives/edgar/data/1373715/000126000004/"
    assert not got[0].is_earnings and got[0].items == ()


def test_filings_from_recent_empty_and_missing_columns():
    assert ws.filings_from_recent("A", {}, 1, TODAY) == []
    only_dates = {"filingDate": ["2026-10-21"], "form": ["8-K"]}      # items·accession 열이 없어도 터지지 않는다
    assert ws.filings_from_recent("A", only_dates, 1, TODAY) == []


# ── 받기 ────────────────────────────────────────────────────────────────────

class _FakeClient:
    def __init__(self, enabled=True, ciks=None, subs=None, boom=False):
        self.enabled, self._ciks, self._subs, self.boom, self.calls = enabled, ciks or {}, subs or {}, boom, []

    def cik_map(self):
        if self.boom:
            raise RuntimeError("sec down")
        return self._ciks

    def _get_json(self, url):
        self.calls.append(url)
        return self._subs.get(url)


def _sub_url(cik):
    return ws.SUBMISSIONS_URL.format(cik=cik)


def test_fetch_new_filings_collects_skips_canadian_and_spy():
    client = _FakeClient(
        ciks={"NOW": 1373715, "PEP": 77476, "SPY": 884394},
        subs={_sub_url(1373715): {"filings": {"recent": _recent(("2026-10-20", "8-K", "2.02", "A1"))}},
              _sub_url(77476): {"filings": {"recent": _recent(("2026-10-20", "8-K", "9.01", "B1"))}}})
    got = ws.fetch_new_filings(["NOW", "PEP", "NBM.V", "SPY", "now"], TODAY, client=client)
    assert [(f.ticker, f.acc) for f in got] == [("NOW", "A1")]
    assert client.calls == [_sub_url(1373715), _sub_url(77476)]          # NBM.V·SPY 는 부르지 않음, 중복 티커 한 번


def test_fetch_new_filings_respects_seen():
    client = _FakeClient(ciks={"NOW": 1}, subs={_sub_url(1): {"filings": {"recent": _recent(("2026-10-21", "8-K", "2.02", "A1"))}}})
    assert ws.fetch_new_filings(["NOW"], TODAY, seen={"A1"}, client=client) == []


def test_fetch_new_filings_disabled_and_failures_give_empty():
    assert ws.fetch_new_filings(["NOW"], TODAY, client=_FakeClient(enabled=False)) == []
    assert ws.fetch_new_filings(["NOW"], TODAY, client=_FakeClient(boom=True)) == []
    # 한 종목의 submissions 가 비어 와도 (404 등) 다음 종목은 계속
    client = _FakeClient(ciks={"AAA": 1, "BBB": 2},
                         subs={_sub_url(2): {"filings": {"recent": _recent(("2026-10-21", "8-K", "1.03", "B9"))}}})
    assert [f.acc for f in ws.fetch_new_filings(["AAA", "BBB"], TODAY, client=client)] == ["B9"]


def test_fetch_new_filings_time_budget():
    client = _FakeClient(ciks={"AAA": 1})
    assert ws.fetch_new_filings(["AAA"], TODAY, client=client, budget_sec=-1) == []
    assert client.calls == []
