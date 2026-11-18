"""실적일 기억 (web_export.report_dates) — 예측 모델 · 종목 상세 캐시가 실적 발표 뒤 다시 받게 하는 공용 도구."""
from __future__ import annotations

from datetime import date

from web_export import report_dates as rd
from web_export import stock_detail


def test_item_dates_reads_sec_last_and_yahoo_next():
    it = {"earnings_date": "2026-11-18", "earnings_move": {"last": [["2026-05-05", 0.01], ["2026-08-04", -0.0154]]}}
    assert rd.item_dates(it) == {"last_report": "2026-08-04", "next_report": "2026-11-18"}
    assert rd.item_dates({"earnings_date": None}) == {"last_report": None, "next_report": None}
    assert rd.item_dates({"earnings_move": {"last": [[None]]}, "earnings_date": "bad"}) == {"last_report": None, "next_report": None}


def test_note_turns_passed_yahoo_date_into_last_report_for_foreign_issuers():
    cache = {"VIPS": {"fetched": "2026-11-10"}, "CNI": {"fetched": "2026-11-10", "next_report": "2026-10-24"}}
    src = {"VIPS": {"last_report": None, "next_report": "2026-11-24"}, "CNI": {"last_report": None, "next_report": "2027-01-28"}}
    assert rd.note(cache, src, date(2026, 11, 17)) == 1                      # CNI — 적어 둔 10-24 가 지남
    assert cache["CNI"] == {"fetched": "2026-11-10", "last_report": "2026-10-24", "next_report": "2027-01-28"}
    assert cache["VIPS"] == {"fetched": "2026-11-10", "next_report": "2026-11-24"}
    src["VIPS"]["next_report"] = "2027-03-01"                                  # 발표 뒤 Yahoo 가 다음 분기로 넘김
    assert rd.note(cache, src, date(2026, 11, 26)) == 1 and cache["VIPS"]["last_report"] == "2026-11-24"
    assert rd.due(cache["VIPS"], date(2026, 11, 26), (2, 10)) == "2026-11-26"


def test_detail_cache_keeps_report_dates_and_refetches_after_report():
    old = {"rec": {"buy": 3}, "fetched": "2026-11-10", "last_report": "2026-11-24", "next_report": "2027-03-01"}
    ent = stock_detail.merge_entry(old, {"rec": {"buy": 4}, "eps": [], "revenue": [], "next": None}, date(2026, 11, 25))
    assert ent["last_report"] == "2026-11-24" and ent["next_report"] == "2027-03-01" and ent["rec"] == {"buy": 4}
    cache = {"VIPS": {"fetched": "2026-11-20", "rec": {}, "eps": [], "revenue": [], "next": None}}
    order = stock_detail.refresh_order(["VIPS"], cache, date(2026, 11, 25), {"VIPS": "2026-11-24"})
    assert order == ["VIPS"]                                                  # 발표 뒤 AFTER_REPORT_DAYS 일까지 매일
