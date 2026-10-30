"""실적 때 움직임 (screener.earnings_moves) — 휴장일 · 반응일 · 요약 · 알림 글자 · 파이프라인 연결."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener import earnings_moves as em


def test_nyse_holidays_2026_and_2027():
    assert sorted(d.isoformat() for d in em.nyse_holidays(2026)) == [
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
        "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
    h27 = em.nyse_holidays(2027)
    assert date(2027, 3, 26) in h27          # 성금요일 (부활절 3-28)
    assert date(2027, 7, 5) in h27           # 7-4 일요일 → 월요일
    assert date(2027, 12, 24) in h27         # 12-25 토요일 → 금요일
    assert em.is_session(date(2027, 12, 31))  # 2028-01-01 토요일이어도 12-31 은 연다


@pytest.mark.parametrize("earn,timing,react,deadline", [
    ("2026-10-28", "amc", "2026-10-29", "2026-10-28"),   # 장 마감 후 → 다음 날 반응, 그날이 마지막 기회
    ("2026-11-09", "bmo", "2026-11-09", "2026-11-06"),   # 월요일 장 전 → 금요일까지
    ("2026-11-05", None, "2026-11-05", "2026-11-04"),    # 시간 모름 → 장 전으로 보고 더 이르게
    ("2026-11-25", "amc", "2026-11-27", "2026-11-25"),   # 추수감사절 건너뜀
    ("2026-11-27", "bmo", "2026-11-27", "2026-11-25"),
])
def test_reaction_and_deadline(earn, timing, react, deadline):
    d = date.fromisoformat(earn)
    assert em.reaction_date(d, timing).isoformat() == react
    assert em.decision_deadline(d, timing).isoformat() == deadline


def test_sessions_until():
    assert em.sessions_until(date(2026, 10, 29), date(2026, 11, 4)) == 4    # 금·월·화·수
    assert em.sessions_until(date(2026, 11, 4), date(2026, 10, 29)) == -4
    assert em.sessions_until(date(2026, 10, 31), date(2026, 11, 2)) == 1    # 토요일 → 월요일
    assert em.sessions_until(date(2026, 10, 29), date(2026, 10, 29)) == 0


def test_accepts_only_8k_item_202():
    block = {"form": ["8-K", "8-K/A", "8-K", "10-Q"], "items": ["2.02,9.01", "2.02", "5.02", ""],
             "acceptanceDateTime": ["2026-10-28T20:11:00.000Z", "2026-10-29T12:00:00.000Z",
                                    "2026-10-01T12:00:00.000Z", "2026-10-28T20:30:00.000Z"]}
    assert em.accepts_from_filings([block]) == ["2026-10-28T20:11:00.000Z"]


def test_scheduled_events_drop_duplicates_and_preannouncements():
    acc = ["2026-07-22T20:05:00.000Z", "2026-07-23T12:00:00.000Z",   # 같은 발표 중복 → 첫 것
           "2026-10-07T11:00:00.000Z",                               # 실적 경고 (3주 뒤 정규 발표) → 빠짐
           "2026-10-28T20:10:00.000Z"]
    ev = em.scheduled_events(acc)
    assert [t.strftime("%Y-%m-%d %H:%M") for t in ev] == ["2026-07-22 16:05", "2026-10-28 16:10"]  # ET


def _close(days: list[str], vals: list[float]) -> pd.Series:
    return pd.Series(vals, index=pd.to_datetime(days))


def test_reaction_moves_amc_and_bmo():
    close = _close(["2026-01-27", "2026-01-28", "2026-01-29", "2026-04-22", "2026-04-23", "2026-04-24"],
                   [100, 100, 90, 50, 55, 56])
    mv = em.reaction_moves(["2026-01-28T21:10:00.000Z",            # 16:10 ET → 다음 날 1-29
                            "2026-04-23T11:00:00.000Z"], close)    # 07:00 ET → 그날 4-23
    assert [(m["date"], round(m["move"], 3), m["timing"]) for m in mv] == [
        ("2026-01-29", -0.1, "amc"), ("2026-04-23", 0.1, "bmo")]
    assert em.reaction_moves(["2026-04-24T21:00:00.000Z"], close) == []   # 반응일이 아직 없음


def test_move_stats_and_bad_case():
    moves = [{"date": f"2026-0{i}-01", "move": m, "timing": "amc"} for i, m in enumerate([0.05, -0.10, 0.02, -0.03], 1)]
    st = em.move_stats(moves)
    assert st["avg"] == 0.05 and st["worst"] == -0.1 and st["best"] == 0.05 and st["n"] == 4 and st["timing"] == "amc"
    assert em.move_stats(moves[:1]) is None
    assert em.bad_case(0.05, -0.10) == pytest.approx(0.15)   # 평균 × 3
    assert em.bad_case(0.03, -0.25) == pytest.approx(0.25)   # 최악이 더 크면 최악
    assert em.bad_case(0.30, -0.20) == pytest.approx(0.50)   # 최대 50%
    assert em.bad_case(None, None) is None


def test_sheet_roundtrip_and_decision():
    st = {"avg": 0.088, "worst": -0.1775, "best": 0.15, "n": 8, "timing": "amc", "last": []}
    vals = em.sheet_values(st)
    assert vals == {"실적 평균": "8.8%", "실적 최악": "-17.8%", "실적 시간": "장 마감 후"}
    back = em.stats_from_sheet(vals)
    assert back["avg"] == pytest.approx(0.088) and back["worst"] == pytest.approx(-0.178) and back["timing"] == "amc"
    assert em.stats_from_sheet({"실적 평균": ""}) is None
    assert em.decision_of("절반 · 2026-11-04", "2026-11-04") == "절반"
    assert em.decision_of("절반 · 2026-08-05", "2026-11-04") is None     # 지난 실적에 정한 것
    assert em.decision_of("몰라 · 2026-11-04", "2026-11-04") is None


def test_alert_text_has_money_and_decision():
    st = {"avg": 0.088, "worst": -0.1775, "n": 8, "timing": "amc"}
    txt = em.alert_text("AAA", "2026-11-18", st, shares=20, price=100.0, account=10000, decision=None,
                        today=date(2026, 11, 16))
    assert "11/18(수) 장 마감 후" in txt and "거래일 2일 남음" in txt
    assert "보통 ±$176" in txt and "나쁜 경우 −$528" in txt and "계좌의 5.3%" in txt and "아직 안 정함" in txt
    done = em.alert_text("AAA", "2026-11-18", st, shares=20, price=100.0, account=None, decision="절반",
                         today=date(2026, 11, 18))
    assert "(오늘)" in done and "대응: 절반" in done
    assert "시간 모름" in em.alert_text("X", "2026-11-05", None, shares=None, price=None, account=None,
                                     decision=None, today=date(2026, 11, 1))


def test_alert_stage():
    assert em.alert_stage(date(2026, 10, 26), "2026-10-28", "amc") == "d2"
    assert em.alert_stage(date(2026, 10, 27), "2026-10-28", "amc") is None
    assert em.alert_stage(date(2026, 10, 28), "2026-10-28", "amc") == "d0"
    assert em.alert_stage(date(2026, 11, 6), "2026-11-09", "bmo") == "d0"     # 월요일 장 전 → 금요일이 마감
    assert em.alert_stage(date(2026, 11, 6), "nope", "bmo") is None


def test_sec_client_collects_earnings_and_reads_older_page():
    from screener import sec_filings as sf

    recent = {"form": ["8-K"], "items": ["2.02"], "acceptanceDateTime": ["2026-10-28T20:11:00.000Z"],
              "filingDate": ["2026-10-28"], "accessionNumber": ["a1"], "primaryDocument": ["d.htm"]}
    older = {"form": ["8-K", "8-K"], "items": ["2.02", "2.02"],
             "acceptanceDateTime": ["2026-04-22T20:11:00.000Z", "2026-07-22T20:11:00.000Z"]}
    pages = {"https://data.sec.gov/submissions/CIK0000000001.json":
             {"filings": {"recent": recent, "files": [{"name": "CIK0000000001-submissions-001.json"}]}},
             "https://data.sec.gov/submissions/CIK0000000001-submissions-001.json": older}

    class Fake(sf.SecClient):
        def _get_json(self, url):
            return pages.get(url)

    c = Fake(user_agent="t t@example.com")
    c._cik = {"AAA": 1}
    got: dict[str, list[str]] = {}
    assert c.flags("AAA", date(2026, 10, 29), got) == []
    assert got["AAA"] == ["2026-04-22T20:11:00.000Z", "2026-07-22T20:11:00.000Z", "2026-10-28T20:11:00.000Z"]


def test_runner_snapshot_keeps_earn_accepts(monkeypatch, tmp_path):
    from paper_trading import runner
    from screener import sec_filings as sf

    monkeypatch.setattr(runner, "_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(runner, "_SP500_SNAPSHOT", tmp_path / "sp500_ranked.parquet")

    def fake(tickers, earnings=None, **kw):
        earnings["BRK-B"] = ["2026-10-28T20:11:00.000Z"]
        return {"BRK-B": []}

    monkeypatch.setattr(sf, "fetch_flags", fake)
    assert runner.save_ranked_snapshot(pd.DataFrame({"티커": ["BRK.B", "AAPL"], "현재가격": [1.0, 2.0]}), "sp500")
    out = pd.read_parquet(tmp_path / "sp500_ranked.parquet")
    assert out["earn_accepts"].tolist() == ['["2026-10-28T20:11:00.000Z"]', ""]


def test_dashboard_earnings_moves_block():
    from web_export.dashboard import earnings_accepts, earnings_moves_block

    ranked = pd.DataFrame({"티커": ["BRK.B", "AAA"],
                           "earn_accepts": ['["2026-01-28T21:10:00.000Z", "2026-04-22T21:10:00.000Z"]', ""]})
    acc = earnings_accepts(ranked, ["BRK.B", "AAA"])
    assert list(acc) == ["BRK.B"]
    days = ["2026-01-28", "2026-01-29", "2026-04-22", "2026-04-23"]
    prices = {"BRK-B": pd.DataFrame({"Close": [100.0, 110.0, 100.0, 80.0]}, index=pd.to_datetime(days))}
    rec = earnings_moves_block(acc, prices)["BRK.B"]
    assert rec["n"] == 2 and rec["worst"] == -0.2 and rec["avg"] == pytest.approx(0.15) and rec["timing"] == "amc"


def test_portfolio_alert_has_numbers_and_urgent_flag():
    from screener.portfolio_report import build_alerts

    st = {"avg": 0.10, "worst": -0.2, "n": 8, "timing": "amc", "last": []}
    records = [{"티커": "AAA", "손절가": "1", "수량": "10"}, {"티커": "AAA", "손절가": "1", "수량": "5"},
               {"티커": "BBB", "손절가": "1", "수량": "2", "실적대응": "그대로 · 2026-11-04"}]
    res = {"AAA": {"_price": 100.0, "_earnings": "2026-11-03", "_earn_stats": st},
           "BBB": {"_price": 50.0, "_earnings": "2026-11-04", "_earn_stats": st}}
    alerts, _ = build_alerts(records, res, today=date(2026, 10, 30))
    earn = {a["ticker"]: a for a in alerts if a["kind"] == "earn"}
    assert "15주 $1,500" in earn["AAA"]["text"] and "나쁜 경우 −$450" in earn["AAA"]["text"]
    assert earn["AAA"]["urgent"] == "1"                       # 마감 11-03 까지 2거래일 · 대응 안 정함
    assert earn["BBB"]["urgent"] == "" and "대응: 그대로" in earn["BBB"]["text"]


def test_account_watch_earnings_alert_days():
    from screener.account_watch import Quote, earnings_alerts

    rows = [{"티커": "AAA", "수량": "20", "매수가": "100", "실적일": "'2026-11-18", "실적 평균": "8.8%",
             "실적 최악": "-17.8%", "실적 시간": "장 마감 후", "실적대응": ""}]
    q = {"AAA": Quote(price=100.0, prev_close=100.0, day_high=None, day_low=None, bar_date=date(2026, 11, 16),
                      stale=False)}
    d2 = earnings_alerts(rows, q, date(2026, 11, 16))
    assert len(d2) == 1 and d2[0]["key"] == "earn:AAA:2026-11-18:d2" and "나쁜 경우" in d2[0]["text"]
    assert earnings_alerts(rows, q, date(2026, 11, 17)) == []
    assert earnings_alerts(rows, q, date(2026, 11, 18))[0]["key"].endswith(":d0")
    rows[0]["실적대응"] = "절반 · 2026-11-18"
    assert "대응: 절반" in earnings_alerts(rows, q, date(2026, 11, 18))[0]["text"]
