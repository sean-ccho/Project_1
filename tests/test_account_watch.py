"""내계좌 장중 30분 체크 (account_watch) 테스트 — 네트워크 없음."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from paper_trading.market_date import MARKET_TZ
from screener import account_watch as aw
from screener.account_watch import Quote

TODAY = date(2026, 10, 21)
NOW = datetime(2026, 10, 21, 11, 7, tzinfo=MARKET_TZ)


def _q(price, prev=None, high=None, low=None, stale=False) -> Quote:
    """오늘 시세. prev 를 안 주면 전날 종가 = 현재가 (등락 0%)."""
    return Quote(price=price, prev_close=prev if prev is not None else price,
                 day_high=high if high is not None else price,
                 day_low=low if low is not None else price, bar_date=TODAY, stale=stale)


def _frame(data: dict[str, list[tuple]], dates: list[str]) -> pd.DataFrame:
    """{심볼: [(open, high, low, close), …]} → yf.download(group_by="ticker") 모양."""
    cols = {}
    for sym, bars in data.items():
        arr = np.array(bars, dtype=float)
        for i, f in enumerate(("Open", "High", "Low", "Close")):
            cols[(sym, f)] = arr[:, i]
    return pd.DataFrame(cols, index=pd.DatetimeIndex(pd.to_datetime(dates), name="Date"))


# ── 시간 · 심볼 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dt, expected", [
    (datetime(2026, 10, 21, 9, 29, tzinfo=MARKET_TZ), False),
    (datetime(2026, 10, 21, 9, 30, tzinfo=MARKET_TZ), True),
    (datetime(2026, 10, 21, 16, 7, tzinfo=MARKET_TZ), True),
    (datetime(2026, 10, 21, 16, 15, tzinfo=MARKET_TZ), False),
    (datetime(2026, 10, 24, 11, 0, tzinfo=MARKET_TZ), False),   # 토요일
    (datetime(2026, 11, 2, 9, 37, tzinfo=MARKET_TZ), True),     # EST 로 바뀐 뒤 월요일
])
def test_in_market_window(dt, expected):
    assert aw.in_market_window(dt) is expected


def test_market_window_follows_dst():
    """같은 UTC 시각(13:37)이 EDT 에서는 09:37(장중), EST 에서는 08:37(장 전)."""
    from datetime import timezone
    edt = datetime(2026, 10, 30, 13, 37, tzinfo=timezone.utc).astimezone(MARKET_TZ)
    est = datetime(2026, 11, 2, 13, 37, tzinfo=timezone.utc).astimezone(MARKET_TZ)
    assert aw.in_market_window(edt) and not aw.in_market_window(est)


def test_yf_symbol_and_cad():
    assert aw._yf_symbol("brk.b") == "BRK-B"
    assert aw._yf_symbol("NBM.V") == "NBM.V"
    assert aw._yf_symbol("SHOP.TO") == "SHOP.TO"
    assert aw.is_cad("NBM.V") and aw.is_cad("ABC", "TSXV") and not aw.is_cad("NOW", "NYSE")


def test_norm_date():
    assert aw._norm_date("2026-10-21") == TODAY
    assert aw._norm_date("2026. 10. 21") == TODAY
    assert aw._norm_date("10/21/2026") == TODAY
    assert aw._norm_date("") is None and aw._norm_date("2026-13-40") is None


# ── 시트 읽기 ───────────────────────────────────────────────────────────────

def test_rows_from_values_header_names_and_closed_rows():
    values = [
        ["티커", "매수가", "수량", "매도일", "손절가"],
        ["now", "140", "5", "", "130"],
        ["샘플", "1", "1", "", ""],
        ["PEP", "150", "2", "2026-10-20", ""],
        ["", "1", "1", "", ""],
        ["CELH"],                                       # 짧은 줄
    ]
    rows = aw._rows_from_values(values)
    assert [r["티커"] for r in rows] == ["NOW", "PEP", "CELH"]
    assert rows[2]["매수가"] == ""

    class _WS:
        def get_all_values(self):
            return values

    class _Sheet:
        def worksheet(self, name):
            if name == aw.MY_ACCOUNT_WORKSHEET:
                return _WS()
            raise KeyError(name)

    assert [r["티커"] for r in aw.read_open_rows(_Sheet())] == ["NOW", "CELH"]
    assert aw.read_watchlist(_Sheet()) == []   # 관심종목 탭 없음


# ── 시세 ────────────────────────────────────────────────────────────────────

def test_quotes_from_frame_today_stale_and_missing():
    df = _frame({
        "NOW": [(1, 1, 1, 100), (1, 1, 1, 110), (108, 112, 104, 111)],
        "NBM.V": [(1, 1, 1, 0.15), (1, 1, 1, 0.14), (np.nan,) * 4],   # 오늘 거래 없음
        "SPY": [(1, 1, 1, 700), (1, 1, 1, 707), (707, 712, 705, 710)],
    }, ["2026-10-19", "2026-10-20", "2026-10-21"])
    q = aw.quotes_from_frame(df, {"NOW": "NOW", "NBM.V": "NBM.V", "SPY": "SPY", "GONE": "GONE"}, TODAY)
    assert set(q) == {"NOW", "NBM.V", "SPY"}
    assert q["NOW"].price == 111 and q["NOW"].prev_close == 110
    assert q["NOW"].day_low == 104 and q["NOW"].day_high == 112
    assert q["NOW"].day_pct == pytest.approx(111 / 110 - 1)
    nbm = q["NBM.V"]
    assert nbm.stale and nbm.price == 0.14 and nbm.day_pct is None and nbm.day_low is None


def test_quotes_from_frame_holiday_spy_stale():
    df = _frame({"SPY": [(1, 1, 1, 700), (1, 1, 1, 707)], "NOW": [(1, 1, 1, 1), (1, 1, 1, 2)]},
                ["2026-10-19", "2026-10-20"])
    q = aw.quotes_from_frame(df, {"SPY": "SPY", "NOW": "NOW"}, TODAY)
    assert q["SPY"].stale


def test_quotes_from_frame_single_level_columns():
    df = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0], "Close": [1.0, 2.0]},
                      index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
    assert aw.quotes_from_frame(df, {"SPY": "SPY"}, TODAY)["SPY"].price == 2.0


# ── 알림 ────────────────────────────────────────────────────────────────────

def _kinds(alerts):
    return [(a["kind"], a["ticker"]) for a in alerts]


def test_stop_by_price_and_by_intraday_low():
    rows = [{"티커": "A", "손절가": "90"}, {"티커": "B", "손절가": "90"}]
    alerts = aw.build_watch_alerts(rows, [], {"A": _q(89), "B": _q(95, low=89.5)})
    assert _kinds(alerts) == [("stop", "A"), ("stop", "B")]
    assert "장중 저가" in alerts[1]["text"] and "$95.00" in alerts[1]["text"]


def test_near_then_stop_have_different_keys():
    rows = [{"티커": "A", "손절가": "90", "매수가": "100"}]
    near = aw.build_watch_alerts(rows, [], {"A": _q(92, prev=95)})
    stop = aw.build_watch_alerts(rows, [], {"A": _q(89.9, prev=92)})
    assert _kinds(near) == [("near", "A")] and _kinds(stop) == [("stop", "A")]
    assert near[0]["key"] != stop[0]["key"]
    assert "매수 대비 -8.0%" in near[0]["text"] and "오늘 -3.2%" in near[0]["text"]


def test_no_alert_above_near_band():
    assert aw.build_watch_alerts([{"티커": "A", "손절가": "90"}], [], {"A": _q(93)}) == []


def test_target_by_price_and_by_intraday_high():
    rows = [{"티커": "A", "목표가": "120"}, {"티커": "B", "목표가": "$1,000"}]
    alerts = aw.build_watch_alerts(rows, [], {"A": _q(100, prev=100, high=121), "B": _q(1001, prev=1000)})
    assert _kinds(alerts) == [("target", "A"), ("target", "B")]
    assert "장중 고가 $121.00" in alerts[0]["text"]


def test_move_alert_boundaries():
    rows = [{"티커": t} for t in "ABCD"]
    q = {t: _q(p, prev=100) for t, p in zip("ABCD", (105, 104.9, 95, 95.1))}
    assert _kinds(aw.build_watch_alerts(rows, [], q)) == [("up", "A"), ("down", "C")]


def test_stale_quote_skips_move_and_intraday_checks():
    rows = [{"티커": "A", "손절가": "90"}]
    q = {"A": Quote(price=95, prev_close=80, day_high=None, day_low=None, bar_date=date(2026, 10, 20), stale=True)}
    assert aw.build_watch_alerts(rows, [], q) == []


def test_duplicate_rows_same_ticker_one_alert():
    rows = [{"티커": "A", "손절가": "90", "수량": "1"}, {"티커": "A", "손절가": "90", "수량": "2"}]
    assert len(aw.build_watch_alerts(rows, [], {"A": _q(80, prev=80)})) == 1


def test_watchlist_below_and_above():
    watch = [{"티커": "X", "목표가": "50", "조건": "이하", "메모": "눌림"},
             {"티커": "Y", "목표가": "50", "조건": "이상"},
             {"티커": "Z", "목표가": "50", "조건": ""},          # 비면 이하
             {"티커": "W", "목표가": "", "조건": "이상"}]
    q = {"X": _q(51, prev=51, low=49.5), "Y": _q(49, prev=49, high=50.2), "Z": _q(55, prev=55), "W": _q(10, prev=10)}
    alerts = aw.build_watch_alerts([], watch, q)
    assert _kinds(alerts) == [("watch", "X"), ("watch", "Y")]
    assert "메모: 눌림" in alerts[0]["text"] and "이상" in alerts[1]["text"]
    assert "(장중 저가 $49.50)" in alerts[0]["text"] and "(장중 고가 $50.20)" in alerts[1]["text"]
    hit_now = aw.build_watch_alerts([], [{"티커": "X", "목표가": "50", "조건": "이하"}], {"X": _q(49, prev=49)})
    assert "장중" not in hit_now[0]["text"]                # 지금 가격으로 닿았으면 장중 표시 없음


# ── 요약 ────────────────────────────────────────────────────────────────────

def test_summary_weighted_buy_cad_and_totals():
    rows = [
        {"티커": "A", "매수가": "100", "수량": "1", "매수일": "2026-10-01"},
        {"티커": "A", "매수가": "130", "수량": "2", "매수일": "2026-10-21"},   # 오늘 산 줄은 매수가 기준 오늘 손익
        {"티커": "NBM.V", "매수가": "0.20", "수량": "1000", "거래소": "TSXV"},
    ]
    quotes = {"A": _q(120, prev=110), "NBM.V": _q(0.14, prev=0.15), "SPY": _q(707, prev=700)}
    title, body = aw.build_summary(rows, quotes, NOW)
    assert title == "내계좌 11:07 · SPY +1.00%"
    lines = body.splitlines()
    assert lines[0] == "A $120.00 · 오늘 +9.1% · 매수 대비 +0.0%"   # 수량 가중 매수가 (100 + 2×130) / 3 = 120
    assert lines[1] == "NBM.V C$0.14 · 오늘 -6.7% · 매수 대비 -30.0%"
    assert lines[2] == "오늘 -$10 · 평가 손익 +$0 (USD)"               # 1×(120−110) + 2×(120−130)
    assert lines[3] == "오늘 -C$10 · 평가 손익 -C$60 (CAD)"


def test_summary_missing_quote_line():
    _, body = aw.build_summary([{"티커": "GONE"}], {}, NOW)
    assert body == "GONE 시세 없음"


# ── 상태 · 실행 ─────────────────────────────────────────────────────────────

def test_state_resets_on_new_day(tmp_path):
    p = tmp_path / "state.json"
    aw.save_state(p, {"date": "2026-10-20", "sent": ["stop:A:90"]})
    assert aw.load_state(p, TODAY) == {"date": "2026-10-21", "sent": []}
    aw.save_state(p, {"date": "2026-10-21", "sent": ["stop:A:90"]})
    assert aw.load_state(p, TODAY)["sent"] == ["stop:A:90"]
    assert aw.load_state(None, TODAY)["sent"] == []


@pytest.fixture
def fake_run(monkeypatch):
    """시트·시세·ntfy·메일을 가짜로 바꾸고 보낸 것을 기록한다."""
    sent = {"push": [], "mail": []}
    rows = [{"티커": "A", "손절가": "90", "매수가": "100", "수량": "1"}]
    quotes = {"A": _q(89), "SPY": _q(700, prev=700)}
    ok = {"push": True, "mail": True}

    monkeypatch.setattr(aw, "_open_sheet", lambda: object())
    monkeypatch.setattr(aw, "read_open_rows", lambda sheet: rows)
    monkeypatch.setattr(aw, "read_watchlist", lambda sheet: [])
    monkeypatch.setattr(aw, "fetch_quotes", lambda tickers, today: quotes)
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")

    def push(topic, title, body, priority=3, tags=None):
        sent["push"].append((title, priority))
        return ok["push"]

    def mail(alerts, rows_, quotes_, now):
        sent["mail"].append(len(alerts))
        return ok["mail"]

    monkeypatch.setattr(aw, "send_ntfy", push)
    monkeypatch.setattr(aw, "send_alert_email", mail)
    return sent, quotes, ok


def test_run_dedupes_within_day(tmp_path, fake_run):
    sent, _, _ = fake_run
    p = tmp_path / "state.json"
    assert aw.run(p, now=NOW) == 0
    assert sent["push"][-1][1] == 4 and sent["mail"] == [1]
    assert aw.run(p, now=NOW) == 0                      # 두 번째: 새 알림 없음 → 조용한 요약만
    assert sent["push"][-1][1] == 2 and sent["mail"] == [1]
    assert aw.run(p, now=NOW.replace(day=22)) == 0      # 다음 날: 다시 알림
    assert sent["mail"] == [1, 1]


def test_run_new_stop_level_is_new_alert(tmp_path, fake_run, monkeypatch):
    sent, _, _ = fake_run
    p = tmp_path / "state.json"
    aw.run(p, now=NOW)
    monkeypatch.setattr(aw, "read_open_rows", lambda sheet: [{"티커": "A", "손절가": "95"}])
    aw.run(p, now=NOW)
    assert sent["mail"] == [1, 1]


def test_run_both_channels_fail_not_marked(tmp_path, fake_run):
    sent, _, ok = fake_run
    p = tmp_path / "state.json"
    ok["push"] = ok["mail"] = False
    aw.run(p, now=NOW)
    assert not p.exists()
    ok["mail"] = True                                   # 한 채널만 성공해도 기록
    aw.run(p, now=NOW)
    assert aw.load_state(p, TODAY)["sent"] == ["stop:A:90"]


def test_run_skips_outside_window_and_holiday(tmp_path, fake_run):
    sent, quotes, _ = fake_run
    assert aw.run(tmp_path / "s.json", now=NOW.replace(hour=8)) == 0
    quotes["SPY"] = Quote(price=700, prev_close=700, day_high=None, day_low=None,
                          bar_date=date(2026, 10, 20), stale=True)
    assert aw.run(tmp_path / "s.json", now=NOW) == 0
    assert sent["push"] == [] and sent["mail"] == []


def test_run_dry_run_sends_nothing(tmp_path, fake_run):
    sent, _, _ = fake_run
    p = tmp_path / "state.json"
    assert aw.run(p, now=NOW, dry_run=True) == 0
    assert sent["push"] == [] and not p.exists()


def test_run_sheet_failure_exit_code(monkeypatch):
    monkeypatch.setattr(aw, "_open_sheet", lambda: None)
    assert aw.run(None, now=NOW) == 1


def test_send_ntfy_json_payload(monkeypatch):
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

    def post(url, json=None, timeout=None):
        seen.update(url=url, **json)
        return _Resp()

    monkeypatch.setattr(aw.requests, "post", post)
    assert aw.send_ntfy("t", "내계좌 11:07", "본문", priority=2)
    assert seen["url"] == aw.NTFY_URL and seen["topic"] == "t" and seen["title"] == "내계좌 11:07"
    assert seen["priority"] == 2 and seen["click"] == aw.ACCOUNT_URL
