"""숏스퀴즈 감시 — 연료·불씨 판정, 공매도 한 줄, 감시 목록 만들기 (가짜 데이터, 네트워크 없음)."""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from screener import short_squeeze as ss


def test_short_line_and_high():
    assert ss.short_line(0.183, 6.12, "2026-09-30") == "공매도 18% · 되사기 6.1일분 (9/30 기준)"
    assert ss.short_line(None, 2.0, None) == "되사기 2.0일분"
    assert ss.short_line(None, None, "2026-09-30") == ""
    assert ss.is_high(0.20, 5.0) and not ss.is_high(0.40, 4.9) and not ss.is_high(0.19, 9.0) and not ss.is_high(None, 9.0)


def test_classify_fuel_and_fire():
    assert ss.classify(0.25, 6.0, False, 3.0) == "fuel"
    assert ss.classify(0.25, 6.0, True, 2.5) == "fire"
    assert ss.classify(0.25, 6.0, True, 1.5) == "fuel"          # 신고가지만 거래량이 모자람
    assert ss.classify(0.19, 9.0, True, 3.0) is None             # 공매도 비중 모자람
    assert ss.classify(0.40, 4.9, True, 3.0) is None             # 되사기 모자람
    assert ss.classify(np.nan, 9.0, True, 3.0) is None


def test_price_trigger_breakout_and_volume():
    idx = pd.bdate_range("2026-07-01", periods=40)
    close = pd.Series(np.r_[np.full(39, 10.0), 11.0], index=idx)
    vol = pd.Series(np.r_[np.full(39, 1e6), 3e6], index=idx)
    t = ss.price_trigger(close, vol)
    assert t["breakout"] and np.isclose(t["vol_ratio"], 3.0) and np.isclose(t["ret_5d"], 0.1)
    assert not ss.price_trigger(close.iloc[:10], vol.iloc[:10])["breakout"]   # 일봉이 모자람


def _sd():
    table = pd.DataFrame({
        "ticker": ["SQZ", "FUEL", "LOW", "TINY", "BIGDTC"],
        "si": [30e6, 25e6, 1e6, 5e6, 9e6], "dtc": [8.0, 6.0, 1.0, 9.0, 12.0],
        "adv": [4e6, 4e6, 4e6, 1e5, 1e6], "si_change": [0.1, -0.05, 0.0, 0.0, 0.2],
    }).set_index("ticker")
    return SimpleNamespace(asof=date(2026, 10, 16), settle="2026-09-30", table=table)


def test_build_watch_with_fake_inputs():
    nasdaq = pd.DataFrame({"ticker": ["SQZ", "FUEL", "LOW", "TINY", "BIGDTC"], "name": list("ABCDE"),
                           "price": [20.0, 15.0, 30.0, 5.0, 40.0], "mcap": [2e9, 1.5e9, 3e9, 5e8, 4e9]})
    idx = pd.bdate_range("2026-07-01", periods=40)
    closes = pd.DataFrame({"SQZ": np.r_[np.full(39, 18.0), 20.0], "FUEL": np.full(40, 15.0), "BIGDTC": np.full(40, 40.0)}, index=idx)
    vols = pd.DataFrame({"SQZ": np.r_[np.full(39, 1e6), 3e6], "FUEL": np.full(40, 1e6), "BIGDTC": np.full(40, 1e6)}, index=idx)
    w = ss.build_watch(_sd(), nasdaq, floats={"SQZ": 60e6, "FUEL": 80e6, "BIGDTC": 9e7}, daily=(closes, vols), seed=1)
    watch = w[w["status"].isin(["fire", "fuel"])]
    assert watch["ticker"].tolist() == ["SQZ", "FUEL"]                 # 🔥 먼저
    assert watch.iloc[0]["status"] == "fire" and np.isclose(watch.iloc[0]["short_pct"], 0.5)
    assert watch.iloc[1]["short_basis"] == "float" and np.isclose(watch.iloc[1]["short_pct"], 25e6 / 80e6)
    assert "TINY" not in set(w["ticker"])                               # 거래량 20만 주 미만은 대상 밖
    assert "BIGDTC" not in set(watch["ticker"])                         # 되사기는 길어도 비중 10% → 연료 아님
    pool = w[w["status"] == "pool"]
    assert len(pool) == 4 and set(pool["ticker"]) <= {"SQZ", "FUEL", "LOW", "BIGDTC"}


def test_squeeze_alerts_for_holdings():
    from screener.portfolio_report import squeeze_alerts
    watch = pd.DataFrame([{"ticker": "SQZ", "status": "fire", "short_pct": 0.5, "dtc": 8.0},
                          {"ticker": "BRK-B", "status": "fuel", "short_pct": 0.22, "dtc": 5.5}])
    out = squeeze_alerts(["SQZ", "BRK.B", "AAPL"], watch=watch)
    assert [a["ticker"] for a in out] == ["SQZ", "BRK.B"]
    assert "진행 가능" in out[0]["text"] and "대기" in out[1]["text"]


def test_dashboard_squeeze_and_short(monkeypatch, tmp_path):
    from web_export import dashboard as wd
    watch = pd.DataFrame([{**{c: None for c in ss.WATCH_COLUMNS}, "ticker": "SQZ", "name": "Squeeze Co", "status": "fire",
                           "price": 20.0, "short_pct": 0.5, "dtc": 8.0, "si_change": 0.1, "ret_5d": 0.11, "vol_ratio": 3.0,
                           "settle": "2026-09-30", "short_basis": "float"}])
    monkeypatch.setattr(ss, "load_watch", lambda path=None: watch)
    blk = wd.squeeze_block()
    assert blk["settle"] == "2026-09-30" and blk["items"][0]["ticker"] == "SQZ" and blk["items"][0]["status"] == "fire"
    row = pd.Series({"short_pct": 0.183, "short_dtc": 6.1, "short_change": 0.05, "short_settle": "2026-09-30", "short_high": "True"})
    assert wd._short(row) == {"pct": 0.183, "dtc": 6.1, "change": 0.05, "settle": "2026-09-30", "high": True}
    assert wd._short(pd.Series({"short_pct": None, "short_dtc": None})) is None


def test_pt_email_short_line_and_squeeze_section(monkeypatch, tmp_path):
    import smtplib
    from screener import exporter

    d = tmp_path / "data" / "paper_trading"
    d.mkdir(parents=True)
    pd.DataFrame({"티커": ["AAA"], "[헬스체크]": ["72 (B) · 매출+10% 마진+5% 이익+3%"], "[시총]": ["$1.0B"],
                  "[공매도]": ["공매도 22% · 되사기 6.1일분 (9/30 기준)"], "short_high": ["True"]}).to_parquet(d / "sp500_ranked.parquet")
    rows = [{**{c: None for c in ss.WATCH_COLUMNS}, "ticker": t, "name": t, "status": s, "price": 10.0, "short_pct": 0.3,
             "dtc": 7.0, "si_change": 0.1, "ret_5d": 0.05, "vol_ratio": 2.5, "settle": "2026-09-30", "short_basis": "float"}
            for t, s in (("AAA", "fire"), ("BBB", "fuel"), ("POOL1", "pool"))]
    pd.DataFrame(rows).to_parquet(d / "squeeze_watch.parquet")
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
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    pos = [{"ticker": "AAA", "entry_price": 10.0, "entry_date": "2026-10-01", "shares": 1, "strategy": "📈 모멘텀"}]
    exporter.send_paper_trading_email({"date": "2026-10-16"}, pos, prices={"AAA": 11.0})
    html = sent["html"]
    assert "공매도 22% · 되사기 6.1일분 (9/30 기준) · 🔥 스퀴즈 진행 가능" in html
    assert "🔥 숏스퀴즈 감시 — 진행 가능 1 · 대기 1" in html and "POOL1" not in html


def test_build_dashboard_marks_candidates_with_squeeze(monkeypatch, tmp_path):
    """후보(candidates)에도 숏스퀴즈 상태가 들어간다 — 후보는 infos 를 복사하므로 그 전에 채워야 한다."""
    from web_export import dashboard as wd
    ranked = pd.DataFrame({"티커": ["SQZ", "AAA"], "회사": ["S", "A"], "판단": ["1. 매수 후보"] * 2,
                           "매수적합도_표시": ["★7.0"] * 2, "_bar_date": ["2026-10-15"] * 2})
    monkeypatch.setattr(wd, "load_ranked", lambda d: ranked)
    monkeypatch.setattr(wd, "build_charts", lambda *a, **k: [])
    monkeypatch.setattr(wd, "squeeze_block", lambda d=None: {"settle": "2026-09-30", "items": [{"ticker": "SQZ", "status": "fire"}]})
    data = wd.build_dashboard(tmp_path, with_prices=False, with_insider=False)
    cands = {c["ticker"]: c["squeeze"] for c in data["candidates"]}
    assert cands.get("SQZ") == "fire" and cands.get("AAA") is None
