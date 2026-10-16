"""웹사이트용 dashboard.json 내보내기 — 파싱·요약·차트 목록 (네트워크·git 없음)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import web_export.dashboard as wd


def test_parse_score_and_health() -> None:
    assert wd.parse_score("★★★★★ (11.5)") == 11.5
    assert wd.parse_score("") is None
    assert wd.parse_health("100 (A+) · 매출+48% 마진+25% 이익+86%") == {
        "score": 100, "grade": "A+", "detail": "매출+48% 마진+25% 이익+86%"}
    assert wd.parse_health("55 (D)") == {"score": 55, "grade": "D", "detail": None}
    assert wd.parse_health(float("nan")) is None


def test_label_strips_numbering_and_emoji() -> None:
    assert wd._label("1. 매수 후보") == "매수 후보"
    assert wd._label("📈 모멘텀") == "모멘텀"
    assert wd._label("—") is None
    assert wd._patterns("이중천장, 골든크로스") == ["이중천장", "골든크로스"]


def test_ticker_info_handles_missing_values() -> None:
    row = pd.Series({"회사": "Figma, Inc.", "현재가격": 21.76, "전략구분": "—", "매수적합도_표시": "★★★★☆ (4.5)",
                     "판단": "5. 관망 약세", "[헬스체크]": "55 (D) · 매출+48%", "RSI": float("nan"),
                     "next_earnings_date": pd.NaT, "일봉패턴": None, "market": "sp500"})
    info = wd.ticker_info(row)
    assert info["strategy"] is None and info["verdict"] == "관망 약세" and info["score"] == 4.5
    assert info["rsi"] is None and info["earnings_date"] is None and info["patterns"]["daily"] == []
    json.dumps(info, allow_nan=False)  # NaN 이 남아 있으면 실패


def test_trade_summary_groups_exit_reasons() -> None:
    trades = [
        {"ticker": "A", "entry_date": "2026-04-02", "exit_date": "2026-04-10", "return_pct": 0.2, "exit_reason": "목표가(+20%)"},
        {"ticker": "B", "entry_date": "2026-04-05", "exit_date": "2026-04-20", "return_pct": -0.05, "exit_reason": "트레일링(-5%)"},
        {"ticker": "C", "entry_date": "2026-04-08", "exit_date": "2026-04-15", "return_pct": -0.03, "exit_reason": "트레일링(-3%)"},
    ]
    s = wd.trade_summary(trades)
    assert s["count"] == 3 and s["wins"] == 1 and s["first_entry"] == "2026-04-02"
    assert [r["reason"] for r in s["by_reason"]] == ["목표가", "트레일링"]
    assert s["by_reason"][1]["count"] == 2
    assert [h["return"] for h in s["history"]] == [0.2, -0.03, -0.05]  # 청산일 순
    assert s["history"][1]["ticker"] == "C"
    assert s["recent"][0]["ticker"] == "B"  # 최근 청산이 먼저


def test_list_chart_files_keeps_latest_per_timeframe(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "charts" / "screenshots_sp500"
    for name in ("NBM.V/NBM.V_Daily_20261014_063540.png", "NBM.V/NBM.V_Daily_20261015_062204.png",
                 "FIG/FIG_Weekly_20261015_063508.png", "FIG/notes.txt"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    found = wd.list_chart_files("charts/screenshots_sp500", "screenshots-sp500")
    assert found == {
        "NBM.V": {"Daily": "charts/screenshots_sp500/NBM.V/NBM.V_Daily_20261015_062204.png"},
        "FIG": {"Weekly": "charts/screenshots_sp500/FIG/FIG_Weekly_20261015_063508.png"},
    }
    assert wd._captured_at(found) == "2026-10-15T06:35:08Z"


def test_ticker_info_fundamentals_keep_yfinance_units() -> None:
    row = pd.Series({"회사": "Airbnb", "fund_revenue_growth": 0.165, "fund_debt_to_equity": 32.004,
                     "fund_free_cashflow": 3206874880.0, "fund_earnings_growth": float("nan"),
                     "52주포지션": 0.7645, "시장상대강도": -0.034})
    info = wd.ticker_info(row)
    f = info["fundamentals"]
    assert f["revenue_growth"] == 0.165 and f["debt_to_equity"] == 32.0 and f["free_cashflow"] == 3206874880.0
    assert f["earnings_growth"] is None and f["roe"] is None  # NaN·컬럼 없음 → None
    assert info["position_52w"] == 0.7645 and info["rel_market"] == -0.034 and info["insider"] is None
    json.dumps(info, allow_nan=False)


def test_insider_info_signals(monkeypatch) -> None:
    import screener.insider as ins

    df = pd.DataFrame([
        {"티커": "AAA", "insider_buy_count": 3, "insider_sell_count": 0, "insider_net_value": 860000.0,
         "insider_last_trade_date": "2026-10-01", "내부자(90일)": "🟢 클러스터매수 · 매수 3/매도 0 · 순 +$860K (~10-01)"},
        {"티커": "BBB", "insider_buy_count": 1, "insider_sell_count": 2, "insider_net_value": -1.2e6,
         "insider_last_trade_date": "2026-09-20", "내부자(90일)": "🟢 매수 · 매수 1/매도 2 · 순 -$1.2M (~09-20)"},
        {"티커": "CCC", "insider_buy_count": 0, "insider_sell_count": 4, "insider_net_value": -5e5,
         "insider_last_trade_date": "2026-09-02", "내부자(90일)": "🔴 매도 · 매수 0/매도 4 · 순 -$500K (~09-02)"},
        {"티커": "DDD", "insider_buy_count": 0, "insider_sell_count": 0, "insider_net_value": 0.0,
         "insider_last_trade_date": "", "내부자(90일)": "—"},
    ])
    monkeypatch.setattr(ins, "fetch_insider_snapshots", lambda tickers, use_cache=True: df)
    out = wd.insider_info(["AAA", "BBB", "CCC", "DDD"], cache_only=False)
    assert [out[t]["signal"] for t in ("AAA", "BBB", "CCC", "DDD")] == ["cluster_buy", "buy", "sell", None]
    assert out["BBB"] == {"buys": 1, "sells": 2, "net_value": -1200000.0, "last_date": "2026-09-20", "signal": "buy"}
    assert out["DDD"]["last_date"] is None


def test_insider_info_failure_is_empty(monkeypatch) -> None:
    import screener.insider as ins

    def boom(tickers, use_cache=True):
        raise RuntimeError("openinsider down")

    monkeypatch.setattr(ins, "fetch_insider_snapshots", boom)
    assert wd.insider_info(["AAA"], cache_only=False) == {}


def test_insider_info_cache_only_skips_uncached(monkeypatch) -> None:
    import screener.insider as ins

    asked: list[list[str]] = []

    def fake(tickers, use_cache=True):
        asked.append(list(tickers))
        return pd.DataFrame([{"티커": t, "insider_buy_count": 1, "insider_sell_count": 0, "insider_net_value": 1.0,
                              "insider_last_trade_date": "2026-10-01", "내부자(90일)": "🟢 매수"} for t in tickers])

    monkeypatch.setattr(ins, "fetch_insider_snapshots", fake)
    monkeypatch.setattr(ins, "has_fresh_cache", lambda t: t == "AAA")
    assert list(wd.insider_info(["AAA", "BBB"])) == ["AAA"] and asked == [["AAA"]]  # 캐시 없는 BBB 는 조회 안 함
    monkeypatch.setattr(ins, "has_fresh_cache", lambda t: False)
    assert wd.insider_info(["BBB"]) == {} and len(asked) == 1  # 전부 없으면 조회 자체를 안 함


def test_golden_cross_lists_by_market() -> None:
    ranked = pd.DataFrame([
        {"티커": "AAA", "market": "sp500", "일봉패턴": "골든크로스임박", "주봉패턴": "골든크로스", "월봉패턴": "",
         "ema_gap_20_50": -0.012, "주봉_MA갭": 0.02, "월봉_MA갭": 0.3},
        {"티커": "BBB", "market": "nasdaq", "일봉패턴": "골든크로스", "주봉패턴": "", "월봉패턴": "",
         "ema_gap_20_50": 0.2, "주봉_MA갭": 0.0, "월봉_MA갭": 0.0},  # 5% 넘게 벌어진 교차는 빠진다
        {"티커": "CCC", "market": "nasdaq", "일봉패턴": "데드크로스", "주봉패턴": "", "월봉패턴": "골든크로스임박",
         "ema_gap_20_50": -0.1, "주봉_MA갭": 0.0, "월봉_MA갭": -0.03},
    ])
    out = wd.golden_cross_lists(ranked)
    assert out["sp500"] == [{"ticker": "AAA", "timeframes": [{"tf": "daily", "gap": -0.012}, {"tf": "weekly", "gap": 0.02}]}]
    assert out["nasdaq"] == [{"ticker": "CCC", "timeframes": [{"tf": "monthly", "gap": -0.03}]}]
    info = wd.ticker_info(pd.Series({"거래량돌파배수": 2.345, "ema_gap_20_50": -0.0123, "주봉_MA갭": float("nan")}))
    assert info["vol_ratio"] == 2.35 and info["ma_gap"] == {"daily": -0.0123, "weekly": None, "monthly": None}
    assert info["golden"] is None


def test_benchmark_block(monkeypatch):
    """SPY 종가 → [[날짜, 값], ...] (tz 제거, 소수 4자리). 받기 실패면 None."""
    import pandas as pd
    import paper_trading.benchmarks as bm
    from web_export.dashboard import benchmark_block

    idx = pd.DatetimeIndex(["2026-10-14", "2026-10-15"]).tz_localize("America/New_York")
    monkeypatch.setattr(bm, "load_spy_close", lambda period="6y": pd.Series([600.123456, 601.5], index=idx))
    assert benchmark_block() == {"ticker": "SPY", "closes": [["2026-10-14", 600.1235], ["2026-10-15", 601.5]]}
    monkeypatch.setattr(bm, "load_spy_close", lambda period="6y": pd.Series(dtype=float))
    assert benchmark_block() is None


def test_all_charts_merges_sources_and_orders_by_score(monkeypatch, tmp_path) -> None:
    """전체차트: 두 시장의 모든 종목 (점수 순, 지수 ETF 제외), 이미지는 차트분석 PNG 먼저 → 전체차트 WebP."""
    from screener import config as cfg
    monkeypatch.chdir(tmp_path)
    for d, f in (("charts/sp500x/AAA", "AAA_Daily_20261016_060000.png"), ("charts/allx/BBB", "BBB_Daily_20261016_070000.webp"),
                 ("charts/allx/CCC", "CCC_Weekly_20261016_070000.webp")):
        Path(d).mkdir(parents=True)
        (Path(d) / f).write_bytes(b"x")
    monkeypatch.setattr(cfg, "CHARTS_OUTPUT_DIR", "charts/sp500x")
    monkeypatch.setattr(cfg, "CHARTS_SIGNALS2_OUTPUT_DIR", "charts/nasdaqx")
    monkeypatch.setattr(wd, "ALL_CHARTS", ("charts/allx", "screenshots-all"))
    ranked = pd.DataFrame({"티커": ["AAA", "BBB", "SPY", "CCC"], "매수적합도": [5.0, 7.0, 1.0, 6.0],
                           "market": ["sp500", "sp500", "sp500", "nasdaq"]})
    out = wd.build_all_charts(ranked, {"AAA": 1}, "o/r")
    sp = next(m for m in out if m["key"] == "sp500")
    assert [it["ticker"] for it in sp["items"]] == ["BBB", "AAA"]
    assert sp["items"][0]["images"]["Daily"].endswith("/screenshots-all/charts/allx/BBB/BBB_Daily_20261016_070000.webp")
    assert sp["items"][1]["held"] and "screenshots-sp500" in sp["items"][1]["images"]["Daily"]
    nq = next(m for m in out if m["key"] == "nasdaq")
    assert nq["items"][0]["images"] == {"Weekly": nq["items"][0]["images"]["Weekly"]}


def test_trade_log_with_spy() -> None:
    idx = pd.to_datetime(["2026-10-01", "2026-10-08", "2026-10-15"])
    spy = pd.Series([100.0, 102.0, 101.0], index=idx)
    trades = [{"ticker": "A", "entry_date": "2026-10-01", "exit_date": "2026-10-08", "return_pct": 0.05,
               "entry_price": 10, "exit_price": 10.5, "exit_reason": "목표가", "strategy": "📈 모멘텀"},
              {"ticker": "B", "entry_date": "2026-10-08", "exit_date": "2026-10-15", "return_pct": -0.02}]
    log = wd.trade_log(trades, spy)
    assert [x["ticker"] for x in log] == ["B", "A"]                      # 최근 청산 먼저
    assert log[1]["spy_return"] == 0.02 and log[1]["vs_spy"] == 0.03
    assert wd.trade_log(trades)[0]["spy_return"] is None
