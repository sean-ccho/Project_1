"""scripts/idea_batch_research.py · scripts/fetch_insider_data.py 테스트 — 신호 조건·미래정보·비중·진입 시점."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetch_insider_data import map_tickers, normalize_symbol, parse_tables
from idea_batch_research import (active_from_entries, breakout_volume_signal, calendar_time_returns,
                                 earnings_premium_entries, episodes, event_stats, final_label, insider_entries,
                                 rsi2_pullback_signal, rsi_wilder, sector_momentum_returns, signal_entries,
                                 vol_managed_weights)
from paper_trading.universe import Membership

IDX = pd.bdate_range("2015-01-01", periods=320)


def _members(*tickers: str) -> Membership:
    return Membership([("2000-01-01", frozenset(tickers))])


def _all_true(index: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(True, index=index, columns=cols)


# ── 신호 ───────────────────────────────────────────────────────────────────────────────────


def test_rsi_wilder_matches_ta_after_warmup():
    from ta.momentum import RSIIndicator
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=IDX[:200])
    ours = rsi_wilder(close.to_frame("A"), 2)["A"]
    ref = RSIIndicator(close, window=2).rsi()
    assert np.allclose(ours.iloc[60:], ref.iloc[60:], atol=1e-6)


def test_rsi2_pullback_needs_uptrend_and_two_down_days():
    up = 100 * 1.002 ** np.arange(260)
    close = np.concatenate([up, [up[-1] * 0.98, up[-1] * 0.96]])
    df = pd.DataFrame({"A": close}, index=IDX[:262])
    sig = rsi2_pullback_signal(df, _all_true(df.index, ["A"]))["A"]
    assert not sig.iloc[:260].any()          # 계속 오르기만 하면 RSI(2) 높음
    assert sig.iloc[-1]                      # 200일선 위에서 이틀 급락 → RSI(2) < 10
    # 하락장(200일선 아래)에서는 같은 급락이어도 신호 없음
    down = 100 * 0.998 ** np.arange(260)
    df2 = pd.DataFrame({"A": np.concatenate([down, [down[-1] * 0.98, down[-1] * 0.96]])}, index=IDX[:262])
    assert not rsi2_pullback_signal(df2, _all_true(df2.index, ["A"]))["A"].iloc[-1]


def test_rsi2_pullback_threshold_is_10_not_looser():
    up = list(100 * 1.002 ** np.arange(260))
    close = pd.DataFrame({"A": up + [up[-1] * 0.993]}, index=IDX[:261])  # 상승 중 하루 소폭 하락 (RSI(2) ≈ 22)
    rsi = rsi_wilder(close, 2)["A"].iloc[-1]
    assert 10 < rsi < 30                                                  # 10 과 30 사이 → 신호 아님
    assert not rsi2_pullback_signal(close, _all_true(close.index, ["A"]))["A"].iloc[-1]
    close2 = pd.DataFrame({"A": up + [up[-1] * 0.975]}, index=IDX[:261])  # 더 큰 하락 (RSI(2) ≈ 7)
    rsi2 = rsi_wilder(close2, 2)["A"].iloc[-1]
    assert 5 < rsi2 < 10                                                  # 5 와 10 사이 → 신호
    assert rsi2_pullback_signal(close2, _all_true(close2.index, ["A"]))["A"].iloc[-1]


def test_rsi2_signal_has_no_lookahead():
    rng = np.random.default_rng(1)
    close = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 300)))}, index=IDX[:300])
    full = rsi2_pullback_signal(close, _all_true(close.index, ["A"]))
    cut = rsi2_pullback_signal(close.iloc[:250], _all_true(close.index[:250], ["A"]))
    assert full.iloc[:250].equals(cut)


def test_breakout_volume_signal():
    n = 300
    close = pd.DataFrame({"A": np.full(n, 100.0), "B": np.full(n, 100.0), "C": np.full(n, 100.0)}, index=IDX[:n])
    vol = pd.DataFrame(1000.0, index=close.index, columns=close.columns)
    close.iloc[-1] = [101.0, 101.0, 100.0]      # A·B 신고가, C 는 아님
    vol.iloc[-1] = [2500.0, 1500.0, 5000.0]     # A 만 거래량 2배 이상
    sig = breakout_volume_signal(close, vol, _all_true(close.index, list(close.columns)))
    assert sig.iloc[-1].tolist() == [True, False, False]
    assert not sig.iloc[:-1].any().any()


def test_signal_entries_enter_next_day_and_drop_last_day():
    sig = pd.DataFrame(False, index=IDX[:5], columns=["A", "B"])
    sig.iloc[1, 0] = True
    sig.iloc[4, 1] = True  # 마지막 날 신호는 다음날이 없어 버림
    ev = signal_entries(sig)
    assert ev[["티커", "entry_pos"]].values.tolist() == [["A", 2]]


# ── 진입 목록 ──────────────────────────────────────────────────────────────────────────────


def test_insider_entries_filters_role_membership_and_dedups():
    idx = pd.bdate_range("2016-01-04", periods=20)
    trades = pd.DataFrame({
        "filing_date": pd.to_datetime(["2016-01-05", "2016-01-05", "2016-01-06", "2016-01-07", "2016-01-09"]),
        "티커": ["AAA", "AAA", "BBB", "CCC", "AAA"],
        "is_officer": [True, False, False, True, False],
        "is_director": [False, True, False, False, True],
    })
    mem = _members("AAA", "BBB", "CCC")
    ev = insider_entries(trades, idx, pd.Index(["AAA", "BBB"]), mem)
    # BBB = 10% 대주주만(임원·이사 아님) 제외, CCC = 가격 열 없음 제외, AAA 같은 날 2건 → 1건
    assert ev["티커"].tolist() == ["AAA", "AAA"]
    # 2016-01-05(화) 제출 → 1-06 진입. 2016-01-09(토) 제출 → 다음 거래일 1-11(월) 진입
    assert [idx[p] for p in ev["entry_pos"]] == [pd.Timestamp("2016-01-06"), pd.Timestamp("2016-01-11")]


def test_insider_entries_require_membership_on_filing_date():
    idx = pd.bdate_range("2016-01-04", periods=10)
    trades = pd.DataFrame({"filing_date": pd.to_datetime(["2016-01-05"]), "티커": ["AAA"],
                           "is_officer": [True], "is_director": [False]})
    mem = Membership([("2000-01-01", frozenset({"BBB"})), ("2016-01-06", frozenset({"AAA", "BBB"}))])
    assert insider_entries(trades, idx, pd.Index(["AAA"]), mem).empty  # 제출일엔 아직 구성종목 아님


def test_earnings_premium_entries_projection_skip_and_hit_rate():
    idx = pd.bdate_range("2016-01-01", "2017-12-29")
    dates = pd.DataFrame({
        "event_date": pd.to_datetime(["2016-03-10", "2017-03-09", "2016-03-10", "2017-02-20"]),
        "티커": ["AAA", "AAA", "BBB", "BBB"],
    })
    ev, hit = earnings_premium_entries(dates, idx, pd.Index(["AAA", "BBB"]), _members("AAA", "BBB"))
    # AAA: 2016-03-10 + 364일 = 2017-03-09(목) = E → E−5 진입. BBB: 2017-02-20 에 이미 발표 → 제외
    assert ev["티커"].tolist() == ["AAA"]
    e_pos = idx.get_loc(pd.Timestamp("2017-03-09"))
    assert ev["entry_pos"].iloc[0] == e_pos - 5
    assert ev["signal_date"].iloc[0] == idx[e_pos - 6]
    assert hit == 1.0  # 실제 2017-03-09 가 [E−5, E+1] 안


# ── 포트폴리오 ─────────────────────────────────────────────────────────────────────────────


def test_active_window_and_extension():
    entries = pd.DataFrame({"티커": ["A", "A"], "entry_pos": [2, 4]})
    act = active_from_entries(entries, IDX[:10], pd.Index(["A", "B"]), hold=3)
    assert act["A"].tolist() == [False, False, True, True, True, True, True, False, False, False]
    assert not act["B"].any()


def test_calendar_time_returns_cap_spy_fill_and_cost():
    idx = IDX[:4]
    r_oo = pd.DataFrame({"A": [0.01] * 4, "B": [0.0] * 4}, index=idx)
    spy = pd.Series(0.005, index=idx)
    act = pd.DataFrame({"A": [False, True, True, False], "B": [False, True, True, False]}, index=idx)
    ret, n, w_sum = calendar_time_returns(act, r_oo, _all_true(idx, ["A", "B"]), spy, max_weight=0.05, cost=0.001)
    assert n.tolist() == [0, 2, 2, 0]
    assert w_sum.tolist() == pytest.approx([0.0, 0.10, 0.10, 0.0])
    assert ret.iloc[0] == pytest.approx(0.005 - 1.0 * 0.001)                          # 첫날 SPY 100% 매수
    assert ret.iloc[1] == pytest.approx(0.05 * 0.01 + 0.9 * 0.005 - 0.2 * 0.001)     # 종목 2개 5%씩, SPY 90%
    assert ret.iloc[2] == pytest.approx(0.05 * 0.01 + 0.9 * 0.005)                   # 비중 그대로 → 비용 없음
    assert ret.iloc[3] == pytest.approx(0.005 - 0.2 * 0.001)                         # 종목 매도, SPY 100%


def test_calendar_time_returns_equal_weight_without_cap_and_untradable_day():
    idx = IDX[:2]
    cols = [f"T{i}" for i in range(30)]
    r_oo = pd.DataFrame(0.01, index=idx, columns=cols)
    act = pd.DataFrame(True, index=idx, columns=cols)
    tradable = _all_true(idx, cols)
    tradable.iloc[1, 0] = False  # 둘째 날 T0 시가 없음 → 그날 제외
    _, n, w_sum = calendar_time_returns(act, r_oo, tradable, pd.Series(0.0, index=idx))
    assert n.tolist() == [30, 29]
    assert w_sum.tolist() == pytest.approx([1.0, 1.0])  # 1/30 < 5% → 전부 종목, SPY 0


def test_sector_momentum_picks_top3_from_next_open():
    idx = pd.bdate_range("2015-01-01", periods=300)
    growth = {"XLB": 1.0001, "XLE": 1.0004, "XLK": 1.0008, "XLV": 1.0006, "SPY": 1.0003}
    frames = {}
    for t, g in growth.items():
        px = 100 * g ** np.arange(len(idx))
        frames[t] = pd.DataFrame({"Open": px, "Close": px}, index=idx)
    etf = pd.concat(frames, axis=1)
    spy_oo = etf["SPY"]["Open"].shift(-1) / etf["SPY"]["Open"] - 1
    ret, w = sector_momentum_returns(etf, spy_oo, cost=0.0)
    first_me = pd.Timestamp("2015-12-31")  # 252일 이력이 처음 생기는 월말
    nxt = idx[idx.get_loc(first_me) + 1]
    assert w.loc[nxt].to_dict() == pytest.approx({"XLB": 0.0, "XLE": 1 / 3, "XLK": 1 / 3, "XLV": 1 / 3})
    assert (w.loc[:first_me] == 0).all().all()                    # 첫 교체 전엔 섹터 비중 0 (SPY)
    assert ret.loc[first_me] == pytest.approx(spy_oo.loc[first_me])


def test_vol_managed_weights_month_end_rule():
    idx = pd.bdate_range("2015-01-01", periods=130)
    rng = np.random.default_rng(2)
    rets = np.concatenate([rng.normal(0, 0.005, 65), rng.normal(0, 0.03, 65)])  # 조용하다가 요동
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    w = vol_managed_weights(close, target=0.15)
    me = pd.Timestamp("2015-05-29")
    sigma = close.pct_change().rolling(21).std().loc[me] * math.sqrt(252)
    assert w.loc[me] == pytest.approx(min(1.0, 0.15 / sigma))
    assert w.loc[me] < 1.0                                         # 요동 구간 → 비중 축소
    assert (w.loc[me:pd.Timestamp("2015-06-29")] == w.loc[me]).all()  # 다음 월말 전까지 유지
    assert w.loc[pd.Timestamp("2015-01-30")] == 1.0                # 21일 안 된 첫 월말 → 100%


# ── 통계·판정 ──────────────────────────────────────────────────────────────────────────────


def test_episodes_counts_overlapping_reentry_once():
    entries = pd.DataFrame({"티커": ["A", "A", "A", "B"], "entry_pos": [10, 12, 20, 11]})
    kept = episodes(entries, hold=5)
    assert sorted(zip(kept["티커"], kept["entry_pos"])) == [("A", 10), ("A", 20), ("B", 11)]


def test_event_stats_excess_vs_member_equal_weight():
    idx = pd.bdate_range("2016-01-04", periods=10)
    opens = pd.DataFrame({"A": 100 * 1.02 ** np.arange(10), "B": np.full(10, 100.0)}, index=idx)
    entries = pd.DataFrame({"티커": ["A"], "entry_pos": [1]})
    r = event_stats(entries, opens, _all_true(idx, ["A", "B"]), hold=2)
    fwd_a, fwd_b = 1.02 ** 2 - 1, 0.0
    assert r["n_events"] == 1
    assert r["mean_ar"] == pytest.approx(fwd_a - (fwd_a + fwd_b) / 2)


def test_final_label():
    good = {"ΔSharpe_하한": 0.1, "p_개선아님": 0.001}
    weak = {"ΔSharpe_하한": 0.05, "p_개선아님": 0.02}
    assert final_label("event", "채택 후보", good, "후보") == "채택 후보"
    assert final_label("event", "채택 후보", good, "관찰") == "관찰"       # 이벤트 판정이 모자람
    assert final_label("etf", "채택 후보", weak, None) == "관찰"           # Bonferroni 미달
    assert final_label("event", "운과 구분 안 됨", {"ΔSharpe_하한": -0.2, "p_개선아님": 0.4}, "역신호") == "역신호"
    assert final_label("etf", "운과 구분 안 됨", {"ΔSharpe_하한": -0.2, "p_개선아님": 0.4}, None) == "탈락"


# ── 내부자 원자료 파싱 ─────────────────────────────────────────────────────────────────────


def test_parse_tables_keeps_form4_open_market_purchases():
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a2", "a3"], "FILING_DATE": ["11-FEB-2016", "12-FEB-2016", "15-FEB-2016"],
                        "DOCUMENT_TYPE": ["4", "4/A", "4"], "ISSUERCIK": ["19617", "19617", "320193"],
                        "ISSUERTRADINGSYMBOL": ["JPM", "JPM", "AAPL"]})
    owners = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a2", "a3", "a3"], "RPTOWNERCIK": ["1", "1", "2", "3"],
                           "RPTOWNER_RELATIONSHIP": ["Director,Officer", "Director,Officer", "TenPercentOwnerOther",
                                                     "Other"]})
    trans = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a1", "a1", "a2", "a3"],
                          "NONDERIV_TRANS_SK": ["1", "2", "5", "3", "4"],
                          "TRANS_DATE": ["11-FEB-2016"] * 4 + ["12-FEB-2016"], "TRANS_CODE": ["P", "S", "P", "P", "P"],
                          "TRANS_SHARES": ["500000", "10", "7", "1", "100"],
                          "TRANS_PRICEPERSHARE": ["53.2", "60", "53", "50", "95"],
                          "TRANS_ACQUIRED_DISP_CD": ["A", "D", "D", "A", "A"]})
    out = parse_tables(sub, owners, trans)
    # a1 매수(P·A)만 (매도 S, 처분 D 로 적힌 P 제외), a2 = 정정본 4/A 제외, a3 = 10% 대주주+기타
    assert out["accession"].tolist() == ["a1", "a3"]
    jpm = out.iloc[0]
    assert jpm["filing_date"] == pd.Timestamp("2016-02-11")
    assert bool(jpm["is_director"]) and bool(jpm["is_officer"]) and not bool(jpm["is_ten_pct"])
    assert jpm["value"] == pytest.approx(500000 * 53.2)
    other = out.iloc[1]
    assert bool(other["is_ten_pct"]) and not bool(other["is_director"]) and not bool(other["is_officer"])
    assert other["n_owners"] == 2


def test_map_tickers_prefers_cik_and_drops_symbol_conflicts():
    df = pd.DataFrame({"issuer_cik": [1652044, 999, 888, 777], "symbol": ["googl", "brk.b", "AAPL", ""]})
    cik_map = pd.DataFrame({"티커": ["GOOG", "GOOGL", "AAPL"], "cik": [1652044.0, 1652044.0, 320193.0]})
    out = map_tickers(df, cik_map)
    # CIK 매핑(같은 CIK 여러 티커면 신고서 기호와 같은 것), CIK 없는 BRK.B → BRK-B,
    # 다른 CIK 인데 AAPL 기호 → 버림, 기호 없음 → 버림
    assert out[["티커", "ticker_source"]].values.tolist() == [["GOOGL", "cik"], ["BRK-B", "symbol"]]
    assert normalize_symbol(" nyse: x ") == ""
