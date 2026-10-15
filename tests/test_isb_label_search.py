"""ISB 라벨 조합 탐색 테스트 — 규칙 목록, 사용자 거래·봉별 표, T1 전량 매핑, 워크포워드 선택, 단독 수익, 조회 표."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import isb_engine as e  # noqa: E402
import isb_label_search as ls  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
from paper_trading.evaluation import max_drawdown, sharpe  # noqa: E402

HAS_DATA = e.PRICE_CACHE.exists() and e.MACRO_PATH.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="가격 캐시·VIX/DXY 파일 없음 (로컬 전용)")


# ── 규칙 목록 ─────────────────────────────────────────────
def test_all_rules_counts_keys_and_validity():
    rules = e.all_rules()
    assert len(rules) == 686 and sum(r.tf == "D" for r in rules) == 252 and sum(r.tf == "W" for r in rules) == 434
    keys = [r.key for r in rules]
    assert len(set(keys)) == len(keys)
    assert all(e.LabelRule.parse(k) == r for k, r in zip(keys, rules))
    assert all(r.exit or r.t1 == "full" or r.top == "full" for r in rules)        # 전량 매도 수단이 꼭 있다
    assert all(r.t1 == "off" for r in rules if r.tf == "D")                        # 일봉엔 T1 이 없다
    assert e.native_rule("D") in rules and e.native_rule("W") in rules and e.WBB_RULE in rules
    assert e.native_rule("W").key == "W:PB+BO+EARLY+REV:X1:T1trim:TOPtrim"
    assert not e.LabelRule("W", ("BOTTOM", "BO"), True, "off", "off").valid()      # 라벨 순서가 정해져 있다
    assert not e.LabelRule("W", ("BO",), False, "trim", "trim").valid()
    assert e.LabelRule("W", ("BO", "BOTTOM"), True, "full", "off").describe() == \
        "주봉 · 매수 BO·BOTTOM · 매도 EXIT 전량 · TRIM(T1) 전량 · TOP 무시"
    assert ls.N_TESTS == 2 and ls.P_BONF == pytest.approx(0.025)


# ── 사용자 거래 · 봉별 표 ─────────────────────────────────
def _signals():
    # 1봉 BO: 5봉 시가 TOP 트림, 7봉 장중 T1, 9봉 EXIT(Trail) → 10봉 시가 · 3봉 BOTTOM(보유 중이라 무시) · 12봉 PB
    return [
        {"s": 1, "kind": "BO", "grade": 3, "stop": 9.0, "x": 9, "why": "Trail", "t1": 14.0, "trims": (("TOP", 5), ("T1", 7))},
        {"s": 3, "kind": "BOTTOM", "grade": 0, "stop": 8.0, "x": 6, "why": "Stop", "t1": 12.0, "trims": ()},
        {"s": 12, "kind": "PB", "grade": 2, "stop": 9.5, "x": -1, "why": "", "t1": 13.0, "trims": ()},
    ]


def test_rule_trades_end_and_partials():
    sig, n = _signals(), 15
    base = e.rule_trades(sig, n, e.LabelRule("W", ("PB", "BO"), True, "trim", "trim"))
    assert [t["kind"] for t in base] == ["BO", "PB"]
    assert base[0]["end"][:3] == (10, 0, "EXIT·Trail") and [x[:2] for x in base[0]["part"]] == [(5, 0), (7, 1)]
    assert base[0]["part"][1][2] == 14.0                                               # T1 가격은 그 거래의 T1
    assert base[1]["end"] is None and base[1]["part"] == []
    t1full = e.rule_trades(sig, n, e.LabelRule("W", ("BO",), True, "full", "trim"))
    assert t1full[0]["end"] == (7, 1, "T1", 14.0) and [x[:2] for x in t1full[0]["part"]] == [(5, 0)]
    topfull = e.rule_trades(sig, n, e.LabelRule("W", ("BO", "BOTTOM"), False, "trim", "full"))
    # TOP 전량(5봉 시가)에서 끝나므로 3봉 BOTTOM 은 보유 중이라 무시된다
    assert topfull[0]["end"][:3] == (5, 0, "TOP") and topfull[0]["part"] == [] and len(topfull) == 1
    only_btm = e.rule_trades(sig, n, e.LabelRule("W", ("BOTTOM",), True, "off", "off"))
    assert [t["s"] for t in only_btm] == [3] and only_btm[0]["end"][:3] == (7, 0, "EXIT·Stop")


def test_rule_trades_ignoring_exit_follows_later_chart_labels():
    # 1봉 BO 거래는 9봉 EXIT 로 끝난다 (그 안의 라벨: 5봉 시가 TOP 트림, 7봉 T1).
    # 차트에는 그 뒤에도 TOP 신호(4봉은 거래 안이라 제외, 11봉 ◆TOP, 16봉)와 다음 ISB 거래의 T1(13봉, 21.0)이 뜬다.
    sig = [{"s": 1, "kind": "BO", "grade": 3, "stop": 9.0, "x": 9, "why": "Trail", "t1": 14.0,
            "trims": (("TOP", 5), ("T1", 7))}]
    nat = {"top": np.array([4, 11, 16]), "t1": [(13, 21.0)]}
    n = 20
    top_only = e.rule_trades(sig, n, e.LabelRule("W", ("BO",), False, "off", "full"), nat)
    assert top_only[0]["end"][:3] == (5, 0, "TOP")                                     # 거래 안의 TOP 이 먼저
    no_top_in_trade = [{**sig[0], "trims": (("T1", 7),)}]
    later = e.rule_trades(no_top_in_trade, n, e.LabelRule("W", ("BO",), False, "off", "full"), nat)
    assert later[0]["end"][:3] == (12, 0, "TOP")                                       # EXIT 뒤 ◆TOP(11봉) → 12봉 시가
    t1_later = e.rule_trades(no_top_in_trade, n, e.LabelRule("W", ("BO",), False, "full", "off"), nat)
    assert t1_later[0]["end"] == (7, 1, "T1", 14.0)
    after_t1 = [{**sig[0], "trims": ()}]
    t1_next = e.rule_trades(after_t1, n, e.LabelRule("W", ("BO",), False, "full", "trim"), nat)
    assert t1_next[0]["end"] == (13, 1, "T1", 21.0) and [x[:2] for x in t1_next[0]["part"]] == [(12, 0)]
    capped = e.rule_trades(sig, n, e.LabelRule("W", ("BO",), False, "full", "trim"), nat)
    assert capped[0]["end"][:2] == (7, 1) and len(capped[0]["part"]) == 1              # 5봉 TOP 트림 뒤 7봉 T1 전량
    many = e.rule_trades(no_top_in_trade, n, e.LabelRule("W", ("BO",), False, "trim", "full"),
                         {"top": np.array([18]), "t1": [(13, 21.0), (15, 23.0)]})
    assert [x[:2] for x in many[0]["part"]] == [(7, 1), (13, 1)] and many[0]["end"][:3] == (19, 0, "TOP")   # ⅓ 은 최대 2회
    keep = e.rule_trades(sig, n, e.LabelRule("W", ("BO",), True, "off", "full"), nat)
    assert keep[0]["end"][:3] == (5, 0, "TOP")                                         # EXIT 를 따르면 뒤 라벨은 볼 일 없음


def test_rule_frame_trims_and_exits():
    sig, n = _signals(), 15
    idx = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    p = e.IsbParams("W")
    fr = e.rule_frame(e.rule_trades(sig, n, e.LabelRule("W", ("PB", "BO"), True, "trim", "trim")), idx,
                      np.full(n, 2), None, p)
    assert list(fr["pos"]) == [0, 0] + [1] * 8 + [0, 0, 0, 1, 1]
    assert list(fr["hold_next"]) == [False] + [True] * 8 + [False, False, False, True, True, True]
    assert fr["ev_buy"].sum() == 2 and fr["ev_top"].tolist()[4] and fr["ev_t1"].tolist()[7]
    assert fr["ev_exit"].tolist()[9] and fr["exit_why"].iloc[9] == "EXIT·Trail" and not fr["t1_full"].any()
    assert fr["rem"].iloc[4] == 1.0 and fr["rem"].iloc[5] == pytest.approx(0.67)      # TOP 은 5봉 시가에 반영
    assert fr["rem"].iloc[7] == pytest.approx(0.67) and fr["rem"].iloc[8] == pytest.approx(0.34)   # T1 은 다음 봉부터
    assert fr["t1_level"].iloc[7] == 14.0
    full = e.rule_frame(e.rule_trades(sig, n, e.LabelRule("W", ("BO",), True, "full", "off")), idx,
                        np.full(n, 2), None, p)
    assert list(full["pos"][:9]) == [0, 0, 1, 1, 1, 1, 1, 1, 0]                       # 7봉 장중 전량 → 7봉까지 보유
    assert full["t1_full"].tolist()[7] and not full["hold_next"].iloc[7] and not full["ev_exit"].any()


def test_daily_events_t1_full_closes_rest_of_week():
    days = pd.bdate_range("2020-07-06", "2020-07-17")
    d = pd.DataFrame({"Open": 16.0, "High": 18.0, "Low": 15.0, "Close": 17.0, "Volume": 1e6}, index=days)
    d.loc["2020-07-15", "High"] = 25.0                         # 둘째 주 수요일에 T1(20) 도달
    d["adj_factor"] = 0.98
    wk = e.to_weekly(d)
    sig = [{"s": 0, "kind": "BO", "grade": 3, "stop": 15.0, "x": -1, "why": "", "t1": 20.0, "trims": (("T1", 1),)}]
    p = e.IsbParams("W")
    fr = e.rule_frame(e.rule_trades(sig, 2, e.LabelRule("W", ("BO",), True, "full", "off")), wk.index,
                      np.full(2, 2), wk["period"].to_numpy(), p)
    ev = e.daily_events(fr, d, p)
    wk2 = ev.loc["2020-07-13":]
    assert list(wk2["hold_next"]) == [True, True, False, False, False]                # 수요일 장중 전량 → 그날부터 미보유
    assert list(wk2["rem"]) == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0])
    assert wk2.loc["2020-07-15", "t1_fill"] == pytest.approx(20.0 * 0.98) and wk2.loc["2020-07-15", "t1_sold"] == 1.0
    trim = e.daily_events(e.rule_frame(e.rule_trades(sig, 2, e.LabelRule("W", ("BO",), True, "trim", "off")), wk.index,
                                       np.full(2, 2), wk["period"].to_numpy(), p), d, p)
    assert list(trim.loc["2020-07-13":, "hold_next"]) == [True] * 5 and trim.loc["2020-07-15", "t1_sold"] == \
        pytest.approx(0.33)


# ── 워크포워드 ─────────────────────────────────────────────
def test_score_penalizes_drawdown_and_needs_trades():
    idx = pd.bdate_range("2016-01-04", periods=300)
    r = pd.Series(np.r_[np.full(150, 0.002), np.full(150, -0.003)], index=idx)
    v = r.tolist()
    assert max_drawdown(v) < -0.30                                                     # 감점 구간
    assert ls.score(r, 50) == pytest.approx(sharpe(v) - (abs(max_drawdown(v)) - 0.30) * 2.0)
    assert math.isnan(ls.score(r, 49)) and math.isnan(ls.score(r.iloc[:10], 100))


def test_wfo_picks_ties_and_stitch():
    sc = pd.DataFrame({y: [1.0, 2.0, np.nan] for y in ls.FOLDS}, index=["b", "a", "c"])
    sc.loc["c", 2021] = 2.0                                                            # 동점 → 키 순서 ("a")
    sc.loc[:, 2022] = np.nan                                                           # 후보 없음
    picks = ls.wfo_picks(sc)
    assert picks[2020] == "a" and picks[2021] == "a" and picks[2022] is None
    idx = pd.bdate_range("2019-12-30", "2025-09-30")
    rets = {"a": pd.Series(0.001, index=idx), "b": pd.Series(-0.001, index=idx)}
    oos = ls.stitch(rets, {**picks, 2023: "b"}, idx)
    assert oos.index.min() == pd.Timestamp("2020-01-01") and oos.index.max() == pd.Timestamp("2025-09-30")
    assert (oos.loc["2022"] == 0).all() and (oos.loc["2023"] == -0.001).all() and (oos.loc["2021"] == 0.001).all()


def test_fold_scores_use_only_past_years():
    idx = pd.bdate_range("2016-01-04", "2025-09-30")
    r = pd.Series(0.0005, index=idx)
    r.loc["2022"] = -0.01                                                              # 2022 폭락은 2023 폴드부터 보인다
    trades = pd.DatetimeIndex(idx[::5])
    sc, sh = ls.fold_scores({"x": r}, {"x": trades})
    assert sc.at["x", 2022] > 0 and sc.at["x", 2023] < 0
    assert list(sc.columns) == list(ls.FOLDS) and sh.shape == sc.shape


# ── 단독 수익 · 조회 표 ───────────────────────────────────
def test_weighted_np_matches_pandas():
    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.random((30, 4)) / 4)
    r = pd.DataFrame(rng.normal(0, 0.01, (30, 4)))
    r.iloc[3, 2] = np.nan
    assert ls.weighted_np(w.to_numpy(), r.to_numpy()) == pytest.approx(iv.weighted_returns(w, r).to_numpy())


class _Mem:
    def __init__(self, names):
        self.names = frozenset(names)

    def on(self, d):
        return self.names


def _raw():
    idx = pd.date_range("2016-01-04", periods=4, freq="B")
    cols = pd.MultiIndex.from_product([["AAA", "BBB", "SPY"], ["Open", "High", "Low", "Close", "Volume"]],
                                      names=["Ticker", "Price"])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw[("AAA", "Open")] = [10.0, 11.0, 12.1, 12.1]
    raw[("BBB", "Open")] = [20.0, 20.0, 18.0, 18.0]
    for t in ("AAA", "BBB"):
        raw[(t, "High")], raw[(t, "Low")], raw[(t, "Volume")] = 30.0, 5.0, 1e6
    return raw, idx


def test_solo_acc_matches_iv_and_corrects_t1_fill():
    raw, idx = _raw()
    mem = _Mem({"AAA", "BBB"})
    ev = pd.DataFrame({"date": list(idx) * 2, "ticker": ["AAA"] * 4 + ["BBB"] * 4,
                       "pos": [1, 1, 1, 1, 0, 0, 0, 0], "rem": [1.0, 1.0, 0.67, 0.67, 1.0, 1.0, 1.0, 1.0]})
    r_isb, r_ew, wsum = iv.solo_returns(ev, raw, mem)
    ctx = ls.SoloCtx(raw, mem, ["AAA", "BBB"])
    acc = ls.SoloAcc(ctx)
    for t, g in ev.groupby("ticker"):
        acc.add(t, g.set_index("date").assign(t1_fill=np.nan, t1_sold=np.nan))
    got, expo = acc.result()
    assert got.to_numpy() == pytest.approx(r_isb.to_numpy()) and ctx.ew.to_numpy() == pytest.approx(r_ew.to_numpy())
    assert expo == pytest.approx(wsum.mean())
    # 둘째 날 AAA 를 11.5 에 ⅓ 장중 매도 → 그 몫은 11→12.1 대신 11→11.5 수익
    acc2 = ls.SoloAcc(ctx)
    a = ev[ev["ticker"] == "AAA"].set_index("date").assign(t1_fill=np.nan, t1_sold=np.nan)
    a.loc[idx[1], ["t1_fill", "t1_sold"]] = [11.5, 0.33]
    acc2.add("AAA", a)
    got2, _ = acc2.result()
    assert got2.iloc[1] - got.iloc[1] == pytest.approx(0.33 * 0.5 * (11.5 - 12.1) / 11.0)


def test_matbook_get_matches_isbbook():
    idx = pd.date_range("2016-01-04", periods=3, freq="B")
    ev = pd.DataFrame({"date": list(idx) * 2, "ticker": ["AAA"] * 3 + ["BBB"] * 3,
                       "hold_next": [True, False, True, False, False, True], "exit_full": [False] * 5 + [True],
                       "trim_ev": [False, True] + [False] * 4, "t1_fill": [np.nan, 10.5] + [np.nan] * 4,
                       "buy_ev": False, "grade": 0})
    ref = iv.IsbBook(ev.iloc[:5])                                                      # BBB 마지막 날 없음
    book = ls.MatBook(idx, ["AAA", "BBB"])
    for t, g in ev.iloc[:5].groupby("ticker"):
        book.put(t, g.set_index("date"))
    for name in ("hold_next", "exit_full", "trim_ev", "t1_fill"):
        for tk in ("AAA", "BBB", "ZZZ"):
            for d in [f"{x:%Y-%m-%d}" for x in idx] + ["2016-01-01"]:
                a, b = ref.get(name, tk, d, "기본"), book.get(name, tk, d, "기본")
                assert (a == b) or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), (name, tk, d)


def test_overlay_t1_mode_overrides_default():
    assert iv.Overlay("x", tf="W:BO:X1:T1full:TOPoff", gate="isb", sells="isb", t1_mode="full").t1
    assert not iv.Overlay("x", tf="W:BO:X1:T1off:TOPoff", gate="isb", sells="isb", t1_mode="off").t1
    assert not iv.Overlay("x", tf="W", gate="isb", sells="isb", t1_mode="trim", probe=True).t1
    assert iv.Overlay("x", tf="W", gate="isb", sells="isb").t1                       # 기본 판단은 그대로


# ── 실제 데이터 ───────────────────────────────────────────
@pytest.fixture(scope="module")
def data():
    return e.load_daily_prices(), e.load_macro()


@needs_data
@pytest.mark.parametrize("tf", ["D", "W"])
def test_label_rules_reproduce_engine_and_bo_bottom(data, tf):
    prices, macro = data
    p = e.IsbParams(tf)
    mkt = e.build_market(prices["SPY"], macro, p)
    tickers = sorted(t for t in prices if t != "SPY")[::12]
    bad, n_sig = [], 0
    for t in tickers:
        item = e.compute_labels(t, prices[t], mkt, p)
        bad += item["check"]
        n_sig += len(item["signals"])
    assert n_sig > 100 and bad == []


@needs_data
@pytest.mark.parametrize("ticker", ["AAPL", "CSCO", "XOM", "CAT"])
def test_rule_events_invariants(data, ticker):
    prices, macro = data
    rng = np.random.default_rng(len(ticker))
    rules = e.all_rules()
    for tf in ("D", "W"):
        p = e.IsbParams(tf)
        item = e.compute_labels(ticker, prices[ticker], e.build_market(prices["SPY"], macro, p), p)
        sub = [r for r in rules if r.tf == tf]
        for r in [sub[i] for i in rng.choice(len(sub), 12, replace=False)]:
            ev, trades = e.rule_events(item, r, p)
            kinds = {sg["s"]: sg["kind"] for sg in item["signals"]}
            assert all(kinds[t["s"]] in r.entries for t in trades)                     # 고른 라벨에서만 산다
            spans = [(t["e"], t["end"][0] if t["end"] else len(item["index"])) for t in trades]
            assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:]))                # 겹치는 거래 없음
            held = ev["pos"] == 1
            assert ev.loc[held, "rem"].between(-1e-6, 1.0 + 1e-6).all()
            if r.t1 == "off" and r.top != "trim":
                assert (ev.loc[held, "rem"] == 1.0).all()                             # ⅓ 매도를 안 하면 늘 전부
            if r.t1 == "off":
                assert ev["t1_fill"].isna().all()
