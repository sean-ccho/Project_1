#!/usr/bin/env python3
"""스퀴즈 모멘텀(SQZMOM_LB) 10년 백테스트 — TradingView 와 같은 계산, 일·주·월봉 (sqz_events_v1 · sqz_solo_v1).

계기: 2026-10-19 사용자 — "패턴들로 백테스팅 해보고싶어 PT1 을 이기는지 … 일단 처음으로 스퀴즈 모멘텀으로 … 주봉이나
월봉으로 보는게 좋겠지?" → 같은 규칙을 일·주·월봉 모두에 사전 등록해서 비교한다.

신호 = src/screener/squeeze_momentum.py (LazyBear SQZMOM_LB 기본값 20·2.0·20·1.5·TrueRange, Pine 그대로).
가격 = data/research/tv_ohlcv_max.parquet (ma_cross 와 같은 분할만 반영 가격, TradingView 종가와 S&P 500 586/586 일치).
**끝난 봉** 종가에서 신호 → 다음 거래일 시가. 수익 = 총수익(배당 포함) 시가. 유니버스 = 그날 S&P 500 구성 종목.
모멘텀 값이 나오기 전(봉 39개 + 직전 2봉) 사건은 없다.

사전 등록 (--count 로 건수만 보고 → --register, 수익률 계산 전):
  F1 sqz_events_v1 — 이벤트 스터디. 봉 {일, 주, 월} × 사건 {FIRE, MOM0, TURN} = 9건
    FIRE = 스퀴즈 풀림(검은 십자 → 회색) + 연두 막대, MOM0 = 모멘텀 0선 상향 돌파, TURN = 빨강 → 갈색
    주 보유기간 = 일 21 · 주 63 · 월 126 거래일 (나머지 21/63/126 은 참고). 초과수익 = 사건 종목 − 같은 구간 구성 종목
    동일가중, 비용 왕복 0.2%, 같은 종목·사건 10봉 쿨다운. 판정 = pattern_confirm_research.grade (ma_cross_v1 과 같음),
    family = 사건 ≥ 300 & 사건 날짜 ≥ 100 인 검정만 (Bonferroni z 는 family 크기로).
  F2 sqz_solo_v1 — 규칙대로 사고팔기. 같은 9개 진입 × 청산 FADE(모멘텀이 떨어진 첫 끝난 봉 — LazyBear "색이 바뀌면 나온다")
    포트폴리오 = 그날 규칙상 보유 중인 구성 종목 동일가중 (없으면 현금), 다음 시가 리밸런스, 비중 변화 × 편도 0.1%.
    판정 (paired_bootstrap 10,000회, 20일 블록, 3구간):
      'PT-1 보다 나음' = PT-1 기준선(pt1_base_dev_equity) 대비 ΔSharpe 하한 > 0 · p_개선아님 ≤ 0.05/9
      '채택 후보'      = 위 + 구성 종목 동일가중 매수보유 대비 verdict '채택 후보' · p ≤ 0.05/9
                         (PT-1 은 그냥 들고 있기보다 낮아서, 신호에 힘이 있으려면 들고 있기도 이겨야 한다)
      '관찰'           = 동일가중 대비 ΔSharpe 하한 > 0 (그 밖)
  확인 구간 2006-01-03 ~ 2015-12-31: F1 '후보' · F2 '채택 후보'만 다시 본다 (같은 부호 · t ≥ 1.96 / 95% 구간 0 안 넘음).
  홀드아웃(2025-10-01~)은 쓰지 않는다.
  참고 (판정 없음): SPY 대비 · 보유 기간 · 평균 보유 종목 수 · 투자 비율 · 사건 빈도 · 스퀴즈 비율.
  PT-1 엔진(3슬롯 계좌·PT-1 타이밍)은 scripts/sqz_vs_pt1.py (F3 sqz_pt1_v1).

추가 사전 등록 (--set lime, 2026-10-19 위 결과를 본 뒤 사용자 규칙 — "스퀴즈로 눌려있다가 연두색으로 변할 때가 좋은 거
아니야? KO 주봉 월봉으로 보면 그렇게 보이긴 하는데"):
  진입 LIME_SQZ = 직전 봉 검은 십자 + 이번 봉 막대가 연두로 바뀜 (직전 봉 연두 아님). 봉 {주, 월} (사용자가 말한 봉)
  F1 sqz_lime_events_v1 2건 (z = family 크기로) · F2 sqz_lime_solo_v1 2건 (p ≤ 0.05/2) — 나머지 정의·판정은 위와 같다.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/sqz_research.py --count [--set lime]
  PYTHONPATH=.:src .venv/bin/python scripts/sqz_research.py --register [--set lime]
  PYTHONPATH=.:src .venv/bin/python scripts/sqz_research.py --run [--set lime]
  PYTHONPATH=.:src .venv/bin/python scripts/sqz_research.py --confirm [--set lime]
산출: tier3/sqz_counts.csv · sqz_events.csv · sqz_solo.csv · sqz_describe.csv (lime 은 sqz_lime_*)
      (보고서 SQZ_REPORT.md 는 sqz_vs_pt1 --report)
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from idea_batch_research import member_mask  # noqa: E402
from ma_cross_research import (BAR_DAYS, HORIZONS, MIN_DAYS, MIN_EVENTS, PERIODS, PRIMARY_H, TF_NAME,  # noqa: E402
                               TV_OHLCV, _cagr, _cooldown, excess_matrix, in_period, membership, state_weights,
                               weighted_returns)
from paper_trading.evaluation import max_drawdown, paired_bootstrap, sharpe, verdict  # noqa: E402
from pattern_confirm_research import grade, stats  # noqa: E402
from research_utils import RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, bonferroni_z, load_ohlcv, log_trials, total_trials  # noqa: E402
from screener.squeeze_momentum import ENTRIES, squeeze_frame, system_position  # noqa: E402

STEP = "sqz"
TFS = ("D", "W", "M")
EVENTS = {"fire": "FIRE 스퀴즈 풀림 + 연두", "mom0": "MOM0 0선 상향 돌파", "turn": "TURN 빨강 → 갈색",
          "lime_sqz": "LIME_SQZ 검은 십자 중 연두로 바뀜"}
N_BOOT = 10_000
BASE_SAVED = RESEARCH_DIR / "pt1_base_dev_equity.parquet"
ETF_OHLCV = RESEARCH_DIR / "etf_ohlcv.parquet"


@dataclass(frozen=True)
class RuleSet:
    """사전 등록 한 묶음: 규칙 목록 · TRIAL_LOG family · 산출 파일 앞붙임."""

    rules: tuple[tuple[str, str], ...]
    family_ev: str
    family_solo: str
    prefix: str

    @property
    def p_solo(self) -> float:
        return 0.05 / len(self.rules)


RULE_SETS = {
    "main": RuleSet(tuple((tf, e) for tf in TFS for e in ENTRIES), "sqz_events_v1", "sqz_solo_v1", "sqz"),
    "lime": RuleSet((("W", "lime_sqz"), ("M", "lime_sqz")), "sqz_lime_events_v1", "sqz_lime_solo_v1", "sqz_lime"),
}


def rule_key(tf: str, entry: str) -> str:
    return f"{tf}-{entry.upper()}"


# ── 데이터 ─────────────────────────────────────────────────────────────────────────────


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(분할만 반영한 OHLC 패널 MultiIndex(Price, Ticker), 총수익 시가) — 개발 구간 끝까지."""
    px = load_ohlcv(TV_OHLCV)
    close = px.xs("Close", axis=1, level="Price").astype("float64")
    adj = px.xs("Adj Close", axis=1, level="Price").astype("float64")
    open_ = px.xs("Open", axis=1, level="Price").astype("float64")
    return px, open_ * (adj / close)


def _ticker(args: tuple[str, pd.DataFrame, str, tuple[tuple[str, str], ...]]
            ) -> tuple[list[dict], dict[str, pd.Series], list[dict]]:
    """한 종목: 사건 행 · 규칙별 봉 보유 상태(끝난 봉) · 규칙별 거래 (진입 봉 → FADE 봉)."""
    ticker, df, as_of, rules = args
    rows: list[dict] = []
    states: dict[str, pd.Series] = {}
    trades: list[dict] = []
    df = df.dropna(subset=["Close"])
    if len(df) < 60:
        return rows, states, trades
    for tf in dict.fromkeys(t for t, _ in rules):
        f = squeeze_frame(df, tf, pd.Timestamp(as_of))
        f = f[f["complete"]]
        if f.empty:
            continue
        pos_bar = f["n_bar"].to_numpy()
        for e in (e for t, e in rules if t == tf):
            idx = np.flatnonzero(f[e].to_numpy())
            idx = idx[_cooldown(pos_bar[idx])]
            for i in idx:
                rows.append({"rule": rule_key(tf, e), "tf": tf, "event": e, "티커": ticker, "date": f.index[i],
                             "sqz_bars": _sqz_run(f, i)})
            pos = system_position(f, e)
            states[rule_key(tf, e)] = pd.Series(pos, index=f.index).where(f["valid"].to_numpy() | pos)
            trades.extend(_trades(f, pos, rule_key(tf, e), ticker))
    return rows, states, trades


def _sqz_run(f: pd.DataFrame, i: int) -> int:
    """i 봉 직전까지 이어진 검은 십자 봉 수 (FIRE 설명용)."""
    on = f["sqz_on"].to_numpy()
    k = 0
    while i - 1 - k >= 0 and on[i - 1 - k]:
        k += 1
    return k


def _trades(f: pd.DataFrame, pos: np.ndarray, key: str, ticker: str) -> list[dict]:
    """보유 상태 → 거래 (신호 봉 날짜·청산 봉 날짜·보유 봉 수·봉 종가 수익). 끝까지 보유 중이면 청산 봉 없음."""
    out = []
    d = f.index
    c = f["close"].to_numpy()
    start = None
    for i, p in enumerate(pos):
        if p and start is None:
            start = i
        elif not p and start is not None:
            out.append({"rule": key, "티커": ticker, "entry_bar": d[start], "exit_bar": d[i], "bars": i - start,
                        "bar_ret": c[i] / c[start] - 1})
            start = None
    return out


def build(px: pd.DataFrame, workers: int, rules: tuple[tuple[str, str], ...]
          ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """모든 종목: (사건, 규칙별 상태 행렬, 거래)."""
    as_of = str(px.index.max().date())
    tickers = list(dict.fromkeys(px.columns.get_level_values("Ticker")))
    jobs = [(t, px.xs(t, axis=1, level="Ticker")[["Open", "High", "Low", "Close"]], as_of, rules) for t in tickers]
    rows: list[dict] = []
    trades: list[dict] = []
    per: dict[str, dict[str, pd.Series]] = {rule_key(tf, e): {} for tf, e in rules}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (r, st, tr) in enumerate(ex.map(_ticker, jobs, chunksize=8)):
            rows.extend(r)
            trades.extend(tr)
            for key, s in st.items():
                per[key][jobs[k][0]] = s
    return pd.DataFrame(rows), {k: pd.DataFrame(v) for k, v in per.items()}, pd.DataFrame(trades)


# ── F1 ───────────────────────────────────────────────────────────────────────────────


def test_id(tf: str, e: str, h: int) -> str:
    return f"{rule_key(tf, e)}|h{h}"


def count_table(ev: pd.DataFrame, rules: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    rows = []
    for tf, e in rules:
        sub = ev[ev["rule"] == rule_key(tf, e)]
        rows.append({"test_id": test_id(tf, e, PRIMARY_H[tf]), "rule": rule_key(tf, e), "n_events": len(sub),
                     "n_days": sub["date"].nunique(),
                     "family": len(sub) >= MIN_EVENTS and sub["date"].nunique() >= MIN_DAYS})
    return pd.DataFrame(rows)


def event_table(ev: pd.DataFrame, open_tr: pd.DataFrame, mem: pd.DataFrame, family: set[str], z: float,
                rules: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    ars = {h: excess_matrix(open_tr, mem, h).stack() for h in HORIZONS}
    rows = []
    for tf, e in rules:
        sub = ev[ev["rule"] == rule_key(tf, e)]
        key = pd.MultiIndex.from_arrays([sub["date"], sub["티커"]])
        for h in HORIZONS:
            s = ars[h].reindex(key)
            s.index = key
            lag = math.ceil(h / BAR_DAYS[tf])
            r = stats(s, lag)
            tid = test_id(tf, e, h)
            primary = h == PRIMARY_H[tf]
            rows.append({"test_id": tid, "rule": rule_key(tf, e), "tf": tf, "event": e, "h": h, "primary": primary,
                         "nw_lag": lag, **r, "판정": grade(r, z) if primary and tid in family else "참고"})
    return pd.DataFrame(rows)


# ── F2 ───────────────────────────────────────────────────────────────────────────────


def _bench(idx: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """(PT-1 기준선 일간 수익, SPY 시가→시가 수익) — idx 날짜로."""
    base = pd.read_parquet(BASE_SAVED).iloc[:, 0]
    base.index = pd.to_datetime(base.index)
    spy = load_ohlcv(ETF_OHLCV)["SPY"]["Open"]
    return base.pct_change().reindex(idx), (spy.shift(-1) / spy - 1).reindex(idx)


def _cmp(a: pd.Series, b: pd.Series, n_boot: int) -> tuple[dict, list[float], list[float]]:
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    st = paired_bootstrap(df["a"].tolist(), df["b"].tolist(), n_boot=n_boot)
    return st, df["a"].tolist(), df["b"].tolist()


def solo_table(states: dict[str, pd.DataFrame], open_tr: pd.DataFrame, mem: pd.DataFrame, period: str,
               trades: pd.DataFrame, rs: RuleSet) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    a, b = PERIODS[period]
    idx = open_tr.loc[a:b].index[:-1]
    r = (open_tr.shift(-1) / open_tr - 1).loc[idx]
    tradable = (open_tr.notna() & open_tr.shift(-1).notna()).loc[idx]
    m = mem.loc[idx]
    ew = weighted_returns(m.astype(float).div(m.sum(axis=1).where(lambda x: x > 0), axis=0).fillna(0.0), r)
    base_r, spy_r = _bench(idx)
    series: dict[str, pd.Series] = {"EW": ew, "PT1": base_r, "SPY": spy_r}
    rows = []
    for tf, e in rs.rules:
        key = rule_key(tf, e)
        w = state_weights(states[key], idx, open_tr.columns, m, tradable)
        ret = weighted_returns(w, r)
        series[key] = ret
        st_ew, ra, rb = _cmp(ret, ew, N_BOOT)
        v_ew = verdict(st_ew, max_drawdown(ra), max_drawdown(rb))
        st_pt, pa, pb = _cmp(ret, base_r, N_BOOT) if period == "dev" else ({}, [], [])
        st_spy, sa, sb = _cmp(ret, spy_r, 1_000)
        beats_pt1 = bool(st_pt) and st_pt["ΔSharpe_하한"] > 0 and st_pt["p_개선아님"] <= rs.p_solo
        beats_ew = v_ew == "채택 후보" and st_ew["p_개선아님"] <= rs.p_solo
        vd = ("채택 후보" if beats_pt1 and beats_ew else "PT-1 보다 나음" if beats_pt1 else
              "관찰" if st_ew and st_ew["ΔSharpe_하한"] > 0 else "탈락")
        n_hold = (w > 0).sum(axis=1)
        tr = trades[(trades["rule"] == key) & (trades["entry_bar"] >= a) & (trades["entry_bar"] <= b)] \
            if len(trades) else trades
        rows.append({"test_id": key, "tf": tf, "event": e,
                     "Sharpe": sharpe(ret.tolist()), "CAGR": _cagr(ret), "MDD": max_drawdown(ret.tolist()),
                     "Sharpe_EW": sharpe(ew.tolist()), "CAGR_EW": _cagr(ew), "MDD_EW": max_drawdown(ew.tolist()),
                     "ΔEW": st_ew.get("ΔSharpe"), "ΔEW_하한": st_ew.get("ΔSharpe_하한"),
                     "ΔEW_상한": st_ew.get("ΔSharpe_상한"), "p_EW": st_ew.get("p_개선아님"),
                     "구간_EW": f"{st_ew.get('구간_개선')}/{st_ew.get('구간_수')}", "verdict_EW": v_ew,
                     "Sharpe_PT1": sharpe(pb) if pb else np.nan, "ΔPT1": st_pt.get("ΔSharpe"),
                     "ΔPT1_하한": st_pt.get("ΔSharpe_하한"), "ΔPT1_상한": st_pt.get("ΔSharpe_상한"),
                     "p_PT1": st_pt.get("p_개선아님"),
                     "Sharpe_SPY": sharpe(sb), "ΔSPY": st_spy.get("ΔSharpe"), "ΔSPY_하한": st_spy.get("ΔSharpe_하한"),
                     "ΔSPY_상한": st_spy.get("ΔSharpe_상한"),
                     "투자일": float((n_hold > 0).mean()), "평균종목": float(n_hold[n_hold > 0].mean()) if (n_hold > 0).any() else 0.0,
                     "회전율_연": float(w.diff().abs().sum(axis=1).sum() / (len(w) / 252)),
                     "거래": len(tr), "보유봉_중앙": float(tr["bars"].median()) if len(tr) else np.nan,
                     "봉수익_평균": float(tr["bar_ret"].mean()) if len(tr) else np.nan,
                     "승률": float((tr["bar_ret"] > 0).mean()) if len(tr) else np.nan,
                     "판정": vd})
    return pd.DataFrame(rows), series


def describe(ev: pd.DataFrame, mem: pd.DataFrame, period: str, rules: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    a, b = PERIODS[period]
    ticker_years = float(mem.loc[a:b].sum().sum()) / 252
    rows = []
    for tf, e in rules:
        sub = ev[ev["rule"] == rule_key(tf, e)]
        rows.append({"rule": rule_key(tf, e), "tf": tf, "event": e, "per_100_ticker_years": len(sub) / ticker_years * 100,
                     "sqz_bars_median": float(sub["sqz_bars"].median()) if e == "fire" and len(sub) else np.nan})
    return pd.DataFrame(rows)


# ── 등록·실행 ─────────────────────────────────────────────────────────────────────────


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame()


def _registered(family: str) -> pd.Series | None:
    log = _log()
    if log.empty:
        return None
    hit = log[(log["family"] == family) & (log["판정"] == "등록")]
    return hit.iloc[-1] if len(hit) else None


def _prepare(workers: int, period: str, rs: RuleSet
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    px, open_tr = load_prices()
    close = px.xs("Close", axis=1, level="Price")
    mem = member_mask(close.index, close.columns, membership())
    ev, states, trades = build(px, workers, rs.rules)
    return open_tr, mem, in_period(ev, mem, period), states, trades


def do_count(workers: int, rs: RuleSet) -> None:
    _, _, ev, _, _ = _prepare(workers, "dev", rs)
    ct = count_table(ev, rs.rules)
    ct.to_csv(TIER3_DIR / f"{rs.prefix}_counts.csv", index=False, encoding="utf-8-sig")
    print(ct.to_string(index=False))
    n = int(ct["family"].sum())
    print(f"\nF1 family: {n} / {len(ct)} · Bonferroni z = {bonferroni_z(n):.2f} · "
          f"F2 {len(rs.rules)}건 p ≤ {rs.p_solo:.4f}")


def do_register(rs: RuleSet) -> None:
    if _registered(rs.family_ev) is not None:
        raise SystemExit(f"{rs.family_ev} 은 이미 등록됐다")
    ct = pd.read_csv(TIER3_DIR / f"{rs.prefix}_counts.csv", encoding="utf-8-sig")
    fam = ct[ct["family"]]
    n = len(rs.rules)
    entries = ",".join(dict.fromkeys(e.upper() for _, e in rs.rules))
    tfs = ",".join(dict.fromkeys(tf for tf, _ in rs.rules))
    log_trials(STEP, rs.family_ev, len(fam), [{
        "test_id": "(등록)", "판정": "등록",
        "메모": f"F1 {len(fam)}건 z={bonferroni_z(len(fam)):.2f}: " + ",".join(fam["test_id"])}])
    log_trials(STEP, rs.family_solo, n, [{
        "test_id": "(등록)", "판정": "등록",
        "메모": f"F2 {n}건 진입 {{{entries}}}×{{{tfs}}} + FADE 청산, 보유 종목 동일가중; "
                f"PT-1 보다 나음 = vs pt1_base ΔSharpe 하한>0·p≤0.05/{n}; 채택 후보 = + vs 동일가중 verdict 채택 후보·"
                f"p≤0.05/{n}; paired_bootstrap {N_BOOT}"}])
    print(f"등록: {rs.family_ev} {len(fam)}건 · {rs.family_solo} {n}건 · 누적 시도 {total_trials()}")


def do_run(workers: int, rs: RuleSet, period: str = "dev") -> None:
    reg = _registered(rs.family_ev)
    if reg is None or _registered(rs.family_solo) is None:
        raise SystemExit("등록 먼저 (--count → --register)")
    family = set(str(reg["메모"]).split(": ", 1)[1].split(","))
    z = bonferroni_z(len(family))
    open_tr, mem, ev, states, trades = _prepare(workers, period, rs)
    a, b = PERIODS[period]
    ot, mm = (open_tr.loc[:b], mem.loc[:b]) if period == "dev" else (open_tr, mem)
    f1 = event_table(ev, ot, mm, family, z, rs.rules)
    f2, series = solo_table(states, open_tr, mem, period, trades, rs)
    desc = describe(ev, mem, period, rs.rules)
    tag = "" if period == "dev" else f"_{period}"
    f1.to_csv(TIER3_DIR / f"{rs.prefix}_events{tag}.csv", index=False, encoding="utf-8-sig")
    f2.to_csv(TIER3_DIR / f"{rs.prefix}_solo{tag}.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(TIER3_DIR / f"{rs.prefix}_describe{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(series).to_parquet(RESEARCH_DIR / f"{rs.prefix}_solo_series{tag}.parquet")
    if period == "dev":
        judged = f1[f1["판정"] != "참고"]
        log_trials(STEP, rs.family_ev, len(family),
                   [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}", "판정": r["판정"]} for _, r in judged.iterrows()])
        log_trials(STEP, rs.family_solo, len(rs.rules),
                   [{"test_id": r.test_id, "t": f"{r['ΔEW']:+.3f}", "판정": r["판정"],
                     "메모": f"Sharpe {r.Sharpe:.2f} vs EW {r.Sharpe_EW:.2f} · PT-1 {r.Sharpe_PT1:.2f} "
                             f"(ΔPT1 {r.ΔPT1:+.2f} [{r.ΔPT1_하한:+.2f}, {r.ΔPT1_상한:+.2f}])"} for _, r in f2.iterrows()])
    pd.set_option("display.width", 250)
    print(f1[f1["primary"]][["test_id", "n_events", "n_days", "mean_ar", "mean_net", "t_nw", "판정"]].to_string(index=False))
    print(f2[["test_id", "Sharpe", "CAGR", "MDD", "Sharpe_EW", "ΔEW", "ΔEW_하한", "p_EW", "Sharpe_PT1", "ΔPT1",
              "ΔPT1_하한", "투자일", "평균종목", "보유봉_중앙", "판정"]].to_string(index=False))
    print(desc.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--confirm", action="store_true")
    ap.add_argument("--set", choices=sorted(RULE_SETS), default="main", help="사전 등록 묶음")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    rs = RULE_SETS[args.set]
    if args.count:
        do_count(args.workers, rs)
    elif args.register:
        do_register(rs)
    elif args.run:
        do_run(args.workers, rs, "dev")
    else:
        do_run(args.workers, rs, "confirm")


if __name__ == "__main__":
    main()
