#!/usr/bin/env python3
"""PT-1 + A 시장 타이밍 (pt1_a_timing_v1): 종목은 PT-1 이 고르고, 언제·얼마나 들고 있을지는 lab_v1 A군 규칙이 정한다.

계기: 사용자 — "PT-1 + A 시장 타이밍" (2026-10-14). 목표는 미국 개별 종목 매매로 좋은 성적.

사전 등록 (TRIAL_LOG family pt1_a_timing_v1, 2건 — 결과 계산 전에 --register):
  A 규칙 (lab_v1 A군이 1950~2007 자료로 고른 값 그대로, 다시 맞추지 않음):
    월 마지막 거래일 종가 기준 — SPY 총수익지수 ≤ 250일 평균이면 w = 0, 아니면 w = min(1, 12% ÷ 최근 21일 변동성).
    결정일 당일부터 다음 결정일 전까지 같은 w (그날 종가로 판단 → 다음 시가 체결, PT-1 엔진과 같은 규칙).
  A추세: w = 0 이면 PT-1 보유를 다음 시가에 전부 팔고(A시장정리) 그 달 새 매수 없음. w > 0 이면 PT-1 그대로.
  A전체: A추세 + 목표 주식 비중 w — 결정일에 보유 평가액이 자산의 w 보다 5%p 넘게 크면 다음 시가에 모든 종목을
         같은 비율로 일부 판다. 새 매수 금액 = min(원래 금액, (w × 자산 − 보유 평가액) ÷ 빈 자리),
         그 여유가 자산의 1% 미만이면 그날 새 매수 없음 (0주 매수 방지).
  같은 날 우선순위: PT-1 매도 > A시장정리 > A 비중 축소.
  기간·엔진·설정 = PT-1 기준선 (validate_upside_walkforward.RUN_KW, 2016-01-04 ~ 2025-09-30, $5,000, 3슬롯).
  src/ 는 고치지 않고 paper_trading.backtest 의 이름만 실행 중에 바꿔 끼운다 (isb_vs_pt1 과 같은 방식).
  현금 이자: 엔진은 현금 수익이 0 이라, 판정은 매일 (전날 현금 비중 × Kenneth French 무위험 수익)을 더한 수익으로
             한다 (BASE 도 같은 방식). Sharpe 는 무위험 초과 기준 (lab_v1 과 같다).
판정: 채택 후보 = BASE 대비 verdict '채택 후보' (ΔSharpe 95% 하한 > 0 · 3구간 중 2+ · MDD 2%p 이내)
      · p_개선아님 ≤ 0.05/2 (paired_bootstrap 10,000회). 참고 = SPY · A 단독 · 이자 없는 엔진 원래 숫자.
주의: A 를 PT-1 에 얹기로 한 건 A 의 2016~2025 결과를 본 뒤다 (숫자는 2008년 전 자료로 정해져 있다).
확인: BASE = 저장된 pt1_base_dev_equity, 통과 모드(래퍼만 끼우고 결정은 원래대로) = BASE (자본곡선·거래 목록).
      둘 중 하나라도 다르면 결과를 쓰지 않고 멈춘다.

사용:
  PYTHONPATH=.:src .venv/bin/python scripts/pt1_market_timing.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/pt1_market_timing.py --run
산출: tier3/PT1_A_TIMING.md · tier3/pt1_a_timing.csv · data/research/pt1_a_timing_runs.pkl (git 제외)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_vs_pt1 as iv  # noqa: E402
import lab_data as ld  # noqa: E402
import paper_trading.backtest as bt  # noqa: E402
from lab_backtest import decision_mask, perf  # noqa: E402
from lab_families import FAMILIES, Universe  # noqa: E402
from paper_trading.evaluation import paired_bootstrap, verdict  # noqa: E402
from research_utils import RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, log_trials, total_trials  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "pt1_a_timing_v1"
STEP = "stage17"
TESTS = ("A추세", "A전체")
MODES = {"PROBE": "probe", "A추세": "trend", "A전체": "full"}
P_BONF = 0.05 / len(TESTS)
N_BOOT = 10_000
A_CFG = {"trend": "sma", "n": 250, "sigma": 0.12, "vl": 21, "sched": "M"}   # lab_v1 A군 선택 (1950~2007)
TRIM_BAND = 0.05
MIN_ROOM = 0.01
EXIT_REASON = "A시장정리"
TRIM_TAG = "A비중축소·"
RUNS_PATH = RESEARCH_DIR / "pt1_a_timing_runs.pkl"
OUT_CSV = TIER3_DIR / "pt1_a_timing.csv"
REPORT = TIER3_DIR / "PT1_A_TIMING.md"


def spec() -> str:
    return ("[2016-01-04~2025-09-30, PT-1 백테스트 엔진·13y PIT S&P 500·$5,000·3슬롯] lab_v1 A군 규칙(1950~2007 로 고른 값 "
            "그대로: 월말 SPY 총수익지수 ≤ 250일 평균이면 w=0, 아니면 w=min(1, 12%÷21일 변동성), 결정일부터 그 달 내내)을 "
            "PT-1 에 얹는다. A추세 = w=0 이면 다음 시가에 보유 전부 매도·그 달 매수 없음. A전체 = A추세 + 결정일 보유 비중이 "
            "w 보다 5%p 넘게 크면 같은 비율로 일부 매도, 새 매수 금액 = min(원래, (w×자산−보유)÷빈 자리), 여유 < 자산 1% 면 "
            "매수 없음. 우선순위 PT-1 매도 > A시장정리 > 비중 축소. 판정 수익 = 엔진 수익 + 전날 현금 비중 × 무위험(BASE 도), "
            "Sharpe 는 무위험 초과. 채택 = BASE 대비 verdict 채택 후보(ΔSharpe 95% 하한>0 · 3구간 중 2+ · MDD 2%p 이내) · "
            "p ≤ 0.05/2 (부트스트랩 10,000회). 참고 = SPY·A 단독·이자 없는 원래 숫자. 주의: A 를 고른 건 2016~2025 결과를 본 뒤.")


# ── A 비중 ───────────────────────────────────────────────────────────────────────────────────

def a_weights() -> tuple[dict[str, float], set[str]]:
    """날짜 → 그날 적용할 w (가장 최근 월말 결정), 결정일 집합. 자료 = lab 고정 시세 (SPY 수정 종가, 1993~)."""
    prices, french = ld.load_prices(), ld.load_french()
    cal = ld.us_calendar(french)
    r = ld.close_returns(prices, "SPY", cal).dropna()
    u = Universe(r.to_frame("EQ"), french["rf"])
    tgt = FAMILIES["A"].target(u, A_CFG)["EQ"]
    dec = decision_mask(tgt.index, "M") & tgt.notna().to_numpy()
    w_dec = tgt[dec]
    daily = w_dec.reindex(tgt.index[tgt.index >= w_dec.index.min()]).ffill()
    return {f"{d:%Y-%m-%d}": float(v) for d, v in daily.items()}, {f"{d:%Y-%m-%d}" for d in w_dec.index}


# ── 순수 판단 함수 (테스트 대상) ─────────────────────────────────────────────────────────────────

def timing_sell(pt1: tuple[bool, str], mode: str, w: float, decision_day: bool,
                invested_frac: float) -> tuple[bool, str]:
    """매도 판단: PT-1 매도 > A시장정리(w = 0) > A 비중 축소(A전체·결정일·보유 비중이 w 보다 5%p 넘게 클 때)."""
    if pt1[0] or mode == "probe":
        return pt1
    if w <= 0:
        return True, EXIT_REASON
    if mode == "full" and decision_day and invested_frac > 0 and invested_frac - w > TRIM_BAND:
        return True, f"{TRIM_TAG}{1.0 - w / invested_frac:.8f}"
    return pt1


def buy_room(cash: float, equity: float, w: float) -> float:
    """A전체 새 매수 여유 = w × 자산 − 보유 평가액 (0 이상, 현금 한도)."""
    return max(0.0, min(cash, w * equity - (equity - cash)))


def capped_allocation(normal: float, cash: float, equity: float, w: float, empty_slots: int, mode: str) -> float:
    """A전체: 새 매수 금액 = min(원래 금액, 여유 ÷ 빈 자리). 그 밖에는 원래 금액."""
    if mode != "full" or w >= 1.0:
        return normal
    return min(normal, buy_room(cash, equity, w) / max(1, empty_slots))


def block_buys(mode: str, w: float, cash: float, equity: float) -> bool:
    """새 매수를 막는 날: A추세·A전체에서 w = 0, 또는 A전체에서 여유가 자산의 1% 미만."""
    if mode == "probe":
        return False
    if w <= 0:
        return True
    return mode == "full" and w < 1.0 and equity > 0 and buy_room(cash, equity, w) < MIN_ROOM * equity


def with_cash_interest(equity: pd.Series, cash: pd.Series, rf: pd.Series) -> pd.Series:
    """엔진 일간 수익 + 전날 현금 비중 × 그날 무위험 수익 (현금이 단기 국채 이자를 받는다고 보고)."""
    r = equity.pct_change().dropna()
    frac = (cash.reindex(equity.index) / equity).shift(1).reindex(r.index).fillna(0.0)
    return r + frac * rf.reindex(r.index).fillna(0.0)


# ── 엔진 실행 ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class Ctx:
    mode: str
    w_map: dict[str, float]
    decision_days: set[str]
    date: pd.Timestamp | None = None
    positions: list | None = None
    closes: pd.DataFrame | None = None
    cash_live: float = float(RUN_KW["initial_capital"])   # 매도 대금·매수 금액을 따라가는 지금 현금 (기록 때 엔진 값과 대조)
    cash_by_day: dict[str, float] = field(default_factory=dict)
    trims: list[dict[str, Any]] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    def w(self, day: str) -> float:
        if day not in self.w_map:
            self.counts["w 없음"] += 1
            return 1.0
        return self.w_map[day]

    def day(self) -> str:
        return f"{self.date:%Y-%m-%d}"

    def book_value(self, day: str) -> tuple[float, float]:
        """(지금 현금, 자산) — 보유 종목은 day 종가로 평가."""
        mv = 0.0
        if self.positions and self.closes is not None:
            ts = pd.Timestamp(day)
            for p in self.positions:
                px = bt._close_on_or_before(self.closes, ts, p.ticker)
                if not np.isnan(px):
                    mv += p.shares * px
        return self.cash_live, self.cash_live + mv


@dataclass
class TimingRun:
    key: str
    equity: pd.Series
    trades: list[dict[str, Any]]
    summary: dict[str, Any]
    cash: pd.Series
    trims: list[dict[str, Any]]
    counts: dict[str, int]


@contextlib.contextmanager
def _patched(ctx: Ctx) -> Iterator[None]:
    """isb_vs_pt1 과 같은 엄격 가격 읽기·고정 종목 목록 + A 시장 타이밍 래퍼."""
    orig = iv._ORIG
    pinned = iv._pinned_tickers()
    bt.fetch_ohlcv = iv._strict_fetch
    bt._pit_universe_tickers = lambda membership, period: list(pinned)

    def snapshot(*args: Any, **kw: Any) -> pd.DataFrame:
        ctx.date = pd.Timestamp(args[1])
        cache = iv._SNAP
        if cache is None:
            return orig["_compute_ranked_snapshot"](*args, **kw)
        hit = cache.get(ctx.date)
        if hit is None:
            hit = orig["_compute_ranked_snapshot"](*args, **kw)
            cache[ctx.date] = hit.copy()
            return hit
        return hit.copy()

    def record(equity_points: list, positions: list, closes: pd.DataFrame, date_ts: pd.Timestamp, today_str: str,
               cash: float = 0.0, use_capital: bool = False) -> None:
        if abs(ctx.cash_live - cash) > 1e-6 * max(1.0, abs(cash)):
            ctx.counts["현금 불일치"] += 1
        ctx.positions, ctx.closes, ctx.cash_live = positions, closes, cash
        ctx.cash_by_day[today_str] = cash
        ctx.counts["기록"] += 1
        return orig["_record_equity"](equity_points, positions, closes, date_ts, today_str, cash=cash,
                                      use_capital=use_capital)

    def alloc(cash: float, equity: float, max_positions: int, empty_slots: int, ranked_df: pd.DataFrame | None,
              ticker: str) -> float:
        normal = orig["slot_allocation"](cash, equity, max_positions, empty_slots, ranked_df, ticker)
        ctx.counts["매수금액"] += 1
        a = capped_allocation(normal, cash, equity, ctx.w(ctx.day()), empty_slots, ctx.mode)
        if a < normal - 1e-9:
            ctx.counts["매수금액 줄임"] += 1
        ctx.cash_live -= a
        return a

    def select(df: pd.DataFrame, holdings: list[dict[str, Any]], k: int = 1):
        picks, debug = orig["select_top_candidates"](df, holdings, k=k)
        ctx.counts["후보"] += 1
        day = ctx.day()
        cash, equity = ctx.book_value(day)
        if picks and block_buys(ctx.mode, ctx.w(day), cash, equity):
            ctx.counts["매수 막음"] += 1
            return [], debug
        return picks, debug

    def sell(pos_dict: dict[str, Any], cur: float, today_str: str) -> tuple[bool, str]:
        pt1 = orig["check_sell_conditions"](pos_dict, cur, today_str)
        ctx.counts["매도판단"] += 1
        dec = today_str in ctx.decision_days
        inv = 0.0
        if ctx.mode == "full" and dec:
            cash, equity = ctx.book_value(today_str)
            inv = (equity - cash) / equity if equity > 0 else 0.0
        out = timing_sell(pt1, ctx.mode, ctx.w(today_str), dec, inv)
        if out[0] and not pt1[0]:
            ctx.counts["A시장정리" if out[1] == EXIT_REASON else "A비중축소"] += 1
        return out

    def close(positions: list, trades: list, ticker: str, exit_price: float, exit_date: str, reason: str):
        if not reason.startswith(TRIM_TAG):
            out = orig["_close_position"](positions, trades, ticker, exit_price, exit_date, reason)
            ctx.cash_live += out[1]
            return out
        frac = float(reason[len(TRIM_TAG):])
        pos = next((p for p in positions if p.ticker == ticker), None)
        if pos is None:
            return None, 0.0
        sold = pos.shares * frac
        pos.shares -= sold
        proceeds = sold * exit_price * (1 - bt.BACKTEST_COST_PER_SIDE)
        ctx.trims.append({"ticker": ticker, "entry_date": pos.entry_date, "date": exit_date, "price": exit_price,
                          "shares": sold, "frac": frac})
        ctx.cash_live += proceeds
        return None, proceeds

    bt._compute_ranked_snapshot = snapshot
    bt._record_equity = record
    bt.slot_allocation = alloc
    bt.select_top_candidates = select
    bt.check_sell_conditions = sell
    bt._close_position = close
    try:
        yield
    finally:
        for name, fn in orig.items():
            setattr(bt, name, fn)


def run_timing(key: str, w_map: dict[str, float], decision_days: set[str]) -> TimingRun:
    ctx = Ctx(MODES[key], w_map, decision_days)
    sink = io.StringIO()
    with _patched(ctx), contextlib.redirect_stdout(sink):
        result = bt.run_paper_trading_backtest(**RUN_KW)
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    cash = pd.Series(ctx.cash_by_day)
    cash.index = pd.to_datetime(cash.index)
    return TimingRun(key, eq, result["trades"], result["summary"], cash.sort_index(), ctx.trims, dict(ctx.counts))


# ── 판정 ─────────────────────────────────────────────────────────────────────────────────────

def judge(st: dict[str, float], v: str) -> str:
    if v == "채택 후보" and st and st["p_개선아님"] <= P_BONF:
        return "채택 후보"
    if st and st["ΔSharpe_하한"] > 0:
        return "관찰"
    return "탈락"


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")


def register() -> None:
    log = _log()
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"[timing] {FAMILY} 이미 등록됨", flush=True)
        return
    log_trials(STEP, FAMILY, len(TESTS), [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"[timing] {FAMILY} 등록 · 누적 시도 {total_trials():,}건", flush=True)


def check_lab_config() -> None:
    """A_CFG 가 lab_v1 A군의 선택과 같은지 (저장된 결과가 있으면)."""
    import lab_search as ls
    if ls.STATE.exists():
        st = ls.load_state()
        chosen = st.searches["A"].grid[st.searches["A"].chosen]
        if chosen != A_CFG:
            raise SystemExit(f"A_CFG 가 lab_v1 선택과 다르다: {chosen}")


def run() -> None:
    log = _log()
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register (결과 계산 전 사전 등록)")
    already = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    check_lab_config()
    w_map, dec_days = a_weights()
    start, end = RUN_KW["start_date"], RUN_KW["end_date"]
    in_win = {d: w for d, w in w_map.items() if start <= d <= end}
    print(f"[timing] A 비중: 기간 {len(in_win)}일 · w=0 인 날 {sum(1 for w in in_win.values() if w <= 0)} · "
          f"평균 {np.mean(list(in_win.values())):.2f}", flush=True)
    iv.ensure_pinned()

    print("===== BASE (래퍼 없음)", flush=True)
    base = iv.run_overlay(iv.Overlay("BASE"))
    err = iv.check_base(base)
    if err:
        raise SystemExit(err)
    iv.enable_snapshot_cache()
    runs: dict[str, TimingRun] = {}
    for k, key in enumerate(("PROBE",) + TESTS, 1):
        print(f"===== {key} [{k / (len(TESTS) + 1):.0%}]", flush=True)
        runs[key] = run_timing(key, w_map, dec_days)
        bad = {k: runs[key].counts.get(k, 0) for k in ("현금 불일치", "w 없음") if runs[key].counts.get(k, 0)}
        if bad:
            raise SystemExit(f"{key}: 현금 추적·A 비중 날짜가 어긋남 {bad}")
        if key == "PROBE":
            p = runs["PROBE"]
            same_eq = p.equity.index.equals(base.equity.index) and float((p.equity / base.equity - 1).abs().max()) <= 1e-12
            same_tr = [iv._trade_key(t) for t in p.trades] == [iv._trade_key(t) for t in base.trades]
            called = p.counts.get("매도판단", 0) > 0 and p.counts.get("후보", 0) > 0 and p.counts.get("기록", 0) > 0
            tracked = p.counts.get("현금 불일치", 0) == 0 and p.counts.get("w 없음", 0) == 0
            if not (same_eq and same_tr and called and tracked):
                raise SystemExit(f"통과 모드가 BASE 와 다르다 (자본 {same_eq}, 거래 {same_tr}, 래퍼 호출 {called}, "
                                 f"현금 추적·w {tracked}: {dict(p.counts)})")
            print(f"[timing] 통과 모드 = BASE (거래 {len(base.trades)}건 · 매도판단 {p.counts['매도판단']} · "
                  f"후보 {p.counts['후보']})", flush=True)
    with RUNS_PATH.open("wb") as fh:
        pickle.dump({"base": (base.equity, base.trades), "runs": runs, "w_map": w_map}, fh)
    finish(base, runs, w_map, already)


def finish(base: Any, runs: dict[str, TimingRun], w_map: dict[str, float], already: bool) -> None:
    french = ld.load_french()
    rf = french["rf"]
    probe = runs["PROBE"]
    series = {"BASE": with_cash_interest(base.equity, probe.cash, rf)}
    raw = {"BASE": base.equity.pct_change().dropna()}
    for key in TESTS:
        series[key] = with_cash_interest(runs[key].equity, runs[key].cash, rf)
        raw[key] = runs[key].equity.pct_change().dropna()
    prices = ld.load_prices()
    spy = ld.close_returns(prices, "SPY", ld.us_calendar(french)).dropna()
    import lab_search as ls
    a_alone = ls.load_state().confirm["A"].r if ls.STATE.exists() else None
    idx = series["BASE"].index
    ex = lambda s: (s.reindex(idx) - rf.reindex(idx).fillna(0.0))  # noqa: E731
    rows, logrows = [], []
    pb = perf(series["BASE"], rf.reindex(idx).fillna(0.0))
    for key in ("BASE",) + TESTS:
        s = series[key]
        p = perf(s, rf.reindex(s.index).fillna(0.0))
        row = {"test_id": key, "Sharpe": p["Sharpe"], "CAGR": p["CAGR"], "MDD": p["MDD"], "변동성": p["vol"],
               "Sharpe_이자없음_원래": perf(raw[key])["Sharpe_raw"], "CAGR_이자없음": perf(raw[key])["CAGR"]}
        if key != "BASE":
            st = paired_bootstrap(ex(s).tolist(), ex(series["BASE"]).tolist(), n_boot=N_BOOT)
            v = verdict(st, p["MDD"], pb["MDD"])
            r = runs[key]
            reasons = Counter(exit_kind(t["exit_reason"]) for t in r.trades)
            invested = 1.0 - (r.cash.reindex(r.equity.index) / r.equity)
            row.update({"ΔSharpe": st["ΔSharpe"], "하한": st["ΔSharpe_하한"], "상한": st["ΔSharpe_상한"],
                        "p": st["p_개선아님"], "구간": f"{st['구간_개선']}/{st['구간_수']}", "verdict": v,
                        "판정": judge(st, v), "거래": len(r.trades), "A시장정리": reasons.get(EXIT_REASON, 0),
                        "일부매도": len(r.trims), "매수막음": r.counts.get("매수 막음", 0),
                        "매수금액줄임": r.counts.get("매수금액 줄임", 0), "평균주식비중": float(invested.mean()),
                        "청산사유": " · ".join(f"{k} {n}" for k, n in reasons.most_common())})
            logrows.append({"test_id": key, "판정": row["판정"],
                            "메모": (f"Sharpe {p['Sharpe']:.2f} vs BASE {pb['Sharpe']:.2f} (무위험 초과, 현금 이자 포함), "
                                     f"ΔSharpe {st['ΔSharpe']:+.2f} [{st['ΔSharpe_하한']:+.2f}, {st['ΔSharpe_상한']:+.2f}] "
                                     f"p {st['p_개선아님']:.4f}, 구간 {st['구간_개선']}/{st['구간_수']}, CAGR {p['CAGR']:+.1%} "
                                     f"vs {pb['CAGR']:+.1%}, MDD {p['MDD']:.1%} vs {pb['MDD']:.1%}, 거래 {len(r.trades)}")})
        else:
            row.update({"거래": len(base.trades), "평균주식비중": float((1.0 - probe.cash.reindex(probe.equity.index)
                                                                    / probe.equity).mean())})
        rows.append(row)
    refs = {"SPY": spy.reindex(idx).dropna()}
    if a_alone is not None:
        refs["A 단독"] = a_alone.reindex(idx).dropna()
    for name, s in refs.items():
        p = perf(s, rf.reindex(s.index).fillna(0.0))
        rows.append({"test_id": name, "Sharpe": p["Sharpe"], "CAGR": p["CAGR"], "MDD": p["MDD"], "변동성": p["vol"],
                     "Sharpe_이자없음_원래": perf(s)["Sharpe_raw"], "판정": "참고"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    if already:
        print("결과 행이 이미 TRIAL_LOG 에 있다 — 다시 쓰지 않는다", flush=True)
    else:
        log_trials(STEP, FAMILY, len(TESTS), logrows)
    write_report(table, series, refs, runs, w_map)
    print(table[["test_id", "Sharpe", "CAGR", "MDD", "판정"]].to_string(index=False), flush=True)


def exit_kind(reason: Any) -> str:
    """청산 사유 종류만: '트레일링(-5.1%)' → '트레일링', '교체→AAPL' → '교체'."""
    return str(reason).split("→")[0].split("(")[0].strip()


def report_only() -> None:
    """저장한 실행 결과로 판정표·보고서만 다시 (엔진 안 돌림, TRIAL_LOG 안 씀)."""
    class _U(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            return super().find_class("pt1_market_timing" if module == "__main__" else module, name)

    with RUNS_PATH.open("rb") as fh:
        saved = _U(fh).load()
    eq, trades = saved["base"]
    base = type("Base", (), {"equity": eq, "trades": trades})()
    finish(base, saved["runs"], saved["w_map"], already=True)


def _f(v: Any, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return format(v, fmt)


def write_report(table: pd.DataFrame, series: dict[str, pd.Series], refs: dict[str, pd.Series],
                 runs: dict[str, TimingRun], w_map: dict[str, float]) -> None:
    t = table.set_index("test_id")
    best = [k for k in TESTS if t.loc[k, "판정"] == "채택 후보"]
    if best:
        head = f"**결론: {', '.join(best)} 이(가) 사전 기준을 통과했다 — PT-1 에 A 시장 타이밍을 얹는 것이 BASE 보다 낫다.**"
    else:
        head = ("**결론: PT-1 에 A 시장 타이밍을 얹어도 사전 기준(BASE 대비 확실한 개선)은 통과하지 못했다.** "
                "숫자는 아래 표대로 — 낙폭·수익의 변화는 참고로 본다.")
    L = [f"# PT-1 + A 시장 타이밍 ({FAMILY}) — 2016-01-04 ~ 2025-09-30", "", head, "",
         f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · 2건 · p ≤ 0.05/2 · 부트스트랩 {N_BOOT:,}회 · 누적 시도 {total_trials():,}건",
         "- 계기: 사용자 — \"PT-1 + A 시장 타이밍\" (2026-10-14). 종목은 PT-1 이 고르고, 언제·얼마나 들고 있을지는 A 가 정한다.",
         "- A 규칙 (lab_v1 A군이 1950~2007 자료로 고른 값 그대로): 월말에 SPY(배당 포함)가 250일 평균 아래면 주식 0%, "
         "위면 주식 비중 = min(100%, 12% ÷ 최근 21일 변동성). 그 달 내내 유지.",
         "- A추세 = 0% 인 달에는 PT-1 보유를 다음 시가에 전부 팔고 새로 안 산다 (나머지 달은 PT-1 그대로). "
         "A전체 = 여기에 주식 비중 한도까지 (결정일에 넘치면 같은 비율로 일부 매도, 새 매수 금액도 한도 안).",
         "- 수익 = 엔진 수익 + 현금 이자(전날 현금 비중 × 1개월 국채). Sharpe = 무위험 초과. 이자 없는 엔진 원래 숫자도 함께.",
         "- 확인: BASE 재현 · 래퍼만 끼운 통과 모드 = BASE (자본곡선·거래 목록) 통과.", "",
         "## 판정", "",
         "| 경우 | Sharpe | 연수익 | 최대 낙폭 | 변동성 | 평균 주식 비중 | 거래 | BASE 대비 ΔSharpe [95%] | p | 구간 | 판정 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("BASE",) + TESTS:
        r = t.loc[k]
        ci = (f"{r['ΔSharpe']:+.2f} [{r['하한']:+.2f}, {r['상한']:+.2f}]" if k != "BASE" else "")
        L.append(f"| {'PT-1 그대로' if k == 'BASE' else 'PT-1 + ' + k} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | "
                 f"{r['변동성']:.1%} | {_f(r.get('평균주식비중'), '.0%')} | {int(r['거래'])} | {ci} | "
                 f"{_f(r.get('p'), '.3f') if k != 'BASE' else ''} | {r.get('구간', '') if k != 'BASE' else ''} | "
                 f"{'기준' if k == 'BASE' else '**' + str(r['판정']) + '**'} |")
    for name in refs:
        r = t.loc[name]
        L.append(f"| {name} (참고) | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | {r['변동성']:.1%} | | | | | | 참고 |")
    L += ["", "- 이자 없는 엔진 원래 Sharpe (지금까지 PT-1 숫자와 같은 방식): "
          + " · ".join(f"{k} {t.loc[k, 'Sharpe_이자없음_원래']:.2f}" for k in ("BASE",) + TESTS)
          + f" (SPY {t.loc['SPY', 'Sharpe_이자없음_원래']:.2f}, 종가 기준)", ""]
    L += ["## A 가 한 일", ""]
    for k in TESTS:
        r = t.loc[k]
        L.append(f"- **{k}**: A시장정리 {int(r['A시장정리'])}건 · 일부 매도 {int(r['일부매도'])}건 · 새 매수 막은 날 "
                 f"{int(r['매수막음'])} · 매수 금액 줄인 횟수 {int(r['매수금액줄임'])} · 청산 사유: {r['청산사유']}")
    w = pd.Series(w_map)
    w.index = pd.to_datetime(w.index)
    w = w.loc[RUN_KW["start_date"]:RUN_KW["end_date"]]
    off = w[w <= 0]
    L += [f"- A 가 주식 0% 로 정한 날: {len(off)}거래일 ({len(off) / len(w):.0%}) · 평균 목표 비중 {w.mean():.0%}", ""]
    L += ["## 연도별 수익 (현금 이자 포함)", ""]
    cols = {("PT-1 그대로" if k == "BASE" else f"PT-1 + {k}"): s for k, s in series.items()}
    cols.update({k: v for k, v in refs.items()})
    yr = pd.DataFrame({k: (1 + s).groupby(s.index.year).prod() - 1 for k, s in cols.items()})
    L += ["| 연도 | " + " | ".join(yr.columns) + " |", "|---|" + "---|" * len(yr.columns)]
    for y, row in yr.iterrows():
        L.append(f"| {y} | " + " | ".join(_f(v, "+.1%") for v in row.values) + " |")
    L += ["", "## 한계", "",
          "- 개발 구간(2016~2025)은 지금까지 많은 시도에 쓰였다. A 를 PT-1 에 얹기로 한 것도 A 의 이 구간 결과를 본 뒤라, "
          "결과가 좋아도 낙관적일 수 있다 (A 의 숫자 자체는 2008년 전 자료로 정해져 다시 맞추지 않았다).",
          "- PT-1 백테스트의 0주 매수 버그(약세장 2슬롯 → 3슬롯 전환 시)는 BASE·변형 모두에 그대로 있다.",
          "- 현금 이자는 전날 현금 비중으로 근사했다. 홀드아웃(2025-10~)은 쓰지 않았다.",
          "- 결과가 좋아도 실전 반영(src 변경·push)은 사용자 결정이다.", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[timing] 보고서: {REPORT.relative_to(ROOT)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true", help="저장한 실행 결과로 보고서만 다시")
    a = ap.parse_args()
    if a.register:
        register()
    elif a.run:
        run()
    else:
        report_only()


if __name__ == "__main__":
    main()
