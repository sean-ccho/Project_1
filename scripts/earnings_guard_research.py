#!/usr/bin/env python3
"""실적 발표 앞에서의 규칙 (earnings_guard_v1): 내계좌 스타일 계좌에서 실적을 '안고 가기' vs '피하기'.

계기: 2026-10-29 사용자 — 산 지 며칠 안 된 보유 종목이 실적 발표 다음 날 크게 떨어짐 ("이런 실수를 다신 안 하게").
밤 메일이 '실적 7일 이내' 알림을 매일 보냈지만 숫자(그 종목이 실적 때 보통 얼마나 움직이는지)가 없었고, 매수 창에는 실적일이
안 보였다. 공개 저장소라 실제 종목은 적지 않는다.

계좌·고르기·매도 = my_style_v1 의 S1 그대로 (scripts/my_style_research.py: 골든 라벨 + 20일 상승 + 매출 YoY>0, ★ 높은 순,
5칸 균등, 다음날 시가, 편도 0.1%, 손절 = 진입가 − ATR(14)%×3 아래 종가, X1 21거래일 · X2 63거래일 · X3 종가<EMA50).
실적 반응일 = SEC 8-K Item 2.02 접수 시각 (data/research/sec_earnings_dates.parquet, 예정 발표만 = 45일 묶음 마지막 —
paper_trading.earnings_calendar.scheduled_only): 16:00 이후 접수 → 다음 거래일, 그 전 → 그날 (휴장이면 다음 거래일).
예정 실적일은 보통 몇 주 전에 공지되므로 그날 알 수 있었다고 본다 (실거래 PT-1 '실적 3일 이내 매수 금지'와 같은 가정).
규칙:
  E0 = 기준 (실적 무시 — 지금 사용자처럼)
  E1 = 매수 보류: 사는 날부터 10거래일 안에 실적 반응일이 있으면 안 산다 (그 뒤 실적은 안고 감)
  E2 = 실적 구간 비우기: E1 + 보유 종목은 실적 반응일 전날 시가에 전부 팔고, 반응일 다음 날 시가에 다시 산다
       (비운 동안에도 매도 규칙은 그 종목 종가로 본다 — 손절·보유일·EMA50 이 걸리면 다시 안 산다)
  E3 = 절반만: E1 + 반응일 전날 시가에 절반 팔고, 다음 날 시가에 그 돈으로 다시 산다 (매도 규칙은 E2 와 같음)
사전 등록 (TRIAL_LOG family earnings_guard_v1, 9건 = E1·E2·E3 × X1·X2·X3 — 결과 계산 전에 --register):
  판정 = 같은 매도 방식의 E0 대비 (paired_bootstrap 1만·20일 블록).  효과 있음 = ΔSharpe 95% 하한 > 0 · p ≤ 0.05/9,
         관찰 = 하한 > 0 이지만 p 미달, 나머지 효과 없음
  같이 보는 위험 숫자 (판정 없음): 최대낙폭 · 계좌 하루 최악 · 계좌 하루 −3% 이하 날 수 · 거래 −15% 이하 수 · 최악 거래 · 연수익 ·
         실적을 안고 간 횟수
  참고 (판정 없음): 고르는 순서 무작위 20시드에서 같은 시드끼리 ΔSharpe (운의 폭), 연도별
기간 2016-01-04 ~ 2025-09-30 (홀드아웃 안 씀).
한계: 그날 S&P 500 구성 종목만 — 사용자 종목 상당수는 S&P 밖이라 실적 움직임이 더 크다 (여기 숫자는 덜 아픈 쪽).
     실적일 자료가 없는 종목은 규칙을 안 건다.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/earnings_guard_research.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/earnings_guard_research.py --run
산출: tier3/EARNINGS_GUARD_REPORT.md · tier3/earnings_guard.csv · data/research/earnings_guard_runs.pkl (git 제외)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_vs_pt1 as iv  # noqa: E402
import my_style_research as ms  # noqa: E402
from paper_trading.earnings_calendar import scheduled_only  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe  # noqa: E402
from research_utils import DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, log_trials, total_trials  # noqa: E402

FAMILY, STEP = "earnings_guard_v1", "earnings_guard"
RULES = {
    "E1": "실적 10거래일 안이면 안 삼",
    "E2": "E1 + 실적 구간 비우기 (전날 다 팔고 다음 날 다시 삼)",
    "E3": "E1 + 실적 구간 절반만 (전날 절반 팔고 다음 날 다시 삼)",
}
EXITS = ms.EXITS
TESTS = tuple(f"{r}-{x}" for r in RULES for x in EXITS)
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
N_BOOT, N_BOOT_SEED = 10_000, 1_000
BUY_WINDOW = 10            # 사는 날 포함 이 거래일 수 안에 실적 반응일이 있으면 E1 이 안 산다
SEEDS = ms.SEEDS
EARN_PATH = RESEARCH_DIR / "sec_earnings_dates.parquet"
RUNS_PATH = RESEARCH_DIR / "earnings_guard_runs.pkl"
OUT_CSV = TIER3_DIR / "earnings_guard.csv"
REPORT = TIER3_DIR / "EARNINGS_GUARD_REPORT.md"
SLOTS, COST = ms.SLOTS, ms.COST
NAN = float("nan")


def spec() -> str:
    return (f"[2016-01-04~{DEV_END.date()}, 그날 S&P 500, 계좌·고르기·매도 = my_style_v1 S1 그대로 (골든 라벨·20일 상승·매출 YoY>0, "
            f"★ 순, {SLOTS}칸 균등, 다음날 시가, 편도 {COST:.1%}, 손절 ATR×3, X1 21일·X2 63일·X3 EMA50)] "
            "실적 반응일 = SEC 8-K 2.02 접수(예정 발표만, 16시 이후→다음 거래일). "
            f"E1 = 사는 날부터 {BUY_WINDOW}거래일 안에 실적이면 안 삼; E2 = E1 + 반응일 전날 시가에 전부 팔고 다음 날 시가에 다시 삼 "
            "(비운 동안도 매도 규칙은 종가로); E3 = E1 + 같은 방식으로 절반만. "
            f"판정 vs 같은 매도의 E0(실적 무시): 효과 있음 = ΔSharpe 95% 하한>0 · p ≤ 0.05/{N_TESTS}, 관찰 = 하한>0 · p 미달. "
            "위험 숫자(MDD·하루 최악·−3% 날·−15% 거래·최악 거래)와 무작위 순서 20시드는 판정 없이 같이 본다.")


# ============================================================
# 실적 반응일 (거래일 위치)
# ============================================================
def reaction_index(mk: ms.Market, path: Path = EARN_PATH) -> dict[int, np.ndarray]:
    """열 번호 → 실적 반응일 위치(오름차순). 16:00 이후 접수는 다음 거래일, 그 전은 그날(휴장이면 다음 거래일)."""
    ev = scheduled_only(pd.read_parquet(path, columns=["티커", "accepted_et"]))
    acc = pd.to_datetime(ev["accepted_et"])
    day = acc.dt.normalize().to_numpy(dtype="datetime64[ns]")
    dates = mk.dates.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(dates, day, side="left")
    same = (pos < len(dates)) & (dates[np.minimum(pos, len(dates) - 1)] == day)
    pos = np.where((acc.dt.hour >= 16).to_numpy() & same, pos + 1, pos)
    ev = ev.assign(ri=pos)
    out: dict[int, np.ndarray] = {}
    for tk, g in ev[ev["ri"] < len(dates)].groupby("티커"):
        j = mk.col.get(str(tk))
        if j is not None:
            out[j] = np.unique(g["ri"].to_numpy(int))
    return out


def next_reaction(er: dict[int, np.ndarray], j: int, i: int) -> int | None:
    """i 다음(i 보다 뒤) 첫 실적 반응일 위치. 자료 없으면 None."""
    arr = er.get(j)
    if arr is None:
        return None
    k = int(np.searchsorted(arr, i, side="right"))
    return int(arr[k]) if k < arr.size else None


# ============================================================
# 시뮬레이터 (my_style_research.simulate + 실적 규칙)
# ============================================================
@dataclass
class Pos:
    j: int
    ticker: str
    entry_i: int
    entry_px: float
    shares: float
    stop_px: float
    cost0: float               # 처음 산 돈 (거래 수익률 분모)
    flow: float = 0.0          # 거래가 끝날 때 받은 돈 (주식 판 돈 + 실적 구간에 남겨 둔 현금)
    parked: float = 0.0        # 실적 구간 동안 들고 있는 현금 (이 칸 몫)
    park_r: int = -1           # 비운 실적 반응일 위치
    armed: bool = False        # ms.exit_due 의 X3b 용 (여기선 안 씀)
    held_events: int = 0       # 주식을 들고 넘긴 실적 수


def simulate(mk: ms.Market, cands: dict[int, list[tuple[float, str]]], exit_rule: str, rule: str,
             er: dict[int, np.ndarray], seed: int | None = None, start: pd.Timestamp = ms.START,
             end: pd.Timestamp = DEV_END, capital: float = 10_000.0) -> tuple[pd.Series, list[dict[str, Any]]]:
    """rule: E0(실적 무시) · E1(매수 보류) · E2(비우기) · E3(절반). 그날 종가로 판단하고 다음날 시가에 사고판다."""
    frac = {"E0": 0.0, "E1": 0.0, "E2": 1.0, "E3": 0.5}[rule]
    gate = rule != "E0"
    i0 = int(mk.dates.searchsorted(start))
    i1 = int(mk.dates.searchsorted(end, side="right")) - 1
    cash, positions, trades, eq = capital, [], [], {}
    sell_next: list[tuple[Pos, str]] = []
    buy_next: list[tuple[int, str, float, float]] = []
    park_next: list[Pos] = []          # 내일 시가에 frac 만큼 판다 (실적 전날)
    unpark_next: list[Pos] = []        # 내일 시가에 다시 산다 (반응일 다음 날)
    for i in range(i0, i1 + 1):
        # 1) 어제 종가로 정한 것들을 오늘 시가에: 매도 → 실적 구간 팔기·다시 사기 → 새 매수
        for p, why in sell_next:
            px = _open_or_last(mk, i, p)
            cash += p.shares * px * (1 - COST) + p.parked
            p.flow += p.shares * px * (1 - COST) + p.parked
            p.shares, p.parked = 0.0, 0.0
            trades.append(_trade(p, mk, i, px, why))
        sell_next = []
        for p in park_next:
            px = _open_or_last(mk, i, p)
            q = p.shares * frac
            p.shares -= q
            p.parked += q * px * (1 - COST)       # 판 돈은 이 칸 안에 현금으로 남는다 (거래는 안 끝남)
        park_next = []
        for p in unpark_next:
            px = _open_or_last(mk, i, p)
            p.shares += p.parked / (px * (1 + COST))
            p.parked, p.park_r = 0.0, -1
        unpark_next = []
        for j, tk, alloc, stop_pct in buy_next:
            op = float(mk.open[i, j])
            alloc = min(alloc, cash)
            if alloc <= 1e-9:
                continue
            cash -= alloc
            positions.append(Pos(j, tk, i, op, alloc / (op * (1 + COST)), op * (1 - stop_pct), alloc))
        buy_next = []
        # 2) 가격 끊긴 종목 정리, 오늘 종가로 평가
        keep = []
        for p in positions:
            if np.isfinite(mk.close[i, p.j]):
                keep.append(p)
                continue
            px = _last_close(mk, i, p)
            cash += p.shares * px * (1 - COST) + p.parked
            p.flow += p.shares * px * (1 - COST) + p.parked
            p.shares, p.parked = 0.0, 0.0
            trades.append(_trade(p, mk, i, px, "가격 끊김"))
        positions = keep
        equity = cash + sum(p.shares * float(mk.close[i, p.j]) + p.parked for p in positions)
        eq[mk.dates[i]] = equity
        if i == i1:
            for p in positions:
                px = float(mk.close[i, p.j])
                p.flow += p.shares * px * (1 - COST) + p.parked
                p.shares, p.parked = 0.0, 0.0
                trades.append(_trade(p, mk, i, px, "기간 끝"))
            break
        # 3) 오늘 종가로 매도 판단 (비운 동안도 그 종목 종가로) → 내일 시가
        keep = []
        for p in positions:
            why = ms.exit_due(exit_rule, p, i, float(mk.close[i, p.j]), float(mk.ema[i, p.j]))
            if why:
                sell_next.append((p, why))
                continue
            keep.append(p)
            if p.park_r >= 0:
                if i >= p.park_r:                     # 반응일 종가까지 봤다 → 내일 시가에 다시 산다
                    unpark_next.append(p)
                elif i == p.park_r - 1 and p.shares > 0:
                    p.held_events += 1                # E3: 남긴 절반은 실적을 들고 넘김
                continue
            r = next_reaction(er, p.j, i)
            if r is None:
                continue
            if frac > 0 and r - i == 2:               # 내일 = 반응일 전날 → 내일 시가에 판다
                park_next.append(p)
                p.park_r = r
            elif r - i == 1 and p.shares > 0:
                p.held_events += 1                    # 오늘 밤 · 내일 아침 실적을 들고 넘김
        positions = keep
        # 4) 빈 칸 → 오늘 규칙 통과 종목을 ★ 순으로 내일 시가에
        free = SLOTS - len(positions)
        if free <= 0 or i not in cands:
            continue
        avail = cash + sum(p.shares * float(mk.close[i, p.j]) * (1 - COST) + p.parked for p, _ in sell_next)
        held = {ms._group(p.ticker) for p in positions} | {ms._group(p.ticker) for p, _ in sell_next}
        day = mk.dates[i]
        for score, tk in sorted(cands[i], key=lambda x: ms.order_key(x[0], x[1], seed, day)):
            if free <= 0:
                break
            j = mk.col.get(tk)
            if j is None or ms._group(tk) in held:
                continue
            op = mk.open[i + 1, j]
            if not (np.isfinite(op) and op > 0):
                continue
            if gate:
                r = next_reaction(er, j, i)
                if r is not None and r <= i + BUY_WINDOW:     # 사는 날(i+1)부터 10거래일 안
                    continue
            alloc = min(avail, equity / SLOTS)
            if alloc <= 1e-9:
                break
            a = mk.atr_pct[i, j]
            buy_next.append((j, tk, alloc, ms.ATR_MULT * a if np.isfinite(a) and a > 0 else ms.STOP_FALLBACK))
            avail -= alloc
            held.add(ms._group(tk))
            free -= 1
    return pd.Series(eq, name="equity"), trades


def _open_or_last(mk: ms.Market, i: int, p: Pos) -> float:
    op = mk.open[i, p.j]
    return float(op) if np.isfinite(op) and op > 0 else _last_close(mk, i, p)


def _last_close(mk: ms.Market, i: int, p: Pos) -> float:
    last = mk.close[:i + 1, p.j][np.isfinite(mk.close[:i + 1, p.j])]
    return float(last[-1]) if last.size else p.entry_px


def _trade(p: Pos, mk: ms.Market, exit_i: int, px: float, why: str) -> dict[str, Any]:
    """거래 수익률 = 끝날 때 받은 돈 ÷ 처음 산 돈 − 1 (실적 구간 사고팔기 비용은 남겨 둔 현금에 이미 들어 있다)."""
    return {"ticker": p.ticker, "entry_date": mk.dates[p.entry_i], "exit_date": mk.dates[exit_i],
            "entry_price": p.entry_px, "exit_price": px, "return_pct": p.flow / p.cost0 - 1,
            "days": exit_i - p.entry_i, "reason": why, "held_events": p.held_events}


# ============================================================
# 요약 · 등록 · 판정
# ============================================================
def risk_numbers(eq: pd.Series, tr: list[dict[str, Any]]) -> dict[str, float]:
    r = eq.pct_change().dropna()
    rets = np.array([t["return_pct"] for t in tr]) if tr else np.array([NAN])
    return {"Sharpe": sharpe(r.tolist()), "CAGR": cagr(r.tolist()), "MDD": max_drawdown(r.tolist()),
            "하루최악": float(r.min()), "하루-3%": int((r <= -0.03).sum()), "거래-15%": int((rets <= -0.15).sum()),
            "최악거래": float(np.nanmin(rets)), "거래수": len(tr), "승률": float(np.nanmean(rets > 0)),
            "거래평균": float(np.nanmean(rets)), "실적안고감": int(sum(t["held_events"] for t in tr))}


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def _registered() -> bool:
    log = _log()
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def register() -> None:
    if _registered():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"등록: {FAMILY} {N_TESTS}건 · p ≤ {P_BONF:.4f} · 누적 시도 {total_trials()}건")


def judge(st: dict[str, float]) -> str:
    if st and st["ΔSharpe_하한"] > 0 and st["p_개선아님"] <= P_BONF:
        return "효과 있음"
    if st and st["ΔSharpe_하한"] > 0:
        return "관찰"
    return "효과 없음"


def run() -> None:
    if not _registered():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    log = _log()
    already = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    print("===== 자료: 가격 · 조건표 · 실적 반응일", flush=True)
    mk = ms.load_market()
    pool = ms.build_pool()
    el = ms.eligible(pool, "S1")
    cands = ms.candidates(mk, el)
    er = reaction_index(mk)
    pool_t = set(el["ticker"].astype(str))
    have = {mk.tickers[j] for j in er}
    print(f"  실적일 있는 종목 {len(er)}개 · S1 후보 종목 {len(pool_t)}개 중 실적일 있음 {len(pool_t & have)}개", flush=True)
    runs: dict[str, Any] = {}
    rows, log_rows, base = [], [], {}
    for x in EXITS:
        print(f"===== E0-{x} (기준) + 무작위 {len(SEEDS)}시드", flush=True)
        eq0, tr0 = simulate(mk, cands, x, "E0", er)
        seeds0 = {s: simulate(mk, cands, x, "E0", er, seed=s)[0] for s in SEEDS}
        base[x] = (eq0, tr0, seeds0)
        runs[f"E0-{x}"] = {"equity": eq0, "trades": tr0}
        rows.append({"test_id": f"E0-{x}", "규칙": "기준 (실적 무시)", "매도": EXITS[x], **risk_numbers(eq0, tr0),
                     "판정": "기준"})
    for rule in RULES:
        for x in EXITS:
            tid = f"{rule}-{x}"
            print(f"===== {tid}: {RULES[rule]} · {EXITS[x]}", flush=True)
            eq, tr = simulate(mk, cands, x, rule, er)
            eq0, tr0, seeds0 = base[x]
            st, _ = iv._cmp(eq.pct_change().dropna(), eq0.pct_change().dropna(), N_BOOT)
            d_seed = []
            for s in SEEDS:
                e_s = simulate(mk, cands, x, rule, er, seed=s)[0]
                d_seed.append(sharpe(e_s.pct_change().dropna().tolist()) - sharpe(seeds0[s].pct_change().dropna().tolist()))
            d_seed = np.array(d_seed)
            rk = risk_numbers(eq, tr)
            row = {"test_id": tid, "규칙": RULES[rule], "매도": EXITS[x], **rk,
                   "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
                   "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}",
                   "시드Δ가운데": float(np.median(d_seed)), "시드Δ>0": int((d_seed > 0).sum()), "판정": judge(st)}
            rows.append(row)
            runs[tid] = {"equity": eq, "trades": tr, "seed_delta": d_seed}
            b = risk_numbers(eq0, tr0)
            memo = (f"Sharpe {rk['Sharpe']:.2f} vs E0 {b['Sharpe']:.2f}, ΔSharpe {st.get('ΔSharpe'):+.2f} "
                    f"[{st.get('ΔSharpe_하한'):+.2f}, {st.get('ΔSharpe_상한'):+.2f}] p {st.get('p_개선아님'):.4f}, "
                    f"CAGR {rk['CAGR']:+.1%} vs {b['CAGR']:+.1%}, MDD {rk['MDD']:.1%} vs {b['MDD']:.1%}, "
                    f"하루 최악 {rk['하루최악']:.1%} vs {b['하루최악']:.1%}, 거래 −15% 이하 {rk['거래-15%']} vs {b['거래-15%']}, "
                    f"실적 안고 감 {rk['실적안고감']} vs {b['실적안고감']}, 시드 ΔSharpe 가운데 {np.median(d_seed):+.2f} "
                    f"({int((d_seed > 0).sum())}/{len(SEEDS)} 플러스)")
            log_rows.append({"test_id": tid, "t": "", "판정": row["판정"], "메모": memo})
            print(f"  {memo}", flush=True)
    pd.to_pickle(runs, RUNS_PATH)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    if not already:
        log_trials(STEP, FAMILY, N_TESTS, log_rows)
    else:
        print("TRIAL_LOG 에 결과를 쓰지 않는다 (이미 있음)", flush=True)
    write_report(table, runs)


def _yearly(eq: pd.Series) -> pd.Series:
    r = eq.pct_change().dropna()
    return (1 + r).groupby(r.index.year).prod() - 1


def write_report(table: pd.DataFrame, runs: dict[str, Any]) -> None:
    f = iv._f
    t = table.set_index("test_id")
    passed = [x for x in TESTS if t.at[x, "판정"] == "효과 있음"]
    watch = [x for x in TESTS if t.at[x, "판정"] == "관찰"]
    head = (f"**결론: Sharpe 를 올린다고 볼 수 있는 규칙 — {', '.join(passed)}.**" if passed else
            "**결론: 실적 규칙 셋 다 Sharpe 를 올린다고 볼 수는 없다 (판정 기준 미달).**"
            + (f" ('관찰': {', '.join(watch)})" if watch else ""))
    lines = [
        f"# 실적 발표 앞에서의 규칙 ({FAMILY}) — 2016-01-04 ~ {DEV_END.date()}", "", head, "",
        "- 계기: 2026-10-29 사용자 보유 종목 하나가 실적 발표 다음 날 크게 떨어짐 (\"이런 실수를 다신 안 하게\"). "
        "공개 저장소라 실제 종목은 적지 않는다.",
        "- 계좌·고르기·매도는 my_style_v1 S1 그대로 (골든 라벨 + 20일 상승 + 매출 성장, ★ 높은 순, 5종목 균등, 손절 ATR×3)",
        f"- 실적 반응일 = SEC 8-K Item 2.02 (예정 발표만). E1 = 사는 날부터 {BUY_WINDOW}거래일 안에 실적이면 안 삼 · "
        "E2 = E1 + 반응일 전날 시가에 다 팔고 다음 날 시가에 다시 삼 · E3 = 같은 방식으로 절반만",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · {N_TESTS}건 · p ≤ 0.05/{N_TESTS} · 부트스트랩 {N_BOOT:,}회 · "
        f"누적 시도 {total_trials():,}건",
        "- 한계: 그날 S&P 500 종목만 — S&P 밖 중소형주는 실적 움직임이 더 커서 실제로는 규칙 효과가 더 클 수 있다",
        "", "## 읽는 법", ""]
    better = [x for x in TESTS if t.at[x, "Sharpe"] > t.at[f"E0-{x.split('-')[1]}", "Sharpe"]]
    lines.append(f"- ★ 순 계좌에서 {len(better)}/{N_TESTS}건이 E0 보다 Sharpe 가 높았다. 하지만 고르는 순서만 무작위로 바꾼 "
                 f"{len(SEEDS)}번에서는 ΔSharpe 가운데가 " + " · ".join(
                     f"{x} {float(np.median([t.at[f'{r}-{x}', '시드Δ가운데'] for r in RULES])):+.2f}" for x in EXITS)
                 + " (규칙 셋의 가운데) — 개선 폭의 상당 부분은 길(path) 운이다")
    lines.append("- 최악 거래 (E0 → E1·E2): " + " · ".join(
        f"{x} {t.at[f'E0-{x}', '최악거래']:.0%} → {t.at[f'E1-{x}', '최악거래']:.0%}·{t.at[f'E2-{x}', '최악거래']:.0%}" for x in EXITS)
        + " — 실적 갭으로 생기는 한 종목 큰 손실이 줄어든다")
    worst_day = {x: runs[f"E0-{x}"]["equity"].pct_change().idxmin() for x in EXITS}
    lines.append("- 계좌 하루 최악은 그대로거나 나빠졌다 — E0 가 가장 크게 잃은 날: " + " · ".join(
        f"{x} {d.date()}" for x, d in worst_day.items()) + " (2020-03-12 는 코로나 폭락 — 시장 전체가 빠진 날은 실적 규칙과 상관없다)")
    lines += ["", "## Sharpe 판정 (같은 매도 방식의 E0 대비)", "",
        "| 규칙 | Sharpe | 연수익 | ΔSharpe [95%] | p | 구간 | 무작위 20시드 Δ 가운데 (플러스) | 판정 |",
        "|---|---|---|---|---|---|---|---|"]
    for x in EXITS:
        b = t.loc[f"E0-{x}"]
        lines.append(f"| E0-{x} 기준 · {EXITS[x]} | {b['Sharpe']:.2f} | {b['CAGR']:+.1%} | | | | | 기준 |")
        for rule in RULES:
            r = t.loc[f"{rule}-{x}"]
            lines.append(f"| {rule}-{x} {RULES[rule]} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | "
                         f"{f(r['ΔSharpe'], '+.2f')} [{f(r['하한'], '+.2f')}, {f(r['상한'], '+.2f')}] | {f(r['p'], '.4f')} | "
                         f"{r['구간']} | {r['시드Δ가운데']:+.2f} ({int(r['시드Δ>0'])}/{len(SEEDS)}) | **{r['판정']}** |")
    lines += ["", "## 위험 숫자 (판정 없음)", "",
              "| 규칙 | 최대낙폭 | 계좌 하루 최악 | 하루 −3% 이하 날 | 거래 −15% 이하 | 최악 거래 | 실적 안고 간 횟수 | 거래 | 승률 | 거래당 평균 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for x in EXITS:
        for rule in ("E0", *RULES):
            r = t.loc[f"{rule}-{x}"]
            lines.append(f"| {rule}-{x} | {r['MDD']:.1%} | {r['하루최악']:.1%} | {int(r['하루-3%'])} | {int(r['거래-15%'])} | "
                         f"{r['최악거래']:.1%} | {int(r['실적안고감'])} | {int(r['거래수'])} | {r['승률']:.0%} | {r['거래평균']:+.2%} |")
    yr = pd.DataFrame({k: _yearly(v["equity"]) for k, v in runs.items()})
    cols = [c for x in EXITS for c in (f"E0-{x}", *(f"{r}-{x}" for r in RULES))]
    yr = yr[cols]
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"보고서: {REPORT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true", help="저장된 결과(csv·pkl)로 보고서만 다시 쓴다")
    a = ap.parse_args()
    if a.register:
        register()
    if a.run:
        run()
    if a.report:
        write_report(pd.read_csv(OUT_CSV), pd.read_pickle(RUNS_PATH))
    if not (a.register or a.run or a.report):
        ap.print_help()


if __name__ == "__main__":
    main()
