#!/usr/bin/env python3
"""연구소(lab) 1~3단계 — 전략군 A~E 탐색 → 검증 → 결합(F) → 재확인 → 보고서.

사전 등록 (TRIAL_LOG family lab_{A..F}_v1, --register 가 결과 계산 전에 SPEC 을 기록):
- 탐색: A·E 는 1950-01 ~ 2007-12 Kenneth French 미국 시장 총수익, B·C·D 는 1990-01 ~ 2007-12 대용 펀드→ETF.
  탐색 자료는 2007-12-31 에서 잘라서만 본다. 조합마다 옵션 A 점수(초과 Sharpe − max(0, |MDD| − 30%) × 2).
- 선택: 자기 + 이웃(손잡이 하나를 한 칸 옮긴 조합) 점수 평균이 가장 높은 조합. 매매 주기만 다른 조합 중
  탐색 Sharpe 가 0.05 이내면 더 느린 주기.
- 통과 (모두): ① PBO < 0.3 ② 이웃 80% 이상이 탐색 벤치마크보다 Sharpe 높음 ③ 검증(2008~2015)·
  ④ 재확인(2016-01 ~ 2025-09) 둘 다 SPY 보다 초과 Sharpe 높고, 변동성 맞춤판(σ* = 탐색 구간 벤치마크 변동성,
  63일 공분산 사전 추정, ≤ 1.5배, 빌린 몫 무위험 + 1%)의 CAGR > SPY · MDD ≥ SPY ⑤ (A·E) 해외 지수 6개 중
  4개 이상에서 매수보유보다 Sharpe 높음 ⑥ 재확인 구간에서 비용 2배·체결 하루 지연에도 ④ 유지.
- F 결합: ①②③(⑤) 을 통과한 군이 2개 이상이면 위험 균등(EW·RP63·RP252 중 탐색 점수 최고)으로 섞고 ④⑥ 으로 판정.
- 참고(판정 없음): ΔSharpe 95% 구간·p (블록 부트스트랩 10,000회), DSR, 시가 체결, 주기별 표, 2배 ETF 비용 대조.

사용:
  .venv/bin/python scripts/lab_search.py --register
  .venv/bin/python scripts/lab_search.py --run       # TRIAL_LOG 결과 행은 한 번만 쓴다
  .venv/bin/python scripts/lab_search.py --report    # 저장한 결과로 보고서만 다시
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lab_data as ld  # noqa: E402
from lab_backtest import (BORROW_SPREAD, SCHEDULE_KO, SCHEDULE_ORDER, SCHEDULES, TRADING_DAYS,  # noqa: E402
                          decision_mask, perf, run_schedule, score, synthetic_2x, vol_overlay)
from lab_families import FAMILIES, MENUS, Family, Universe, schedule_of, target_key  # noqa: E402
from paper_trading.evaluation import deflated_sharpe, paired_bootstrap  # noqa: E402
from research_utils import COST_PER_SIDE, DEV_END, TIER3_DIR, TRIAL_LOG, log_trials, pbo_cscv, total_trials  # noqa: E402

STEP = "stage16"
VERSION = "v1"
FAMILY_IDS = {c: f"lab_{c}_{VERSION}" for c in "ABCDEF"}
SEARCH = {"index": (pd.Timestamp("1950-01-03"), pd.Timestamp("2007-12-31")),
          "multi": (pd.Timestamp("1990-01-02"), pd.Timestamp("2007-12-31"))}
VALID = (pd.Timestamp("2008-01-02"), pd.Timestamp("2015-12-31"))
CONFIRM = (pd.Timestamp("2016-01-04"), DEV_END)
KF_START = pd.Timestamp("1948-01-02")
OPEN_START = pd.Timestamp("2015-01-02")     # 시가 체결 확인은 이날부터 돌려 재확인 구간만 본다
PBO_MAX = 0.3
NEIGHBOR_MIN = 0.8
FREQ_TOL = 0.05
LEV_CAP = 1.5
OVERLAY_L = 63
INTL_MIN = 4
N_BOOT = 10_000
F_WEIGHTINGS = ("EW", "RP63", "RP252")
ETF_FOR = {"US": "SPY", "DEV": "EFA", "EM": "EEM", "LTT": "TLT", "ITT": "IEF", "REIT": "VNQ", "GOLD": "GLD",
           "CMDTY": "DBC"}
OUT_DIR = ld.LAB_DIR / f"search_{VERSION}"
STATE = OUT_DIR / "state.pkl"
OPTUNA_DB = OUT_DIR / "optuna.db"
REPORT = TIER3_DIR / "LAB_REPORT.md"
TRIALS_CSV = TIER3_DIR / "lab_trials.csv"
RESULTS_CSV = TIER3_DIR / "lab_results.csv"
GATE_KO = {"g1": "① PBO < 0.3", "g2": "② 이웃 80%+", "g3": "③ 검증", "g4": "④ 재확인", "g5": "⑤ 해외",
           "g6": "⑥ 스트레스"}


# ── 사전 등록 ─────────────────────────────────────────────────────────────────────────────────

def spec(code: str) -> str:
    """TRIAL_LOG 등록 문구 (결과 보기 전에 고정)."""
    common = ("검증 2008-01~2015-12 · 재확인 2016-01~2025-09 (SPY, 현금 = Kenneth French 1개월 국채) · "
              "t 종가 신호 → t+1 종가 체결 · 편도 0.1% · Sharpe = 무위험 초과수익 기준 · "
              "선택 = 옵션 A 점수(초과 Sharpe − max(0,|MDD|−30%)×2)의 자기+이웃 평균 최고, 주기만 다른 조합은 "
              "탐색 Sharpe 0.05 이내면 느린 주기 · 통과 = ① PBO(CSCV 16구간) < 0.3 ② 이웃 80%+ 가 탐색 벤치마크보다 "
              "Sharpe 높음 ③ 검증·④ 재확인 모두 SPY 보다 초과 Sharpe 높고 변동성 맞춤판(σ* = 탐색 벤치마크 변동성, "
              "63일 공분산, ≤1.5배, 빌린 몫 무위험+1%) CAGR > SPY·MDD ≥ SPY")
    stress = " ⑥ 재확인 구간 비용 2배·체결 하루 지연에도 ④ 유지"
    if code == "A":
        return ("[lab A 지수 위험관리] 탐색 1950-01~2007-12 Kenneth French 미국 시장 총수익 · 격자 960 = 추세 {없음, "
                "총수익지수>SMA 50·100·150·200·250, 초과 모멘텀>0 63·126·189·252일, 이평 교차 20/100·50/200} × 변동성 "
                "목표 {없음, min(1, σ/σ̂) σ 10·12·15·18·20% × 21·63·126일} × 주기 {월, 주, 매일 폭 10·5·2%p} · "
                + common + " ⑤ 해외 6개(닛케이·FTSE·DAX·항셍·KOSPI·TSX, 가격 지수·현금 0) 중 4+ 에서 매수보유보다 Sharpe 높음"
                + stress)
    if code == "B":
        return ("[lab B 자산배분 추세추종] 탐색 1990-01~2007-12 대용 펀드→ETF (미국 VFINX→SPY, 선진국 VEURX·VPACX "
                "60/40→EFA, 신흥국 VEIEX→EEM, 장기채 VUSTX→TLT, 중기채 VFITX→IEF, 리츠 VGSIX→VNQ, 금 GC=F→GLD, "
                "원자재 DBC), 기록 253일 이상 자산만 · 격자 320 = 메뉴 {4자산, 8자산} × 신호 {초과 모멘텀>0 63·126·"
                "189·252일, 총수익지수>SMA 100·150·200, 1·3·6·12개월 플러스 비율} × 비중 {동일, 역변동성 21·63·126일} "
                "× 주기 5 · 신호 꺼진 몫은 현금 · " + common + stress)
    if code == "C":
        return ("[lab C 상대·듀얼 모멘텀] 탐색 1990-01~2007-12 (B 와 같은 자료) · 격자 360 = 위험자산 {미국·선진국·"
                "신흥국, +리츠·금·원자재} × 대피 {현금, 중기채, 장기채} × 점수 {초과 모멘텀 63·126·252일, 12·4·2·1 "
                "가중 1·3·6·12개월} × 상위 k {1,2,3} × 주기 5 · 고른 자산 점수 ≤ 0 이면 그 몫은 대피 자산 · "
                + common + stress)
    if code == "D":
        return ("[lab D 위험 균등 정적 배분] 탐색 1990-01~2007-12 (B 와 같은 자료) · 격자 100 = 메뉴 {4자산, 8자산, "
                "주식+중기채, 미국·장기채·중기채·금·원자재} × 비중 {동일, 역변동성 21·63·126·252일} × 주기 5 · "
                + common + stress)
    if code == "E":
        return ("[lab E 달력 효과] 탐색 1950-01~2007-12 Kenneth French 미국 시장 총수익 · 격자 25 = 월말 마지막 k "
                "거래일(1~5) + 월초 첫 j 거래일(1~5)만 보유, 나머지 현금, 매일 · " + common
                + " ⑤ 해외 6개 중 4+ 에서 매수보유보다 Sharpe 높음" + stress)
    return ("[lab F 결합] ①②③(A·E 는 ⑤도)을 통과한 군이 2개 이상이면 각 군의 목표 비중을 위험 균등 {동일, 63일·252일 "
            "역변동성(군 수익 기준)} 중 탐색(1990-01~2007-12, A·E 는 미국 대용 시세) 옵션 A 점수 최고로 섞고, 매매 주기는 "
            "군 중 가장 빠른 것 · 변동성 맞춤판 기준 ④ 재확인·⑥ 스트레스로 판정 (군 고르기에 검증 구간을 써서 F 의 "
            "검증 숫자는 참고)")


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")


def registered() -> dict[str, bool]:
    log = _log()
    return {c: bool(((log["family"] == fid) & (log["판정"] == "등록")).any()) for c, fid in FAMILY_IDS.items()}


def results_logged() -> bool:
    log = _log()
    return bool((log["family"].isin(list(FAMILY_IDS.values())) & (log["판정"] != "등록")).any())


def register() -> None:
    """6개 전략군을 TRIAL_LOG 에 등록한다 (이미 있으면 건너뜀)."""
    done = registered()
    sizes = {c: len(FAMILIES[c].grid()) for c in FAMILIES} | {"F": len(F_WEIGHTINGS)}
    for c in "ABCDEF":
        if done[c]:
            print(f"[lab] {FAMILY_IDS[c]} 이미 등록됨", flush=True)
            continue
        log_trials(STEP, FAMILY_IDS[c], sizes[c], [{"test_id": "(등록)", "판정": "등록", "메모": spec(c)}])
        print(f"[lab] {FAMILY_IDS[c]} 등록 · 조합 {sizes[c]}", flush=True)
    print(f"[lab] 누적 시도 {total_trials():,}건", flush=True)


# ── 자료 ─────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Data:
    prices: pd.DataFrame
    french: pd.DataFrame
    calendar: pd.DatetimeIndex
    multi: Universe
    kf: Universe
    spy: Universe
    spy_ret: pd.Series
    open_multi: pd.DataFrame
    intl: dict[str, Universe] = field(default_factory=dict)


def load_data() -> Data:
    prices, french = ld.load_prices(), ld.load_french()
    cal = ld.us_calendar(french)
    rf = french["rf"]
    multi = Universe(ld.asset_returns(prices, cal), rf)
    kf = Universe(french[["mkt"]].rename(columns={"mkt": "EQ"}).loc[KF_START:], rf)
    spy_ret = ld.close_returns(prices, "SPY", cal).dropna()
    spy = Universe(spy_ret.to_frame("EQ"), rf)
    open_multi = pd.DataFrame({a: ld.open_returns(prices, t) for a, t in ETF_FOR.items()}).reindex(cal)
    intl = {}
    for t in ld.INTL:
        r = ld.close_returns(prices, t)
        intl[t] = Universe(r.to_frame("EQ"), pd.Series(0.0, index=r.index))
    return Data(prices, french, cal, multi, kf, spy, spy_ret, open_multi, intl)


def search_setup(d: Data, fam: Family) -> tuple[Universe, pd.Series, tuple[pd.Timestamp, pd.Timestamp]]:
    """탐색용 자료 (2007-12-31 에서 자른 것), 벤치마크 수익, 탐색 구간."""
    lo, hi = SEARCH[fam.kind]
    if fam.kind == "index":
        u = d.kf.truncate(hi)
        return u, u.ret["EQ"], (lo, hi)
    u = d.multi.truncate(hi)
    return u, u.ret["US"], (lo, hi)


def full_universe(d: Data, fam: Family) -> Universe:
    return d.spy if fam.kind == "index" else d.multi


def open_returns_for(d: Data, fam: Family) -> pd.DataFrame:
    if fam.kind == "index":
        return d.open_multi[["US"]].rename(columns={"US": "EQ"})
    return d.open_multi


# ── 1단계: 탐색 ───────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    code: str
    grid: list[dict[str, Any]]
    table: pd.DataFrame
    bench: dict[str, float]
    sigma_star: float
    chosen: int
    smooth_best: int
    pbo: dict[str, float]
    neighbor_share: float
    n_neighbors: int
    dsr: float


def smoothed_scores(fam: Family, grid: list[dict[str, Any]], scores: np.ndarray) -> np.ndarray:
    """자기 + 이웃 점수 평균 (점수가 nan 인 조합은 빼고)."""
    keys = [fam.key(c) for c in grid]
    pos = {k: i for i, k in enumerate(keys)}
    keyset = set(keys)
    out = np.full(len(grid), np.nan)
    for i, c in enumerate(grid):
        idx = [i] + [pos[k] for k in fam.neighbors(c, keyset)]
        vals = scores[idx]
        vals = vals[np.isfinite(vals)]
        if np.isfinite(scores[i]) and len(vals):
            out[i] = float(vals.mean())
    return out


def select(fam: Family, grid: list[dict[str, Any]], sharpe: np.ndarray, smooth: np.ndarray) -> tuple[int, int]:
    """(최종 선택, 이웃 평균 최고) — 주기만 다른 조합 중 Sharpe 가 0.05 이내면 더 느린 주기."""
    best = int(np.nanargmax(smooth))
    cfg = grid[best]
    if "sched" not in cfg:
        return best, best
    same = [i for i, c in enumerate(grid) if all(c[k] == cfg[k] for k in c if k != "sched")]
    ok = [i for i in same if np.isfinite(sharpe[i]) and sharpe[i] >= sharpe[best] - FREQ_TOL]
    return min(ok, key=lambda i: SCHEDULE_ORDER.index(grid[i]["sched"])), best


def neighbor_share(fam: Family, grid: list[dict[str, Any]], sharpe: np.ndarray, i: int,
                   bench_sharpe: float) -> tuple[float, int]:
    """선택한 조합의 이웃 중 탐색 벤치마크보다 Sharpe 가 높은 비율, 이웃 수."""
    keys = [fam.key(c) for c in grid]
    pos = {k: j for j, k in enumerate(keys)}
    nb = [pos[k] for k in fam.neighbors(grid[i], set(keys))]
    if not nb:
        return float("nan"), 0
    return float(np.mean([sharpe[j] > bench_sharpe for j in nb])), len(nb)


def search_family(d: Data, code: str) -> SearchResult:
    fam = FAMILIES[code]
    u, bench_r, (lo, hi) = search_setup(d, fam)
    grid = fam.grid()
    rf_w = u.rf.loc[lo:hi]
    days = u.ret.loc[lo:hi].index
    years = len(days) / TRADING_DAYS
    mat = np.empty((len(days), len(grid)), dtype=np.float32)
    rows = []
    cache_key, cache_tgt = None, None
    t0 = time.time()
    for j, cfg in enumerate(grid):
        tk = target_key(fam, cfg)
        if tk != cache_key:
            cache_key, cache_tgt = tk, fam.target(u, cfg)
        sim = run_schedule(cache_tgt, u.ret[list(cache_tgt.columns)], u.rf, schedule_of(cfg))
        r = sim.ret.loc[lo:hi]
        p = perf(r, rf_w)
        rows.append({"family": code, "key": fam.key(cfg), "설명": fam.describe(cfg), **p, "score": score(p),
                     "노출": float(sim.exposure.loc[lo:hi].mean()),
                     "매매_연": float((sim.turnover.loc[lo:hi] > 0).sum() / years),
                     "회전율_연": float(sim.turnover.loc[lo:hi].sum() / years)})
        mat[:, j] = (r - rf_w).to_numpy(dtype=np.float32)
        if (j + 1) % max(1, len(grid) // 10) == 0 or j + 1 == len(grid):
            print(f"[lab] {code} 탐색 {j + 1}/{len(grid)} [{(j + 1) / len(grid):.1%}] {time.time() - t0:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    bench = perf(bench_r.loc[lo:hi], rf_w)
    sh = table["Sharpe"].to_numpy(dtype=float)
    smooth = smoothed_scores(fam, grid, table["score"].to_numpy(dtype=float))
    table["smooth"] = smooth
    chosen, sbest = select(fam, grid, sh, smooth)
    table["선택"] = ""
    table.loc[sbest, "선택"] = "이웃평균 최고"
    table.loc[chosen, "선택"] = "선택" if chosen == sbest else "선택(느린 주기)"
    pbo = pbo_cscv(mat)
    share, n_nb = neighbor_share(fam, grid, sh, chosen, bench["Sharpe"])
    ex = mat[:, chosen].astype(float)
    xs = pd.Series(ex)
    dsr = deflated_sharpe(sh[chosen] / math.sqrt(TRADING_DAYS), list(sh / math.sqrt(TRADING_DAYS)), len(ex),
                          float(xs.skew()), float(xs.kurt()) + 3.0)
    print(f"[lab] {code} 선택: {fam.describe(grid[chosen])} · Sharpe {sh[chosen]:.2f} vs 벤치 {bench['Sharpe']:.2f} · "
          f"PBO {pbo['PBO']:.2f} · 이웃 {share:.0%} ({n_nb}) · DSR {dsr:.2f}", flush=True)
    return SearchResult(code, grid, table, bench, float(bench["vol"]), chosen, sbest, pbo, share, n_nb, float(dsr))


# ── 2·3단계: 검증 · 재확인 ─────────────────────────────────────────────────────────────────────

@dataclass
class Eval:
    p: dict[str, float]
    pv: dict[str, float]
    pb: dict[str, float]
    gate: bool
    lev_mean: float
    lev_max: float
    exposure: float
    trades_yr: float
    r: pd.Series
    rv: pd.Series
    rb: pd.Series
    boot: dict[str, float] = field(default_factory=dict)
    boot_v: dict[str, float] = field(default_factory=dict)


def gate_ok(p: dict[str, float], pv: dict[str, float], pb: dict[str, float]) -> bool:
    """③④: 초과 Sharpe > SPY, 변동성 맞춤판 CAGR > SPY, MDD ≥ SPY."""
    return bool(p["Sharpe"] > pb["Sharpe"] and pv["CAGR"] > pb["CAGR"] and pv["MDD"] >= pb["MDD"])


def evaluate_target(tgt: pd.DataFrame, u: Universe, sched: str, bench: pd.Series, window: tuple[pd.Timestamp, pd.Timestamp],
                    sigma_star: float, *, cost: float = COST_PER_SIDE, delay: int = 0, exec_mode: str = "close",
                    open_ret: pd.DataFrame | None = None, boot: bool = False) -> Eval:
    """목표 비중 → 그대로 · 변동성 맞춤판을 window 구간에서 SPY(bench) 와 비교."""
    cols = list(tgt.columns)
    freq = SCHEDULES[sched][0]
    ov = vol_overlay(tgt, u.ret[cols], sigma_star, freq=freq, lookback=OVERLAY_L, cap=LEV_CAP)
    if exec_mode == "open":
        assert open_ret is not None
        ar = open_ret[cols].loc[OPEN_START:]
        t_use, ov_use, rf = tgt.loc[OPEN_START:], ov.loc[OPEN_START:], u.rf.loc[OPEN_START:]
    else:
        ar, t_use, ov_use, rf = u.ret[cols], tgt, ov, u.rf
    sim = run_schedule(t_use, ar, rf, sched, cost=cost, delay=delay, exec_mode=exec_mode)
    simv = run_schedule(ov_use, ar, rf, sched, cost=cost, delay=delay, exec_mode=exec_mode)
    lo, hi = window
    rb = bench.loc[lo:hi].dropna()                 # 시가→시가 기준은 마지막 날 다음 시가가 없다
    common = sim.ret.loc[lo:hi].index.intersection(rb.index)
    r, rv, rb = sim.ret.loc[common], simv.ret.loc[common], rb.loc[common]
    rf_w = u.rf.reindex(r.index).fillna(0.0)
    p, pv, pb = perf(r, rf_w), perf(rv, rf_w), perf(rb, u.rf.reindex(rb.index).fillna(0.0))
    years = len(r) / TRADING_DAYS
    ev = Eval(p, pv, pb, gate_ok(p, pv, pb), float(simv.exposure.loc[lo:hi].mean()),
              float(simv.exposure.loc[lo:hi].max()), float(sim.exposure.loc[lo:hi].mean()),
              float((sim.turnover.loc[lo:hi] > 0).sum() / years), r, rv, rb)
    if boot:
        df = pd.concat({"s": r - rf_w, "v": rv - rf_w, "b": rb - u.rf.reindex(rb.index).fillna(0.0)}, axis=1).dropna()
        ev.boot = paired_bootstrap(df["s"].tolist(), df["b"].tolist(), n_boot=N_BOOT)
        ev.boot_v = paired_bootstrap(df["v"].tolist(), df["b"].tolist(), n_boot=N_BOOT)
    return ev


def evaluate(d: Data, fam: Family, cfg: dict[str, Any], window: tuple[pd.Timestamp, pd.Timestamp], sigma_star: float,
             **kw: Any) -> Eval:
    u = full_universe(d, fam)
    bench = d.spy_ret if kw.get("exec_mode", "close") == "close" else d.open_multi["US"]
    return evaluate_target(fam.target(u, cfg), u, schedule_of(cfg), bench, window, sigma_star,
                           open_ret=open_returns_for(d, fam), **kw)


def no_lookahead_check(d: Data, fam: Family, cfg: dict[str, Any], sr: SearchResult) -> float:
    """B·C·D: 전체 자료로 돌린 탐색 구간 수익 = 잘라서 돌린 수익 (미래 정보가 안 섞였는지)."""
    u_full = full_universe(d, fam)
    u_cut, _, (lo, hi) = search_setup(d, fam)
    a = run_schedule(fam.target(u_full, cfg), u_full.ret[list(fam.target(u_full, cfg).columns)], u_full.rf,
                     schedule_of(cfg)).ret.loc[lo:hi]
    t_cut = fam.target(u_cut, cfg)
    b = run_schedule(t_cut, u_cut.ret[list(t_cut.columns)], u_cut.rf, schedule_of(cfg)).ret.loc[lo:hi]
    return float((a - b).abs().max())


def intl_check(d: Data, fam: Family, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """⑤ 해외 지수: 같은 규칙 vs 매수보유 (가격 지수, 현금 0, 지수별 전체 기간 ~ 2025-09)."""
    out = []
    for t, u in d.intl.items():
        tgt = fam.target(u, cfg)
        sim = run_schedule(tgt, u.ret[["EQ"]], None, schedule_of(cfg))
        start = tgt.dropna(how="all").index.min()
        r = sim.ret.loc[start:]
        b = u.ret["EQ"].loc[start:]
        p, pb = perf(r), perf(b)
        out.append({"지수": ld.INTL[t], "시작": start.date(), "Sharpe": p["Sharpe"], "매수보유_Sharpe": pb["Sharpe"],
                    "CAGR": p["CAGR"], "매수보유_CAGR": pb["CAGR"], "MDD": p["MDD"], "매수보유_MDD": pb["MDD"],
                    "이김": bool(p["Sharpe"] > pb["Sharpe"])})
    return out


# ── F: 결합 ──────────────────────────────────────────────────────────────────────────────────

def sleeve_target(d: Data, code: str, cfg: dict[str, Any], u: Universe) -> pd.DataFrame:
    """군의 목표 비중을 여러 자산 자료(u) 위에서: A·E 는 EQ = 미국(US)."""
    fam = FAMILIES[code]
    if fam.kind == "index":
        ue = Universe(u.ret[["US"]].rename(columns={"US": "EQ"}), u.rf)
        return fam.target(ue, cfg).rename(columns={"EQ": "US"})
    return fam.target(u, cfg)


def combo_target(d: Data, sleeves: dict[str, dict[str, Any]], u: Universe, weighting: str) -> tuple[pd.DataFrame, str]:
    """F 목표 = Σ 군 비중 α × 군 목표(군 결정일 값을 다음 결정일까지 유지). α = 동일 또는 군 수익 역변동성."""
    cols = list(u.ret.columns)
    held, rets = {}, {}
    scheds = []
    for code, cfg in sleeves.items():
        sched = schedule_of(cfg)
        scheds.append(sched)
        t = sleeve_target(d, code, cfg, u).reindex(columns=cols)
        m = decision_mask(t.index, SCHEDULES[sched][0]) & t.notna().any(axis=1).to_numpy()
        held[code] = t.where(np.broadcast_to(m[:, None], t.shape)).ffill()
        rets[code] = run_schedule(t, u.ret[cols], u.rf, sched).ret
    if weighting == "EW":
        alpha = pd.DataFrame(1.0 / len(sleeves), index=u.ret.index, columns=list(sleeves))
    else:
        L = int(weighting[2:])
        vol = pd.DataFrame({c: r.rolling(L).std() for c, r in rets.items()})
        inv = 1.0 / vol.where(vol > 0)
        alpha = inv.div(inv.sum(axis=1), axis=0)
    tgt = sum(held[c].fillna(0.0).mul(alpha[c], axis=0) for c in sleeves)
    ok = pd.concat([held[c].notna().any(axis=1) for c in sleeves], axis=1).all(axis=1) & alpha.notna().all(axis=1)
    tgt = tgt.copy()
    tgt.loc[~ok] = np.nan
    order = SCHEDULE_ORDER + ["D0"]
    fastest = max(scheds, key=order.index)
    return tgt, ("D2" if fastest == "D0" else fastest)


# ── 상태 저장 ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class State:
    searches: dict[str, SearchResult] = field(default_factory=dict)
    valid: dict[str, Eval] = field(default_factory=dict)
    confirm: dict[str, Eval] = field(default_factory=dict)
    stress: dict[str, dict[str, Eval]] = field(default_factory=dict)
    open_exec: dict[str, Eval] = field(default_factory=dict)
    intl: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sched_table: dict[str, pd.DataFrame] = field(default_factory=dict)
    neighbors: dict[str, dict[str, float]] = field(default_factory=dict)
    gates: dict[str, dict[str, bool | None]] = field(default_factory=dict)
    lookahead: dict[str, float] = field(default_factory=dict)
    f: dict[str, Any] = field(default_factory=dict)
    lev: list[dict[str, Any]] = field(default_factory=list)
    started: str = ""
    finished: str = ""


def save_state(st: State) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with STATE.open("wb") as fh:
        pickle.dump(st, fh)


class _StateUnpickler(pickle.Unpickler):
    """스크립트로 실행해 저장한 상태(__main__.State 등)를 모듈 이름으로 읽는다."""

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class("lab_search" if module == "__main__" else module, name)


def load_state() -> State:
    with STATE.open("rb") as fh:
        return _StateUnpickler(fh).load()


# ── 실행 ──────────────────────────────────────────────────────────────────────────────────────

def lev_check(d: Data) -> list[dict[str, Any]]:
    """2배 ETF 실제 수익 vs 가상 2배(2r − 무위험 − 1%/252): 빌린 몫 비용 가정이 맞는지."""
    rf = d.french["rf"]
    out = []
    for lev, base in ld.LEV2X.items():
        a = ld.close_returns(d.prices, lev)
        b = ld.close_returns(d.prices, base)
        x = pd.concat({"a": a, "b": b}, axis=1, sort=True).dropna()
        for lo, label in ((x.index.min(), "상장 이후"), (CONFIRM[0], "2016~2025-09")):
            y = x.loc[max(lo, x.index.min()):]
            syn = synthetic_2x(y["b"], rf)
            ca, cs = perf(y["a"])["CAGR"], perf(syn)["CAGR"]
            # 실제와 같아지는 연 비용(무위험 위) — 이분법
            lo_s, hi_s = -0.05, 0.10
            for _ in range(60):
                mid = (lo_s + hi_s) / 2
                if perf(synthetic_2x(y["b"], rf, mid))["CAGR"] > ca:
                    lo_s = mid
                else:
                    hi_s = mid
            out.append({"2배 ETF": lev, "기초": base, "구간": label, "시작": y.index.min().date(), "실제_CAGR": ca,
                        "가상_CAGR": cs, "차이": ca - cs, "일간 상관": float(y["a"].corr(syn)),
                        "실제와 맞는 비용(무위험 위)": (lo_s + hi_s) / 2})
    return out


def run() -> None:
    reg = registered()
    if not all(reg.values()):
        raise SystemExit("먼저 --register 로 6개 전략군을 등록한다 (결과 계산 전 사전 등록)")
    st = State(started=f"{pd.Timestamp.now():%Y-%m-%d %H:%M}")
    print("===== 자료 읽기 (해시 확인)", flush=True)
    d = load_data()

    print("===== 1단계 탐색 (2007-12-31 까지만)", flush=True)
    for code in FAMILIES:
        st.searches[code] = search_family(d, code)
    save_state(st)

    print("===== 2단계 검증 (2008~2015) · 해외", flush=True)
    for code, sr in st.searches.items():
        fam = FAMILIES[code]
        cfg = sr.grid[sr.chosen]
        if fam.kind == "multi":
            st.lookahead[code] = no_lookahead_check(d, fam, cfg, sr)
            if st.lookahead[code] > 1e-12:
                raise SystemExit(f"{code}: 전체 자료와 자른 자료의 탐색 구간 수익이 다르다 ({st.lookahead[code]:.2e})")
        st.valid[code] = evaluate(d, fam, cfg, VALID, sr.sigma_star, boot=True)
        if fam.kind == "index":
            st.intl[code] = intl_check(d, fam, cfg)
        v = st.valid[code]
        print(f"[lab] {code} 검증: Sharpe {v.p['Sharpe']:.2f} vs SPY {v.pb['Sharpe']:.2f} · 맞춤 CAGR {v.pv['CAGR']:.1%} "
              f"vs {v.pb['CAGR']:.1%} · MDD {v.pv['MDD']:.1%} vs {v.pb['MDD']:.1%} → {'통과' if v.gate else '탈락'}", flush=True)
    save_state(st)

    print("===== 3단계 재확인 (2016-01 ~ 2025-09) · 스트레스 · 이웃 · 주기 표", flush=True)
    for code, sr in st.searches.items():
        fam = FAMILIES[code]
        cfg = sr.grid[sr.chosen]
        st.confirm[code] = evaluate(d, fam, cfg, CONFIRM, sr.sigma_star, boot=True)
        st.stress[code] = {"비용 2배": evaluate(d, fam, cfg, CONFIRM, sr.sigma_star, cost=2 * COST_PER_SIDE),
                           "하루 지연": evaluate(d, fam, cfg, CONFIRM, sr.sigma_star, delay=1)}
        st.open_exec[code] = evaluate(d, fam, cfg, CONFIRM, sr.sigma_star, exec_mode="open")
        keys = [fam.key(c) for c in sr.grid]
        nb = [keys.index(k) for k in fam.neighbors(cfg, set(keys))]
        nsh = [evaluate(d, fam, sr.grid[j], CONFIRM, sr.sigma_star).p["Sharpe"] for j in nb]
        st.neighbors[code] = {"이웃": len(nsh), "재확인_Sharpe_중앙": float(np.median(nsh)) if nsh else float("nan"),
                              "SPY보다_높은_비율": float(np.mean([x > st.confirm[code].pb["Sharpe"] for x in nsh]))
                              if nsh else float("nan")}
        rows = []
        if "sched" in cfg:
            for s in SCHEDULE_ORDER:
                c2 = {**cfg, "sched": s}
                j = keys.index(fam.key(c2))
                ev_v = evaluate(d, fam, c2, VALID, sr.sigma_star)
                ev_c = evaluate(d, fam, c2, CONFIRM, sr.sigma_star)
                rows.append({"주기": SCHEDULE_KO[s], "탐색 Sharpe": sr.table.loc[j, "Sharpe"],
                             "탐색 매매/년": sr.table.loc[j, "매매_연"], "검증 Sharpe": ev_v.p["Sharpe"],
                             "재확인 Sharpe": ev_c.p["Sharpe"], "재확인 맞춤 CAGR": ev_c.pv["CAGR"],
                             "재확인 매매/년": ev_c.trades_yr, "선택": "◀" if s == cfg["sched"] else ""})
        st.sched_table[code] = pd.DataFrame(rows)
        c = st.confirm[code]
        print(f"[lab] {code} 재확인: Sharpe {c.p['Sharpe']:.2f} vs SPY {c.pb['Sharpe']:.2f} · 맞춤 CAGR {c.pv['CAGR']:.1%} "
              f"vs {c.pb['CAGR']:.1%} · MDD {c.pv['MDD']:.1%} vs {c.pb['MDD']:.1%}", flush=True)

    for code, sr in st.searches.items():
        g5 = None
        if FAMILIES[code].kind == "index":
            g5 = sum(x["이김"] for x in st.intl[code]) >= INTL_MIN
        st.gates[code] = {"g1": bool(sr.pbo["PBO"] < PBO_MAX), "g2": bool(sr.n_neighbors >= 2 and sr.neighbor_share >= NEIGHBOR_MIN),
                          "g3": st.valid[code].gate, "g4": st.confirm[code].gate, "g5": g5,
                          "g6": all(e.gate for e in st.stress[code].values())}
    save_state(st)

    print("===== F 결합", flush=True)
    entry = [c for c in st.searches if all(st.gates[c][g] is not False for g in ("g1", "g2", "g3", "g5"))]
    st.f = {"들어간 군": entry}
    if len(entry) >= 2:
        sleeves = {c: st.searches[c].grid[st.searches[c].chosen] for c in entry}
        lo, hi = SEARCH["multi"]
        u_cut = d.multi.truncate(hi)
        sig = float(perf(u_cut.ret["US"].loc[lo:hi], u_cut.rf.loc[lo:hi])["vol"])
        rows = []
        for wgt in F_WEIGHTINGS:
            tgt, sched = combo_target(d, sleeves, u_cut, wgt)
            r = run_schedule(tgt, u_cut.ret, u_cut.rf, sched).ret.loc[lo:hi]
            p = perf(r, u_cut.rf.loc[lo:hi])
            rows.append({"비중": wgt, **p, "score": score(p), "주기": sched})
        ft = pd.DataFrame(rows)
        best = ft.loc[ft["score"].idxmax()]
        tgt_full, sched = combo_target(d, sleeves, d.multi, str(best["비중"]))
        evk = dict(open_ret=d.open_multi)
        st.f.update({"탐색표": ft, "비중": str(best["비중"]), "주기": sched, "sigma_star": sig,
                     "검증": evaluate_target(tgt_full, d.multi, sched, d.spy_ret, VALID, sig, boot=True, **evk),
                     "재확인": evaluate_target(tgt_full, d.multi, sched, d.spy_ret, CONFIRM, sig, boot=True, **evk),
                     "스트레스": {"비용 2배": evaluate_target(tgt_full, d.multi, sched, d.spy_ret, CONFIRM, sig,
                                                       cost=2 * COST_PER_SIDE, **evk),
                               "하루 지연": evaluate_target(tgt_full, d.multi, sched, d.spy_ret, CONFIRM, sig, delay=1, **evk)},
                     "시가": evaluate_target(tgt_full, d.multi, sched, d.open_multi["US"], CONFIRM, sig, exec_mode="open",
                                           **evk)})
        st.f["통과"] = bool(st.f["재확인"].gate and all(e.gate for e in st.f["스트레스"].values()))
        print(f"[lab] F ({'+'.join(entry)}, {st.f['비중']}): 재확인 Sharpe {st.f['재확인'].p['Sharpe']:.2f} → "
              f"{'통과' if st.f['통과'] else '탈락'}", flush=True)
    else:
        print(f"[lab] F 없음 — 검증까지 통과한 군 {len(entry)}개", flush=True)

    print("===== 2배 ETF 비용 대조", flush=True)
    st.lev = lev_check(d)
    st.finished = f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"
    save_state(st)
    if results_logged():
        print("결과 행이 이미 TRIAL_LOG 에 있다 — 다시 쓰지 않는다", flush=True)
    else:
        write_trial_rows(st)
    export_optuna(st)
    write_outputs(st)


def final_label(st: State, code: str) -> str:
    g = st.gates[code]
    return "실전 후보" if all(v is not False for v in g.values()) else "탈락"


def write_trial_rows(st: State) -> None:
    for code, sr in st.searches.items():
        fam = FAMILIES[code]
        cfg = sr.grid[sr.chosen]
        g = st.gates[code]
        v, c = st.valid[code], st.confirm[code]
        rows = [
            {"test_id": f"{code}-탐색", "판정": "통과" if g["g1"] and g["g2"] else "탈락",
             "메모": (f"선택: {fam.describe(cfg)} · 탐색 Sharpe {sr.table.loc[sr.chosen, 'Sharpe']:.2f} vs 벤치 "
                      f"{sr.bench['Sharpe']:.2f} · PBO {sr.pbo['PBO']:.2f} · 이웃 {sr.neighbor_share:.0%}({sr.n_neighbors}) · "
                      f"DSR {sr.dsr:.2f}")},
            {"test_id": f"{code}-검증", "판정": "통과" if g["g3"] else "탈락", "메모": _eval_memo(v)},
            {"test_id": f"{code}-재확인", "판정": "통과" if g["g4"] else "탈락", "메모": _eval_memo(c)},
            {"test_id": f"{code}-스트레스", "판정": "통과" if g["g6"] else "탈락",
             "메모": " · ".join(f"{k}: Sharpe {e.p['Sharpe']:.2f}, 맞춤 CAGR {e.pv['CAGR']:+.1%}, MDD {e.pv['MDD']:.1%}"
                               for k, e in st.stress[code].items())},
        ]
        if g["g5"] is not None:
            wins = sum(x["이김"] for x in st.intl[code])
            rows.append({"test_id": f"{code}-해외", "판정": "통과" if g["g5"] else "탈락",
                         "메모": f"{wins}/6 이 매수보유보다 Sharpe 높음 (기준 4+)"})
        rows.append({"test_id": f"{code}-최종", "판정": final_label(st, code),
                     "메모": " · ".join(f"{GATE_KO[k]} {'O' if x else 'X'}" for k, x in g.items() if x is not None)})
        log_trials(STEP, FAMILY_IDS[code], len(sr.grid), rows)
    f = st.f
    if "재확인" in f:
        log_trials(STEP, FAMILY_IDS["F"], len(F_WEIGHTINGS), [
            {"test_id": "F-재확인", "판정": "통과" if f["재확인"].gate else "탈락",
             "메모": f"{'+'.join(f['들어간 군'])} · {f['비중']} · " + _eval_memo(f["재확인"])},
            {"test_id": "F-최종", "판정": "실전 후보" if f["통과"] else "탈락",
             "메모": " · ".join(f"{k}: Sharpe {e.p['Sharpe']:.2f}, 맞춤 CAGR {e.pv['CAGR']:+.1%}"
                               for k, e in f["스트레스"].items())}])
    else:
        log_trials(STEP, FAMILY_IDS["F"], len(F_WEIGHTINGS), [
            {"test_id": "F-최종", "판정": "해당 없음", "메모": f"검증까지 통과한 군 {len(f['들어간 군'])}개 (2개 이상 필요)"}])


def _eval_memo(e: Eval) -> str:
    b = e.boot or {}
    s = (f"Sharpe {e.p['Sharpe']:.2f} vs SPY {e.pb['Sharpe']:.2f}"
         + (f", ΔSharpe {b['ΔSharpe']:+.2f} [{b['ΔSharpe_하한']:+.2f}, {b['ΔSharpe_상한']:+.2f}] p {b['p_개선아님']:.3f}"
            if b else "")
         + f" · 맞춤(평균 {e.lev_mean:.2f}배) CAGR {e.pv['CAGR']:+.1%} vs {e.pb['CAGR']:+.1%}, MDD {e.pv['MDD']:.1%} vs "
           f"{e.pb['MDD']:.1%}")
    return s


def export_optuna(st: State) -> None:
    """탐색 조합 전부를 Optuna 저장소로 (optuna-dashboard 로 보기). 값 = 탐색 구간 옵션 A 점수 (표본 내)."""
    import optuna
    from optuna.distributions import CategoricalDistribution as Cat
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_DB.unlink(missing_ok=True)
    storage = f"sqlite:///{OPTUNA_DB}"
    for code, sr in st.searches.items():
        names = list(sr.grid[0])
        dist = {n: Cat(sorted({str(c[n]) for c in sr.grid})) for n in names}
        study = optuna.create_study(study_name=f"lab_{code}", storage=storage, direction="maximize")
        study.set_user_attr("설명", f"{FAMILIES[code].title} · 값 = 탐색 구간 옵션 A 점수 (표본 내라 낙관적, 판정은 보고서)")
        trials = []
        for i, c in enumerate(sr.grid):
            row = sr.table.iloc[i]
            val = float(row["score"])
            ok = math.isfinite(val)
            attrs = {"설명": row["설명"], "Sharpe": float(row["Sharpe"]), "CAGR": float(row["CAGR"]),
                     "MDD": float(row["MDD"]), "이웃평균": float(row["smooth"]) if math.isfinite(row["smooth"]) else None,
                     "선택": row["선택"]}
            trials.append(optuna.trial.create_trial(
                params={n: str(c[n]) for n in names}, distributions=dist, value=val if ok else None,
                state=optuna.trial.TrialState.COMPLETE if ok else optuna.trial.TrialState.FAIL, user_attrs=attrs))
        study.add_trials(trials)
    print(f"[lab] Optuna 저장소: {OPTUNA_DB} (optuna-dashboard {storage})", flush=True)


# ── 보고서 ───────────────────────────────────────────────────────────────────────────────────

def _f(v: Any, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return format(v, fmt)


def _ox(v: bool | None) -> str:
    return "—" if v is None else ("O" if v else "X")


def results_frame(st: State) -> pd.DataFrame:
    rows = []
    for code, sr in st.searches.items():
        fam = FAMILIES[code]
        cfg = sr.grid[sr.chosen]
        for stage, e in (("검증", st.valid[code]), ("재확인", st.confirm[code]),
                         *((f"재확인·{k}", e) for k, e in st.stress[code].items()), ("재확인·시가 체결", st.open_exec[code])):
            b = e.boot or {}
            rows.append({"군": code, "이름": fam.title, "선택": fam.describe(cfg), "단계": stage,
                         "Sharpe": e.p["Sharpe"], "CAGR": e.p["CAGR"], "MDD": e.p["MDD"], "노출": e.exposure,
                         "맞춤_Sharpe": e.pv["Sharpe"], "맞춤_CAGR": e.pv["CAGR"], "맞춤_MDD": e.pv["MDD"],
                         "맞춤_평균배율": e.lev_mean, "맞춤_최대배율": e.lev_max, "SPY_Sharpe": e.pb["Sharpe"],
                         "SPY_CAGR": e.pb["CAGR"], "SPY_MDD": e.pb["MDD"], "ΔSharpe": b.get("ΔSharpe"),
                         "하한": b.get("ΔSharpe_하한"), "상한": b.get("ΔSharpe_상한"), "p": b.get("p_개선아님"),
                         "통과": e.gate, "판정": final_label(st, code)})
    f = st.f
    if "재확인" in f:
        for stage, e in (("검증(참고)", f["검증"]), ("재확인", f["재확인"]),
                         *((f"재확인·{k}", e) for k, e in f["스트레스"].items()), ("재확인·시가 체결", f["시가"])):
            b = e.boot or {}
            rows.append({"군": "F", "이름": "결합", "선택": f"{'+'.join(f['들어간 군'])} · {f['비중']}", "단계": stage,
                         "Sharpe": e.p["Sharpe"], "CAGR": e.p["CAGR"], "MDD": e.p["MDD"], "노출": e.exposure,
                         "맞춤_Sharpe": e.pv["Sharpe"], "맞춤_CAGR": e.pv["CAGR"], "맞춤_MDD": e.pv["MDD"],
                         "맞춤_평균배율": e.lev_mean, "맞춤_최대배율": e.lev_max, "SPY_Sharpe": e.pb["Sharpe"],
                         "SPY_CAGR": e.pb["CAGR"], "SPY_MDD": e.pb["MDD"], "ΔSharpe": b.get("ΔSharpe"),
                         "하한": b.get("ΔSharpe_하한"), "상한": b.get("ΔSharpe_상한"), "p": b.get("p_개선아님"),
                         "통과": e.gate, "판정": "실전 후보" if f["통과"] else "탈락"})
    return pd.DataFrame(rows)


def yearly(st: State) -> pd.DataFrame:
    cols = {}
    spy = None
    for code in st.searches:
        r = pd.concat([st.valid[code].rv, st.confirm[code].rv]).sort_index()
        cols[f"{code} 맞춤"] = (1 + r).groupby(r.index.year).prod() - 1
        if spy is None:
            b = pd.concat([st.valid[code].rb, st.confirm[code].rb]).sort_index()
            spy = (1 + b).groupby(b.index.year).prod() - 1
    if "재확인" in st.f:
        r = pd.concat([st.f["검증"].rv, st.f["재확인"].rv]).sort_index()
        cols["F 맞춤"] = (1 + r).groupby(r.index.year).prod() - 1
    out = pd.DataFrame(cols)
    out["SPY"] = spy
    return out


def write_outputs(st: State) -> None:
    trials = pd.concat([sr.table for sr in st.searches.values()], ignore_index=True)
    trials.to_csv(TRIALS_CSV, index=False)
    res = results_frame(st)
    res.to_csv(RESULTS_CSV, index=False)
    REPORT.write_text(render_report(st, res), encoding="utf-8")
    print(f"[lab] 보고서: {REPORT.relative_to(TIER3_DIR.parents[2])}", flush=True)


def notes(st: State) -> list[str]:
    """맞춤 목표 변동성·SPY 숫자·주기 요약 (보고서 머리)."""
    any_code = next(iter(st.searches))
    sig = {c: sr.sigma_star for c, sr in st.searches.items()}
    sig_txt = " · ".join(f"{c} {v:.1%}" for c, v in sig.items())
    v_vol, c_vol = st.valid[any_code].pb["vol"], st.confirm[any_code].pb["vol"]
    sigma_free = []
    for c in st.searches:                         # σ* 와 무관한 조건(PBO·Sharpe)만으로도 떨어졌는지
        g = st.gates[c]
        sharpe_fail = (st.valid[c].p["Sharpe"] <= st.valid[c].pb["Sharpe"]
                       or st.confirm[c].p["Sharpe"] <= st.confirm[c].pb["Sharpe"])
        sigma_free.append(bool(not g["g1"] or sharpe_fail or g["g5"] is False))
    out = [f"- 변동성 맞춤 목표 σ* (탐색 구간 벤치마크 변동성): {sig_txt}. 같은 기간 SPY 실제 변동성은 검증 {v_vol:.0%}, "
           f"재확인 {c_vol:.0%} 라 맞춤판은 SPY 보다 낮은 위험에서 비교됐다."
           + (" 다만 모든 군이 σ* 와 상관없는 조건(PBO·Sharpe·해외)에서도 떨어져 결론은 같다." if all(sigma_free) else ""),
           f"- 참고: 재확인 구간 SPY Sharpe — 이 보고서의 비교 기준(초과수익·종가→종가) {st.confirm[any_code].pb['Sharpe']:.2f}, "
           f"무위험을 안 뺀 종가→종가 {st.confirm[any_code].pb['Sharpe_raw']:.2f}, 무위험을 안 뺀 시가→시가 "
           f"{st.open_exec[any_code].pb['Sharpe_raw']:.2f} (지금까지 보고서들의 SPY 0.89 와 같은 방식).",
           "- DSR 은 참고: 같은 군 안 조합들이 서로 비슷해 Sharpe 분산이 작으면 1 에 가깝게 나온다. 고르기의 운은 PBO 로 본다.",
           "- 맞춤판 최대 배율이 1.5 를 조금 넘는 날은 다음 리밸런싱 전 가격 변동 때문이다."]
    summ = []
    for c, tab in st.sched_table.items():
        if tab.empty:
            continue
        best = tab.loc[tab["재확인 Sharpe"].idxmax()]
        mon = tab.loc[tab["주기"] == SCHEDULE_KO["M"], "재확인 Sharpe"].iloc[0]
        summ.append(f"{c} 최고 {best['주기']} {best['재확인 Sharpe']:.2f} (월 1회 {mon:.2f})")
    if summ:
        out.append("- 매매 주기 (재확인 Sharpe): " + " · ".join(summ))
    return out + [""]


def render_report(st: State, res: pd.DataFrame) -> str:
    cand = [c for c in st.searches if final_label(st, c) == "실전 후보"]
    f_ok = bool(st.f.get("통과"))
    if f_ok:
        concl = (f"**결론: 결합 F({'+'.join(st.f['들어간 군'])})가 모든 기준을 통과했다 — 실전 페이퍼 계좌 후보.**"
                 + (f" 단독 통과 군: {', '.join(cand)}." if cand else ""))
    elif cand:
        concl = f"**결론: {', '.join(cand)}군이 모든 기준을 통과했다 — 실전 페이퍼 계좌 후보.**"
    else:
        concl = ("**결론: 사전 등록한 기준을 모두 통과한 전략군은 없다.** 지금 자료로는 SPY 를 같은 위험에서 이기는 규칙을 "
                 "확인하지 못했다 — SPY(또는 변동성 조절 SPY) 보유가 기본값이다.")
    L: list[str] = [f"# 연구소 1~3단계 (lab_{VERSION}) — 전략군 A~F 탐색·검증·재확인", "", concl, ""]
    L += [f"- 사전 등록: TRIAL_LOG family `lab_A~F_{VERSION}` · 누적 시도 {total_trials():,}건 · 실행 {st.started} ~ {st.finished}",
          "- 계획: 탐색(1990~2007, 지수 규칙은 1950~) → 검증(2008~2015, 군당 1회) → 재확인(2016-01~2025-09) → "
          "통과하면 실전 페이퍼 계좌. 홀드아웃(2025-10~)은 쓰지 않았다.",
          "- Sharpe = 3개월 국채(Kenneth French 1개월) 초과수익 기준. 비용 편도 0.1%. 신호 t 종가 → t+1 종가 체결.",
          f"- 변동성 맞춤판 = 사용자 목표(같은 위험에서 더 높은 수익): 탐색 구간 벤치마크 변동성(σ*)에 맞춰 최대 "
          f"{LEV_CAP}배, 빌린 몫은 무위험 + {BORROW_SPREAD:.0%}.",
          "- 통과 기준(모두): ① PBO < 0.3 ② 이웃 80%+ 가 벤치마크보다 Sharpe 높음 ③ 검증 ④ 재확인 (Sharpe > SPY, "
          "맞춤판 CAGR > SPY·MDD ≥ SPY) ⑤ 해외 6개 중 4+ (A·E) ⑥ 비용 2배·하루 지연에도 ④.", ""]
    L += notes(st)

    L += ["## 판정", "", "| 군 | 선택된 규칙 | " + " | ".join(GATE_KO.values()) + " | 판정 |", "|---|---|" + "---|" * 7]
    for code, sr in st.searches.items():
        g = st.gates[code]
        L.append(f"| {code} {FAMILIES[code].title} | {FAMILIES[code].describe(sr.grid[sr.chosen])} | "
                 + " | ".join(_ox(g[k]) for k in GATE_KO) + f" | **{final_label(st, code)}** |")
    if "재확인" in st.f:
        f = st.f
        L.append(f"| F 결합 | {'+'.join(f['들어간 군'])} · {f['비중']} · {SCHEDULE_KO.get(f['주기'], f['주기'])} | — | — | 참고 | "
                 f"{_ox(f['재확인'].gate)} | — | {_ox(all(e.gate for e in f['스트레스'].values()))} | "
                 f"**{'실전 후보' if f_ok else '탈락'}** |")
    else:
        L.append(f"| F 결합 | 검증까지 통과한 군 {len(st.f.get('들어간 군', []))}개 → 만들지 않음 | | | | | | | 해당 없음 |")
    L.append("")

    L += ["## 1단계 탐색 (표본 내)", "", "| 군 | 조합 | 벤치마크 Sharpe | 선택 Sharpe | CAGR | MDD | 노출 | 매매/년 | PBO | 이웃 > 벤치 | DSR |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for code, sr in st.searches.items():
        t = sr.table.loc[sr.chosen]
        L.append(f"| {code} | {len(sr.grid)} | {sr.bench['Sharpe']:.2f} | {t['Sharpe']:.2f} | {t['CAGR']:+.1%} | {t['MDD']:.1%} | "
                 f"{t['노출']:.0%} | {t['매매_연']:.1f} | {sr.pbo['PBO']:.2f} | {_f(sr.neighbor_share, '.0%')} ({sr.n_neighbors}) | "
                 f"{_f(sr.dsr, '.2f')} |")
    L += ["", "- 벤치마크: A·E = 미국 시장 총수익(1950~2007), B·C·D = 미국 주식 대용→SPY(1990~2007). 탐색 숫자는 고른 구간이라 낙관적이다.",
          "- 군별 이웃 평균 상위 5개:", ""]
    for code, sr in st.searches.items():
        top = sr.table.sort_values("smooth", ascending=False).head(5)
        L.append(f"  - **{code}** " + " / ".join(f"{r['설명']} ({r['Sharpe']:.2f})" for _, r in top.iterrows()))
    L.append("")

    L += ["## 2·3단계 검증·재확인 (SPY 대비)", "",
          "| 군 | 구간 | Sharpe | SPY | ΔSharpe [95%] · p | CAGR | MDD | 맞춤 배율(평균/최대) | 맞춤 CAGR | 맞춤 MDD | SPY CAGR | SPY MDD | 통과 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res[res["단계"].isin(["검증", "재확인", "검증(참고)"])].iterrows():
        ci = (f"{r['ΔSharpe']:+.2f} [{r['하한']:+.2f}, {r['상한']:+.2f}] · {r['p']:.3f}" if pd.notna(r["ΔSharpe"]) else "—")
        L.append(f"| {r['군']} | {r['단계']} | {r['Sharpe']:.2f} | {r['SPY_Sharpe']:.2f} | {ci} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | "
                 f"{r['맞춤_평균배율']:.2f}/{r['맞춤_최대배율']:.2f} | {r['맞춤_CAGR']:+.1%} | {r['맞춤_MDD']:.1%} | {r['SPY_CAGR']:+.1%} | "
                 f"{r['SPY_MDD']:.1%} | {'O' if r['통과'] else 'X'} |")
    L.append("")

    L += ["## 매매 주기 비교 (월·주·매일)", "", "선택된 규칙의 다른 손잡이는 그대로 두고 주기만 바꿨다. 판정은 선택된 주기로만 한다.", ""]
    for code, tab in st.sched_table.items():
        if tab.empty:
            continue
        L += [f"**{code} {FAMILIES[code].title}**", "", "| 주기 | 탐색 Sharpe | 탐색 매매/년 | 검증 Sharpe | 재확인 Sharpe | 재확인 맞춤 CAGR | 재확인 매매/년 | |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in tab.iterrows():
            L.append(f"| {r['주기']} | {r['탐색 Sharpe']:.2f} | {r['탐색 매매/년']:.1f} | {r['검증 Sharpe']:.2f} | {r['재확인 Sharpe']:.2f} | "
                     f"{r['재확인 맞춤 CAGR']:+.1%} | {r['재확인 매매/년']:.1f} | {r['선택']} |")
        L.append("")

    L += ["## ⑤ 해외 지수 (A·E)", ""]
    for code, rows in st.intl.items():
        L += [f"**{code}** — {sum(x['이김'] for x in rows)}/6", "", "| 지수 | 시작 | Sharpe | 매수보유 | CAGR | 매수보유 | MDD | 매수보유 |",
              "|---|---|---|---|---|---|---|---|"]
        for x in rows:
            L.append(f"| {x['지수']} | {x['시작']} | {x['Sharpe']:.2f} | {x['매수보유_Sharpe']:.2f} | {x['CAGR']:+.1%} | "
                     f"{x['매수보유_CAGR']:+.1%} | {x['MDD']:.1%} | {x['매수보유_MDD']:.1%} |")
        L.append("")

    L += ["## ⑥ 스트레스 · 시가 체결 · 이웃 (재확인 구간)", "",
          "| 군 | 경우 | Sharpe | SPY | 맞춤 CAGR | 맞춤 MDD | 통과 |", "|---|---|---|---|---|---|---|"]
    for _, r in res[res["단계"].str.startswith("재확인·")].iterrows():
        L.append(f"| {r['군']} | {r['단계'].replace('재확인·', '')} | {r['Sharpe']:.2f} | {r['SPY_Sharpe']:.2f} | {r['맞춤_CAGR']:+.1%} | "
                 f"{r['맞춤_MDD']:.1%} | {'O' if r['통과'] else 'X'} |")
    L += ["", "- 시가 체결은 참고(판정 없음): 다음 거래일 시가에 사고팔고, SPY 도 시가→시가로 비교.", ""]
    L += ["| 군 | 이웃 수 | 이웃 재확인 Sharpe 중앙값 | SPY 보다 높은 비율 |", "|---|---|---|---|"]
    for code, nb in st.neighbors.items():
        L.append(f"| {code} | {nb['이웃']} | {_f(nb['재확인_Sharpe_중앙'], '.2f')} | {_f(nb['SPY보다_높은_비율'], '.0%')} |")
    L.append("")

    if "탐색표" in st.f:
        L += ["## F 결합", "", f"- 들어간 군: {', '.join(st.f['들어간 군'])} · 매매 주기 {SCHEDULE_KO.get(st.f['주기'], st.f['주기'])} · "
              f"σ* {st.f['sigma_star']:.1%}", "", "| 비중 | 탐색 Sharpe | CAGR | MDD | 점수 |", "|---|---|---|---|---|"]
        for _, r in st.f["탐색표"].iterrows():
            L.append(f"| {r['비중']}{' ◀' if r['비중'] == st.f['비중'] else ''} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | {r['score']:.2f} |")
        L.append("")

    L += ["## 연도별 수익 (변동성 맞춤판, 2008~2025-09)", ""]
    yt = yearly(st)
    L += ["| 연도 | " + " | ".join(yt.columns) + " |", "|---|" + "---|" * len(yt.columns)]
    for y, row in yt.iterrows():
        L.append(f"| {y} | " + " | ".join(_f(v, "+.1%") for v in row.values) + " |")
    L.append("")

    L += ["## 2배 ETF 비용 대조 (레버리지 가정 확인)", "",
          "| 2배 ETF | 구간 | 실제 CAGR | 가상 2배 CAGR | 차이 | 일간 상관 | 실제와 맞는 비용(무위험 위) |", "|---|---|---|---|---|---|---|"]
    for x in st.lev:
        L.append(f"| {x['2배 ETF']}({x['기초']}) | {x['구간']} ({x['시작']}~) | {x['실제_CAGR']:+.1%} | {x['가상_CAGR']:+.1%} | "
                 f"{x['차이']:+.1%} | {x['일간 상관']:.3f} | {x['실제와 맞는 비용(무위험 위)']:.1%} |")
    L += ["", f"- 백테스트 가정은 빌린 몫에 무위험 + {BORROW_SPREAD:.0%}. 실제와 맞는 비용이 이보다 크면 레버리지 결과는 그만큼 낙관적이다.", ""]

    cov = pd.read_csv(ld.COVERAGE) if ld.COVERAGE.exists() else None
    if cov is not None:
        L += ["## 자료", "", "| 자산 | 이름 | 시작 | 출처 | 겹친 구간 상관 | 연수익 차(대용−ETF) |", "|---|---|---|---|---|---|"]
        for _, r in cov.iterrows():
            L.append(f"| {r['자산']} | {r['이름']} | {r['시작']} | {r['출처']} | {_f(r.get('겹친 구간 상관'), '.3f')} | "
                     f"{_f(r.get('겹친 구간 연수익 차(대용−ETF)'), '+.2%')} |")
        L.append("")

    L += ["## 한계", "",
          "- 탐색 구간의 해외·채권·리츠·금 시세는 뱅가드 인덱스 펀드·금 선물 대용이다 (ETF 와 일간 상관 0.88~0.99). "
          "금은 2000-08, 원자재는 2006-02 부터라 탐색 구간 대부분은 그 자산 없이 배분했다.",
          "- 해외 지수는 배당 빠진 가격 지수이고 현금 수익을 0 으로 뒀다 (매수보유도 같은 조건).",
          "- 검증 구간(2008~2015)은 F 를 만들 군을 고르는 데도 썼다. F 의 독립 확인은 재확인 구간뿐이다.",
          "- 재확인 구간(2016~2025)은 지금까지 개별주식 연구에 많이 쓰였다. 자산배분 규칙으로는 처음이지만, 시장 흐름은 이미 알려져 있다.",
          "- 레버리지는 매일 다시 맞추는 2배 ETF 를 무위험 + 1% 비용으로 근사했다. 위의 2배 ETF 대조를 함께 본다.",
          "- 결과가 좋아도 실전 반영(새 페이퍼 계좌, src 변경·push)은 사용자 결정이다.", ""]
    return "\n".join(L)


def report_only() -> None:
    write_outputs(load_state())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.register:
        register()
    elif a.run:
        run()
    else:
        report_only()


if __name__ == "__main__":
    main()
