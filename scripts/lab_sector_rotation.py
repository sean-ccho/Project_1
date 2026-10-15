#!/usr/bin/env python3
"""연구소 G군 — 경기순환주·방어주 전환 (lab_G_v1): 시장이 좋으면 순환주(또는 시장 전체), 나쁘면 방어주(또는 현금).

계기: 사용자 — 섹터 분류(경기순환주·경기방어주)로 그때그때 맞게 사면 도움이 될까 (2026-10-14).

사전 등록 (TRIAL_LOG family lab_G_v1 — 결과 계산 전에 --register):
- 자료: Kenneth French 12 산업 일간 (시가총액 가중, 1926~). 순환주 8 = 내구재·제조업·에너지·화학·IT장비·유통·금융·기타
  (건설·운송·호텔 등), 방어주 4 = 필수소비재·통신·유틸리티·헬스케어. 바구니 안은 산업 동일 비중. 시장 = 미국 시장 총수익.
- 탐색 1950-01 ~ 2007-12 → 검증 2008-01 ~ 2015-12 (1회) → 재확인 2016-01 ~ 2025-09. 벤치마크 = 미국 시장 총수익.
- 격자 180 = 추세 판단 {시장 총수익지수 > SMA 100·150·200·250, 시장 초과 모멘텀 > 0 126·252일,
  순환주 − 방어주 수익 > 0 63·126·252일} × 상승일 때 {순환주, 시장} × 하락일 때 {방어주, 현금} × 주기 {월, 주, 매일 폭 10·5·2%p}.
- 선택·통과 기준은 lab_v1 과 같다 (옵션 A 자기+이웃 평균 최고, 주기는 Sharpe 0.05 이내면 느린 쪽 · ① PBO < 0.3
  ② 이웃 80%+ > 벤치마크 ③ 검증 ④ 재확인: 초과 Sharpe > 벤치마크, 변동성 맞춤판(σ* = 탐색 벤치마크 변동성, ≤1.5배,
  빌린 몫 무위험+1%) CAGR > 벤치마크·MDD ≥ 벤치마크 ⑥ 재확인 비용 2배·하루 지연에도 ④). 신호 t 종가 → t+1 종가 체결, 편도 0.1%.
- 참고(판정 없음): 같은 신호의 상승·하락 조합 4가지 비교(방어주 vs 현금), 실제 섹터 ETF(순환 XLY·XLF·XLI·XLB·XLE·XLK,
  방어 XLP·XLV·XLU, 시장 SPY)로 2016~2025 다시 계산.

사용:
  .venv/bin/python scripts/lab_data.py --build-industries
  .venv/bin/python scripts/lab_sector_rotation.py --register
  .venv/bin/python scripts/lab_sector_rotation.py --run
산출: tier3/LAB_SECTOR_ROTATION.md · tier3/lab_sector_rotation.csv
"""

from __future__ import annotations

import argparse
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
import lab_search as ls  # noqa: E402
from lab_backtest import SCHEDULE_KO, SCHEDULE_ORDER, TRADING_DAYS, perf, run_schedule, score  # noqa: E402
from lab_families import Family, Universe, schedule_of, target_key  # noqa: E402
from paper_trading.evaluation import deflated_sharpe  # noqa: E402
from research_utils import (COST_PER_SIDE, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv, log_trials,  # noqa: E402
                            pbo_cscv, total_trials)

FAMILY = "lab_G_v1"
STEP = "stage18"
SEARCH = (pd.Timestamp("1950-01-03"), pd.Timestamp("2007-12-31"))
VALID, CONFIRM = ls.VALID, ls.CONFIRM
START = pd.Timestamp("1948-01-02")
CYC = ["Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Shops", "Money", "Other"]
DEF = ["NoDur", "Telcm", "Utils", "Hlth"]
IND_KO = {"NoDur": "필수소비재", "Durbl": "내구재", "Manuf": "제조업", "Enrgy": "에너지", "Chems": "화학", "BusEq": "IT장비",
          "Telcm": "통신", "Utils": "유틸리티", "Shops": "유통", "Hlth": "헬스케어", "Money": "금융", "Other": "기타"}
CYC_ETF = ["XLY", "XLF", "XLI", "XLB", "XLE", "XLK"]
DEF_ETF = ["XLP", "XLV", "XLU"]
SIG_N = {"sma": [100, 150, 200, 250], "mom": [126, 252], "rel": [63, 126, 252]}
ON_KO = {"CYC": "순환주", "MKT": "시장 전체"}
OFF_KO = {"DEF": "방어주", "CASH": "현금"}
STATE = ld.LAB_DIR / "sector_rotation_v1.pkl"
REPORT = TIER3_DIR / "LAB_SECTOR_ROTATION.md"
OUT_CSV = TIER3_DIR / "lab_sector_rotation.csv"


# ── 전략군 G ───────────────────────────────────────────────────────────────────────────────────

def grid_g() -> list[dict[str, Any]]:
    return [{"sig": k, "n": n, "on": on, "off": off, "sched": s} for k, ns in SIG_N.items() for n in ns
            for on in ("CYC", "MKT") for off in ("DEF", "CASH") for s in SCHEDULE_ORDER]


def ladders_g(cfg: dict[str, Any]) -> dict[str, list[Any]]:
    return {"n": SIG_N[cfg["sig"]], "sched": SCHEDULE_ORDER}


def regime(u: Universe, kind: str, n: int, cyc: list[str], dfn: list[str], mkt: str) -> pd.Series:
    """상승(True)·하락(False): sma = 시장 총수익지수 > n일 평균, mom = 시장 n일 초과 모멘텀 > 0,
    rel = 순환주 바구니 n일 수익 − 방어주 바구니 n일 수익 > 0 (바구니 = 산업 동일 비중, 매일 다시 맞춤)."""
    if kind == "sma":
        return u.tri[mkt] > u.sma(n)[mkt]
    if kind == "mom":
        return u.mom(n)[mkt] > 0
    key = ("baskets", tuple(cyc), tuple(dfn))
    if key not in u._cache:
        c = (1.0 + u.ret[cyc].mean(axis=1, skipna=False)).cumprod()
        d = (1.0 + u.ret[dfn].mean(axis=1, skipna=False)).cumprod()
        u._cache[key] = (c, d)
    c, d = u._cache[key]
    return (c / c.shift(n) - d / d.shift(n)) > 0


def make_target(cyc: list[str], dfn: list[str], mkt: str):
    """자료의 열 이름(산업 또는 ETF)에 맞춘 목표 비중 함수."""
    cols = cyc + dfn + [mkt]

    def target(u: Universe, cfg: dict[str, Any]) -> pd.DataFrame:
        on = regime(u, cfg["sig"], int(cfg["n"]), cyc, dfn, mkt)
        w = pd.DataFrame(0.0, index=u.ret.index, columns=cols)
        if cfg["on"] == "CYC":
            w.loc[on, cyc] = 1.0 / len(cyc)
        else:
            w.loc[on, mkt] = 1.0
        if cfg["off"] == "DEF":
            w.loc[~on, dfn] = 1.0 / len(dfn)
        n = int(cfg["n"])
        ok = u.eligible[cols].all(axis=1) & (np.arange(len(w)) >= n)
        w.loc[~ok.to_numpy()] = np.nan
        return w

    return target


def describe_g(cfg: dict[str, Any]) -> str:
    s = {"sma": f"시장이 {cfg['n']}일 평균 위", "mom": f"시장 {cfg['n']}일 초과 모멘텀 > 0",
         "rel": f"최근 {cfg['n']}일 순환주가 방어주보다 강함"}[cfg["sig"]]
    return f"{s}이면 {ON_KO[cfg['on']]}, 아니면 {OFF_KO[cfg['off']]} · {SCHEDULE_KO[cfg['sched']]}"


FAM = Family("G", "경기순환주·방어주 전환", "industry", grid_g, ladders_g, make_target(CYC, DEF, "MKT"), describe_g)


# ── 자료 ─────────────────────────────────────────────────────────────────────────────────────

def industry_universe() -> Universe:
    ind, fr = ld.load_french_industries(), ld.load_french()
    ret = ind.join(fr["mkt"].rename("MKT"), how="inner").loc[START:]
    return Universe(ret, fr["rf"])


def etf_universe() -> tuple[Universe, pd.Series]:
    """실제 섹터 ETF (idea_batch 와 같은 자료, 2015~) — 재확인 구간 참고용."""
    etf = load_ohlcv(RESEARCH_DIR / "etf_ohlcv.parquet")
    close = pd.DataFrame({t: etf[t]["Close"] for t in CYC_ETF + DEF_ETF + ["SPY"]})
    ret = close.pct_change(fill_method=None).iloc[1:]
    return Universe(ret, ld.load_french()["rf"]), ret["SPY"]


# ── 실행 ──────────────────────────────────────────────────────────────────────────────────────

@dataclass
class GState:
    search: ls.SearchResult | None = None
    valid: ls.Eval | None = None
    confirm: ls.Eval | None = None
    stress: dict[str, ls.Eval] = field(default_factory=dict)
    neighbors: dict[str, float] = field(default_factory=dict)
    sched_table: pd.DataFrame | None = None
    combos: pd.DataFrame | None = None
    etf: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    bear: pd.DataFrame | None = None
    started: str = ""


def search(u: Universe) -> ls.SearchResult:
    """lab_v1 과 같은 탐색·선택 (자료는 2007-12-31 에서 자른 것만)."""
    lo, hi = SEARCH
    ut = u.truncate(hi)
    grid = FAM.grid()
    rf_w = ut.rf.loc[lo:hi]
    days = ut.ret.loc[lo:hi].index
    years = len(days) / TRADING_DAYS
    mat = np.empty((len(days), len(grid)), dtype=np.float32)
    rows, ck, ct = [], None, None
    t0 = time.time()
    for j, cfg in enumerate(grid):
        tk = target_key(FAM, cfg)
        if tk != ck:
            ck, ct = tk, FAM.target(ut, cfg)
        sim = run_schedule(ct, ut.ret[list(ct.columns)], ut.rf, schedule_of(cfg))
        r = sim.ret.loc[lo:hi]
        p = perf(r, rf_w)
        rows.append({"family": "G", "key": FAM.key(cfg), "설명": FAM.describe(cfg), **p, "score": score(p),
                     "노출": float(sim.exposure.loc[lo:hi].mean()),
                     "매매_연": float((sim.turnover.loc[lo:hi] > 0).sum() / years),
                     "회전율_연": float(sim.turnover.loc[lo:hi].sum() / years)})
        mat[:, j] = (r - rf_w).to_numpy(dtype=np.float32)
        if (j + 1) % 36 == 0:
            print(f"[G] 탐색 {j + 1}/{len(grid)} [{(j + 1) / len(grid):.0%}] {time.time() - t0:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    bench = perf(ut.ret["MKT"].loc[lo:hi], rf_w)
    sh = table["Sharpe"].to_numpy(dtype=float)
    smooth = ls.smoothed_scores(FAM, grid, table["score"].to_numpy(dtype=float))
    table["smooth"] = smooth
    chosen, sbest = ls.select(FAM, grid, sh, smooth)
    table["선택"] = ""
    table.loc[sbest, "선택"] = "이웃평균 최고"
    table.loc[chosen, "선택"] = "선택" if chosen == sbest else "선택(느린 주기)"
    pbo = pbo_cscv(mat)
    share, n_nb = ls.neighbor_share(FAM, grid, sh, chosen, bench["Sharpe"])
    xs = pd.Series(mat[:, chosen].astype(float))
    dsr = deflated_sharpe(sh[chosen] / math.sqrt(TRADING_DAYS), list(sh / math.sqrt(TRADING_DAYS)), len(xs),
                          float(xs.skew()), float(xs.kurt()) + 3.0)
    print(f"[G] 선택: {FAM.describe(grid[chosen])} · Sharpe {sh[chosen]:.2f} vs 벤치 {bench['Sharpe']:.2f} · "
          f"PBO {pbo['PBO']:.2f} · 이웃 {share:.0%} ({n_nb}) · DSR {dsr:.2f}", flush=True)
    return ls.SearchResult("G", grid, table, bench, float(bench["vol"]), chosen, sbest, pbo, share, n_nb, float(dsr))


def evaluate(u: Universe, cfg: dict[str, Any], window: tuple[pd.Timestamp, pd.Timestamp], sigma: float,
             target=None, bench: pd.Series | None = None, **kw: Any) -> ls.Eval:
    tgt = (target or FAM.target)(u, cfg)
    b = u.ret["MKT"] if bench is None else bench
    return ls.evaluate_target(tgt, u, schedule_of(cfg), b, window, sigma, **kw)


def bear_table(u: Universe) -> pd.DataFrame:
    """큰 하락장에서 순환주·방어주·시장 바구니 수익 (참고)."""
    spans = [("1973-01-11", "1974-10-03", "1973~74 오일쇼크"), ("1987-08-25", "1987-12-04", "1987 블랙먼데이"),
             ("2000-03-24", "2002-10-09", "2000~02 닷컴"), ("2007-10-09", "2009-03-09", "2007~09 금융위기"),
             ("2020-02-19", "2020-03-23", "2020 코로나"), ("2022-01-03", "2022-10-12", "2022 하락장")]
    rows = []
    for a, b, name in spans:
        r = u.ret.loc[a:b]
        rows.append({"구간": name, "기간": f"{a} ~ {b}", "시장": float((1 + r["MKT"]).prod() - 1),
                     "순환주": float((1 + r[CYC].mean(axis=1)).prod() - 1),
                     "방어주": float((1 + r[DEF].mean(axis=1)).prod() - 1)})
    return pd.DataFrame(rows)


def register() -> None:
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"[G] {FAMILY} 이미 등록됨", flush=True)
        return
    doc = (__doc__ or "")
    memo = doc[doc.index("- 자료:"):doc.index("사용:")].replace("\n", " ").strip()
    log_trials(STEP, FAMILY, len(FAM.grid()), [{"test_id": "(등록)", "판정": "등록", "메모": "[lab G 순환·방어 전환] " + memo}])
    print(f"[G] {FAMILY} 등록 · 조합 {len(FAM.grid())} · 누적 시도 {total_trials():,}건", flush=True)


def run() -> None:
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register (결과 계산 전 사전 등록)")
    already = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    st = GState(started=f"{pd.Timestamp.now():%Y-%m-%d %H:%M}")
    print("===== 자료 (해시 확인)", flush=True)
    u = industry_universe()
    print("===== 1단계 탐색 (2007-12-31 까지만)", flush=True)
    st.search = sr = search(u)
    cfg = sr.grid[sr.chosen]
    full = run_schedule(FAM.target(u, cfg), u.ret[CYC + DEF + ["MKT"]], u.rf, schedule_of(cfg)).ret.loc[SEARCH[0]:SEARCH[1]]
    ut = u.truncate(SEARCH[1])
    cut = run_schedule(FAM.target(ut, cfg), ut.ret[CYC + DEF + ["MKT"]], ut.rf, schedule_of(cfg)).ret.loc[SEARCH[0]:]
    gap = float((full - cut).abs().max())
    if gap > 1e-12:
        raise SystemExit(f"전체 자료와 자른 자료의 탐색 구간 수익이 다르다 ({gap:.2e})")
    print("===== 2단계 검증 (2008~2015)", flush=True)
    st.valid = evaluate(u, cfg, VALID, sr.sigma_star, boot=True)
    v = st.valid
    print(f"[G] 검증: Sharpe {v.p['Sharpe']:.2f} vs 시장 {v.pb['Sharpe']:.2f} · 맞춤 CAGR {v.pv['CAGR']:.1%} vs {v.pb['CAGR']:.1%} · "
          f"MDD {v.pv['MDD']:.1%} vs {v.pb['MDD']:.1%} → {'통과' if v.gate else '탈락'}", flush=True)
    print("===== 3단계 재확인 (2016-01 ~ 2025-09) · 스트레스 · 이웃 · 주기 · 조합 · ETF", flush=True)
    st.confirm = evaluate(u, cfg, CONFIRM, sr.sigma_star, boot=True)
    c = st.confirm
    print(f"[G] 재확인: Sharpe {c.p['Sharpe']:.2f} vs 시장 {c.pb['Sharpe']:.2f} · 맞춤 CAGR {c.pv['CAGR']:.1%} vs {c.pb['CAGR']:.1%} · "
          f"MDD {c.pv['MDD']:.1%} vs {c.pb['MDD']:.1%}", flush=True)
    st.stress = {"비용 2배": evaluate(u, cfg, CONFIRM, sr.sigma_star, cost=2 * COST_PER_SIDE),
                 "하루 지연": evaluate(u, cfg, CONFIRM, sr.sigma_star, delay=1)}
    keys = [FAM.key(x) for x in sr.grid]
    nb = [keys.index(k) for k in FAM.neighbors(cfg, set(keys))]
    nsh = [evaluate(u, sr.grid[j], CONFIRM, sr.sigma_star).p["Sharpe"] for j in nb]
    st.neighbors = {"이웃": len(nsh), "재확인_Sharpe_중앙": float(np.median(nsh)) if nsh else float("nan"),
                    "시장보다_높은_비율": float(np.mean([x > c.pb["Sharpe"] for x in nsh])) if nsh else float("nan")}
    rows = []
    for s in SCHEDULE_ORDER:
        c2 = {**cfg, "sched": s}
        j = keys.index(FAM.key(c2))
        rows.append({"주기": SCHEDULE_KO[s], "탐색 Sharpe": sr.table.loc[j, "Sharpe"], "검증 Sharpe":
                     evaluate(u, c2, VALID, sr.sigma_star).p["Sharpe"], "재확인 Sharpe":
                     evaluate(u, c2, CONFIRM, sr.sigma_star).p["Sharpe"], "선택": "◀" if s == cfg["sched"] else ""})
    st.sched_table = pd.DataFrame(rows)
    rows = []
    for on in ("CYC", "MKT"):
        for off in ("DEF", "CASH"):
            c2 = {**cfg, "on": on, "off": off}
            j = keys.index(FAM.key(c2))
            ev_v, ev_c = evaluate(u, c2, VALID, sr.sigma_star), evaluate(u, c2, CONFIRM, sr.sigma_star)
            rows.append({"상승일 때": ON_KO[on], "하락일 때": OFF_KO[off], "탐색 Sharpe": sr.table.loc[j, "Sharpe"],
                         "탐색 MDD": sr.table.loc[j, "MDD"], "검증 Sharpe": ev_v.p["Sharpe"], "검증 CAGR": ev_v.p["CAGR"],
                         "검증 MDD": ev_v.p["MDD"], "재확인 Sharpe": ev_c.p["Sharpe"], "재확인 CAGR": ev_c.p["CAGR"],
                         "재확인 MDD": ev_c.p["MDD"], "선택": "◀" if (on, off) == (cfg["on"], cfg["off"]) else ""})
    st.combos = pd.DataFrame(rows)
    eu, spy = etf_universe()
    etf_target = make_target(CYC_ETF, DEF_ETF, "SPY")
    st.etf = {"선택": evaluate(eu, cfg, CONFIRM, sr.sigma_star, target=etf_target, bench=spy)}
    for on in ("CYC", "MKT"):
        for off in ("DEF", "CASH"):
            st.etf[f"{ON_KO[on]}/{OFF_KO[off]}"] = evaluate(eu, {**cfg, "on": on, "off": off}, CONFIRM, sr.sigma_star,
                                                            target=etf_target, bench=spy)
    st.bear = bear_table(u)
    st.gates = {"g1": bool(sr.pbo["PBO"] < ls.PBO_MAX), "g2": bool(sr.n_neighbors >= 2 and sr.neighbor_share >= ls.NEIGHBOR_MIN),
                "g3": st.valid.gate, "g4": st.confirm.gate, "g6": all(e.gate for e in st.stress.values())}
    with STATE.open("wb") as fh:
        pickle.dump(st, fh)
    final = "실전 후보" if all(st.gates.values()) else "탈락"
    if already:
        print("결과 행이 이미 TRIAL_LOG 에 있다 — 다시 쓰지 않는다", flush=True)
    else:
        g = st.gates
        log_trials(STEP, FAMILY, len(sr.grid), [
            {"test_id": "G-탐색", "판정": "통과" if g["g1"] and g["g2"] else "탈락",
             "메모": (f"선택: {FAM.describe(cfg)} · 탐색 Sharpe {sr.table.loc[sr.chosen, 'Sharpe']:.2f} vs 시장 "
                      f"{sr.bench['Sharpe']:.2f} · PBO {sr.pbo['PBO']:.2f} · 이웃 {sr.neighbor_share:.0%}({sr.n_neighbors}) · "
                      f"DSR {sr.dsr:.2f}")},
            {"test_id": "G-검증", "판정": "통과" if g["g3"] else "탈락", "메모": ls._eval_memo(st.valid).replace("SPY", "시장")},
            {"test_id": "G-재확인", "판정": "통과" if g["g4"] else "탈락", "메모": ls._eval_memo(st.confirm).replace("SPY", "시장")},
            {"test_id": "G-스트레스", "판정": "통과" if g["g6"] else "탈락",
             "메모": " · ".join(f"{k}: Sharpe {e.p['Sharpe']:.2f}, 맞춤 CAGR {e.pv['CAGR']:+.1%}, MDD {e.pv['MDD']:.1%}"
                               for k, e in st.stress.items())},
            {"test_id": "G-최종", "판정": final,
             "메모": " · ".join(f"{ls.GATE_KO[k]} {'O' if x else 'X'}" for k, x in g.items())}])
    write_outputs(st)


def _f(v: Any, fmt: str) -> str:
    return ls._f(v, fmt)


def write_outputs(st: GState) -> None:
    sr = st.search
    cfg = sr.grid[sr.chosen]
    sr.table.to_csv(OUT_CSV, index=False)
    final = all(st.gates.values())
    v, c = st.valid, st.confirm
    head = ("**결론: 경기순환주·방어주 전환 규칙이 사전 기준을 모두 통과했다 — 실전 후보.**" if final else
            "**결론: 경기순환주·방어주 전환은 사전 기준을 통과하지 못했다.** 아래 표에서 어디서 떨어졌는지, 하락장에서 "
            "방어주가 현금보다 나았는지를 본다.")
    L = [f"# 연구소 G군 — 경기순환주·방어주 전환 ({FAMILY})", "", head, "",
         f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · 조합 {len(sr.grid)} · 누적 시도 {total_trials():,}건 · 실행 {st.started}",
         "- 계기: 사용자 — 섹터 분류(경기순환주·경기방어주)로 그때그때 맞게 사면 도움이 될까 (2026-10-14).",
         "- 자료: Kenneth French 12 산업 일간(시가총액 가중). 순환주 8 = " + "·".join(IND_KO[x] for x in CYC)
         + ", 방어주 4 = " + "·".join(IND_KO[x] for x in DEF) + " (바구니 안은 산업 동일 비중). 시트의 칸 구분 대신 일반 구분.",
         "- 탐색 1950~2007 → 검증 2008~2015 → 재확인 2016~2025-09. 벤치마크 = 미국 시장 총수익. Sharpe = 무위험 초과. 편도 0.1%.",
         "- 통과 기준: lab_v1 과 같음 (① PBO < 0.3 ② 이웃 80%+ ③ 검증 ④ 재확인 ⑥ 비용 2배·하루 지연).", "",
         "## 판정", "", "| 선택된 규칙 | ① PBO | ② 이웃 | ③ 검증 | ④ 재확인 | ⑥ 스트레스 | 판정 |", "|---|---|---|---|---|---|---|",
         f"| {FAM.describe(cfg)} | {sr.pbo['PBO']:.2f} {ls._ox(st.gates['g1'])} | {_f(sr.neighbor_share, '.0%')} "
         f"({sr.n_neighbors}) {ls._ox(st.gates['g2'])} | {ls._ox(st.gates['g3'])} | {ls._ox(st.gates['g4'])} | "
         f"{ls._ox(st.gates['g6'])} | **{'실전 후보' if final else '탈락'}** |", "",
         "## 구간별 (시장 대비)", "",
         "| 구간 | Sharpe | 시장 | ΔSharpe [95%] · p | CAGR | MDD | 맞춤 CAGR | 맞춤 MDD | 시장 CAGR | 시장 MDD | 통과 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    t = sr.table.loc[sr.chosen]
    L.append(f"| 탐색 1950~2007 | {t['Sharpe']:.2f} | {sr.bench['Sharpe']:.2f} | | {t['CAGR']:+.1%} | {t['MDD']:.1%} | | | "
             f"{sr.bench['CAGR']:+.1%} | {sr.bench['MDD']:.1%} | |")
    for name, e in (("검증 2008~2015", v), ("재확인 2016~2025", c)):
        b = e.boot or {}
        ci = f"{b['ΔSharpe']:+.2f} [{b['ΔSharpe_하한']:+.2f}, {b['ΔSharpe_상한']:+.2f}] · {b['p_개선아님']:.3f}" if b else ""
        L.append(f"| {name} | {e.p['Sharpe']:.2f} | {e.pb['Sharpe']:.2f} | {ci} | {e.p['CAGR']:+.1%} | {e.p['MDD']:.1%} | "
                 f"{e.pv['CAGR']:+.1%} | {e.pv['MDD']:.1%} | {e.pb['CAGR']:+.1%} | {e.pb['MDD']:.1%} | {ls._ox(e.gate)} |")
    for k, e in st.stress.items():
        L.append(f"| 재확인·{k} | {e.p['Sharpe']:.2f} | {e.pb['Sharpe']:.2f} | | {e.p['CAGR']:+.1%} | {e.p['MDD']:.1%} | "
                 f"{e.pv['CAGR']:+.1%} | {e.pv['MDD']:.1%} | | | {ls._ox(e.gate)} |")
    L += ["", f"- 탐색: PBO {sr.pbo['PBO']:.2f} · 이웃 {sr.n_neighbors}개 중 시장보다 Sharpe 높은 비율 {_f(sr.neighbor_share, '.0%')} · "
          f"DSR {sr.dsr:.2f} · 재확인 구간 이웃 Sharpe 중앙 {_f(st.neighbors.get('재확인_Sharpe_중앙'), '.2f')}", ""]
    L += ["## 하락일 때 방어주 vs 현금 (같은 추세 신호, 판정 없음)", "",
          "| 상승일 때 | 하락일 때 | 탐색 Sharpe | 탐색 MDD | 검증 Sharpe | 검증 CAGR | 검증 MDD | 재확인 Sharpe | 재확인 CAGR | 재확인 MDD | |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in st.combos.iterrows():
        L.append(f"| {r['상승일 때']} | {r['하락일 때']} | {r['탐색 Sharpe']:.2f} | {r['탐색 MDD']:.1%} | {r['검증 Sharpe']:.2f} | "
                 f"{r['검증 CAGR']:+.1%} | {r['검증 MDD']:.1%} | {r['재확인 Sharpe']:.2f} | {r['재확인 CAGR']:+.1%} | "
                 f"{r['재확인 MDD']:.1%} | {r['선택']} |")
    L += ["", f"- 시장 매수보유: 검증 Sharpe {v.pb['Sharpe']:.2f} · CAGR {v.pb['CAGR']:+.1%} · MDD {v.pb['MDD']:.1%} / 재확인 "
          f"Sharpe {c.pb['Sharpe']:.2f} · CAGR {c.pb['CAGR']:+.1%} · MDD {c.pb['MDD']:.1%}", ""]
    L += ["## 실제 섹터 ETF 로 다시 계산 (2016~2025-09, 참고)", "",
          "순환 = XLY·XLF·XLI·XLB·XLE·XLK, 방어 = XLP·XLV·XLU (동일 비중), 시장 = SPY. 같은 신호를 SPY·ETF 로 계산.", "",
          "| 조합 | Sharpe | CAGR | MDD | SPY Sharpe | SPY CAGR | SPY MDD |", "|---|---|---|---|---|---|---|"]
    for k, e in st.etf.items():
        L.append(f"| {k} | {e.p['Sharpe']:.2f} | {e.p['CAGR']:+.1%} | {e.p['MDD']:.1%} | {e.pb['Sharpe']:.2f} | {e.pb['CAGR']:+.1%} | "
                 f"{e.pb['MDD']:.1%} |")
    L += ["", "## 매매 주기 비교", "", "| 주기 | 탐색 Sharpe | 검증 Sharpe | 재확인 Sharpe | |", "|---|---|---|---|---|"]
    for _, r in st.sched_table.iterrows():
        L.append(f"| {r['주기']} | {r['탐색 Sharpe']:.2f} | {r['검증 Sharpe']:.2f} | {r['재확인 Sharpe']:.2f} | {r['선택']} |")
    L += ["", "## 큰 하락장에서 바구니별 수익 (참고)", "", "| 구간 | 기간 | 시장 | 순환주 | 방어주 |", "|---|---|---|---|---|"]
    for _, r in st.bear.iterrows():
        L.append(f"| {r['구간']} | {r['기간']} | {r['시장']:+.1%} | {r['순환주']:+.1%} | {r['방어주']:+.1%} |")
    top = sr.table.sort_values("smooth", ascending=False).head(5)
    L += ["", "## 탐색 이웃 평균 상위 5 (표본 내)", ""] + [f"- {r['설명']} — Sharpe {r['Sharpe']:.2f}, MDD {r['MDD']:.1%}"
                                                      for _, r in top.iterrows()]
    L += ["", "## 한계", "",
          "- 산업 수익은 Kenneth French(CRSP 전 종목) 시가총액 가중이다. 실제로는 섹터 ETF 로 사야 하고 구성이 조금 다르다 (위 ETF 표).",
          "- 순환·방어 구분은 일반 구분을 썼다 (사용자 시트의 칸 구분과 다를 수 있다). 유통에는 필수소비재 유통(월마트 등)도 섞여 있다.",
          "- 재확인 구간(2016~2025)은 지금까지 개별주식 연구에 많이 쓰였다. 홀드아웃(2025-10~)은 쓰지 않았다.",
          "- 결과가 좋아도 PT-1 결합·실전 반영은 사용자 결정이다.", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[G] 보고서: {REPORT.relative_to(TIER3_DIR.parents[2])}", flush=True)


def report_only() -> None:
    class _U(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "__main__":
                module = "lab_sector_rotation"
            return super().find_class(module, name)

    with STATE.open("rb") as fh:
        write_outputs(_U(fh).load())


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
