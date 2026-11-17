#!/usr/bin/env python3
"""예측 모델 v3 (수석 애널리스트 · PM) — 사전 등록 과거 성과 (2026-11-16, 한 번만 실행).

규칙은 운영 코드 그대로 (web_export/value_model.py — analyze · pm_metrics · v3_eligible · top_v3):
  v2 후보(품질 5개 통과 · 반증 조건 없음) 중 ① SBC 를 비용으로 뺀 FCF 로도 v1 가격(알파 ≥ 5%p) · v2 하방(알파 ≥ 0) 통과
  ② (FCF − SBC + 세후 이자) ÷ EV 의 알파(vs WACC) ≥ 5%p ③ 주주환원율 ≥ 0 · 최근 2년 희석주식수 연 1% 이하
  ④ 같은 업종 2개까지 → SBC 차감 하방 알파 큰 순 10칸, 칸당 10%, 빈칸은 SPY.
데이터 · 유니버스 · 수익률 · 비용은 value_v3_backtest (value_v3_recheck) 그대로 — SEC companyfacts(값마다 공시일) ·
배당 포함 월봉 · 그날 시가총액 · 매년 5/31 → 다음 5/31 (마지막 2026-10-31). SBC · 이자 · 희석주식수(가중평균)는 SEC 값.
운영은 매달 다시 맞추고 반증 조건이면 그날 SPY 로 바꾸지만, 여기선 1년에 한 번 (월간 재무 시점이 없어서).
판정은 실행 전에 docs/quant_improvement/tier3/TRIAL_LOG.csv 에 등록 (value_v3pm). 결과를 보고 고치지 않는다.

사용: PYTHONPATH=.:src python scripts/value_v3pm_backtest.py run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import value_v3_backtest as vb  # noqa: E402
from paper_trading.evaluation import sharpe  # noqa: E402
from web_export import value_model as vm  # noqa: E402

warnings.filterwarnings("ignore", message="Sorting by default when concatenating")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

OUT = vb.OUT
STRATS = ("v3_pm", "v3_picks", "top_v2", "pass_v2")
NAMES = {"v3_pm": "v3 PM 10칸 (빈칸 SPY)", "v3_picks": "v3 고른 종목만 (같은 비중)", "top_v2": "v2 상위 10",
         "pass_v2": "v2 통과"}


def add_pm_rows(ent: dict[str, Any], co: Mapping[str, Any], asof: str) -> dict[str, Any]:
    """build_ent 결과에 v3 가 읽는 희석주식수(SEC 연간 가중평균 희석 → 기본)를 같은 회계연도로 넣고 항목 판을 올린다."""
    ends = sorted((ent.get("inc") or {}).get("Total Revenue", {}))
    sh = {e: x for e in ends if (x := vb.pick(co["rows"].get("shares"), e, asof)) is not None and x > 0}
    if sh:
        ent["inc"]["Diluted Average Shares"] = sh
    ent["rows_v"] = vm.ROWS_VERSION
    return ent


def run_weighted(baskets: Sequence[Mapping[str, float]], paths: Sequence[Mapping[str, np.ndarray]],
                 cost: float = vb.COST) -> tuple[list[float], list[float]]:
    """해마다 비중대로 사서 1년 보유(비중은 흘러감), 리밸런싱 때 매매 금액 × cost — value_v3_backtest.run_series 의 비중판."""
    monthly: list[float] = []
    period: list[float] = []
    prev: dict[str, float] = {}
    for w, pth in zip(baskets, paths):
        w = {t: x for t, x in w.items() if t in pth and x > 0}
        tot = sum(w.values())
        w = {t: x / tot for t, x in w.items()}
        turn = sum(abs(w.get(t, 0.0) - prev.get(t, 0.0)) for t in set(w) | set(prev))
        g = sum(x * pth[t] for t, x in w.items())
        rets = np.diff(np.concatenate([[1.0], g])) / np.concatenate([[1.0], g[:-1]])
        rets[0] = (1 + rets[0]) * (1 - cost * turn) - 1
        monthly.extend(rets.tolist())
        period.append(float(g[-1] * (1 - cost * turn) - 1))
        last = {t: x * pth[t][-1] for t, x in w.items()}
        s = sum(last.values())
        prev = {t: v / s for t, v in last.items()} if s > 0 else {}
    return monthly, period


def slots(picks: Sequence[str], n: int = vm.PM_SLOTS) -> dict[str, float]:
    """10칸 — 종목마다 1/n, 나머지 SPY."""
    w = {t: 1.0 / n for t in picks[:n]}
    if len(picks) < n:
        w["SPY"] = w.get("SPY", 0.0) + (n - len(picks)) / n
    return w


def equal(names: Sequence[str]) -> dict[str, float]:
    return {t: 1.0 / len(names) for t in names} if names else {"SPY": 1.0}


def run_backtest(inp: Mapping[str, Any]) -> list[dict[str, Any]]:
    """value_v3_backtest.run_backtest 와 같은 회사 · 시가총액 · 유니버스. 회사마다 운영 analyze(v3 포함)를 부른다."""
    facts, px, cands, scan, sm, mem = (inp[k] for k in ("facts", "px", "cands", "scan", "sector_map", "mem"))
    spy, tnx = px["SPY"], px["^TNX"]
    now = date.today().isoformat()
    by_cik: dict[int, dict[str, Any]] = {}
    for t, c in cands.items():
        if not c.get("cik") or not c.get("px") or c["px"] not in px:
            continue
        if c["cik"] not in facts and (c.get("alt") or 0) not in facts:
            continue
        cur = by_cik.get(c["cik"])
        mc = (scan.get(c["px"]) or {}).get("mcap_now") or 0.0
        if cur is None or (c["scan"] and mc > cur["mcap_now"]):
            by_cik[c["cik"]] = {"px": c["px"], "mcap_now": mc, "scan": c["scan"], "alt": c.get("alt") or (cur or {}).get("alt")}
    cos = {cik: vb.merge_facts(facts.get(cik), facts.get(m["alt"]) if m.get("alt") else None) for cik, m in by_cik.items()}
    tick2cik = {t: c["cik"] for t, c in cands.items() if c.get("cik")}
    years = []
    for d0s, d1s in zip(vb.REBAL[:-1], vb.REBAL[1:]):
        d0, d1 = vb.month_end(d0s), vb.month_end(d1s)
        months = [m for m in spy.index if d0 < m <= d1]
        rf = vb.px_at(tnx, d0)
        rf = min(max(rf / 100.0, 0.015), 0.06) if math.isfinite(rf) else 0.04
        a_ciks = {tick2cik[t] for t in vb.members_on(mem, d0s) if t in tick2cik}
        rows_u: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
        for cik, meta in by_cik.items():
            in_a = cik in a_ciks
            if not (in_a or meta["scan"]):
                continue
            t = meta["px"]
            df = px[t]
            p0 = vb.px_at(df, d0)
            if not math.isfinite(p0):
                continue
            sec = (scan.get(t) or {}).get("sector") or sm.get(t, (None, None))[0]
            ind = (scan.get(t) or {}).get("industry") or sm.get(t, (None, None))[1]
            if not sec or vm.excluded({"sector": sec, "industry": ind}):
                continue
            co = cos[cik]
            mcap_now = meta["mcap_now"]
            p_now = float(df["close"].iloc[-1])
            if not (mcap_now and math.isfinite(p_now) and p_now > 0):
                continue
            ratio, _how = vb.share_ratio(co, df, d0s, now)
            mcap_usd = mcap_now * (p0 / p_now) * ratio
            in_b = bool(meta["scan"] and mcap_usd >= 1e9)
            if not (in_a or in_b):
                continue
            ccy = co["ccy"]
            fx = 1.0 if ccy == "USD" else vb.px_at(px.get(f"{ccy}USD=X"), d0)
            if not math.isfinite(fx):
                continue
            ent, _fl = vb.build_ent(co, d0s)
            if ent is None:
                continue
            ent = add_pm_rows(ent, co, d0s)
            beta = vb.monthly_beta_tr(df, spy, d0)
            res = vb.evaluate(ent, vb.quarters_asof(co, d0s), price=p0, mcap_fc=mcap_usd / fx, beta=beta, rf=rf)
            if res is None:
                continue
            res.update({"ticker": t, "industry": ind, "sector": sec, "own": True})
            res["pass_v3"] = vm.v3_eligible(res)
            for u, inside in (("A", in_a), ("B", in_b)):
                if inside:
                    rows_u[u].append(res)
        ports: dict[str, dict[str, Any]] = {}
        for u, rows in rows_u.items():
            rows.sort(key=lambda r: r["ticker"])
            picks = vm.top_v3(rows)
            sel = vb.select_portfolios(rows)
            ports[u] = {"v3_pm": slots(picks), "v3_picks": equal(picks), "top_v2": equal([r["ticker"] for r in sel["top_v2"]]),
                        "pass_v2": equal([r["ticker"] for r in sel["pass_v2"]]), "picks": picks,
                        "pool": [r["ticker"] for r in rows if not r.get("kill")],
                        "n_pm_ready": sum(1 for r in rows if r.get("pm_ready")), "n_rows": len(rows)}
        all_t = {t for u in ports for k in STRATS for t in ports[u][k]} | {t for u in ports for t in ports[u]["pool"]} | {"SPY"}
        paths = vb.period_paths(px, all_t, d0, months)
        years.append({"d0": d0s, "d1": d1s, "ports": ports, "paths": paths, "months": months})
        vb.log(f"{d0s}: v3 A {ports['A']['picks']} | B′ {ports['B']['picks']} (v2 후보 A {len(ports['A']['pool'])} · "
               f"B′ {len(ports['B']['pool'])})")
    return years


def summarize(years: Sequence[Mapping[str, Any]], seed: int = 13) -> dict[str, Any]:
    paths = [y["paths"] for y in years]
    spy_m, spy_p = vb.run_series([["SPY"]] * len(years), paths, cost=0.0)
    windows = {"full": (0, len(years)), "from2016": (2, len(years)), "to2025": (0, len(years) - 1)}
    rng = random.Random(seed)
    out: dict[str, Any] = {"spy": {"full": vb.stats_block(spy_m, spy_m), "period": spy_p}, "U": {}}
    for u in ("A", "B"):
        res_u: dict[str, Any] = {}
        series_m: dict[str, list[float]] = {}
        for s in STRATS:
            bk = [y["ports"][u][s] for y in years]
            m, p = run_weighted(bk, paths)
            series_m[s] = m
            blk: dict[str, Any] = {"period": p, "beat_years": sum(1 for a, b in zip(p, spy_p) if a > b),
                                   "n_avg": float(np.mean([len([t for t in b if t != "SPY"]) for b in bk]))}
            for w, (i0, i1) in windows.items():
                a = sum(len(y["months"]) for y in years[:i0])
                b = sum(len(y["months"]) for y in years[:i1])
                blk[w] = vb.stats_block(m[a:b], spy_m[a:b])
                blk[w]["vs_spy"] = vb.boot(m[a:b], spy_m[a:b])
            res_u[s] = blk
        res_u["v3_pm"]["full"]["vs_v2top"] = vb.boot(series_m["v3_pm"], series_m["top_v2"])
        res_u["v3_picks"]["full"]["vs_v2top"] = vb.boot(series_m["v3_picks"], series_m["top_v2"])
        sims = []  # 플라시보 — 같은 해 v2 후보(품질 통과 · 반증 없음)에서 같은 개수 무작위, 빈칸 SPY
        for _ in range(vb.N_PLACEBO):
            bk = []
            for y in years:
                pool = [t for t in y["ports"][u]["pool"] if t in y["paths"]]
                n = len(y["ports"][u]["picks"])
                bk.append(slots(rng.sample(pool, min(n, len(pool))) if n and pool else []))
            sims.append(sharpe(run_weighted(bk, paths)[0], 12))
        sims.sort()
        act = res_u["v3_pm"]["full"]["sharpe"]
        res_u["v3_pm"]["placebo"] = {"pct": 100.0 * sum(1 for x in sims if x < act) / len(sims),
                                     "p50": sims[len(sims) // 2], "p95": sims[int(0.95 * len(sims)) - 1]}
        res_u["picks"] = [{"d0": y["d0"], "picks": y["ports"][u]["picks"], "ready": y["ports"][u]["n_pm_ready"],
                           "rows": y["ports"][u]["n_rows"]} for y in years]
        out["U"][u] = res_u
    ok, rej = [], []
    for u in ("A", "B"):
        b = out["U"][u]["v3_pm"]
        vs = b["full"]["vs_spy"] or {}
        ok.append(bool(vs and vs["ΔSharpe_하한"] > 0 and b["placebo"]["pct"] >= 95.0))
        rej.append(bool(vs and vs["ΔSharpe_상한"] < 0))
    out["verdict"] = "후보" if all(ok) else ("기각" if all(rej) else "운과 구분 안 됨")
    return out


def report_md(res: Mapping[str, Any], years: Sequence[Mapping[str, Any]]) -> str:
    pct = vb.pct
    sp = res["spy"]["full"]

    def ci(vs: Mapping[str, float] | None) -> str:
        return (f"{vs['ΔSharpe']:+.2f} [{vs['ΔSharpe_하한']:+.2f}, {vs['ΔSharpe_상한']:+.2f}]") if vs else "—"

    L = [f"SPY(배당 포함) 2014-06~2026-10: CAGR {pct(sp['cagr'])} · 샤프 {sp['sharpe']:.2f} · MDD {pct(sp['mdd'])}\n"]
    for u, title in (("A", "방식 A — 그날 S&P 500"), ("B", "방식 B′ — 그날 시총 US$10억 이상 (지금 상장 명단)")):
        L.append(f"\n### {title}\n")
        L.append("| 묶음 | 평균 종목 수 | CAGR | 샤프 | MDD(월) | SPY 이긴 해 | 샤프 차이 vs SPY [95%] | 샤프 차이 vs v2 상위 10 [95%] | "
                 "2016~ 샤프 | 2025-05 구간 뺀 샤프 |")
        L.append("|---|---:|---:|---:|---:|---:|---|---|---:|---:|")
        for s in STRATS:
            b = res["U"][u][s]
            f = b["full"]
            L.append(f"| {NAMES[s]} | {b['n_avg']:.1f} | {pct(f['cagr'])} | {f['sharpe']:.2f} | {pct(f['mdd'])} | "
                     f"{b['beat_years']}/{len(b['period'])} | {ci(f['vs_spy'])} | {ci(f.get('vs_v2top'))} | "
                     f"{b['from2016']['sharpe']:.2f} | {b['to2025']['sharpe']:.2f} |")
        pl = res["U"][u]["v3_pm"]["placebo"]
        L.append(f"\n플라시보(v2 후보에서 같은 개수 무작위 · 빈칸 SPY) 백분위 {pl['pct']:.0f} (가운데 {pl['p50']:.2f} · 95% {pl['p95']:.2f})")
        L.append("\n| 기준일 | v3 PM 종목 |")
        L.append("|---|---|")
        for p in res["U"][u]["picks"]:
            L.append(f"| {p['d0']} | {', '.join(p['picks']) or '전부 SPY'} |")
    L.append(f"\n### 사전 등록 판정: **{res['verdict']}**")
    return "\n".join(L)


def cmd_run(_args: argparse.Namespace) -> int:
    inp = vb.load_inputs()
    years = run_backtest(inp)
    res = summarize(years)
    slim = json.loads(json.dumps(res, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)))
    (OUT / "v3pm_results.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), "utf-8")
    md = report_md(res, years)
    (OUT / "v3pm_report.md").write_text(md, "utf-8")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    args = ap.parse_args(argv)
    return {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
