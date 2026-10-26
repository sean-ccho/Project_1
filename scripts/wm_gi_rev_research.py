#!/usr/bin/env python3
"""주봉 골든임박 + 매출 성장 (wm_gi_rev_v1): wm_gi_research 와 같은 규칙에 '그날까지 공시된 최근 분기 매출 YoY > 0' 만 더한다.
S&P 500 (그날 구성)만 — SEC 매출 자료(gc_insider_rev.RevBook, 첫 제출 값, 분기 말 200일 넘으면 불통과)가 있는 종목.
사전 등록 1건 WR-SP vs SPY (p ≤ 0.05). 대조(판정 없음): 같은 종목(매출 자료 있는 종목)에서 매출 조건 없이 · 무작위 20시드."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gc_insider_rev as gr  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
import wm_gi_research as wg  # noqa: E402
import wm_sqz_research as ws  # noqa: E402
from paper_trading.evaluation import sharpe  # noqa: E402

FAMILY, STEP = "wm_gi_rev_v1", "wm_gi_rev"
REPORT = ROOT / "docs" / "quant_improvement" / "tier3" / "WM_GI_REV_REPORT.md"
REV_ALIAS = {v: k for k, v in ws.ALIAS.items()}      # 새 티커 → SEC 표의 옛 티커


def main() -> None:
    import argparse
    from research_utils import TRIAL_LOG, log_trials, total_trials
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    reg = ((log["family"] == FAMILY) & (log["판정"] == "등록")).any()
    if a.register:
        if not reg:
            log_trials(STEP, FAMILY, 1, [{"test_id": "(등록)", "판정": "등록", "메모":
                "[2021-10-22~2026-10-21, 그날 S&P 500] wm_gi_v1 규칙(주봉 EMA10/20 임박·$5·$5M·간격 순·5칸·교체 없음·ATR×2 손절·골든 뒤 데드·8주)에 "
                "그날까지 공시된 최근 분기 매출 YoY>0 (SEC 첫 제출 값, 분기 말 200일 넘거나 모르면 불통과) 추가. WR-SP vs SPY, "
                "SPY 이김=ΔSharpe 하한>0·p≤0.05. 대조=매출 자료 있는 같은 종목에서 매출 조건 없이, 무작위 20시드."}])
        print("등록됨", total_trials())
        return
    assert reg, "먼저 --register"
    lists = json.loads(ws.TICKERS_PATH.read_text())
    sig = wg.build()
    mk = ws.load_mkt()
    shape = (len(mk.dates), len(mk.tickers))
    we, gi, golden = (np.zeros(shape, bool) for _ in range(3))
    gap = np.full(shape, np.nan)
    i = mk.dates.get_indexer(pd.to_datetime(sig["date"]))
    j = np.array([mk.col.get(t, -1) for t in sig["ticker"]])
    ok = (i >= 0) & (j >= 0)
    i, j, x = i[ok], j[ok], sig[ok]
    we[i, j] = True; gi[i, j] = x["gi"].to_numpy(); golden[i, j] = x["golden"].to_numpy(); gap[i, j] = x["gap"].to_numpy()
    win = (mk.dates >= ws.TEST_START) & (mk.dates <= ws.TEST_END)
    spy = pd.Series(mk.adj[:, mk.col["SPY"]], index=mk.dates)[win].pct_change().dropna()
    with np.errstate(invalid="ignore"):
        liq = (mk.close >= ws.LIQ_PRICE) & (mk.dv20 >= ws.LIQ_DV)
    base = we & gi & liq & ws.universe_mask(mk, "SP", lists)
    _, rb = gr.books()
    known = np.zeros(shape, bool)      # 매출 자료가 있는 종목-날
    pos = np.zeros(shape, bool)        # 매출 YoY > 0
    for d, c in zip(*np.nonzero(base)):
        t = mk.tickers[c]
        y = rb.yoy(t, mk.dates[d])
        if y != y:
            y = rb.yoy(REV_ALIAS.get(t, t), mk.dates[d])
        known[d, c] = y == y
        pos[d, c] = y == y and y > 0
    n_all, n_known, n_pos = int(base[win].sum()), int(known[win].sum()), int(pos[win].sum())
    print(f"후보 종목-주: 전체 {n_all} · 매출 아는 것 {n_known} · 매출 성장 {n_pos}", flush=True)
    runs = {}
    for key, cand, judged in (("WR-SP", base & pos, True), ("WR-대조(매출 조건 없이, 같은 종목)", base & known, False)):
        eq, tr = wg.simulate(mk, we, gi, gap, golden, cand)
        r = ws._stats(key, eq, tr, spy, 10_000 if judged else 1_000)
        r["판정"] = ("SPY 이김" if judged and r["_st"]["ΔSharpe_하한"] > 0 and r["_st"]["p_개선아님"] <= 0.05 else
                   "관찰" if judged and r["_st"]["ΔSharpe_하한"] > 0 else "탈락") if judged else "참고"
        sh = [sharpe(wg.simulate(mk, we, gi, gap, golden, cand, seed=s)[0].pct_change().dropna().tolist()) for s in ws.SEEDS]
        r.update({"무작위 가운데": float(np.median(sh)), "무작위 최소": min(sh), "무작위 최대": max(sh),
                  "무작위 넘음": int(sum(v > r["기준_Sharpe"] for v in sh))})
        runs[key] = r
        print(key, "거래", len(tr), "Sharpe", round(r["Sharpe"], 2), r["판정"], flush=True)
    r = runs["WR-SP"]
    if not ((log["family"] == FAMILY) & (log["판정"] != "등록")).any():
        log_trials(STEP, FAMILY, 1, [{"test_id": "WR-SP", "t": "", "판정": r["판정"], "메모":
            f"Sharpe {r['Sharpe']:.2f} vs SPY {r['기준_Sharpe']:.2f}, ΔSharpe {r['ΔSharpe']:+.2f} [{r['하한']:+.2f}, {r['상한']:+.2f}] "
            f"p {r['p']:.4f}, CAGR {r['CAGR']:+.1%} vs {r['기준_CAGR']:+.1%}, 거래 {r['거래수']}, 무작위 가운데 {r['무작위 가운데']:.2f}"}])
    f = iv._f
    L = ["# 주봉 골든임박 + 매출 성장 (wm_gi_rev_v1) — 2021-10-22 ~ 2026-10-21, 그날 S&P 500", "",
         f"후보 종목-주: 주봉 임박 전체 {n_all} → 매출 자료 있는 것 {n_known} → 매출 YoY>0 {n_pos}", "",
         "| 검정 | Sharpe | 연수익 | 최대낙폭 | SPY 대비 ΔSharpe [95%] | p | 거래 | 승률 | 거래당 평균·중앙 | 무작위 가운데 (SPY 넘음) | 판정 |",
         "|---|---|---|---|---|---|---|---|---|---|---|",
         f"| SPY | {r['기준_Sharpe']:.2f} | {r['기준_CAGR']:+.1%} | {r['기준_MDD']:.1%} | | | | | | | 기준 |"]
    for k, v in runs.items():
        L.append(f"| {k} | {v['Sharpe']:.2f} | {v['CAGR']:+.1%} | {v['MDD']:.1%} | {f(v['ΔSharpe'], '+.2f')} [{f(v['하한'], '+.2f')}, "
                 f"{f(v['상한'], '+.2f')}] | {f(v['p'], '.4f')} | {v['거래수']} | {f(v['승률'], '.0%')} | {f(v['거래당 평균'], '+.2%')}·"
                 f"{f(v['거래당 중앙'], '+.2%')} | {v['무작위 가운데']:.2f} ({v['무작위 넘음']}/20) | **{v['판정']}** |")
    L += ["", "한계: 매출 자료는 SEC 표에 있는 종목만(없으면 불통과), 5년은 짧아 ±0.5 이내 차이는 운과 구분 안 됨, 옛 홀드아웃 1년 포함."]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
