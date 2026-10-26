#!/usr/bin/env python3
"""주봉 골든크로스 임박만 (wm_gi_v1): 월봉·스퀴즈 없이 주봉 EMA10/EMA20 임박(앱 규칙, 차트 선)만으로 — 최근 5년.

사용자 2026-10-22 "only 주봉 골든임박으로 한 번만 더". wm_sqz_research 의 자료·시장 행렬·가격 규칙을 그대로 쓴다.
후보 = 주봉 임박(screener.ma_cross gi, 끝난 주봉) · 종가≥$5 · 20일 거래대금≥$5M. 순위 = 간격(EMA10−EMA20)/EMA20 이 0 에 가까운 순.
5칸 균등 · 빈 칸 그날 모두 · 다음날 시가 · 편도 0.1% · 교체 없음.
매도: 일봉 종가 < 진입가×(1−2×ATR14%) / 주봉 골든이 된 뒤 다시 EMA10<EMA20 / 8주 안에 골든이 안 되면.
사전 등록 family wm_gi_v1 2건 (WG-SP, WG-ALL) vs SPY, p ≤ 0.05/2. 참고: 나스닥·NYSE 만, 무작위 20시드, 비용 2배.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_vs_pt1 as iv  # noqa: E402
import wm_sqz_research as ws  # noqa: E402
from paper_trading.evaluation import sharpe  # noqa: E402

FAMILY, STEP = "wm_gi_v1", "wm_gi"
TESTS = {"WG-SP": ("SP", "그날 S&P 500"), "WG-ALL": ("ALL", "S&P 500 + 나스닥·NYSE (지금 상장)")}
P_BONF = 0.05 / len(TESTS)
MAX_WEEKS = 8
SIG = ws.DATA_DIR / "weekly_gi.parquet"
REPORT = ROOT / "docs" / "quant_improvement" / "tier3" / "WM_GI_REPORT.md"


def ticker_gi(args):
    from screener.ma_cross import cross_frame
    t, d = args
    d = d[d["Close"].notna()]
    if len(d) < 60:
        return pd.DataFrame()
    as_of = min(d.index[-1], ws.TEST_END)
    d = d.loc[:as_of]
    w = cross_frame(d["Close"], 10, 20, "ema", "W", as_of)
    w = w[w["complete"]]
    return pd.DataFrame({"date": w.index, "ticker": t, "gi": w["gi"].to_numpy(bool), "gap": w["gap"].to_numpy(float),
                         "golden": w["golden"].to_numpy(bool)})


def build() -> pd.DataFrame:
    if SIG.exists():
        return pd.read_parquet(SIG)
    from concurrent.futures import ProcessPoolExecutor
    long = pd.read_parquet(ws.OHLCV_PATH)
    long["date"] = pd.to_datetime(long["date"])
    groups = [(t, g.set_index("date").sort_index()[["Close"]].astype(float)) for t, g in long.groupby("ticker")]
    with ProcessPoolExecutor(max_workers=6) as ex:
        parts = list(ex.map(ticker_gi, groups, chunksize=32))
    sig = pd.concat([p for p in parts if len(p)], ignore_index=True)
    sig.to_parquet(SIG, index=False)
    return sig


def simulate(mk, we, gi, gap, golden, cand, seed=None, cost=ws.COST):
    i0 = int(mk.dates.searchsorted(ws.TEST_START))
    i1 = int(mk.dates.searchsorted(ws.TEST_END, side="right")) - 1
    cash, pos, trades, eq, sell_next, buy_next = 10_000.0, [], [], {}, [], []
    for i in range(i0, i1 + 1):
        later = []
        for p, why in sell_next:
            op, f = mk.open[i, p.j], mk.factor[i, p.j]
            if not (np.isfinite(op) and op > 0 and np.isfinite(f)):
                later.append((p, why)); continue
            cash += p.units * op * f * (1 - cost)
            trades.append(ws._trade(p, mk, i, op, f, why, cost))
        sell_next = later
        for j, alloc, sp in buy_next:
            op, f = mk.open[i, j], mk.factor[i, j]
            alloc = min(alloc, cash)
            if not (np.isfinite(op) and op > 0 and np.isfinite(f)) or alloc <= 1e-9:
                continue
            cash -= alloc
            p = ws.Pos(j, mk.tickers[j], i, float(op), alloc / (op * f * (1 + cost)), float(op * (1 - sp)), alloc, float(f))
            p.armed, p.weeks = False, 0
            pos.append(p)
        buy_next = []
        held_v = sum(p.units * mk.adj[i, p.j] for p in pos + [x[0] for x in sell_next] if np.isfinite(mk.adj[i, p.j]))
        equity = cash + held_v
        eq[mk.dates[i]] = equity
        if i == i1:
            for p in pos + [x[0] for x in sell_next]:
                c, f = mk.close[i, p.j], mk.factor[i, p.j]
                if np.isfinite(c) and np.isfinite(f):
                    trades.append(ws._trade(p, mk, i, c, f, "기간 끝", cost))
            break
        keep = []
        for p in pos:
            c, why = mk.close[i, p.j], None
            if np.isfinite(c) and c < p.stop_px:
                why = "손절"
            elif we[i, p.j] and i > p.entry_i:
                p.weeks += 1
                if golden[i, p.j]:
                    p.armed = True
                elif p.armed:
                    why = "주봉 데드크로스"
                elif p.weeks >= MAX_WEEKS:
                    why = f"{MAX_WEEKS}주 안에 골든 안 됨"
            (sell_next.append((p, why)) if why else keep.append(p))
        pos = keep
        free = ws.SLOTS - len(pos)
        js = [j for j in np.flatnonzero(cand[i]) if j not in {p.j for p in pos} | {p.j for p, _ in sell_next}
              and np.isfinite(mk.open[i + 1, j]) and mk.open[i + 1, j] > 0]
        if free <= 0 or not js:
            continue
        if seed is None:
            js.sort(key=lambda j: (abs(gap[i, j]), mk.tickers[j]))
        else:
            js.sort(key=lambda j: (iv.stable_uniform("wm_gi", seed, mk.dates[i], mk.tickers[j]), mk.tickers[j]))
        avail = cash + sum(p.units * mk.adj[i, p.j] * (1 - cost) for p, _ in sell_next if np.isfinite(mk.adj[i, p.j]))
        for j in js[:free]:
            alloc = min(avail, equity / ws.SLOTS)
            if alloc <= 1e-9:
                break
            buy_next.append((j, alloc, ws._stop_pct(mk, i, j)))
            avail -= alloc
    return pd.Series(eq), trades


def main():
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
            log_trials(STEP, FAMILY, 2, [{"test_id": "(등록)", "판정": "등록", "메모":
                "[2021-10-22~2026-10-21] 주봉 EMA10/20 임박(앱 규칙)만, 월봉·스퀴즈 없음; 종가≥$5·거래대금≥$5M; 순위=간격 0에 가까운 순; "
                "5칸 균등·교체 없음·다음날 시가·편도0.1%; 매도=ATR×2 손절·골든 뒤 데드·8주 안 골든 안 됨. WG-SP(그날 S&P 500)·"
                f"WG-ALL(+나스닥·NYSE) vs SPY, SPY 이김=ΔSharpe 하한>0·p≤{P_BONF:.3f}. 참고=나스닥·NYSE만, 무작위 20시드, 비용 2배."}])
        print("등록됨", total_trials())
        return
    assert reg, "먼저 --register"
    lists = json.loads(ws.TICKERS_PATH.read_text())
    sig = build()
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
    bench = {k: pd.Series(mk.adj[:, mk.col[k]], index=mk.dates)[win].pct_change().dropna() for k in ("SPY", "QQQ")}
    with np.errstate(invalid="ignore"):
        liq = (mk.close >= ws.LIQ_PRICE) & (mk.dv20 >= ws.LIQ_DV)
    rows, yearly = [], {"SPY": bench["SPY"]}
    spec = [("WG-SP", "SP", 1, True), ("WG-ALL", "ALL", 1, True), ("WG-NQ", "NQ", 1, False),
            ("WG-SP-비용2배", "SP", 2, False), ("WG-ALL-비용2배", "ALL", 2, False)]
    for key, uni, cm, judged in spec:
        cand = we & gi & liq & ws.universe_mask(mk, uni, lists)
        eq, tr = simulate(mk, we, gi, gap, golden, cand, cost=cm * ws.COST)
        r = ws._stats(key, eq, tr, bench["SPY"], 10_000 if judged else 1_000)
        r.update({"test_id": key, "판정": ws._judge(r["_st"]) if judged else "참고"})
        if cm == 1:
            sh = [sharpe(simulate(mk, we, gi, gap, golden, cand, seed=s)[0].pct_change().dropna().tolist()) for s in ws.SEEDS]
            r.update({"무작위 가운데": float(np.median(sh)), "무작위 최소": min(sh), "무작위 최대": max(sh),
                      "무작위 넘음": int(sum(v > r["기준_Sharpe"] for v in sh))})
        yearly[key] = eq.pct_change().dropna()
        rows.append(r)
        print(key, "거래", len(tr), "Sharpe", round(r["Sharpe"], 2), r["판정"], flush=True)
    log_rows = [{"test_id": r["test_id"], "t": "", "판정": r["판정"], "메모":
                 f"Sharpe {r['Sharpe']:.2f} vs SPY {r['기준_Sharpe']:.2f}, ΔSharpe {r['ΔSharpe']:+.2f} [{r['하한']:+.2f}, {r['상한']:+.2f}] "
                 f"p {r['p']:.4f}, CAGR {r['CAGR']:+.1%} vs {r['기준_CAGR']:+.1%}, 거래 {r['거래수']}, 무작위 가운데 {r['무작위 가운데']:.2f}"}
                for r in rows if r["test_id"] in TESTS]
    if not ((log["family"] == FAMILY) & (log["판정"] != "등록")).any():
        log_trials(STEP, FAMILY, 2, log_rows)
    f = iv._f
    L = [f"# 주봉 골든크로스 임박만 ({FAMILY}) — 2021-10-22 ~ 2026-10-21", "",
         "주봉 EMA10/EMA20 임박(앱 규칙)만, 월봉·스퀴즈 없음. 5칸 균등 · 교체 없음 · 매도: ATR×2 손절 / 골든 뒤 데드 / 8주 안 골든 안 됨.", "",
         "| 검정 | Sharpe | 연수익 | 최대낙폭 | SPY 대비 ΔSharpe [95%] | p | 거래 | 승률 | 거래당 평균·중앙 | 보유일 중앙 | 무작위 가운데 (넘음) | 판정 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|",
         f"| SPY | {rows[0]['기준_Sharpe']:.2f} | {rows[0]['기준_CAGR']:+.1%} | {rows[0]['기준_MDD']:.1%} | | | | | | | | 기준 |"]
    for r in rows:
        rnd = f"{r['무작위 가운데']:.2f} ({r['무작위 넘음']}/20)" if "무작위 가운데" in r else ""
        L.append(f"| {r['test_id']} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | {f(r['ΔSharpe'], '+.2f')} "
                 f"[{f(r['하한'], '+.2f')}, {f(r['상한'], '+.2f')}] | {f(r['p'], '.4f')} | {r['거래수']} | {f(r['승률'], '.0%')} | "
                 f"{f(r['거래당 평균'], '+.2%')}·{f(r['거래당 중앙'], '+.2%')} | {f(r['보유일 중앙'], '.0f')}일 | {rnd} | **{r['판정']}** |")
    L += ["", "청산: " + " / ".join(f"{r['test_id']}: {r['청산']}" for r in rows[:2]), "",
          "한계: 나스닥·NYSE 는 지금 상장 종목만(생존 편향), 5년은 짧아 ±0.5 이내 차이는 운과 구분 안 됨, 옛 홀드아웃 1년 포함."]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
