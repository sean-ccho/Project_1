#!/usr/bin/env python3
"""Tier 3-3: 합성 점수 IC 검증 (횡단면 순위 합성) + 다중검정 보정.

두 종류:
  A) 가설 기반 고정 합성 (REV 반전 / LIQ 유동성 / VOL 변동성 및 조합)
     - 1단계 팩터 리서치 방향에서 착안했으므로 전체기간 결과는 낙관 편향 가능
       → 학습구간(~2024-08)/검증구간(2024-08~) 분리 결과를 같이 본다
  B) 워크포워드 데이터기반 합성
     - 학습창(과거 전부)에서 |t|가 큰 팩터 top-K를 부호와 함께 뽑아 동일가중 순위합
     - 검증은 그 이후 구간(OOS)에서만. 연 1회 재학습, 최초 학습창 1년

다중검정: 1단계 132건 + 여기서 시도한 합성 수를 합쳐 Bonferroni 임계 |t|를 출력.
t-stat은 h일 간격 비중첩 샘플. 선행수익률은 t+1 시가 진입 → t+1+h 시가 청산.

사용법: PYTHONPATH=.:src .venv/bin/python scripts/composite_research.py --set pit
산출: docs/quant_improvement/tier3/composite_{set}.csv, COMPOSITE_REPORT_{set}.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import norm

from factor_research import EXCLUDE, HORIZONS, ic_by_date, tstat

SPLIT = pd.Timestamp("2024-08-16")  # 학습/검증 경계 (고정 합성용)
FIRST_OOS = pd.Timestamp("2023-08-18")  # 워크포워드 최초 검증 시작
TOP_K = 5
PRIOR_TRIALS = 132  # 1단계 팩터 검정 수

FIXED = {
    # 이름: [(팩터, 부호)]
    "REV": [("10일고점괴리", -1), ("5일수익률", -1), ("obv_z20", -1)],
    "LIQ": [("최근20일평균거래대금", +1)],
    "VOL": [("ATR%", +1)],
    "REV+LIQ": [("10일고점괴리", -1), ("5일수익률", -1), ("obv_z20", -1), ("최근20일평균거래대금", +1)],
    "REV+LIQ+VOL": [("10일고점괴리", -1), ("5일수익률", -1), ("obv_z20", -1),
                    ("최근20일평균거래대금", +1), ("ATR%", +1)],
}


def composite_score(ranks: pd.DataFrame, spec: list[tuple[str, int]]) -> pd.Series:
    parts = [ranks[f] if s > 0 else 1.0 - ranks[f] for f, s in spec]
    return pd.concat(parts, axis=1).mean(axis=1)


def evaluate(score: pd.Series, panel: pd.DataFrame, h: int, mask: pd.Series | None,
             all_dates: np.ndarray) -> dict:
    ycol = f"fwd_ret_{h}d"
    m = panel[ycol].notna() & score.notna()
    if mask is not None:
        m &= mask
    d = panel.loc[m, ["date", ycol]].copy()
    d["s"] = score[m]
    if d["date"].nunique() < 20:
        return {}
    steps = set(all_dates[::h])
    ic = ic_by_date(d["s"], d[ycol], d["date"])
    ic_ns = ic[ic.index.isin(steps)]
    yr = ic.groupby(ic.index.year).mean()
    # 5분위 / 상위분위 초과수익 (윈저라이즈)
    lo, hi = d[ycol].quantile(0.005), d[ycol].quantile(0.995)
    d["y"] = d[ycol].clip(lo, hi)
    pct = d.groupby("date")["s"].rank(pct=True, method="first")
    d["q"] = np.ceil(pct * 5).clip(1, 5).astype(int)
    qm = d.groupby(["date", "q"])["y"].mean().unstack("q").dropna()
    ew = d.groupby("date")["y"].mean().reindex(qm.index)
    sp = (qm[5] - qm[1])
    ex = (qm[5] - ew)
    ns = lambda s: s[s.index.isin(steps)]
    return {
        "n_dates": len(ic), "mean_ic": ic.mean(), "t_nonoverlap": tstat(ic_ns),
        "hit_rate": float((np.sign(ic) == np.sign(ic.mean())).mean()),
        "yearly_ic": ";".join(f"{y}:{v:+.3f}" for y, v in yr.items()),
        "Q5_minus_Q1": sp.mean(), "spread_t": tstat(ns(sp)),
        "Q5_excess_vs_EW": ex.mean(), "Q5_excess_t": tstat(ns(ex)),
        "Q1": qm[1].mean(), "Q3": qm[3].mean(), "Q5": qm[5].mean(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pit", choices=["pit", "nonpit"])
    ap.add_argument("--out", default="docs/quant_improvement/tier3")
    args = ap.parse_args()

    panel = pd.read_parquet(ROOT / "data/research" / f"panel_{args.set}.parquet")
    panel = panel.sort_values(["date", "티커"]).reset_index(drop=True)
    skip = {"date", "티커"} | EXCLUDE | {f"fwd_ret_{h}d" for h in HORIZONS}
    factors = [c for c in panel.columns if c not in skip and pd.api.types.is_numeric_dtype(panel[c])
               and panel[c].notna().mean() > 0.5 and panel[c].nunique() > 5]
    ranks = panel[factors].replace([np.inf, -np.inf], np.nan).groupby(panel["date"]).rank(pct=True)
    all_dates = np.array(sorted(panel["date"].unique()))
    rows = []

    # ── A) 고정 합성
    n_trials = 0
    for name, spec in FIXED.items():
        score = composite_score(ranks, spec)
        for h in HORIZONS:
            n_trials += 1
            for seg, mask in (("full", None), ("train<2024-08", panel["date"] < SPLIT),
                              ("test>=2024-08", panel["date"] >= SPLIT)):
                r = evaluate(score, panel, h, mask, all_dates)
                if r:
                    rows.append({"kind": "fixed", "name": name, "horizon": h, "segment": seg, **r})

    # ── B) 워크포워드 합성
    refit_dates = [FIRST_OOS, pd.Timestamp("2024-08-16")]
    bounds = refit_dates + [panel["date"].max() + pd.Timedelta(days=1)]
    wf_info = []
    for h in HORIZONS:
        n_trials += 1
        ycol = f"fwd_ret_{h}d"
        score = pd.Series(np.nan, index=panel.index)
        for a, b in zip(bounds[:-1], bounds[1:]):
            tr = (panel["date"] < a) & panel[ycol].notna()
            # 학습창 끝 h일은 선행수익률이 검증구간과 겹치므로 제외 (룩어헤드 방지)
            tr_dates = np.array(sorted(panel.loc[tr, "date"].unique()))
            if len(tr_dates) > h + 5:
                tr &= panel["date"] <= tr_dates[-(h + 1)]
            steps_tr = set(np.array(sorted(panel.loc[tr, "date"].unique()))[::h])
            stats = []
            for f in factors:
                ic = ic_by_date(panel.loc[tr, f].replace([np.inf, -np.inf], np.nan),
                                panel.loc[tr, ycol], panel.loc[tr, "date"])
                ic = ic[ic.index.isin(steps_tr)]
                if len(ic) >= 10:  # 20일 기간은 1년 학습창에서 비중첩 표본이 ~12개
                    stats.append((f, tstat(ic), ic.mean()))
            top = sorted([s for s in stats if np.isfinite(s[1])], key=lambda s: -abs(s[1]))[:TOP_K]
            spec = [(f, 1 if t > 0 else -1) for f, t, _ in top]
            if not spec:
                wf_info.append(f"h={h} 학습<{a.date()}: 선택 가능한 팩터 없음 (구간 건너뜀)")
                continue
            wf_info.append(f"h={h} 학습<{a.date()}: " + ", ".join(
                f"{f}({'+' if s > 0 else '-'},t={t:+.2f})" for (f, s), (_, t, _) in zip(spec, top)))
            seg = (panel["date"] >= a) & (panel["date"] < b)
            sc = composite_score(ranks, spec)
            score[seg] = sc[seg]
        r = evaluate(score, panel, h, panel["date"] >= FIRST_OOS, all_dates)
        if r:
            rows.append({"kind": "walkforward", "name": f"WF_top{TOP_K}", "horizon": h,
                         "segment": f"oos>={FIRST_OOS.date()}", **r})

    res = pd.DataFrame(rows)
    out_dir = ROOT / args.out
    res.to_csv(out_dir / f"composite_{args.set}.csv", index=False)

    total = PRIOR_TRIALS + n_trials
    z_bonf = norm.isf(0.05 / total / 2)
    lines = [f"# Tier 3-3 합성 점수 검증 ({args.set})", "",
             f"- 시도 횟수: 1단계 팩터 {PRIOR_TRIALS}건 + 합성 {n_trials}건 = **{total}건**",
             f"- Bonferroni 보정 임계 |t| ≈ **{z_bonf:.2f}** (5% 유의수준, 양측)",
             "- t는 비중첩 샘플. 합성 A는 1단계 결과에서 방향을 착안 → 'full'은 낙관적, **test 구간이 핵심**",
             "- 합성 B(워크포워드)는 학습창 데이터만 사용 → OOS 결과가 비교적 정직", "",
             "## 워크포워드가 고른 팩터", ""] + [f"- {x}" for x in wf_info] + [""]
    cols = ["name", "horizon", "segment", "n_dates", "mean_ic", "t_nonoverlap", "hit_rate",
            "Q5_minus_Q1", "spread_t", "Q5_excess_vs_EW", "Q5_excess_t", "yearly_ic"]
    for h in HORIZONS:
        lines += [f"## {h}일", "", "| " + " | ".join(cols[:-1]) + " | yearly_ic |",
                  "|" + "---|" * len(cols)]
        for _, r in res[res.horizon == h].iterrows():
            lines.append("| " + " | ".join(
                f"{r[c]:+.4f}" if c in ("mean_ic", "Q5_minus_Q1", "Q5_excess_vs_EW")
                else f"{r[c]:+.2f}" if c in ("t_nonoverlap", "spread_t", "Q5_excess_t")
                else f"{r[c]:.0%}" if c == "hit_rate" else str(r[c]) for c in cols) + " |")
        lines.append("")
    (out_dir / f"COMPOSITE_REPORT_{args.set}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
