#!/usr/bin/env python3
"""Tier 3-2: 팩터 리서치 — 전 종목 × 전 날짜 횡단면 IC.

입력: data/research/panel_{set}.parquet  (build_research_panel.py 산출)
방법:
  - 날짜별 Spearman IC(팩터, 선행수익률) → 평균 IC, t-stat, hit-rate
  - t-stat은 선행수익률 겹침(autocorrelation) 때문에 **h일 간격 비중첩 샘플**로 계산
  - 연도별 평균 IC 부호 일관성 (전체 평균 부호와 같은 연도 비율)
  - 5분위 수익률 (Q5-Q1 스프레드, 비중첩 t-stat)
  - 레짐(bull/neutral/bear)별 IC — 레짐은 패널 종목 동일가중 지수의 50/200일 이동평균으로 판정
산출(기본 docs/quant_improvement/tier3/):
  factor_ic_{set}.csv, factor_quintile_{set}.csv, factor_regime_{set}.csv, FACTOR_REPORT_{set}.md

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/factor_research.py --set pit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20)
# 스케일이 종목마다 달라 횡단면 비교가 무의미한 원시 수준값 제외
EXCLUDE = {
    "close", "ema20", "ema50", "ema200", "volume", "volume_ma20", "obv", "obv_ma20",
    "atr_value", "atr_med_252", "atr_buy_max", "atr_sell_max", "stop_dist",
    "position_size", "annual_dividend",
}


def _step_dates(dates: np.ndarray, h: int) -> set:
    return set(dates[::h])


def ic_by_date(x: pd.Series, y: pd.Series, date: pd.Series) -> pd.Series:
    """날짜별 Spearman IC (벡터화). NaN 쌍은 제외."""
    df = pd.DataFrame({"x": x, "y": y, "d": date}).dropna()
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
    g = df.groupby("d")
    rx = g["x"].rank()
    ry = g["y"].rank()
    rxc = rx - rx.groupby(df["d"]).transform("mean")
    ryc = ry - ry.groupby(df["d"]).transform("mean")
    num = (rxc * ryc).groupby(df["d"]).sum()
    den = np.sqrt((rxc**2).groupby(df["d"]).sum() * (ryc**2).groupby(df["d"]).sum())
    n = df.groupby("d").size()
    ic = (num / den).where(n >= 30)
    return ic.dropna()


def tstat(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 5 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))


def market_regime(panel: pd.DataFrame) -> pd.Series:
    close = panel.pivot(index="date", columns="티커", values="close").sort_index()
    ew = (1 + close.pct_change(fill_method=None).clip(-0.5, 0.5).mean(axis=1)).cumprod()
    ma50, ma200 = ew.rolling(50).mean(), ew.rolling(200).mean()
    reg = pd.Series("neutral", index=ew.index)
    reg[(ew > ma200) & (ma50 > ma200)] = "bull"
    reg[(ew < ma200) & (ma50 < ma200)] = "bear"
    reg[ma200.isna()] = "warmup"
    return reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pit", choices=["pit", "nonpit", "px10y"])
    ap.add_argument("--out", default="docs/quant_improvement/tier3")
    args = ap.parse_args()

    panel = pd.read_parquet(ROOT / "data/research" / f"panel_{args.set}.parquet")
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    skip = {"date", "티커"} | EXCLUDE | {f"fwd_ret_{h}d" for h in HORIZONS}
    factors = [
        c for c in panel.columns
        if c not in skip and pd.api.types.is_numeric_dtype(panel[c])
        and panel[c].notna().mean() > 0.5 and panel[c].nunique() > 5
    ]
    print(f"[{args.set}] 팩터 {len(factors)}개, 행 {len(panel):,}, 날짜 {panel['date'].nunique()}")

    regime = market_regime(panel)
    all_dates = np.array(sorted(panel["date"].unique()))
    panel["year"] = panel["date"].dt.year
    # 극단 선행수익률 윈저라이즈(횡단면 순위 IC에는 무관, 분위 수익률 안정화용)
    ic_rows, q_rows, reg_rows = [], [], []

    for h in HORIZONS:
        ycol = f"fwd_ret_{h}d"
        steps = _step_dates(all_dates, h)
        sub = panel[panel[ycol].notna()]
        yw = sub[ycol].clip(sub[ycol].quantile(0.005), sub[ycol].quantile(0.995))
        for f in factors:
            ic = ic_by_date(sub[f], sub[ycol], sub["date"])
            if len(ic) < 30:
                continue
            ic_ns = ic[ic.index.isin(steps)]
            yr = ic.groupby(ic.index.year).mean()
            sign = np.sign(ic.mean())
            ic_rows.append({
                "factor": f, "horizon": h, "n_dates": len(ic),
                "mean_ic": ic.mean(), "ic_std": ic.std(),
                "t_nonoverlap": tstat(ic_ns), "t_naive": tstat(ic),
                "hit_rate": float((np.sign(ic) == sign).mean()),
                "year_sign_consistency": float((np.sign(yr) == sign).mean()),
                "yearly_ic": ";".join(f"{y}:{v:+.3f}" for y, v in yr.items()),
            })
            # 레짐별 (비중첩 날짜만)
            for rg in ("bull", "neutral", "bear"):
                s = ic_ns[regime.reindex(ic_ns.index) == rg]
                if len(s) >= 8:
                    reg_rows.append({
                        "factor": f, "horizon": h, "regime": rg, "n": len(s),
                        "mean_ic": s.mean(), "t": tstat(s),
                    })
            # 5분위
            d = pd.DataFrame({"x": sub[f], "y": yw, "d": sub["date"]}).dropna()
            d = d[np.isfinite(d["x"])]
            pct = d.groupby("d")["x"].rank(pct=True, method="first")
            d["q"] = np.ceil(pct * 5).clip(1, 5).astype(int)
            qm = d.groupby(["d", "q"])["y"].mean().unstack("q")
            qm = qm.dropna()
            if len(qm) < 30:
                continue
            spread = qm[5] - qm[1]
            sp_ns = spread[spread.index.isin(steps)]
            row = {"factor": f, "horizon": h}
            row.update({f"Q{q}": qm[q].mean() for q in range(1, 6)})
            row.update({
                "Q5_minus_Q1": spread.mean(),
                "spread_t_nonoverlap": tstat(sp_ns),
                "monotonic": bool(all(qm[q].mean() <= qm[q + 1].mean() for q in range(1, 5))
                                  or all(qm[q].mean() >= qm[q + 1].mean() for q in range(1, 5))),
            })
            q_rows.append(row)
        print(f"  horizon {h}d 완료")

    ic_df = pd.DataFrame(ic_rows)
    q_df = pd.DataFrame(q_rows)
    reg_df = pd.DataFrame(reg_rows)
    ic_df["pass_t2"] = ic_df["t_nonoverlap"].abs() >= 2
    ic_df["pass_t3"] = ic_df["t_nonoverlap"].abs() >= 3
    ic_df["pass_gate"] = ic_df["pass_t2"] & (ic_df["year_sign_consistency"] >= 0.67)

    ic_df.to_csv(out_dir / f"factor_ic_{args.set}.csv", index=False)
    q_df.to_csv(out_dir / f"factor_quintile_{args.set}.csv", index=False)
    reg_df.to_csv(out_dir / f"factor_regime_{args.set}.csv", index=False)

    # ── 요약 리포트
    lines = [f"# Tier 3 팩터 리서치 리포트 ({args.set})", ""]
    lines += [
        f"- 패널: 행 {len(panel):,} / 날짜 {panel['date'].nunique()} / 종목 {panel['티커'].nunique()}",
        f"- 기간: {panel['date'].min().date()} ~ {panel['date'].max().date()}",
        f"- 팩터 {len(factors)}개 × 기간 {list(HORIZONS)} = 검정 {len(ic_df)}건 "
        f"(다중검정 주의: |t|≥2는 우연히도 ~5% 나옴 → |t|≥3 우선 신뢰)",
        "- t-stat은 h일 간격 비중첩 샘플 기준. 선행수익률 = t+1 시가 진입 → t+1+h 시가 청산",
        f"- 게이트(참고): |t|≥2 & 연도별 부호 일관성≥67% → 통과 {int(ic_df['pass_gate'].sum())}건",
        "",
    ]
    for h in HORIZONS:
        s = ic_df[ic_df.horizon == h].sort_values("t_nonoverlap", key=abs, ascending=False)
        lines += [f"## 상위 팩터 (|t| 순) — {h}일", "",
                  "| 팩터 | 평균IC | t(비중첩) | 연도부호일관 | 연도별 IC |",
                  "|---|---|---|---|---|"]
        for _, r in s.head(15).iterrows():
            lines.append(
                f"| {r.factor} | {r.mean_ic:+.4f} | {r.t_nonoverlap:+.2f} | "
                f"{r.year_sign_consistency:.0%} | {r.yearly_ic} |")
        lines.append("")
    (out_dir / f"FACTOR_REPORT_{args.set}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:45]))
    print(f"\n저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
