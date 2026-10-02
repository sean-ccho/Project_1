#!/usr/bin/env python3
"""백테스트 거래 로그로 피처별 예측력(IC, 5분위 수익률 차이)을 분석한다.

사용법:
    python scripts/analyze_trade_features.py                       # 기본: 캐시 미사용 run 3개
    python scripts/analyze_trade_features.py 2026-04-26_17-03 ...  # run 지정
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["2026-04-22_00-07", "2026-04-24_02-08", "2026-04-26_17-03"]


def main() -> None:
    runs = sys.argv[1:] or DEFAULT_RUNS
    df = pd.concat(
        [pd.read_csv(ROOT / "output/runs" / r / "backtest_trades_enhanced.csv").assign(run=r) for r in runs],
        ignore_index=True,
    ).drop_duplicates(subset=["ticker", "entry_date"])
    df["strat"] = df["strategy"].astype(str).str.extract(r"(바닥반등|모멘텀|숏스퀴즈|골든크로스|단타-눌림목|단타-돌파)")[0]
    feats = [c for c in df.columns if c.startswith("feat_") and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for strat, g in [("전체", df)] + list(df.groupby("strat")):
        for f in feats:
            x = g[[f, "return_pct"]].dropna()
            if len(x) < 30 or x[f].nunique() < 5:
                continue
            ic, p = spearmanr(x[f], x["return_pct"])
            q = pd.qcut(x[f].rank(method="first"), 5, labels=False)
            by_q = x.groupby(q)["return_pct"].mean()
            rows.append({
                "전략": strat, "피처": f[5:], "n": len(x), "IC": round(ic, 3),
                "p": round(p, 3), "Q5-Q1": round(by_q.iloc[-1] - by_q.iloc[0], 4),
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("분석할 피처가 없습니다 (거래 30건 미만이거나 feat_ 컬럼 없음)")
        return
    res["absIC"] = res["IC"].abs()
    pd.set_option("display.width", 200)
    print(f"고유 거래 {len(df)}건 | |IC| > {2 / len(df) ** 0.5:.2f} 이면 대략 유의")
    for strat, g in res.sort_values("absIC", ascending=False).groupby("전략"):
        print(f"\n=== {strat} ===")
        print(g.drop(columns="absIC").head(15).to_string(index=False))


if __name__ == "__main__":
    main()
