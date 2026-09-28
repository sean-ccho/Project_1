"""백테스트 트레이드 진단 분석 스크립트.

output/backtest_trades_enhanced.json 을 읽어 4가지 분석 수행:
  1. 피처-수익률 상관관계 + 골든 룰 탐색
  2. 청산 사유 분석 (손절 회복 / 익절 잔여수익 / 트레일링)
  3. 전략별 세부 분석 (모멘텀 / 바닥반등)
  4. 시장 상황 & 섹터 분석

Usage:
    python scripts/analyze_trades.py [--input output/backtest_trades_enhanced.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────

def _load_trades(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        trades = json.load(f)
    flat: list[dict] = []
    for t in trades:
        row = {k: v for k, v in t.items() if k != "entry_features"}
        for k, v in (t.get("entry_features") or {}).items():
            row[f"feat_{k}"] = v
        flat.append(row)
    df = pd.DataFrame(flat)
    df["win"] = (df["return_pct"] > 0).astype(int)
    return df


def _sep(char: str = "═", width: int = 60) -> str:
    return char * width


def _header(title: str) -> None:
    print(f"\n{_sep()}")
    print(f"  {title}")
    print(_sep())


# ─────────────────────────────────────────────────────────────────
# 1. 피처-수익률 상관관계 분석
# ─────────────────────────────────────────────────────────────────

def analyze_feature_correlations(df: pd.DataFrame) -> None:
    _header("1. FEATURE-OUTCOME CORRELATIONS")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        print("  entry_features 없음 — 백테스트를 재실행하세요.")
        return

    results: list[tuple[str, float, float]] = []
    for col in feat_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = df[["return_pct"]].join(series.rename("x")).dropna()
        if len(valid) < 10:
            continue
        r, p = stats.spearmanr(valid["x"], valid["return_pct"])
        if not (math.isnan(r) or math.isnan(p)):
            results.append((col, r, p))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    top = results[:20]

    print(f"\n  상위 피처-수익률 Spearman 상관계수 (N={len(df)}):\n")
    for i, (col, r, p) in enumerate(top, 1):
        name = col.replace("feat_", "")
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
        print(f"  #{i:<2} {name:<30} r={r:+.3f}  p={p:.3f} {sig}")

    # 분위별 승률 테이블 (상위 5개 피처)
    print(f"\n  분위별 승률 (상위 5개 피처):")
    for col, r, p in top[:5]:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = df[["return_pct", "win"]].join(series.rename("x")).dropna()
        if len(valid) < 10:
            continue
        try:
            valid["quintile"] = pd.qcut(valid["x"], q=5, labels=False, duplicates="drop")
            n_bins = valid["quintile"].nunique()
            if n_bins < 2:
                continue
            label_map = {i: f"Q{i+1}" for i in range(n_bins)}
            valid["quintile"] = valid["quintile"].map(label_map)
        except Exception:
            continue
        name = col.replace("feat_", "")
        print(f"\n    {name} (r={r:+.3f}):")
        print(f"    {'분위':<6} {'건수':>5} {'승률':>8} {'평균수익':>10}")
        for q, grp in valid.groupby("quintile", observed=True):
            wr = grp["win"].mean()
            ar = grp["return_pct"].mean()
            print(f"    {str(q):<6} {len(grp):>5} {wr:>8.1%} {ar:>+10.2%}")

    # 골든 룰 탐색
    print(f"\n  골든 룰 (승률 > 65%, N ≥ 10):")
    found = 0
    for col, r, p in results:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = df[["return_pct", "win"]].join(series.rename("x")).dropna()
        if len(valid) < 10:
            continue
        name = col.replace("feat_", "")
        thresholds = valid["x"].quantile([0.25, 0.33, 0.5, 0.67, 0.75]).tolist()
        for thresh in thresholds:
            for op, label in (("lt", "<"), ("gt", ">")):
                if op == "lt":
                    subset = valid[valid["x"] < thresh]
                else:
                    subset = valid[valid["x"] > thresh]
                n = len(subset)
                if n < 10:
                    continue
                wr = subset["win"].mean()
                ar = subset["return_pct"].mean()
                if wr > 0.65:
                    w = int(subset["win"].sum())
                    print(f"    {name} {label} {thresh:.2f} → {wr:.0%} win ({w}/{n}), avg {ar:+.2%}")
                    found += 1
                    if found >= 15:
                        break
            if found >= 15:
                break
        if found >= 15:
            break
    if found == 0:
        print("    (없음 — 데이터가 더 많아지면 재분석 권장)")


# ─────────────────────────────────────────────────────────────────
# 2. 청산 사유 분석
# ─────────────────────────────────────────────────────────────────

def analyze_exit_reasons(df: pd.DataFrame) -> None:
    _header("2. EXIT REASON ANALYSIS")

    print(f"\n  {'청산사유':<20} {'건수':>5} {'승률':>8} {'평균수익':>10} {'평균보유일':>10}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*10} {'-'*10}")
    for reason, grp in df.groupby("exit_reason"):
        n = len(grp)
        wr = grp["win"].mean()
        ar = grp["return_pct"].mean()
        ah = grp["holding_days"].mean() if "holding_days" in grp else float("nan")
        print(f"  {str(reason):<20} {n:>5} {wr:>8.1%} {ar:>+10.2%} {ah:>9.1f}일")

    # 손절 회복 분석
    stop_loss = df[df["exit_reason"].str.contains("손절|stop|stoploss", case=False, na=False)]
    if len(stop_loss) >= 3:
        print(f"\n  손절 후 가격 회복률 (N={len(stop_loss)}):")
        for lag, col in ((5, "post_exit_return_5d"), (10, "post_exit_return_10d"), (20, "post_exit_return_20d")):
            series = pd.to_numeric(stop_loss[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            recovered = (series > 0).mean()
            avg = series.mean()
            print(f"    {lag:>2}일 후: 회복률 {recovered:.1%}, 평균 추가변동 {avg:+.2%} (N={len(series)})")
        print("    ※ 회복률 높으면 손절이 너무 타이트함")

    # 익절 잔여 수익
    profit_exit = df[df["exit_reason"].str.contains("목표|profit|익절", case=False, na=False)]
    if len(profit_exit) >= 3:
        print(f"\n  익절 후 추가 상승분 (N={len(profit_exit)}):")
        for lag, col in ((5, "post_exit_return_5d"), (10, "post_exit_return_10d"), (20, "post_exit_return_20d")):
            series = pd.to_numeric(profit_exit[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            further_up = (series > 0).mean()
            avg = series.mean()
            print(f"    {lag:>2}일 후: 추가상승률 {further_up:.1%}, 평균 {avg:+.2%} (N={len(series)})")
        print("    ※ 추가상승률 높으면 익절이 너무 이름")

    # 트레일링 분석
    trailing = df[df["exit_reason"].str.contains("트레일|trail", case=False, na=False)]
    if len(trailing) >= 3:
        print(f"\n  트레일링 청산 후 분석 (N={len(trailing)}):")
        for lag, col in ((5, "post_exit_return_5d"), (10, "post_exit_return_10d")):
            series = pd.to_numeric(trailing[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            down = (series < 0).mean()
            avg = series.mean()
            print(f"    {lag:>2}일 후: 추가하락률 {down:.1%}, 평균 {avg:+.2%} (N={len(series)})")
        print("    ※ 추가하락률 높으면 트레일링이 적절함")


# ─────────────────────────────────────────────────────────────────
# 3. 전략별 세부 분석
# ─────────────────────────────────────────────────────────────────

def _strategy_key(s: str) -> str:
    return str(s).replace("📉 ", "").replace("📈 ", "").replace("🔄 ", "").strip()


def _print_breakdown(label: str, df: pd.DataFrame, feat_col: str, bins: list, bin_labels: list[str]) -> None:
    series = pd.to_numeric(df[feat_col], errors="coerce")
    valid = df[["return_pct", "win"]].join(series.rename("x")).dropna()
    if len(valid) < 5:
        return
    valid["bin"] = pd.cut(valid["x"], bins=bins, labels=bin_labels, include_lowest=True)
    print(f"\n  {label}:")
    print(f"    {'구간':<18} {'건수':>5} {'승률':>8} {'평균수익':>10}")
    for b, grp in valid.groupby("bin", observed=True):
        if len(grp) < 2:
            continue
        print(f"    {str(b):<18} {len(grp):>5} {grp['win'].mean():>8.1%} {grp['return_pct'].mean():>+10.2%}")


def analyze_strategies(df: pd.DataFrame) -> None:
    _header("3. STRATEGY BREAKDOWN")

    df["_strat"] = df["strategy"].apply(_strategy_key)

    momentum_df = df[df["_strat"].str.contains("모멘텀|momentum", case=False, na=False)]
    bottom_df = df[df["_strat"].str.contains("바닥|bottom|반등", case=False, na=False)]

    # 모멘텀 분석
    if len(momentum_df) >= 5:
        print(f"\n  ── 모멘텀 전략 (N={len(momentum_df)}) ──")
        overall_wr = momentum_df["win"].mean()
        overall_ar = momentum_df["return_pct"].mean()
        print(f"  전체: 승률 {overall_wr:.1%}, 평균수익 {overall_ar:+.2%}")

        if "feat_adx" in momentum_df.columns:
            _print_breakdown(
                "ADX 구간별",
                momentum_df, "feat_adx",
                bins=[0, 15, 20, 25, 30, 100],
                bin_labels=["0-15", "15-20", "20-25", "25-30", "30+"],
            )
        if "feat_buy_support_count" in momentum_df.columns:
            _print_breakdown(
                "buy_support_count별",
                momentum_df, "feat_buy_support_count",
                bins=[-0.5, 1.5, 2.5, 3.5, 4.5, 20],
                bin_labels=["≤1", "2", "3", "4", "5+"],
            )
        if "feat_factor_momentum" in momentum_df.columns:
            _print_breakdown(
                "팩터_모멘텀 사분위",
                momentum_df, "feat_factor_momentum",
                bins=[-1.5, -0.5, 0.0, 0.5, 1.5],
                bin_labels=["약(-1~-0.5)", "보통(-0.5~0)", "양호(0~0.5)", "강(0.5~1)"],
            )

    # 바닥반등 분석
    if len(bottom_df) >= 5:
        print(f"\n  ── 바닥반등 전략 (N={len(bottom_df)}) ──")
        overall_wr = bottom_df["win"].mean()
        overall_ar = bottom_df["return_pct"].mean()
        print(f"  전체: 승률 {overall_wr:.1%}, 평균수익 {overall_ar:+.2%}")

        if "feat_RSI" in bottom_df.columns:
            _print_breakdown(
                "RSI 구간별",
                bottom_df, "feat_RSI",
                bins=[0, 25, 30, 35, 40, 50, 100],
                bin_labels=["<25", "25-30", "30-35", "35-40", "40-50", "50+"],
            )
        if "feat_reversal_score" in bottom_df.columns:
            _print_breakdown(
                "반등스코어별",
                bottom_df, "feat_reversal_score",
                bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 20],
                bin_labels=["0", "1", "2", "3", "4+"],
            )
        if "feat_factor_mean_reversion" in bottom_df.columns:
            _print_breakdown(
                "팩터_평균회귀 사분위",
                bottom_df, "feat_factor_mean_reversion",
                bins=[-1.5, -0.5, 0.0, 0.5, 1.5],
                bin_labels=["약(-1~-0.5)", "보통(-0.5~0)", "양호(0~0.5)", "강(0.5~1)"],
            )


# ─────────────────────────────────────────────────────────────────
# 4. 시장 상황 & 섹터 분석
# ─────────────────────────────────────────────────────────────────

def analyze_market_and_sector(df: pd.DataFrame) -> None:
    _header("4. MARKET & SECTOR ANALYSIS")

    # 레짐별
    if "feat_regime" in df.columns:
        print(f"\n  시장 레짐별 성과:")
        print(f"  {'레짐':<12} {'건수':>5} {'승률':>8} {'평균수익':>10}")
        for regime, grp in df.groupby("feat_regime"):
            if len(grp) < 2:
                continue
            print(f"  {str(regime):<12} {len(grp):>5} {grp['win'].mean():>8.1%} {grp['return_pct'].mean():>+10.2%}")

    # 섹터별
    sector_col = "feat_sector" if "feat_sector" in df.columns else ("sector" if "sector" in df.columns else None)
    if sector_col:
        print(f"\n  섹터별 성과 (N ≥ 5):")
        print(f"  {'섹터':<25} {'건수':>5} {'승률':>8} {'평균수익':>10}")
        sector_stats = []
        for sector, grp in df.groupby(sector_col):
            if len(grp) < 5:
                continue
            sector_stats.append((sector, len(grp), grp["win"].mean(), grp["return_pct"].mean()))
        sector_stats.sort(key=lambda x: x[2], reverse=True)
        for sector, n, wr, ar in sector_stats:
            print(f"  {str(sector):<25} {n:>5} {wr:>8.1%} {ar:>+10.2%}")

    # 월별 패턴
    if "entry_date" in df.columns:
        df["_month"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.month
        print(f"\n  월별 성과:")
        print(f"  {'월':<6} {'건수':>5} {'승률':>8} {'평균수익':>10}")
        month_names = {1:"1월",2:"2월",3:"3월",4:"4월",5:"5월",6:"6월",
                       7:"7월",8:"8월",9:"9월",10:"10월",11:"11월",12:"12월"}
        for m, grp in df.dropna(subset=["_month"]).groupby("_month"):
            if len(grp) < 2:
                continue
            print(f"  {month_names.get(int(m), str(m)):<6} {len(grp):>5} {grp['win'].mean():>8.1%} {grp['return_pct'].mean():>+10.2%}")


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="백테스트 트레이드 진단 분석")
    parser.add_argument(
        "--input",
        default="output/backtest_trades_enhanced.json",
        help="향상된 거래 로그 JSON 경로",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"[오류] 파일 없음: {path}")
        print("  먼저 실행하세요: python scripts/run_sp500_backtest.py --period 3y --capital 5000")
        return

    df = _load_trades(str(path))

    print(f"\n{'═'*60}")
    print(f"  TRADE ANALYSIS REPORT — {len(df)} trades")
    print(f"  파일: {path}")
    overall_wr = df["win"].mean()
    overall_ar = df["return_pct"].mean()
    avg_win = df[df["win"] == 1]["return_pct"].mean() if df["win"].sum() > 0 else 0.0
    avg_loss = df[df["win"] == 0]["return_pct"].mean() if (len(df) - df["win"].sum()) > 0 else 0.0
    print(f"  전체 승률: {overall_wr:.1%} | 평균수익: {overall_ar:+.2%} | 평균승: {avg_win:+.2%} | 평균패: {avg_loss:+.2%}")
    print(f"{'═'*60}")

    analyze_feature_correlations(df)
    analyze_exit_reasons(df)
    analyze_strategies(df)
    analyze_market_and_sector(df)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
