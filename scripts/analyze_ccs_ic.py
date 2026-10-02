#!/usr/bin/env python3
"""CCS v1 vs v2 예측력(IC) 비교 — CCS v2 채택 판단용.

N거래일마다 그날 스냅샷에서
  - 후보 풀: PT-1 하드필터를 통과한 종목
  - 유니버스: 전략점수(바닥반등·모멘텀 중 큰 값) 4점 이상 종목
에 v1·v2 점수를 매기고, H거래일 선행수익률(종가→종가)과의 Spearman IC를 날짜별로 계산한다.
채택 기준(계획서 10절): v2의 평균 IC t값 ≥ 2, v1보다 높음, 연도별 부호가 일관됨.

사용법:
  PYTHONPATH=.:src python scripts/analyze_ccs_ic.py --period 5y --max-tickers 300 --step 5 --horizon 10
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.fetch import fetch_ohlcv
from paper_trading.backtest import _code_hash, default_universe
from paper_trading.candidate_selector import (
    REGIME_WEIGHTS,
    _apply_hard_filters,
    _ccs_v2_components,
    _safe_float,
    _score_alpha_factor,
    _score_confluence,
    _score_entry_timing,
    _score_risk_quality,
    _score_strategy_fit,
    detect_market_regime,
)
from screener.backtest import _compute_ranked_snapshot, _prepare_price_map
from screener.cache import write_cache_meta


def _ccs_v1(row: pd.Series, regime: str) -> float:
    """보유 종목이 없을 때의 CCS v1 (섹터 페널티 0, 레짐 보너스 포함)."""
    w = REGIME_WEIGHTS[regime]
    ccs = (
        w["strategy"] * _score_strategy_fit(row)
        + w["timing"] * _score_entry_timing(row)
        + w["alpha"] * _score_alpha_factor(row)
        + w["risk"] * _score_risk_quality(row)
        + w["confluence"] * _score_confluence(row)
    )
    strategy = str(row.get("전략구분", ""))
    dominant = max(_safe_float(row.get("바닥반등_적합도")), _safe_float(row.get("모멘텀_적합도")))
    if (regime == "bull" and "모멘텀" in strategy) or (regime == "bear" and "바닥반등" in strategy):
        ccs += 0.015 * dominant / 10.0
    return ccs


def _ic(scores: pd.Series, fwd: pd.Series) -> float:
    x = pd.concat([scores, fwd], axis=1).dropna()
    if len(x) < 5 or x.iloc[:, 0].nunique() < 3:
        return np.nan
    return float(spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0])


def _summarize(df: pd.DataFrame, col: str) -> str:
    s = df[col].dropna()
    if s.empty:
        return f"{col}: 데이터 없음"
    t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
    by_year = s.groupby(df.loc[s.index, "date"].dt.year).mean().round(3).to_dict()
    return f"{col:12s} 평균 {s.mean():+.3f} | t {t:+.2f} | IC>0 {(s > 0).mean():.0%} | n {len(s)} | 연도별 {by_year}"


def main() -> None:
    p = argparse.ArgumentParser(description="CCS v1 vs v2 IC 비교")
    p.add_argument("--period", default="5y")
    p.add_argument("--max-tickers", type=int, default=300)
    p.add_argument("--step", type=int, default=5, help="몇 거래일마다 측정할지")
    p.add_argument("--horizon", type=int, default=10, help="선행수익률 기간(거래일)")
    p.add_argument("--min-history", type=int, default=220)
    args = p.parse_args()

    tickers = ["SPY"] + [t for t in default_universe() if t != "SPY"][: args.max_tickers]
    raw = fetch_ohlcv(tickers, period=args.period)
    closes = raw.xs("Close", level=1, axis=1)
    dates = list(closes.index)

    key = ",".join(sorted(tickers)) + "|" + args.period + f"|fund=0|code={_code_hash()}"
    ohlcv_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    write_cache_meta(ohlcv_hash, args.period, False)
    price_map = _prepare_price_map(raw)
    ic_cache: dict = {}

    records = []
    for i in range(args.min_history, len(dates) - args.horizon, args.step):
        d, d_fwd = dates[i], dates[i + args.horizon]
        ranked = _compute_ranked_snapshot(
            price_map, d, include_fundamentals=False, cache={}, ic_weights_cache=ic_cache, ohlcv_hash=ohlcv_hash,
        )
        if ranked.empty or "티커" not in ranked.columns:
            continue
        regime = detect_market_regime(ranked)
        tick = ranked["티커"].astype(str)
        fwd = tick.map(lambda t: closes.at[d_fwd, t] / closes.at[d, t] - 1 if t in closes.columns else np.nan)
        v2 = _ccs_v2_components(ranked)["ccs_v2"]
        v1 = ranked.apply(lambda r: _ccs_v1(r, regime), axis=1)

        strat = pd.concat([
            pd.to_numeric(ranked.get("바닥반등_적합도"), errors="coerce"),
            pd.to_numeric(ranked.get("모멘텀_적합도"), errors="coerce"),
        ], axis=1).max(axis=1)
        universe = strat >= 4.0
        filtered, _ = _apply_hard_filters(ranked, [], 70 if regime == "bear" else None)
        pool = ranked.index.isin(filtered.index)

        records.append({
            "date": d, "regime": regime, "n_pool": int(pool.sum()), "n_universe": int(universe.sum()),
            "ic_v1_pool": _ic(v1[pool], fwd[pool]), "ic_v2_pool": _ic(v2[pool], fwd[pool]),
            "ic_v1_universe": _ic(v1[universe], fwd[universe]), "ic_v2_universe": _ic(v2[universe], fwd[universe]),
        })
        if len(records) % 20 == 0:
            print(f"  {d.date()} 측정 {len(records)}회")

    df = pd.DataFrame(records)
    if df.empty:
        print("측정 결과가 없습니다.")
        return
    out = ROOT / "output" / f"ccs_ic_{datetime.now():%Y-%m-%d_%H-%M}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n=== CCS IC ({len(df)}회 측정, {args.horizon}거래일 선행수익률) ===")
    for col in ("ic_v1_pool", "ic_v2_pool", "ic_v1_universe", "ic_v2_universe"):
        print(_summarize(df, col))
    print(f"\n레짐별 평균:\n{df.groupby('regime')[['ic_v1_pool', 'ic_v2_pool']].mean().round(3)}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
