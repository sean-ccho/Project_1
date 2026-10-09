#!/usr/bin/env python3
"""PT-1 Optuna 파라미터 탐색 (Tier 4).

원칙 (계획서 5절·10절):
  - 마지막 N개월은 홀드아웃으로 봉인하고, 최적화는 그 전 구간만 쓴다.
  - 개발 구간을 연속된 K개 구간(fold)으로 나눠 각각 백테스트 → 평균 점수 - 구간 간 편차 페널티.
  - 점수 = 일간 Sharpe - 2 × max(0, |MDD| - 0.30). 거래 50건 미만·승률 40% 미만은 prune.
  - 탐색 파라미터는 피처 캐시 이후 단계(선정·청산)만 → 첫 trial 이후는 디스크 캐시로 빠르게 돈다.
  - 홀드아웃은 --evaluate-holdout 으로 최종 1회만 평가한다.

사용법:
  pip install -r requirements-research.txt
  PYTHONPATH=.:src python scripts/optimize_optuna.py --account pt1 --trials 100
  PYTHONPATH=.:src python scripts/optimize_optuna.py --account pt1 --evaluate-holdout
  optuna-dashboard sqlite:///output/optuna.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import optuna
import pandas as pd

from data.fetch import fetch_ohlcv
from paper_trading.backtest import default_universe

MIN_TRADES = 50
MIN_WIN_RATE = 0.40
MDD_LIMIT = 0.30


# ── 탐색 공간 ────────────────────────────────────────────────


def suggest_pt1(trial: optuna.Trial) -> dict[str, Any]:
    """PT-1: 1차는 청산 핵심 4개 × 전략 2개 + CCS 버전 + 교체 on/off (CCS 가중치는 2차로 분리).

    반환값은 config 모양 그대로라 config_overrides()에 바로 넣는다.
    """
    exit_params: dict[str, dict[str, float]] = {}
    for strat, short in (("모멘텀", "mom"), ("바닥반등", "bot")):
        exit_params[strat] = {
            "profit_target": trial.suggest_float(f"{short}_profit_target", 0.08, 0.25, step=0.01),
            "stop_loss": trial.suggest_float(f"{short}_stop_loss", 0.05, 0.12, step=0.01),
            "trailing_stop": trial.suggest_float(f"{short}_trailing_stop", 0.04, 0.10, step=0.01),
            "trail_activate_pct": trial.suggest_float(f"{short}_trail_activate_pct", 0.0, 0.08, step=0.01),
        }
    return {
        "CCS_VERSION": trial.suggest_categorical("ccs_version", ["v1", "v2"]),
        "PT1_REPLACE_ENABLED": trial.suggest_categorical("replace_enabled", [True, False]),
        "EXIT_PARAMS": exit_params,
    }


# ── 백테스트 실행 ────────────────────────────────────────────


def run_window(account: str, params: dict[str, Any], start: str, end: str, args: argparse.Namespace) -> dict[str, Any]:
    """한 구간 PT-1 백테스트 → summary. config를 잠시 바꿨다가 원복한다."""
    from paper_trading.backtest import run_paper_trading_backtest
    from paper_trading.config_override import config_overrides

    with config_overrides(params):
        r = run_paper_trading_backtest(
            period=args.period, max_tickers=args.max_tickers, rebalance_every=1,
            initial_capital=5000.0, start_date=start, end_date=end, save_run=False,
            pit_universe=args.pit_universe,
        )
    return r["summary"]


def score_summary(s: dict[str, Any]) -> float:
    sharpe = s.get("Sharpe_일간")
    if sharpe is None:
        sharpe = s.get("Sharpe", 0.0)
    return float(sharpe) - 2.0 * max(0.0, abs(float(s.get("MDD", 0.0))) - MDD_LIMIT)


def make_windows(args: argparse.Namespace) -> tuple[list[tuple[str, str]], tuple[str, str]]:
    """개발 구간 fold들과 홀드아웃 구간 (거래일 기준으로 자름)."""
    tickers = ["SPY"] + [t for t in default_universe() if t != "SPY"][: args.max_tickers]
    dates = list(fetch_ohlcv(tickers, period=args.period).index)
    sim = dates[args.min_history:]
    holdout_start = sim[-1] - pd.DateOffset(months=args.holdout_months)
    dev = [d for d in sim if d < holdout_start]
    hold = [d for d in sim if d >= holdout_start]
    chunks = np.array_split(np.arange(len(dev)), args.folds)
    folds = [(str(dev[c[0]].date()), str(dev[c[-1]].date())) for c in chunks if len(c) > 1]
    return folds, (str(hold[0].date()), str(hold[-1].date()))


SUGGEST = {"pt1": suggest_pt1}


def main() -> None:
    p = argparse.ArgumentParser(description="PT-1 Optuna 탐색")
    p.add_argument("--account", choices=["pt1"], default="pt1")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--period", default="5y")
    p.add_argument("--max-tickers", type=int, default=100)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--holdout-months", type=int, default=12)
    p.add_argument("--min-history", type=int, default=220)
    p.add_argument("--storage", default=f"sqlite:///{ROOT / 'output' / 'optuna.db'}")
    p.add_argument("--study-name", default=None)
    p.add_argument("--evaluate-holdout", action="store_true", help="최적 파라미터로 홀드아웃 1회 평가")
    p.add_argument("--pit-universe", action="store_true", help="그날의 S&P 500 구성종목만 후보로 (생존 편향 완화)")
    args = p.parse_args()

    (ROOT / "output").mkdir(exist_ok=True)
    folds, holdout = make_windows(args)
    study_name = args.study_name or f"{args.account}-{args.period}-{args.max_tickers}{'-pit' if args.pit_universe else ''}"
    study = optuna.create_study(
        study_name=study_name, storage=args.storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    print(f"[Optuna] {study_name} | 개발 구간 {folds} | 홀드아웃 {holdout} (봉인)")

    if args.evaluate_holdout:
        best = SUGGEST[args.account](optuna.trial.FixedTrial(study.best_params))
        s = run_window(args.account, best, holdout[0], holdout[1], args)
        print(f"[홀드아웃] 점수 {score_summary(s):+.3f} | {s}")
        return

    def objective(trial: optuna.Trial) -> float:
        params = SUGGEST[args.account](trial)
        scores, trades, wins = [], 0, 0.0
        for i, (start, end) in enumerate(folds):
            s = run_window(args.account, params, start, end, args)
            n = int(s.get("총거래수", 0) or 0)
            trades += n
            wins += float(s.get("승률", 0.0) or 0.0) * n
            scores.append(score_summary(s) if n else -1.0)
            trial.set_user_attr(f"fold{i}", {k: s.get(k) for k in ("총거래수", "승률", "MDD", "Sharpe_일간", "CAGR", "총수익률_자본기준")})
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if trades < MIN_TRADES or (trades and wins / trades < MIN_WIN_RATE):
            raise optuna.TrialPruned()
        trial.set_user_attr("trades", trades)
        return float(np.mean(scores) - 0.5 * np.std(scores))

    study.optimize(objective, n_trials=args.trials)
    print(f"[Optuna] 최고 점수 {study.best_value:+.3f}\n최적 파라미터: {study.best_params}")


if __name__ == "__main__":
    main()
