"""ML 학습 데이터 로드 및 피처/라벨 생성.

upside_model._load_recent_trades() 패턴을 기반으로 전체 runs를 풀링하여
XGBoost 학습에 필요한 DataFrame을 생성한다.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS_DIR = _PROJECT_ROOT / "output" / "runs"

# 숫자형 피처 (entry_features 키와 1:1 대응)
FEATURE_COLS: list[str] = [
    # 기술지표 (15)
    "RSI", "adx", "macd_hist", "bollinger_pband", "atr_pct",
    "vol_z20", "obv_z20", "cmf_20", "pos_52w",
    "ret_5d", "ret_20d", "ema_gap_20_50", "ema_gap_50_200",
    "volatility_contraction", "buy_support_count",
    # 알파팩터 (6)
    "alpha_score", "factor_momentum", "factor_trend",
    "factor_volume", "factor_volatility", "factor_mean_reversion",
    # 전략점수 (4)
    "bottom_reversal_fit", "momentum_fit", "reversal_score", "low_prob",
    # 섹터/시장 (3)
    "in_strong_sector", "sector_relative_strength", "spy_close_to_ema200",
    # CCS 서브스코어 (6)
    "ccs_strategy_fit", "ccs_timing", "ccs_alpha",
    "ccs_risk", "ccs_confluence", "ccs_penalty",
]

CATEGORICAL_COLS: list[str] = ["sector", "regime"]

LABEL_COLS: list[str] = ["y_win", "y_return", "y_big_win", "y_drawdown"]


def load_training_trades(
    runs_dir: Path | None = None,
    max_runs: int = 10,
) -> pd.DataFrame:
    """백테스트 run들에서 거래 로드 → 중복 제거 → DataFrame 반환.

    Columns: entry_date, ticker, strategy, sector, regime, [FEATURE_COLS...],
             return_pct, post_exit_return_5d, post_exit_return_10d, post_exit_return_20d
    """
    runs_dir = runs_dir or _DEFAULT_RUNS_DIR
    run_files = sorted(runs_dir.glob("*/backtest_trades_enhanced.json"), reverse=True)
    if not run_files:
        logger.warning(f"No backtest trade files found in {runs_dir}")
        return pd.DataFrame()

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []

    for rf in run_files[:max_runs]:
        try:
            with rf.open() as f:
                trades = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load {rf}: {exc}")
            continue

        for t in trades:
            key = (t.get("ticker", ""), t.get("entry_date", ""))
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)

            feats = t.get("entry_features") or {}
            row: dict[str, Any] = {
                "ticker": t.get("ticker"),
                "entry_date": t.get("entry_date"),
                "strategy": t.get("strategy"),
                "return_pct": t.get("return_pct"),
                "post_exit_return_5d": t.get("post_exit_return_5d"),
                "post_exit_return_10d": t.get("post_exit_return_10d"),
                "post_exit_return_20d": t.get("post_exit_return_20d"),
                "sector": feats.get("sector"),
                "regime": feats.get("regime"),
            }
            for col in FEATURE_COLS:
                row[col] = feats.get(col)
            records.append(row)

    logger.info(f"Loaded {len(records)} unique trades from {min(max_runs, len(run_files))} runs")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df = df.sort_values("entry_date").reset_index(drop=True)

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_labels(trades: pd.DataFrame) -> pd.DataFrame:
    """4종 라벨을 trades DataFrame에 추가하여 반환.

    - y_win: return_pct > 0 (분류)
    - y_return: return_pct (회귀)
    - y_big_win: return_pct > 0.07 (강한 상승)
    - y_drawdown: post_exit_return_5d < -0.03 (조기 하락 위험)
    """
    df = trades.copy()
    df["y_win"] = (df["return_pct"] > 0).astype(int)
    df["y_return"] = df["return_pct"].astype(float)
    df["y_big_win"] = (df["return_pct"] > 0.07).astype(int)
    df["y_drawdown"] = (df["post_exit_return_5d"].fillna(0) < -0.03).astype(int)
    return df


def build_feature_matrix(
    trades: pd.DataFrame,
    target: str = "y_win",
) -> tuple[pd.DataFrame, pd.Series]:
    """학습용 (X, y) 반환. sector/regime one-hot 인코딩 포함."""
    df = make_labels(trades)
    df = df.dropna(subset=["return_pct"])

    # one-hot
    cat_dummies = pd.get_dummies(df[CATEGORICAL_COLS].fillna("Unknown"), prefix=CATEGORICAL_COLS)
    X = pd.concat([df[FEATURE_COLS], cat_dummies], axis=1)
    y = df[target]
    return X, y
