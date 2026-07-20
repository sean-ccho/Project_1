"""XGBoost 모델 훈련 파이프라인.

Walk-forward CV 검증 후 전체 데이터로 최종 모델 훈련 → data/ml_models/ 에 저장.

실행:
    python -m ml.training
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from screener.config import ML_MODEL_DIR, ML_MIN_TRAINING_SAMPLES, ML_TIME_SERIES_N_SPLITS
from ml.feature_engineering import (
    FEATURE_COLS,
    CATEGORICAL_COLS,
    build_feature_matrix,
    load_training_trades,
    make_labels,
)
from ml.evaluation import print_cv_report, walk_forward_cv

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TARGETS = ["y_win", "y_big_win", "y_drawdown"]

# HistGradientBoostingClassifier: NaN 네이티브 처리 + XGBoost급 성능
_MODEL_PARAMS = {
    "max_iter": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 10,
    "random_state": 42,
}


def build_entry_quality_model(
    trades_df: pd.DataFrame,
    target: str = "y_win",
    n_splits: int = ML_TIME_SERIES_N_SPLITS,
) -> tuple[HistGradientBoostingClassifier, dict]:
    """Walk-forward CV → 최종 모델 훈련 → (model, metrics) 반환."""
    X, y = build_feature_matrix(trades_df, target)

    if len(X) < ML_MIN_TRAINING_SAMPLES:
        logger.warning(f"[{target}] 학습 데이터 부족: {len(X)} < {ML_MIN_TRAINING_SAMPLES}")

    # walk-forward CV
    fold_results = walk_forward_cv(
        HistGradientBoostingClassifier,
        X, y,
        n_splits=n_splits,
        model_kwargs=_MODEL_PARAMS,
    )
    print_cv_report(fold_results, target)

    aucs = [r["auc"] for r in fold_results if not np.isnan(r["auc"])]
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    # 전체 데이터로 최종 모델 훈련
    final_model = HistGradientBoostingClassifier(**_MODEL_PARAMS)
    final_model.fit(X, y)

    # 피처 중요도 상위 15개 (permutation importance 대신 간단히 None)
    feat_names = list(X.columns)
    importances = getattr(final_model, "feature_importances_", np.zeros(len(feat_names)))
    top_features = sorted(
        zip(feat_names, importances.tolist()),
        key=lambda x: x[1], reverse=True
    )[:15]

    metrics = {
        "target": target,
        "train_samples": len(X),
        "cv_folds": fold_results,
        "mean_auc": mean_auc,
        "feature_cols": feat_names,
        "top_features": top_features,
        "trained_at": datetime.now().isoformat(),
    }
    return final_model, metrics


def save_model(
    model: HistGradientBoostingClassifier,
    metrics: dict,
    output_dir: Path,
    target: str,
) -> None:
    """joblib으로 모델 저장 + meta.json 기록 (upside_model 캐시 패턴 준수)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"entry_quality_{target}_v1.pkl"
    meta_path = output_dir / f"entry_quality_{target}_meta.json"

    joblib.dump(model, model_path)
    with meta_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Saved: {model_path.name}  AUC={metrics.get('mean_auc', 'n/a'):.3f}")


def train_all_models(
    runs_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """CLI 엔트리포인트. 3개 타겟 모두 훈련 + 저장."""
    output_dir = output_dir or (_PROJECT_ROOT / ML_MODEL_DIR)

    print("=== ML 모델 훈련 시작 ===")
    trades_df = load_training_trades(runs_dir)
    print(f"학습 데이터: {len(trades_df)}건  전략 분포: {trades_df['strategy'].value_counts().to_dict()}")

    if trades_df.empty:
        print("ERROR: 학습 데이터 없음. 백테스트 run이 output/runs/ 에 있는지 확인하세요.")
        sys.exit(1)

    trades_df = make_labels(trades_df)

    for target in _TARGETS:
        print(f"\n--- {target} ---")
        model, metrics = build_entry_quality_model(trades_df, target)
        save_model(model, metrics, output_dir, target)

    print("\n=== 완료 ===")
    print(f"artifact 저장 위치: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_all_models()
