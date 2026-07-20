"""런타임 예측 API.

모델이 없거나 실패 시 0.5(중립)를 반환하므로 블렌딩 시 rule-based 점수만 반영된다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from screener.config import ML_FALLBACK_ON_MISSING, ML_MODEL_DIR
from ml.feature_engineering import CATEGORICAL_COLS, FEATURE_COLS

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_CACHE: dict[str, Any] = {}
_NEUTRAL = 0.5  # 모델 없거나 실패 시 반환값


def _load_model(target: str) -> Any | None:
    """joblib lazy load + module-level 캐시."""
    if target in _MODEL_CACHE:
        return _MODEL_CACHE[target]

    import joblib
    model_path = _PROJECT_ROOT / ML_MODEL_DIR / f"entry_quality_{target}_v1.pkl"
    if not model_path.exists():
        if ML_FALLBACK_ON_MISSING:
            return None
        raise FileNotFoundError(f"ML model not found: {model_path}")

    model = joblib.load(model_path)
    _MODEL_CACHE[target] = model
    logger.info(f"Loaded ML model: {model_path.name}")
    return model


def _row_to_features(row: dict | pd.Series) -> pd.DataFrame:
    """row (entry_features dict 또는 ranked_df row)를 학습 시와 동일한 피처 DataFrame으로 변환."""
    if isinstance(row, pd.Series):
        row = row.to_dict()

    numeric = {col: row.get(col) for col in FEATURE_COLS}
    cat = {col: row.get(col) for col in CATEGORICAL_COLS}

    df_num = pd.DataFrame([numeric])
    df_cat = pd.get_dummies(
        pd.DataFrame([cat]).fillna("Unknown"),
        prefix=CATEGORICAL_COLS,
    )
    return pd.concat([df_num, df_cat], axis=1)


def predict_entry_quality(
    row: dict | pd.Series,
    target: str = "y_win",
) -> float:
    """[0, 1] 신뢰도 점수. 모델 없거나 실패 시 0.5(중립) 반환.

    Args:
        row: entry_features dict 또는 ranked_df row (Series)
        target: "y_win" | "y_big_win" | "y_drawdown"
    """
    try:
        model = _load_model(target)
        if model is None:
            return _NEUTRAL

        X = _row_to_features(row)

        # 훈련 시 존재했던 컬럼만 사용 (one-hot 불일치 방지)
        train_cols = getattr(model, "feature_names_in_", None)
        if train_cols is not None:
            for c in list(train_cols):
                if c not in X.columns:
                    X[c] = 0
            X = X[list(train_cols)]

        proba = model.predict_proba(X)[0, 1]
        return float(proba)

    except Exception as e:
        logger.debug(f"predict_entry_quality({target}) failed: {e}")
        return _NEUTRAL


def predict_exit_signal(pos_features: dict) -> float:
    """Phase E에서 활성화. 현재는 중립 반환."""
    return _NEUTRAL
