"""저점/고점 극단 구간 탐지를 위한 라벨링 및 간단한 모델 도우미."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    EXTREME_HIGH_MAX_DRAWDOWN,
    EXTREME_HISTORY_STEP,
    EXTREME_LABEL_LOOKAHEAD,
    EXTREME_LOW_MIN_RETURN,
    EXTREME_MODEL_FEATURES,
    EXTREME_MODEL_MIN_SAMPLES,
    EXTREME_MODEL_REG_C,
)
from data.fetch import fetch_ohlcv
from features import compute_features_for_ticker, feature_row_from_set

MIN_HISTORY_BARS = 120


@dataclass
class ModelResult:
    """저점/고점 판별 모델과 검증 통계를 보관하는 컨테이너."""

    label: str
    features: list[str]
    pipeline: Pipeline
    metrics: list[dict]


def _create_pipeline() -> Pipeline:
    """Logistic regression 파이프라인(결측치 대체 + 표준화)."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    C=EXTREME_MODEL_REG_C,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _forward_returns(close: pd.Series, start_idx: int, lookahead: int) -> tuple[float, float]:
    """주어진 인덱스 이후 lookahead 기간 동안의 최대 상승/하락률을 계산."""

    future = close.iloc[start_idx + 1 : start_idx + 1 + lookahead]
    if future.empty:
        return float("nan"), float("nan")

    base = close.iloc[start_idx]
    returns = (future / base) - 1.0
    return float(returns.max()), float(returns.min())


def build_labeled_dataset(
    price_map: Dict[str, pd.DataFrame],
    *,
    lookahead: int | None = None,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """슬라이딩 윈도우로 특징·라벨 조합을 생성한다."""

    lookahead = lookahead or EXTREME_LABEL_LOOKAHEAD
    low_threshold = low_threshold or EXTREME_LOW_MIN_RETURN
    high_threshold = high_threshold or EXTREME_HIGH_MAX_DRAWDOWN
    step = max(step or EXTREME_HISTORY_STEP, 1)

    rows: list[dict] = []

    for ticker, frame in price_map.items():
        data = frame.dropna(how="all").copy()
        if len(data) <= lookahead + MIN_HISTORY_BARS:
            continue

        data = data.sort_index()
        close_series = data["Close"]
        for idx in range(MIN_HISTORY_BARS, len(data) - lookahead):
            if (idx - MIN_HISTORY_BARS) % step != 0:
                continue

            window = data.iloc[: idx + 1]
            feature_set = compute_features_for_ticker(window)
            if feature_set is None:
                continue

            row = feature_row_from_set(ticker, feature_set)
            row["날짜"] = window.index[-1]

            max_gain, max_loss = _forward_returns(close_series, idx, lookahead)
            if np.isnan(max_gain) or np.isnan(max_loss):
                continue

            row["fwd_max_gain"] = max_gain
            row["fwd_max_loss"] = max_loss
            row["label_low"] = float(max_gain >= low_threshold)
            row["label_high"] = float(max_loss <= high_threshold)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    dataset = pd.DataFrame(rows)
    dataset["날짜"] = pd.to_datetime(dataset["날짜"])

    if {"5일수익률", "시장"}.issubset(dataset.columns):
        market_mean = dataset.groupby(["날짜", "시장"])["5일수익률"].transform("mean")
        dataset["시장상대강도"] = dataset["5일수익률"] - market_mean

    if {"5일수익률", "섹터"}.issubset(dataset.columns):
        sector_mean = dataset.groupby(["날짜", "섹터"])["5일수익률"].transform("mean")
        dataset["섹터상대강도"] = dataset["5일수익률"] - sector_mean

    dataset["label_low"] = dataset["label_low"].astype(int)
    dataset["label_high"] = dataset["label_high"].astype(int)

    return dataset


def _walk_forward_evaluate(
    dataset: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: Sequence[str],
) -> list[dict]:
    """시간 축을 따라 walk-forward 검증을 수행."""

    if dataset.empty:
        return []

    ordered = dataset.sort_values("날짜").reset_index(drop=True)
    unique_dates = ordered["날짜"].drop_duplicates().tolist()
    if len(unique_dates) < 80:
        return []

    min_train = max(int(len(unique_dates) * 0.4), 60)
    step = max(int(len(unique_dates) * 0.1), 20)

    metrics: list[dict] = []
    for split_idx in range(min_train, len(unique_dates) - 10, step):
        train_end = unique_dates[split_idx - 1]
        test_end_idx = min(split_idx + step, len(unique_dates))
        test_dates = unique_dates[split_idx:test_end_idx]
        if len(test_dates) < 5:
            continue

        train_mask = ordered["날짜"] <= train_end
        test_mask = ordered["날짜"].isin(test_dates)

        X_train = ordered.loc[train_mask, feature_cols]
        y_train = ordered.loc[train_mask, label_col]
        X_test = ordered.loc[test_mask, feature_cols]
        y_test = ordered.loc[test_mask, label_col]

        if y_train.sum() < 5 or y_test.sum() == len(y_test) or y_test.sum() == 0:
            continue

        pipe = _create_pipeline()
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        metrics.append(
            {
                "train_end": train_end,
                "test_end": test_dates[-1],
                "roc_auc": float(roc),
                "avg_precision": float(ap),
                "positives": int(y_test.sum()),
                "samples": int(len(y_test)),
            }
        )

    return metrics


def fit_extreme_model(
    dataset: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: Sequence[str] | None = None,
) -> ModelResult:
    """주어진 라벨에 대해 최종 모델을 학습하고 검증 통계를 함께 반환."""

    if dataset.empty:
        raise ValueError("Dataset is empty")

    feature_cols = [
        col for col in (feature_cols or EXTREME_MODEL_FEATURES) if col in dataset.columns
    ]
    if len(feature_cols) < 5:
        raise ValueError("Not enough feature columns available for training")

    model_data = dataset.dropna(subset=[label_col])
    if len(model_data) < EXTREME_MODEL_MIN_SAMPLES:
        raise ValueError(
            f"Insufficient samples for {label_col}: {len(model_data)} < {EXTREME_MODEL_MIN_SAMPLES}"
        )

    if model_data[label_col].sum() < 20:
        raise ValueError(f"Not enough positive samples for {label_col}")

    metrics = _walk_forward_evaluate(model_data, label_col=label_col, feature_cols=feature_cols)

    pipeline = _create_pipeline()
    pipeline.fit(model_data[feature_cols], model_data[label_col])

    return ModelResult(
        label=label_col,
        features=list(feature_cols),
        pipeline=pipeline,
        metrics=metrics,
    )


def _score_with_model(frame: pd.DataFrame, result: ModelResult) -> np.ndarray:
    """학습된 모델을 현재 스냅샷 데이터에 적용."""

    missing = [col for col in result.features if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features for scoring: {missing}")

    proba = result.pipeline.predict_proba(frame[result.features])[:, 1]
    return proba


def score_extremes_for_snapshot(
    tickers: Iterable[str],
    snapshot: pd.DataFrame,
    *,
    period: str = "3y",
) -> tuple[pd.DataFrame, dict]:
    """현재 스냅샷에 대해 저점/고점 확률 점수를 추가한다."""

    unique_tickers = sorted(set(ticker for ticker in tickers if ticker))
    if len(unique_tickers) < 5:
        return snapshot, {"global": {"status": "insufficient_tickers"}}

    price_df = fetch_ohlcv(unique_tickers, period=period)
    if isinstance(price_df.columns, pd.MultiIndex):
        symbols = price_df.columns.levels[0]
        price_map = {
            symbol: price_df[symbol].dropna(how="all") for symbol in symbols
        }
    else:
        price_map = {unique_tickers[0]: price_df.dropna(how="all")}

    dataset = build_labeled_dataset(price_map)
    if dataset.empty:
        return snapshot, {"global": {"status": "empty_dataset"}}

    metrics_info: dict[str, dict] = {}
    models: dict[str, ModelResult] = {}

    for label in ("label_low", "label_high"):
        try:
            result = fit_extreme_model(dataset, label_col=label)
        except ValueError as exc:
            metrics_info[label] = {"status": "failed", "reason": str(exc)}
            continue
        models[label] = result
        metrics_info[label] = {"status": "ok", "folds": result.metrics}

    if not models:
        return snapshot, metrics_info

    scored = snapshot.copy()
    if "label_low" in models:
        try:
            scored["저점확률"] = _score_with_model(scored, models["label_low"])
        except ValueError as exc:
            metrics_info["label_low"].update({"status": "score_failed", "reason": str(exc)})
    if "label_high" in models:
        try:
            scored["고점확률"] = _score_with_model(scored, models["label_high"])
        except ValueError as exc:
            metrics_info["label_high"].update({"status": "score_failed", "reason": str(exc)})

    if "저점확률" in scored.columns and "고점확률" in scored.columns:
        scored["극점편차"] = scored["저점확률"] - scored["고점확률"]

    return scored, metrics_info
