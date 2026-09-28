"""Walk-forward CV 성능 지표 계산 헬퍼."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_cv(
    model_cls: type,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_kwargs: dict | None = None,
) -> list[dict]:
    """TimeSeriesSplit walk-forward CV → fold별 성능 지표 반환."""
    model_kwargs = model_kwargs or {}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(y_train) < 20 or y_train.nunique() < 2:
            continue

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # precision@top20%
        n_top = max(1, int(len(proba) * 0.2))
        top_idx = np.argsort(proba)[::-1][:n_top]
        top_precision = y_test.iloc[top_idx].mean()

        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")

        results.append({
            "fold": fold + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "auc": auc,
            "accuracy": accuracy_score(y_test, pred),
            "precision_top20": top_precision,
            "baseline_win_rate": y_test.mean(),
        })

    return results


def print_cv_report(fold_results: list[dict], target: str = "") -> None:
    """CV 결과 콘솔 출력."""
    if not fold_results:
        print("  [CV] No results")
        return

    header = f"  [CV] {target}" if target else "  [CV]"
    print(header)
    print(f"  {'Fold':>4}  {'Train':>6}  {'Test':>5}  {'AUC':>6}  {'Acc':>6}  {'P@20%':>6}  {'Base':>6}")
    aucs = []
    for r in fold_results:
        auc_str = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "  nan"
        print(f"  {r['fold']:>4}  {r['train_size']:>6}  {r['test_size']:>5}  {auc_str}  "
              f"{r['accuracy']:.3f}  {r['precision_top20']:.3f}  {r['baseline_win_rate']:.3f}")
        if not np.isnan(r["auc"]):
            aucs.append(r["auc"])

    if aucs:
        mean_auc = np.mean(aucs)
        flag = "✓" if mean_auc >= 0.55 else "✗"
        print(f"  Mean AUC: {mean_auc:.3f} {flag} (threshold 0.55)")
