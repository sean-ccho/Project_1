#!/usr/bin/env python3
"""Tier 3-B 연구 공통 함수: 패널·OHLCV 로드(개발 구간만), Newey-West t값, 다중검정 임계, 시도 기록.

홀드아웃(2025-10-01~) 보호: 연구 스크립트는 패널·가격을 반드시 `load_panel` / `load_ohlcv`로 읽는다.
(`data/cache/ohlcv_e7842ef0a144.parquet`는 2026-10까지 들어 있어 그냥 읽으면 홀드아웃이 섞인다.)
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "data" / "research"
TIER3_DIR = ROOT / "docs" / "quant_improvement" / "tier3"
TRIAL_LOG = TIER3_DIR / "TRIAL_LOG.csv"
DEV_END = pd.Timestamp("2025-09-30")
COST_PER_SIDE = 0.001  # screener.config.BACKTEST_COST_PER_SIDE 와 같게
ROUND_TRIP = 2 * COST_PER_SIDE

TRIAL_FIELDS = ["실행시각", "step", "family", "n_family", "test_id", "t", "판정", "메모"]


def load_panel(name: str) -> pd.DataFrame:
    """data/research/panel_{name}.parquet 을 개발 구간(≤ DEV_END)만 읽는다."""
    panel = pd.read_parquet(RESEARCH_DIR / f"panel_{name}.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    return panel[panel["date"] <= DEV_END].reset_index(drop=True)


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """OHLCV 캐시(parquet, 컬럼 MultiIndex)를 개발 구간(≤ DEV_END)만 읽는다."""
    p = Path(path)
    ohlcv = pd.read_parquet(p if p.is_absolute() else ROOT / p)
    ohlcv.index = pd.to_datetime(ohlcv.index)
    return ohlcv.sort_index().loc[:DEV_END]


def nw_tstat(x: pd.Series | np.ndarray, lag: int) -> float:
    """평균의 Newey-West(Bartlett) t값. h일 겹치는 수익률이면 lag ≥ h-1."""
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = v.size
    if n < max(20, 2 * lag + 2):
        return float("nan")
    e = v - v.mean()
    s = float(e @ e) / n
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1)) * float(e[k:] @ e[:-k]) / n
    return float(v.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def bonferroni_z(n_tests: int, alpha: float = 0.05) -> float:
    """양측 alpha / n_tests 에 해당하는 |t| 임계."""
    return float(norm.isf(alpha / max(1, n_tests) / 2))


def yearly_mean(s: pd.Series) -> pd.Series:
    """날짜 인덱스 시계열 → 연도별 평균."""
    return s.groupby(pd.DatetimeIndex(s.index).year).mean()


def halves_mean(s: pd.Series) -> tuple[float, float]:
    """날짜 순으로 반을 나눈 앞·뒤 평균."""
    s = s.dropna().sort_index()
    mid = len(s) // 2
    return float(s.iloc[:mid].mean()), float(s.iloc[mid:].mean())


def log_trials(step: str, family: str, n_family: int, rows: list[dict]) -> None:
    """검정 등록·결과를 TRIAL_LOG.csv 에 덧붙인다 (git에 커밋)."""
    new = not TRIAL_LOG.exists()
    TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRIAL_LOG.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        if new:
            w.writeheader()
        now = f"{datetime.now():%Y-%m-%d %H:%M}"
        for r in rows:
            w.writerow({"실행시각": now, "step": step, "family": family, "n_family": n_family,
                        "test_id": r.get("test_id", ""), "t": r.get("t", ""),
                        "판정": r.get("판정", ""), "메모": r.get("메모", "")})


def total_trials() -> int:
    """TRIAL_LOG 의 family별 n_family 합 (같은 family는 한 번만)."""
    if not TRIAL_LOG.exists():
        return 0
    df = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return int(df.drop_duplicates("family")["n_family"].sum())
