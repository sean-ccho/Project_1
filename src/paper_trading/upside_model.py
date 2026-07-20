"""상승 예측 모델 (경험적 분포).

과거 백테스트 거래 데이터에서 전략/CCS/★ 버킷별 P25/P50/P75/P90 분포를 빌드.
매수 시점에 유사 거래의 과거 max_drawup 분포를 조회해서 예상 목표가 제공.

사용법:
    # 1. 캐시 빌드 (한 번만)
    python -m paper_trading.upside_model

    # 2. 런타임 조회
    from paper_trading.upside_model import predict_upside
    dist = predict_upside(strategy="📈 모멘텀", ccs_score=0.53, star_rating="★★★★★ (10.0)")
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screener.config import (
    UPSIDE_MODEL_CCS_BUCKETS,
    UPSIDE_MODEL_CACHE_PATH,
    UPSIDE_MODEL_MIN_SAMPLES,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS_DIR = _PROJECT_ROOT / "output" / "runs"
_DEFAULT_CACHE_PATH = _PROJECT_ROOT / UPSIDE_MODEL_CACHE_PATH

_STAR_PATTERN = re.compile(r"\(([\d.]+)\)")

_PERCENTILES = [25, 50, 75, 90]


def _ccs_bucket(ccs: float) -> str:
    """CCS 점수를 버킷 레이블로 변환."""
    boundaries = UPSIDE_MODEL_CCS_BUCKETS  # [0.40, 0.45, 0.50, 0.55]
    for i, edge in enumerate(boundaries):
        if ccs < edge:
            if i == 0:
                return f"<{edge:.2f}"
            return f"{boundaries[i-1]:.2f}-{edge:.2f}"
    return f"{boundaries[-1]:.2f}+"


def _star_bucket(star_rating: str) -> str:
    """'★★★★★ (10.0)' → '5★(9+)' 같은 버킷 레이블로 변환."""
    star_count = star_rating.count("★")
    if star_count < 5:
        return f"{star_count}★이하"
    # 5★는 세부 점수로 나눔
    m = _STAR_PATTERN.search(star_rating)
    if not m:
        return "5★(기본)"
    score = float(m.group(1))
    if score >= 9.0:
        return "5★(9+)"
    if score >= 7.0:
        return "5★(7-9)"
    return "5★(<7)"


def _strategy_key(strategy: str) -> str:
    """전략명에서 이모지 제거."""
    return re.sub(r"[^\w가-힣]", "", strategy).strip()


def _bucket_key(strategy: str, ccs: float, star_rating: str) -> str:
    return f"{_strategy_key(strategy)}|{_ccs_bucket(ccs)}|{_star_bucket(star_rating)}"


def _load_recent_trades(runs_dir: Path, max_runs: int = 3) -> list[dict[str, Any]]:
    """최근 N개 run의 거래 데이터를 로드하고 (ticker, entry_date) 기준 dedup."""
    run_files = sorted(runs_dir.glob("*/backtest_trades_enhanced.json"), reverse=True)
    if not run_files:
        logger.warning(f"No backtest trade files found in {runs_dir}")
        return []

    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for rf in run_files[:max_runs]:
        try:
            with rf.open() as f:
                trades = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load {rf}: {exc}")
            continue
        for t in trades:
            key = (t.get("ticker", ""), t.get("entry_date", ""))
            if key in seen or not key[0] or not key[1]:
                continue
            seen.add(key)
            merged.append(t)
    logger.info(f"Loaded {len(merged)} unique trades from {min(max_runs, len(run_files))} runs")
    return merged


def _compute_max_drawup(
    ticker: str,
    entry_date: str,
    exit_date: str,
    entry_price: float,
    ohlc_cache: dict[str, pd.DataFrame],
) -> tuple[float, int] | None:
    """보유 기간 동안의 최대 상승률 + 도달까지 걸린 일수 계산.

    Returns (max_drawup_pct, days_to_peak) or None if OHLC unavailable.
    """
    if ticker not in ohlc_cache:
        return None
    df = ohlc_cache[ticker]
    try:
        entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
        exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
    except ValueError:
        return None

    mask = (df.index >= entry_dt) & (df.index <= exit_dt)
    window = df.loc[mask]
    if window.empty or "High" not in window.columns:
        return None

    highs = window["High"].dropna()
    if highs.empty or entry_price <= 0:
        return None

    peak_idx = highs.idxmax()
    max_high = float(highs.loc[peak_idx])
    max_drawup = (max_high - entry_price) / entry_price
    days_to_peak = max(0, (peak_idx - entry_dt).days)
    return max_drawup, days_to_peak


def _fetch_ohlc_for_tickers(
    tickers: list[str],
    period: str = "2y",
) -> dict[str, pd.DataFrame]:
    """거래 기간을 커버하는 OHLC를 배치로 받아 ticker별 dict 반환."""
    from data.fetch import fetch_ohlcv

    tickers = sorted(set(tickers))
    if not tickers:
        return {}
    logger.info(f"Fetching OHLC for {len(tickers)} tickers (period={period})...")
    raw = fetch_ohlcv(tickers, period=period)

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            if tk not in raw.columns.get_level_values(0):
                continue
            sub = raw[tk].dropna(how="all")
            if not sub.empty:
                result[tk] = sub
    else:
        # 단일 티커 경우
        if tickers:
            result[tickers[0]] = raw.dropna(how="all")
    logger.info(f"Fetched OHLC for {len(result)}/{len(tickers)} tickers")
    return result


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in _PERCENTILES}


def _build_bucket_stats(
    records: list[tuple[str, float, float]],
    # records: list of (bucket_key, max_drawup, days_to_peak)
) -> dict[str, dict[str, float | int]]:
    """버킷별 통계 + 폴백 계층(strategy / global) 구축."""
    by_bucket: dict[str, list[tuple[float, float]]] = {}
    by_strategy: dict[str, list[tuple[float, float]]] = {}
    all_records: list[tuple[float, float]] = []

    for key, drawup, days in records:
        by_bucket.setdefault(key, []).append((drawup, days))
        strat = key.split("|", 1)[0]
        by_strategy.setdefault(strat, []).append((drawup, days))
        all_records.append((drawup, days))

    def _stats(entries: list[tuple[float, float]], level: str) -> dict[str, Any]:
        drawups = [d for d, _ in entries]
        days = [dy for _, dy in entries]
        out: dict[str, Any] = _percentiles(drawups)
        out["n"] = len(drawups)
        out["mean_days_to_peak"] = float(np.mean(days)) if days else 0.0
        out["level"] = level
        return out

    result: dict[str, dict[str, float | int]] = {}
    for key, entries in by_bucket.items():
        if len(entries) >= UPSIDE_MODEL_MIN_SAMPLES:
            result[key] = _stats(entries, "bucket")
        else:
            # strategy 레벨 폴백
            strat = key.split("|", 1)[0]
            if len(by_strategy.get(strat, [])) >= UPSIDE_MODEL_MIN_SAMPLES:
                result[key] = _stats(by_strategy[strat], f"strategy:{strat}")
            else:
                result[key] = _stats(all_records, "global")

    # 폴백 전용 키도 저장 (매수 시 버킷이 아예 없을 때)
    for strat, entries in by_strategy.items():
        fkey = f"__strategy__|{strat}"
        if len(entries) >= UPSIDE_MODEL_MIN_SAMPLES:
            result[fkey] = _stats(entries, f"strategy:{strat}")
    if all_records:
        result["__global__"] = _stats(all_records, "global")

    return result


def build_upside_distribution(
    runs_dir: Path | None = None,
    output_path: Path | None = None,
    max_runs: int = 3,
    period: str = "2y",
) -> dict[str, Any]:
    """경험적 상승 분포 빌드 → JSON 캐시 저장.

    Returns: 빌드된 분포 dict.
    """
    runs_dir = runs_dir or _DEFAULT_RUNS_DIR
    output_path = output_path or _DEFAULT_CACHE_PATH

    trades = _load_recent_trades(runs_dir, max_runs=max_runs)
    if not trades:
        logger.error("No trades to build distribution")
        return {}

    tickers = list({t["ticker"] for t in trades if t.get("ticker")})
    ohlc_cache = _fetch_ohlc_for_tickers(tickers, period=period)

    records: list[tuple[str, float, float]] = []
    skipped = 0
    for t in trades:
        drawup = _compute_max_drawup(
            t["ticker"], t["entry_date"], t["exit_date"],
            float(t["entry_price"]), ohlc_cache,
        )
        if drawup is None:
            skipped += 1
            continue
        max_drawup, days_to_peak = drawup
        key = _bucket_key(t["strategy"], float(t["ccs_score"]), t["star_rating"])
        records.append((key, max_drawup, days_to_peak))

    logger.info(f"Processed {len(records)} trades (skipped {skipped} with no OHLC)")
    stats = _build_bucket_stats(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "built_at": datetime.now().isoformat(),
        "n_trades": len(records),
        "n_buckets": len(stats),
        "buckets": stats,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved upside distribution → {output_path}")
    return payload


_cache: dict[str, Any] | None = None


def _load_cache(path: Path | None = None) -> dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    path = path or _DEFAULT_CACHE_PATH
    if not path.exists():
        _cache = {}
        return _cache
    try:
        with path.open(encoding="utf-8") as f:
            _cache = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to load upside cache: {exc}")
        _cache = {}
    return _cache


def predict_upside(
    strategy: str,
    ccs_score: float,
    star_rating: str,
    cache_path: Path | None = None,
) -> dict[str, float | int | str]:
    """매수 시점 상승 분포 조회.

    Returns: {p25, p50, p75, p90, n, mean_days_to_peak, level, bucket_key}
    캐시 없거나 데이터 부족 시 빈 dict 반환 (n=0).
    """
    payload = _load_cache(cache_path)
    buckets: dict[str, dict[str, Any]] = payload.get("buckets", {})
    if not buckets:
        return {"n": 0}

    key = _bucket_key(strategy, ccs_score, star_rating)
    result = buckets.get(key)
    if result is None:
        # strategy 폴백
        strat_key = f"__strategy__|{_strategy_key(strategy)}"
        result = buckets.get(strat_key)
    if result is None:
        result = buckets.get("__global__")
    if result is None:
        return {"n": 0}

    return {
        "p25": result.get("p25", 0.0),
        "p50": result.get("p50", 0.0),
        "p75": result.get("p75", 0.0),
        "p90": result.get("p90", 0.0),
        "n": result.get("n", 0),
        "mean_days_to_peak": result.get("mean_days_to_peak", 0.0),
        "level": result.get("level", "unknown"),
        "bucket_key": key,
    }


def invalidate_cache() -> None:
    """테스트/재빌드 후 강제 리로드용."""
    global _cache
    _cache = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = build_upside_distribution()
    print(json.dumps({k: v for k, v in result.items() if k != "buckets"}, indent=2, ensure_ascii=False))
    print(f"\nBuckets: {len(result.get('buckets', {}))}")
    for key in list(result.get("buckets", {}).keys())[:10]:
        bucket = result["buckets"][key]
        print(f"  {key}: n={bucket.get('n')} p50={bucket.get('p50', 0):.2%} p75={bucket.get('p75', 0):.2%}")
