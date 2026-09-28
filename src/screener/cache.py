"""백테스트용 feature 스냅샷 디스크 캐시 유틸리티.

구조:
  data/cache/features/{run_hash}/
      meta.json          — 캐시 메타데이터 (OHLCV 해시, feature 버전, period 등)
      YYYY-MM-DD.parquet — 날짜별 feature DataFrame (compute_features_snapshot 결과)
  data/cache/ic_weights/
      {ohlcv_hash}_{cycle}.json — IC 가중치 (20 거래일 주기)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

# feature 계산 로직이 변경될 때 이 버전을 올리면 기존 캐시가 자동 무효화된다.
FEATURE_VERSION = "v1"

_CACHE_ROOT = Path("data/cache")
_FEATURE_CACHE_BASE = _CACHE_ROOT / "features"
_IC_CACHE_DIR = _CACHE_ROOT / "ic_weights"


# ── 캐시 디렉토리 경로 ──────────────────────────────────────────────────────────

def feature_cache_dir(ohlcv_hash: str) -> Path:
    run_key = f"{ohlcv_hash}_{FEATURE_VERSION}"
    return _FEATURE_CACHE_BASE / run_key


def _meta_path(ohlcv_hash: str) -> Path:
    return feature_cache_dir(ohlcv_hash) / "meta.json"


# ── 메타데이터 ──────────────────────────────────────────────────────────────────

def write_cache_meta(ohlcv_hash: str, period: str, include_fundamentals: bool) -> None:
    meta = {
        "ohlcv_hash": ohlcv_hash,
        "feature_version": FEATURE_VERSION,
        "period": period,
        "include_fundamentals": include_fundamentals,
    }
    path = _meta_path(ohlcv_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def read_cache_meta(ohlcv_hash: str) -> Optional[dict]:
    path = _meta_path(ohlcv_hash)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def cache_meta_valid(ohlcv_hash: str, period: str, include_fundamentals: bool) -> bool:
    """저장된 메타데이터가 현재 실행 파라미터와 일치하는지 확인."""
    meta = read_cache_meta(ohlcv_hash)
    if meta is None:
        return False
    return (
        meta.get("ohlcv_hash") == ohlcv_hash
        and meta.get("feature_version") == FEATURE_VERSION
        and meta.get("period") == period
        and meta.get("include_fundamentals") == include_fundamentals
    )


# ── Feature 스냅샷 저장/로드 ────────────────────────────────────────────────────

def _snapshot_path(ohlcv_hash: str, cutoff: pd.Timestamp) -> Path:
    date_str = cutoff.strftime("%Y-%m-%d")
    return feature_cache_dir(ohlcv_hash) / f"{date_str}.parquet"


def save_feature_snapshot(ohlcv_hash: str, cutoff: pd.Timestamp, df: pd.DataFrame) -> None:
    path = _snapshot_path(ohlcv_hash, cutoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_feature_snapshot(ohlcv_hash: str, cutoff: pd.Timestamp) -> Optional[pd.DataFrame]:
    path = _snapshot_path(ohlcv_hash, cutoff)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def count_cached_snapshots(ohlcv_hash: str) -> int:
    cache_dir = feature_cache_dir(ohlcv_hash)
    if not cache_dir.exists():
        return 0
    return len(list(cache_dir.glob("????-??-??.parquet")))


# ── IC weights 저장/로드 ────────────────────────────────────────────────────────

def _ic_path(ohlcv_hash: str, cycle: int) -> Path:
    return _IC_CACHE_DIR / f"{ohlcv_hash}_{cycle}.json"


def save_ic_weights(ohlcv_hash: str, cycle: int, weights: dict) -> None:
    path = _ic_path(ohlcv_hash, cycle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights))


def load_ic_weights(ohlcv_hash: str, cycle: int) -> Optional[dict]:
    path = _ic_path(ohlcv_hash, cycle)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
