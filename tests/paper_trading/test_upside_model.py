"""Upside 예측 모델 유닛 테스트.

테스트 대상:
- _ccs_bucket, _star_bucket, _bucket_key: 버킷 키 생성
- _build_bucket_stats: 폴백 계층 (bucket → strategy → global)
- predict_upside: 캐시 조회 + 폴백 순서
- invalidate_cache: 캐시 리셋
"""
import json
import tempfile
from pathlib import Path

import pytest

from paper_trading.upside_model import (
    _bucket_key,
    _build_bucket_stats,
    _ccs_bucket,
    _star_bucket,
    _strategy_key,
    invalidate_cache,
    predict_upside,
)
from screener.config import UPSIDE_MODEL_MIN_SAMPLES


# ── _ccs_bucket ───────────────────────────────────────────────────────────────

class TestCcsBucket:
    def test_below_first_boundary(self):
        result = _ccs_bucket(0.38)
        assert result.startswith("<")

    def test_exactly_on_boundary_goes_to_next(self):
        # 0.45 → "<0.45" 아니고 "0.45-0.50" (i+1 버킷)
        result = _ccs_bucket(0.45)
        assert "0.45" in result

    def test_middle_range(self):
        result = _ccs_bucket(0.52)
        assert "0.50" in result and "0.55" in result

    def test_above_last_boundary(self):
        result = _ccs_bucket(0.70)
        assert result.endswith("+")


# ── _star_bucket ──────────────────────────────────────────────────────────────

class TestStarBucket:
    def test_five_star_high_score(self):
        result = _star_bucket("★★★★★ (9.5)")
        assert "9+" in result

    def test_five_star_mid_score(self):
        result = _star_bucket("★★★★★ (8.0)")
        assert "7-9" in result

    def test_five_star_low_score(self):
        result = _star_bucket("★★★★★ (5.0)")
        assert "<7" in result

    def test_four_star(self):
        result = _star_bucket("★★★★ (7.0)")
        assert "4★" in result

    def test_three_star(self):
        result = _star_bucket("★★★ (6.0)")
        assert "이하" in result


# ── _build_bucket_stats ───────────────────────────────────────────────────────

class TestBuildBucketStats:
    def _make_records(self, key: str, n: int, drawup: float = 0.10) -> list:
        return [(key, drawup, 5.0)] * n

    def test_bucket_level_when_enough_samples(self):
        key = "모멘텀|0.50-0.55|5★(9+)"
        records = self._make_records(key, UPSIDE_MODEL_MIN_SAMPLES)
        stats = _build_bucket_stats(records)
        assert stats[key]["level"] == "bucket"
        assert stats[key]["n"] == UPSIDE_MODEL_MIN_SAMPLES

    def test_strategy_fallback_when_bucket_sparse(self):
        """버킷 샘플 < MIN이지만 strategy 수준은 충분."""
        strategy = "모멘텀"
        # 같은 strategy, 다른 버킷 → strategy 수준 충분
        base_key = f"{strategy}|0.50-0.55|5★(9+)"
        other_key = f"{strategy}|0.45-0.50|5★(9+)"
        records = (
            self._make_records(base_key, 2) +  # 버킷 부족
            self._make_records(other_key, UPSIDE_MODEL_MIN_SAMPLES)
        )
        stats = _build_bucket_stats(records)
        assert f"strategy:{strategy}" in stats[base_key]["level"]

    def test_global_fallback_when_strategy_sparse(self):
        """버킷도, strategy도 샘플 부족 → global."""
        key = "모멘텀|0.50-0.55|5★(9+)"
        other_key = "바닥반등|0.40-0.45|5★(<7)"
        records = (
            self._make_records(key, 2) +
            self._make_records(other_key, UPSIDE_MODEL_MIN_SAMPLES)
        )
        stats = _build_bucket_stats(records)
        assert stats[key]["level"] == "global"

    def test_global_key_always_present(self):
        key = "모멘텀|0.50-0.55|5★(9+)"
        records = self._make_records(key, UPSIDE_MODEL_MIN_SAMPLES)
        stats = _build_bucket_stats(records)
        assert "__global__" in stats

    def test_percentile_ordering(self):
        """P25 <= P50 <= P75 <= P90."""
        key = "모멘텀|0.50-0.55|5★(9+)"
        # 다양한 값으로 분포 생성
        records = [(key, 0.05 * (i + 1), 5.0) for i in range(UPSIDE_MODEL_MIN_SAMPLES)]
        stats = _build_bucket_stats(records)
        s = stats[key]
        assert s["p25"] <= s["p50"] <= s["p75"] <= s["p90"]

    def test_mean_days_to_peak(self):
        key = "모멘텀|0.50-0.55|5★(9+)"
        records = [(key, 0.10, float(d)) for d in range(1, UPSIDE_MODEL_MIN_SAMPLES + 1)]
        stats = _build_bucket_stats(records)
        expected_mean = sum(range(1, UPSIDE_MODEL_MIN_SAMPLES + 1)) / UPSIDE_MODEL_MIN_SAMPLES
        assert abs(stats[key]["mean_days_to_peak"] - expected_mean) < 0.01


# ── predict_upside ────────────────────────────────────────────────────────────

class TestPredictUpside:
    def _write_cache(self, buckets: dict) -> Path:
        """임시 캐시 파일 작성 후 Path 반환."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        )
        json.dump({"built_at": "2026-01-01", "n_trades": 10, "n_buckets": len(buckets), "buckets": buckets}, tmp)
        tmp.close()
        invalidate_cache()
        return Path(tmp.name)

    def _sample_stats(self, level: str = "bucket") -> dict:
        return {"p25": 0.05, "p50": 0.10, "p75": 0.18, "p90": 0.30,
                "n": 20, "mean_days_to_peak": 7.0, "level": level}

    def test_exact_bucket_hit(self):
        strategy = "모멘텀"
        ccs = 0.52
        star = "★★★★★ (9.5)"
        key = _bucket_key(strategy, ccs, star)
        buckets = {key: self._sample_stats()}
        path = self._write_cache(buckets)
        result = predict_upside(strategy, ccs, star, cache_path=path)
        assert result["p50"] == pytest.approx(0.10)
        assert result["level"] == "bucket"
        assert result["n"] == 20

    def test_strategy_fallback(self):
        """버킷 키 없음 → strategy 폴백."""
        strategy = "모멘텀"
        ccs = 0.52
        star = "★★★★★ (9.5)"
        strat_key = f"__strategy__|{_strategy_key(strategy)}"
        buckets = {strat_key: self._sample_stats(level="strategy:모멘텀")}
        path = self._write_cache(buckets)
        result = predict_upside(strategy, ccs, star, cache_path=path)
        assert "strategy" in result["level"]

    def test_global_fallback(self):
        """버킷도 strategy도 없음 → global."""
        buckets = {"__global__": self._sample_stats(level="global")}
        path = self._write_cache(buckets)
        result = predict_upside("없는전략", 0.99, "★ (1.0)", cache_path=path)
        assert result["level"] == "global"

    def test_empty_cache_returns_n0(self):
        """캐시 파일 없음 → n=0."""
        invalidate_cache()
        result = predict_upside("모멘텀", 0.52, "★★★★★ (9.5)",
                                cache_path=Path("/nonexistent/path.json"))
        assert result.get("n", 0) == 0

    def test_returns_bucket_key(self):
        strategy = "모멘텀"
        ccs = 0.52
        star = "★★★★★ (9.5)"
        key = _bucket_key(strategy, ccs, star)
        buckets = {key: self._sample_stats()}
        path = self._write_cache(buckets)
        result = predict_upside(strategy, ccs, star, cache_path=path)
        assert "bucket_key" in result
        assert result["bucket_key"] == key
