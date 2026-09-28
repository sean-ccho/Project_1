"""백테스트 거래 데이터를 이용한 Upside 모델 LOO 교차검증.

Leave-One-Out: 각 거래를 제외한 나머지로 분포 재계산 → 예측 → 실제와 비교.
결과를 data/paper_trading/backtest_validation.json에 저장.

사용법:
    python scripts/validate_upside_model.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from paper_trading.upside_model import (
    _bucket_key,
    _build_bucket_stats,
    _compute_max_drawup,
    _fetch_ohlc_for_tickers,
    _load_recent_trades,
    _strategy_key,
)
from screener.config import UPSIDE_MODEL_MIN_SAMPLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_RUNS_DIR = _PROJECT_ROOT / "output" / "runs"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "paper_trading" / "backtest_validation.json"


def _predict_from_stats(
    buckets: dict, strategy: str, ccs: float, star_rating: str,
) -> dict:
    """분포 dict에서 예측값 조회 (predict_upside와 동일 로직, 캐시 안 씀)."""
    key = _bucket_key(strategy, ccs, star_rating)
    result = buckets.get(key)
    if result is None:
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
        "level": result.get("level", "unknown"),
        "bucket_key": key,
    }


def run_loo_validation(max_runs: int = 3, period: str = "5y") -> dict:
    """LOO 교차검증 실행. 결과 dict 반환."""
    # 1. 거래 로드
    trades = _load_recent_trades(_DEFAULT_RUNS_DIR, max_runs=max_runs)
    if not trades:
        logger.error("No trades found")
        return {}

    # 2. OHLC 데이터 가져오기
    tickers = list({t["ticker"] for t in trades if t.get("ticker")})
    ohlc_cache = _fetch_ohlc_for_tickers(tickers, period=period)

    # 3. 각 거래의 max_drawup 계산
    enriched: list[dict] = []
    for t in trades:
        drawup = _compute_max_drawup(
            t["ticker"], t["entry_date"], t["exit_date"],
            float(t["entry_price"]), ohlc_cache,
        )
        if drawup is None:
            continue
        max_drawup, days_to_peak = drawup
        key = _bucket_key(t["strategy"], float(t["ccs_score"]), t["star_rating"])
        enriched.append({
            **t,
            "max_drawup": max_drawup,
            "days_to_peak": days_to_peak,
            "bucket_key": key,
        })

    logger.info(f"Enriched {len(enriched)} trades with max_drawup")

    # 4. LOO 교차검증
    results: list[dict] = []
    for i, held_out in enumerate(enriched):
        # 이 거래를 제외한 나머지로 분포 재구축
        remaining = [
            (e["bucket_key"], e["max_drawup"], e["days_to_peak"])
            for j, e in enumerate(enriched) if j != i
        ]
        stats = _build_bucket_stats(remaining)

        # 제외된 거래에 대해 예측
        pred = _predict_from_stats(
            stats, held_out["strategy"],
            float(held_out["ccs_score"]), held_out["star_rating"],
        )

        if pred.get("n", 0) == 0:
            continue

        actual = held_out["max_drawup"]
        p50_pred = pred["p50"]
        p75_pred = pred["p75"]

        results.append({
            "ticker": held_out["ticker"],
            "entry_date": held_out["entry_date"],
            "exit_date": held_out["exit_date"],
            "strategy": held_out["strategy"],
            "ccs_score": held_out["ccs_score"],
            "star_rating": held_out["star_rating"],
            "sector": held_out.get("sector", ""),
            "entry_price": held_out["entry_price"],
            "exit_price": held_out["exit_price"],
            "return_pct": held_out.get("return_pct", 0.0),
            "holding_days": held_out.get("holding_days", 0),
            "actual_max_upside_pct": round(actual, 4),
            "predicted_upside_p50": round(p50_pred, 4),
            "predicted_upside_p75": round(p75_pred, 4),
            "upside_bucket_key": pred["bucket_key"],
            "upside_level": pred["level"],
            "upside_n": pred["n"],
            "hit_p50": 1 if actual >= p50_pred else -1,
            "hit_p75": 1 if actual >= p75_pred else -1,
            "source": "backtest_cv",
        })

        if (i + 1) % 20 == 0:
            logger.info(f"  LOO progress: {i + 1}/{len(enriched)}")

    # 5. 집계
    n = len(results)
    p50_hits = sum(1 for r in results if r["hit_p50"] == 1)
    p75_hits = sum(1 for r in results if r["hit_p75"] == 1)
    avg_pred_p50 = np.mean([r["predicted_upside_p50"] for r in results]) if results else 0
    avg_actual = np.mean([r["actual_max_upside_pct"] for r in results]) if results else 0

    summary = {
        "n_trades": n,
        "p50_hit_rate": round(p50_hits / n, 4) if n else 0,
        "p75_hit_rate": round(p75_hits / n, 4) if n else 0,
        "p50_hits": p50_hits,
        "p75_hits": p75_hits,
        "avg_predicted_p50": round(float(avg_pred_p50), 4),
        "avg_actual_max_upside": round(float(avg_actual), 4),
    }

    payload = {
        "validation_type": "leave_one_out",
        "n_source_trades": len(enriched),
        "summary": summary,
        "trades": results,
    }

    logger.info(f"\n=== LOO Cross-Validation Results ===")
    logger.info(f"  Total trades: {n}")
    logger.info(f"  P50 hit rate: {p50_hits}/{n} ({summary['p50_hit_rate']:.0%})")
    logger.info(f"  P75 hit rate: {p75_hits}/{n} ({summary['p75_hit_rate']:.0%})")
    logger.info(f"  Avg predicted P50: {summary['avg_predicted_p50']:+.1%}")
    logger.info(f"  Avg actual max upside: {summary['avg_actual_max_upside']:+.1%}")

    return payload


def main() -> None:
    payload = run_loo_validation()
    if not payload:
        logger.error("Validation failed")
        return

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
