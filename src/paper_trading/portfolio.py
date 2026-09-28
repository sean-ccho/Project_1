"""JSON 기반 포지션 및 거래 히스토리 관리."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _default_data_dir() -> Path:
    """config 순환 임포트 방지용 — 런타임에 config에서 경로를 가져온다."""
    try:
        from screener.config import PAPER_TRADING_DATA_DIR
        return Path(PAPER_TRADING_DATA_DIR)
    except ImportError:
        return Path("data/paper_trading")


def _positions_path(data_dir: Path | None = None) -> Path:
    return (data_dir or _default_data_dir()) / "positions.json"


def _trades_path(data_dir: Path | None = None) -> Path:
    return (data_dir or _default_data_dir()) / "trades.json"


# ── Load / Save ──────────────────────────────────────────────


def load_positions(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """현재 보유 포지션 리스트를 JSON에서 로드."""
    path = _positions_path(data_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json_write(path: Path, data: list | dict) -> None:
    """JSON을 atomic write 방식으로 저장 (임시파일 → rename).

    1) 기존 파일이 있으면 .bak 백업 생성
    2) 같은 디렉토리에 임시파일 생성 → JSON 기록 → fsync
    3) os.replace로 원자적 교체 (중간에 죽어도 파일 손상 없음)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # 백업
    if path.exists():
        try:
            shutil.copy2(path, path.with_suffix(".json.bak"))
        except Exception as e:
            logger.warning(f"백업 생성 실패 ({path.name}): {e}")

    # Atomic write
    content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        # 실패 시 임시 파일 정리
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_positions(
    positions: list[dict[str, Any]], data_dir: Path | None = None
) -> None:
    """현재 보유 포지션 리스트를 JSON에 저장 (atomic write + 백업)."""
    path = _positions_path(data_dir)
    _atomic_json_write(path, positions)


def load_trades(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """전체 거래 히스토리 로드."""
    path = _trades_path(data_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_trades(
    trades: list[dict[str, Any]], data_dir: Path | None = None
) -> None:
    """전체 거래 히스토리 저장 (atomic write + 백업)."""
    path = _trades_path(data_dir)
    _atomic_json_write(path, trades)


# ── Position Operations ──────────���──────────────────────────


def add_position(
    positions: list[dict[str, Any]],
    *,
    ticker: str,
    entry_price: float,
    entry_date: str | date,
    strategy: str,
    star_rating: str,
    ccs_score: float,
    sector: str = "",
    entry_rsi: float = 0.0,
    predicted_upside_p25: float = 0.0,
    predicted_upside_p50: float = 0.0,
    predicted_upside_p75: float = 0.0,
    predicted_upside_p90: float = 0.0,
    upside_sample_size: int = 0,
    upside_days_to_peak: float = 0.0,
    upside_level: str = "",
    upside_bucket_key: str = "",
) -> dict[str, Any]:
    """매수 포지션 추가. 새 포지션 dict를 반환."""
    pos = {
        "ticker": ticker,
        "entry_price": entry_price,
        "entry_date": str(entry_date),
        "strategy": strategy,
        "star_rating": star_rating,
        "ccs_score": ccs_score,
        "sector": sector,
        "highest_price": entry_price,
        # Hold-Winners 재평가 상태
        "defer_count": 0,
        "trailing_stop_override": None,
        "last_defer_date": None,
        "defer_debug": None,
        # 매수 시점 스냅샷
        "entry_rsi": entry_rsi,
        # 상승 예측 분포
        "predicted_upside_p25": predicted_upside_p25,
        "predicted_upside_p50": predicted_upside_p50,
        "predicted_upside_p75": predicted_upside_p75,
        "predicted_upside_p90": predicted_upside_p90,
        "upside_sample_size": upside_sample_size,
        "upside_days_to_peak": upside_days_to_peak,
        "upside_level": upside_level,
        "upside_bucket_key": upside_bucket_key,
    }
    positions.append(pos)
    return pos


def close_position(
    positions: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    *,
    ticker: str,
    exit_price: float,
    exit_date: str | date,
    reason: str,
) -> dict[str, Any] | None:
    """포지션 청산. 거래 기록을 trades에 추가하고 positions에서 제거.
    해당 ticker 포지션이 없으면 None 반환."""
    idx = None
    for i, p in enumerate(positions):
        if p["ticker"] == ticker:
            idx = i
            break
    if idx is None:
        return None

    pos = positions.pop(idx)

    entry_price = pos["entry_price"]
    return_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0

    entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d") if isinstance(pos["entry_date"], str) else pos["entry_date"]
    exit_dt = datetime.strptime(str(exit_date), "%Y-%m-%d") if isinstance(exit_date, str) else exit_date
    holding_days = (exit_dt - entry_dt).days

    # 예측 vs 실제 비교 (ML 학습 데이터용)
    highest_price = pos.get("highest_price", entry_price)
    actual_max_upside_pct = (
        round((highest_price - entry_price) / entry_price, 4)
        if entry_price > 0 else 0.0
    )
    p50_pred = pos.get("predicted_upside_p50", 0.0)
    p75_pred = pos.get("predicted_upside_p75", 0.0)

    trade = {
        "ticker": ticker,
        "entry_date": pos["entry_date"],
        "exit_date": str(exit_date),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return_pct": round(return_pct, 4),
        "holding_days": holding_days,
        "highest_price": highest_price,
        "strategy": pos.get("strategy", ""),
        "star_rating": pos.get("star_rating", ""),
        "ccs_score": pos.get("ccs_score", 0.0),
        "sector": pos.get("sector", ""),
        "exit_reason": reason,
        # Hold-Winners 분석용
        "defer_count": pos.get("defer_count", 0),
        "entry_rsi": pos.get("entry_rsi", 0.0),
        "predicted_upside_p50": p50_pred,
        "predicted_upside_p75": p75_pred,
        "upside_sample_size": pos.get("upside_sample_size", 0),
        # 예측 검증 (ML 학습 데이터)
        "actual_max_upside_pct": actual_max_upside_pct,
        "upside_level": pos.get("upside_level", ""),
        "upside_bucket_key": pos.get("upside_bucket_key", ""),
        "hit_p50": (
            1 if p50_pred > 0 and actual_max_upside_pct >= p50_pred else
            -1 if p50_pred > 0 else 0
        ),
        "hit_p75": (
            1 if p75_pred > 0 and actual_max_upside_pct >= p75_pred else
            -1 if p75_pred > 0 else 0
        ),
    }
    trades.append(trade)
    return trade


def update_highest_price(
    positions: list[dict[str, Any]], prices: dict[str, float]
) -> None:
    """현재가로 각 포지션의 고점(highest_price)을 업데이트."""
    for pos in positions:
        ticker = pos["ticker"]
        if ticker in prices:
            current = prices[ticker]
            if current > pos.get("highest_price", 0):
                pos["highest_price"] = current


def get_worst_position(
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    ranked_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """수익률 + 최신 스크리너 점수를 종합해서 가장 약한 포지션 반환.

    weakness = -return_pct_norm + (1 - latest_score_norm)
    둘 다 [0,1] 정규화. 값이 클수록 약한 포지션.
    """
    if not positions:
        return None

    scored: list[tuple[float, dict]] = []
    for pos in positions:
        ticker = pos["ticker"]
        entry_price = pos["entry_price"]
        current_price = prices.get(ticker, entry_price)
        ret = (current_price - entry_price) / entry_price if entry_price else 0.0

        latest_score = 0.0
        if ranked_df is not None and "티커" in ranked_df.columns:
            match = ranked_df[ranked_df["티커"] == ticker]
            if not match.empty:
                latest_score = float(
                    match.iloc[0].get("매수적합도", 0.0) or 0.0
                )

        # weakness: 수익률 낮을수록 + 점수 낮을수록 = 더 약함
        # return: -0.07 ~ +0.15 범위 가정 → 정규화
        ret_norm = max(0.0, min(1.0, (ret + 0.10) / 0.25))
        score_norm = min(latest_score / 10.0, 1.0)
        weakness = (1.0 - ret_norm) * 0.6 + (1.0 - score_norm) * 0.4

        scored.append((weakness, pos))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def get_position_return(pos: dict[str, Any], current_price: float) -> float:
    """포지션의 현재 수익률 계산."""
    entry = pos["entry_price"]
    return (current_price - entry) / entry if entry else 0.0


def get_holding_days(pos: dict[str, Any], as_of: str | date | None = None) -> int:
    """포지션의 보유일수 계산."""
    entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
    if as_of is None:
        as_of = date.today()
    elif isinstance(as_of, str):
        as_of = datetime.strptime(as_of, "%Y-%m-%d").date()
    return (as_of - entry_dt).days
