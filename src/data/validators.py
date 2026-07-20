"""OHLCV 데이터 및 포지션 파일 무결성 검증 모듈.

프로덕션 파이프라인에서 데이터 품질 이슈를 조기에 발견하기 위한 검증 유틸리티.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── OHLCV 데이터 검증 ────────────────────────────────────────


class DataValidationWarning:
    """검증 경고를 담는 경량 컨테이너."""

    def __init__(self, level: str, ticker: str, message: str):
        self.level = level  # "warning" | "error"
        self.ticker = ticker
        self.message = message

    def __repr__(self) -> str:
        return f"[{self.level.upper()}] {self.ticker}: {self.message}"


def validate_ohlcv(
    data: pd.DataFrame,
    *,
    max_nan_ratio: float = 0.30,
    max_price_change: float = 0.50,
    max_zero_volume_streak: int = 5,
) -> list[DataValidationWarning]:
    """OHLCV 데이터의 품질을 검증하고 경고 목록을 반환.

    Args:
        data: (ticker, field) 멀티인덱스 DataFrame
        max_nan_ratio: NaN 비율 경고 기준 (기본 30%)
        max_price_change: 전일 대비 최대 가격 변동률 (기본 50%)
        max_zero_volume_streak: 연속 거래량 0 경고 기준 (기본 5일)

    Returns:
        DataValidationWarning 리스트
    """
    warnings_list: list[DataValidationWarning] = []

    if data.empty:
        warnings_list.append(DataValidationWarning("error", "ALL", "OHLCV 데이터가 비어있음"))
        return warnings_list

    # 멀티인덱스가 아닌 경우 스킵
    if not isinstance(data.columns, pd.MultiIndex):
        return warnings_list

    # 티커 목록 추출
    tickers = data.columns.get_level_values(0).unique().tolist()

    for ticker in tickers:
        if ticker not in data.columns.get_level_values(0):
            continue

        ticker_data = data[ticker]

        # 1. NaN 비율 체크
        if "Close" in ticker_data.columns:
            close = ticker_data["Close"]
            nan_ratio = close.isna().sum() / len(close) if len(close) > 0 else 0
            if nan_ratio > max_nan_ratio:
                warnings_list.append(DataValidationWarning(
                    "warning", ticker,
                    f"Close NaN 비율 {nan_ratio:.1%} (기준: {max_nan_ratio:.0%})"
                ))

        # 2. 가격 이상치 감지
        if "Close" in ticker_data.columns:
            close = ticker_data["Close"].dropna()
            if len(close) > 1:
                pct_changes = close.pct_change().dropna().abs()
                extreme = pct_changes[pct_changes > max_price_change]
                if len(extreme) > 0:
                    worst_date = extreme.idxmax()
                    worst_val = extreme.max()
                    warnings_list.append(DataValidationWarning(
                        "warning", ticker,
                        f"비정상 가격 변동 {worst_val:.1%} ({worst_date.date()}) — 스플릿/데이터 오류 가능"
                    ))

        # 3. 거래량 0 연속일 체크 (최근 1년만 검사 — 합병 전/상장 전 데이터 오탐 방지)
        if "Volume" in ticker_data.columns:
            vol = ticker_data["Volume"].dropna()
            # 최근 252 거래일(≈1년)만 검사
            vol = vol.tail(252)
            if len(vol) > 0:
                zero_streaks = (vol == 0).astype(int)
                # 연속 0 카운트
                streak = 0
                max_streak = 0
                for v in zero_streaks:
                    if v == 1:
                        streak += 1
                        max_streak = max(max_streak, streak)
                    else:
                        streak = 0
                if max_streak >= max_zero_volume_streak:
                    warnings_list.append(DataValidationWarning(
                        "warning", ticker,
                        f"거래량 0 연속 {max_streak}일 (기준: {max_zero_volume_streak}일) — 상장폐지/거래정지 가능"
                    ))

    if warnings_list:
        logger.warning(f"[DataValidator] {len(warnings_list)}건의 데이터 품질 경고 발생")
        for w in warnings_list:
            logger.warning(str(w))

    return warnings_list


# ── 포지션 파일 검증 ──────────────────────────────────────────

_REQUIRED_POSITION_FIELDS = [
    "ticker", "entry_price", "entry_date", "strategy",
    "star_rating", "ccs_score", "highest_price",
]


def validate_positions(positions: list[dict[str, Any]]) -> list[DataValidationWarning]:
    """positions.json 스키마 검증.

    Returns:
        경고 리스트 (비어있으면 정상)
    """
    warnings_list: list[DataValidationWarning] = []

    if not isinstance(positions, list):
        warnings_list.append(DataValidationWarning(
            "error", "ALL", "positions가 리스트가 아님"
        ))
        return warnings_list

    for i, pos in enumerate(positions):
        if not isinstance(pos, dict):
            warnings_list.append(DataValidationWarning(
                "error", f"index_{i}", "포지션이 dict가 아님"
            ))
            continue

        ticker = pos.get("ticker", f"index_{i}")

        # 필수 필드 존재 확인
        for field in _REQUIRED_POSITION_FIELDS:
            if field not in pos:
                warnings_list.append(DataValidationWarning(
                    "error", ticker, f"필수 필드 누락: {field}"
                ))

        # 가격 유효성
        entry_price = pos.get("entry_price", 0)
        if not isinstance(entry_price, (int, float)) or entry_price <= 0:
            warnings_list.append(DataValidationWarning(
                "error", ticker, f"유효하지 않은 entry_price: {entry_price}"
            ))

        highest = pos.get("highest_price", 0)
        if isinstance(highest, (int, float)) and isinstance(entry_price, (int, float)):
            if highest > 0 and highest < entry_price * 0.5:
                warnings_list.append(DataValidationWarning(
                    "warning", ticker,
                    f"highest_price({highest:.2f})가 entry_price({entry_price:.2f})의 50% 미만"
                ))

    return warnings_list


def validate_positions_file(path: Path | str) -> list[DataValidationWarning]:
    """positions.json 파일을 로드하고 검증.

    Returns:
        경고 리스트
    """
    path = Path(path)
    if not path.exists():
        return []  # 파일 없음 = 빈 포지션 (정상)

    try:
        positions = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [DataValidationWarning("error", "ALL", f"positions.json 파싱 실패: {e}")]

    return validate_positions(positions)
