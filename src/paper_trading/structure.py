"""PT-1 구조 실험 스위치 (워크포워드 2단계용). 기본값 = 현재 동작.

- PT1_BEAR_MODE: "none"(기존) | "liquidate"(약세장이 되면 보유 종목을 다음날 시가에 모두 정리)
    · 신규 매수 중단은 기존 스위치 CANDIDATE_BEAR_BLOCK_NEW 를 쓴다
- PT1_SIZING: "equal"(기존: 남은 현금 ÷ 빈자리) | "inv_vol"(평가액 ÷ 최대 보유 수 × min(1, 그날 중앙 ATR% ÷ 종목 ATR%))
    · 변동성이 중앙값보다 큰 종목은 그만큼 덜 산다. 레버리지는 없다 (남는 현금은 그대로 둔다)

screener/config.py 는 피처 캐시 해시 대상이라 고치면 10년 캐시를 다시 만들어야 한다. 그래서 검증 전 실험 스위치의
기본값은 여기서 config 모듈에 붙인다. 채택되면 config.py 로 옮기고 실거래 엔진에도 같은 규칙을 넣는다.
"""

from __future__ import annotations

import math

import pandas as pd

from screener import config as _cfg

DEFAULTS: dict[str, str] = {"PT1_BEAR_MODE": "none", "PT1_SIZING": "equal"}
for _name, _value in DEFAULTS.items():
    if not hasattr(_cfg, _name):
        setattr(_cfg, _name, _value)


def bear_liquidate() -> bool:
    """약세장이면 보유 종목을 모두 정리하는가."""
    return getattr(_cfg, "PT1_BEAR_MODE", "none") == "liquidate"


def inv_vol_factor(ranked_df: pd.DataFrame | None, ticker: str) -> float:
    """변동성 반비례 비중 계수 (0~1). 그날 스냅샷 중앙 ATR% ÷ 종목 ATR%, 1 이 상한. 값이 없으면 1."""
    if ranked_df is None or ranked_df.empty or "ATR%" not in ranked_df.columns or "티커" not in ranked_df.columns:
        return 1.0
    atr = pd.to_numeric(ranked_df["ATR%"], errors="coerce")
    atr = atr[(atr > 0) & (atr < 1)]  # 소수 단위 정상값만 (이상치·결측 제외)
    row = ranked_df.loc[ranked_df["티커"] == ticker, "ATR%"]
    if atr.empty or row.empty:
        return 1.0
    own = pd.to_numeric(row, errors="coerce").iloc[0]
    if own is None or not math.isfinite(own) or own <= 0:
        return 1.0
    return float(min(1.0, atr.median() / own))


def slot_allocation(cash: float, equity: float, max_positions: int, empty_slots: int,
                    ranked_df: pd.DataFrame | None, ticker: str) -> float:
    """새 종목에 쓸 금액. equal = 기존 공식 그대로, inv_vol = 평가액 기준 한 자리 × 변동성 계수 (현금 한도)."""
    if getattr(_cfg, "PT1_SIZING", "equal") != "inv_vol":
        return cash / max(1, empty_slots)
    slot = equity / max(1, max_positions)
    return max(0.0, min(cash, slot * inv_vol_factor(ranked_df, ticker)))
