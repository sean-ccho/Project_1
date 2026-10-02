"""페이퍼 트레이딩 계좌 정의 (PT-2, PT-3).

PT-1(기존)은 engine.run_daily_trading을 그대로 쓰고, PT-2/PT-3는 account_engine을 쓴다.
params는 config의 dict 객체를 그대로 참조한다 (Optuna in-place 수정이 바로 반영됨).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ACCOUNT_KEYS = ("pt2", "pt3")


@dataclass
class AccountProfile:
    """계좌 1개의 설정 + 전략 모듈.

    strategy 모듈이 제공하는 함수 (account_engine에서 호출):
      - prefilter_tickers(rows, ctx, params) -> list[str]   후보 선정에 일봉이 필요한 종목
      - select_candidates(rows, ctx, params) -> (list[dict], dict)
      - init_position(pos, order, params) -> None            체결 직후 손절/목표 설정
      - on_bar_close(pos, ind, bar_date, params) -> str|None 장 마감 후 청산 사유
    """

    key: str
    name: str
    params: dict[str, Any]
    worksheets: dict[str, str]
    enabled: bool
    strategy: ModuleType

    @property
    def data_dir(self) -> Path:
        return Path(self.params["data_dir"])

    @property
    def version(self) -> str:
        return str(self.params.get("version", self.key))


def get_profile(key: str) -> AccountProfile:
    """계좌 키(pt2/pt3)로 프로필을 만든다."""
    from screener import config as cfg

    if key == "pt2":
        from paper_trading import pt2_golden_cross

        return AccountProfile(
            "pt2", "PT-2 골든크로스 스윙", cfg.PT2_PARAMS, cfg.PT2_WORKSHEETS,
            cfg.PT2_ENABLED, pt2_golden_cross,
        )
    if key == "pt3":
        from paper_trading import pt3_short_term

        return AccountProfile(
            "pt3", "PT-3 일봉 단타", cfg.PT3_PARAMS, cfg.PT3_WORKSHEETS,
            cfg.PT3_ENABLED, pt3_short_term,
        )
    raise ValueError(f"알 수 없는 계좌: {key} (가능: {', '.join(ACCOUNT_KEYS)})")
