"""페이퍼 트레이딩 계좌 정의 (account_engine 으로 도는 계좌: PT-SPY).

PT-1·PT-1S 는 engine.run_daily_trading 을 그대로 쓰고, PT-SPY 는 account_engine 을 쓴다.
(PT-2 골든크로스 스윙·PT-3 일봉 단타는 2026-10-09 모두 제거)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ACCOUNT_KEYS = ("pt_spy",)


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
    """계좌 키로 프로필을 만든다."""
    if key == "pt_spy":
        from paper_trading import pt_spy

        return AccountProfile("pt_spy", "PT-SPY 지수 보유", pt_spy.PARAMS, pt_spy.WORKSHEETS, True, pt_spy)
    raise ValueError(f"알 수 없는 계좌: {key} (가능: {', '.join(ACCOUNT_KEYS)})")
