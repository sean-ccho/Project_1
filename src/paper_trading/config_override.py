"""실행 중 config 덮어쓰기 (A/B 실험·Optuna용). with 블록이 끝나면 원래 값으로 되돌린다."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator

from screener import config as cfg

# 실행 중에 cfg.X로 읽히거나 dict라서 in-place 변경이 반영되는 이름만 허용한다.
# (from config import X 로 묶인 스칼라는 덮어써도 반영되지 않아 실험 결과가 조용히 틀린다)
RUNTIME_OVERRIDABLE = frozenset({
    "CCS_VERSION", "CCS_V2_WEIGHTS", "CCS_V2_MIN",
    "CCS_REPLACE_MARGIN", "PT1_REPLACE_ENABLED",
    "CANDIDATE_ALLOWED_STRATEGIES", "CANDIDATE_BEAR_BLOCK_NEW", "CANDIDATE_ALPHA_WEIGHTS",
    "EXIT_PARAMS", "EXIT_PARAMS_DEFAULT", "PAPER_TRADING_MAX_POSITIONS",
    "PAPER_TRADING_MAX_DAILY_BUY", "BACKTEST_PIT_UNIVERSE",
    "PT2_PARAMS", "PT3_PARAMS",
})


def _deep_update(target: dict, patch: dict) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_update(target[k], v)
        else:
            target[k] = v


def _restore(target: dict, original: dict) -> None:
    """target을 original 내용으로 되돌린다. 중첩 dict도 같은 객체를 유지한다."""
    for k in [k for k in target if k not in original]:
        del target[k]
    for k, v in original.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _restore(target[k], v)
        else:
            target[k] = v


@contextmanager
def config_overrides(overrides: dict[str, Any]) -> Iterator[None]:
    """{config 이름: 값}을 잠시 적용한다.

    dict 값은 기존 dict에 재귀적으로 합친다 (다른 모듈이 같은 dict 객체를 들고 있어도 반영됨).
    """
    unknown = set(overrides) - RUNTIME_OVERRIDABLE
    if unknown:
        raise ValueError(f"런타임에 바꿀 수 없는 config: {sorted(unknown)} (RUNTIME_OVERRIDABLE 참고)")

    saved: dict[str, Any] = {}
    try:
        for name, value in overrides.items():
            current = getattr(cfg, name)
            saved[name] = copy.deepcopy(current)
            if isinstance(current, dict) and isinstance(value, dict):
                _deep_update(current, value)
            else:
                setattr(cfg, name, value)
        yield
    finally:
        for name, original in saved.items():
            current = getattr(cfg, name)
            if isinstance(current, dict) and isinstance(original, dict):
                _restore(current, original)
            else:
                setattr(cfg, name, original)
