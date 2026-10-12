"""OHLCV 품질 경고 — 상장 전 빈 값·오래된 급등락은 경고하지 않고, 출력은 종류별 한 줄."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.validators import (  # noqa: E402
    DataValidationWarning,
    summarize_warnings,
    validate_ohlcv,
)

IDX = pd.bdate_range("2024-06-03", periods=500)


def _frame(close: np.ndarray, volume: float = 1_000_000.0) -> pd.DataFrame:
    return pd.DataFrame({"Close": close, "Volume": np.where(np.isnan(close), np.nan, volume)}, index=IDX)


def _ohlcv(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """fetch_ohlcv 와 같은 (티커, 필드) 멀티인덱스 표."""
    return pd.concat(frames, axis=1)


def _flat(price: float = 50.0) -> np.ndarray:
    return np.full(len(IDX), price)


def _by_ticker(warnings: list[DataValidationWarning]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for w in warnings:
        out.setdefault(w.ticker, []).append(w.message)
    return out


def test_nan_ratio_counts_only_after_listing():
    new = _flat()
    new[:400] = np.nan                    # 최근 상장 — 상장 전 400일은 빈 값 (경고 아님)
    gappy = _flat()
    gappy[100:300] = np.nan               # 상장 후 40% 가 빈 값 (경고)
    never = np.full(len(IDX), np.nan)     # 데이터가 아예 없음 (경고)
    got = _by_ticker(validate_ohlcv(_ohlcv({"OLD": _frame(_flat()), "NEW": _frame(new),
                                            "GAPPY": _frame(gappy), "NEVER": _frame(never)})))
    assert "OLD" not in got and "NEW" not in got
    assert got["GAPPY"] == ["Close NaN 비율 40.0% (상장 이후, 기준: 30%)"]
    assert got["NEVER"] == ["Close NaN 비율 100.0% (상장 이후, 기준: 30%)"]


def test_price_jump_only_within_last_year_with_sign():
    old_jump = _flat()
    old_jump[100:] *= 1.8                 # 400 거래일 전 +80% — 1년 넘게 지나 경고 안 함
    up = _flat(63.0)
    up[-30:] = 174.4                      # 30 거래일 전 +177%
    down = _flat()
    down[-10:] = 20.0                     # 10 거래일 전 -60%
    warnings = validate_ohlcv(_ohlcv({"OLDJ": _frame(old_jump), "UP": _frame(up), "DOWN": _frame(down)}))
    got = _by_ticker(warnings)
    assert "OLDJ" not in got
    assert got["UP"] == [f"비정상 가격 변동 +176.8% ({IDX[-30].date()}) — 스플릿/데이터 오류 가능"]
    assert got["DOWN"] == [f"비정상 가격 변동 -60.0% ({IDX[-10].date()}) — 스플릿/데이터 오류 가능"]
    assert [w.detail for w in warnings] == [f"+177%({IDX[-30].date()})", f"-60%({IDX[-10].date()})"]


def test_zero_volume_streak_in_last_year():
    frame = _frame(_flat())
    frame.iloc[-8:, frame.columns.get_loc("Volume")] = 0.0
    warnings = validate_ohlcv(_ohlcv({"HALT": frame}))
    assert [w.message for w in warnings] == ["거래량 0 연속 8일 (기준: 5일) — 상장폐지/거래정지 가능"]
    assert warnings[0].detail == "8일"


def test_summary_is_one_line_per_kind():
    gappy = _flat()
    gappy[100:300] = np.nan
    up = _flat(63.0)
    up[-30:] = 174.4
    down = _flat()
    down[-10:] = 20.0
    lines = summarize_warnings(validate_ohlcv(_ohlcv({
        "GAPPY": _frame(gappy), "UP": _frame(up), "DOWN": _frame(down),
    })))
    assert lines[0] == "[DataValidator] 데이터 품질 경고 3건 — 정보용, 데이터는 그대로 씀"
    assert len(lines) == 3  # 머리줄 + 종류 2줄
    assert lines[1] == "  상장 이후 종가 빈 값 30% 넘음 — 1종목: GAPPY 40%"
    assert lines[2].startswith("  최근 1년 하루 ±50% 넘는 변동")
    assert lines[2].endswith(f"— 2종목: UP +177%({IDX[-30].date()}), DOWN -60%({IDX[-10].date()})")
    assert summarize_warnings([]) == []


def test_position_warnings_keep_old_constructor():
    # validate_positions 쪽은 그룹 없이 만든다 — 요약에선 메시지 자체가 그룹
    w = DataValidationWarning("error", "ABT", "필수 필드 누락: ticker")
    assert repr(w) == "[ERROR] ABT: 필수 필드 누락: ticker"
    assert summarize_warnings([w])[1] == "  필수 필드 누락: ticker — 1종목: ABT"
