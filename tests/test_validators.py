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


# ── Yahoo 가 아직 반영 안 한 액면분할 (2026-11-18 EMBJ 4:1 · KALA 1:30) ─────────────

from data.validators import (  # noqa: E402
    repair_unapplied_splits,
    repair_unapplied_splits_frame,
    split_explains_jump,
    split_label,
    summarize_split_fixes,
)

SIDX = pd.bdate_range("2026-11-02", periods=12)   # 11-02 … 11-17


def _split_frame(closes: list[float], split_at: int, ratio: float) -> pd.DataFrame:
    """종가 · 고가 · 저가 · 거래량 · Stock Splits (split_at 번째 봉에 분할 기록)."""
    c = np.array(closes, dtype=float)
    sp = np.zeros(len(c))
    sp[split_at] = ratio
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                         "Volume": np.full(len(c), 1000, dtype=np.int64), "Stock Splits": sp}, index=SIDX[:len(c)])


def test_unapplied_forward_split_is_fixed():
    # 4:1 분할 — Yahoo 가 분할 전 가격 80 을 그대로 둠 → 분할 뒤 19.5
    df = _split_frame([80.0] * 9 + [19.5, 19.6, 19.4], split_at=9, ratio=4.0)
    out, fixes = repair_unapplied_splits_frame(df, "EMBJ")
    assert len(fixes) == 1 and fixes[0].ratio == 4.0
    assert out["Close"].iloc[0] == 20.0 and out["High"].iloc[0] == 80.0 * 1.01 / 4
    assert out["Volume"].iloc[0] == 4000.0          # 주식 수는 4배
    assert out["Close"].iloc[-1] == 19.4             # 분할 뒤 봉은 그대로
    assert abs(fixes[0].raw_move - (19.5 / 80 - 1)) < 1e-9 and abs(fixes[0].fixed_move) < 0.03


def test_event_on_empty_row_before_first_bar():
    # yfinance: 분할 기록이 종가 없는 빈 줄(분할일, 그날 봉 누락)에 찍히고 첫 종가는 다음 날
    df = _split_frame([0.14] * 9 + [np.nan, 3.43, 3.26], split_at=9, ratio=1 / 30)
    out, fixes = repair_unapplied_splits_frame(df, "KALA")
    assert len(fixes) == 1 and fixes[0].first_bar == SIDX[10]
    assert abs(out["Close"].iloc[0] - 4.2) < 1e-9    # 0.14 × 30
    assert abs(fixes[0].fixed_move - (3.43 / 4.2 - 1)) < 1e-9


def test_already_applied_split_is_left_alone():
    # Yahoo 가 이미 반영 — 분할 전 20, 뒤 19.5 → 나누면 오히려 점프가 생기므로 그대로
    df = _split_frame([20.0] * 9 + [19.5, 19.6, 19.4], split_at=9, ratio=4.0)
    out, fixes = repair_unapplied_splits_frame(df, "AAPL")
    assert fixes == [] and out is df


def test_real_jump_without_split_record_is_kept():
    df = _split_frame([7.5] * 9 + [19.15, 19.9, 18.0], split_at=0, ratio=0.0)  # SLS +156% — 분할 기록 없음
    out, fixes = repair_unapplied_splits_frame(df, "SLS")
    assert fixes == [] and out is df


def test_small_split_three_for_two():
    assert split_explains_jump(60.0, 40.4, 1.5)       # 3:2 를 반영 안 함 (−33%)
    assert not split_explains_jump(40.0, 40.4, 1.5)   # 이미 반영
    assert split_explains_jump(50.0, 40.1, 1.25)      # 5:4


def test_stock_dividend_is_not_treated_as_split():
    # Yahoo 는 1~3% 주식배당도 분할로 적는다 — 그날 −3% 가 우연히 맞아도 과거 가격은 그대로
    assert not split_explains_jump(100.0, 96.8, 1.033)
    df = _split_frame([100.0] * 9 + [96.8, 97.0, 97.5], split_at=9, ratio=1.033)
    assert repair_unapplied_splits_frame(df, "LEN")[1] == []


def test_multiindex_repair_touches_only_split_tickers():
    a = _split_frame([80.0] * 9 + [19.5, 19.6, 19.4], split_at=9, ratio=4.0)
    b = _split_frame([50.0] * 12, split_at=0, ratio=0.0)
    data = pd.concat({"EMBJ": a, "XOM": b}, axis=1)
    out, fixes = repair_unapplied_splits(data)
    assert [f.ticker for f in fixes] == ["EMBJ"]
    assert out[("EMBJ", "Close")].iloc[0] == 20.0
    assert out[("XOM", "Close")].equals(data[("XOM", "Close")])
    assert data[("EMBJ", "Close")].iloc[0] == 80.0    # 받은 표는 그대로 (복사본을 고침)
    assert repair_unapplied_splits(b.pipe(lambda x: pd.concat({"XOM": x}, axis=1)))[1] == []


def test_split_labels_and_summary():
    assert [split_label(r) for r in (4.0, 1 / 30, 1.5, 0.02, 10.0)] == ["4:1", "1:30", "3:2", "1:50", "10:1"]
    df = _split_frame([80.0] * 9 + [19.5, 19.6, 19.4], split_at=9, ratio=4.0)
    line = summarize_split_fixes(repair_unapplied_splits_frame(df, "EMBJ")[1])
    assert line and "EMBJ 4:1" in line[0] and "-76%" in line[0]
