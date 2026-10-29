"""지금 받는 배당 (screener.dividends) — 최근 정기 배당 × 1년 횟수, 특별배당·끊긴 배당 처리."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener.dividends import dividend_now, forward_dividend  # noqa: E402


def _daily(payouts: dict[str, float], end: str = "2026-10-28", tz: str | None = None) -> pd.Series:
    """일봉 Dividends 칸 모양 — 거래일마다 0, 배당락일에만 금액."""
    idx = pd.bdate_range("2023-01-02", end, tz=tz)
    s = pd.Series(0.0, index=idx)
    for d, a in payouts.items():
        s.loc[pd.Timestamp(d, tz=tz)] = a
    return s.sort_index()


def test_cut_uses_latest_regular_dividend() -> None:
    """TELUS: 0.418 → 0.188 (2026-09) — 합계(1.44)가 아니라 0.188 × 4."""
    s = _daily({"2025-12-11": 0.418, "2026-03-11": 0.418, "2026-06-10": 0.418, "2026-09-10": 0.188},
               tz="America/Toronto")
    assert forward_dividend(s) == pytest.approx(0.752)


def test_monthly_and_raise() -> None:
    monthly = _daily({f"2026-{m:02d}-15": 0.06 for m in range(1, 11)})
    assert forward_dividend(monthly) == pytest.approx(0.72)
    # 크게 올린 배당이 두 번 나오면 정기 배당 (NVDA 0.01 → 0.25)
    raised = _daily({"2025-12-04": 0.01, "2026-03-11": 0.01, "2026-06-04": 0.25, "2026-09-10": 0.25})
    assert forward_dividend(raised) == pytest.approx(1.0)


def test_special_dividend_is_skipped() -> None:
    """정기 1.30/분기 사이의 특별배당 15 는 빼고 본다 (COST)."""
    s = _daily({"2025-11-01": 1.30, "2026-01-28": 15.0, "2026-02-01": 1.30, "2026-05-01": 1.30, "2026-08-01": 1.30})
    assert forward_dividend(s) == pytest.approx(5.2)
    # 마지막이 특별배당이어도 직전 정기 배당 기준
    s2 = _daily({"2025-12-31": 0.25, "2026-03-31": 0.25, "2026-06-30": 0.25, "2026-09-15": 3.75})
    assert forward_dividend(s2) == pytest.approx(1.0)


def test_stopped_or_no_dividend_is_zero() -> None:
    """정기 배당이 끊긴 지 오래면 0 (ARI: 3월 뒤로 특별배당만)."""
    s = _daily({"2025-09-30": 0.25, "2025-12-31": 0.25, "2026-03-31": 0.25, "2026-07-16": 3.75})
    assert forward_dividend(s) == 0.0
    assert forward_dividend(_daily({})) == 0.0
    assert forward_dividend(pd.Series(dtype=float)) == 0.0


def test_dividend_now() -> None:
    assert dividend_now(0.752, 1.44, 11.99) == (0.0627, 0.752)
    assert dividend_now(22.0, 0.0, 31.8) == (0.0, 0.0)  # Yahoo: 정기 배당 없음 → 특별배당만
    assert dividend_now(453.6, 453.6, 0.84) == (None, None)  # 데이터 오류
    assert dividend_now(None, None, 10.0) == (None, None)
    assert dividend_now(1.0, float("nan"), None) == (None, 1.0)
