"""거래일 판정·중복 실행 방지 테스트."""

from datetime import datetime, timezone

from paper_trading.market_date import (
    MARKET_TZ,
    check_run_guard,
    is_bar_complete,
    load_state,
    mark_processed,
)


def _at(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=MARKET_TZ)


def test_today_bar_incomplete_during_market_hours():
    assert not is_bar_complete("2026-09-29", _at(2026, 9, 29, 15, 0))


def test_today_bar_complete_after_close():
    assert is_bar_complete("2026-09-29", _at(2026, 9, 29, 20, 45))


def test_utc_runner_time_maps_to_previous_toronto_day():
    # GitHub 러너 00:45 UTC 토요일 = 토론토 금요일 20:45 → 금요일 일봉은 확정
    now = datetime(2026, 9, 26, 0, 45, tzinfo=timezone.utc)
    assert is_bar_complete("2026-09-25", now)


def test_future_bar_is_not_complete():
    assert not is_bar_complete("2026-09-30", _at(2026, 9, 29, 20, 0))


def test_guard_requires_bar_date(tmp_path):
    ok, reason = check_run_guard(None, tmp_path)
    assert not ok and "_bar_date" in reason


def test_guard_blocks_weekend_rerun_of_same_bar(tmp_path):
    friday_evening = _at(2026, 9, 25, 20, 45)
    ok, _ = check_run_guard("2026-09-25", tmp_path, friday_evening)
    assert ok
    mark_processed(tmp_path, "2026-09-25", account="pt1")

    # 토요일·일요일 실행에도 스냅샷의 마지막 일봉은 여전히 금요일
    saturday_evening = _at(2026, 9, 26, 20, 45)
    ok, reason = check_run_guard("2026-09-25", tmp_path, saturday_evening)
    assert not ok and "이미 처리" in reason
    assert load_state(tmp_path)["account"] == "pt1"


def test_guard_allows_next_trading_day(tmp_path):
    mark_processed(tmp_path, "2026-09-25")
    ok, _ = check_run_guard("2026-09-28", tmp_path, _at(2026, 9, 28, 20, 45))
    assert ok


def test_guard_blocks_intraday_push_run(tmp_path):
    ok, reason = check_run_guard("2026-09-29", tmp_path, _at(2026, 9, 29, 11, 0))
    assert not ok and "장중" in reason
