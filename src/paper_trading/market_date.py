"""거래일(마지막 완성 일봉) 판정과 계좌별 중복 실행 방지.

GitHub Actions 러너는 UTC라서 date.today()가 미국 동부 날짜보다 하루 늦게 바뀌고,
cron이 주말에도 돌아서 금요일 데이터로 매매가 다시 일어났다.
페이퍼 트레이딩 날짜는 항상 "마지막으로 완성된 일봉 날짜(토론토 시간)"를 쓰고,
같은 일봉은 계좌마다 한 번만 처리한다.
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from paper_trading.json_store import atomic_json_write, read_json

MARKET_TZ = ZoneInfo("America/Toronto")
# 정규장 마감 16:00 + 데이터 반영 여유
BAR_COMPLETE_TIME = time(16, 30)
STATE_FILE = "state.json"


def market_now() -> datetime:
    """토론토(미국 동부) 현재 시각."""
    return datetime.now(MARKET_TZ)


def market_today() -> date:
    """토론토(미국 동부) 기준 오늘 날짜."""
    return market_now().date()


def is_bar_complete(bar_date: str, now: datetime | None = None) -> bool:
    """bar_date 일봉이 장 마감으로 확정됐는지. 오늘 일봉은 16:30 이후에만 확정으로 본다."""
    now = (now or market_now()).astimezone(MARKET_TZ)
    bar = date.fromisoformat(bar_date)
    if bar < now.date():
        return True
    if bar > now.date():
        return False
    return now.time() >= BAR_COMPLETE_TIME


def load_state(data_dir: Path) -> dict[str, Any]:
    """계좌 실행 상태(state.json). 없으면 빈 dict."""
    return read_json(Path(data_dir) / STATE_FILE, {})


def check_run_guard(
    bar_date: str | None,
    data_dir: Path,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """(실행 여부, 건너뛰는 사유).

    건너뛰는 경우: 일봉 날짜를 모름 / 장중이라 일봉 미완성 / 이미 처리한 일봉
    (주말·휴일 실행과 push 재실행은 마지막 조건에 걸린다)
    """
    if not bar_date:
        return False, "스냅샷에 일봉 날짜(_bar_date)가 없음"
    if not is_bar_complete(bar_date, now):
        return False, f"{bar_date} 일봉이 아직 확정되지 않음 (장중 실행)"
    last = load_state(data_dir).get("last_processed_bar_date")
    if last and bar_date <= last:
        return False, f"이미 처리한 일봉 ({bar_date}, 마지막 처리 {last})"
    return True, ""


def mark_processed(data_dir: Path, bar_date: str, **extra: Any) -> None:
    """bar_date 일봉을 처리 완료로 기록한다."""
    state = load_state(data_dir)
    state.update(extra)
    state["last_processed_bar_date"] = bar_date
    state["last_run_at"] = market_now().isoformat(timespec="seconds")
    atomic_json_write(Path(data_dir) / STATE_FILE, state)
