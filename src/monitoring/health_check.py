"""파이프라인 실행 후 헬스 체크 모듈.

GitHub Actions에서 파이프라인 실행 후 호출하여
핵심 산출물 상태를 자동으로 검증한다.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _check_file_recent(
    path: Path, max_age_hours: float = 2.0
) -> tuple[bool, str]:
    """파일이 존재하고 max_age_hours 이내에 수정되었는지 확인."""
    if not path.exists():
        return False, f"파일 없음: {path}"
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return False, f"파일이 {age_hours:.1f}시간 전에 수정됨 (기준: {max_age_hours}시간)"
    return True, f"OK (최근 {age_hours:.1f}시간 내 수정)"


def _check_json_valid(path: Path) -> tuple[bool, str]:
    """JSON 파일이 파싱 가능한지 확인."""
    if not path.exists():
        return True, "파일 없음 (빈 포지션 가능)"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return True, f"OK ({len(data)}개 항목)"
        return True, "OK"
    except json.JSONDecodeError as e:
        return False, f"JSON 파싱 실패: {e}"


def _check_positions_schema(path: Path) -> tuple[bool, str]:
    """positions.json 스키마 유효성 확인."""
    if not path.exists():
        return True, "파일 없음"
    try:
        from data.validators import validate_positions_file
        warnings = validate_positions_file(path)
        errors = [w for w in warnings if w.level == "error"]
        if errors:
            return False, f"{len(errors)}개 스키마 에러: {errors[0]}"
        return True, f"OK ({len(warnings)}개 경고)"
    except Exception as e:
        return False, f"검증 실패: {e}"


def _check_log_exists(log_date: str | None = None) -> tuple[bool, str]:
    """오늘 날짜 로그 파일이 생성되었는지 확인."""
    if log_date is None:
        log_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(f"logs/{log_date}")
    if not log_dir.exists():
        return False, f"로그 디렉토리 없음: {log_dir}"
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return False, f"로그 파일 없음: {log_dir}"
    return True, f"OK ({len(log_files)}개 로그 파일)"


def run_health_check(
    data_dir: str = "data/paper_trading",
    max_age_hours: float = 2.0,
) -> dict[str, Any]:
    """전체 헬스 체크 실행.

    Returns:
        {"status": "healthy" | "degraded" | "unhealthy", "checks": [...]}
    """
    data_path = Path(data_dir)
    checks: list[dict[str, Any]] = []

    # 1. positions.json 상태
    pos_path = data_path / "positions.json"
    ok, msg = _check_json_valid(pos_path)
    checks.append({"name": "positions.json 파싱", "ok": ok, "message": msg})

    ok, msg = _check_positions_schema(pos_path)
    checks.append({"name": "positions.json 스키마", "ok": ok, "message": msg})

    # 2. trades.json 상태
    trades_path = data_path / "trades.json"
    ok, msg = _check_json_valid(trades_path)
    checks.append({"name": "trades.json 파싱", "ok": ok, "message": msg})

    # 3. 스냅샷 파일 최신 여부
    for snapshot in ["sp500_ranked.parquet", "nasdaq_ranked.parquet"]:
        snap_path = data_path / snapshot
        ok, msg = _check_file_recent(snap_path, max_age_hours)
        checks.append({"name": f"{snapshot} 최신", "ok": ok, "message": msg})

    # 4. 로그 파일 존재
    ok, msg = _check_log_exists()
    checks.append({"name": "오늘 로그 파일", "ok": ok, "message": msg})

    # 집계
    total = len(checks)
    passed = sum(1 for c in checks if c["ok"])
    failed = total - passed

    if failed == 0:
        status = "healthy"
    elif failed <= 2:
        status = "degraded"
    else:
        status = "unhealthy"

    result = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "failed": failed,
        "total": total,
        "checks": checks,
    }

    # 콘솔 출력
    emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}
    print(f"\n{'='*60}")
    print(f"  {emoji[status]} Health Check: {status.upper()} ({passed}/{total} passed)")
    print(f"{'='*60}")
    for check in checks:
        icon = "✅" if check["ok"] else "❌"
        print(f"  {icon} {check['name']}: {check['message']}")
    print(f"{'='*60}\n")

    if status == "unhealthy":
        logger.error(f"[HealthCheck] UNHEALTHY — {failed}/{total} checks failed")
    elif status == "degraded":
        logger.warning(f"[HealthCheck] DEGRADED — {failed}/{total} checks failed")
    else:
        logger.info(f"[HealthCheck] HEALTHY — all {total} checks passed")

    return result


if __name__ == "__main__":
    run_health_check()
