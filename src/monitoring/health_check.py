"""파이프라인 실행 후 헬스 체크 — GitHub Actions 'Health check'·'Verify state pushed' 스텝.

산출물을 파일 시각이 아니라 내용(일봉 날짜·계좌 상태·로그 줄)으로 확인한다.
하나라도 실패하면 exit 1 → 잡 실패 → 'Notify on failure' 메일 (본문에 이 결과 표).

검사
  1. 계좌 상태    pt1·pt1s·pt1n·pt_spy 의 state.json 이 있고, 일봉이 확정됐으면 마지막 처리 일봉 == 스냅샷 일봉
  2. 스냅샷 최신  sp500·nasdaq 스냅샷의 _bar_date 가 서로 같고 LOG_DATE − 4일 이후 (주말 + 공휴일 하루)
  3. 에러 로그    logs/LOG_DATE/*.log 에 우리 모듈의 [ERROR]·Traceback 이 없음 (외부 라이브러리 [ERROR] 는 경고만)
  4. 메일         paper_trading.log 에 PT-1 메일 발송 완료(또는 보낼 것 없음) 줄이 있음
  5. 계좌 파일    positions/trades(/account).json 이 있고 파싱되며 positions 스키마가 맞음
  6. 스텝 로그    스텝마다 로그 파일이 있음

--verify-pushed: 커밋·push 뒤 origin/main 의 state.json 이 이번 실행 값과 같은지 본다.
  2026-10-09 ~ 10-11 에 스크린샷 스텝의 checkout -f 가 커밋 전 상태를 지워, 같은 일봉을 사흘 연속 다시 처리했다.

사용법 (저장소 루트, PYTHONPATH=.:src):
  python -m monitoring.health_check --log-date 2026-10-12 --expected-out "$RUNNER_TEMP/expected_state.json"
  python -m monitoring.health_check --verify-pushed "$RUNNER_TEMP/expected_state.json"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from paper_trading.market_date import is_bar_complete, market_today
from paper_trading.runner import BAR_DATE_COL, snapshot_bar_date

DATA_DIR = Path("data/paper_trading")
LOG_ROOT = Path("logs")
# 계좌 키 → 데이터 폴더 (runner.PT1_ACCOUNTS · accounts.py 와 같게 — 계좌를 늘리면 여기도 늘린다)
ACCOUNTS: dict[str, str] = {"pt1": "", "pt1s": "pt1s", "pt1n": "pt1n", "pt_spy": "pt_spy"}
ACCOUNT_FILES: dict[str, tuple[str, ...]] = {
    "pt1": ("positions.json", "trades.json"),
    "pt1s": ("positions.json", "trades.json"),
    "pt1n": ("positions.json", "trades.json"),
    "pt_spy": ("positions.json", "trades.json", "account.json"),
}
SNAPSHOTS = ("sp500_ranked.parquet", "nasdaq_ranked.parquet")
STEP_LOGS = ("sp500_analysis.log", "full_scan.log", "paper_trading.log",
             "paper_trading_pt1s.log", "paper_trading_pt1n.log", "paper_trading_pt_spy.log", "portfolio_report.log")
MAX_BAR_AGE_DAYS = 4
OWN_LOGGERS = ("paper_trading", "screener", "data", "monitoring", "ml", "analytics", "__main__", "root")
EMAIL_SENT = ("페이퍼 트레이딩 알림 발송 완료", "이메일 발송 생략")
REPORT_NAME = "health_check.log"
_ERROR_LINE = re.compile(r"\[ERROR\]\s+([\w.]+)")

Check = dict[str, str]


def _check(name: str, status: str, message: str) -> Check:
    """status: ok | warn | fail."""
    return {"name": name, "status": status, "message": message}


def _account_dir(data_dir: Path, key: str) -> Path:
    sub = ACCOUNTS[key]
    return data_dir / sub if sub else data_dir


def _snapshot_bar(path: Path) -> str | None:
    """스냅샷 parquet 의 일봉 날짜 (컬럼이 없거나 못 읽으면 None)."""
    try:
        return snapshot_bar_date(pd.read_parquet(path, columns=[BAR_DATE_COL]))
    except Exception:  # noqa: BLE001 — 못 읽는 것 자체를 실패로 보고한다
        return None


def check_snapshots(data_dir: Path, log_date: date) -> tuple[list[Check], str | None]:
    """스냅샷마다 일봉 날짜가 있고 최근인지, 둘이 같은지. (검사 목록, 최신 일봉) 반환."""
    checks: list[Check] = []
    bars: dict[str, str] = {}
    oldest_ok = log_date - timedelta(days=MAX_BAR_AGE_DAYS)
    for name in SNAPSHOTS:
        path = data_dir / name
        bar = _snapshot_bar(path) if path.exists() else None
        if not path.exists():
            checks.append(_check(f"{name} 일봉", "fail", f"파일 없음: {path}"))
        elif bar is None:
            checks.append(_check(f"{name} 일봉", "fail", f"{BAR_DATE_COL} 을 읽지 못함"))
        elif date.fromisoformat(bar) < oldest_ok:
            checks.append(_check(f"{name} 일봉", "fail",
                                 f"일봉 {bar} 가 LOG_DATE {log_date} 보다 {(log_date - date.fromisoformat(bar)).days}일 오래됨 "
                                 f"— 스캔이 새 데이터를 못 썼다"))
        else:
            checks.append(_check(f"{name} 일봉", "ok", f"일봉 {bar}"))
        if bar:
            bars[name] = bar
    if len(set(bars.values())) > 1:
        checks.append(_check("스냅샷 일봉 일치", "fail",
                             " · ".join(f"{k} {v}" for k, v in bars.items()) + " — 한쪽 스캔이 실패했을 수 있다"))
    return checks, max(bars.values()) if bars else None


def _read_json(path: Path) -> tuple[Any, str | None]:
    """(데이터, 오류 메시지)."""
    if not path.exists():
        return None, f"파일 없음: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"JSON 파싱 실패: {exc}"


def check_accounts(data_dir: Path, bar: str | None, now: datetime | None = None
                   ) -> tuple[list[Check], dict[str, str | None]]:
    """계좌별 상태·파일. (검사 목록, 계좌별 마지막 처리 일봉) 반환 — 뒤 값은 verify-pushed 의 기대값."""
    from data.validators import validate_positions

    checks: list[Check] = []
    expected: dict[str, str | None] = {}
    complete = bar is not None and is_bar_complete(bar, now)
    for key in ACCOUNTS:
        acct_dir = _account_dir(data_dir, key)
        state, err = _read_json(acct_dir / "state.json")
        last = state.get("last_processed_bar_date") if isinstance(state, dict) else None
        expected[key] = last
        if err:
            checks.append(_check(f"{key} 상태", "fail", err))
        elif bar is None:
            checks.append(_check(f"{key} 상태", "warn", f"마지막 처리 {last} (스냅샷 일봉을 몰라 비교 못 함)"))
        elif not complete:
            checks.append(_check(f"{key} 상태", "ok", f"장중 실행 — 일봉 {bar} 미확정이라 비교 생략 (마지막 처리 {last})"))
        elif last == bar:
            checks.append(_check(f"{key} 상태", "ok", f"일봉 {bar} 처리됨"))
        else:
            checks.append(_check(f"{key} 상태", "fail",
                                 f"마지막 처리 {last} ≠ 스냅샷 일봉 {bar} — 이번 실행이 이 계좌를 처리하지 못했거나 상태가 저장되지 않음"))

        for name in ACCOUNT_FILES[key]:
            data, err = _read_json(acct_dir / name)
            if err:
                checks.append(_check(f"{key}/{name}", "fail", err))
                continue
            if name == "positions.json":
                errors = [w for w in validate_positions(data) if w.level == "error"]
                if errors:
                    checks.append(_check(f"{key}/{name}", "fail", f"스키마 오류 {len(errors)}개: {errors[0]}"))
                    continue
            size = f"{len(data)}개 항목" if isinstance(data, list) else "OK"
            checks.append(_check(f"{key}/{name}", "ok", size))
    return checks, expected


def check_logs(log_dir: Path) -> list[Check]:
    """스텝 로그가 다 있는지, 우리 모듈의 [ERROR]·Traceback 이 없는지, PT-1 메일이 나갔는지."""
    if not log_dir.exists():
        return [_check("로그 폴더", "fail", f"없음: {log_dir}")]
    checks: list[Check] = []
    missing = [n for n in STEP_LOGS if not (log_dir / n).exists()]
    checks.append(_check("스텝 로그", "fail" if missing else "ok",
                         f"없음: {', '.join(missing)} (스텝이 건너뛰어졌다)" if missing else f"{len(STEP_LOGS)}개 모두 있음"))

    own: list[str] = []
    foreign: list[str] = []
    for path in sorted(log_dir.glob("*.log")):
        if path.name == REPORT_NAME:  # 이 검사 자신의 출력 (tee 로 쓰는 중)
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            where = f"{path.name}:{i} {line.strip()[:160]}"
            if "Traceback (most recent call last)" in line:
                own.append(where)
                continue
            m = _ERROR_LINE.search(line)
            if m:
                (own if m.group(1).split(".")[0] in OWN_LOGGERS else foreign).append(where)
    if own:
        checks.append(_check("에러 로그", "fail", f"{len(own)}줄\n      " + "\n      ".join(own[:5])))
    else:
        checks.append(_check("에러 로그", "ok", "우리 모듈의 [ERROR]·Traceback 없음"))
    if foreign:
        checks.append(_check("외부 라이브러리 [ERROR]", "warn", f"{len(foreign)}줄 (예: {foreign[0]})"))

    pt_log = log_dir / "paper_trading.log"
    text = pt_log.read_text(encoding="utf-8", errors="replace") if pt_log.exists() else ""
    sent = any(marker in text for marker in EMAIL_SENT)
    checks.append(_check("PT-1 메일", "ok" if sent else "fail",
                         "발송 완료" if sent else "paper_trading.log 에 발송 완료 줄이 없음 (발송 실패 또는 조기 종료)"))
    return checks


def _summary(checks: list[Check], **extra: Any) -> dict[str, Any]:
    counts = {s: sum(c["status"] == s for c in checks) for s in ("ok", "warn", "fail")}
    status = "unhealthy" if counts["fail"] else "warning" if counts["warn"] else "healthy"
    return {"status": status, "counts": counts, "checks": checks, **extra}


def run_health_check(data_dir: Path = DATA_DIR, log_date: str | None = None, log_root: Path = LOG_ROOT,
                     now: datetime | None = None) -> dict[str, Any]:
    """전체 헬스 체크. log_date 는 워크플로의 LOG_DATE (없으면 토론토 오늘)."""
    day = date.fromisoformat(log_date) if log_date else market_today()
    snap_checks, bar = check_snapshots(data_dir, day)
    acct_checks, expected = check_accounts(data_dir, bar, now)
    checks = acct_checks + snap_checks + check_logs(log_root / day.isoformat())
    return _summary(checks, title=f"Health Check (LOG_DATE {day}, 스냅샷 일봉 {bar})", expected=expected)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def verify_pushed(expected_path: Path, git: Callable[..., subprocess.CompletedProcess] = _git,
                  ref: str = "origin/main") -> dict[str, Any]:
    """origin/main 의 계좌 state.json 이 이번 실행의 기대값과 같은지."""
    title = f"State push 확인 ({ref})"
    expected, err = _read_json(expected_path)
    if err:
        return _summary([_check("기대 상태", "fail", f"{err} — Health check 스텝이 끝까지 돌았는지 확인")], title=title)
    fetch = git("fetch", "--quiet", "origin", "main")
    if fetch.returncode != 0:
        return _summary([_check("git fetch", "fail", fetch.stderr.strip()[:200])], title=title)
    checks: list[Check] = []
    for key, want in expected.items():
        rel = (_account_dir(DATA_DIR, key) / "state.json").as_posix()
        shown = git("show", f"{ref}:{rel}")
        if shown.returncode != 0:
            checks.append(_check(f"{key} 상태 push", "fail", f"{ref} 에 {rel} 없음"))
            continue
        try:
            got = json.loads(shown.stdout).get("last_processed_bar_date")
        except json.JSONDecodeError as exc:
            checks.append(_check(f"{key} 상태 push", "fail", f"{rel} 파싱 실패: {exc}"))
            continue
        if got == want:
            checks.append(_check(f"{key} 상태 push", "ok", f"마지막 처리 {got}"))
        else:
            checks.append(_check(f"{key} 상태 push", "fail",
                                 f"{ref} 마지막 처리 {got} ≠ 이번 실행 {want} — 상태가 push 되지 않아 다음 실행이 같은 일봉을 다시 처리한다"))
    return _summary(checks, title=title)


def print_report(result: dict[str, Any]) -> None:
    icon = {"ok": "✅", "warn": "⚠️", "fail": "❌"}
    head = {"healthy": "✅", "warning": "⚠️", "unhealthy": "❌"}[result["status"]]
    c = result["counts"]
    print("=" * 60)
    print(f"  {head} {result['title']}: {result['status'].upper()} — 실패 {c['fail']} · 경고 {c['warn']} · 통과 {c['ok']}")
    print("=" * 60)
    for check in result["checks"]:
        print(f"  {icon[check['status']]} {check['name']}: {check['message']}")
    print("=" * 60, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="파이프라인 실행 후 헬스 체크")
    ap.add_argument("--log-date", help="logs/ 아래 날짜 폴더 (워크플로 LOG_DATE). 없으면 토론토 오늘")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--expected-out", help="계좌별 마지막 처리 일봉을 이 JSON 에 기록 (--verify-pushed 의 입력)")
    ap.add_argument("--verify-pushed", metavar="EXPECTED_JSON", help="origin/main 의 state.json 이 기대값과 같은지만 본다")
    args = ap.parse_args(argv)

    if args.verify_pushed:
        result = verify_pushed(Path(args.verify_pushed))
    else:
        result = run_health_check(Path(args.data_dir), args.log_date)
        if args.expected_out:
            Path(args.expected_out).write_text(json.dumps(result["expected"], indent=2), encoding="utf-8")
    print_report(result)
    return 1 if result["status"] == "unhealthy" else 0


if __name__ == "__main__":
    sys.exit(main())
