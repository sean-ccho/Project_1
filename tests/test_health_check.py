"""실행 후 헬스 체크 — 가짜 data/·logs/ 폴더로 (네트워크·git 없음)."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import monitoring.health_check as hc

ET = ZoneInfo("America/Toronto")
NOW = datetime(2026, 10, 12, 3, 0, tzinfo=ET)  # 월요일 새벽 실행 → 금요일(10-09) 일봉 확정
EMAIL_LINE = "[Email] 페이퍼 트레이딩 알림 발송 완료 (매도 0건, 매수 1건, 재평가 0건, 재평가실패 0건, 후보 5건) → me\n"


def _layout(tmp_path: Path, *, bar: str = "2026-10-09", nasdaq_bar: str | None = None,
            states: dict | None = None, logs: dict | None = None, log_date: str = "2026-10-12"):
    """정상 실행 직후 모양의 data/paper_trading · logs/LOG_DATE 를 만든다. states 값 None = state.json 없음."""
    data = tmp_path / "data" / "paper_trading"
    for key in hc.ACCOUNTS:
        d = hc._account_dir(data, key)
        d.mkdir(parents=True, exist_ok=True)
        for name in hc.ACCOUNT_FILES[key]:
            (d / name).write_text("{}" if name == "account.json" else "[]", encoding="utf-8")
        state = (states or {}).get(key, bar)
        if state is not None:
            (d / "state.json").write_text(json.dumps({"last_processed_bar_date": state}), encoding="utf-8")
    pd.DataFrame({"티커": ["AAA"], "_bar_date": [bar]}).to_parquet(data / "sp500_ranked.parquet")
    pd.DataFrame({"티커": ["BBB"], "_bar_date": [nasdaq_bar or bar]}).to_parquet(data / "nasdaq_ranked.parquet")
    log_dir = tmp_path / "logs" / log_date
    log_dir.mkdir(parents=True)
    for name in hc.STEP_LOGS:
        (log_dir / name).write_text("ok\n", encoding="utf-8")
    (log_dir / "paper_trading.log").write_text(EMAIL_LINE, encoding="utf-8")
    for name, text in (logs or {}).items():
        (log_dir / name).write_text(text, encoding="utf-8")
    return data, tmp_path / "logs"


def _run(tmp_path, now=NOW, log_date="2026-10-12", **kw):
    data, log_root = _layout(tmp_path, log_date=log_date, **kw)
    return hc.run_health_check(data, log_date, log_root=log_root, now=now)


def _failed(result) -> list[str]:
    return [c["name"] for c in result["checks"] if c["status"] == "fail"]


def test_healthy_run_passes_and_records_expected_state(tmp_path):
    r = _run(tmp_path)
    assert r["status"] == "healthy", _failed(r)
    assert r["expected"] == {"pt1": "2026-10-09", "pt1s": "2026-10-09", "pt1n": "2026-10-09", "pt_spy": "2026-10-09"}


def test_state_not_advanced_fails(tmp_path):
    """10-10 ~ 10-11: 상태가 저장되지 않아 pt1·pt1s 가 10-08 에 머문 경우."""
    r = _run(tmp_path, states={"pt1": "2026-10-08", "pt1s": "2026-10-08"})
    assert r["status"] == "unhealthy"
    assert _failed(r) == ["pt1 상태", "pt1s 상태"]


def test_missing_state_file_fails(tmp_path):
    r = _run(tmp_path, states={"pt_spy": None})
    assert "pt_spy 상태" in _failed(r)


def test_own_error_line_fails(tmp_path):
    line = "2026-10-11 09:49:41,920 [ERROR] paper_trading.sheet_sync — [SheetSync] 거래로그 append 실패: list index out of range\n"
    r = _run(tmp_path, logs={"paper_trading_pt1s.log": line})
    assert _failed(r) == ["에러 로그"]
    assert "paper_trading_pt1s.log:1" in next(c["message"] for c in r["checks"] if c["name"] == "에러 로그")


def test_library_error_line_only_warns(tmp_path):
    line = "2026-10-08 09:10:00,000 [ERROR] yfinance — ABT: Failed to retrieve the news\n"
    r = _run(tmp_path, logs={"full_scan.log": line})
    assert r["status"] == "warning"
    assert _failed(r) == []


def test_traceback_fails(tmp_path):
    r = _run(tmp_path, logs={"full_scan.log": "Traceback (most recent call last):\n  File ...\nValueError: x\n"})
    assert _failed(r) == ["에러 로그"]


def test_own_report_file_is_not_scanned(tmp_path):
    """health_check.log 는 다른 로그의 [ERROR] 줄을 인용하므로 다시 세면 안 된다."""
    quoted = "  ❌ 에러 로그: 1줄\n      x.log:1 [ERROR] paper_trading.sheet_sync — 실패\n"
    r = _run(tmp_path, logs={hc.REPORT_NAME: quoted})
    assert r["status"] == "healthy", _failed(r)


def test_snapshots_with_different_bars_fail(tmp_path):
    r = _run(tmp_path, nasdaq_bar="2026-10-08")
    assert "스냅샷 일봉 일치" in _failed(r)


def test_stale_snapshot_fails(tmp_path):
    r = _run(tmp_path, bar="2026-10-01")  # 계좌 상태도 10-01 이라 상태 검사는 통과, 일봉이 11일 묵음
    assert _failed(r) == ["sp500_ranked.parquet 일봉", "nasdaq_ranked.parquet 일봉"]


def test_long_weekend_is_not_stale(tmp_path):
    """월요일 휴장 뒤 화요일 새벽 실행: 마지막 일봉은 금요일 (4일 전)."""
    r = _run(tmp_path, bar="2026-10-09", log_date="2026-10-13", now=datetime(2026, 10, 13, 3, 0, tzinfo=ET))
    assert r["status"] == "healthy", _failed(r)


def test_intraday_run_skips_state_comparison(tmp_path):
    """장중 push 실행: 오늘 일봉이 미확정이라 계좌는 어제 일봉에 머무는 게 정상."""
    r = _run(tmp_path, bar="2026-10-12", states={k: "2026-10-09" for k in hc.ACCOUNTS},
             now=datetime(2026, 10, 12, 13, 0, tzinfo=ET))
    assert r["status"] == "healthy", _failed(r)


INTRADAY_LOGS = {
    "sp500_analysis.log": "[[SP500 분석]] 2026-10-12 일봉은 장중 데이터 → 페이퍼 트레이딩 대상 아님\n",
    "full_scan.log": "[[NASDAQ/NYSE 분석]] 2026-10-12 일봉은 장중 데이터 → 페이퍼 트레이딩 대상 아님\n",
}


def _intraday(tmp_path, *, manual: bool = True, states: dict | None = None, logs: dict = INTRADAY_LOGS):
    """10-28 실제 모양: 장중 push 실행이라 스캔이 _bar_date 를 비우고, 계좌는 전날(10-09) 일봉에 머문다."""
    data, log_root = _layout(tmp_path, bar="", states=states or {k: "2026-10-09" for k in hc.ACCOUNTS}, logs=logs)
    return hc.run_health_check(data, "2026-10-12", log_root=log_root,
                               now=datetime(2026, 10, 12, 15, 18, tzinfo=ET), manual=manual)


def test_manual_intraday_run_only_warns(tmp_path):
    r = _intraday(tmp_path)
    assert r["status"] == "warning", _failed(r)
    assert [c["name"] for c in r["checks"] if c["status"] == "warn"] == ["sp500_ranked.parquet 일봉", "nasdaq_ranked.parquet 일봉"]
    assert "장중 2026-10-12" in r["title"]


def test_scheduled_intraday_run_still_fails(tmp_path):
    """정기 실행이 장중에 돌면 그날 일봉을 놓친다 → 그대로 실패."""
    r = _intraday(tmp_path, manual=False)
    assert _failed(r) == ["sp500_ranked.parquet 일봉", "nasdaq_ranked.parquet 일봉"]


def test_manual_blank_bar_without_intraday_line_fails(tmp_path):
    """사람이 돌린 실행이라도 '장중 데이터' 줄 없이 날짜가 비면 (스캔이 데이터를 못 받음 등) 실패."""
    r = _intraday(tmp_path, logs={"full_scan.log": INTRADAY_LOGS["full_scan.log"]})
    assert _failed(r) == ["sp500_ranked.parquet 일봉"]


def test_intraday_bar_processed_fails(tmp_path):
    """장중 실행인데 계좌가 미확정 일봉을 처리했으면 실패."""
    r = _intraday(tmp_path, states={**{k: "2026-10-09" for k in hc.ACCOUNTS}, "pt1": "2026-10-12"})
    assert _failed(r) == ["pt1 상태"]


def test_main_reads_event_name(tmp_path, monkeypatch):
    data, _ = _layout(tmp_path, bar="", states={k: "2026-10-09" for k in hc.ACCOUNTS}, logs=INTRADAY_LOGS)
    monkeypatch.chdir(tmp_path)  # LOG_ROOT = logs/ (상대 경로)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    assert hc.main(["--log-date", "2026-10-12", "--data-dir", str(data)]) == 0
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    assert hc.main(["--log-date", "2026-10-12", "--data-dir", str(data)]) == 1


def test_missing_email_line_fails(tmp_path):
    r = _run(tmp_path, logs={"paper_trading.log": "[Email] 페이퍼 트레이딩 이메일 발송 실패: SMTP\n"})
    assert _failed(r) == ["PT-1 메일"]


def test_missing_step_log_and_account_file_fail(tmp_path):
    data, log_root = _layout(tmp_path)
    (log_root / "2026-10-12" / "full_scan.log").unlink()
    (data / "pt1s" / "trades.json").unlink()
    r = hc.run_health_check(data, "2026-10-12", log_root=log_root, now=NOW)
    assert _failed(r) == ["pt1s/trades.json", "스텝 로그"]


def test_positions_schema_error_fails(tmp_path):
    data, log_root = _layout(tmp_path)
    (data / "positions.json").write_text(json.dumps([{"ticker": "AAA", "entry_price": 0}]), encoding="utf-8")
    r = hc.run_health_check(data, "2026-10-12", log_root=log_root, now=NOW)
    assert _failed(r) == ["pt1/positions.json"]


def _fake_git(remote_states: dict, fetch_ok: bool = True):
    def git(*args):
        if args[0] == "fetch":
            return subprocess.CompletedProcess(args, 0 if fetch_ok else 1, "", "" if fetch_ok else "network down")
        path = args[1].split(":", 1)[1]
        for key in hc.ACCOUNTS:
            if path == (hc._account_dir(hc.DATA_DIR, key) / "state.json").as_posix() and key in remote_states:
                return subprocess.CompletedProcess(args, 0, json.dumps({"last_processed_bar_date": remote_states[key]}), "")
        return subprocess.CompletedProcess(args, 128, "", "fatal: path does not exist")
    return git


def test_verify_pushed(tmp_path):
    expected = tmp_path / "expected.json"
    want = {"pt1": "2026-10-09", "pt1s": "2026-10-09", "pt_spy": "2026-10-09"}
    expected.write_text(json.dumps(want), encoding="utf-8")

    assert hc.verify_pushed(expected, git=_fake_git(want))["status"] == "healthy"
    lost = hc.verify_pushed(expected, git=_fake_git({**want, "pt1": "2026-10-08", "pt1s": "2026-10-08"}))
    assert _failed(lost) == ["pt1 상태 push", "pt1s 상태 push"]
    missing = hc.verify_pushed(expected, git=_fake_git({"pt1": "2026-10-09", "pt1s": "2026-10-09"}))
    assert _failed(missing) == ["pt_spy 상태 push"]
    assert _failed(hc.verify_pushed(expected, git=_fake_git(want, fetch_ok=False))) == ["git fetch"]
    assert _failed(hc.verify_pushed(tmp_path / "nope.json", git=_fake_git(want))) == ["기대 상태"]


def test_main_exit_code_and_expected_out(tmp_path, monkeypatch):
    data, _ = _layout(tmp_path)
    monkeypatch.chdir(tmp_path)  # LOG_ROOT = logs/ (상대 경로)
    out = tmp_path / "expected.json"
    assert hc.main(["--log-date", "2026-10-12", "--data-dir", str(data), "--expected-out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["pt1"] == "2026-10-09"
    (tmp_path / "logs" / "2026-10-12" / "paper_trading.log").write_text("no email\n", encoding="utf-8")
    assert hc.main(["--log-date", "2026-10-12", "--data-dir", str(data)]) == 1
