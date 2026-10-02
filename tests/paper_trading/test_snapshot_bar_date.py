"""스냅샷 일봉 날짜 테스트: 장중에 받은 스냅샷(빈 날짜)은 페이퍼 트레이딩 대상에서 빠진다."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading import runner


def _snapshot(tickers: list[str], bar_date: str) -> pd.DataFrame:
    return pd.DataFrame({"티커": tickers, "모멘텀_적합도": [7.0] * len(tickers), "_bar_date": bar_date})


def _write(tmp_path: Path, monkeypatch, sp500: pd.DataFrame, nasdaq: pd.DataFrame) -> None:
    paths = {"_SP500_SNAPSHOT": tmp_path / "sp500.parquet", "_NASDAQ_SNAPSHOT": tmp_path / "nasdaq.parquet"}
    sp500.to_parquet(paths["_SP500_SNAPSHOT"])
    nasdaq.to_parquet(paths["_NASDAQ_SNAPSHOT"])
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)


def test_blank_bar_date_is_unknown():
    assert runner.snapshot_bar_date(_snapshot(["AAA"], "")) is None
    assert runner.snapshot_bar_date(_snapshot(["AAA"], "2026-09-29")) == "2026-09-29"


def test_merge_keeps_only_completed_snapshot(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, _snapshot(["AAA"], ""), _snapshot(["BBB"], "2026-09-29"))
    merged = runner.load_and_merge_snapshots()
    assert list(merged["티커"]) == ["BBB"]
    assert runner.snapshot_bar_date(merged) == "2026-09-29"


def test_all_intraday_snapshots_have_no_trading_date(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, _snapshot(["AAA"], ""), _snapshot(["BBB"], ""))
    merged = runner.load_and_merge_snapshots()
    assert runner.snapshot_bar_date(merged) is None
