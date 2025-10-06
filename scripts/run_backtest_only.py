#!/usr/bin/env python3
"""Run configured backtest scenarios and upload results to Google Sheets.

This bypasses the full pipeline (signals/portfolio) and executes only the
backtest portion defined in config.BACKTEST_RUNS.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import run_backtest
from config import BACKTEST_RUNS, BACKTEST_WORKSHEET_NAME
from exporter import export_backtest_results


def main() -> None:
    results: dict[str, dict] = {}
    for run in BACKTEST_RUNS:
        label = str(run.get("label", "Backtest"))
        params = {k: v for k, v in run.items() if k != "label"}
        try:
            results[label] = run_backtest(**params)
            print(f"[Backtest] '{label}' 실행 완료")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[Backtest] '{label}' 실행 실패: {exc}")

    if not results:
        print("[Backtest] 실행 가능한 시나리오가 없습니다.")
        return

    if export_backtest_results(results, BACKTEST_WORKSHEET_NAME):
        print(f"[{BACKTEST_WORKSHEET_NAME}] 백테스트 결과 업데이트 완료")
    else:
        print(f"[{BACKTEST_WORKSHEET_NAME}] 백테스트 결과 업데이트를 건너뜁니다.")


if __name__ == "__main__":
    main()
