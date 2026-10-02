#!/usr/bin/env python3
"""PT-2/PT-3 계좌 백테스트 CLI.

사용법:
  PYTHONPATH=.:src python scripts/run_account_backtest.py --account pt2 --period 5y
  PYTHONPATH=.:src python scripts/run_account_backtest.py --account pt3 --period 5y --max-tickers 300
  PYTHONPATH=.:src python scripts/run_account_backtest.py --account pt2 --start 2023-01-01 --end 2024-12-31
  PYTHONPATH=.:src python scripts/run_account_backtest.py --account pt2 --max-tickers 0 --pit-universe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.account_backtest import run_account_backtest
from paper_trading.backtest import print_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PT-2/PT-3 계좌 백테스트")
    parser.add_argument("--account", choices=["pt2", "pt3"], required=True)
    parser.add_argument("--period", default="5y", help="데이터 기간 (기본 5y)")
    parser.add_argument("--max-tickers", type=int, default=100, help="종목 수 (0이면 전체)")
    parser.add_argument("--start", default=None, help="시뮬레이션 시작일 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="시뮬레이션 종료일 YYYY-MM-DD")
    parser.add_argument("--fundamentals", action="store_true", help="펀더멘털 포함 (PIT-safe 설정 적용)")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시하고 새로 계산")
    parser.add_argument("--pit-universe", action="store_true", help="그날의 S&P 500 구성종목만 후보로 (생존 편향 완화)")
    args = parser.parse_args()

    result = run_account_backtest(
        args.account,
        period=args.period,
        max_tickers=args.max_tickers or None,
        include_fundamentals=args.fundamentals,
        start_date=args.start,
        end_date=args.end,
        no_cache=args.no_cache,
        pit_universe=args.pit_universe,
    )
    print_summary(result["summary"], result["trades"])


if __name__ == "__main__":
    main()
