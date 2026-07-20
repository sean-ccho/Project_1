#!/usr/bin/env python3
"""S&P 500 전체 유니버스 백테스팅 전용 CLI.

사용법:
    python scripts/run_sp500_backtest.py
    python scripts/run_sp500_backtest.py --period 2y --rebalance 5 --fundamentals
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.sp500_tickers import SP500_TICKERS
from paper_trading.backtest import run_paper_trading_backtest, print_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S&P 500 전체 유니버스 페이퍼 트레이딩 백테스트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--period", default="2y", help="데이터 기간 (예: 1y, 2y, 3y)")
    parser.add_argument("--rebalance", type=int, default=1, metavar="DAYS", help="리밸런스 주기 (거래일)")
    parser.add_argument("--fundamentals", action="store_true", help="펀더멘탈 데이터 포함 (시간이 더 걸립니다)")
    parser.add_argument("--capital", type=float, default=0.0, metavar="USD", help="초기 자본금 (예: 5000). 0이면 수익률 모드")
    parser.add_argument("--no-cache", action="store_true", help="디스크 캐시 무시, 처음부터 전체 재계산 (캐시 갱신)")
    args = parser.parse_args()

    print(f"\n페이퍼 트레이딩 백테스트 시작")
    print(f"  기간: {args.period}  |  리밸런스: {args.rebalance}일  |  종목수: {len(SP500_TICKERS)}개 (S&P500 전체)")
    if args.fundamentals:
        print("  펀더멘탈 데이터 포함 (시간이 더 걸립니다)")
    if args.capital > 0:
        print(f"  초기자본: ${args.capital:,.0f}")
    if args.no_cache:
        print("  캐시 무시 모드: 처음부터 전체 재계산합니다")

    result = run_paper_trading_backtest(
        tickers=list(SP500_TICKERS),
        period=args.period,
        rebalance_every=args.rebalance,
        include_fundamentals=args.fundamentals,
        max_tickers=None,
        initial_capital=args.capital,
        no_cache=args.no_cache,
    )

    print_summary(result["summary"], result["trades"])


if __name__ == "__main__":
    main()
