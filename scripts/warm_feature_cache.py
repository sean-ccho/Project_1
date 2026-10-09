#!/usr/bin/env python3
"""백테스트 피처 캐시를 날짜 구간별로 미리 계산한다 (여러 프로세스로 나눠 돌려 긴 백테스트를 빠르게).

날짜별 피처 스냅샷은 그날까지의 시장 데이터만으로 정해지고 디스크 캐시(data/cache/features/<해시>)에
날짜별 파일로 저장된다. 같은 인자(period·종목·코드)로 겹치지 않는 구간을 동시에 돌리면, 나중의 본 실행은
캐시만 읽어 빨리 끝난다. 매매 시뮬레이션 결과는 버린다.

사용법: PYTHONPATH=.:src python scripts/warm_feature_cache.py --period 13y --start 2016-02-17 --end 2018-10-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.backtest import run_paper_trading_backtest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="13y")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    run_paper_trading_backtest(period=args.period, max_tickers=1000, rebalance_every=1, initial_capital=5000.0,
                               start_date=args.start, end_date=args.end, save_run=False, pit_universe=True)
    print(f"[warm] 끝: {args.start} ~ {args.end}")


if __name__ == "__main__":
    main()
