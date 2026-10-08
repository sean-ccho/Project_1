#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""통합 페이퍼 트레이딩 CLI 진입점.

GitHub Actions 실행 순서:
  1. main.py          → SP500 ranked_df → data/paper_trading/sp500_ranked.parquet
  2. run_full_scan.py → NASDAQ ranked_df → data/paper_trading/nasdaq_ranked.parquet
  3. 이 스크립트     → 두 파일 합산 → 계좌별 paper trading 실행

로컬 실행:
  PYTHONPATH=.:src python src/paper_trading/run_paper_trading.py                 # PT-1 (기존)
  PYTHONPATH=.:src python src/paper_trading/run_paper_trading.py --account pt1s  # PT-1S (PT-1 규칙, S&P 500만)
  PYTHONPATH=.:src python src/paper_trading/run_paper_trading.py --account pt2   # PT-2 골든크로스 스윙
  PYTHONPATH=.:src python src/paper_trading/run_paper_trading.py --account pt3   # PT-3 일봉 단타
  ... --dry-run [--as-of 2026-09-29]   # 저장·시트·이메일 없이 결과만 출력
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# src/ 디렉토리를 sys.path에 추가 (GitHub Actions에서 PYTHONPATH로 주입되지만 안전망)
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="페이퍼 트레이딩 실행")
    parser.add_argument("--account", choices=["pt1", "pt1s", "pt2", "pt3", "all"], default="pt1",
                        help="pt1=기존, pt1s=PT-1 규칙·S&P 500만, pt2=골든크로스 스윙, pt3=일봉 단타, all=전부")
    parser.add_argument("--dry-run", action="store_true", help="파일 저장·시트·이메일 없이 결과만 출력")
    parser.add_argument("--as-of", default=None, help="거래일 강제 지정 (YYYY-MM-DD)")
    args = parser.parse_args()

    accounts = ["pt1", "pt1s", "pt2", "pt3"] if args.account == "all" else [args.account]
    for key in accounts:
        if key in ("pt1", "pt1s"):
            from paper_trading.runner import run_unified_paper_trading

            run_unified_paper_trading(dry_run=args.dry_run, as_of=args.as_of, account=key)
        else:
            from paper_trading.account_engine import run_account_daily

            run_account_daily(key, dry_run=args.dry_run, as_of=args.as_of)


if __name__ == "__main__":
    main()
