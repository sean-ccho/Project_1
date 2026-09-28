#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""통합 페이퍼 트레이딩 CLI 진입점.

GitHub Actions 실행 순서:
  1. main.py          → SP500 ranked_df → data/paper_trading/sp500_ranked.parquet
  2. run_full_scan.py → NASDAQ ranked_df → data/paper_trading/nasdaq_ranked.parquet
  3. 이 스크립트     → 두 파일 합산 → 단일 paper trading 실행

로컬 실행:
  PYTHONPATH=.:src python src/paper_trading/run_paper_trading.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# src/ 디렉토리를 sys.path에 추가 (GitHub Actions에서 PYTHONPATH로 주입되지만 안전망)
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def main() -> None:
    from paper_trading.runner import run_unified_paper_trading
    run_unified_paper_trading()


if __name__ == "__main__":
    main()
