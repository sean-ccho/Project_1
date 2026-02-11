#!/usr/bin/env python3
"""
로컬 차트 생성 도구 (Google Sheets 인증 없이 사용 가능)
사용법: python3 generate_charts_local.py [티커1] [티커2] ...
예: python3 generate_charts_local.py AAPL TSLA
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from charts.tradingview_capture import capture_multiple_timeframes
from config import CHARTS_OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description="로컬 차트 생성 도구")
    parser.add_argument("tickers", nargs="*", help="차트를 생성할 티커 목록 (공백으로 구분)")
    args = parser.parse_args()

    tickers = args.tickers
    if not tickers:
        print("티커를 입력하지 않아 기본 티커(AAPL, TSLA, NVDA)로 실행합니다.")
        tickers = ["AAPL", "TSLA", "NVDA"]

    print(f"대상 티커: {tickers}")
    print(f"저장 경로: {Path(CHARTS_OUTPUT_DIR).absolute()}")
    print("=" * 60)

    success_count = 0
    
    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\n[{ticker}] 차트 캡처 시작...")
        
        try:
            # headless=True로 설정하여 백그라운드 실행
            results = capture_multiple_timeframes(ticker, headless=True)
            
            if results:
                print(f"✅ [{ticker}] 성공 ({len(results)}개 파일)")
                for tf, path in results.items():
                    print(f"   - {tf}: {path}")
                success_count += 1
            else:
                print(f"❌ [{ticker}] 실패: 캡처된 이미지 없음")
                
        except Exception as e:
            print(f"❌ [{ticker}] 에러 발생: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"완료: {success_count}/{len(tickers)} 성공")
    print("=" * 60)

if __name__ == "__main__":
    main()
