#!/usr/bin/env python3
"""차트 캡처 기능 테스트 (Google Sheets 없이)"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from charts.tradingview_capture import capture_multiple_timeframes

def test_chart_capture():
    """차트 캡처 기본 테스트"""
    test_tickers = ["AAPL", "TSLA", "NVDA"]
    
    print("=" * 60)
    print("차트 캡처 테스트 시작")
    print("=" * 60)
    
    for ticker in test_tickers:
        print(f"\n[{ticker}] 캡처 시작...")
        try:
            results = capture_multiple_timeframes(
                ticker,
                timeframes=["Daily", "Weekly"],  # 2개만 테스트
                headless=True
            )
            
            if results:
                print(f"✅ [{ticker}] 성공: {len(results)}개 캡처")
                for tf, path in results.items():
                    print(f"   - {tf}: {path}")
            else:
                print(f"❌ [{ticker}] 실패: 캡처된 차트 없음")
                
        except Exception as e:
            print(f"❌ [{ticker}] 에러: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print(f"저장 위치: {project_root}/charts/screenshots/")
    print("=" * 60)

if __name__ == "__main__":
    test_chart_capture()
