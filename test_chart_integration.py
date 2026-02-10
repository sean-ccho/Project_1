#!/usr/bin/env python3
"""차트 통합 테스트 - 단일 종목"""

import sys
import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from charts.tradingview_capture import capture_multiple_timeframes
from exporter import generate_chart_paths

def test_chart_integration():
    """차트 캡처 및 경로 생성 테스트"""
    test_ticker = "AAPL"
    
    print(f"\n=== {test_ticker} 차트 캡처 테스트 ===\n")
    
    # 1. 차트 캡처
    print("[1/3] 차트 캡처 중...")
    chart_paths = capture_multiple_timeframes(test_ticker, headless=True)
    
    if not chart_paths:
        print("❌ 차트 캡처 실패")
        return False
    
    print(f"✅ {len(chart_paths)}개 타임프레임 캡처 완료:")
    for tf, path in chart_paths.items():
        print(f"  - {tf}: {path}")
    
    # 2. 차트 경로 생성
    print("\n[2/3] 차트 경로 생성 중...")
    generated_paths = generate_chart_paths(test_ticker)
    
    print(f"✅ 생성된 경로:")
    for col, path in generated_paths.items():
        print(f"  - {col}: {path}")
    
    # 3. DataFrame 시뮬레이션
    print("\n[3/3] DataFrame 통합 시뮬레이션...")
    
    # 가상 DataFrame 생성
    df = pd.DataFrame({
        "티커": [test_ticker],
        "매수적합도_표시": ["★★★★"]
    })
    
    # 차트 컬럼 추가
    for col, path in generated_paths.items():
        df[col] = path
    
    print("✅ DataFrame 생성 완료:")
    print(df[["티커", "매수적합도_표시", "차트_Daily"]])
    
    print("\n🎉 모든 테스트 통과!\n")
    return True

if __name__ == "__main__":
    success = test_chart_integration()
    sys.exit(0 if success else 1)
