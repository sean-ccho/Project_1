#!/usr/bin/env python3
"""TradingView 차트 캡처 로컬 테스트 스크립트."""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from charts.tradingview_capture import capture_tradingview_chart, capture_multiple_timeframes


def test_single_chart():
    """단일 차트 캡처 테스트."""
    print("=" * 60)
    print("테스트 1: 단일 차트 캡처 (AAPL Daily)")
    print("=" * 60)
    
    result = capture_tradingview_chart(
        ticker="AAPL",
        timeframe="Daily",
        headless=False,  # 브라우저를 화면에 표시
    )
    
    if result:
        print(f"✅ 성공! 파일 저장: {result}")
    else:
        print("❌ 실패")
    
    print()


def test_multiple_timeframes():
    """여러 타임프레임 캡처 테스트."""
    print("=" * 60)
    print("테스트 2: 여러 타임프레임 캡처 (TSLA)")
    print("=" * 60)
    
    results = capture_multiple_timeframes(
        ticker="TSLA",
        timeframes=["1H", "Daily", "Weekly"],
    )
    
    print(f"\n캡처 완료: {len(results)}/{3}개")
    for tf, path in results.items():
        print(f"  - {tf}: {path}")
    
    print()


if __name__ == "__main__":
    print("\n🚀 TradingView 차트 캡처 테스트 시작\n")
    
    # Playwright 설치 확인
    try:
        from playwright.sync_api import sync_playwright
        print("✅ Playwright 설치 확인됨\n")
    except ImportError:
        print("❌ Playwright가 설치되지 않았습니다!")
        print("\n설치 방법:")
        print("  pip install playwright")
        print("  playwright install chromium")
        sys.exit(1)
    
    # 테스트 선택
    print("실행할 테스트를 선택하세요:")
    print("  1. 단일 차트 (AAPL Daily)")
    print("  2. 여러 타임프레임 (TSLA 1H/Daily/Weekly)")
    print("  3. 둘 다 실행")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == "1":
        test_single_chart()
    elif choice == "2":
        test_multiple_timeframes()
    elif choice == "3":
        test_single_chart()
        test_multiple_timeframes()
    else:
        print("잘못된 선택입니다.")
        sys.exit(1)
    
    print("✨ 테스트 완료!")
    print(f"📁 스크린샷 위치: {project_root}/charts/screenshots/")
