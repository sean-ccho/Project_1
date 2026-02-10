"""TradingView 차트 캡처 모듈 (Playwright 기반)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    sync_playwright = None
    PlaywrightTimeout = Exception


# TradingView 타임프레임 코드 매핑
TIMEFRAME_MAP = {
    "1H": "60",
    "4H": "240",
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "M",
}


def capture_tradingview_chart(
    ticker: str,
    timeframe: str = "D",
    output_dir: str = "charts/screenshots",
    headless: bool = True,
) -> str | None:
    """
    TradingView 차트를 캡처하여 이미지 파일로 저장합니다.
    
    Args:
        ticker: 종목 심볼 (예: "AAPL", "TSLA")
        timeframe: 타임프레임 ("1H", "4H", "Daily", "Weekly", "Monthly")
        output_dir: 스크린샷 저장 디렉토리
        headless: 브라우저를 백그라운드에서 실행할지 여부
        
    Returns:
        저장된 이미지 파일 경로, 실패 시 None
    """
    if sync_playwright is None:
        print("[TradingView] Playwright가 설치되지 않았습니다. pip install playwright 후 playwright install 실행")
        return None
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 파일명 생성
    output_path = Path(output_dir) / f"{ticker}_{timeframe}.png"
    
    # 타임프레임 코드 변환
    interval = TIMEFRAME_MAP.get(timeframe, "D")
    
    # TradingView Widget Embed URL with TEMA indicator
    # 올바른 study ID: STD;TEMA (브라우저 리서치를 통해 확인)
    # Widget embed URL이 더 안정적으로 작동함
    url = (
        f"https://s.tradingview.com/widgetembed/"
        f"?symbol={ticker}"
        f"&interval={interval}"
        f"&hidesidetoolbar=1"
        f"&symboledit=1"
        f"&saveimage=1"
        f"&studies=STD%3BTEMA"  # %3B는 ; (세미콜론)의 URL 인코딩
        f"&theme=light"
        f"&style=1"
        f"&timezone=Etc%2FUTC"
    )
    
    try:
        with sync_playwright() as p:
            # 브라우저 실행
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=2,  # Retina 디스플레이 품질
            )
            page = context.new_page()
            
            print(f"[TradingView] {ticker} {timeframe} 차트 로딩 중 (TEMA 포함)...")
            
            # 페이지 열기
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 차트 및 인디케이터 로딩 대기
            time.sleep(8)  # TEMA 로딩을 위해 조금 더 대기
            
            print(f"[TradingView] TEMA 인디케이터 로딩 완료")
            
            # 광고/팝업 닫기 (있다면)
            try:
                page.click('button[aria-label="Close"]', timeout=2000)
            except:
                pass
            
            # 차트 영역만 스크린샷
            chart_selector = 'div[data-role="chart"]'
            try:
                chart_element = page.query_selector(chart_selector)
                if chart_element:
                    chart_element.screenshot(path=str(output_path))
                else:
                    # 전체 페이지 스크린샷 (fallback)
                    page.screenshot(path=str(output_path), full_page=False)
            except:
                # 전체 페이지 스크린샷 (fallback)
                page.screenshot(path=str(output_path), full_page=False)
            
            print(f"[TradingView] 스크린샷 저장: {output_path}")
            
            browser.close()
            
            return str(output_path)
            
    except PlaywrightTimeout:
        print(f"[TradingView] {ticker} 차트 로딩 타임아웃")
        return None
    except Exception as exc:
        print(f"[TradingView] {ticker} 차트 캡처 실패: {exc}")
        return None


def capture_multiple_timeframes(
    ticker: str,
    timeframes: list[str] = None,
    output_dir: str = "charts/screenshots",
    headless: bool = True,
) -> Dict[str, str]:
    """
    여러 타임프레임의 차트를 한 번에 캡처합니다.
    
    Args:
        ticker: 종목 심볼
        timeframes: 타임프레임 리스트 (기본값: 전체)
        output_dir: 저장 디렉토리
        headless: 브라우저 headless 모드 여부
        
    Returns:
        {timeframe: image_path} 딕셔너리
    """
    if timeframes is None:
        timeframes = ["1H", "4H", "Daily", "Weekly", "Monthly"]
    
    results = {}
    
    for tf in timeframes:
        path = capture_tradingview_chart(ticker, tf, output_dir, headless=headless)
        if path:
            results[tf] = path
    
    return results


if __name__ == "__main__":
    # 테스트
    print("TradingView 차트 캡처 테스트...")
    
    test_ticker = "AAPL"
    result = capture_tradingview_chart(test_ticker, "Daily", headless=False)
    
    if result:
        print(f"✅ 성공! 저장 위치: {result}")
    else:
        print("❌ 실패")
