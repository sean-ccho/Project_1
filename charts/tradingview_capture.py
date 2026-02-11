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
    
    # 티커 매핑 적용 (예: NBM.V -> NBM)
    try:
        from config import TRADINGVIEW_TICKER_MAP, TRADINGVIEW_CHART_ID
    except ImportError:
        TRADINGVIEW_TICKER_MAP = {}
        TRADINGVIEW_CHART_ID = None

    tv_ticker = TRADINGVIEW_TICKER_MAP.get(ticker, ticker)
    if tv_ticker != ticker:
        print(f"[TradingView] 티커 매핑 적용: {ticker} -> {tv_ticker}")
    
    # TradingView Widget Embed URL 생성
    # TradingView Widget Embed URL 생성
    if TRADINGVIEW_CHART_ID:
        # 저장된 차트(커스텀 지표) 사용 - 위젯이 아닌 실제 차트 페이지로 직접 이동
        url = f"https://www.tradingview.com/chart/{TRADINGVIEW_CHART_ID}/?symbol={tv_ticker}&interval={interval}&theme=light"
        print(f"[TradingView] 저장된 차트 레이아웃({TRADINGVIEW_CHART_ID}) 로딩 중... (Timeframe: {interval})")
        
        try:
            with sync_playwright() as p:
                # 브라우저 실행
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    device_scale_factor=2,  # Retina 디스플레이 품질
                )
                page = context.new_page()
                
                # 페이지 열기
                page.goto(url, wait_until="domcontentloaded", timeout=40000)
                
                # 차트 로딩 대기 (Full Page)
                try:
                    # 차트 캔버스 또는 메인 영역이 뜰 때까지 대기
                    page.wait_for_selector('div[class*="chart-container"]', timeout=30000)
                    time.sleep(5)  # 인디케이터 렌더링 대기
                except:
                    print(f"[TradingView] 차트 로딩 시간이 길어짐")
                
                # 팝업/광고 닫기 시도
                try:
                    page.click('button[class*="close"]', timeout=2000)
                except:
                    pass

                # 스크린샷 영역 지정 (레이아웃 중심부)
                # .layout__area--center 또는 canvas 컨테이너
                chart_element = page.query_selector('.layout__area--center')
                if not chart_element:
                    chart_element = page.query_selector('div[class*="chart-container"]')

                if chart_element:
                    chart_element.screenshot(path=str(output_path))
                else:
                    page.screenshot(path=str(output_path), full_page=False)
                
                print(f"[TradingView] 스크린샷 저장: {output_path}")
                browser.close()
                return str(output_path)

        except Exception as e:
            print(f"[TradingView] {ticker} 커스텀 차트 캡처 실패: {e}")
            return None

    else:
        # 기본 TEMA(9) 적용
        # 올바른 study ID: STD;TEMA
        url = (
            f"https://s.tradingview.com/widgetembed/"
            f"?symbol={tv_ticker}"
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
    여러 타임프레임의 차트를 한 번에 캡처합니다 (브라우저 세션 재사용).
    
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
    
    if sync_playwright is None:
        print("[TradingView] Playwright가 설치되지 않았습니다.")
        return {}

    # 출력 디렉토리 준비
    ticker_dir = Path(output_dir) / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    
    # 설정 로드
    try:
        from config import TRADINGVIEW_TICKER_MAP, TRADINGVIEW_CHART_ID
    except ImportError:
        TRADINGVIEW_TICKER_MAP = {}
        TRADINGVIEW_CHART_ID = None

    tv_ticker = TRADINGVIEW_TICKER_MAP.get(ticker, ticker)
    if tv_ticker != ticker:
        print(f"[TradingView] 티커 매핑 적용: {ticker} -> {tv_ticker}")

    try:
        with sync_playwright() as p:
            # 브라우저 실행 (한 번만)
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=2,
            )
            page = context.new_page()
            
            print(f"[TradingView] {ticker} 브라우저 세션 시작...")
            
            # 타임스탬프 생성 (구글 시트 캐싱 방지)
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for tf in timeframes:
                try:
                    interval = TIMEFRAME_MAP.get(tf, "D")
                    # 기존 파일 삭제 (공간 절약 및 최신 파일만 유지)
                    for old_file in ticker_dir.glob(f"{ticker}_{tf}_*.png"):
                        # 혹시 모를 에러 방지
                        try:
                            old_file.unlink()
                        except Exception as e:
                            print(f"[TradingView] 기존 파일 삭제 실패 ({old_file}): {e}")

                    # 새 파일명 (타임스탬프 포함)
                    output_path = ticker_dir / f"{ticker}_{tf}_{timestamp}.png"
                    
                    # URL 생성
                    if TRADINGVIEW_CHART_ID:
                        url = f"https://www.tradingview.com/chart/{TRADINGVIEW_CHART_ID}/?symbol={tv_ticker}&interval={interval}&theme=light"
                    else:
                        url = (
                            f"https://s.tradingview.com/widgetembed/"
                            f"?symbol={tv_ticker}"
                            f"&interval={interval}"
                            f"&hidesidetoolbar=1"
                            f"&symboledit=1"
                            f"&saveimage=1"
                            f"&studies=STD%3BTEMA"
                            f"&theme=light"
                            f"&style=1"
                            f"&timezone=Etc%2FUTC"
                        )
                    
                    # 페이지 이동 (같은 탭에서 URL만 변경 -> 리로딩)
                    # 주의: 위젯 임베드는 URL 파라미터 변경 시 새로고침이 필요함
                    # 하지만 브라우저 인스턴스는 유지되므로 속도 향상됨
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    
                    # 로딩 대기
                    try:
                        if TRADINGVIEW_CHART_ID:
                            page.wait_for_selector('div[class*="chart-container"]', timeout=30000)
                            # 인디케이터 로딩 등 추가 대기 (첫 로딩때만 길게, 이후엔 캐시 효과 기대)
                            time.sleep(3 if len(results) > 0 else 5)
                            
                            # 팝업 닫기 시도
                            try:
                                page.click('button[class*="close"]', timeout=1000)
                            except:
                                pass
                                
                            # 캡처
                            chart_element = page.query_selector('.layout__area--center')
                            if not chart_element:
                                chart_element = page.query_selector('div[class*="chart-container"]')
                                
                            if chart_element:
                                chart_element.screenshot(path=str(output_path))
                            else:
                                page.screenshot(path=str(output_path), full_page=False)
                                
                        else:
                            # 위젯 모드
                            time.sleep(2 if len(results) > 0 else 5) # 첫 로딩 이후엔 조금 더 짧게
                            
                            # 광고 닫기
                            try:
                                page.click('button[aria-label="Close"]', timeout=1000)
                            except:
                                pass
                                
                            chart_selector = 'div[data-role="chart"]'
                            chart_element = page.query_selector(chart_selector)
                            if chart_element:
                                chart_element.screenshot(path=str(output_path))
                            else:
                                page.screenshot(path=str(output_path), full_page=False)
                                
                        print(f"[TradingView] {tf} 캡처 완료 ({interval})")
                        results[tf] = str(output_path)
                        
                    except PlaywrightTimeout:
                        print(f"[TradingView] {ticker} {tf} 로딩 타임아웃")
                        continue
                        
                except Exception as e:
                    print(f"[TradingView] {ticker} {tf} 처리 중 에러: {e}")
                    continue
            
            browser.close()
            
    except Exception as e:
        print(f"[TradingView] 브라우저 실행 실패: {e}")
        
    return results


if __name__ == "__main__":
    # 테스트
    print("TradingView 차트 캡처 테스트 (세션 재사용)...")
    
    test_ticker = "AAPL"
    # test_timeframes = ["1H", "Daily"]
    test_timeframes = ["Daily"]
    
    results = capture_multiple_timeframes(test_ticker, test_timeframes, headless=False)
    
    if results:
        print(f"✅ 성공! {len(results)}개 파일 저장됨")
        for k, v in results.items():
            print(f"  - {k}: {v}")
    else:
        print("❌ 실패")
