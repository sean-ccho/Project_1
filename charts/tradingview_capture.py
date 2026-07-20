"""TradingView 차트 캡처 모듈 (Playwright 기반).

로그인 세션(auth.json)이 있으면 커스텀 인디케이터(SeanEMA 등) 포함 캡처,
없으면 위젯 임베드에서 표준 EMA를 createStudy로 추가하여 캡처합니다.
"""

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

# 인증 파일 경로
AUTH_FILE = Path(__file__).parent / "auth.json"

# 위젯 모드에서 추가할 EMA 설정 (auth.json 없을 때 사용)
WIDGET_EMA_PERIODS = [5, 10, 20, 50, 100, 200]
WIDGET_EMA_COLORS = [
    "#FF0000",   # 5 - 빨강
    "#FF8C00",   # 10 - 주황
    "#FFD700",   # 20 - 금색
    "#00BFFF",   # 50 - 파랑
    "#8A2BE2",   # 100 - 보라
    "#0000FF",   # 200 - 진파랑
]


def _build_widget_html(ticker: str, interval: str) -> str:
    """TradingView 위젯을 포함한 HTML 페이지를 생성합니다.

    createStudy API로 EMA 인디케이터를 프로그래매틱하게 추가합니다.
    """
    # EMA createStudy 호출 생성
    ema_studies = ""
    for period, color in zip(WIDGET_EMA_PERIODS, WIDGET_EMA_COLORS):
        ema_studies += f"""
                widget.chart().createStudy(
                    "Moving Average Exponential", false, false,
                    [{period}],
                    {{"plot.color": "{color}", "plot.linewidth": 2}}
                );"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; background: #fff; }}
        #tradingview_chart {{ width: 1920px; height: 1080px; }}
    </style>
</head>
<body>
    <div id="tradingview_chart"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
        var widget = new TradingView.widget({{
            "container_id": "tradingview_chart",
            "symbol": "{ticker}",
            "interval": "{interval}",
            "timezone": "America/New_York",
            "theme": "light",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": false,
            "save_image": false,
            "width": "1920",
            "height": "1080",
            "autosize": false
        }});

        widget.onChartReady(function() {{
            // EMA 인디케이터 추가{ema_studies}

            // EMA 로딩 완료 마커
            setTimeout(function() {{
                document.title = "CHART_READY";
            }}, 3000);
        }});
    </script>
</body>
</html>"""
    return html


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
        from screener.config import TRADINGVIEW_TICKER_MAP, TRADINGVIEW_CHART_ID
    except ImportError:
        TRADINGVIEW_TICKER_MAP = {}
        TRADINGVIEW_CHART_ID = None

    tv_ticker = TRADINGVIEW_TICKER_MAP.get(ticker, ticker)
    if tv_ticker != ticker:
        print(f"[TradingView] 티커 매핑 적용: {ticker} -> {tv_ticker}")
    
    # 인증 파일 확인
    has_auth = AUTH_FILE.exists()

    # TradingView Widget Embed URL 생성
    if TRADINGVIEW_CHART_ID and has_auth:
        # 저장된 차트(커스텀 지표) 사용 - 로그인 세션 포함
        url = f"https://www.tradingview.com/chart/{TRADINGVIEW_CHART_ID}/?symbol={tv_ticker}&interval={interval}&theme=light"
        print(f"[TradingView] 저장된 차트 레이아웃({TRADINGVIEW_CHART_ID}) + 로그인 세션 로딩 중... (Timeframe: {interval})")
        
        try:
            with sync_playwright() as p:
                # 브라우저 실행 (로그인 세션 포함)
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    device_scale_factor=2,  # Retina 디스플레이 품질
                    storage_state=str(AUTH_FILE),
                )
                page = context.new_page()
                
                # 페이지 열기
                page.goto(url, wait_until="domcontentloaded", timeout=40000)
                
                # 차트 로딩 대기 (Full Page)
                try:
                    # 차트 캔버스 또는 메인 영역이 뜰 때까지 대기
                    page.wait_for_selector('div[class*="chart-container"]', timeout=30000)
                    time.sleep(8)  # 인디케이터 렌더링 대기 (이동평균선 포함)
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

    elif TRADINGVIEW_CHART_ID and not has_auth:
        # auth.json 없음 → 위젯 모드로 폴백 (EMA createStudy)
        print(f"[TradingView] ⚠️  auth.json 없음 → 위젯 모드(표준 EMA {WIDGET_EMA_PERIODS})로 폴백")
        print(f"[TradingView]    커스텀 인디케이터를 사용하려면: python charts/save_tradingview_session.py")
        return _capture_with_widget_ema(tv_ticker, interval, timeframe, output_path, headless)

    else:
        # 기본 위젯 모드 (EMA createStudy)
        return _capture_with_widget_ema(tv_ticker, interval, timeframe, output_path, headless)


def _capture_with_widget_ema(
    tv_ticker: str,
    interval: str,
    timeframe: str,
    output_path: Path,
    headless: bool,
) -> str | None:
    """위젯 임베드에서 createStudy API로 EMA를 추가하여 캡처합니다."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=2,
            )
            page = context.new_page()

            print(f"[TradingView] {tv_ticker} {timeframe} 차트 로딩 중 (EMA {WIDGET_EMA_PERIODS})...")

            # 로컬 HTML 생성 및 로드
            html_content = _build_widget_html(tv_ticker, interval)
            page.set_content(html_content, wait_until="domcontentloaded")

            # EMA 로딩 완료 대기 (title이 CHART_READY로 변경될 때까지)
            try:
                page.wait_for_function("document.title === 'CHART_READY'", timeout=30000)
                time.sleep(3)  # 추가 렌더링 대기
            except:
                print(f"[TradingView] EMA 로딩 타임아웃, 기본 대기로 진행")
                time.sleep(8)

            print(f"[TradingView] EMA 인디케이터 로딩 완료")

            # 광고/팝업 닫기 (있다면)
            try:
                page.click('button[aria-label="Close"]', timeout=2000)
            except:
                pass

            # 차트 영역만 스크린샷
            chart_element = page.query_selector('#tradingview_chart')
            if chart_element:
                chart_element.screenshot(path=str(output_path))
            else:
                page.screenshot(path=str(output_path), full_page=False)

            print(f"[TradingView] 스크린샷 저장: {output_path}")
            browser.close()
            return str(output_path)

    except PlaywrightTimeout:
        print(f"[TradingView] {tv_ticker} 차트 로딩 타임아웃")
        return None
    except Exception as exc:
        print(f"[TradingView] {tv_ticker} 차트 캡처 실패: {exc}")
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
        from screener.config import TRADINGVIEW_TICKER_MAP, TRADINGVIEW_CHART_ID
    except ImportError:
        TRADINGVIEW_TICKER_MAP = {}
        TRADINGVIEW_CHART_ID = None

    tv_ticker = TRADINGVIEW_TICKER_MAP.get(ticker, ticker)
    if tv_ticker != ticker:
        print(f"[TradingView] 티커 매핑 적용: {ticker} -> {tv_ticker}")

    # 인증 파일 확인
    has_auth = AUTH_FILE.exists()
    use_saved_chart = bool(TRADINGVIEW_CHART_ID and has_auth)

    if TRADINGVIEW_CHART_ID and not has_auth:
        print(f"[TradingView] ⚠️  auth.json 없음 → 위젯 모드(표준 EMA {WIDGET_EMA_PERIODS})로 폴백")
        print(f"[TradingView]    커스텀 인디케이터를 사용하려면: python charts/save_tradingview_session.py")

    try:
        with sync_playwright() as p:
            # 브라우저 실행 (한 번만)
            browser = p.chromium.launch(headless=headless)

            context_kwargs = {
                "viewport": {"width": 1920, "height": 1080},
                "device_scale_factor": 2,
            }
            if use_saved_chart:
                context_kwargs["storage_state"] = str(AUTH_FILE)

            context = browser.new_context(**context_kwargs)
            page = context.new_page()
            
            print(f"[TradingView] {ticker} 브라우저 세션 시작{'  (로그인 세션)' if use_saved_chart else '  (위젯 모드)'}...")
            
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
                    
                    if use_saved_chart:
                        # ━━━ 저장된 차트 모드 (로그인 세션 포함) ━━━
                        url = f"https://www.tradingview.com/chart/{TRADINGVIEW_CHART_ID}/?symbol={tv_ticker}&interval={interval}&theme=light"
                        page.goto(url, wait_until="domcontentloaded", timeout=30000)

                        try:
                            page.wait_for_selector('div[class*="chart-container"]', timeout=30000)
                            # 인디케이터 로딩 대기 (첫 로딩때만 길게)
                            time.sleep(4 if len(results) > 0 else 8)

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

                        except PlaywrightTimeout:
                            print(f"[TradingView] {ticker} {tf} 로딩 타임아웃")
                            continue

                    else:
                        # ━━━ 위젯 모드 (EMA createStudy) ━━━
                        html_content = _build_widget_html(tv_ticker, interval)
                        page.set_content(html_content, wait_until="domcontentloaded")

                        try:
                            page.wait_for_function("document.title === 'CHART_READY'", timeout=30000)
                            time.sleep(2 if len(results) > 0 else 4)
                        except:
                            time.sleep(6)

                        # 광고 닫기
                        try:
                            page.click('button[aria-label="Close"]', timeout=1000)
                        except:
                            pass

                        chart_element = page.query_selector('#tradingview_chart')
                        if chart_element:
                            chart_element.screenshot(path=str(output_path))
                        else:
                            page.screenshot(path=str(output_path), full_page=False)

                    print(f"[TradingView] {tf} 캡처 완료 ({interval})")
                    results[tf] = str(output_path)
                        
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
