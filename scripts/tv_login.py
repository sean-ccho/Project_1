import os
import sys
import time

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright가 설치되어 있지 않습니다. 'pip install playwright'를 먼저 실행하세요.")
    sys.exit(1)

def main():
    print("==================================================")
    print("TradingView 완벽 동기화 스크립트")
    print("==================================================")
    
    with sync_playwright() as p:
        # headless=False로 브라우저 창을 사용자에게 보여줌
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        
        print("\n브라우저가 열리면 다음 과정을 따라주세요:")
        print("1. 필요시 우측 상단 프로필/메뉴를 통해 로그인합니다.")
        print("2. 차트 화면에 캔들과 인디케이터(SeanEMA 등)가 정상적으로 다 렌더링될 때까지 기다립니다.")
        print("3. 모든 게 완벽하게 나오는 것을 눈으로 확인했다면, **직접 브라우저 창(X버튼)을 닫아주세요!**\n")
        
        # 아예 문제의 차트 ID 페이지로 바로 이동
        page.goto("https://www.tradingview.com/chart/gr1tsHcA/")
        
        state_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tv_state.json'))
        
        # 창이 닫힐 때까지 무한 대기 (최대 10분)
        closed_by_user = False
        for i in range(600):
            try:
                if page.is_closed():
                    closed_by_user = True
                    break
                time.sleep(1)
            except Exception:
                closed_by_user = True
                break
                
        if closed_by_user:
            print("\n브라우저 창이 닫힌 것을 감지했습니다. 현재 상태(쿠키+로컬스토리지)를 저장합니다...")
            try:
                context.storage_state(path=state_path)
                print(f"🎉 성공적으로 렌더링 상태와 쿠키를 모두 저장했습니다!\n경로: {state_path}")
            except Exception as e:
                print(f"상태 저장 중 오류: {e}")
        else:
            print("\n❌ 10분 초과로 자동 종료됩니다.")
            
        browser.close()

if __name__ == "__main__":
    main()
