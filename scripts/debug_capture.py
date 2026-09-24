import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    state_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tv_state.json'))
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) # 디버깅을 위해 브라우저 보이게
        
        context_args = {
            "viewport": {"width": 1920, "height": 1080},
            "device_scale_factor": 2,
        }
        if os.path.exists(state_path):
            print("Using saved state.")
            context_args["storage_state"] = state_path
            
        context = browser.new_context(**context_args)
        page = context.new_page()
        
        url = "https://www.tradingview.com/chart/gr1tsHcA/?symbol=PG&interval=D&theme=light"
        print("Navigating to", url)
        
        page.goto(url)
        print("Waiting 15 seconds for rendering...")
        time.sleep(15)
        
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'charts', 'screenshots', 'test', 'debug_full.png'))
        page.screenshot(path=out_path, full_page=True)
        print("Saved to", out_path)
        
        browser.close()

if __name__ == "__main__":
    main()
