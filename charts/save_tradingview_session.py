"""TradingView 로그인 세션 저장 스크립트.

persistent context(실제 브라우저 프로필)를 사용하여 Google 로그인 차단을 우회합니다.
로그인 후 세션을 auth.json으로 저장합니다.

사용법:
    python3 charts/save_tradingview_session.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("❌ Playwright가 설치되지 않았습니다.")
    print("   pip install playwright && playwright install")
    sys.exit(1)


AUTH_FILE = Path(__file__).parent / "auth.json"
USER_DATA_DIR = Path(__file__).parent / ".browser_profile"


def save_session():
    """TradingView에 로그인하고 세션을 auth.json으로 저장합니다."""
    print("=" * 60)
    print("TradingView 로그인 세션 저장")
    print("=" * 60)
    print()
    print("1. 브라우저가 열리면 TradingView에 로그인하세요.")
    print("   (Google 로그인, Email 로그인 모두 가능)")
    print("2. 로그인 완료 후 차트 페이지가 표시되면")
    print("   터미널에서 Enter를 눌러주세요.")
    print()

    with sync_playwright() as p:
        # persistent context: 실제 브라우저 프로필 사용 → Google 로그인 차단 우회
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
            ignore_default_args=["--enable-automation"],
        )
        page = context.new_page()

        # TradingView 로그인 페이지로 이동
        page.goto("https://www.tradingview.com/accounts/signin/", wait_until="domcontentloaded")

        input("\n✅ 로그인 완료 후 Enter를 눌러주세요... ")

        # 세션 저장
        context.storage_state(path=str(AUTH_FILE))
        print(f"\n💾 세션 저장 완료: {AUTH_FILE}")
        print("   이후 차트 캡처 시 자동으로 이 세션을 사용합니다.")

        context.close()


if __name__ == "__main__":
    save_session()
