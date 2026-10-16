"""pytest 공통 설정."""

# 손으로 돌리는 스크립트라 자동 테스트에서 뺀다 — 브라우저로 TradingView 차트를 캡처하거나 구글 드라이브에 올린다
collect_ignore = [
    "test_chart_capture.py",
    "test_chart_integration.py",
    "test_chart_simple.py",
    "test_drive_upload.py",
]


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_live_sec_or_finra(monkeypatch):
    """테스트는 SEC·FINRA 에 진짜 요청을 보내지 않는다 (CI 는 작업 전체에 SEC_USER_AGENT 가 있다).

    필요한 테스트는 자기 안에서 가짜 클라이언트·데이터를 넘긴다.
    """
    monkeypatch.setenv("SEC_USER_AGENT", "")
    import screener.short_squeeze as ss
    monkeypatch.setattr(ss, "_CACHE", {"data": None})
