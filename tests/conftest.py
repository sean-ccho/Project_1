"""pytest 공통 설정."""

# 손으로 돌리는 스크립트라 자동 테스트에서 뺀다 — 브라우저로 TradingView 차트를 캡처하거나 구글 드라이브에 올린다
collect_ignore = [
    "test_chart_capture.py",
    "test_chart_integration.py",
    "test_chart_simple.py",
    "test_drive_upload.py",
]
