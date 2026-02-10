# TradingView Chart Screenshots Setup

## 로컬 테스트 방법

### 1. Playwright 설치
```bash
pip install playwright
playwright install chromium
```

### 2. 테스트 실행
```bash
python test_chart_capture.py
```

브라우저가 자동으로 열리고 TradingView 차트를 캡처합니다.

### 3. 결과 확인
캡처된 이미지는 `charts/screenshots/` 폴더에 저장됩니다.

---

## 주의사항

- **첫 실행**: Playwright가 Chromium 브라우저를 다운로드합니다 (~100MB)
- **헤드리스 모드**: `headless=True`로 설정하면 백그라운드에서 실행
- **타임아웃**: 차트 로딩에 시간이 걸릴 수 있습니다 (5-10초/차트)

---

## GitHub Actions에서 실행

워크플로우에 다음 단계가 추가되었습니다:
```yaml
- name: Install Playwright browsers
  run: |
    playwright install chromium
    playwright install-deps chromium
```

이제 GitHub Actions에서도 자동으로 차트를 캡처할 수 있습니다!
