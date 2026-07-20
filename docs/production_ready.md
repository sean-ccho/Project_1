# 프로덕션 레벨 업그레이드 계획

> **최종 업데이트:** 2026-04-18 / **상태 점검:** 2026-05-24
> **목표:** 퀀트 트레이딩 시스템을 프로덕션 레벨로 업그레이드
>
> **2026-05-24 시점 진행 상황**: Phase 1(인프라 안정화) 완료 상태 유지. Phase 2(테스트 커버리지 확장)와 Phase 3+(모니터링/장애 복원/리스크 레이어)는 미진행. 그동안의 변경은 주로 도메인 기능(이메일 4테이블 통일, 내부자 90일, 골든크로스 임박, MACD/차트 개선, Hold-Winners 튜닝)에 집중됨.

---

## 현재 상태 분석

| 영역 | 현재 수준 | 프로덕션 갭 |
|------|-----------|-------------|
| 전략 로직 | ⭐⭐⭐⭐⭐ 매우 성숙 | 거의 없음 |
| 자동화 (GH Actions) | ⭐⭐⭐⭐ 잘 구축 | 모니터링/알림 부족 |
| 에러 핸들링 | ⭐⭐⭐ 기본적 | 체계적 retry/fallback 필요 |
| 테스트 커버리지 | ⭐⭐ 부분적 | 핵심 로직 커버리지 확대 필요 |
| 설정 관리 | ⭐⭐ 하드코딩 혼재 | 환경별 분리 필요 |
| 보안 | ⭐⭐ 기본적 | secrets/credential 관리 강화 |
| 데이터 무결성 | ⭐⭐⭐ 보통 | 검증 파이프라인 필요 |
| 코드 품질 | ⭐⭐⭐ 양호 | 린팅/포맷팅 자동화 필요 |
| 문서화 | ⭐⭐⭐⭐ 우수 | API 문서 추가 |

---

## Phase 1: 인프라 안정화 ✅ 완료 (2026-04-18)

### 1.1 에러 핸들링 & 재시도 로직 ✅

**문제:** yfinance 다운로드 실패 시 전체 파이프라인이 중단될 수 있음

**해결:**
- `src/data/fetch.py`: yfinance 호출에 `tenacity` retry 적용 (3회 재시도, 지수 백오프)
- `engine.py`: 개별 종목 실패 시 skip → 나머지 정상 처리
- 부분 실패 허용 (Partial Failure Handling)

### 1.2 헬스 체크 & 모니터링 ✅

**문제:** 파이프라인 실패 시 다음 날까지 모름

**해결:**
- `src/monitoring/health_check.py` 신규 생성
- positions.json 파싱 가능 여부, Google Sheets 업데이트 확인
- `.github/workflows/run-screener.yml`: 실패 시 이메일 알림 step 추가 + 각 step 타임아웃 설정

### 1.3 데이터 무결성 검증 ✅

**문제:** OHLCV 데이터 NaN/이상치가 전략 로직에 영향

**해결:**
- `src/data/validators.py` 신규 생성
- NaN 비율, 가격 이상치, 거래량 0 연속일 체크
- positions.json 스키마 검증
- `src/data/fetch.py`에 검증 연동 완료

### 1.4 설정 관리 체계화 ✅

**문제:** 운영 설정이 코드에 하드코딩

**해결:**
- 핵심 운영 파라미터 환경변수 오버라이드 지원
- `.env.example` 템플릿 생성

### 1.5 의존성 고정 & 코드 품질 ✅

**해결:**
- `requirements-prod.txt` 정확한 버전 고정
- `pyproject.toml` ruff 린팅 설정
- `.github/workflows/run-screener.yml`에 pytest + ruff step 추가

### 1.6 포지션 파일 안전 장치 ✅

**문제:** positions.json 쓰기 중 프로세스 종료 → 파일 손상 가능

**해결:**
- `src/paper_trading/portfolio.py`: Atomic Write (임시파일 → rename) + 매 실행 전 자동 백업 (.bak)

---

## Phase 2: 코드 품질 & 테스트 (1-2주) ← 다음 진행

### 2.1 테스트 커버리지 확대

| 테스트 파일 | 검증 대상 | 우선순위 |
|---|---|---|
| `tests/screener/test_signals.py` | 시그널 스코어링 (각 패턴별 점수 계산) | 높음 |
| `tests/screener/test_features.py` | 기술적 지표 계산 (RSI, MACD, BB 등) | 높음 |
| `tests/paper_trading/test_candidate_selector.py` | Hard Filter 통과 조건 + CCS 순위 매기기 | 높음 |
| `tests/paper_trading/test_engine.py` | 매도 조건, 교체 로직, 엣지 케이스 (빈 포트폴리오, 동점 등) | 중간 |
| `tests/data/test_validators.py` | Phase 1에서 추가한 validators.py 검증 | 중간 |

- CI에서 `pytest` 자동 실행 (Phase 1에서 workflow에 이미 추가됨 ✅)

### 2.2 코드 린팅 자동화
- `ruff check` + `ruff format` CI 연동 (Phase 1에서 pyproject.toml 설정 완료 ✅)
- pre-commit hook 설정 (선택)

---

## Phase 3: 운영 안정성 (2-3주)

### 3.1 구조화 로깅

**현재:** `print()` + 기본 `logging` 혼용 → 로그 검색/분석 어려움

**목표:**
- `src/utils/logger.py` 신규 생성 — 프로젝트 공용 로거
- JSON 포맷 로그 출력 (`structlog` 또는 `python-json-logger`)
- 로그 레벨별 분리:
  - `DEBUG` — 로컬 개발 시 상세 출력
  - `INFO` — 프로덕션 실행 기록 (종목 수, 필터링 결과, 매매 실행)
  - `ERROR` — 알림 트리거 (이메일/Slack)
- GitHub Actions 로그에서 바로 검색 가능한 포맷

### 3.2 Graceful Degradation

**목표:** 단일 장애가 전체 파이프라인을 중단시키지 않도록 폴백 체계 구축

| 장애 시나리오 | 폴백 전략 | 구현 위치 |
|---|---|---|
| yfinance 데이터 수집 실패 | 캐시 데이터로 폴백 (24시간 이내) | `src/data/fetch.py` |
| Google Sheets API 실패 | 로컬 CSV 저장 + 다음 실행 시 retry | `src/output/sheets.py` |
| 이메일 발송 실패 | 로그 기록, 다음 실행에 누적 발송 | `src/output/email.py` |
| 특정 종목 데이터 누락 | 해당 종목만 skip (Phase 1에서 부분 구현 ✅) | `src/data/fetch.py` |

---

## Phase 4: 백테스트 신뢰성 강화 (3-4주)

### 4.1 거래 비용 모델링

**목표:** 백테스트 결과와 실거래 성과 간 갭 최소화

| 비용 항목 | 모델링 방식 | 기본값 |
|---|---|---|
| 슬리피지 | 매수 시 +0.1%, 매도 시 -0.1% | 0.1% |
| 수수료 | 설정 가능 파라미터 | $0 (IB 기준 $0.005/주) |
| 스프레드 | ATR 기반 동적 추정 | ATR × 0.02 |

- `src/backtesting/cost_model.py` 신규 생성
- 기존 백테스트 엔진에 cost_model 플러그인 방식 연동

### 4.2 Walk-Forward 검증

**목표:** 과적합 방지 — 미래 데이터 누출 없는 성과 검증

- Rolling Window: 학습 180일 → 테스트 60일 (슬라이딩)
- Out-of-Sample 성과 집계 → 리포트 자동 생성
- 전략 파라미터 안정성 확인 (파라미터 변경 시 성과 변동 폭)

---

## Phase 5: 실거래 준비 (4-8주)

### 5.1 브로커 API 연동

| 브로커 | API | 수수료 | 캐나다 사용 | 추천 |
|--------|-----|--------|------------|------|
| **Interactive Brokers** | TWS API / `ib_insync` | $0.005/주 | ✅ | ⭐ 최추천 |
| Alpaca | REST API | $0 | ❌ 미국만 | — |
| Questrade | REST API | $4.95 | ✅ | 대안 |

**구현 계획:**
- `src/broker/ib_adapter.py` — IB API 래퍼 (주문 제출, 상태 조회, 포지션 동기화)
- `src/broker/order_manager.py` — 주문 큐 관리 + 중복 방지
- 페이퍼 트레이딩 → 실거래 전환 스위치 (설정 기반)
- IB Paper Trading 계좌로 먼저 검증 → 실계좌 전환

### 5.2 리스크 관리 레이어

| 리스크 규칙 | 설명 | 트리거 시 행동 |
|---|---|---|
| 일일 최대 손실 | 포트폴리오 -3% 도달 | 신규 매수 중단 + 알림 |
| 포지션 사이징 | ATR 기반 동적 사이징 | 변동성 높은 종목 비중 축소 |
| 상관관계 모니터링 | 보유 종목 간 상관계수 > 0.7 경고 | 섹터 집중 리스크 알림 |
| 최대 포지션 수 | 동시 보유 상한 | 설정 파라미터로 제어 |

---

## 진행 상황 요약

### Phase 1 ✅ 완료 (2026-04-18)

- [x] `requirements-prod.txt` 버전 고정
- [x] `pyproject.toml` ruff 설정
- [x] `.env.example` 생성
- [x] GitHub Actions 타임아웃 + pytest + 헬스체크 + 실패알림
- [x] positions.json atomic write + 백업 (`portfolio.py`)
- [x] 데이터 무결성 검증 (`validators.py`)
- [x] `fetch.py` 검증 연동
- [x] 헬스 체크 모듈 (`health_check.py`)

### Phase 2 — 다음 목표

- [ ] `tests/screener/test_signals.py` 시그널 스코어링 테스트
- [ ] `tests/screener/test_features.py` 지표 계산 테스트
- [ ] `tests/paper_trading/test_candidate_selector.py` CCS 테스트
- [ ] `tests/paper_trading/test_engine.py` 엔진 엣지 케이스 테스트
- [ ] `tests/data/test_validators.py` 검증 모듈 테스트
- [ ] ruff CI 연동 확인 + pre-commit hook (선택)

### Phase 3~5 — 예정

- [ ] 구조화 로깅 (`structlog` / JSON 포맷)
- [ ] Graceful Degradation (캐시 폴백, CSV 저장)
- [ ] 슬리피지/수수료 모델 (`cost_model.py`)
- [ ] Walk-Forward 검증
- [ ] 브로커 API 연동 (Interactive Brokers)
- [ ] 리스크 관리 레이어
