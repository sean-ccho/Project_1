# CLAUDE.md — Project Rules

이 파일은 AI가 이 프로젝트에서 작업할 때 **항상** 따르는 규칙입니다.

---

## 0. 기본 정보

- **사용자 위치:** 캐나다 온타리오 (EST/EDT 시간대)
- **날짜 기준:** 항상 온타리오 현지 시간 기준으로 날짜를 표기한다

---

## 1. 문서 기록 규칙 (완화됨)

> **2026-04-18 변경:** 프로젝트가 안정기에 접어들어 자동 변경 로그 규칙을 **제거**했다.
> docs는 최소한으로 유지하며, 커밋마다 문서를 갱신하지 않는다.

### 현재 docs 구조 (최소 셋업)

| 파일 | 용도 |
|---|---|
| `docs/architecture.md` | 스크리너 + 페이퍼 트레이딩 통합 아키텍처 |
| `docs/screener_filtering_guide.md` | 필터링 가이드 + 점수 계산 원리 |
| `docs/paper_trading_candidate_selection.md` | 페이퍼 트레이딩 후보 선정 로직 |
| `docs/patterns_reference.md` | 차트 패턴 25종 레퍼런스 |
| `docs/Machine_Learning.md` | ML 통합 계획 (진행 중) |
| `docs/archive/` | 과거 기록 (진행 상황·최적화 로그·사건 기록 등) |

### 언제 문서를 수정하나?

**사용자가 명시적으로 요청할 때만.** 다음은 자동 기록하지 않는다:
- 일반 커밋 / 리팩토링 / 버그 수정
- 파라미터 조정 / 임계값 변경
- 새 파일 생성 (의미 있는 아키텍처 변경이 아닌 한)

### 문서를 수정해야 하는 경우 (드물게)

- **구조적 변경** (새 모듈/파이프라인 단계) → `architecture.md`
- **점수 계산 로직 변경** (가중치·조건) → `screener_filtering_guide.md`
- **페이퍼 트레이딩 로직 변경** (CCS·매매 조건) → `paper_trading_candidate_selection.md`
- **새 패턴 추가** → `patterns_reference.md`

---

## 2. 커밋 규칙

- 커밋 메시지: `type: 한국어 설명`
  - type: `feat` / `fix` / `docs` / `refactor` / `test` / `chore`
- CI 스킵이 필요한 경우 메시지 끝에 `[skip ci]` 추가
- 사용자가 특별히 요청하지 않는 한 **변경된 파일만** stage한다 (`git add <file>` 방식)

---

## 3. 코드 스타일

- Python: type hint 필수, docstring 작성 (한국어 or 영어 혼용 OK)
- 함수명/변수명: snake_case (영어)
- 컬럼명/레이블: 한국어 허용 (기존 코드베이스 관행 유지)

---

## 4. 프로젝트 컨텍스트

- **프로젝트:** 미국 주식 퀀트 트레이딩 시스템 (스크리너 + 백테스팅 + 페이퍼 트레이딩)
- **주요 파일:**
  - `src/screener/` — 스크리너 핵심 로직
  - `src/paper_trading/` — 페이퍼 트레이딩 엔진
  - `src/main.py` — 전체 파이프라인 진입점
  - `docs/architecture.md` — 전체 아키텍처 레퍼런스
- **자동화:** GitHub Actions (매일 5PM EST)
- **출력:** Google Sheets + 이메일 알림
