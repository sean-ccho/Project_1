# 2026-04-19 ~ 2026-05-24 주요 변경 로그

> 사용자 요청에 따라 일괄 docs 갱신(2026-05-24) 시 정리한 약 5주간의 굵직한 코드 변경 한 줄 요약.
> 일상 차트 스크린샷 자동 커밋(`Update chart screenshots …`)·페이퍼 트레이딩 상태 스냅샷(`update logs and paper trading state … [skip ci]`)은 제외.
> 카테고리별 시간 역순(최신 위) 정렬.

---

## 이메일 / 리포트

| 날짜 | 커밋 | 요약 |
|---|---|---|
| 2026-05-23 | `9bdb9efcb` | 시장 분석 박스를 단일 카드 가로 컬럼으로 통합 |
| 2026-05-22 | `01c508340` | 페이퍼 트레이딩 이메일 4개 테이블 통일 — 내부자(90일) 추가 + 골든크로스 컬럼 확장 |
| 2026-05-22 | `74004c23c` | 내부자 거래(openinsider) 90일 요약 컬럼 추가 — 이메일 + 차트분석 시트 |
| 2026-05-22 | `1adf2fa1e` | 페이퍼 트레이딩 이메일 컬럼 개편 + 삼각형 패턴 수평선 검증 개선 |
| 2026-05-19 | `f33bc7b5c` | 페이퍼 트레이딩 이메일에 골든크로스임박 섹션 추가 |
| 2026-05-19 | `2a99e0ad2` | 이메일 레이아웃 변경 — 차트 먼저 모아보기 / 뉴스 이후 모아보기 |
| 2026-05-19 | `cd0f8a96b` | 페이퍼 트레이딩 이메일에 지수(SP500/NASDAQ-NYSE) 컬럼 추가 |
| 2026-05-19 | `cea6459ab` | 티커 정렬 순서 변경 (모멘텀↓ → 바닥반등↓) |
| 2026-05-17 | `82dee8066` | 이메일 차트 향상 (improve Chart in Email) |
| 2026-05-17 | `3e3847b96` | 이메일 차트 향상 |
| 2026-04-20 | `7aa0b2ab5` | email updated & TradingView Screenshot Enabled |

## 패턴 / 시그널

| 날짜 | 커밋 | 요약 |
|---|---|---|
| 2026-05-18 | `9aa4bdef6` | 골든크로스 패턴 버그 수정 — DDOG 등 정배열 상태 종목을 "임박"으로 오판하던 문제 (50봉 sustained-below 검증 추가) |
| 2026-05-17 | `a40694d61` | 이동평균선 + MACD 추가 (Alpha 모델 추세 팩터에 MACD 25% 비중) |
| 2026-04-26 | `f851bc980` | 숏스퀴즈 전략 제거 + 잠복 버그 2건 수정 |

## 차트 캡처

| 날짜 | 커밋 | 요약 |
|---|---|---|
| 2026-05-16 | `881b401bf` | 차트 스크린샷에 이동평균선 오버레이 추가 |

## 페이퍼 트레이딩 로직

특별한 단일 커밋 없음 — 4월 15~16에 정착된 Hold-Winners Defer 로직이 그대로 운영 중. 2026-05 점검 결과:
- `HOLD_WINNERS_MIN_CHECKS = 5` (5/5 전부 통과 시 defer, all() 동등) 의도 확정
- `HOLD_WINNERS_DEFER_FREEZE_DAYS = 0` (freeze 비활성)
- 전략별 `EXIT_PARAMS`: 바닥반등 +18%/-10%/-6%/25일, 모멘텀 +12%/-10%/-5%/18일

## 인프라

| 날짜 | 커밋 | 요약 |
|---|---|---|
| 2026-04-19 | `83a1151e2` | Production Prep 2 |
| 2026-04-19 | `493e96072` | Production Prep |
| 2026-04-19 | `0cfbb1cdc` | fetch_ohlcv 배치 다운로드 적용 — 거래량 0 오탐 / 타임아웃 해결 |

---

## docs 갱신 메모

2026-05-24 일괄 docs 갱신 시점:
- `docs/architecture.md`: dual-output 시트 구조, insider/sector strength/extreme model/chart hosting, Hold-Winners 추가
- `docs/screener_filtering_guide.md`: MACD 25% / 골든크로스 50봉 검증 / 볼린저 바운스 / 5-support / 내부자(표시 전용) 명시
- `docs/paper_trading_candidate_selection.md`: **모멘텀 alpha 가중치 수치 정정** (0.35/0.30/0.20/0.10/0.05 → 0.20/0.20/0.25/0.15/0.20), Hold-Winners 5/5·freeze 0 반영, EXIT_PARAMS 전략별 명시
- `docs/design_spec.md`: 7-2-1·7-2-2 이메일 4테이블/골든크로스 표 섹션 추가
- `docs/Machine_Learning.md`, `docs/production_ready.md`: 2026-05-24 시점 상태 노트
- `docs/patterns_reference.md`, `docs/Pattern_Review.md`: 골든크로스 50봉 검증 / 후속 작업 메모
- `docs/strategy_overview.md`, `README.md`: 골든크로스임박 + 이메일 4테이블 통일 반영
- `docs/TO_DO.md`: 각 항목 상태(✅/⏳/⚠️) 갱신, 새 작업 추가 안 함
