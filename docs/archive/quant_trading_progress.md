# 퀀트 트레이딩 애플리케이션 완성도 트래커

> **최종 업데이트:** 2026-04-04
> **전반적 완성도:** ~65%

---

## 변경 로그

---
*최근 변경: 2026-04-18 — Hold-Winners v2 완화 롤백 + 백테스트 meta.json 확장*
- **원인 진단**: 커밋 `23c84acc` 의 HW 조건 완화(`MIN_CHECKS=5→4`, `DEFER_FREEZE_DAYS=5` 도입)가 성과 회귀의 주범으로 확인. HW v1 (04-15): AvgRet +1.35% / WinRate 54.19% → HW v2 (04-16~): +0.49~0.51% / 48~49% 로 급락
- **롤백**: `HOLD_WINNERS_MIN_CHECKS=5`, `HOLD_WINNERS_DEFER_FREEZE_DAYS=0` 으로 환원 → HW v1 거동 복원 (engine.py 코드는 미변경, config 값만 환원)
- **meta.json 확장**: 앞으로 `output/runs/*/meta.json` 에 `flags`(include_fundamentals 등), `git`(sha/branch/dirty), `benchmark`(SPY대비 수익률/MDD/Sharpe), `hold_winners`(defer 발동 통계 + 현재 파라미터), `simulation` 블록 기록 → 로그만 보고 런 간 비교 가능
- **fundamentals 영향**: 04-16(OFF) +0.51% vs 04-17(ON) +0.49% — 주범 아님으로 확인. 다만 단독 격리 A/B 검증은 다음 백테스트 런 2회로 확인
- 관련 파일: `src/screener/config.py`, `src/paper_trading/backtest.py`

---
*최근 변경: 2026-04-16 — 백테스트 디스크 캐싱 도입 (재실행 속도 40-50시간 → ~5분)*
- **3단계 캐싱 구조**: OHLCV(24h parquet) → Fundamentals(1회 사전 로드) → Feature 스냅샷(날짜별 parquet)
- **핵심 개선**: `_compute_ranked_snapshot()` 분리 — feature 계산(캐싱 대상)과 scoring/필터(매번 실행) 분리
- **재실행 흐름**: 동일 커맨드 재실행 시 캐시 자동 감지 → `liquidity_filter → apply_neutralization → attach_signals_and_sort`만 재실행
- **`--no-cache` 플래그**: 기존 캐시 삭제 + 처음부터 전체 재계산 (캐시 갱신용)
- **임계치 변경 가이드**: `BACKTEST_*` 계열은 캐시 사용 가능. `RSI_*`, `MACD_*`, `ALPHA_*` 등 feature 계산 파라미터 변경 시 `--no-cache` 필요
- 캐시 저장 위치: `data/cache/` (`.gitignore` 등록)
- 관련 파일: `src/screener/cache.py` (신규), `src/data/fetch.py`, `src/screener/backtest.py`, `src/screener/fundamentals.py`, `src/screener/features.py`, `src/paper_trading/backtest.py`

---
*최근 변경: 2026-04-16 — ML 통합 계획 문서 신설 (`docs/Machine_Learning.md`)*
- **6단계 점진적 ML 롤아웃 계획 수립**: Bootstrap → Alpha 블렌딩 → Risk/Strategy 블렌딩 → 상승 예측 업그레이드 → 매도 타이밍 → 재학습 파이프라인
- **목표**: 승률 53% → 60%+, ML 신뢰도 기반 포지션 차등화, 시장 국면 자동 적응
- **전략**: 기존 규칙 유지한 채 ML 점수 블렌딩 (`final = w × ML + (1-w) × rules`) → 안전한 점진 도입
- **선결 조건**: 거래 250건+ (현재 202건 → `--period 5y --fundamentals` 백테스트로 확보 필요)
- 관련 파일: `docs/Machine_Learning.md` (신규)

---
*최근 변경: 2026-04-16 — Hold-Winners 로직 대폭 강화 + 이메일 upside 표시 + 안정화*

#### Hold-Winners 재평가 로직 개선 (`engine.py`, `config.py`)
- **RSI_MIN 제거**: 기존 `RSI >= 60` 조건 삭제 → `RSI < 75`(과매수 아님)만 체크  
  - 배경: RSI 55~60 구간이 오히려 상승 여지가 더 많음. ACHR(RSI 55), JOBY(RSI 51) 같은 케이스에서 불필요하게 defer 차단했던 문제 해결
- **다수결 defer 조건 (`MIN_CHECKS=4`)**: 기존 `all()` → 5개 조건 중 4개 이상 통과 시 defer 허용  
  - 배경: 거래량 하나가 약해도 나머지 모멘텀 신호가 강하면 defer 되어야 함 (ACHR 케이스)
- **defer freeze 기간 (`DEFER_FREEZE_DAYS=5`)**: defer 발동 후 5일간 시간익절/목표가 재발동 금지  
  - 배경: defer 후 다음날 또 `시간익절`이 발동되어 바로 2번째 defer를 소진하는 버그 방지  
  - 작동: freeze 중에는 tight trailing(-3.5%)과 손절만 작동, 5일 후 재평가 재개
- **defer 실패 로그 추가**: `DEFER SKIP {ticker}: rsi_not_overbought, volume_strong (RSI=55.0, ADX=28.3...)` 형태로 실패 원인 명시

#### 이메일 리포트 개선 (`exporter.py`)
- **보유 현황에 P50/P75 목표가 컬럼 추가**: 매수 당시 저장된 `predicted_upside_p50/p75` 기반 목표가 표시
- **후보 테이블에 upside 예측 추가**: 후보 종목별 `upside_model` 실시간 조회 → P50/P75 표시 (샘플 n≥5 기준)
- **보유 종목 있으면 이메일 항상 발송**: 기존에는 매수/매도/후보 모두 없으면 무조건 스킵 → 보유 종목 있으면 일일 현황 보고로 발송

#### 안정화 (`backtest.py`)
- **RuntimeWarning 수정**: `cumprod()` 전 inf/극단값 클리핑 (`clip(-0.99, 10.0)`) — `screener/backtest.py` + `paper_trading/backtest.py` 동시 적용
- **backtest.py defer 실패 로그 동기화**: `engine.py`와 동일한 DEFER SKIP 패턴으로 통일

#### 주요 파라미터 변경 요약
| 파라미터 | 이전 | 이후 | 의미 |
|----------|------|------|------|
| `HOLD_WINNERS_RSI_MIN` | 60.0 | 삭제 | RSI 하한 없앰 (낮을수록 여지 많음) |
| `HOLD_WINNERS_RSI_MAX` | 75.0 | 75.0 | 과매수 기준 유지 |
| `HOLD_WINNERS_MIN_CHECKS` | 미존재(all) | 4 | 5개 중 4개 통과 시 defer |
| `HOLD_WINNERS_DEFER_FREEZE_DAYS` | 미존재 | 5 | defer 후 5일 재발동 금지 |

---
*최근 변경: 2026-04-15 — Hold-Winners 재평가 + 상승 예측 모델 + 리포트 투명성 강화*
- **Hold-Winners 재평가 (engine.py/backtest.py)**: 목표가·시간익절 도달 시 즉시 매도 대신 당일 factor로 모멘텀 재평가. 강하면 최대 2회 defer + tight trailing stop(-3.5%). 손절/트레일링은 defer 금지.
- **상승 예측 모델 (upside_model.py 신규)**: 과거 백테스트 거래에서 (전략, CCS버킷, ★버킷)별 P25/P50/P75/P90 경험적 분포 빌드. 매수 payload에 예상 상승률 + 목표가 포함. 초기 캐시: 128 거래 / 18 버킷.
- **이메일/PDF 리포트 개선 (exporter.py/report_generator.py)**: 보유 현황에 현재가·수익률·재평가뱃지 추가. 고RSI 매수 rationale 박스. 예상 상승 분포(P25~P90 + 목표가) 박스.
- **이메일 수신자 추가 (config.py)**: `ssamjungtan@naver.com` 추가.
- **유닛 테스트 추가 (tests/paper_trading/)**: `test_hold_winners.py` 24개, `test_upside_model.py` 20개 — 전체 44/44 통과.
- **pytest 인프라 구축**: `pytest.ini` (pythonpath=src), `conftest.py` 추가.

---
*최근 변경: 2026-04-08 — 백테스트 버그 수정 및 성능 최적화*
- `candidate_selector.py:141` — Boolean Series 인덱스 불일치 버그 수정 (`strategy_col2.reindex(df.index)` 추가)
- `screener/backtest.py` — IC 가중치를 매일 재계산하던 것을 20거래일 주기 캐싱으로 변경 (`_IC_WEIGHTS_REFRESH_DAYS=20`)
- `screener/features.py` — `compute_features_snapshot`에 `ic_weights_override` 파라미터 추가
- `paper_trading/backtest.py` — `ic_weights_cache` 딕셔너리를 생성해 `_compute_ranked_snapshot`에 전달

---
*최근 변경: 2026-04-06 — 트레이드 진단 분석 프레임워크 구축 (Phase 1 & 2)*
- `backtest.py`: `BtPosition`에 `entry_features` 필드 추가, `_extract_entry_features()` 신규 (기술지표/알파팩터/CCS 서브스코어/시장상태/섹터 30+개 피처 캡처)
- `backtest.py`: 청산 후 가격 변동 추적 (`post_exit_return_5d/10d/20d`) — 손절 회복 / 익절 잔여수익 진단용
- `backtest.py`: `output/backtest_trades_enhanced.json` + `.csv` 자동 저장
- `scripts/analyze_trades.py` 신규 — 피처-수익률 상관관계, 청산사유 분석, 전략별 세부 분석, 시장/섹터 분석 4개 섹션
- 관련 파일: `src/paper_trading/backtest.py`, `scripts/analyze_trades.py`

---
*최근 변경: 2026-04-05 — 백테스트 전용 GitHub Actions 워크플로우 추가*
- `workflow_dispatch` 수동 트리거로 백테스트만 독립 실행 가능
- UI에서 period, rebalance, capital, max_tickers 파라미터 조정 가능
- 결과를 Job Summary + artifact 로그로 확인
- 관련 파일: `.github/workflows/run-backtest.yml`

---
*최근 변경: 2026-04-04 — 로그 구조 개선, 배치 파라미터 조정, 버그 수정*
- GitHub Actions: 로그를 `logs/YYYY-MM-DD/` 날짜별 폴더로 저장, 기존 루트 `.log` 방식 제거
- `.gitignore`: `logs/**/*.log` git 추적 허용 예외 추가
- `ticker_fetcher.py`, `run_full_scan.py`: 배치 크기 100→150, 딜레이 2.0→1.5s, 실패 배치 카운팅 로그 추가
- `candidate_selector.py`: 전략점수 필터 로직 리팩터 (`combine(max)`) + 점수 분포 로그 추가
- `paper_trading/backtest.py`, `screener/backtest.py`: MDD 계산 시 `returns.dropna()` 적용 (NaN 포함 버그 수정)
- `sp500_tickers.py`: `K`, `MMC` 제거
- 관련 파일: `src/data/ticker_fetcher.py`, `src/run_full_scan.py`, `src/paper_trading/candidate_selector.py`, `src/paper_trading/backtest.py`, `src/screener/backtest.py`

---
*최근 변경: 2026-04-04 — 페이퍼 트레이딩 수치 docs 동기화*
- 손절 -7% → -5%, 트레일링 -8% → -5%, 장기보유 14일→21일 / 3%→2%, CCS 교체 마진 0.05→0.10 반영 (config.py 기준)
- 관련 파일: `docs/quant_trading_progress.md`, `docs/architecture_diagram.md`

---
*최근 변경: 2026-04-04 — S&P500 전용 백테스팅 CLI 추가 및 기본 기간 2y 설정*
- S&P500 501개 전체를 유니버스로 사용하는 독립 백테스트 스크립트 생성
- 기본 기간 `2y` (1y는 워밍업 220일 제외 시 ~30거래일/6건으로 통계 신뢰 불가, 2y는 ~284거래일/50~80건)
- 확실한 검증 시 `--period 3y` 옵션 권장 (~100~150건)
- 기존 `run_paper_trading_backtest()` 재사용, 로직 중복 없음
- 관련 파일: `scripts/run_sp500_backtest.py`

---

### 2026-04-02 — paper_trading_candidate_selection.md 전면 상세화
- Phase 1 Hard Filters: 7개 필터를 각각 개별 섹션으로 분리, 목적/조건/탈락키/판단근거 추가
- Phase 2 CCS: 5개 서브스코어(A~E) 상세 계산식 + 예시 + 만점 구성 케이스 문서화
- Phase 3 레짐 감지: SPY EMA 기반 감지 코드 + 가중치 변화 의도 + bull/bear CCS 비교 예시 추가
- Phase 4 최종 선정: 동점 처리 3단계 + 타이브레이킹 로직 + 반환값 구조 주석 추가
- 엔진 연동: 매도 조건 4개 코드+예시, 교체 판단 흐름 + 수치 예시 추가
- 관련 파일: `docs/paper_trading_candidate_selection.md`

### 2026-04-02 — CLAUDE.md 생성
- 프로젝트 영구 규칙 파일 생성 (커밋/변경 시 자동 변경 로그 기록 규칙 포함)
- 관련 파일: `CLAUDE.md`

---

## 현재 구축된 부분 (강점)

| 영역 | 완성도 | 내용 |
|------|--------|------|
| 시그널 생성 | 90% | 30+ 기술 지표, 25개 차트 패턴, 역발상/모멘텀 전략 |
| 알파 모델 | 80% | 5-팩터 IC 기반 동적 가중치 모델 |
| 백테스팅 | 70% | ATR 포지션 사이징, Sharpe/MDD 계산, 다중 청산 조건 |
| 섹터 로테이션 | 75% | 11개 섹터 ETF 추적, SPY 200일선 시장 필터 |
| 데이터 파이프라인 | 65% | 4,000+ 종목 스캔, 기본적 분석 포함 |
| 자동화/알림 | 85% | GitHub Actions + 구글 시트 + 이메일 알림 |

---

## 미비 사항 체크리스트

퀀트 트레이딩 시스템으로 완성하기 위해 추가해야 할 항목들입니다.
(브로커 API 실거래 연동은 나중 단계 — 지금은 페이퍼 트레이딩으로 검증)

### 0. 자동 페이퍼 트레이딩 (Paper Trading) ← 구현 완료 (통합 테스트 대기)

스크리너가 매일 자동으로 매수/매도를 시뮬레이션하고 구글 시트에 기록.

**핵심 규칙:**
| 규칙 | 값 |
|------|-----|
| 최대 보유 종목 수 | 3개 |
| 일일 최대 매수 | 1개 (안 살 수도 있음) |
| 목표 보유 기간 | ~2주 |
| 매수가 기준 | 다음 거래일 시가(Open) |
| 최약 종목 판단 | 수익률 + 스크리너 점수 종합 |

**구현 체크리스트:**

#### Day 1: 기반 구조 ✅
- [x] `src/paper_trading/__init__.py` 생성
- [x] `src/paper_trading/portfolio.py` — JSON 기반 포지션 저장/로드
- [x] `data/paper_trading/` 디렉토리 생성
- [x] 단위 테스트: 포지션 추가/삭제/저장/로드 (9/9 통과)

#### Day 2: 의사결정 엔진 ✅
- [x] `src/paper_trading/candidate_selector.py` — 7단계 Hard Filter + 5-서브스코어 CCS (109→1) → [상세 문서](paper_trading_candidate_selection.md)
- [x] `src/paper_trading/engine.py` — 매수/매도/교체 로직
- [x] `check_sell_conditions()` — 목표가(+15%) / 손절(-5%) / 트레일링(-5%) / 장기보유(21일+수익<2%)
- [x] `should_replace()` — 보유 3개일 때 교체 판단 (CCS 마진 0.10)
- [x] `run_daily_trading()` — 전체 일일 로직 통합 (엔진 6/6 테스트 통과)

#### Day 3: 구글 시트 연동 ✅
- [x] `src/paper_trading/sheet_sync.py` — 3탭 동기화
- [x] `[페이퍼_거래로그]` — 매수/매도 기록 append
- [x] `[페이퍼_포지션현황]` — 현재 보유 종목 + 수익률 덮어쓰기
- [x] `[페이퍼_성과요약]` — 전략별/별점별 승률 집계

#### Day 4: 파이프라인 연결 + CLI ✅
- [x] `src/screener/config.py` — 페이퍼 트레이딩 설정 추가 (~30개 파라미터)
- [x] `src/main.py` — 스크리너 끝난 후 자동 호출
- [x] `scripts/paper_trading.py` — CLI (auto/status/history/reset/backtest)
- [x] `scripts/run_sp500_backtest.py` — S&P500 501개 전체 유니버스 전용 백테스트 CLI (argparse)

#### Day 5: 테스트 + 검증 ⏳
- [ ] 로컬에서 수동 실행 테스트 (30종목 dev 모드)
- [ ] 구글 시트 3탭 정상 생성 확인
- [ ] GitHub Actions에서 자동 실행 확인

**작동 Flow:**
```
매일 5PM EST (GitHub Actions)
  │
  ├─ 1. 스크리너 → "오늘의 최고 후보" 1개 선정 (★4 이상, 없으면 패스)
  │     매수가 = 다음 거래일 시가(Open)
  │
  ├─ 2. 보유 종목 현재가 업데이트 (yfinance)
  │
  ├─ 3. 매도 판단
  │     ├─ 목표가(+15%) → 자동 매도
  │     ├─ 손절(-5%) → 자동 매도
  │     ├─ 트레일링 스탑(고점 대비 -5%) → 자동 매도
  │     └─ 21일 초과 + 수익 < 2% → 자동 매도
  │
  ├─ 4. 매수 판단
  │     ├─ 보유 < 3개 → 후보 있으면 매수
  │     └─ 보유 = 3개 → 후보 > 최약 종목이면 교체
  │
  └─ 5. 구글 시트 3탭 업데이트
```

### 1. 포트폴리오 관리 (Portfolio Management)
- [ ] 종목 간 상관관계 매트릭스 추적
- [ ] 포트폴리오 리밸런싱 로직
- [ ] 현금 배분 규칙 (Cash Allocation Rules)
- [ ] Markowitz 효율적 프론티어 최적화
- [ ] 최대 보유 종목 수 제한 및 자동 교체

### 2. 백테스트 통계적 검증 (Statistical Validation)
- [ ] Walk-forward 테스트 (비표본 검증)
- [ ] 몬테카를로 시뮬레이션
- [ ] 파라미터 최적화 / 그리드 서치
- [ ] 유의성 검증 (p-value, Sharpe 신뢰 구간)
- [ ] 생존자 편향(Survivorship Bias) 보정

### 3. 거래 비용 모델링 (Transaction Cost Modeling)
- [ ] 슬리피지(Slippage) 모델링
- [ ] 수수료/세금 반영
- [ ] Bid-Ask 스프레드 영향
- [ ] 대형 포지션의 시장 충격(Market Impact) 계산

### 4. 실시간 데이터 (Real-time Data)
- [ ] 실시간 데이터 피드 연동
- [ ] 장중(Intraday) 신호 생성
- [ ] 틱(Tick) 데이터 분석
- [ ] 호가창(Level 2) 데이터

### 5. 고급 리스크 관리 (Advanced Risk Management)
- [ ] VaR / CVaR 계산
- [ ] 상관관계 기반 포트폴리오 리스크 지표
- [ ] 스트레스 테스트 / 시나리오 분석
- [ ] 헤지 구성 로직

### 6. 머신러닝 고도화 (ML Enhancement)
- [ ] LSTM / RNN 시계열 예측 모델
- [ ] 앙상블 방법론 (Random Forest, XGBoost)
- [ ] 피처 엔지니어링 자동화
- [ ] 모델 드리프트 감지 및 재학습 파이프라인

### 7. 대체 데이터 (Alternative Data)
- [ ] 뉴스 감성 분석 (News Sentiment)
- [ ] 소셜 미디어 감성 (Reddit, Twitter)
- [ ] 옵션 플로우(Options Flow) 추적
- [ ] 내부자 거래 데이터
- [ ] 거시경제 지표 연동

### 8. 데이터 인프라 (Data Infrastructure)
- [ ] 시계열 데이터베이스 도입 (TimescaleDB / InfluxDB)
- [ ] 데이터 검증 파이프라인
- [ ] 이상치 탐지
- [ ] 데이터 버전 관리 / 감사

---

## 개발 워크플로우 원칙

**핵심: 전체 파이프라인을 매번 돌리지 않는다**

### 데이터 캐싱
- `yfinance` 다운로드 결과를 `.parquet`으로 저장 → 재다운로드 없이 반복 테스트
- 캐시 경로: `data/cache/` (날짜별 저장)

### 개발용 소규모 유니버스
- `--dev` 플래그 또는 환경변수로 30종목만 실행
- 전략 로직 변경 시 30종목으로 빠르게 검증 → 통과하면 전체 실행

### 모듈 독립 실행
- 각 개선 사항은 독립 스크립트로 개발 및 테스트
  - `scripts/run_backtest_only.py` — 스크리너 백테스트 (config 기반)
  - `scripts/paper_trading.py` — 페이퍼 트레이딩 CLI (status/history/reset/backtest)
  - `scripts/run_sp500_backtest.py` — S&P500 전체 전용 백테스트 CLI (argparse)
  - `scripts/run_portfolio.py` — 신규 추가 예정

---

## 단계별 개발 로드맵

| 단계 | 목표 | 핵심 작업 | 예상 완성도 |
|------|------|-----------|------------|
| **Phase 1** | 자동 페이퍼 트레이딩 | 스크리너 연동 자동 매매 시뮬레이션 + 구글 시트 로그 | 70% |
| **Phase 2** | 백테스트 강화 | 슬리피지/수수료 반영, Walk-forward 검증 | 75% |
| **Phase 3** | 포트폴리오 관리 | 상관관계 추적, 리밸런싱, 자산배분 최적화 | 83% |
| **Phase 4** | ML 고도화 | 앙상블 모델, 피처 엔지니어링 자동화 | 90% |
| **Phase 5** | 실거래 연동 | 브로커 API, 자동 주문 집행 | 95% |

---

## 현재 시스템 정의

> 이 애플리케이션은 현재 **"퀀트 스크리너 + 백테스팅 시스템"**입니다.
> 신호를 생성하고 검증하는 리서치 툴로서는 완성도가 높으나,
> 실전 검증(페이퍼 트레이딩) 및 포트폴리오 관리 기능이 필요합니다.

- **퀀트 리서치 툴**: ✅ 완성 (~90%)
- **페이퍼 트레이딩 검증**: 🔨 구현 완료, 통합 테스트 대기 (Phase 1)
- **자동 퀀트 트레이딩**: ⏳ 장기 목표 (Phase 5)
