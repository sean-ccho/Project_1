# Design Specification — Project_1
### Quantitative Stock Screener & Paper Trading System

> **버전:** 2026-04-21  
> **대상 독자:** 개발자, 시스템 리뷰어  
> **Repository:** `sean-ccho/Project_1`

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [스크리너 파이프라인](#3-스크리너-파이프라인)
4. [페이퍼 트레이딩 엔진](#4-페이퍼-트레이딩-엔진)
5. [백테스트 모듈](#5-백테스트-모듈)
6. [데이터 흐름 & 상태 관리](#6-데이터-흐름--상태-관리)
7. [외부 연동](#7-외부-연동)
8. [모듈 맵](#8-모듈-맵)
9. [핵심 설정 상수](#9-핵심-설정-상수)
10. [주요 설계 결정 (Design Decisions)](#10-주요-설계-결정-design-decisions)

---

## 1. 시스템 개요

### 목적

Python 기반 자동화 주식 스크리닝 & 가상 매매(페이퍼 트레이딩) 시스템.  
매일 장 마감 후 S&P 500 / NASDAQ / NYSE 종목을 분석해 **매수 적합 후보를 선별**하고, 이를 Google Sheets와 이메일로 배포. 동시에 가상 포트폴리오를 운용해 전략의 실효성을 검증한다.

### 전략 철학

| 전략 | 설명 | 핵심 지표 |
|------|------|-----------|
| **📉 바닥반등 (Bottom Reversal)** | 과매도 구간에서 반등 타이밍 狙い | RSI < 40, 볼린저 하단, ML 저점확률 |
| **🚀 모멘텀 (Momentum)** | 확인된 상승 추세에 올라타기 | EMA 정배열, ADX > 25, 거래량 급증 |

두 전략은 **독립적으로 점수화**되어 단독 또는 **전환구간(Hybrid)** 으로 사용된다.

### 실행 환경

- **런타임:** Python 3.x, GitHub Actions (매일 5 PM EST, 평일)
- **데이터 소스:** yfinance (OHLCV), Nasdaq FTP (티커 목록), yfinance fundamentals
- **출력:** Google Sheets 4탭 + 이메일 알림 + PDF 리포트(페이퍼 트레이딩)

---

## 2. 전체 아키텍처

```
GitHub Actions (05:00 PM EST, Mon–Fri)
│
├─① SP500 파이프라인 (src/main.py)
│    고정 티커 ~600개 (S&P 500 + Nasdaq 100 + ETF)
│    5년 일봉 OHLCV
│    → [SP500 분석] 시트 + [SP500 차트분석] 시트 + 이메일
│
└─② 전체 시장 스캔 (src/run_full_scan.py)
     NYSE / NASDAQ 전 종목 ~4,000+개
     1년 일봉 OHLCV, 3단계 필터링
     → [NASDAQ/NYSE 분석] 시트 + [NASDAQ/NYSE 차트분석] 시트 + 이메일
│
└─③ 페이퍼 트레이딩 (src/paper_trading/runner.py)
     ① ② 분석 결과를 parquet로 읽어 가상 포트폴리오 운용
     → Google Sheets 3탭(거래로그/포지션현황/성과요약) + PDF 리포트
```

### 공통 분석 파이프라인 (두 스크리너 공유)

```
데이터 수집 (yfinance OHLCV)
    ↓
Feature 계산 (features.py)   ← 30+ 기술지표
    ↓
유동성 필터 (processing.py)  ← 거래대금 ≥ $5M / 하위 25% 제거
    ↓
섹터 중립화 (processing.py)  ← 60% 원본 + 25% 시장중립 + 15% 섹터중립
    ↓
Alpha 모델 (alpha_model.py)  ← 5팩터 IC-가중 합산 (60일 재계산)
    ↓
차트 패턴 감지 (patterns.py) ← 25종 패턴, 일/주/월 3 타임프레임
    ↓
신호 스코어링 (signals.py)   ← 바닥반등 / 모멘텀 독립 별점 계산
    ↓
출력 (exporter.py)           ← Google Sheets + 이메일 + Parquet 스냅샷
```

---

## 3. 스크리너 파이프라인

### 3-1. 데이터 수집

| 파이프라인 | 소스 | 기간 | 종목 수 |
|------------|------|------|---------|
| SP500 (`main.py`) | 고정 티커 풀 | 5년 일봉 | ~600 |
| 전체 시장 (`run_full_scan.py`) | Nasdaq FTP 자동 다운로드 | 1년 일봉 | ~4,000+ |

**전체 시장 스캔 3단계 필터:**

```
1단계 기본 필터   → 주가 ≥ $1, 거래량 ≥ 100K, 우선주·ETF·워런트 제외
2단계 후보 필터   → Bottom Reversal 또는 Momentum 조건 충족 여부
3단계 정밀 분석   → 통과 종목 대상 30+ 피처 계산 + 별점 평가
```

### 3-2. Feature 계산 (`features.py`)

| 카테고리 | 주요 지표 |
|----------|-----------|
| **수익률** | 1/5/20/63/126일 수익률, ROC(10) |
| **추세** | EMA(20/50/200) 갭, ADX, MACD 히스토그램, RSI, 스토캐스틱 |
| **변동성** | ATR%, 볼린저 밴드(pband), 변동성 압축 비율(ATR/ATR_med_252) |
| **거래량/수급** | 거래량 Z-score(20), OBV, CMF(20), A/D 기울기 |
| **캔들/장중** | 해머 캔들, 장중 반등률, 갭 하락률 |
| **펀더멘탈** | ROE, 부채비율, 배당수익률, 실적 발표 임박 여부, 기관보유%, 공매도% |

### 3-3. Alpha 모델 (`alpha_model.py`)

5개 팩터를 **IC(정보계수) 가중** 으로 합산. 가중치는 60일마다 재계산.

| 팩터 | 사용 지표 | 범위 |
|------|-----------|------|
| `팩터_모멘텀` | 20/63/126일 수익률 가중 평균 | [-1, 1] |
| `팩터_추세` | EMA 정배열 + ADX + MACD | [-1, 1] |
| `팩터_거래량` | OBV Z-score + CMF + 거래량 Z | [-1, 1] |
| `팩터_변동성` | ATR% 역수 (낮을수록 Good) | [-1, 1] |
| `팩터_평균회귀` | RSI 편차 + 볼린저 위치 | [-1, 1] |

### 3-4. 신호 스코어링 (`signals.py`)

#### 반등스코어 (최대 10점, 보조지표)

| 신호 | 점수 |
|------|:----:|
| RSI 과매도 후 반등 (≤30 → ≥35) | +2.0 |
| 볼린저 하단 바운스 | +1.5 |
| 저점 거래량 급증 (MA20×1.5+) | +2.0 |
| 지지선 2회 테스트 (3% 이내) | +2.5 |
| MACD 다이버전스 | +2.0 |

#### 바닥반등 적합도 (별점 원점수, 최대 ~12점)

| 조건 | 점수 |
|------|:----:|
| ML 저점확률 ≥ 40% | +2.5 |
| 반등스코어 ≥ 3.0 | +2.5 |
| RSI < 30 (과매도) | +1.5 |
| RSI 30~40 (탈출) | +1.0 |
| RSI 40~45 (중립 하단) | +0.5 |
| 상승형 차트 패턴 | +1.0 |
| 5일 수익률 < -8% | +0.5 |
| 강한 섹터 | +0.5 |
| 팩터_평균회귀 (강/약) | +1.5 / +0.5 |
| 팩터_변동성 압축 | +0.5 |
| 주봉/월봉 패턴 | 최대 +3.0 |
| RSI > 75 / > 80 | -1.5 / -2.0 |
| 급등 (5일 > 20%) | -1.0 |
| 볼린저 과열 (pband > 0.95) | -1.0 |

#### 모멘텀 적합도 (최대 ~12점)

| 조건 | 점수 |
|------|:----:|
| 매수 신호 (EMA+MACD+보조 3개+) | +2.5 |
| 트렌드점수 > 0.1 | +2.0 |
| EMA20 > EMA50 | +1.5 |
| 거래량 > MA20×1.2 | +1.0 |
| ADX > 25 | +1.0 |
| 강한 섹터 | +0.5 |
| 상승 차트 패턴 | +0.5 |
| 팩터_모멘텀 (강/약) | +1.5 / +0.5 |
| 팩터_추세 | +1.0 |
| 주봉/월봉 패턴 | 최대 +3.0 |
| RSI > 75 | -1.5 |
| 급등 (5일 > 20%) | -1.0 |
| 볼린저 과열 | -1.0 |

#### 별점 변환

| 별점 | 점수 범위 |
|:----:|-----------|
| ★★★★★ | 5.0 이상 |
| ★★★★☆ | 4.0 ~ 4.9 |
| ★★★☆☆ | 3.5 ~ 3.9 |
| ★★☆☆☆ | 3.0 ~ 3.4 |
| ★☆☆☆☆ | 2.0 ~ 2.9 |
| ☆☆☆☆☆ | 2.0 미만 |

이메일/차트 배포 기준: **4.0점(★★★★) 이상**

### 3-5. 차트 패턴 (`patterns.py`)

일봉·주봉·월봉 3개 타임프레임에서 **25종 패턴** 감지.

| 유형 | 패턴 예시 |
|------|-----------|
| 반전 | 이중바닥, 역헤드앤숄더, 하락쐐기, 강세잉걸핑, 모닝스타 |
| 지속 | 상승삼각형, 컵앤핸들 |
| 이평선 | 골든크로스, 골든크로스임박 |

---

## 4. 페이퍼 트레이딩 엔진

### 4-1. 시스템 파라미터

| 항목 | 값 |
|------|-----|
| 최대 보유 종목 | 3개 |
| 수익 목표 | +15% |
| 손절 | -5% |
| 트레일링 스탑 | -5% (고점 대비) |
| 최대 보유 기간 | 21일 |
| 실행 시각 | 매일 5 PM EST (평일) |
| 후보 선정 방식 | CCS (Composite Conviction Score) |
| 포지션 교체 조건 | 새 후보 CCS > 최약 CCS + 0.10 |

### 4-2. 일일 거래 로직 (`engine.py`)

**실행 순서: SELL → SELECT → BUY/REPLACE**

```
Step 1  보유 종목 현재가 + 고점 갱신
Step 2  매도 조건 체크 (check_sell_conditions)
Step 3  후보 선정 (select_best_candidate)
Step 4  매수 / 교체 판단
Step 5  positions.json / trades.json 저장
```

#### 매도 조건 4가지 (순서대로 체크, 하나 해당 시 즉시 매도)

| 우선순위 | 조건 | 매도 이유 |
|----------|------|-----------|
| 1 | `수익률 ≥ +15%` | 목표가 도달 |
| 2 | `수익률 ≤ -5%` | 손절 |
| 3 | `고점 대비 -5% 이하` | 트레일링 스탑 |
| 4 | `보유 21일 초과 AND 수익률 < +2%` | 장기보유 청산 |

#### 매수 / 교체 로직

```
빈 슬롯(보유 < 3)  →  즉시 신규 매수
풀 슬롯            →  새 CCS > 최약 CCS + 0.10 이면 교체, 아니면 SKIP
```

최약 포지션 선정 기준: `weakness = (1 - return_norm) × 0.6 + (1 - ccs_norm) × 0.4`

### 4-3. 후보 선정 엔진 — CCS (`candidate_selector.py`)

**파이프라인: N개 → 1개 최적 종목**

```
스크리너 DataFrame (100+ 종목)
    ↓
[Phase 1] Hard Filters (7개)       → 부적합 종목 제거
    ↓
[Phase 2] CCS 계산 (5개 서브스코어) → 통과 종목 점수화
    ↓
[Phase 3] 시장 레짐 감지 + 가중치 조정
    ↓
[Phase 4] 섹터 페널티 + 정렬 + 동점 처리
    ↓
최종 1개 선정 (또는 None)
```

#### Phase 1 — Hard Filters

| # | 필터 | 기준 | 탈락 키 |
|---|------|------|---------|
| 1 | 전략 점수 | 바닥반등 OR 모멘텀 ≥ 6.0점 | `전략점수<5.0` |
| 2 | 판단 등급 | `"1. 매수 후보"` / `"1. 저점 반등"` / `"1. 즉시 진입"` / `"1. 반등 매수"` | `판단등급_미달` |
| 3 | 과열 차단 | RSI < 75 AND pband < 0.95 AND 5일수익률 < 15% | `과열_차단` |
| 4 | 유동성 | 20일 평균 거래대금 ≥ $10M | `유동성_부족` |
| 5 | 어닝 회피 | 다음 실적 발표까지 > 3일 or NaN | `어닝_임박` |
| 6 | 보유 중복 | 이미 보유 중인 종목 제외 | `보유_중복` |
| 7 | 섹터 집중 | 동일 섹터 보유 2개 이상이면 해당 섹터 전체 차단 | `섹터_집중` |

> 약세장(bear)에서는 Filter 이후 추가 조건: `buy_signal == True` 종목만 통과

#### Phase 2 — CCS 5개 서브스코어 (각 [0, 1])

**A. Strategy Fit**
```
score = max(바닥반등_적합도, 모멘텀_적합도) / 10.0
전환구간 보너스: +0.1 (cap 1.0)
```

**B. Entry Timing** (4가지 합산, cap 1.0)

| 지표 | 바닥반등 | 모멘텀 | 최대 점수 |
|------|----------|--------|----------|
| RSI 스윗스팟 | 25~40: +0.4, 40~50: +0.2 | 45~60: +0.4, 35~45: +0.2 | +0.4 |
| 볼린저 위치 | pband<0.2: +0.3, <0.4: +0.15 | (동일) | +0.3 |
| 거래량 Z-score | Z>1.0: +0.15, Z>0: +0.05 | (동일) | +0.15 |
| MACD 히스토그램 | hist>-0.5: +0.1 | hist>0: +0.15 | +0.15 |

**C. Alpha Factor** (전략별 가중합산 후 [-1,1] → [0,1] 정규화)

| 팩터 | 바닥반등 | 모멘텀 |
|------|----------|--------|
| 모멘텀 | 10% | **35%** |
| 추세 | 10% | **30%** |
| 거래량 | 15% | 20% |
| 변동성 | **25%** | 10% |
| 평균회귀 | **40%** | 5% |

**D. Risk-Adjusted Quality** (합산, cap 1.0)

| 항목 | 조건 | 점수 |
|------|------|------|
| 변동성 압축 | ATR/ATR_med < 0.8: +0.3, < 1.0: +0.15 | 최대 +0.3 |
| 52주 포지션 | 바닥반등 10~40%: +0.2 / 모멘텀 55~85%: +0.2 | 최대 +0.2 |
| ROE | > 10% | +0.15 |
| 부채비율 | < 100 | +0.10 |
| 기관 보유 | > 60% | +0.10 |
| 공매도 | < 5%: +0.05 / > 15%: -0.05 | ±0.05 |

**E. Pattern & ML Confluence** (합산, cap 1.0)

| 항목 | 조건 | 점수 |
|------|------|------|
| ML 저점확률 | ≥ 0.70: +0.4, ≥ 0.50: +0.2 (바닥반등 전용) | 최대 +0.4 |
| 반등스코어 | `min(반등스코어/10, 1) × 0.2` | 최대 +0.2 |
| 멀티타임프레임 패턴 | 3개: +0.3, 2개: +0.15, 1개: +0.05 | 최대 +0.3 |
| EMA 정배열 | EMA20 > EMA50 > EMA200 | +0.10 |

#### Phase 3 — 시장 레짐 & 가중치

**레짐 감지 (SPY EMA 상태 기반)**

| 레짐 | 조건 |
|------|------|
| `bull` | SPY 종가 > EMA50 > EMA200 |
| `bear` | SPY 종가 < EMA200 |
| `neutral` | 그 외 |

**레짐별 CCS 가중치**

| 서브스코어 | bull | neutral | bear |
|-----------|:----:|:-------:|:----:|
| Strategy Fit | 0.25 | 0.25 | 0.20 |
| Entry Timing | 0.15 | 0.20 | **0.25** |
| Alpha Factor | **0.25** | 0.20 | 0.15 |
| Risk Quality | 0.15 | 0.20 | **0.30** |
| Confluence | 0.20 | 0.15 | 0.10 |

**최종 CCS 계산식:**
```
CCS = Σ(weight_i × subscore_i) - sector_penalty
```

#### Phase 4 — 섹터 페널티 & 동점 처리

**섹터 페널티**

| 동일 섹터 보유 수 | 페널티 |
|------------------|--------|
| 0개 | 0.00 |
| 1개 | -0.05 |
| 2개+ | -0.15 (안전망, 이론상 Filter 7에서 차단) |

**동점 처리 (CCS 차이 ≤ 0.02)**

| 우선순위 | 기준 | 방향 |
|----------|------|------|
| 1 | 전략 타입 | 전환구간(0) > 바닥반등(1) > 모멘텀(2) |
| 2 | `buy_support_count` | 높을수록 우선 |
| 3 | `ATR%` | 낮을수록 우선 |

**CCS 최소 임계값 (통과 기준)**

| 레짐 | 최소 CCS |
|------|----------|
| Bull / Neutral | 0.40 |
| Bear | 0.45 |

### 4-4. 상승 예측 모델 (`upside_model.py`)

과거 백테스트 거래에서 **(전략, CCS 버킷, ★ 버킷)** 조합별로 수익률 분포(P25/P50/P75/P90)를 구축.  
매수 payload에 `upside_pXX` 값 및 예상 목표가 포함.  
캐시 파일: `data/paper_trading/upside_distribution.json`

### 4-5. Hold-Winners 재평가

목표가(+15%) 또는 시간익절(21일) 트리거 시 바로 청산하는 대신,  
**당일 최신 팩터**로 모멘텀 재평가 후 강하면 **최대 2회 defer** + 타이트한 트레일링 스탑(-3.5%) 적용.

재평가 통과 조건:
- RSI 60~75
- ADX ≥ 20
- 거래량 ≥ 1.2×
- BB pband < 0.95
- 5일 수익률 > 0

> 손절 / 트레일링 스탑 조건은 defer 대상 아님.

---

## 5. 백테스트 모듈

### 5-1. 페이퍼 트레이딩 백테스트 (`src/paper_trading/backtest.py`)

라이브 매매 로직을 과거 OHLCV 데이터로 그대로 시뮬레이션.

| 항목 | 라이브 | 백테스트 |
|------|--------|---------|
| 데이터 소스 | 실시간 parquet 스냅샷 | 과거 OHLCV + `_compute_ranked_snapshot()` |
| 체결 가격 | 당일 현재가 | 다음 거래일 **시가** (슬리피지 반영) |
| 리밸런싱 | 매 거래일 | `rebalance_every` 거래일마다 |
| 포지션 저장 | positions.json | 메모리 `BtPosition` dataclass |
| 자본 모드 | — | `initial_capital=0` → % 모드 / `>0` → 달러 시뮬레이션 |

**출력:**
- `trades` — 전체 거래 로그
- `summary` — 승률, 평균수익, Sharpe, MDD, 전략별/매도사유별 breakdown
- `equity_curve` — 자본 시계열

### 5-2. 백테스트 CLI

| 스크립트 | 대상 | 특징 |
|---------|------|------|
| `scripts/run_sp500_backtest.py` | S&P 500 501개 전체 | `max_tickers=None`, 기본 2y |
| `scripts/paper_trading.py backtest` | 임의 유니버스 | 종목 수 / 기간 / 자본 등 유연한 옵션 |

**기간별 통계 신뢰도 (워밍업 220일 제외 기준)**

| 기간 | 실제 시뮬 일수 | 예상 거래 건수 | 신뢰도 |
|------|--------------|-------------|--------|
| 1y | ~30일 | ~6건 | ❌ 불충분 |
| 2y | ~284일 | ~50~80건 | ✅ 충분 |
| 3y | ~536일 | ~100~150건 | ✅✅ 확실 |

---

## 6. 데이터 흐름 & 상태 관리

### 6-1. 스냅샷 파일 (parquet)

```
src/main.py          → data/paper_trading/sp500_ranked.parquet
src/run_full_scan.py → data/paper_trading/nasdaq_ranked.parquet
```

`runner.py`의 `load_and_merge_snapshots()` 가 두 파일을 병합 (~600~1,000개).  
동일 티커 중복 시 `(바닥반등_적합도 + 모멘텀_적합도)` 합이 높은 행 유지.

### 6-2. 포지션 상태 파일 (JSON)

**`data/paper_trading/positions.json`** — 현재 보유 포지션

```json
[{
  "ticker": "ACHR",
  "entry_price": 5.35,
  "entry_date": "2026-04-02",
  "strategy": "📉 바닥반등",
  "star_rating": "★★★★★",
  "ccs_score": 0.4996,
  "sector": "Industrials",
  "highest_price": 5.42
}]
```

**`data/paper_trading/trades.json`** — 종료 포지션 누적 이력  
추가 필드: `exit_date`, `exit_price`, `return_pct`, `holding_days`, `exit_reason`

### 6-3. portfolio.py 주요 함수

| 함수 | 설명 |
|------|------|
| `add_position()` | positions에 신규 추가 |
| `close_position()` | positions 제거 → trades에 기록 |
| `update_highest_price()` | 신고가 갱신 |
| `get_worst_position()` | weakness 점수로 최약 포지션 선정 |

---

## 7. 외부 연동

### 7-1. Google Sheets

| 워크시트 | 업데이트 방식 | 주요 컬럼 |
|----------|--------------|----------|
| `[SP500 분석]` | Clear + 재작성 | 전체 분석 결과 |
| `[SP500 차트분석]` | Clear + 재작성 | 별점 4.0+ TradingView 차트 이미지 |
| `[NASDAQ/NYSE 분석]` | Clear + 재작성 | 전체 시장 스캔 결과 |
| `[NASDAQ/NYSE 차트분석]` | Clear + 재작성 | 별점 4.0+ TradingView 차트 이미지 |
| `페이퍼_거래로그` | Append-only | 날짜·티커·액션·매수/매도가·수익률·보유일·전략·별점·CCS·사유 |
| `페이퍼_포지션현황` | Clear + 재작성 | 현재가·수익률·재평가 배지 포함 |
| `페이퍼_성과요약` | Clear + 재작성 | 총 거래수·승률·평균수익률·누적수익률·전략/별점별 breakdown |
| `[백테스트_결과]` | Clear + 재작성 | 요약·전략별·매도사유·전체 거래 로그 |

### 7-2. 이메일 알림

| 이메일 제목 | 발송 조건 | 수신자 |
|------------|-----------|--------|
| `[SP500 분석] YYYY-MM-DD 전략별 매수 적합 종목 리스트` | 별점 4.0+ 종목 존재 | `chunghwan14@gmail.com`, `ssamjungtan@naver.com` |
| `[NASDAQ/NYSE 분석] YYYY-MM-DD ...` | 별점 4.0+ 종목 존재 | 동일 |
| 페이퍼 트레이딩 일일 알림 | 매수/매도 발생 시 | 동일 |

이메일 구성: **📉 바닥반등 전략** 섹션 + **🚀 모멘텀 전략** 섹션  
페이퍼 트레이딩 이메일: 고RSI 매수 rationale 박스 + 예상 상승 분포 박스(P25/P50/P75/P90 + 목표가) 포함

#### 7-2-1. 이메일 4테이블 통일 (2026-05-22, commit 01c508340)

페이퍼 트레이딩 이메일의 4개 테이블(매도/보유/후보/골든크로스) 컬럼이 통일됐다.

| 항목 | 변경 |
|---|---|
| **내부자(90일)** 컬럼 | 4테이블 모두에 추가 (openinsider 90일 클러스터 매수 요약) |
| 골든크로스 컬럼 | 폭 확장 + 임박/완료 동시 표시 |
| 목표가 표기 | `+X% → $price` 형식으로 통일 |
| 차트 / 뉴스 | "차트 모아보기" + "뉴스 모아보기" 섹션으로 재배치 (2026-05-19) |
| 시장 분석 박스 | 단일 카드 가로 컬럼으로 통합 (2026-05-23, commit 9bdb9efcb) |

#### 7-2-2. 골든크로스 표 (신규)

페이퍼 트레이딩 이메일에 별도 골든크로스 섹션이 추가됐다 (2026-05-19, commit f33bc7b5c).

| 항목 | 내용 |
|---|---|
| 신호 종류 | 골든크로스(완료) + 골든크로스임박 |
| 임박 검증 | SMA50/200 갭 2% 이내 + 수렴 중 + **단기 MA가 장기 MA 아래 50봉 이상 지속** (2026-05-18 sustained-below 추가) |
| 표 컬럼 | 티커, 전략, 별점, CCS, 거래량 z, 내부자(90일) |
| 정렬 | 바닥반등 적합도 ↓ → 모멘텀 적합도 ↓ (2026-05-19 변경) |

### 7-3. GitHub Actions

| 워크플로우 | 트리거 | 내용 |
|-----------|--------|------|
| `run-screener.yml` | 매일 5 PM EST (평일) + push to main | ① main.py → ② run_full_scan.py 순차 실행 |
| `run-backtest.yml` | 수동 (Actions 탭) | 파라미터 입력 후 백테스트 실행, Job Summary + artifact 로그 |

> `[skip ci]` 커밋 메시지로 자동 실행 스킵 가능.

### 7-4. TradingView 차트 캡처 (`charts/tradingview_capture.py`)

별점 4.0+ 종목 대상 1H / 4H / Daily / Weekly / Monthly 차트 스크린샷 캡처 후 Google Sheets 업로드.  
`CHARTS_ENABLED` 설정으로 on/off 가능.

---

## 8. 모듈 맵

### 스크리너

| 파일 | 역할 |
|------|------|
| `src/main.py` | SP500 파이프라인 진입점 |
| `src/run_full_scan.py` | 전체 시장 스캔 진입점 |
| `src/data/ticker_fetcher.py` | Nasdaq FTP 티커 다운로드 |
| `src/data/fetch.py` | yfinance OHLCV 다운로드 |
| `src/screener/features.py` | 30+ 기술지표 계산 |
| `src/screener/processing.py` | 유동성 필터 + 섹터 중립화 |
| `src/screener/alpha_model.py` | 5팩터 Alpha 모델 (IC 가중) |
| `src/screener/signals.py` | 바닥반등 / 모멘텀 신호 스코어링 |
| `src/screener/patterns.py` | 차트 패턴 25종 감지 |
| `src/screener/exporter.py` | Google Sheets + 이메일 + parquet 출력 |
| `src/screener/config.py` | 모든 파라미터 중앙 관리 |
| `src/screener/fundamentals.py` | 펀더멘탈 데이터 수집 |
| `src/screener/sector_rotation.py` | 강한 섹터 분류 |
| `charts/tradingview_capture.py` | TradingView 차트 스크린샷 |

### 페이퍼 트레이딩

| 파일 | 역할 |
|------|------|
| `src/paper_trading/run_paper_trading.py` | CLI 진입점 |
| `src/paper_trading/runner.py` | 통합 오케스트레이터 (스냅샷 병합 → engine 호출) |
| `src/paper_trading/engine.py` | 일일 거래 로직 (SELL → SELECT → BUY/REPLACE) |
| `src/paper_trading/portfolio.py` | JSON 포지션 / 이력 관리 |
| `src/paper_trading/candidate_selector.py` | CCS 기반 후보 선정 (Phase 1~4) |
| `src/paper_trading/sheet_sync.py` | Google Sheets 3탭 동기화 |
| `src/paper_trading/backtest.py` | 백테스트 엔진 |
| `src/paper_trading/upside_model.py` | 상승 예측 분포 모델 |
| `src/paper_trading/report_generator.py` | PDF 리포트 생성 |

### 데이터 파일

| 파일 | 내용 |
|------|------|
| `data/paper_trading/positions.json` | 현재 보유 포지션 |
| `data/paper_trading/trades.json` | 전체 거래 이력 |
| `data/paper_trading/sp500_ranked.parquet` | SP500 스냅샷 (매일 갱신) |
| `data/paper_trading/nasdaq_ranked.parquet` | NASDAQ 스냅샷 (매일 갱신) |
| `data/paper_trading/upside_distribution.json` | 상승 예측 분포 캐시 |

---

## 9. 핵심 설정 상수

> 모두 `src/screener/config.py`에 정의.

```python
# ── 포지션 관리 (기본값 — 전략별 EXIT_PARAMS가 오버라이드) ──
PAPER_TRADING_MAX_POSITIONS    = 3       # 최대 동시 보유 종목 수
PAPER_TRADING_PROFIT_TARGET    = 0.15   # 목표 수익률 +15% (전략별: 바닥반등 18% / 모멘텀 12%)
PAPER_TRADING_STOP_LOSS        = 0.07   # 손절 -7% (전략별 EXIT_PARAMS는 -10%)
PAPER_TRADING_TRAILING_STOP    = 0.05   # 트레일링 스탑 -5% (바닥반등은 -6%)
PAPER_TRADING_MAX_HOLDING_DAYS = 21     # 최대 보유 기간 (바닥반등 25 / 모멘텀 18)
PAPER_TRADING_STALE_MIN_RETURN = 0.02   # 장기보유 청산 최소 수익률 +2%

# ── 후보 선정 하드 필터 ──────────────────────
CANDIDATE_MIN_STRATEGY_SCORE   = 6.0
CANDIDATE_RSI_MAX              = 75
CANDIDATE_BOLLINGER_MAX        = 0.95
CANDIDATE_5D_RETURN_MAX        = 0.15
CANDIDATE_LIQUIDITY_MIN        = 10_000_000
CANDIDATE_EARNINGS_BUFFER_DAYS = 3
CANDIDATE_MAX_SAME_SECTOR      = 2

# ── CCS & 교체 ──────────────────────────────
CCS_REPLACE_MARGIN             = 0.10   # 교체 최소 마진
CANDIDATE_CCS_MIN_NORMAL       = 0.40   # Bull/Neutral 최소 CCS
CANDIDATE_CCS_MIN_BEAR         = 0.45   # Bear 최소 CCS

# ── 스크리너 전략 모드 ───────────────────────
MARKET_FILTER_ENABLED          = True   # SPY EMA200 하회 시 신호 차단
STRATEGY_MODE                  = "STANDARD"  # vs "AGGRESSIVE"
SECTOR_ROTATION_ENABLED        = True   # 강한 섹터 종목만 매수 신호 허용
CHARTS_ENABLED                 = True   # TradingView 차트 캡처 on/off
```

---

## 10. 주요 설계 결정 (Design Decisions)

| # | 결정 | 이유 |
|---|------|------|
| 1 | **CCS 임계값 Bear에서 높임** (0.40 → 0.45) | 약세장에서 잘못된 진입 최소화 |
| 2 | **레짐별 가중치 동적 조정** | 시장 상태에 따라 리스크/알파 중요도가 달라짐 |
| 3 | **섹터 이중 차단** (Hard Filter 7 + 섹터 페널티) | Edge Case까지 커버하는 방어적 설계 |
| 4 | **동점 처리 — 전환구간 우선** | 두 전략 신호가 겹치는 구간이 통계적으로 더 높은 정밀도 |
| 5 | **어닝 발표 3일 버퍼** | 발표 직전 방향성 불확실성 회피, 발표 후 결과 확인 후 진입 |
| 6 | **Trailing Stop과 목표가 defer 분리** | 수익 보호는 defer 불가, 익절 타이밍만 Hold-Winners 적용 |
| 7 | **스냅샷 parquet 병합** | 스크리너와 페이퍼 트레이딩 완전 분리 — 독립적으로 실행 가능 |
| 8 | **백테스트 슬리피지 — 다음 거래일 시가** | 진입가 낙관적 bias 제거 |
| 9 | **IC 기반 Alpha 팩터 가중치 60일 재계산** | 시장 국면에 따른 팩터 유효성 변화 자동 반영 |
| 10 | **이메일 수신자 2인** | 실시간 모니터링 용이성 및 알림 이중화 |
