# Stock Market Screener

Python 기반 주식 스크리너 파이프라인입니다.  
yfinance 데이터를 활용하여 기술·수급·재무 지표를 종합 분석하고, 매수 적합 종목을 Google Sheets 4개 시트에 자동 배포하며 이메일로 알립니다.

---

## 문서

| 파일 | 설명 |
|---|---|
| [strategy_overview.md](docs/strategy_overview.md) | ⭐ 전략 소개서 — 작동 원리 & 백테스트 수치 (신규 사용자 시작점) |
| [design_spec.md](docs/design_spec.md) | 전체 시스템 Design Specification |
| [architecture.md](docs/architecture.md) | 스크리너 + 페이퍼 트레이딩 통합 아키텍처 |
| [screener_filtering_guide.md](docs/screener_filtering_guide.md) | 필터링 가이드 + 점수 계산 원리 |
| [paper_trading_candidate_selection.md](docs/paper_trading_candidate_selection.md) | 페이퍼 트레이딩 후보 선정 로직 |
| [patterns_reference.md](docs/patterns_reference.md) | 차트 패턴 레퍼런스 (25종) |
| [Machine_Learning.md](docs/Machine_Learning.md) | ML 통합 계획 |

---

## 전체 구조

```
GitHub Actions (매일 5 PM EST 자동 실행)
│
├── main.py ─────────────────────────────────────────────▶ [SP500 분석]       시트
│   S&P 500 + Nasdaq 100 (~600개)                        [SP500 차트분석]    시트
│   · 5년 데이터, 별점 4.0+ 종목 이메일 [SP500 분석]    이메일
│   · 상위 종목 차트 캡처 (TradingView)
│
└── run_full_scan.py ────────────────────────────────────▶ [NASDAQ/NYSE 분석]     시트
    NYSE/NASDAQ 전 종목 (~4,000+개)                       [NASDAQ/NYSE 차트분석]  시트
    · 1년 데이터, 3단계 필터링                             이메일 [NASDAQ/NYSE 분석]
    · 별점 4.0+ 종목 이메일 및 차트 캡처
```

---

## Google Sheets 시트 구성

| 시트 | 소스 | 설명 |
|---|---|---|
| **[SP500 분석]** | `main.py` | S&P 500 + Nasdaq 100 전체 분석 결과 |
| **[SP500 차트분석]** | `main.py` | 별점 4.0+ 종목 + 고정 관심 종목(NBM.V)의 TradingView 차트 |
| **[NASDAQ/NYSE 분석]** | `run_full_scan.py` | 전 종목 스캔 상위 종목 분석 결과 |
| **[NASDAQ/NYSE 차트분석]** | `run_full_scan.py` | 별점 4.0+ 종목의 TradingView 차트 |

---

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# SP500 + Nasdaq 100 분석
PYTHONPATH=src python src/main.py

# 전 종목 스캔
PYTHONPATH=src python src/run_full_scan.py
```

`config.py`에서 Google Sheets 연동 정보(`GOOGLE_SHEETS_SPREADSHEET_ID`, `GOOGLE_SHEETS_CREDENTIALS_PATH`)와 이메일 설정(`EMAIL_*`)을 지정해야 합니다.

---

## 파이프라인 상세

### main.py — [SP500 분석] / [SP500 차트분석]

1. **데이터 수집** – 고정 티커풀(S&P 500 + Nasdaq 100 + ETF) 대상 5년 일봉 OHLCV 다운로드
2. **피처 계산** – 30+ 기술·수급·재무 지표 산출 (아래 참조)
3. **유동성 필터** – 20일 평균 거래대금 하위 25% 및 $5M 미만 제거
4. **섹터 중립화** – 시장/섹터 편향 제거 (`트렌드점수_최종`)
5. **신호 평가** – 바닥반등 / 모멘텀 별점 계산
6. **[SP500 분석] 업로드** + 이메일 발송 (별점 4.0+)
7. **[SP500 차트분석] 업로드** – 별점 4.0+ 종목 + NBM.V 대상 TradingView 차트 캡처 후 업로드

### run_full_scan.py — [NASDAQ/NYSE 분석] / [NASDAQ/NYSE 차트분석]

1. **1단계: 전 종목 수집** – Nasdaq FTP에서 NYSE/NASDAQ 전 종목 다운로드, 우선주/워런트/유닛/권리 사전 제거
2. **2단계: 빠른 필터링** – 가격 $1+, 거래량 100,000+ 이상만 통과, 1년 데이터 기반 바닥탈출/강세돌파 후보 선별 (rate limit 재시도 포함)
3. **3단계: 정밀 분석** – 통과 종목에 동일한 피처 계산 + 별점 평가
4. **[NASDAQ/NYSE 분석] 업로드** + 이메일 발송 (별점 4.0+)
5. **[NASDAQ/NYSE 차트분석] 업로드** – 별점 4.0+ 종목 TradingView 차트 캡처 후 업로드

---

## 이메일 알림

두 파이프라인 모두 별점 **4.0점(★★★★) 이상** 종목이 있을 때 자동 발송됩니다.

| 이메일 제목 | 소스 |
|---|---|
| `[SP500 분석] YYYY-MM-DD 전략별 매수 적합 종목 리스트` | `main.py` |
| `[NASDAQ/NYSE 분석] YYYY-MM-DD 전략별 매수 적합 종목 리스트` | `run_full_scan.py` |
| 페이퍼 트레이딩 일일 알림 | `paper_trading/runner.py` |

각 이메일은 **📉 바닥반등 전략**과 **🚀 모멘텀 전략** 두 섹션으로 구성됩니다.

### 페이퍼 트레이딩 이메일 (2026-05-22 통일)

페이퍼 트레이딩 이메일은 4개 테이블(매도/보유/후보/골든크로스) 컬럼이 통일돼 있다.

| 항목 | 설명 |
|---|---|
| **내부자(90일)** 컬럼 | 4테이블 공통 — `screener/insider.py::attach_insider_summary()`로 openinsider 클러스터 매수 90일 요약 |
| **골든크로스 표** | 골든크로스 완료 + 골든크로스임박 동시 노출. 임박은 단기 MA가 장기 MA 아래 50봉 이상 지속 시에만 인정 (2026-05-18 sustained-below) |
| **목표가 표기** | `+X% → $price` 형식으로 통일 |
| **차트·뉴스 모아보기** | 차트는 표 다음 한 곳에 모아 표시, 뉴스는 끝부분에 모아 표시 (2026-05-19) |
| **시장 분석 카드** | 단일 카드 가로 컬럼으로 통합 (2026-05-23) |
| **정렬** | 바닥반등 적합도 ↓ → 모멘텀 적합도 ↓ (2026-05-19) |

---

## 계산 파이프라인

### 피처 계산 (`features.py`)

| 카테고리 | 주요 지표 |
|---|---|
| **수익률** | 1/5/20/63/126일 수익률, ROC(10) |
| **추세** | EMA(20/50/200) 갭, ADX, MACD 히스토그램, RSI, 스토캐스틱 |
| **변동성** | ATR%, 볼린저 밴드, 변동성 압축 비율 |
| **거래량/수급** | 거래량 Z-score, OBV, CMF(20), A/D 기울기 |
| **캔들/장중** | 해머 캔들, 장중 반등률, 갭 하락률 |
| **펀더멘탈** | ROE, 부채비율, 배당수익률, 실적 발표 임박 여부 |

### 반등스코어 (최대 10점)

| 신호 | 점수 |
|---|:---:|
| RSI 과매도 후 반등 (≤30 터치 → ≥35) | +2.0 |
| 볼린저 하단 바운스 | +1.5 |
| 저점 거래량 급증 (MA20×1.5 이상) | +2.0 |
| 지지선 2회 테스트 (3% 이내) | +2.5 |
| MACD 다이버전스 | +2.0 |

### 알파 모델 팩터 (`alpha_model.py`)

매 실행 시 최근 60일 IC(정보계수)를 분석해 가중치를 자동 조절합니다.

| 팩터 | 사용 지표 |
|---|---|
| 모멘텀 | 20/63/126일 수익률 가중 평균 |
| 추세 | EMA 정배열 + ADX + MACD |
| 거래량 | OBV Z-score + CMF + 거래량 Z |
| 변동성 | ATR% 역수 (낮을수록 유리) |
| 평균회귀 | RSI 편차 + 볼린저 위치 |

---

## 두 전략 별점 시스템

### 📉 바닥반등 적합도

> "많이 빠진 종목이 바닥을 찍고 반등하는 타이밍"

| 조건 | 점수 |
|---|:---:|
| AI 저점확률 ≥ 40% | +2.5 |
| 반등스코어 ≥ 3.0 | +2.5 |
| RSI < 30 (과매도) | +1.5 |
| RSI 30~40 (탈출) | +1.0 |
| RSI 40~45 (중립 하단) | +0.5 |
| 상승형 차트 패턴 | +1.0 |
| 5일 수익률 < -8% | +0.5 |
| 강한 섹터 | +0.5 |
| 팩터_평균회귀 강/약 | +1.5 / +0.5 |
| 팩터_변동성 압축 | +0.5 |
| 주봉/월봉 패턴 | 최대 +3.0 |
| RSI > 75 / > 80 | -1.5 / -2.0 |
| 급등 (5일 > 20%) | -1.0 |
| 볼린저 과열 (pband > 0.95) | -1.0 |

### 🚀 모멘텀 적합도

> "이미 상승 추세가 확인된 종목의 강한 흐름에 올라타기"

| 조건 | 점수 |
|---|:---:|
| 매수 신호 (EMA+MACD+보조 3개+) | +2.5 |
| 트렌드점수 > 0.1 | +2.0 |
| EMA20 > EMA50 | +1.5 |
| 거래량 > MA20×1.2 | +1.0 |
| ADX > 25 | +1.0 |
| 강한 섹터 | +0.5 |
| 상승 차트 패턴 | +0.5 |
| 팩터_모멘텀 강/약 | +1.5 / +0.5 |
| 팩터_추세 | +1.0 |
| 주봉/월봉 패턴 | 최대 +3.0 |
| RSI > 75 | -1.5 |
| 급등 (5일 > 20%) | -1.0 |
| 볼린저 과열 | -1.0 |

### 별점 기준

| 별점 | 점수 |
|:---:|---|
| ★★★★★ | 5.0 이상 |
| ★★★★☆ | 4.0 ~ 4.9 |
| ★★★☆☆ | 3.5 ~ 3.9 |
| ★★☆☆☆ | 3.0 ~ 3.4 |
| ★☆☆☆☆ | 2.0 ~ 2.9 |
| ☆☆☆☆☆ | 2.0 미만 |

---

## GitHub Actions 자동 실행

매일 **5:00 PM EST (UTC 22:00)**에 `run-screener.yml` 워크플로우 하나로 두 파이프라인이 순서대로 실행됩니다.

```
① main.py       → [SP500 분석] + [SP500 차트분석] 업데이트 + 이메일
② run_full_scan.py → [NASDAQ/NYSE 분석] + [NASDAQ/NYSE 차트분석] 업데이트 + 이메일
```

`push to main` 시에도 자동으로 실행됩니다. (`[skip ci]` 커밋 메시지로 스킵 가능)

---

## 주요 설정 (`config.py`)

| 설정 | 설명 |
|---|---|
| `TICKERS` | SP500 + Nasdaq 100 분석 대상 |
| `LIQUIDITY_QUANTILE` | 유동성 하위 몇 % 제거 (기본 0.25) |
| `EMAIL_BOTTOM_SCORE_THRESHOLD` | 바닥반등 이메일 기준 (기본 4.0) |
| `EMAIL_MOMENTUM_SCORE_THRESHOLD` | 모멘텀 이메일 기준 (기본 4.0) |
| `GOOGLE_SHEETS_SIGNALS_WORKSHEET` | `[SP500 분석]` 시트명 |
| `GOOGLE_SHEETS_PORTFOLIO_WORKSHEET` | `[SP500 차트분석]` 시트명 |
| `GOOGLE_SHEETS_SIGNALS2_WORKSHEET` | `[NASDAQ/NYSE 분석]` 시트명 |
| `GOOGLE_SHEETS_SIGNALS2_CHART_WORKSHEET` | `[NASDAQ/NYSE 차트분석]` 시트명 |
| `TURNAROUND_MIN_DROP` | 전 종목 스캔 바닥탈출 최소 낙폭 (기본 -50%) |
| `MOMENTUM_HIGH_THRESHOLD` | 전 종목 스캔 강세돌파 52주 포지션 (기본 0.95) |
| `CHARTS_ENABLED` | TradingView 차트 캡처 on/off |
| `MARKET_FILTER_ENABLED` | SPY가 EMA200 아래일 때 신규 매수 신호 전체 차단 (기본 True) |
| `STRATEGY_MODE` | `"STANDARD"` vs `"AGGRESSIVE"` — AGGRESSIVE 시 매수신호 RSI 상한 60, ADX 하한 15로 완화 |
| `SECTOR_ROTATION_ENABLED` | 강한 섹터 종목만 매수 신호 허용 (기본 True) |

---

## 문제 해결

- **Rate limit 오류**: `run_full_scan.py`는 배치 크기 100, 배치 간 2초 딜레이, rate limit 발생 시 30/60/120초 자동 재시도가 적용되어 있습니다.
- **yfinance 다운로드 실패**: 간헐적으로 발생합니다. 잠시 후 재시도하거나 `TICKERS` 수를 줄이세요.
- **서비스 계정 권한 오류**: Google Sheets 공유 설정에서 서비스 계정을 편집 권한으로 추가해야 합니다.
- **실행 시간이 길다**: 전 종목 스캔은 수천 개 종목을 순차 처리하므로 수 시간이 소요될 수 있습니다.
