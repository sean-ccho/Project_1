# Trend Ranking Screener

Python 기반 트렌드/저점 탐지 스크립트입니다. yfinance 일봉 데이터를 내려받아 기술 지표·거래/자금 흐름·기초 재무 지표를 종합 평가하고, 보유주식을 위한 Google Sheets `"주식찾기"` 워크시트를 업데이트합니다.

---
## 빠른 시작
1. 프로젝트 루트에서 가상환경 생성 및 활성화
   ```bash
   python3 -m .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 환경 설정(`config.py`)에서 필요한 값 수정
   - `TICKERS`: 분석할 티커 목록 (기본 S&P500)
   - `GOOGLE_SHEETS_*`: 서비스 계정 JSON 경로와 스프레드시트 ID 등
3. 실행
   ```bash
   python main.py
   ```
4. 종료 후 `deactivate`로 가상환경을 빠져나옵니다.

콘솔에는 업로드 결과와 총 실행 시간이 출력되고, 세부 결과는 Google Sheets에 반영됩니다.

---
## 파이프라인 개요
1. **데이터 수집** – `data.fetch.fetch_ohlcv`가 5년 일봉 OHLCV와 배당 정보를 다운로드합니다. 다운로드 실패 시 `MAX_DOWNLOAD_RETRIES`와 `RETRY_DELAY_SECONDS` 동안 재시도합니다.
2. **특징 생성(`features.py`)**
   - 수익률 & 모멘텀
     - 1일/5일/20일/63일 수익률, 10일 ROC, RSI, MACD(+시그널/히스토그램), Stochastic %K/%D
     - 5일 대비 20일 수익률 가속도(`accel`)로 단·중기 모멘텀 변화를 추적
   - 추세 & 변동성
     - 트렌드 점수 (가중합), ADX(+DI/-DI), 52주 포지션, EMA(20/50/200) 갭, EMA200 기울기, ATR%, Bollinger/Keltner 밴드 위치·폭, EMA200 이탈률, 10일 저점 괴리
     - ATR%가 `VOLATILITY_PENALTY_START~END` 범위를 벗어나면 비선형 패널티 부여
   - 거래량 & 자금 흐름
     - 20일 거래량 Z-score, 60일 대비 거래대금 안정 비율, Chaikin Money Flow(20), OBV Z-score, Acc/Dist 기울기, 장중 반등률
   - 이벤트 & 패턴
     - 갭 하락률, 망치형(반전) 캔들 탐지, 최근 실적 발표 일정(Days to Earnings)
   - 재무 체력(`fundamentals.py`)
     - ROE, 부채/자본 비율, 매출/이익 성장률, 이익률, 유동·당좌 비율, 자유/영업 현금흐름, 다음 실적 발표일 등
   - 상대 강도
     - 20일 수익률 기준 시장/섹터 평균 대비 초과 성과(`시장상대강도`, `섹터상대강도`)
3. **유동성 필터(`processing.liquidity_filter`)** – 시장별 20일 평균 거래대금 분위수 하위 `LIQUIDITY_QUANTILE` 비율을 제거하고, 절대 거래대금이 `LIQUIDITY_DOLLAR_MIN` 미만이면 제외합니다. `LIQUIDITY_WHITELIST`는 항상 유지됩니다.
4. **중립화(`processing.apply_neutralization`)** – 트렌드 점수를 시장/섹터 z-score로 스무딩하여 비교 가능성을 높입니다.
5. **랭킹 정렬(`main.py`)**
   - 유동성/중립화가 끝난 스냅샷을 `트렌드점수_최종` 등 핵심 지표 기준으로 정렬합니다.
   - `features.py`에서 계산한 스퀴즈/RSI 신호를 일봉·주봉·월봉까지 확장해 변동성 압축 상태와 중기/장기 모멘텀을 함께 확인합니다.
6. **출력(`main.py`)**
   - `exporter.prepare_export_dataframe`가 최종 테이블을 정돈한 뒤 Google Sheets에 업로드합니다.
   - 현재 버전은 `GOOGLE_SHEETS_PORTFOLIO_WORKSHEET`(기본 `"주식찾기"`)에 적힌 티커 목록만 업데이트하며, 보유 종목은 유동성 필터를 거치지 않고 모두 유지합니다.
   - Google Sheets 업로드 성공 여부를 콘솔에 출력하고, 전체 파이프라인 수행 시간을 요약합니다.

---
## Google Sheets로 전달되는 컬럼
아래 항목만 `config.EXPORT_COLUMNS`에 정의되어 있으며, 시트에서도 같은 순서로 표시됩니다.

| 컬럼 | 설명 |
| --- | --- |
| `티커` | 종목 코드 (대문자) |
| `회사` | yfinance에서 조회한 종목명 |
| `현재가격` | `fetch_latest_prices`가 가져온 최신 종가 |
| `SQZ_On(1H)` | 1시간봉 기준 볼린저 밴드 압축 여부(TRUE) |
| `SQZ_Off(1H)` | 1시간봉에서 직전 스퀴즈 해제 여부(TRUE) |
| `SQZ_M(1H)` | 1시간봉 스퀴즈 모멘텀 (양수 상방, 음수 하방) |
| `SQZ_On(1D)` | 일봉 기준 스퀴즈 압축 여부 |
| `SQZ_Off(1D)` | 일봉에서 스퀴즈 해제 여부 |
| `SQZ_M(1D)` | 일봉 스퀴즈 모멘텀 값 |
| `SQZ_On(1W)` | 주봉 기준 스퀴즈 압축 여부 |
| `SQZ_Off(1W)` | 주봉 스퀴즈 해제 여부 |
| `SQZ_M(1W)` | 주봉 스퀴즈 모멘텀 값 |
| `SQZ_On(1M)` | 월봉 기준 스퀴즈 압축 여부 |
| `SQZ_Off(1M)` | 월봉 스퀴즈 해제 여부 |
| `SQZ_M(1M)` | 월봉 스퀴즈 모멘텀 값 |
| `RSI(1H)` | 1시간봉 RSI |
| `RSI(1D)` | 일봉 RSI |
| `RSI(1W)` | 주봉 RSI |
| `RSI(1M)` | 월봉 RSI |

원하는 컬럼만 노출하고 싶으면 `config.EXPORT_COLUMNS`에서 문자열을 주석 처리하거나 제거하면 됩니다.

---
## 주요 기술·재무 신호 요약
- **모멘텀**
  - RSI ≤35 : 과매도, ≥78 : 과열
  - MACD 히스토그램 ≤ -0.3 : 하락 심화, ≥ +0.5 : 강세
  - Stochastic %K ≤20 : 침체, ≥85 : 과열
  - SQZ_M > 0 : 압축 해제 후 상방 모멘텀, < 0 : 하방 압력
  - 주봉/월봉 RSI와 스퀴즈 모멘텀으로 중기·장기 추세 확인
- **추세/저점 파악**
  - EMA20/50/200 역배열, EMA200 이탈 ≤ -25% → 과도한 하락 여부 평가
  - 20·63일 수익률 급락(≤ -15%, ≤ -20%) → 저점 탐색 가중치 상승
  - 10일 저점 괴리 ≤ 3% → 직전 저점 근접
- **거래/자금 흐름**
  - 거래량 Z ≥ 0 또는 거래대금 안정비 ≥ 0.6 → 저점 지지 가점
  - CMF ≥ 0, OBV Z ≥ 0 → 자금 유입
  - 장중 반등률 ≥ 4% → intraday 매수세 확인
- **상대 강도**
  - 최근 20일 수익률이 시장/섹터 평균보다 높으면 `방어`, 미달 시 `약세`
  - 시장·섹터 상대 강도가 `BUY_REL_STRENGTH_MIN`, `SECTOR_REL_STRENGTH_MIN` 미만이면 매수 후보에서 제외
- **재무 체력**
  - ROE ≥ 8%, 매출/이익 성장률 ≥ 설정값, 부채/자본 ≤ 200, 유동비율 ≥ 1.0 등 충족 시 `저점건강 = 건강`
  - 자유/영업 현금흐름이 음수면 감점
- **이벤트 리스크**
  - 실적 발표 D-day ±5일 안이면 저점 점수 감점, 표에서 `이벤트주의 = 실적 임박`

상세 조건은 `config.py` 임계치(예: `REL_STRENGTH_*`, `GAP_DOWN_EXTREME`, `HAMMER_*`, 재무 임계값 등)로 조정할 수 있습니다.

---
## 설정 가이드 (`config.py`)
- **티커/섹터 관리**
  - `TICKERS`: 분석 대상 목록 (S&P500 기본 제공)
  - `SECTOR_MAP`: 섹터 중립화용 매핑, 없으면 `Unknown`
- **유동성 필터링**
  - `LIQUIDITY_QUANTILE`: 제거할 하위 분위수 (기본 0.25 → 하위 25% 제거)
  - `LIQUIDITY_DOLLAR_MIN`: 최근 20일 평균 거래대금이 해당 금액(기본 5M USD) 미만이면 제외
  - `LIQUIDITY_WHITELIST`: 필수 포함 종목 리스트
- **저점 탐지 임계치**
  - `FUND_HEALTH_*`, `REL_STRENGTH_*`, `VOLUME_STABILITY_RATIO_MIN`, `EMA200_DISTANCE_MIN`, `GAP_DOWN_EXTREME`, `INTRADAY_RECOVERY_MIN` 등
- **Google Sheets 설정**
  - `GOOGLE_SHEETS_ENABLED`, `GOOGLE_SHEETS_CREDENTIALS_PATH`, `GOOGLE_SHEETS_SPREADSHEET_ID`, `GOOGLE_SHEETS_PORTFOLIO_WORKSHEET`
  - `GOOGLE_SHEETS_PORTFOLIO_TICKER_COLUMN`: 보유주식 워크시트에서 티커가 적힌 컬럼 이름(기본 `"티커"`)
- **출력 형식**
  - `EXPORT_COLUMNS`: 시트에 보낼 컬럼 순서
  - `PERCENT_COLUMNS`, `TECH_COLUMN_LABELS`: 값 포맷팅과 헤더 이름 매핑

---
## 트렌드 점수 가중치
- `ret5` (0.45): 5일 수익률을 그대로 반영
- `vol` (0.20): 거래량 Z-score를 `tanh`로 스무딩해 급증 여부 판단
- `break` (0.30): 52주 범위 대비 위치
- `vola` (-0.15): ATR% 선형 감점
- `rsi` (0.10): RSI 기반 스무딩 점수
- `accel` (0.12): 5일 대비 20일 수익률 가속도
- `vola_penalty` (-0.18): ATR%가 `VOLATILITY_PENALTY_START~END` 범위를 넘어설 때 추가 감점

트렌드 점수는 이후 시장/섹터 z-score와 혼합돼 `트렌드점수_최종`(0.6/0.25/0.15 비율)으로 정규화됩니다.

---
## 백테스트/신호 모듈 변경 사항
이 버전에서는 과거에 제공하던 신호 평가(`signals.py`), 백테스트(`backtest.py`), 극점 모델(`analytics/extremes.py`)을 제거했습니다. 따라서 `main.py`는 보유주식 워크시트(`"주식찾기"`) 업데이트만 수행하며, 별도의 백테스트 스크립트(`scripts/run_backtest_only.py`)도 더 이상 제공되지 않습니다.

---
## 실행 결과 예시 (콘솔)
```
Google Sheets 업데이트 완료
[요약] 파이프라인 완료 – 총 152.6초 (2.54분) 소요
```

상세 데이터는 Google Sheets에서 확인하세요. (콘솔 표는 출력하지 않습니다.)

---
## 문제 해결 팁
- **yfinance 다운로드 실패**: 간헐적으로 발생합니다. 잠시 후 재시도하거나 티커 수를 줄이세요.
- **실행 시간이 길다**: `TICKERS`를 섹터별로 나누거나, `fetch_ohlcv`의 `period`를 `1y` 등으로 줄일 수 있습니다.
- **재실행 시 결과가 조금씩 다르다**: 파이프라인은 실행할 때마다 yfinance에서 최신 데이터를 다시 받기 때문에 시점에 따라 수치가 달라질 수 있습니다.
- **서비스 계정 권한 오류**: Google Sheets 공유 설정에서 서비스 계정을 편집 권한으로 추가해야 합니다.
- **캐시/가상환경이 Git에 잡힘**: `.gitignore`에 `__pycache__/`, `.venv/` 등을 이미 포함해 두었으니, 과거 이력에 있다면 `git rm --cached`로 한 번 정리하세요.

---
문의나 개선 아이디어가 있다면 과거 커밋에 포함된 `signals.py`의 저점 스코어링 로직을 참고해 원하는 지표를 추가해 보세요.
