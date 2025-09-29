# Trend Ranking Screener

Python 기반으로 S&P 500 전 종목을 내려받아 기술적 지표를 계산하고, 종합 스코어 및 매수/매도 권장 메시지를 생성하는 스크립트입니다.

## 1. 워크플로우

1. **데이터 수집** – `yfinance`로 1년 일봉 데이터를 다운로드합니다. 분할·배당을 반영한 수정주가와 배당금(`actions=True`)을 함께 가져옵니다.
2. **피처 계산** – `features.py`가 다음 지표를 생성합니다.
   - 수익률: 1일(`ret_1d`), 5일(`ret_5d`), 10일 ROC
   - 추세 & 모멘텀: 트렌드 점수(가중합), RSI, MACD(+시그널, 히스토그램), Stochastic %K/%D, ADX(+DI/-DI), 52주 포지션, 이동평균 갭(EMA20-50/50-200/20-200)
   - 변동성: ATR%, Bollinger Band P-band/width, Keltner Channel P-band/width
   - 거래량 & 자금 흐름: 20일 거래량 Z-score, OBV Z-score, Chaikin Money Flow(20), Acc/Dist 기울기(5)
   - 배당: 최근 1년 배당 합계, 배당수익률
3. **중립화 & 필터** – 시장·섹터별 z-score를 추가해 트렌드 점수를 조정하고(`processing.apply_neutralization`), 거래대금 하위 40%를 제거합니다. 화이트리스트(`LIQUIDITY_WHITELIST`)에 등록된 종목은 항상 유지됩니다.
4. **신호 & 추천** – `signals.py`가 지표들을 긍정/부정/과열 요소로 분류해 `판단`(매수 후보/관심 관찰/관망 과열/관망 약세)과 `추천`(적극 매수, 조건 충족 시 매수, 차익 실현 고려 등)을 생성합니다.
5. **출력** – `main.py`가 결과를 콘솔과 `output/신호_최신.xlsx`에 저장합니다. 숫자는 소수점 1자리, 거래대금은 백만 달러 단위로 반올림되며, `메모` 컬럼에 주요 긍정/경계 지표 요약이 들어갑니다.
6. **Google Sheets(선택)** – `config.py`에서 연결 정보를 채우면 같은 DataFrame이 구글 시트에도 업로드됩니다.

## 2. 주요 기술 지표와 해석

| 분류 | 지표 | 매수 관점 | 매도/관망 관점 |
| --- | --- | --- | --- |
| 모멘텀 | RSI | 55~75: 건강한 상승 추세, 과매수가 아니면 매수 가점 | ≥80: 과열, 40 이하: 약세 관망 |
| | MACD 히스토그램 | ≥ +0.5: 상승 가속 | ≤ –0.5: 하락 위험 |
| | Stochastic %K/%D | ≥40: 단기 반등, ≥85: 과열 경계 | ≤20: 침체 구간 |
| 추세 | ADX(+DI/-DI) | ≥20(또는 30 이상): 추세 확립, +DI 우위 | ≤15: 추세 약, –DI 우위 |
| | EMA 20/50/200 갭 | 양수이면 정배열, 추세 동행 | 음수이면 역배열, 하락 가능 |
| | 52주 포지션 | ≥0.7: 돌파/강세 | ≤0.3: 저점 부근 |
| 변동성 | ATR% | ≤5%: 안정적 추세 | ≥8%: 변동성 확대 |
| | Bollinger P-band | ≥0.85: 상단 돌파(강세) | ≥0.98: 과열, ≤0.2: 하단 테스트 |
| 거래량/자금 | 거래량 Z-score | ≥0: 거래량 지지 | ≤-1.5: 거래량 감소 |
| | OBV Z-score & CMF | ≥0: 자금 유입, 지지 | <0: 자금 유출 |
| 배당 | 배당수익률 | ≥2%: 안정적 현금흐름 | ≤0.2%: 배당 매력 낮음 |

`signals.collect_signal_evidence`는 위 기준을 긍정/부정/과열 증거로 분류해 종합 판단을 내립니다.

### 트렌드 점수 계산 방법

1. **기본 트렌드 점수(`트렌드점수`)** – `features.py`에서 아래 가중합으로 계산합니다. (가중치는 `config.WEIGHTS`)

   
   `0.30 × 5일 수익률` (단기 모멘텀)  
   `+ 0.30 × tanh(거래량Z(20)/3)` (거래량 급증)  
   `+ 0.25 × clip(52주포지션, 0, 1)` (52주 돌파 정도)  
   `+ 0.05 × clip(ATR%, 0, 0.1)/0.1` (낮은 변동성 보상)  
   `+ 0.10 × tanh((RSI-55)/10)` (RSI 기반 스무딩)

2. **최종 트렌드 점수(`트렌드점수_최종`)** – `processing.apply_neutralization`에서 시장/섹터 편향을 제거합니다.

   - 시장별 z-score (`트렌드점수_mktz`) + 섹터별 z-score (`_secz`)를 추가
   - 섹터 정보가 있으면 `0.5 × 원본 + 0.3 × 시장z + 0.2 × 섹터z`
   - 섹터가 Unknown이면 `0.7 × 원본 + 0.3 × 시장z`

   → 결과적으로 전체 시장 흐름과 섹터 편차를 보정한 비교 가능한 점수가 `트렌드점수_최종`입니다.

## 3. 판단 & 추천 로직 요약

- **매수 후보 → 적극/분할 매수**: 긍정 요소가 6개 이상이며 부정 요소가 거의 없고, MACD와 ADX가 강세일 때 `적극 매수`, 그 외엔 `분할 매수`.
- **관심 관찰**: 긍정 요소가 부정보다 많지만 추세 강도가 부족하면 `조건 충족 시 매수` 또는 `추가 관찰`.
- **관망 과열 → 차익 실현 고려**: RSI/스토캐스틱/볼린저가 과열 신호를 2개 이상 띄우면 차익 실현 권장.
- **관망 약세**: 부정 요인이 우위일 때 `관망/보유` 혹은 `추가 관찰`.

## 4. 설정 파일(`config.py`) 주요 항목

- `SP500_TICKERS`: 위키에서 스크랩한 S&P 500 티커 목록. 필요 시 직접 추가 가능(`LIQUIDITY_WHITELIST` 포함).
- `SECTOR_MAP`: 섹터 중립화를 위한 매핑. Unknown은 시장 중립화만 적용됩니다.
- `LIQUIDITY_QUANTILE`: 거래대금 컷 비율(0.40이면 하위 40% 제거). 손쉽게 조절 가능.
- `LIQUIDITY_WHITELIST`: 필터를 통과시키고 싶은 종목 목록. `GRRR`, `COIN` 등이 기본 등록되어 있습니다.
- `WEIGHTS`: 트렌드 점수에서 각 요소에 부여할 가중치.
- `PERCENT_COLUMNS`, `TECH_COLUMN_LABELS`: 출력 포맷과 헤더 명칭을 정의합니다.

## 5. 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # yfinance, pandas, numpy, ta, openpyxl 등
python main.py
```

S&P 500 전체를 처리하면 2~3분 이상 걸릴 수 있으며, Yahoo Finance 요청이 간헐적으로 실패할 수 있습니다. `data.fetch.MAX_DOWNLOAD_RETRIES`와 `RETRY_DELAY_SECONDS`로 재시도 횟수와 대기 시간을 조절하세요.

## 6. 출력 파일

- `output/신호_최신.xlsx`: 최신 랭킹과 주석이 포함된 시그널 표 (엑셀 시트 `Signals`).
  - `메모`는 긍정·경계 지표를 3개까지 요약해 추천 이유를 빠르게 파악할 수 있게 합니다.
  - `최근20일평균거래대금(M)`은 백만 달러 단위로 환산됩니다.
- `output/신호_YYYYMMDD_HHMMSS.xlsx`: 백업 파일(옵션). `EXPORT_WITH_BACKUP = False`로 되어 있으니 필요 시 `True`로 변경.
 - *(옵션)* `config.GOOGLE_SHEETS_ENABLED = True`로 설정하면 동일 데이터를 Google Sheets에도 업데이트합니다.

콘솔에서도 같은 표가 출력되며, 거래대금은 `XX,XXX.XM` 포맷을 유지합니다.

## 7. 트러블슈팅

- **DNS 오류 / Failed download**: yfinance 서버가 응답하지 않을 때 발생합니다. 스크립트가 최대 3회 재시도하며, 반복될 경우 잠시 후 다시 실행하거나 티커 리스트를 나눠서 처리하세요.
- **속도 문제**: S&P 500 전체 요청은 데이터가 많으므로, 필요 시 `TICKERS`를 섹터별로 나눠 실행하거나 `period`를 줄여 가속할 수 있습니다.
- **Unknown 섹터**: `SECTOR_MAP`을 보강하면 중립화精度가 향상됩니다.
- **Google Sheets**: `pip install gspread google-auth` 후 서비스 계정 JSON 경로(`GOOGLE_SHEETS_CREDENTIALS_PATH`)와 `GOOGLE_SHEETS_SPREADSHEET_ID`를 설정하세요. 서비스 계정을 해당 시트에 편집 권한으로 공유해야 합니다.

---
지표 해석 기준과 추천 로직은 README에 모두 요약되어 있으므로, 생성된 추천을 확인할 때 각 지표가 어떤 역할을 했는지 빠르게 추적할 수 있습니다. 필요하다면 `signals.collect_signal_evidence`에 로그를 추가하여 어떤 증거가 쌓였는지 더 자세히 확인할 수 있습니다.
