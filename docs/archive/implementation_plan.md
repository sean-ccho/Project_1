# 전 종목 스캔 및 전략 구현 계획서

NYSE와 NASDAQ의 모든 종목(약 6,000개+)을 스캔하여 새로운 구글 시트 워크시트(**Signals2**)에 결과를 내보내는 것을 목표로 합니다.

## 사용자 검토 필요 사항

> [!IMPORTANT]
> **기존 로직 유지 (Backward Compatibility)**: 이번 작업은 기존 `Signals` 시트를 업데이트하는 `main.py` 로직을 **절대 건드리지 않습니다.** 
> - 모든 새로운 분석 로직은 별도의 새로운 컬럼으로만 추가됩니다.
> - 기존 `Signals` 시트에 나가는 '매수적합도', '판단' 등의 기존 데이터는 100% 동일하게 유지됩니다.
> - 새로운 '바닥 탈출' 및 '강세 돌파' 로직은 오직 **Signals2** 전용 실행 파일(`run_full_scan.py`)에서만 필터링 기준으로 사용됩니다.
>
> **성능 최적화 (3단계 필터링)**: 수천 개의 종목을 효율적으로 처리하기 위해 다음과 같은 단계별 접근 방식을 사용합니다:
> 1. **1단계 (고속 추출)**: 6,000개 전 종목의 기본 데이터(현재가, 거래량, 52주 고점)를 대량으로 가져옵니다. (약 2~3분)
> 2. **2단계 (유효성 검사)**: 바닥 탈출/강세 돌파 전략에 해당하지 않는 종목을 즉시 제외합니다.
> 3. **3단계 (정밀 분석)**: 선별된 후보 종목들에 대해서만 정밀 기술적 분석을 수행합니다. (총 15~20분 내외 예상)

## 주요 변경 사항

### 분석 로직
#### [MODIFY] [features.py](file:///Users/seancho/Documents/Code/Signals/Project_1/features.py)
`compute_features_for_ticker` 함수에 다음 지표 추가:
- `drop_from_52w_high`: `(종가 - 52주 고점) / 52주 고점`.
- `ma200_breakout_recent`: 최근 5일 이내에 주가가 200일선(EMA)을 하향에서 상향으로 돌파했는지 여부.
- `volume_ma50`: 50일 평균 거래량.

#### [MODIFY] [signals.py](file:///Users/seancho/Documents/Code/Signals/Project_1/signals.py)
`attach_signals_and_sort` 함수에 두 가지 전략 그룹 식별 로직 추가:

1.  **바닥 탈출 (Turnaround)**: "지옥에서 돌아온 턴어라운드 종목"
    - **핵심**: 장기 소외주가 대량 거래를 동반하며 하락 추세를 끝내고 상방으로 방향을 트는 지점 포착.
    - **조건**: 고점 대비 -50% 이상 하락 AND 최근 200일선 돌파 AND 거래량 > 50일 평균의 2배.

2.  **강세 돌파 (Momentum)**: "달리는 말에 올라타기"
    - **핵심**: 이미 시장의 주도주로 정배열을 유지하며 추가 상승 에너지가 응축된 종목 포착.
    - **조건**: 52주 고점 대비 5% 이내 AND 최근 20일 수익률 15% 이상 AND 정배열(20>50>200).

### 설정
#### [MODIFY] [config.py](file:///Users/seancho/Documents/Code/Signals/Project_1/config.py)
- `GOOGLE_SHEETS_SIGNALS2_WORKSHEET = "Signals2"` 추가.
- 전략별 임계값 상수(낙폭 기준, 모멘텀 기준 등) 추가.

### 실행
#### [NEW] [run_full_scan.py](file:///Users/seancho/Documents/Code/Signals/Project_1/run_full_scan.py)
전 종목 분석을 위한 전용 스크립트 생성.
1. `ticker_fetcher.py`를 통해 전 종목 리스트 획득.
2. 3단계 필터링(깔때기 전략) 적용.
3. 최종 선정된 종목들을 `Signals2` 워크시트에 내보내기.

## 검증 계획
### 자동화 테스트
- `run_full_scan.py`를 50개 종목으로 제한하여 실행 후 `Signals2` 시트 업데이트 확인.
- `ticker_fetcher.py`가 나스닥 FTP 서버에서 데이터를 정확히 파싱하는지 확인.
