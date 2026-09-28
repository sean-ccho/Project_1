# Analyze Trading Strategy — 진단 프레임워크

> 백테스트 승률 53% 문제를 **데이터 기반**으로 진단하고 개선하기 위한 프레임워크.
> 단순 파라미터 조정이 아닌, 무엇이 작동하고 무엇이 실패하는지 측정한다.

---

*최근 변경: 2026-04-16 — 백테스트 디스크 캐싱 도입*
- **재실행 속도 40-50시간 → ~5분**: 동일 파라미터 재실행 시 캐시에서 로드
- **캐시 안전한 변경**: `BACKTEST_PROFIT_TARGET`, `BACKTEST_TRAILING_STOP`, `BACKTEST_STOP_LOSS`, `BACKTEST_ENTRY_SCORE_MIN` 등 `BACKTEST_*` 계열은 자유롭게 변경 후 재실행 가능
- **캐시 무효화 필요**: `RSI_BUY_MAX`, `MACD_BUY_HIST_THRESHOLD`, `ALPHA_DEFAULT_WEIGHTS` 등 feature 계산 파라미터 변경 시 `--no-cache` 플래그 추가
- 커맨드: `python scripts/run_sp500_backtest.py --period 5y --capital 5000 --fundamentals [--no-cache]`

---
*최근 변경: 2026-04-15 — Phase 2 진단 결과 적용 완료 (Hold-Winners + 상승 예측 모델)*
- **Phase 2 진단 → 실제 개선 반영**: "익절 후 추가 상승 분석"에서 목표가 도달 시 모멘텀이 남은 경우가 있음을 확인 → Hold-Winners 재평가 로직으로 구현
- **Hold-Winners 재평가** (`engine.py`, `backtest.py`): 목표가/시간익절 트리거 시 당일 factor 재평가 후 강하면 defer + tight trailing(-3.5%). 최대 2회.
- **상승 예측 모델 B** (`upside_model.py` 신규): 과거 백테스트 거래에서 전략/CCS/★ 버킷별 P25/P50/P75/P90 경험적 분포 빌드 → Phase 3 ML 대안. 현재 128 거래 / 18 버킷.
- 상승 예측 캐시 빌드: `python -m paper_trading.upside_model` (새 백테스트 후 재실행 권장)

---

## 진행 현황

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | 백테스트 트레이드 진입 피처 기록 | ✅ 완료 |
| Phase 2 | 진단 분석 스크립트 (`analyze_trades.py`) | ✅ 완료 |
| Phase 2B | Phase 2 진단 결과 → Hold-Winners 재평가 적용 | ✅ 완료 (2026-04-15) |
| Phase 2C | 상승 예측 모델 B (경험적 분포) | ✅ 완료 (2026-04-15) |
| Phase 3 | ML 기반 진입 품질 예측 | ⏳ 대기 (데이터 300건+ 이후) |

---

## 검증 방법 (실행 순서)

### Step 1 — 백테스트 실행

```bash
python scripts/run_sp500_backtest.py --period 5y --capital 5000

python scripts/run_sp500_backtest.py --period 5y --capital 5000 --fundamentals

```

끝날 때 아래 두 줄이 보이면 성공:
```
[백테스트] 향상된 거래 로그 저장: output/backtest_trades_enhanced.json, output/backtest_trades_enhanced.csv
[백테스트] 버전 저장: output/runs/YYYY-MM-DD_HH-MM/
```

> 실행할 때마다 `output/runs/` 아래에 타임스탬프 폴더가 자동 생성된다.
> 파라미터를 바꾸고 여러 번 돌리면 모든 결과가 누적 보존된다.

---

### Step 2 — 피처 캡처 확인

```bash
python -c "import json; t=json.load(open('output/backtest_trades_enhanced.json'))[0]; print('entry_features keys:', list(t.get('entry_features',{}).keys())); print('post_exit_5d:', t.get('post_exit_return_5d'))"
```

성공 시 출력:
```
entry_features keys: ['RSI', 'adx', 'macd_hist', 'bollinger_pband', ..., 'ccs_strategy_fit', 'regime', ...]
post_exit_5d: 0.0312
```
> `entry_features keys: []` 이거나 `post_exit_5d: None` 이면 피처 추출 실패

---

### Step 3 — 진단 분석 실행

```bash
python scripts/analyze_trades.py
```

4개 섹션 출력:
1. **FEATURE-OUTCOME CORRELATIONS** — 어떤 지표가 수익률과 관련 있는지 + 골든 룰 자동 탐색
2. **EXIT REASON ANALYSIS** — 손절 회복률 (높으면 손절이 너무 타이트), 익절 후 추가 상승 여부
3. **STRATEGY BREAKDOWN** — 모멘텀/바닥반등 각각 어떤 셋업에서 승률 높은지
4. **MARKET & SECTOR ANALYSIS** — 레짐/섹터/월별 승률 패턴

---

### Step 3 이후 — 결과 활용

분석 결과에서 발견한 패턴을 `src/screener/config.py` 수정에 활용:
- 골든 룰 조건 (예: ADX < 15 → 승률 낮음) → 필터 강화
- 손절 회복률 높음 → 손절 퍼센트 넓히기
- bear 레짐 승률 낮음 → 약세장 진입 조건 강화

---

### Step 4 — 파라미터 수정 후 재실행 & 비교

백테스트를 실행할 때마다 `output/runs/` 아래에 타임스탬프 폴더가 **자동 생성**된다.
각 폴더에는 거래 로그 + `meta.json`(성과 수치 & 당시 파라미터 스냅샷)이 저장된다.

```
output/runs/
  2026-04-07_baseline/     ← run #1 (변경 전)
    backtest_trades_enhanced.json
    meta.json              ← 승률·평균수익 + ALPHA_DEFAULT_WEIGHTS 등 파라미터 기록
  2026-04-07_14-30/        ← run #2 (1차 수정 후)
    ...
  2026-04-07_16-00/        ← run #3 (2차 수정 후)
    ...
```

**전체 run 요약 한눈에 보기** (몇 번을 돌려도 모두 표시):

```bash
python scripts/compare_runs.py --all
```

출력 예시:
```
  #   run                  거래수    승률    평균수익   평균승    평균패  모멘텀WR  바닥WR
  1   baseline               217   50.7%    +0.95%  +5.93%  -4.17%    48.5%   57.7%
  2   2026-04-07_14-30       203   54.2%    +1.38%  +6.10%  -3.90%    52.1%   60.3%
  3   2026-04-07_16-00       198   56.1%    +1.72%  +6.45%  -3.80%    54.3%   61.5%
```

**두 run 상세 비교** (파라미터 diff + 전략별/섹터별/월별):

```bash
# 가장 최근 2개 자동 선택
python scripts/compare_runs.py

# 특정 두 run 지정
python scripts/compare_runs.py output/runs/2026-04-07_baseline output/runs/2026-04-07_14-30
```

상세 비교 출력 내용:
1. **파라미터 diff** — 변경된 config 값 나란히 표시 (변경 없는 항목은 요약)
2. **전체 성과** — 승률/평균수익/평균승패 before → after + delta
3. **전략별** — 모멘텀/바닥반등 각각 건수·승률·평균수익 비교
4. **섹터별** — Unknown 비율 변화 확인
5. **exit reason** — 손절/익절/트레일링 비율 변화
6. **월별 패턴** — 계절성 변화

> **반복 개선 사이클**: `config.py` 수정 → Step 1 백테스트 → `--all`로 전체 흐름 확인 → 상세 비교로 원인 분석 → 다시 수정

---

### (상세 버전)

### Phase 1 완료 후 테스트 가능
 
```bash
# Step 1: 백테스트 실행 (entry_features + post_exit 데이터 포함)
python scripts/run_sp500_backtest.py --period 3y --capital 5000

# Step 2: 출력 파일 구조 확인
python -c "
import json
with open('output/backtest_trades_enhanced.json') as f:
    trades = json.load(f)
t = trades[0]
print('entry_features keys:', list(t.get('entry_features', {}).keys()))
print('post_exit_5d:', t.get('post_exit_return_5d'))
"
```

### Phase 1 + 2 완료 후 테스트 가능

```bash
# Step 3: 진단 분석 실행
python scripts/analyze_trades.py

# Step 4: 골든 룰 수동 검증
# 출력된 "Rule 1: RSI < 35 → N% win" 등 조건을 trades_enhanced.csv에서 필터링해 직접 확인
```

---

## Phase 1 — 백테스트 트레이드 진입 피처 기록 ✅

**파일**: `src/paper_trading/backtest.py`

### 변경 내용

- **`BtPosition`에 `entry_features` 필드 추가** — 진입 시점 피처 저장
- **`_extract_entry_features()` 함수 신규** — `ranked_df`에서 아래 피처를 추출:
  - 기술지표: RSI, adx, macd_hist, bollinger_pband, ATR%, 거래량Z(20), obv_z20, cmf_20, 52주포지션, 5/20일수익률, ema_gap, 변동성압축, buy_support_count
  - 알파팩터: 팩터_모멘텀/추세/거래량/변동성/평균회귀, 알파점수
  - 전략점수: 바닥반등_적합도, 모멘텀_적합도, 반등스코어, 저점확률
  - CCS 서브스코어: strategy_fit, timing, alpha, risk, confluence, penalty
  - 시장상태: regime, spy_close_to_ema200
  - 섹터: sector, in_strong_sector, 섹터상대강도
- **`_close_position`에서 피처 전파** — trade dict에 `entry_features` 포함
- **청산 후 가격 변동** — `post_exit_return_5d / 10d / 20d` 계산 (손절 회복 / 익절 잔여수익 진단용)
- **향상된 거래 로그 저장** — `output/backtest_trades_enhanced.json` + `.csv`

---

## Phase 2 — 진단 분석 스크립트 ✅

**파일**: `scripts/analyze_trades.py` (신규)

`output/backtest_trades_enhanced.json`을 읽어 4가지 분석 수행:

### 분석 1: 피처-수익률 상관관계
- 모든 진입 피처와 `return_pct`의 Spearman 상관계수 → 상위 20개 출력
- 상위 5개 피처를 5분위로 나눠 분위별 승률/평균수익 테이블
- **골든 룰 탐색**: 승률 > 65% + N ≥ 10 조건 자동 발견

### 분석 2: 청산 사유 분석
- 청산 유형별 승률/평균수익/건수 테이블
- **손절 회복 분석**: 손절 후 5/10/20일 내 회복률 → 손절이 너무 타이트한지 진단
- **익절 잔여수익 분석**: 익절 후 추가 상승분 → 너무 일찍 나가는지 진단
- **트레일링 분석**: 청산 후 추가 하락/상승 비교

### 분석 3: 전략별 세부 분석
- **모멘텀**: ADX 구간별 / buy_support_count별 / 팩터_모멘텀 사분위별
- **바닥반등**: RSI 구간별 / 반등스코어별 / 팩터_평균회귀 사분위별

### 분석 4: 시장 상황 & 섹터
- 시장 레짐별 (bull/bear/neutral) 승률
- 섹터별 승률 (N ≥ 5)
- 월별 승률 패턴

---

## Phase 2B — Hold-Winners 재평가 ✅

> Phase 2 "익절 후 추가 상승 분석"에서 목표가 도달 후에도 모멘텀이 남는 케이스 확인 → 즉시 매도가 아닌 재평가 로직 도입

**파일**: `src/paper_trading/engine.py`, `src/paper_trading/backtest.py`

### 로직 요약
- 목표가(`profit_target`) / 시간익절(`long_hold`) 트리거 시 `_should_defer_sell()` 호출
- **Defer 조건** (모두 충족 필요):
  - RSI 60–75 (강세 구간, 하드 컷오프 75 미만)
  - ADX ≥ 20 (추세 유지)
  - 거래량돌파배수 ≥ 1.2× OR 거래량Z ≥ 1.0
  - 5일수익률 > 0
  - 볼린저 pband < 0.95 (고점 아님)
- Defer 시: `trailing_stop_override = 3.5%` (기존 5–6%보다 타이트) + 고점 리셋
- 최대 2회 — 3번째 트리거 시 무조건 매도
- **손절/트레일링은 절대 defer 안 함**

### 캐시 빌드 후 분석 명령
```bash
# 새 백테스트 실행 후 defer 효과 확인
python scripts/compare_runs.py   # 전후 비교
# exit reason 섹션에서 "목표가→tight trail" 비율 확인
```

---

## Phase 2C — 상승 예측 모델 (경험적 분포) ✅

> Phase 3 ML 대안 — 데이터 300건 미만에서도 활용 가능한 경험적 분포

**파일**: `src/paper_trading/upside_model.py` (신규)

### 버킷 구조
- 키: `(전략 | CCS버킷 | ★버킷)`
  - CCS 버킷: `<0.40 / 0.40-0.45 / 0.45-0.50 / 0.50-0.55 / 0.55+`
  - ★ 버킷: `5★(9+) / 5★(7-9) / 5★(<7) / 4★이하`
- 버킷별 P25/P50/P75/P90 + 평균 도달 일수
- 샘플 < 10 시 strategy 레벨 → global 레벨 폴백

### 캐시 관리
```bash
# 캐시 빌드 (새 백테스트 완료 후 실행)
python -m paper_trading.upside_model
# → data/paper_trading/upside_distribution.json 업데이트
```

현재 캐시: **128 거래 / 18 버킷** (2026-04-15 기준)

### 이메일/PDF 반영
매수 시 자동으로 예상 상승 분포 박스 표시:
```
보수적 (P25): +6%  → $5.11
중간값 (P50): +14% → $5.50
낙관  (P75): +24% → $5.98
홈런  (P90): +41% → $6.80
```

---

## Phase 3 — ML 기반 진입 품질 예측 ⏳

> **조건**: 거래 데이터 300건+ 축적 후 진행

- Phase 2에서 발견한 주요 피처 10-15개로 LightGBM/sklearn 분류기 학습
- Walk-forward validation (미래 데이터 사용 금지)
- Phase 2C(경험적 분포)가 이미 부분 대체 중 → 300건 축적 후 ML로 고도화

---

## 수정/생성 파일 목록

| 파일 | 변경 내용 | 상태 |
|------|-----------|------|
| `src/paper_trading/backtest.py` | BtPosition 피처 필드, 피처 추출 함수, post_exit 계산, JSON/CSV 저장 | ✅ |
| `scripts/analyze_trades.py` | 신규 — 진단 분석 스크립트 | ✅ |
| `src/paper_trading/engine.py` | `_should_defer_sell`, `_activate_tight_trail`, `_build_high_rsi_rationale` 추가 | ✅ |
| `src/paper_trading/upside_model.py` | 신규 — 경험적 상승 분포 빌드/조회 | ✅ |
| `data/paper_trading/upside_distribution.json` | 신규 캐시 (128 거래 / 18 버킷) | ✅ |
