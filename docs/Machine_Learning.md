# Machine Learning 통합 계획

> 규칙 기반 퀀트 트레이딩 시스템에 XGBoost 기반 ML 레이어를 점진적으로 추가하기 위한 실행 문서.
> 작업 중 이 파일을 참고하면서 Phase별로 진행한다.

> **2026-05-24 시점 상태**: Phase A+B 코드가 통합돼 있으나 `ML_ENABLED=False`(prod). 모델 마지막 학습 2026-04-18 이후 재학습·재평가 없음. Phase C~F는 미진행. 켜려면 `src/screener/config.py`의 `ML_ENABLED=True` 후 `data/ml_models/` 확인.

---

*최근 변경: 2026-04-17 — Phase A + Phase B 구현 완료*
- `src/ml/` 패키지 신규 생성 (feature_engineering, training, prediction, evaluation)
- `requirements.txt` xgboost/lightgbm/joblib 추가 (로컬은 sklearn HistGBM 사용)
- `src/screener/config.py` ML_* 상수 블록 추가 (ML_ENABLED=False)
- `data/ml_models/` 훈련 artifact 저장소 생성 (561건 학습, y_win AUC 0.478)
- `src/paper_trading/candidate_selector.py` `_score_alpha_factor()`에 ML 블렌딩 추가
- 관련 파일: `src/ml/`, `src/paper_trading/candidate_selector.py`

---

*최근 변경: 2026-04-16 — ML 통합 계획 문서 신설*
- `docs/Machine_Learning.md` 신규 생성 (6단계 ML 롤아웃 계획)
- 관련 파일: `src/ml/` (신규 예정), `src/paper_trading/candidate_selector.py`, `src/screener/signals.py`

---

## Context (배경)

현재 시스템은 완전한 **규칙 기반(rule-based)** 퀀트 트레이딩 파이프라인이다:

- 스크리너(S&P500/NASDAQ) → 전략 점수(바닥반등/모멘텀) → CCS(Composite Conviction Score) → 후보 선정 → 페이퍼 트레이딩
- 매일 5PM EST에 GitHub Actions로 자동 실행
- **진입 피처 데이터는 이미 수집 중**: `output/backtest_trades_enhanced.json`에 202건의 거래가 있고, 각 거래마다 35+ 피처(`entry_features`) + 라벨(`return_pct`, `post_exit_return_5d/10d/20d`)을 보유

### 왜 지금 ML을 도입하는가

1. 규칙 기반 CCS의 5개 서브스코어는 사람이 정한 **고정 가중치** — 조건 간 비선형 상호작용을 포착 못함
2. `_score_alpha_factor()`, `_score_risk_quality()` 등 5개 스코어링 함수가 이미 분리돼 있어 **최소 침습적(ML 스코어 블렌딩)** 삽입이 가능
3. 이미 `upside_model.py`에서 "오프라인 빌드 → JSON 캐시 → 런타임 조회" 패턴이 정립돼 있어 ML도 같은 패턴 재사용 가능

### 목표 성과

- **승률 53% → 60%+ 개선** (Phase 2 진단 문서의 목표)
- ML 신뢰도 점수 기반 **포지션 배분 차등화**
- 시장 국면 변화에 자동 적응 (드리프트 감지 → 주간 재학습)

---

## 전체 단계 (6단계 점진적 롤아웃)

| Phase | 목표 | 위험도 | 선결 조건 | 상태 |
|-------|------|--------|-----------|------|
| A. Bootstrap | 의존성 추가 + `src/ml/` 패키지 골격 + 훈련 스크립트 | 낮음 | 거래 250건+ | ✅ 완료 |
| B. Alpha 블렌딩 | `_score_alpha_factor()`에 ML 점수 블렌딩 (CCS 15%) | 낮음 | A 완료 | ✅ 완료 (ML_ENABLED=False) |
| C. Risk + Strategy 블렌딩 | `_score_risk_quality()` + 전략 점수에 ML 블렌딩 (가장 큰 영향) | 중간 | B 검증 완료 | ⏳ 대기 |
| D. 상승 예측 업그레이드 | `upside_model.py`의 empirical 버킷을 XGBoost 회귀로 교체 | 중간 | C 완료 | ⏳ 대기 |
| E. 종목 매도 타이밍 | `check_sell_conditions()` + `_should_defer_sell()` 보강 | 높음 | D 완료, 보유 데이터 더 확보 | ⏳ 대기 |
| F. 재학습 + 드리프트 감지 | 주간 재학습 워크플로우 + 성능 저하 경보 | 중간 | B~E 중 하나 이상 운영 중 | ⏳ 대기 |

---

## Phase A — Bootstrap (의존성 + 모듈 골격 + 훈련 스크립트)

### 체크리스트
- [x] A-1. `requirements.txt`에 `xgboost`, `lightgbm`, `joblib` 추가
- [x] A-2. `src/ml/` 패키지 생성 + `__init__.py`
- [x] A-3. `src/screener/config.py`에 ML_ENABLED=False 포함 ML_* 상수 추가
- [x] A-4. `src/ml/feature_engineering.py` — `load_training_trades()`, `make_labels()`
- [x] A-5. `src/ml/training.py` — walk-forward CV + 모델 저장
- [x] A-6. `src/ml/prediction.py` — 런타임 예측 API + 폴백 로직
- [x] A-7. `python -m ml.training` 실행 → 561건 학습, artifact 생성 완료
- [ ] A-8. CV 성능 리포트 검토 (AUC > 0.55) — 현재 y_win 0.478, 재학습 필요

### A-1. 의존성 추가

**수정:** `requirements.txt`

```
xgboost>=2.0
lightgbm>=4.0  # 대안 모델용
joblib>=1.3    # 모델 직렬화 (이미 sklearn 경유로 설치되지만 명시)
```

### A-2. 새 모듈 구조

```
src/ml/
├── __init__.py
├── config.py              # ML 전용 설정 (blend_weight, min_samples, paths 등)
├── feature_engineering.py # 거래 JSON → 학습용 DataFrame 변환
├── training.py            # build_entry_quality_model() 등 훈련 파이프라인
├── prediction.py          # predict_entry_quality(row) 런타임 API
├── evaluation.py          # walk-forward 교차검증, 성능 지표
└── drift_detection.py     # Phase F에서 확장

data/ml_models/
├── entry_quality_v1.pkl   # XGBoost 모델 artifact (joblib)
├── entry_quality_meta.json # 훈련 날짜, 피처 목록, 성능 지표, 버전
└── feature_importance.json # 상위 피처 중요도 (운영 참고용)
```

### A-3. `src/ml/config.py` (신규)

```python
# 모델 사용 on/off — 기본 False로 시작해 검증 후 켜기
ML_ENABLED = False

# 블렌딩 가중치: final = ML_BLEND_WEIGHT × ML + (1 - w) × rules
ML_BLEND_WEIGHT_ALPHA = 0.4      # Phase B
ML_BLEND_WEIGHT_RISK = 0.5       # Phase C
ML_BLEND_WEIGHT_STRATEGY = 0.5   # Phase C (바닥반등/모멘텀 적합도)

# 훈련 설정
ML_MIN_TRAINING_SAMPLES = 250    # 최소 거래 수 (현재 202 → 5y 백테스트로 확보)
ML_TIME_SERIES_N_SPLITS = 5      # walk-forward 폴드 수
ML_MODEL_DIR = "data/ml_models"
ML_MODEL_MAX_AGE_DAYS = 14       # 이보다 오래되면 stale 경고
ML_FALLBACK_ON_MISSING = True    # 모델 없으면 rule-based로 폴백

# 재학습 트리거
ML_DRIFT_ROLLING_WINDOW = 30     # 최근 N거래 기준 정확도 체크
ML_DRIFT_ACCURACY_THRESHOLD = 0.45 # 이하로 떨어지면 경보/재학습
```

### A-4. `src/ml/feature_engineering.py` (신규)

**입력:** `output/runs/*/backtest_trades_enhanced.json` (Phase 2C `upside_model._load_recent_trades()` 패턴 그대로 재사용)

```python
def load_training_trades(
    runs_dir: Path | None = None,
    max_runs: int = 5,
) -> pd.DataFrame:
    """백테스트 run들에서 거래 로드 → 중복 제거 → 엔트리 피처 평탄화.
    Columns: entry_date, ticker, strategy, [feature_cols...], return_pct, post_exit_return_10d
    """

FEATURE_COLS = [
    # 기술지표 (13)
    "RSI", "adx", "macd_hist", "bollinger_pband", "atr_pct",
    "vol_z20", "obv_z20", "cmf_20", "pos_52w",
    "ret_5d", "ret_20d", "ema_gap_20_50", "ema_gap_50_200",
    # 알파팩터 (6)
    "alpha_score", "factor_momentum", "factor_trend",
    "factor_volume", "factor_volatility", "factor_mean_reversion",
    # 전략 점수 (4)
    "bottom_reversal_fit", "momentum_fit", "reversal_score", "low_prob",
    # 섹터/시장 (3)
    "sector_relative_strength", "spy_close_to_ema200", "in_strong_sector",
    # CCS 서브 (6)
    "ccs_strategy_fit", "ccs_timing", "ccs_alpha",
    "ccs_risk", "ccs_confluence", "ccs_penalty",
]
CATEGORICAL_COLS = ["sector", "regime"]  # one-hot 인코딩

def make_labels(trades: pd.DataFrame) -> dict[str, pd.Series]:
    """4종 라벨 생성:
    - y_win: return_pct > 0 (분류)
    - y_return: return_pct (회귀)
    - y_big_win: return_pct > 0.07 (강한 상승 분류)
    - y_drawdown: post_exit_return_5d < -0.03 (손실 회피 분류)
    """
```

**중요 포인트:**
- 피처 선택은 `_extract_entry_features()` ([src/paper_trading/backtest.py:235](../src/paper_trading/backtest.py#L235))와 **1:1 매칭** → 런타임에 동일 피처 보장
- `sector`는 S&P500 11개 섹터 one-hot, `regime`은 3종(bull/neutral/bear) one-hot
- NaN 처리: `volatility_contraction`, `low_prob` 등 일부 피처는 계산 실패 시 NaN → XGBoost가 native로 처리 (imputation 불필요)

### A-5. `src/ml/training.py` (신규)

```python
def build_entry_quality_model(
    trades_df: pd.DataFrame,
    target: str = "y_win",
    n_splits: int = 5,
) -> tuple[XGBClassifier, dict]:
    """Walk-forward 교차검증 → 최종 모델 훈련 → (model, metrics) 반환.

    1. entry_date 기준 시간순 정렬
    2. TimeSeriesSplit(n_splits=5) — 랜덤 분할 금지
    3. 각 폴드에서: AUC, accuracy, precision@top20%, sharpe_proxy 측정
    4. 최종 모델은 전체 데이터로 재훈련
    """

def save_model(
    model: XGBClassifier,
    metrics: dict,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    """joblib으로 모델 + meta.json 저장 (upside_model 패턴 준수)."""

def train_all_models(runs_dir: Path, output_dir: Path) -> None:
    """CLI 엔트리포인트. 4개 타겟 모두 훈련 + 저장."""
```

**실행:**
```bash
python -m ml.training   # data/ml_models/ 에 artifact 생성
```

### A-6. `src/ml/prediction.py` (신규)

```python
# 모듈 레벨 캐시 (upside_model._load_cache 패턴 준수)
_MODEL_CACHE: dict[str, Any] = {}

def _load_model(name: str) -> tuple[Any, dict] | None:
    """joblib 모델 + meta.json 로드. 없으면 None → 호출자가 폴백."""

def predict_entry_quality(row: pd.Series | dict, target: str = "y_win") -> float:
    """[0, 1] 신뢰도 점수. 모델 없으면 0.5 (중립) 반환.

    1. row에서 FEATURE_COLS 추출 → sector/regime one-hot 적용
    2. 모델 로드 → predict_proba()[:, 1]
    3. 모델 없거나 실패 시 0.5 (블렌드 시 rule-based만 사용하도록)
    """

def predict_exit_signal(pos_features: dict) -> float:
    """Phase E에서 활성화. 현재 포지션의 매도 확률 [0, 1]."""
```

### A-7. 검증

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 학습 데이터 생성 확인
python -c "from ml.feature_engineering import load_training_trades; df = load_training_trades(); print(df.shape, df.columns.tolist())"

# 3. 모델 훈련
python -m ml.training
# → data/ml_models/entry_quality_v1.pkl, meta.json 생성 확인
# → 콘솔 출력에 CV AUC, accuracy 리포트 확인

# 4. 런타임 예측 확인
python -c "from ml.prediction import predict_entry_quality; import pandas as pd; print(predict_entry_quality(pd.Series({'RSI': 35, 'adx': 25})))"
```

**성공 기준:**
- `y_win` AUC > 0.55 (랜덤보다 유의하게 우수)
- top-20% precision > baseline 승률
- `ML_ENABLED=False` 이므로 프로덕션 영향 없음 (artifact 생성 확인만)

---

## Phase B — Alpha Factor 블렌딩 (첫 프로덕션 삽입)

### 체크리스트
- [x] B-1. `_score_alpha_factor()` 수정 — ML 블렌딩 분기 추가
- [ ] B-2. `ML_ENABLED=False` 상태로 baseline 백테스트 실행
- [ ] B-3. `ML_ENABLED=True` 상태로 ML 백테스트 실행
- [ ] B-4. `compare_runs.py`로 A/B 비교
- [ ] B-5. 승률 +3%p 이상 개선 시 main branch 반영, 아니면 블렌딩 가중치 조정

### 목표
CCS의 "alpha factor" 서브스코어(15% 가중)에 ML 점수 블렌딩 → **안전한 첫 ML 노출**.

### 변경사항

**수정:** [src/paper_trading/candidate_selector.py:222-238](../src/paper_trading/candidate_selector.py#L222-L238)

```python
def _score_alpha_factor(row: pd.Series) -> float:
    """C. Alpha Factor [0, 1] — ML 블렌딩 지원."""
    # 기존 규칙 기반 계산
    mom = _safe_float(row.get("팩터_모멘텀"))
    # ... 기존 로직 그대로 ...
    rule_based = max(0.0, min((raw + 1.0) / 2.0, 1.0))

    # ML 블렌딩 (ML_ENABLED=False면 즉시 반환)
    from screener.config import ML_ENABLED, ML_BLEND_WEIGHT_ALPHA
    if not ML_ENABLED:
        return rule_based
    try:
        from ml.prediction import predict_entry_quality
        ml_score = predict_entry_quality(row)
        return ML_BLEND_WEIGHT_ALPHA * ml_score + (1 - ML_BLEND_WEIGHT_ALPHA) * rule_based
    except Exception as e:
        logger.warning(f"ML alpha blend failed, fallback to rules: {e}")
        return rule_based
```

**수정:** `src/screener/config.py` — `ML_ENABLED` 등 상수 import 경로 설정

### 검증

```bash
# Baseline (ML 꺼짐)
# config.py: ML_ENABLED = False
python scripts/run_sp500_backtest.py --period 5y --capital 5000 --fundamentals

# ML 적용
# config.py: ML_ENABLED = True
python scripts/run_sp500_backtest.py --period 5y --capital 5000 --fundamentals

# 비교
python scripts/compare_runs.py --all
python scripts/compare_runs.py  # 최근 2개 상세 비교
python scripts/analyze_trades.py  # 피처별 성과
```

**성공 기준:**
- 승률 +3%p 이상 개선 (53% → 56%+)
- 평균 수익 악화 없음
- 최대 손실 악화 없음
- 거래 수 ±20% 이내 (급격한 변화 없음)

---

## Phase C — Risk + Strategy Score 블렌딩 (최대 영향)

### 체크리스트
- [ ] C-1. `_score_risk_quality()` 블렌딩 (CCS 25% 가중)
- [ ] C-2. `_calc_bottom_reversal_score()` 블렌딩 (signals.py)
- [ ] C-3. `_calc_momentum_score()` 블렌딩 (signals.py)
- [ ] C-4. 백테스트 A/B 비교 → 거래 수 ±30% 이내 확인
- [ ] C-5. 섹터/레짐별 승률 편향 검토

### 목표
- `_score_risk_quality()` (CCS 25% 가중) 블렌딩
- `_calc_bottom_reversal_score()` / `_calc_momentum_score()` — **hard filter 게이트를 통과시키는 점수** → 가장 큰 레버리지

### C-1. Risk Quality 블렌딩

**수정:** [src/paper_trading/candidate_selector.py:241](../src/paper_trading/candidate_selector.py#L241) `_score_risk_quality()`

Phase B와 같은 패턴. 단, 여기서는 **손실 회피 전용 모델** 사용:

```python
ml_score = predict_entry_quality(row, target="y_drawdown_avoid")
# y_drawdown_avoid = 1 - y_drawdown (손실 회피 확률)
```

### C-2/C-3. Strategy Fit (바닥반등/모멘텀 적합도) 블렌딩

**수정:** `src/screener/signals.py` — `_calc_bottom_reversal_score()` / `_calc_momentum_score()`

```python
def _calc_bottom_reversal_score(row) -> float:
    rule_score = _existing_logic(row)  # 0-10 스케일
    from screener.config import ML_ENABLED, ML_BLEND_WEIGHT_STRATEGY
    if not ML_ENABLED:
        return rule_score
    try:
        from ml.prediction import predict_entry_quality
        # big_win 분류기 (return_pct > 7% 확률)
        ml_conf = predict_entry_quality(row, target="y_big_win")
        ml_score_10pt = ml_conf * 10.0
        return ML_BLEND_WEIGHT_STRATEGY * ml_score_10pt + (1 - ML_BLEND_WEIGHT_STRATEGY) * rule_score
    except Exception:
        return rule_score
```

⚠️ **주의:** 이 점수는 [candidate_selector.py](../src/paper_trading/candidate_selector.py)의 hard filter (`CANDIDATE_MIN_STRATEGY_SCORE = 6.0`)를 통과하는 게이트 — 블렌딩으로 **게이트 통과 후보가 달라질 수 있음**. 반드시 `compare_runs.py`로 거래 수 변화 확인.

### 검증

1. 백테스트 A/B 비교 (Phase B와 동일 방식)
2. 거래 건수 급격한 변화(±30%+) 시 `ML_BLEND_WEIGHT_STRATEGY` 조정
3. 섹터/레짐별 승률 변화 검토 — 특정 섹터 편향 발견 시 훈련 데이터 보강

---

## Phase D — 상승 예측 모델 업그레이드 (Empirical → XGBoost)

### 체크리스트
- [ ] D-1. `src/ml/upside_regression.py` 작성 (XGBoost 분위수 회귀)
- [ ] D-2. `upside_model.predict_upside()` ML 폴백 추가
- [ ] D-3. `engine.py`에서 row 전달하도록 수정
- [ ] D-4. 백테스트 log에 `predicted_upside_*` 기록 → 실제 max_drawup과 MAE 비교

### 목표
[src/paper_trading/upside_model.py](../src/paper_trading/upside_model.py)의 Phase 2C empirical 분포를 XGBoost 회귀로 교체. 현재는 18개 버킷에 128거래뿐이라 희소성(sparsity) 심각 → 모델이 버킷 간 일반화 가능.

### 변경사항

**신규:** `src/ml/upside_regression.py`
- 타겟: `max_drawup` (기존 `upside_model._compute_max_drawup()` 재사용)
- 4종 분위수 회귀 모델 (XGBoost `objective=reg:quantileerror`, alpha=0.25/0.5/0.75/0.9)
- 또는 단일 mean 회귀 + 잔차 분포로 분위수 추정

**수정:** `src/paper_trading/upside_model.py` `predict_upside()`

```python
def predict_upside(strategy, ccs_score, star_rating, row=None):
    if ML_ENABLED and row is not None:
        try:
            from ml.upside_regression import predict_upside_ml
            return predict_upside_ml(row)  # {p25, p50, p75, p90, level="ml"}
        except Exception:
            pass
    # 기존 empirical 로직 폴백
    return _existing_predict_upside(strategy, ccs_score, star_rating)
```

**호출측 영향:** [src/paper_trading/engine.py:392-417](../src/paper_trading/engine.py) — `row` 추가 전달 필요 (최소 변경)

---

## Phase E — 매도 타이밍 최적화

### 체크리스트
- [ ] E-1. defer 사례 데이터셋 구축 (defer_count > 0, 최소 50건)
- [ ] E-2. `predict_exit_signal()` 모델 훈련
- [ ] E-3. `_should_defer_sell()`에 ML override 추가
- [ ] E-4. A/B로 defer 거래의 사후 수익률 비교

### 목표
매도 시점을 ML로 보강. 위험이 크므로 **직접 매도 결정은 안 하고 "defer 판단" 보조용**으로만 사용.

### 변경사항

**수정:** [src/paper_trading/engine.py:159](../src/paper_trading/engine.py#L159) `_should_defer_sell()`

```python
# 기존 5개 체크 → 다수결 로직 뒤에 ML 보조
if passed >= HOLD_WINNERS_MIN_CHECKS:
    # ML이 강한 반대 신호면 defer 취소 (보수적 사용)
    if ML_ENABLED:
        exit_prob = predict_exit_signal({
            **pos,
            "current_rsi": rsi, "current_adx": adx,
            "current_vol_z": vol_z, "current_ret_5d": ret_5d,
        })
        if exit_prob > 0.75:
            debug["ml_override"] = f"exit_prob={exit_prob:.2f}"
            return False, debug
    return True, debug
```

**훈련 데이터:** 기존 거래 중 **defer_count > 0** 사례들 — defer 후 결과(추가 상승 vs 되돌림)를 라벨로 사용. 최소 50건 확보 필요.

---

## Phase F — 재학습 파이프라인 + 드리프트 감지

### 체크리스트
- [ ] F-1. `.github/workflows/retrain-ml.yml` 생성 (매주 월요일 4AM EST)
- [ ] F-2. `src/ml/drift_detection.py` 작성
- [ ] F-3. `run_paper_trading.py` 말미에 drift 체크 호출 추가
- [ ] F-4. Drift 감지 시 이메일 경보 연동
- [ ] F-5. 주간 피처 중요도 리포트 자동 업데이트

### F-1. 주간 재학습 (GitHub Actions)

**신규:** `.github/workflows/retrain-ml.yml`

```yaml
name: Retrain ML Models
on:
  schedule:
    - cron: "0 8 * * 1"  # 매주 월요일 4AM EST
  workflow_dispatch:
jobs:
  retrain:
    steps:
      - checkout
      - setup-python 3.11
      - pip install -r requirements.txt
      - python -m ml.training
      - commit data/ml_models/* with message "chore: ML 모델 재학습 [skip ci]"
```

**의도:** 기존 `run-screener.yml`과 분리 → 데일리 파이프라인 런타임에 영향 없음.

### F-2. 드리프트 감지

**신규:** `src/ml/drift_detection.py`

```python
def check_model_drift(
    trades_df: pd.DataFrame,
    model_name: str = "entry_quality",
    window: int = 30,
    threshold: float = 0.45,
) -> dict:
    """최근 N거래에서 모델 예측 vs 실제 결과 정확도 측정.
    threshold 미만이면 drift=True + 경보.
    """
```

**통합:** 데일리 `run_paper_trading.py` 말미에 호출 → drift 감지 시 이메일 알림 + `data/ml_models/drift_alerts.json` 기록

### F-3. 피처 중요도 자동 리포트

주 1회 재학습 시 상위 10개 피처를 `docs/ml_feature_importance.md` 에 자동 업데이트 — 수동 튜닝 참고용.

---

## 수정되는 파일 요약

| Phase | 신규/수정 | 경로 |
|-------|-----------|------|
| A | 수정 | [requirements.txt](../requirements.txt) |
| A | 수정 | [src/screener/config.py](../src/screener/config.py) — ML_* 상수 추가 |
| A | 신규 | [src/ml/__init__.py](../src/ml/), `config.py`, `feature_engineering.py`, `training.py`, `prediction.py`, `evaluation.py` |
| A | 신규 | [data/ml_models/](../data/ml_models/) — artifact 저장소 |
| B | 수정 | [src/paper_trading/candidate_selector.py:222-238](../src/paper_trading/candidate_selector.py#L222-L238) `_score_alpha_factor()` |
| C | 수정 | [src/paper_trading/candidate_selector.py:241](../src/paper_trading/candidate_selector.py#L241) `_score_risk_quality()` |
| C | 수정 | [src/screener/signals.py:261-432](../src/screener/signals.py#L261-L432) `_calc_bottom_reversal_score()`, `_calc_momentum_score()` |
| D | 신규 | [src/ml/upside_regression.py](../src/ml/) |
| D | 수정 | [src/paper_trading/upside_model.py](../src/paper_trading/upside_model.py) `predict_upside()` |
| D | 수정 | [src/paper_trading/engine.py:392-417](../src/paper_trading/engine.py) — row 전달 |
| E | 수정 | [src/paper_trading/engine.py:159](../src/paper_trading/engine.py#L159) `_should_defer_sell()` |
| F | 신규 | `.github/workflows/retrain-ml.yml` |
| F | 신규 | [src/ml/drift_detection.py](../src/ml/) |
| 문서 | 수정 | [docs/analyze_trading_strategy.md](analyze_trading_strategy.md) Phase 3 섹션 갱신 |

---

## 재사용할 기존 유틸리티

| 유틸리티 | 위치 | 용도 |
|---------|------|------|
| `_load_recent_trades()` | [upside_model.py:80](../src/paper_trading/upside_model.py#L80) | 학습 데이터 로드 (run 중복 제거 포함) |
| `_compute_max_drawup()` | [upside_model.py](../src/paper_trading/upside_model.py) | Phase D 타겟 계산 |
| `_extract_entry_features()` | [backtest.py:235](../src/paper_trading/backtest.py#L235) | 피처 스키마 1:1 매칭 기준 |
| `_safe_float()` / `_safe_num()` | candidate_selector, engine | NaN 안전 변환 |
| `compare_runs.py` | [scripts/compare_runs.py](../scripts/compare_runs.py) | A/B 백테스트 비교 |
| JSON 캐시 패턴 | `upside_model._load_cache()` | 모델 artifact lazy 로드 |

---

## 리스크 & 완화

| 리스크 | 완화 |
|--------|------|
| 데이터 부족 (202건 < 최소 250건) | 먼저 `--period 5y --fundamentals` 백테스트로 데이터 확보 후 Phase A 착수 |
| 과적합 | TimeSeriesSplit + 피처 중요도 상위 20개만 사용 옵션 |
| ML 장애로 프로덕션 중단 | 모든 호출부에 try/except + `ML_FALLBACK_ON_MISSING=True` |
| 블렌딩 가중치 튜닝 난이도 | `ML_BLEND_WEIGHT_*`를 config로 분리 → 코드 수정 없이 A/B |
| GitHub Actions 시간 초과 | 재학습은 별도 워크플로우 분리 (데일리와 무관) |
| 모델 artifact 크기 | XGBoost 직렬화 시 `.pkl` 크기 < 1MB 예상 — git에 커밋 OK |

---

## 추천 착수 순서

1. **즉시 (오늘)**: 데이터 확보 — `python scripts/run_sp500_backtest.py --period 5y --capital 5000 --fundamentals` 실행 → 거래 수 확인
2. **거래 250건+ 이면 Phase A** 착수 (순차 진행)
3. **Phase B 성공 시에만 Phase C** 진행 (블렌딩 가중치 점진적 상향: 0.4 → 0.5 → 0.6)
4. **Phase D/E는 Phase B~C 운영 데이터 축적 후** (1~2개월 후)
5. **Phase F는 Phase B부터 병행 가능** (재학습 워크플로우만 먼저 세팅해도 OK)

---

## 참고 문서

- [docs/paper_trading_candidate_selection.md](paper_trading_candidate_selection.md) — CCS 스코어링 상세
- [docs/architecture.md](architecture.md) — 전체 시스템 아키텍처
- [docs/archive/analyze_trading_strategy.md](archive/analyze_trading_strategy.md) — Phase 2C 배경 (아카이브)
- [docs/archive/quant_trading_progress.md](archive/quant_trading_progress.md) — 마일스톤 기록 (아카이브)
