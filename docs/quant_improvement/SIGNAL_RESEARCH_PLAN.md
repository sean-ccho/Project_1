# 신호 연구 2차 (Tier 3-B) — 구현 가이드 A→Z

> 작성 2026-10-05 · 회사 노트북에서 코드를 읽고 작성 (pandas가 없어 **실행 검증은 못 함**).
> **메인 컴퓨터에서 이 문서 순서대로 진행한다.** 진행 규칙은 [HANDOFF.md](HANDOFF.md) 2절을 그대로 따른다.
> 코드 블록은 **골격**이다. 실제 컬럼명·파일 경로는 메인 컴퓨터에서 확인하며 완성한다.

---

## 0. 목표 · 배경 · 전체 순서

**목표**: 지금 스타일(스크리너 · 차트 패턴 · 스윙)로 **비용을 빼고도 남는 진입 신호**를 찾는다. 찾으면 PT-1에 넣고, 그 다음에만 Optuna로 청산을 다듬는다.

**Tier 3 결론의 범위**: 지금까지는 "S&P 500 · 가격/거래량 팩터 · 5~20일 · 팩터 하나씩 또는 순위 평균"만 검증했다. 아직 안 본 것:

1. 스크리너가 **실제로 내놓는 출력**: 차트 패턴, 전략구분, 판단, buy_signal, 적합도, CCS.
   - 패턴은 콤마 문자열이라 숫자 필터에서 빠졌다 ([factor_research.py](../../scripts/factor_research.py#L88-L92)).
   - 나머지는 피처 캐시 **이후**에 계산돼 패널에 아예 없다 ([screener/backtest.py](../../src/screener/backtest.py#L182-L220)).
   - 백테스트는 "1등만, 최대 3개"라 181건뿐이다.
2. 팩터끼리의 **조건부 효과** (단기 반전 × 거래량·갭)
3. **실적 이벤트** (PEAD)
4. **비선형 결합** (패널 ML)

**Optuna를 뒤로 미루는 이유**: 엣지가 0인 전략도 3년 백테스트에서 100번 시도하면 최고 Sharpe 기댓값이 약 1.4다(독립 시도 가정). 청산 파라미터는 기대수익보다 승률↔손익비 모양을 바꾼다. → 진입 신호가 검증된 뒤(7단계)에만 쓴다.

| 단계 | 내용 | 새로 만들거나 고치는 파일 | 다음으로 넘어가는 조건 |
|---|---|---|---|
| 0 | 준비: 동기화, 공통 도구, 시도 기록, **버그 진단** | `research_utils.py`, `signal_portfolio_sim.py`, `TRIAL_LOG.csv`, `evaluation.py` | 테스트 통과 |
| 1 | 스크리너 출력 이벤트 스터디 (**최우선**) | `signal_event_study.py` | 후보 유무와 상관없이 2로 |
| 2 | 조건부 반전 (거래량·갭) | `conditional_research.py` | 〃 |
| 3 | 실적 이벤트 (PEAD) | `fetch_earnings_dates.py`, `pead_research.py` | 〃 |
| 4 | 패널 ML | `panel_ml.py` | 〃 |
| 5 | (선택) 보유 기간 늘리기 | `add_forward_returns.py` | 〃 |
| 6 | 백테스트 일괄 정비 → BASE 재측정, PT-2·3 사전 기준, 후보 A/B | 해시 파일 **한 번에** 수정 | 판정 `채택 후보` |
| 7 | Optuna (채택 후보가 있을 때만) | `optimize_optuna.py` | WFO `채택 후보` + DSR ≥ 0.95 |
| 8 | 홀드아웃 1회 | — | 8-3 기준 |
| 9 | 반영 · 봇 처리 · 문서 | config 🛑 | — |

**중단 규칙 (미리 정함)**: 1~4단계에서 `후보`가 하나도 없으면 종목 선택 연구를 끝낸다. 6단계는 버그 수정 · BASE 재측정 · PT-2·3 사전 기준만 하고 9단계(지수 기준선 · 봇 처리)로 간다.

---

## 1. 모든 단계 공통 규칙

### 1-1 데이터 · 기간 · 비용
- 개발 구간은 **2025-09-30까지**. 홀드아웃(2025-10-01~)은 8단계에서 **한 번만** 본다.
- 수익률: 날짜 t 종가로 신호 → **t+1 시가 진입 → t+1+h 시가 청산** (패널의 `fwd_ret_{h}d`와 같음).
- 비용: 편도 0.1% (`BACKTEST_COST_PER_SIDE`). 이벤트 1건 = 왕복 0.2%.

### 1-2 사전 등록 (수익률을 보기 전에 정한다)
1. 이벤트·조건 목록과 기간(h)을 정하고 **개수만** 센다.
2. 검정 수 `n_family`와 임계 `z = bonferroni_z(n_family)`를 `TRIAL_LOG.csv`에 적는다 (판정 = `등록`).
3. 그 다음에 수익률을 계산하고 결과를 같은 파일에 적는다.
- 결과를 본 뒤 목록을 바꾸면 **새 family로 다시 등록**한다. 버린 시도도 기록에 남긴다.

### 1-3 판정 (1~5단계 공통)

| 등급 | 조건 | 다음 |
|---|---|---|
| **후보** | `t_nw ≥ z` · 비용 후 평균 > 0 · 연도 2/3 이상 양수 · 앞/뒤 반 모두 양수 | 0-6 포트폴리오 시뮬 → SPY 대비 ΔSharpe 95% 하한 > 0이면 6단계 A/B |
| 관찰 | `t_nw ≥ 2` · 비용 후 > 0 · 앞/뒤 반 모두 양수 | 채택하지 않음. 다른 표본(1-5, 실거래)에서만 재확인 |
| 역신호 | `t_nw ≤ −z` | 스크리너 점수·필터에서 빼는 후보 → 6단계 A/B |
| 탈락 | 그 외 | 기록만 |

- t값은 **Newey-West**(`nw_tstat`, lag ≥ h)로 계산한다. 기존 `t_nonoverlap`(시작점 하나만 쓰는 방식)은 참고용으로만 본다.
- 보고서 머리에 **누적 시도 수**(TRIAL_LOG 합계)를 적는다. 최종 채택의 마지막 방어선은 DSR(7단계)과 홀드아웃(8단계)이다.

### 1-4 해시 파일 — 6단계 전에는 고치지 않는다
`src/paper_trading/backtest.py`의 `_HASHED_SOURCES`(features · alpha_model · patterns · fundamentals · processing · signals · screener/backtest · **config**) 중 하나라도 바꾸면 피처 캐시 키가 바뀐다. 그러면 **전체 피처를 다시 계산**한다(HANDOFF 7절 방법 B 규모). 1~5단계는 `scripts/`와 해시 대상이 아닌 파일만 고친다. 해시 파일 수정은 6단계에서 **한 번에** 모으고, 캐시도 한 번만 만든다.

### 1-5 커밋
- 커밋 메시지는 `type: 한국어 설명 [skip ci]`. 바뀐 파일만 `git add <file>`.
- **커밋함**: `scripts/*.py`, `tests/*`, `docs/quant_improvement/tier3/*`(md·csv, TRIAL_LOG 포함), `output/hypothesis_ab.csv`.
- **커밋 안 함**: `data/research/*`, `data/cache/*`, `output/runs/*`, `data/paper_trading/*`(봇 전용).
- 단계가 끝날 때마다 [WORK_SUMMARY.md](WORK_SUMMARY.md) 9절에 한 줄 쓰고 [HANDOFF.md](HANDOFF.md) 0·5절을 갱신한다.

---

## 2. 0단계 — 준비

### 0-1 동기화
- px10y 작업분이 커밋 안 돼 있으면 커밋한다. 대상: `scripts/build_price_panel.py`, 수정된 `factor_research.py`·`composite_research.py`, `tier3/*px10y*`.
  ```bash
  git status --short
  git add scripts/build_price_panel.py scripts/factor_research.py scripts/composite_research.py docs/quant_improvement/tier3/
  git commit -m "feat: 10년 가격 패널(px10y)·팩터/합성 리서치 [skip ci]"
  ```
- 노트북에서 온 문서 3개(이 문서, HANDOFF.md, WORK_SUMMARY.md)를 반영한다.
  - 메인 쪽 HANDOFF·WORK_SUMMARY가 2026-10-05 이후 바뀌지 않았으면 덮어쓴다.
  - 바뀌었으면 HANDOFF 0·5·6절과 WORK_SUMMARY 9절 마지막 줄만 옮긴다.
- HANDOFF 3·4·6·7절을 px10y 기준으로 고친다. 파일 지도에 `build_price_panel.py`를, 데이터 번들에 `panel_px10y.parquet`와 10년 OHLCV 캐시 파일명을 넣는다. 메인에만 있는 정보라 거기서 한다.

### 0-2 환경 · 데이터
```bash
source .venv/bin/activate
pip install -r requirements.txt -r requirements-research.txt pytest   # lightgbm·scipy·optuna·ta 포함
PYTHONPATH=.:src pytest tests -q   # 168 통과 · 2 실패(기존 portfolio_report)면 정상
```

| 파일 | 쓰는 단계 | 비고 |
|---|---|---|
| `data/research/panel_pit.parquet` | 1 | 3년, 스크리너 피처 + 패턴 문자열. **SPY 행 없음** |
| `data/cache/ohlcv_e7842ef0a144.parquet` | 1, 0-6 | PIT 5년 일봉, **SPY 포함**. ⚠️ 2026-10-02까지 있음(홀드아웃 포함) → 로드 직후 `.loc[:"2025-09-30"]` |
| `data/research/panel_px10y.parquet` | 2·3·4 | 10년 가격 팩터 |
| `data/cache/ohlcv_px10y.parquet` (px10y용 10년 OHLCV, `build_price_panel.py`가 읽고 씀) | 2·3·4, 0-6 | 2015-01-02 ~ 2025-09-30. **SPY·섹터 ETF 없음** → 0-6 SPY 기준은 `etf_ohlcv.parquet` 또는 e784 캐시(잘라서)에서 |
| `data/research/etf_ohlcv.parquet` | 1 | 섹터 ETF 11개 + SPY 10년 (1-1에서 생성) |
| `data/universe/sp500_membership.csv` | 전부 | git |

### 0-3 공통 함수 — `scripts/research_utils.py` (새 파일)
```python
#!/usr/bin/env python3
"""Tier 3-B 연구 공통 함수: 패널 로드, Newey-West t값, 다중검정 임계, 시도 기록."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "data" / "research"
TIER3_DIR = ROOT / "docs" / "quant_improvement" / "tier3"
TRIAL_LOG = TIER3_DIR / "TRIAL_LOG.csv"
DEV_END = pd.Timestamp("2025-09-30")
COST_PER_SIDE = 0.001  # screener.config.BACKTEST_COST_PER_SIDE 와 같게
ROUND_TRIP = 2 * COST_PER_SIDE


def load_panel(name: str) -> pd.DataFrame:
    """data/research/panel_{name}.parquet 을 개발 구간(≤ DEV_END)만 읽는다."""
    panel = pd.read_parquet(RESEARCH_DIR / f"panel_{name}.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    return panel[panel["date"] <= DEV_END].reset_index(drop=True)


def nw_tstat(x: pd.Series | np.ndarray, lag: int) -> float:
    """평균의 Newey-West(Bartlett) t값. h일 겹치는 수익률이면 lag ≥ h-1."""
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = v.size
    if n < max(20, 2 * lag + 2):
        return float("nan")
    e = v - v.mean()
    s = float(e @ e) / n
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1)) * float(e[k:] @ e[:-k]) / n
    return float(v.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def bonferroni_z(n_tests: int, alpha: float = 0.05) -> float:
    """양측 alpha / n_tests 에 해당하는 |t| 임계."""
    return float(norm.isf(alpha / max(1, n_tests) / 2))


def yearly_mean(s: pd.Series) -> pd.Series:
    """날짜 인덱스 시계열 → 연도별 평균."""
    return s.groupby(pd.DatetimeIndex(s.index).year).mean()


def halves_mean(s: pd.Series) -> tuple[float, float]:
    """날짜 순으로 반을 나눈 앞·뒤 평균."""
    s = s.dropna().sort_index()
    mid = len(s) // 2
    return float(s.iloc[:mid].mean()), float(s.iloc[mid:].mean())


def log_trials(step: str, family: str, n_family: int, rows: list[dict]) -> None:
    """검정 등록·결과를 TRIAL_LOG.csv 에 덧붙인다 (git에 커밋)."""
    fields = ["실행시각", "step", "family", "n_family", "test_id", "t", "판정", "메모"]
    new = not TRIAL_LOG.exists()
    TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRIAL_LOG.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        now = f"{datetime.now():%Y-%m-%d %H:%M}"
        for r in rows:
            w.writerow({"실행시각": now, "step": step, "family": family, "n_family": n_family,
                        "test_id": r.get("test_id", ""), "t": r.get("t", ""),
                        "판정": r.get("판정", ""), "메모": r.get("메모", "")})


def total_trials() -> int:
    """TRIAL_LOG 의 family별 n_family 합 (같은 family는 한 번만)."""
    if not TRIAL_LOG.exists():
        return 0
    df = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return int(df.drop_duplicates("family")["n_family"].sum())
```
- 테스트 `tests/test_research_utils.py`:
  - iid 정규 난수에서 `nw_tstat(x, 0)`은 단순 t와 같아야 한다.
  - `bonferroni_z(1)`은 약 1.96, `bonferroni_z(150)`은 약 3.59.

### 0-4 시도 기록 — `docs/quant_improvement/tier3/TRIAL_LOG.csv` (새 파일)
과거 시도를 먼저 넣는다 (누적 276건).
```csv
실행시각,step,family,n_family,test_id,t,판정,메모
2026-10-03 00:00,tier2,ab_t2,3,(합계),,기록,T2-5·10·20 모두 운과 구분 안 됨 (hypothesis_ab.csv)
2026-10-03 00:00,tier3-2,pit_factor_ic,132,(합계),,기록,44팩터×3기간 · 게이트 통과 1
2026-10-03 00:00,tier3-3,pit_composite,18,(합계),,기록,통과 0
2026-10-05 00:00,tier3-2,px10y_factor_ic,105,(합계),,기록,35팩터×3기간 · 게이트 통과 1
2026-10-05 00:00,tier3-3,px10y_composite,18,(합계),,기록,통과 0
```

### 0-5 기존 스크립트 보강 (권장)
- `factor_research.py`: 결과에 `t_nw = nw_tstat(ic, lag=h - 1)` 열을 추가한다. 정렬 기준은 그대로 둔다.
- `composite_research.py`: 상수를 세트별로 나눈다. px10y는 매년 재학습해 OOS를 2019~2025로 늘린다. 4단계 ML과 같은 OOS 구간이라 비교 기준이 된다.
  ```python
  SET_PARAMS: dict[str, dict] = {
      "pit":   {"split": "2024-08-16", "refits": ["2023-08-18", "2024-08-16"], "prior_trials": 132},
      "px10y": {"split": "2021-01-01", "refits": [f"{y}-01-01" for y in range(2019, 2026)], "prior_trials": 105},
  }
  ```
  - `SPLIT`·`FIRST_OOS`(= `refits[0]`)·`refit_dates`·`PRIOR_TRIALS`를 `SET_PARAMS[args.set]`에서 읽는다.
  - px10y를 다시 돌리면 시도 18건을 TRIAL_LOG에 새로 적는다 (family `px10y_composite_wf_annual`).

### 0-6 패널 포트폴리오 시뮬레이터 — `scripts/signal_portfolio_sim.py` (새 파일)
후보를 무거운 PT 엔진에 넣기 전에 빠르게 거른다. 피처 캐시나 config에 의존하지 않는다.
```python
"""Tier 3-B 공통: 신호 → 겹치는 h일 보유 포트폴리오 (패널 기반 빠른 백테스트).

규칙: 날짜 d 신호 상위 k개 → d+1 시가 진입, d+1+h 시가 청산. 매일 새 하위 포트폴리오(종목당 비중 1/(k·h))를
만들어 h개가 겹친다. 신호가 k개보다 적으면 남는 비중은 현금. 비용은 편도 COST_PER_SIDE.
입력 events: date, 티커, event_id, score(높을수록 우선)
"""

def open_to_open(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """t 시가 → t+1 시가 수익률 (t에 기록)."""
    opens = ohlcv.xs("Open", axis=1, level="Price").sort_index()
    return opens.shift(-1) / opens - 1.0


def simulate(signals: pd.DataFrame, r_oo: pd.DataFrame, k: int, hold: int,
             cost: float = COST_PER_SIDE) -> pd.Series:
    """신호 → 일간 포트폴리오 수익률 (비용 차감)."""
    dates = r_oo.index
    pos = {d: i for i, d in enumerate(dates)}
    col = {t: j for j, t in enumerate(r_oo.columns)}
    w = np.zeros(r_oo.shape)
    for d, g in signals.groupby("date"):
        i = pos.get(pd.Timestamp(d))
        if i is None or i + 1 + hold >= len(dates):
            continue
        names = [t for t in g.sort_values("score", ascending=False)["티커"] if t in col][:k]
        for t in names:
            w[i + 1 : i + 1 + hold, col[t]] += 1.0 / (k * hold)
    weights = pd.DataFrame(w, index=dates, columns=r_oo.columns)
    gross = (weights * r_oo.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return (gross - turnover * cost).loc[signals["date"].min():]
```
- 비교 대상:
  - SPY 시가→시가 (`r_oo["SPY"]`)
  - 그날 S&P 500 구성종목 동일가중 (`Membership.on(date)`로 열 선택 후 평균)
- 통계는 `paper_trading.evaluation`의 `sharpe`, `cagr`, `max_drawdown`, `alpha_beta(s, {"SPY": spy})`, `paired_bootstrap`을 그대로 쓴다.
- **시뮬 통과** = SPY 대비 `paired_bootstrap`의 `ΔSharpe_하한 > 0`. MDD는 6단계 A/B에서 본다.
- CLI 예:
  ```bash
  PYTHONPATH=.:src python scripts/signal_portfolio_sim.py --events data/research/events_pit.parquet \
      --event-id S7_top1 --ohlcv data/cache/ohlcv_e7842ef0a144.parquet --k 3 --hold 10 --name s7_top1
  ```
- 산출: `tier3/PORTFOLIO_SIM_{name}.md` (CAGR · Sharpe · MDD · 알파(t) · SPY/동일가중 대비 ΔSharpe 95% 구간 · 연도별 수익).
- 한계: 상장폐지 이후 결측 수익은 0(현금)으로 처리한다. 비중 드리프트도 무시한다.

### 0-7 Deflated Sharpe — `src/paper_trading/evaluation.py`에 추가 (해시 대상 아님)
```python
from statistics import NormalDist

_EULER_GAMMA = 0.5772156649015329


def deflated_sharpe(sr: float, sr_trials: Sequence[float], n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014). sr·sr_trials 는 기간당(일간) Sharpe.

    시도 N개의 Sharpe 분산으로 "운으로 기대되는 최고 Sharpe" SR0 를 구하고, sr 이 그보다 클 확률을 낸다.
    """
    n = len(sr_trials)
    if n < 2 or n_obs < 3:
        return float("nan")
    mean = sum(sr_trials) / n
    var = sum((s - mean) ** 2 for s in sr_trials) / (n - 1)
    nd = NormalDist()
    sr0 = math.sqrt(var) * ((1 - _EULER_GAMMA) * nd.inv_cdf(1 - 1 / n)
                            + _EULER_GAMMA * nd.inv_cdf(1 - 1 / (n * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return nd.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denom)
```
테스트 (`tests/paper_trading/test_evaluation.py`에 추가):
```python
def test_deflated_sharpe_penalizes_many_trials():
    rng = random.Random(0)
    one = deflated_sharpe(0.1, [0.1, 0.1], n_obs=756)                     # 분산 0 → PSR(0) ≈ 0.997
    many = deflated_sharpe(0.1, [rng.gauss(0.0, 0.05) for _ in range(200)], n_obs=756)
    assert 0.0 <= many < one <= 1.0
    assert math.isnan(deflated_sharpe(0.1, [0.1], n_obs=756))
```

### 0-8 진단: buy_signal 불일치 (확인만 하고, 수정은 6단계)
노트북에서 코드를 읽다가 발견한 문제 두 개다. 둘 다 **buy_signal이 의도와 다르게 꺼지는** 문제다.

**(a) 백테스트: buy_signal이 항상 False로 보인다**
- `screener/backtest.py`는 강한 섹터를 `price_map`의 섹터 ETF로 계산한다 ([L206-L218](../../src/screener/backtest.py#L206-L218)). 그런데 PT-1·PT-2·PT-3 백테스트의 `tickers`(PIT 구성종목 또는 `TICKERS` + SPY)에는 XLK 같은 섹터 ETF가 없다.
- 그래서 강한 섹터 = ∅ → `in_strong_sector` 전부 False → [signals.py](../../src/screener/signals.py#L196-L201)에서 **buy_signal 전부 False**가 된다.
- 실거래([main.py](../../src/main.py#L83-L103))는 ETF를 따로 받고, `Unknown` 섹터도 통과시킨다.
- 결과: 백테스트에서는 판단 "1. 매수 후보"가 한 번도 안 나와서, 하드필터 2번을 "1. 저점 반등"만 통과한다. **BASE(Sharpe 0.35)는 실거래와 다른 전략을 잰 것**이다.
- 영향 없는 것: Tier 3 팩터 IC(개별 피처), PT-2·3 규칙(buy_signal을 안 씀).

**(b) 실거래: 섹터 이름 불일치**
- fundamentals를 켜면 `섹터`가 yfinance 이름(Technology, Healthcare, …)으로 바뀐다 ([features.py](../../src/screener/features.py#L920-L921), [fundamentals.py](../../src/screener/fundamentals.py#L120)). 강한 섹터는 GICS 이름(`SECTOR_ETFS` 키: Information Technology, Health Care, …)이다.
- 그래서 **Technology · Healthcare · Financial Services · Consumer Cyclical · Consumer Defensive · Basic Materials** 종목은 섹터가 강해도 buy_signal이 꺼진다.
- 노트북에서 실거래 스냅샷 문자열을 확인했을 때 "Technology"·"Healthcare"만 있고 GICS 이름은 없었다.

**확인 방법 (메인, 수정 전)**
```python
import pandas as pd
df = pd.read_parquet("data/paper_trading/sp500_ranked.parquet")   # 실거래 최근 스냅샷
print(df["섹터"].value_counts().head(12))
print(pd.crosstab(df["섹터"], df["buy_signal"]))   # Technology 등에 True가 0이면 (b) 확정
```
- (a)는 코드상 거의 확실하다. 수치로 보려면 `paper_trading/backtest.py`의 `ranked_df` 계산 직후에 **임시로** 아래 출력을 넣고 짧게 돌린다. 임시 코드는 커밋하지 않는다.
  - 넣을 줄: `print(today_str, int(ranked_df["buy_signal"].sum()))`
  - 실행: `run_paper_trading_backtest(period="5y", max_tickers=1000, pit_universe=True, start_date="2025-09-01", end_date="2025-09-30", save_run=False)` (python에서 직접 호출. `run_hypothesis_ab.py`는 A/B 기록에 행이 남으므로 쓰지 않는다)
- **1단계 연구 파이프라인은 처음부터 고친 규칙**(ETF 있음 + 이름 매핑 + Unknown 통과)으로 계산한다. 수정 자체는 6단계에서 한다.

---

## 3. 1단계 — 스크리너 출력 이벤트 스터디 (최우선)

**질문**: 스크리너의 각 출력(패턴 · 전략구분 · 판단 · buy_signal · 하드필터 통과 · CCS 순위)이 뜬 뒤 h일 동안 같은 날 유니버스보다 더 오르는가?

**파일**: `scripts/signal_event_study.py` (새 파일)
- 중간 산출물: `data/research/signals_pit.parquet`, `data/research/events_pit.parquet` (커밋 안 함)
- 결과 (커밋): `tier3/EVENT_COUNTS_pit.csv`, `tier3/EVENT_FAMILY_pit.json`, `tier3/EVENT_STUDY_pit.md`, `tier3/event_study_pit.csv`

```bash
PYTHONPATH=.:src python scripts/signal_event_study.py --set pit --build     # 신호 재계산 (--limit 20 으로 먼저 시험)
PYTHONPATH=.:src python scripts/signal_event_study.py --set pit --count     # 이벤트 수만 → family 등록
PYTHONPATH=.:src python scripts/signal_event_study.py --set pit --analyze   # 수익률·판정
```

### 1-1 `--build`: 날짜마다 실거래 순서를 재현한다

| 순서 | 처리 | 실거래와 다른 점 |
|---|---|---|
| 1 | 패널 하루치 (그날 S&P 500 구성종목) | 실거래 유니버스는 `TICKERS`(S&P 500 + 나스닥) |
| 2 | `섹터`를 GICS 이름으로 매핑 (아래 `SECTOR_ALIASES`) | 고친 규칙 |
| 3 | `최근20일평균거래대금 ≥ LIQUIDITY_DOLLAR_MIN`($5M)만 | 분위수 기준(하위 25% 제거)은 단면 구성에 따라 달라져서 쓰지 않음 |
| 4 | SPY 행 추가 (`close`·`현재가격`·`ema50`·`ema200`, features.py와 같은 `EMAIndicator`) | 시장 필터·레짐 판정용 (패널엔 SPY 없음) |
| 5 | `apply_neutralization` | 같음 |
| 6 | `in_strong_sector` = 강한 섹터 **또는 Unknown** (섹터 ETF 10년 일봉) | 고친 규칙 |
| 7 | `buy_signal_raw` = `_compute_buy_signal` (섹터·시장 필터 전) | 추가 기록 |
| 8 | `attach_signals_and_sort` | 같음 |
| 9 | `select_top_candidates(ranked, [], k=3)` → `debug["all_scores"]`의 ccs, 하드필터 통과, top1·top3 | 보유 종목 없음 가정 |
| 10 | SPY 행 제거 → 필요한 열 + `fwd_ret_*` 저장 | |

```python
import logging

from ta.trend import EMAIndicator

from data.fetch import fetch_ohlcv
from paper_trading.candidate_selector import select_top_candidates
from screener.backtest import _prepare_price_map
from screener.config import LIQUIDITY_DOLLAR_MIN, SECTOR_ETFS
from screener.processing import apply_neutralization
from screener.sector_rotation import get_strong_sectors
from screener.signals import _compute_buy_signal, attach_signals_and_sort

OHLCV_PIT = ROOT / "data/cache/ohlcv_e7842ef0a144.parquet"
ETF_FILE = RESEARCH_DIR / "etf_ohlcv.parquet"
# yfinance 섹터명 → GICS(SECTOR_ETFS 키). 6단계에서 sector_rotation.py 로 옮긴다
SECTOR_ALIASES = {
    "Technology": "Information Technology", "Healthcare": "Health Care",
    "Financial Services": "Financials", "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples", "Basic Materials": "Materials",
}
KEEP = ["date", "티커", "섹터", "최근20일평균거래대금", "in_strong_sector", "buy_signal_raw", "buy_signal",
        "판단", "전략구분", "매수적합도", "바닥반등_적합도", "모멘텀_적합도", "hard_pass", "ccs", "top1", "top3",
        "일봉패턴", "주봉패턴", "월봉패턴", "5일수익률", "거래량Z(20)", "RSI"]


def _spy_frame() -> pd.DataFrame:
    """SPY 종가·EMA (features.py 220~221행과 같은 EMAIndicator)."""
    close = pd.read_parquet(OHLCV_PIT)["SPY"]["Close"].dropna()
    return pd.DataFrame({"close": close, "현재가격": close,
                         "ema50": EMAIndicator(close, window=50).ema_indicator(),
                         "ema200": EMAIndicator(close, window=200).ema_indicator()})


def _etf_map() -> dict[str, pd.DataFrame]:
    """섹터 ETF + SPY 10년 일봉 (처음 한 번 받아 data/research 에 보관)."""
    if not ETF_FILE.exists():
        fetch_ohlcv(list(SECTOR_ETFS.values()) + ["SPY"], period="10y").to_parquet(ETF_FILE)
    return _prepare_price_map(pd.read_parquet(ETF_FILE))


def _strong_flags(sectors: pd.Series, etf_map: dict[str, pd.DataFrame], d: pd.Timestamp) -> pd.Series:
    """실거래 규칙: 강한 섹터 또는 Unknown 이면 True."""
    etf_data = {t: f.loc[:d] for t, f in etf_map.items() if t != "SPY"}
    strong = get_strong_sectors(etf_data, etf_map["SPY"].loc[:d])
    return sectors.isin(strong) | (sectors == "Unknown")


def build(set_name: str, limit: int | None) -> None:
    """패널 → 날짜별 신호 재계산 → signals_{set}.parquet."""
    logging.getLogger().setLevel(logging.WARNING)   # 하드필터 로그가 많다
    panel = load_panel(set_name)
    spy, etf_map = _spy_frame(), _etf_map()
    fwd_cols = [c for c in panel.columns if c.startswith("fwd_ret_")]
    frames = []
    for n, (d, day) in enumerate(panel.groupby("date")):
        if (limit and n >= limit) or d not in spy.index:
            continue
        day = day.copy()
        day["섹터"] = day["섹터"].fillna("Unknown").map(lambda s: SECTOR_ALIASES.get(s, s))
        day = day[day["최근20일평균거래대금"] >= LIQUIDITY_DOLLAR_MIN]
        spy_row = {"티커": "SPY", "시장": "US", "섹터": "Unknown", "최근20일평균거래대금": 1e12, **spy.loc[d].to_dict()}
        day = pd.concat([day, pd.DataFrame([spy_row])], ignore_index=True)
        neutral = apply_neutralization(day)
        neutral["in_strong_sector"] = _strong_flags(neutral["섹터"], etf_map, d)
        strong = neutral["in_strong_sector"].copy()
        raw_buy, _ = _compute_buy_signal(neutral)
        ranked = attach_signals_and_sort(neutral)          # 정렬돼도 인덱스 라벨은 유지된다
        ranked["in_strong_sector"] = strong.reindex(ranked.index)
        ranked["buy_signal_raw"] = raw_buy.reindex(ranked.index).fillna(False).astype(bool)
        picks, debug = select_top_candidates(ranked, [], k=3)
        scores = debug.get("all_scores", {})
        top = [p["ticker"] for p in picks]
        ranked["hard_pass"] = ranked["티커"].isin(list(scores))
        ranked["ccs"] = ranked["티커"].map(lambda t: scores.get(t, {}).get("ccs"))
        ranked["top1"] = ranked["티커"].eq(top[0]) if top else False
        ranked["top3"] = ranked["티커"].isin(top)
        ranked = ranked[ranked["티커"] != "SPY"].assign(date=d)
        frames.append(ranked[[c for c in KEEP + fwd_cols if c in ranked.columns]])
    pd.concat(frames, ignore_index=True).to_parquet(RESEARCH_DIR / f"signals_{set_name}.parquet")
```
- 시작할 때 패널에 필요한 열이 다 있는지 확인한다. 빠진 게 있으면 목록을 출력하고 멈춘다.
  - 필요한 열: `시장`, `섹터`, `close`, `ema20`, `ema50`, `ema200`, `macd_hist`, `RSI`, `volume`, `volume_ma20`, `adx`, `obv`, `obv_ma20`, `obv_mom_5`, `반등스코어`, `5일수익률`, `bollinger_pband`, `트렌드점수`, `거래량Z(20)`, `최근20일평균거래대금`, 패턴 3열
- 끝나면 **날짜별 buy_signal 비율**을 출력한다. 약세장(SPY<EMA200)이 아닌 날에 0이 아니어야 파이프라인이 맞다.

### 1-2 `--count`: 이벤트 수만 센다 (수익률 안 봄)

| ID | 이벤트 | 정의 |
|---|---|---|
| S1_buy_raw | 기술적 매수신호 | `buy_signal_raw` 시작일 |
| S2_buy | 최종 매수신호 | `buy_signal` (시장·섹터 필터 후) 시작일 |
| S3_mom | 모멘텀 전략 | `전략구분`에 "모멘텀" (실제 값 "📈 모멘텀") 시작일 |
| S4_bottom | 바닥반등 전략 | `전략구분`에 "바닥반등" 시작일 |
| S5_judge | 최상위 판단 | `판단` ∈ {"1. 매수 후보", "1. 저점 반등"} 시작일 |
| S6_hard | PT-1 후보군 | `hard_pass` 시작일 |
| S7_top1 | PT-1이 실제 사는 종목 | `top1` (쿨다운 적용) |
| S8_top3 | CCS 상위 3 | `top3` (쿨다운 적용) |
| PD_/PW_/PM_<패턴> | 일봉/주봉/월봉 패턴 | 패턴 문자열(구분자 `", "`)에 등장한 시작일 |
| C1~C3 | 점수 IC | `매수적합도`, `바닥반등_적합도`, `모멘텀_적합도` (`ic_by_date`) |

- **시작일(onset)**: 전 거래일엔 없다가 오늘 생긴 날. 구성종목에서 빠졌다 들어온 경우도 시작일로 본다.
- **쿨다운**: 같은 종목·같은 이벤트는 10거래일 안에 한 번만 센다.
- **family**: `n_events ≥ 300`이고 `n_days ≥ 100`인 이벤트 × h ∈ {5, 10, 20}에 C1~C3 × 3을 더한다.
  - `n_family`와 `z`를 `EVENT_FAMILY_pit.json`과 TRIAL_LOG에 `등록`으로 적는다.
  - `--analyze`는 이 파일이 없으면 실행을 거부한다.

```python
TOP_JUDGMENTS = {"1. 매수 후보", "1. 저점 반등"}
COOLDOWN, MIN_EVENTS, MIN_DAYS = 10, 300, 100


def event_flags(sig: pd.DataFrame) -> pd.DataFrame:
    """행별 이벤트 상태 (시작일 처리 전). 열 = 이벤트 ID."""
    f = pd.DataFrame(index=sig.index)
    f["S1_buy_raw"] = sig["buy_signal_raw"].astype(bool)
    f["S2_buy"] = sig["buy_signal"].astype(bool)
    f["S3_mom"] = sig["전략구분"].astype(str).str.contains("모멘텀")
    f["S4_bottom"] = sig["전략구분"].astype(str).str.contains("바닥반등")
    f["S5_judge"] = sig["판단"].isin(TOP_JUDGMENTS)
    f["S6_hard"] = sig["hard_pass"].astype(bool)
    f["S7_top1"] = sig["top1"].astype(bool)
    f["S8_top3"] = sig["top3"].astype(bool)
    for col, tag in (("일봉패턴", "PD"), ("주봉패턴", "PW"), ("월봉패턴", "PM")):
        s = sig[col].fillna("").astype(str).str.split(", ").explode()
        s = s[(s.str.len() > 0) & (s != "nan")]
        dummies = pd.crosstab(s.index, s).reindex(sig.index, fill_value=0) > 0
        for name in dummies.columns:
            f[f"{tag}_{name}"] = dummies[name]
    return f


def onsets(sig: pd.DataFrame, flag: pd.Series, date_index: dict, cooldown: int = COOLDOWN) -> pd.Series:
    """전 거래일엔 없고 오늘 생긴 이벤트. 같은 종목은 cooldown 거래일 안에 한 번만."""
    df = pd.DataFrame({"t": sig["티커"], "di": sig["date"].map(date_index), "f": flag.astype(bool)})
    df = df.sort_values(["t", "di"])
    prev_f = df.groupby("t")["f"].shift(1).fillna(False).astype(bool)
    prev_di = df.groupby("t")["di"].shift(1)
    start = df["f"] & ~(prev_f & (df["di"] - prev_di == 1))
    keep = pd.Series(False, index=df.index)
    for _, g in df[start].groupby("t"):
        last = -10**9
        for idx, di in zip(g.index, g["di"]):
            if di - last >= cooldown:
                keep[idx] = True
                last = di
    return keep.reindex(sig.index, fill_value=False)
```

### 1-3 `--analyze`: 초과수익 · NW t · 연도별
```python
def event_stats(sig: pd.DataFrame, ev: pd.Series, h: int) -> dict:
    """이벤트 후 h일 초과수익 (같은 날 유니버스 동일가중 평균 대비)."""
    y = f"fwd_ret_{h}d"
    bench = sig.groupby("date")[y].transform("mean")
    ar = (sig[y] - bench)[ev].dropna()
    if ar.empty:
        return {}
    daily = ar.groupby(sig.loc[ar.index, "date"]).mean()     # 같은 날 이벤트는 한 묶음
    yr = yearly_mean(daily)
    first, second = halves_mean(daily)
    return {"n_events": int(ar.size), "n_days": int(daily.size),
            "mean_ar": float(ar.mean()), "mean_net": float(ar.mean() - ROUND_TRIP),
            "hit": float((ar > 0).mean()), "t_nw": nw_tstat(daily, lag=h),
            "years_pos": int((yr > 0).sum()), "n_years": int(yr.size),
            "first_half": first, "second_half": second,
            "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items())}


def grade(r: dict, z: float) -> str:
    """1-3 공통 판정."""
    if not r:
        return "표본 없음"
    halves_up = r["first_half"] > 0 and r["second_half"] > 0
    if r["t_nw"] >= z and r["mean_net"] > 0 and halves_up and r["years_pos"] >= math.ceil(r["n_years"] * 2 / 3):
        return "후보"
    if r["t_nw"] <= -z:
        return "역신호"
    if r["t_nw"] >= 2 and r["mean_net"] > 0 and halves_up:
        return "관찰"
    return "탈락"
```
- C1~C3은 `factor_research.ic_by_date(sig[col], sig[y], sig["date"])`의 평균 IC와 `nw_tstat(ic, lag=h-1)`로 판정한다.
- family 이벤트의 시작일 목록을 `events_pit.parquet`(date, 티커, event_id, score)로 저장한다. score는 S7·S8이면 ccs, 나머지는 1이다. 0-6 시뮬레이터의 입력이 된다.
- 보고서(`EVENT_STUDY_pit.md`)에 넣을 것:
  - 등급별 표
  - 이벤트별 n · 평균 초과수익(비용 전·후) · 적중률 · t_nw · 연도별
  - 누적 시도 수
  - 날짜별 buy_signal 비율

### 1-4 후속
- **후보** → 0-6 시뮬. S7·S8은 k=1·3, h=10이고, 패턴·전략 이벤트는 k=3, h=해당 h다. 시뮬 통과면 6단계 A/B 목록에 올린다.
- **역신호** → 6단계에서 "점수·필터에서 빼기" A/B. 예: 월봉 골든크로스 계열, 알파점수.
- 패턴은 3년 표본이라 드문 패턴은 family에서 빠진다(표본 부족). 그건 결론이 아니라 "판단 불가"다.

### 1-5 (선택, 후보가 있을 때만) 2017~2022 독립 표본으로 재현
발견 구간(2022-08~2025-09)과 겹치지 않는 구간에서 **후보만** 다시 검정한다. family = 후보 수라 임계가 낮다.
1. 피처를 계산한다. 오래 걸리니 한 번만 한다. `run_hypothesis_ab.py`는 A/B 기록을 남기므로 python에서 직접 호출한다.
   ```python
   run_paper_trading_backtest(period="10y", max_tickers=1000, pit_universe=True, end_date="2022-08-17", save_run=False)
   ```
2. 새로 생긴 `data/cache/features/<hash>_v1/` 폴더와 OHLCV 파일 이름으로 `build_research_panel.py`의 `SETS`에 `"pit_early"`를 추가하고 `--set pit_early`로 패널을 만든다.
3. `signal_event_study.py --set pit_early --build`를 돌린다. 그 다음 `--analyze --only <후보 ID들>`로 분석한다 (family 따로 등록).

---

## 4. 2단계 — 조건부 반전 (거래량 · 갭)

**질문**: 유일하게 일관된 신호(단기 반전, 10년 중 10년 같은 부호)가 "거래량·갭이 있는 움직임"과 "없는 움직임"에서 다르게 작동하는가? 뉴스가 있는 움직임은 이어지고 뉴스 없는 움직임은 되돌아간다는 연구가 있다(Chan 2003). 둘이 섞여 서로 상쇄됐을 수 있다.

**파일**: `scripts/conditional_research.py` (새 파일). **데이터**는 px10y 패널과 10년 OHLCV다.
**산출**: `tier3/CONDITIONAL_REPORT_px10y.md`, `tier3/conditional_px10y.csv`

**조건 변수**: 패널 정의가 모호하면 OHLCV에서 직접 계산한다.
```python
def conditioning_vars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """날짜 t 종가 기준: 5일 수익률, 거래량 급증(5일/직전 60일), 최근 5일 최대 갭."""
    close = ohlcv.xs("Close", axis=1, level="Price")
    open_ = ohlcv.xs("Open", axis=1, level="Price")
    vol = ohlcv.xs("Volume", axis=1, level="Price")
    out = pd.concat({
        "ret_5d_c": (close / close.shift(5) - 1).stack(),
        "abn_vol": (vol.rolling(5).mean() / vol.shift(5).rolling(60).mean()).stack(),
        "max_gap_5": (open_ / close.shift(1) - 1).abs().rolling(5).max().stack(),
    }, axis=1)
    out.index.names = ["date", "티커"]          # px10y 패널의 종목 열 이름에 맞춘다
    return out.reset_index()
```

**검정 18건 (사전 등록)**: h ∈ {5, 10, 20} × 아래 6개. 날짜별로 독립 정렬한다(`rank(method="first")` 후 `qcut`). `ret_5d_c`는 5분위(Q1=하락 큰 쪽), `abn_vol`은 3분위(V1=조용함)다.

| ID | 정의 |
|---|---|
| A1 | 조용한 반전폭: V1 안에서 Q1 − Q5 |
| A2 | 거래량 급증 반전폭: V3 안에서 Q1 − Q5 |
| A3 | A1 − A2 |
| B1 | 갭 없는 반전폭: `max_gap_5 < 3%` 안에서 Q1 − Q5 |
| B2 | 갭 있는 반전폭: `max_gap_5 ≥ 3%` 안에서 Q1 − Q5 |
| B3 | B1 − B2 |

- 날짜별 스프레드 시계열 → 평균, `nw_tstat(lag=h-1)`, 연도별, 앞(2016~2020)/뒤(2021~2025-09). `z = bonferroni_z(18) ≈ 2.99`.
- **롱 온리로 번역**: 가장 좋은 셀(예: V1∩Q1)의 동일가중 대비 초과수익에서 왕복 0.2%를 뺀다. 이것도 판정 조건에 넣는다.
- 후보 → 날짜별 셀 종목을 `events_cond.parquet`로 저장 → 0-6 시뮬.
- 주의할 점:
  - `abn_vol`은 65거래일 이력이 필요하다.
  - 10년 패널의 2016~2017년은 구성종목 가격 커버리지가 약 78%다 (`tier3/px10y_coverage.csv`, 생존 편향 잔존). 결과는 앞/뒤 반으로 나눠 본다.

---

## 5. 3단계 — 실적 이벤트 (PEAD)

지금 시스템은 실적 **전** 3일만 피한다 ([config.py](../../src/screener/config.py#L805)). PEAD는 실적 **후** 큰 반응을 보인 종목이 몇 주간 같은 방향으로 가는 현상이다. 오래 검증된 이상현상이지만, 대형주에서는 최근 많이 약해졌다는 연구가 있다. 그래서 직접 확인한다.

### 3-1 대리 이벤트 (새 데이터 없이 바로)
- **이벤트일 t**: `|open_t / close_{t-1} − 1| ≥ 4%`이고 `volume_t ≥ 3 × 직전 20일 평균`이며, t에 S&P 500 구성종목인 경우.
- **방향**: t일 수익률에서 같은 날 구성종목 동일가중 수익률을 뺀 값의 부호.
- **진입·청산**: t+1 시가 진입 → t+1+h 시가 청산. h ∈ {5, 10, 20, 40, 60}. 40·60은 OHLCV에서 직접 계산한다.
- **검정 10건**: {상승 이벤트 초과수익, 하락 이벤트 초과수익} × 5. `z = bonferroni_z(10) ≈ 2.81`.
- 날짜별 평균 → `nw_tstat(lag=h)`. 판정은 1-3과 같다.

### 3-2 실적 발표일 수집 — `scripts/fetch_earnings_dates.py` (새 파일)
SEC EDGAR(무료, 과거 전체, 발표 시각 포함)에서 **8-K Item 2.02**(실적 발표) 제출 기록을 받는다.
- 엔드포인트:
  - 티커→CIK: `https://www.sec.gov/files/company_tickers.json` (현재 상장사만)
  - 제출 기록: `https://data.sec.gov/submissions/CIK##########.json` (10자리, 앞을 0으로 채움). `filings.recent` 배열과 `filings.files`에 나열된 과거 파일들이 있다.
- **SEC 규칙**: User-Agent에 이름·이메일을 넣는다. 초당 10회 이하로 호출한다. 이메일은 코드에 쓰지 말고 환경변수 `SEC_USER_AGENT`로 받는다 (공개 리포).
```python
UA = os.environ["SEC_USER_AGENT"]          # 예: "이름 you@example.com"
SUB = "https://data.sec.gov/submissions/"


def _get(url: str) -> dict:
    time.sleep(0.15)                        # 초당 10회 이하
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json()


def earnings_filings(cik: int) -> list[dict]:
    """8-K(Item 2.02) 제출 기록: form, acceptanceDateTime, filingDate."""
    sub = _get(f"{SUB}CIK{cik:010d}.json")
    blocks = [sub["filings"]["recent"]] + [_get(SUB + f["name"]) for f in sub["filings"].get("files", [])]
    rows = []
    for b in blocks:
        for form, items, acc, filed in zip(b["form"], b["items"], b["acceptanceDateTime"], b["filingDate"]):
            if form in ("8-K", "8-K/A") and "2.02" in str(items).split(","):
                rows.append({"form": form, "accepted": acc, "filed": filed})
    return rows


def first_session(accepted_et: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    """발표 시각 → 처음 반응하는 거래일. 16:00(ET) 이후면 다음 거래일."""
    day = accepted_et.normalize() + pd.Timedelta(days=1 if accepted_et.hour >= 16 else 0)
    i = sessions.searchsorted(day)
    return sessions[i] if i < len(sessions) else pd.NaT
```
- ⚠️ **시간대 확인**: `acceptanceDateTime`이 ET인지 UTC인지 먼저 검증한다. 장 마감 후 발표하는 회사(AAPL 등)의 시각이 16~17시대로 나오면 ET로 처리하고, 20~21시대면 UTC에서 ET로 변환한다.
- 산출: `data/research/earnings_dates.parquet`(ticker, cik, accepted_et, event_date, form)와 `tier3/earnings_coverage.csv`(연도별 구성종목 중 실적일을 찾은 비율).
- 상장폐지 종목은 CIK 매핑이 안 될 수 있다. 커버리지로만 보고한다.

### 3-3 분석 — `scripts/pead_research.py` (새 파일)
- **EAR**: 3일 초과수익 `close(d0+1) / close(d0−2) − 1`에서 같은 구간 구성종목 동일가중을 뺀다.
- **강도 순위**: 직전 365일 이벤트들 중 EAR 백분위. 미래 이벤트를 쓰지 않는다. 직전 이벤트가 100건 미만이면 건너뛴다.
- **진입**: `open(d0+2)` → `open(d0+2+h)`, h ∈ {5, 10, 20, 40, 60}. 같은 구간 동일가중 대비 초과수익.
- **검정 10건**: {상위 20% 초과수익, 상위 − 하위 20%} × 5. 진입일별 평균 → `nw_tstat(lag=h)`.
- 보너스: 과거 실적일이 생기면 하드필터 "어닝 3일 회피"를 백테스트에서도 재현할 수 있다. 지금은 백테스트에서 `days_to_next_earnings`가 비어 있어 항상 통과한다. 6단계에서 함께 반영할 수 있다.

---

## 6. 4단계 — 패널 ML

기존 `entry_quality` 모델은 거래 561건으로 학습해서 CV AUC 평균이 0.48 / 0.54 / 0.54였다(동전 수준). 같은 아이디어를 **패널 수십만 행**에 적용한다.

**파일**: `scripts/panel_ml.py` (새 파일). **데이터**: px10y.
**산출**: `data/research/ml_pred_px10y.parquet`(date, 티커, pred), `tier3/ML_REPORT_px10y.md`, `tier3/ml_importance_px10y.csv`

```python
from lightgbm import LGBMRegressor

H, EMBARGO = 10, 5
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=2000,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              random_state=42, n_jobs=-1, verbose=-1)          # 고정 (사전 등록, 튜닝 안 함)

panel = load_panel("px10y")
feats = [c for c in panel.columns if c not in {"date", "티커"} and not c.startswith("fwd_ret_")
         and pd.api.types.is_numeric_dtype(panel[c])]
X = panel[feats].replace([np.inf, -np.inf], np.nan).groupby(panel["date"]).rank(pct=True).astype("float32")
y = panel.groupby("date")[f"fwd_ret_{H}d"].rank(pct=True) - 0.5
dates = np.array(sorted(panel["date"].unique()))
pred = pd.Series(np.nan, index=panel.index)
for year in range(2019, 2026):                                  # 매년 재학습 (확장 창)
    test_start = pd.Timestamp(f"{year}-01-01")
    train_dates = dates[dates < test_start][: -(H + EMBARGO)]   # 라벨이 검증 구간과 겹치는 끝부분 제거
    tr = panel["date"].isin(train_dates) & y.notna()
    te = (panel["date"] >= test_start) & (panel["date"] <= min(pd.Timestamp(f"{year}-12-31"), DEV_END))
    model = LGBMRegressor(**PARAMS).fit(X[tr], y[tr])
    pred[te] = model.predict(X[te])
```
- 평가 (OOS 2019-01~2025-09만):
  - 날짜별 IC `ic_by_date(pred, fwd_ret_10d)`: 평균, `nw_tstat(lag=9)`, 연도별
  - 상위 10% 초과수익(비용 후)
  - 0-6 시뮬 (k=20, h=10)
- **판정**: 시도 1건으로 사전 등록한다. 아래를 모두 만족해야 한다.
  - OOS IC `t_nw ≥ 3.0`
  - 상위 10% 비용 후 초과수익 > 0인 해가 7년 중 5년 이상
  - 시뮬 SPY 대비 ΔSharpe 하한 > 0
- **비교 기준**: 0-5에서 다시 돌린 px10y 선형 워크포워드(같은 OOS 구간). ML이 이것보다 나아야 의미가 있다.
- **확인 사항**:
  - ① 타깃을 날짜 안에서 섞으면 OOS IC가 약 0이어야 한다.
  - ② `build_price_panel.py`의 `_sn`(섹터 중립)·`resid_*`(잔차)·`beta_252`가 **그날까지의 데이터만** 쓰는지 확인한다. 전체 기간으로 추정한 베타나 평균이면 누설이다.
- 중요도(gain)를 fold 평균해서 저장한다.
- Optuna는 쓰더라도 **각 학습 창 안에서만**(시계열 분할) 쓴다. 검증 연도는 건드리지 않는다.
- ML 후보를 PT-1에 넣으려면 예측값을 스크리너 파이프라인에 넣는 별도 설계가 필요하다. 6단계에서는 "예측 상위 N만 허용" 같은 필터 형태로 시작한다.

---

## 7. 5단계 — (선택) 보유 기간 늘리기

같은 약한 신호라도 매주 교체하면 비용이 연 8% 안팎, 매달 교체하면 연 1% 안팎이다.
- `scripts/add_forward_returns.py --set {pit,px10y} --h 40 60`: OHLCV 시가로 `fwd_ret_40d`·`fwd_ret_60d`를 계산해 패널에 붙인다. 공식은 `build_research_panel.py`와 같다.
  ```python
  fwd = opens.shift(-(1 + h)) / opens.shift(-1) - 1
  s = fwd.stack().rename(f"fwd_ret_{h}d")
  s.index.names = ["date", "티커"]
  panel = panel.merge(s.reset_index(), on=["date", "티커"], how="left")
  ```
- `factor_research.py`·`composite_research.py`에 `--horizons` 인자를 추가하고 px10y로 (20, 60)을 다시 돌린다. 새 family로 등록한다.
- 60일은 10년에서도 독립 표본이 약 40개뿐이다. 판정은 1-3과 같고, 시뮬(h=20·60)로 비용 후 결과를 같이 본다.

---

## 8. 6단계 — 백테스트 일괄 정비 + 후보 A/B

### 6-1 해시 파일 일괄 수정 (한 커밋 · 캐시 재생성 한 번)

| # | 파일 | 내용 | 해시 | 확인 |
|---|---|---|---|---|
| a | `src/screener/sector_rotation.py` | `SECTOR_ALIASES` + `mark_strong_sectors()` 추가 | 아님 | 새 테스트 |
| b | `src/screener/backtest.py` | `mark_strong_sectors` 사용 + `snapshot`에서 섹터 ETF 제외 | **예** | |
| c | `src/paper_trading/backtest.py`, `account_backtest.py` | 섹터 ETF 일봉을 따로 받아 `price_map`에만 추가 (`tickers`·해시는 그대로) | 아님 | buy_signal > 0 |
| d | `src/screener/config.py` | `CANDIDATE_SIGNAL_FILTER = None` (+ `RUNTIME_OVERRIDABLE`, 구현은 candidate_selector) | **예** | 새 테스트 |
| e | `src/screener/features.py` | 1~5단계 후보가 새 피처를 요구할 때만 | **예** | |
| f | 🛑 `src/main.py`, `src/run_full_scan.py` | `mark_strong_sectors` 사용 = **실거래 동작 변경** | 아님 | 별도 커밋 · PT-1 버전 표시 |
| g | 🛑 `TICKERS`의 비주식(VXX 등) 제외 | 실거래 유니버스 변경 | **예** | 별도 커밋 |

```python
# sector_rotation.py
# yfinance 섹터명 → GICS(SECTOR_ETFS 키). 실거래 '섹터'가 yfinance 이름이라 강한 섹터와 매칭이 안 됐다
SECTOR_ALIASES: dict[str, str] = {
    "Technology": "Information Technology", "Healthcare": "Health Care",
    "Financial Services": "Financials", "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples", "Basic Materials": "Materials",
}


def mark_strong_sectors(sectors: pd.Series, strong: set[str]) -> pd.Series:
    """섹터 열 → 강한 섹터 여부 (실거래 규칙: Unknown 은 통과)."""
    norm = sectors.fillna("Unknown").map(lambda s: SECTOR_ALIASES.get(s, s))
    return norm.isin(strong) | (norm == "Unknown")
```
```python
# screener/backtest.py::_compute_ranked_snapshot
_SECTOR_ETF_SET = set(SECTOR_ETFS.values())
for ticker, frame in price_map.items():
    if ticker in _SECTOR_ETF_SET:          # 섹터 강도 계산용일 뿐 후보가 아니다
        continue
    ...
neutral["in_strong_sector"] = mark_strong_sectors(neutral["섹터"], strong_sectors)

# paper_trading/backtest.py · account_backtest.py — price_map 만든 직후
etf_raw = fetch_ohlcv(list(SECTOR_ETFS.values()), period=period, force_download=no_cache)
price_map.update(_prepare_price_map(etf_raw))
```
```python
# config.py — Tier 3-B 후보 규칙 (None = 끔). 일반형이라 이후 후보는 런타임 덮어쓰기로만 바꾼다 (캐시 재생성 없음)
# 예: {"require_any": {"일봉패턴": ["상승삼각형"]}, "exclude_any": {"월봉패턴": ["골든크로스"]},
#      "min": {"모멘텀_적합도": 6.0}, "max": {"5일수익률": -0.03}}
CANDIDATE_SIGNAL_FILTER: dict | None = None
```
```python
# candidate_selector.py — _apply_hard_filters 의 1c(허용 전략) 다음에
def _signal_filter_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    """CANDIDATE_SIGNAL_FILTER 규칙 → 통과 마스크."""
    mask = pd.Series(True, index=df.index)
    for key, want in (("require_any", True), ("exclude_any", False)):
        for col, names in rule.get(key, {}).items():
            parts = df.get(col, pd.Series("", index=df.index)).fillna("").astype(str).str.split(", ")
            hit = parts.apply(lambda xs: any(n in xs for n in names)).astype(bool)
            mask &= hit if want else ~hit
    for col, lo in rule.get("min", {}).items():
        mask &= pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce") >= lo
    for col, hi in rule.get("max", {}).items():
        mask &= pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce") <= hi
    return mask

    # (_apply_hard_filters 안)
    rule = _cfg.CANDIDATE_SIGNAL_FILTER
    if rule:
        mask = _signal_filter_mask(df, rule)
        rejections["후보규칙_외"] = int((~mask).sum())
        df = df[mask]
```
- 테스트:
  - `tests/paper_trading/test_sector_rotation.py`: 별칭 매핑, Unknown 통과
  - `tests/paper_trading/test_pt1_switches.py`: 규칙 None이면 그대로, require/exclude/min/max 동작
- 커밋 예: `fix: 백테스트 섹터 ETF 누락·섹터명 불일치 수정, 후보 규칙 스위치 추가 [skip ci]`. f·g는 사용자 확인 후 따로 커밋한다.

### 6-2 캐시 재생성 + BASE 재측정
```bash
PYTHONPATH=.:src python scripts/run_hypothesis_ab.py --only BASE --period 5y --max-tickers 1000 --pit-universe --end 2025-09-30
```
- 첫 실행이 피처를 다시 계산한다. 끝나면 진단 출력으로 buy_signal이 0이 아닌지 확인한다.
- 새 BASE를 옛 BASE(Sharpe 0.35, 알파 t −0.45)와 비교해 HANDOFF 3절에 적는다. `hypothesis_ab.csv`는 `code_hash`로 구분된다.
- **이후 모든 A/B의 기준선은 새 BASE**다.

### 6-3 🛑 PT-2 · PT-3 사전 기준 백테스트 (계획서 10-8절)
```bash
PYTHONPATH=.:src python scripts/run_account_backtest.py --account pt2 --max-tickers 0 --pit-universe --end 2025-09-30
PYTHONPATH=.:src python scripts/run_account_backtest.py --account pt3 --max-tickers 0 --pit-universe --end 2025-09-30
```
- 같은 피처 캐시를 쓰므로 6-2 다음에 돌린다.
- 일간 Sharpe가 SPY보다 낮으면 계획서 규칙대로 재설계 또는 중단한다. 결정은 사용자 확인 후.

### 6-4 후보 · 역신호 A/B
- `run_hypothesis_ab.py`의 `VARIANTS`에 후보를 하나씩 추가한다.
  ```python
  "S1": ("1단계 후보: <설명>", {"CANDIDATE_SIGNAL_FILTER": {"require_any": {"일봉패턴": ["<패턴>"]}}}),
  "X1": ("역신호 제외: <설명>", {"CANDIDATE_SIGNAL_FILTER": {"exclude_any": {"월봉패턴": ["골든크로스"]}}}),
  ```
- 실행:
  ```bash
  PYTHONPATH=.:src python scripts/run_hypothesis_ab.py --only BASE S1 X1 --period 5y --max-tickers 1000 --pit-universe --end 2025-09-30
  PYTHONPATH=.:src python scripts/run_hypothesis_ab.py --base T2-10 --only S1 --period 5y --max-tickers 1000 --pit-universe --end 2025-09-30
  ```
  - 두 번째 명령은 거래 수를 늘려 검정력을 높이기 위한 것이다.
- 판정이 **`채택 후보`인 것만** 채택한다. 한 번에 하나씩 채택하고, 합치면 새 기준선이 된다. A/B 하나 = 시도 1건으로 TRIAL_LOG에 기록한다.

---

## 9. 7단계 — Optuna (6단계에서 `채택 후보`가 나왔을 때만)

### 7-1 탐색 공간
청산 핵심(기존 `suggest_pt1`)에 후보 규칙의 임계 1~2개를 더한다. **파라미터는 6개 이하, 간격은 굵게** 잡는다.

### 7-2 워크포워드 모드 (`--wfo`)
지금 스크립트는 개발 구간 fold 평균을 최적화하므로 그 점수 자체가 in-sample이다. 정직한 추정은 "최적화 절차"의 OOS 성과다.
```python
def robust_center(study: optuna.Study, top_frac: float = 0.1, min_n: int = 5) -> dict:
    """상위 시도들의 '가운데' 파라미터 (최고점 하나 대신 고원 선택). 숫자는 실제 시도값 중 중앙값."""
    done = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                  key=lambda t: t.value, reverse=True)
    top = done[: max(min_n, int(len(done) * top_frac))]
    center: dict = {}
    for name in top[0].params:
        vals = [t.params[name] for t in top]
        if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            center[name] = sorted(vals)[len(vals) // 2]
        else:
            center[name] = max(set(vals), key=vals.count)
    return center

# main() 안 (--wfo, --folds 4 권장)
oos, base = [], []
for k in range(1, len(folds)):
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(lambda t: objective_on(t, folds[:k]), n_trials=args.trials)   # 기존 objective 를 fold 인자화
    params = SUGGEST[args.account](optuna.trial.FixedTrial(robust_center(st)))
    oos.append(_daily(run_window_full(args.account, params, *folds[k], args)["equity_curve"]))
    base.append(_daily(run_window_full(args.account, {}, *folds[k], args)["equity_curve"]))
a, b = pd.concat(oos), pd.concat(base)
common = a.index.intersection(b.index)
stats = paired_bootstrap(a[common].tolist(), b[common].tolist())
print("[WFO] 이어 붙인 OOS:", verdict(stats), stats)
```
- `run_window_full`은 `run_window`를 고쳐 summary 대신 결과 전체(`equity_curve` 포함)를 돌려주는 함수다. `_daily`는 `run_hypothesis_ab.py`와 같다.
- WFO 전체 trial 수를 TRIAL_LOG에 적는다.

### 7-3 최종 파라미터 + DSR
1. WFO가 `채택 후보`면 개발 구간 전체로 study를 한 번 더 돌리고 `robust_center`를 최종 파라미터로 삼는다.
2. objective에서 trial마다 fold를 이어 붙인 일간 Sharpe(`mean/std`, 연율화 안 함)를 `trial.set_user_attr("sr_daily", …)`로 남긴다.
3. 최종 파라미터의 개발 구간 일간 수익률로 `deflated_sharpe(sr, [t.user_attrs["sr_daily"] for t in trials], n_obs, skew, kurt)`를 계산한다 (`scipy.stats.skew`, `kurtosis(fisher=False)`).
4. **DSR ≥ 0.95**일 때만 8단계로 간다.
- `--evaluate-holdout`이 `study.best_params` 대신 `robust_center(study)`를 쓰도록 고친다.

---

## 10. 8단계 — 홀드아웃 1회

- **8-1 대상**: 최종 설정 1개, 새 BASE, SPY. 구간은 2025-10-01부터 실행일까지다.
- **8-2 실행**: `optimize_optuna.py --evaluate-holdout`을 쓰거나, Optuna를 안 썼으면 `run_hypothesis_ab.py --only BASE S1 --start 2025-10-01 …`. **딱 한 번** 돌린다.
- **8-3 기준**: 약 1년이라 유의성 대신 방향만 본다.
  - 후보 Sharpe ≥ SPY Sharpe
  - 알파 > 0
  - MDD가 새 BASE보다 2%p 넘게 나쁘지 않음
- 결과는 무엇이든 기록한다. 실패하면 채택하지 않는다. **다시 튜닝해서 재평가하지 않는다** (홀드아웃은 소진됨).

---

## 11. 9단계 — 반영 · 봇 처리 · 문서

- 🛑 PT-1 기본 config 변경은 사용자 확인 후 한다.
  - 버전을 표시한다 (계획서 10-8: 성과는 버전별로).
  - 실거래 3개월 또는 30건이 쌓이면 같은 기간 백테스트와 비교한다. 거래당 평균수익 차이가 표준오차의 2배를 넘으면 체결·데이터부터 조사한다.
- 🛑 PT-2 · PT-3: 6-3 결과로 계속할지 중단할지 정한다.
- 후보가 하나도 없으면 종목 선택 연구를 종료한다. 지수 기준선과 봇 처리 결정(이전 HANDOFF 5절)으로 돌아간다.
- 문서: `tier3/` 결과, TRIAL_LOG, WORK_SUMMARY 9절, HANDOFF 0·3·5절.

---

## 12. 체크리스트

- [ ] 0-1 px10y 코드 커밋, 문서 3개 반영, HANDOFF 3·4·6·7절 px10y 갱신
- [ ] 0-2 테스트 168 통과 · 데이터 파일 표 채우기
- [ ] 0-3 `research_utils.py` + 테스트
- [ ] 0-4 `TRIAL_LOG.csv` (과거 276건)
- [ ] 0-5 (권장) `t_nw` 열, composite 세트별 상수 → px10y 연 1회 재학습 재실행
- [ ] 0-6 `signal_portfolio_sim.py`
- [ ] 0-7 `deflated_sharpe` + 테스트
- [ ] 0-8 buy_signal 진단 (a)(b) 결과 기록
- [ ] 1 `signal_event_study.py`: build → count(등록) → analyze → 후보 시뮬
- [ ] 2 `conditional_research.py` (18건 등록 → 분석)
- [ ] 3-1 대리 이벤트 (10건) → 3-2 EDGAR 수집(시간대 확인) → 3-3 PEAD (10건)
- [ ] 4 `panel_ml.py` (1건, 섞기 확인 · 누설 확인 · 선형 WF와 비교)
- [ ] 5 (선택) 40·60일
- [ ] 6-1 해시 파일 일괄 수정 (a~e) · 🛑 f·g는 따로
- [ ] 6-2 캐시 재생성 + 새 BASE
- [ ] 6-3 🛑 PT-2·3 사전 기준 백테스트
- [ ] 6-4 후보·역신호 A/B
- [ ] 7 (채택 후보 있을 때) Optuna WFO + DSR
- [ ] 8 홀드아웃 1회
- [ ] 9 🛑 반영·봇 처리, WORK_SUMMARY 9절 · HANDOFF 0·5절

---

## 부록. 새로 만들거나 고치는 파일

| 파일 | 단계 | 해시 대상 | 커밋 |
|---|---|---|---|
| `scripts/research_utils.py` | 0 | 아님 | O |
| `scripts/signal_portfolio_sim.py` | 0 | 아님 | O |
| `src/paper_trading/evaluation.py` (`deflated_sharpe`) | 0 | 아님 | O |
| `scripts/factor_research.py`, `composite_research.py` (보강) | 0·5 | 아님 | O |
| `scripts/signal_event_study.py` | 1 | 아님 | O |
| `scripts/conditional_research.py` | 2 | 아님 | O |
| `scripts/fetch_earnings_dates.py`, `pead_research.py` | 3 | 아님 | O |
| `scripts/panel_ml.py` | 4 | 아님 | O |
| `scripts/add_forward_returns.py` | 5 | 아님 | O |
| `src/screener/sector_rotation.py` | 6 | 아님 | O |
| `src/screener/backtest.py`, `config.py` (+ `features.py` 필요 시) | 6 | **예** | O (한 커밋) |
| `src/paper_trading/backtest.py`, `account_backtest.py`, `candidate_selector.py`, `config_override.py` | 6 | 아님 | O |
| 🛑 `src/main.py`, `src/run_full_scan.py` | 6 | 아님 | O (따로) |
| `scripts/run_hypothesis_ab.py` (`VARIANTS`) | 6 | 아님 | O |
| `scripts/optimize_optuna.py` (`--wfo`, `robust_center`, DSR) | 7 | 아님 | O |
| `docs/quant_improvement/tier3/*` (보고서, TRIAL_LOG) | 전부 | — | O |
| `data/research/*` (signals, events, etf, earnings, ml_pred) | 전부 | — | X |
