# 코드 변경 가이드 (GitHub 웹에서 수동 적용)

> 대상 리포: `sean-ccho/Project_1`
> 목적: Tier 0(백테스트 재현성) + Tier 1(측정 정합성) 코드 변경
> 배경과 이유는 `QUANT_IMPROVEMENT_PLAN.md` 참고
> ⚠️ 이 코드는 **실행 테스트를 하지 않은 상태**다. 변경 후 반드시 [변경 7](#변경-7--검증-스크립트-추가-새-파일)의 검증 스크립트를 먼저 돌린다.

---

## 시작 전에 꼭 읽기

### 1) main이 아니라 브랜치에서 작업한다
README에 따르면 **main에 push하면 GitHub Actions가 자동 실행**된다(스크리너 실행, 이메일 발송). 웹 편집으로 main에 바로 커밋하면 변경할 때마다 워크플로우가 돈다.

**브랜치 만들기 (웹)**
1. https://github.com/sean-ccho/Project_1 접속
2. 왼쪽 위 `main` 브랜치 버튼 클릭
3. `tier0-integrity` 입력 → **Create branch tier0-integrity from main** 클릭
4. 이후 모든 편집은 이 브랜치가 선택된 상태에서 한다

### 2) 커밋 메시지 끝에 `[skip ci]`를 붙인다
브랜치에서 작업하더라도 안전하게 모든 커밋 메시지 끝에 `[skip ci]`를 붙인다.
예: `fix: cache key includes code hash [skip ci]`

### 3) 웹 편집 방법
1. 브랜치가 `tier0-integrity`인지 확인
2. 파일을 열고 오른쪽 위 ✏️ (Edit this file) 클릭
3. 편집기 안에서 `Cmd + F`로 **찾을 텍스트** 검색
4. 안내대로 교체하거나 추가
5. **Commit changes...** → 메시지 입력 → *Commit directly to the `tier0-integrity` branch* 선택 → 커밋

### 4) 들여쓰기 주의
Python은 들여쓰기가 문법이다. 아래 코드 블록의 **앞 공백까지 그대로** 복사한다. 붙여넣은 뒤 주변 줄과 들여쓰기가 맞는지 눈으로 확인한다.

### 변경 목록

| # | 파일 | 내용 | 해결하는 문제 |
|---|------|------|---------------|
| 1 | `src/screener/config.py` | 설정값 2개 추가 | 준비 |
| 2 | `src/paper_trading/backtest.py` | import 추가 | 준비 |
| 3 | `src/paper_trading/backtest.py` | 헬퍼 함수 2개 추가 + 캐시 키 수정 | A (캐시 버그) |
| 4 | `src/paper_trading/backtest.py` | 펀더멘털 미래 정보 차단 | B, C (미래 정보, 어닝 필터) |
| 5 | `src/paper_trading/backtest.py` | 거래비용 반영 | Tier 1 |
| 6 | `src/paper_trading/backtest.py` | 자본곡선 지표 + `meta.json` 보강 | Tier 1, 재현성 |
| 7 | `scripts/verify_backtest_integrity.py` | 새 파일: 검증 스크립트 | A, E 검증 |
| 8 | `scripts/analyze_trade_features.py` | 새 파일: 피처 분석 스크립트 | Tier 3 |
| 9 | (검색만) | 저점확률 누락 확인 | D |

`backtest.py`는 변경 2 ~ 6을 **한 번 편집 세션에서 모두 적용하고 한 번에 커밋**해도 된다.

---

## 변경 1 — config.py에 설정값 추가

**파일**: https://github.com/sean-ccho/Project_1/blob/tier0-integrity/src/screener/config.py

**찾을 텍스트**
```python
PAPER_TRADING_DATA_DIR = "data/paper_trading"
```

**이 줄 바로 아래에 추가**
```python

# 백테스트 측정 정합성 (Tier 1)
BACKTEST_FUNDAMENTALS_PIT_SAFE = True  # yfinance info는 현재 값이라 과거 날짜에 쓰면 미래 정보가 섞임
BACKTEST_COST_PER_SIDE = 0.001         # 수수료+슬리피지 편도 10bp
```

**커밋 메시지**: `feat: add backtest PIT-safe and cost settings [skip ci]`

---

## 변경 2 — backtest.py import 추가

**파일**: https://github.com/sean-ccho/Project_1/blob/tier0-integrity/src/paper_trading/backtest.py

### 2-1. `json` import 추가

**찾을 텍스트** (파일 맨 위쪽, 1번만 나옴)
```python
import logging
```

**바꿀 내용**
```python
import json
import logging
```

### 2-2. config import에 새 설정 추가

파일 위쪽 `from screener.config import (` 블록 안에서 찾는다.

**찾을 텍스트**
```python
    TICKERS,
)
```

**바꿀 내용**
```python
    BACKTEST_COST_PER_SIDE,
    BACKTEST_FUNDAMENTALS_PIT_SAFE,
    TICKERS,
)
```

---

## 변경 3 — 헬퍼 함수 추가 + 캐시 키 수정 (문제 A)

### 3-1. 헬퍼 함수 2개 추가

`_git_info()` 함수의 마지막 줄을 찾는다.

**찾을 텍스트**
```python
        return {"sha": None, "branch": None, "dirty": None}
```

**이 줄 바로 아래에 추가** (빈 줄 2개로 시작)
```python


_HASHED_SOURCES = (
    "screener/features.py", "screener/alpha_model.py", "screener/patterns.py",
    "screener/fundamentals.py", "screener/processing.py", "screener/signals.py",
    "screener/backtest.py", "screener/config.py",
)
_SECRET_MARKERS = ("PASSWORD", "SECRET", "TOKEN", "CREDENTIAL", "EMAIL", "SPREADSHEET", "DRIVE", "GITHUB", "KEY")
_SNAPSHOT_SKIP = {"TICKERS", "SECTOR_MAP", "COMPANY_NAME_MAP"}


def _code_hash() -> str:
    # 파일 수정 시에만 바뀜. Optuna의 런타임 오버라이드는 캐시를 무효화하지 않음 (의도된 동작)
    src_root = Path(__file__).resolve().parents[1]
    h = hashlib.md5()
    for rel in _HASHED_SOURCES:
        p = src_root / rel
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:8]


def _config_snapshot() -> dict[str, Any]:
    import screener.config as cfg

    snap: dict[str, Any] = {}
    for k in dir(cfg):
        if not k.isupper() or k in _SNAPSHOT_SKIP or any(m in k for m in _SECRET_MARKERS):
            continue
        v = getattr(cfg, k)
        try:
            json.dumps(v)
        except TypeError:
            continue
        snap[k] = v
    return snap
```

> 🔒 `meta.json`은 공개 리포에 커밋된다. `_SECRET_MARKERS` 필터 덕분에 `EMAIL_PASSWORD`, 스프레드시트 ID, 이메일 주소 등은 저장되지 않는다. **이 필터는 지우지 않는다.**

### 3-2. 캐시 키 수정

**찾을 텍스트**
```python
    ohlcv_key = ",".join(sorted(tickers)) + "|" + period
```

**바꿀 내용**
```python
    ohlcv_key = ",".join(sorted(tickers)) + "|" + period + f"|fund={int(include_fundamentals)}|code={_code_hash()}"
```

**효과**: 코드나 `config.py`를 수정하거나 fundamentals 플래그를 바꾸면 캐시 키가 바뀐다. 옛 피처를 불러오는 일이 사라진다. 기존 캐시(`data/cache/`)는 더 이상 쓰이지 않으니 로컬에서 지워도 된다.

---

## 변경 4 — 펀더멘털 미래 정보 차단 (문제 B, C)

**찾을 텍스트** (fundamentals 미리 불러오는 블록 바로 다음 줄)
```python
    price_map = _prepare_price_map(raw)
```

**이 줄 바로 위에 추가**
```python
    if prefetched_fundamentals is not None and BACKTEST_FUNDAMENTALS_PIT_SAFE:
        keep = [c for c in ("티커", "fund_sector") if c in prefetched_fundamentals.columns]
        prefetched_fundamentals = prefetched_fundamentals[keep]

```

**효과**
- 백테스트에서 ROE · 부채 · 기관보유 · 공매도 · 시총 같은 **현재 시점 값**이 과거 판단에 쓰이지 않는다.
- `days_to_next_earnings`도 빠지므로 잘못된 어닝 필터가 "전 종목 통과"로 바뀐다.
- 섹터 정보(`fund_sector`)만 유지한다.
- 실거래(`main.py`)에는 영향이 없다.

---

## 변경 5 — 거래비용 반영 (Tier 1)

### 5-1. `_close_position()` 수익률 계산

**찾을 텍스트**
```python
            return_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
```

**바꿀 내용**
```python
            entry_eff = entry_price * (1 + BACKTEST_COST_PER_SIDE)
            exit_eff = exit_price * (1 - BACKTEST_COST_PER_SIDE)
            return_pct = (exit_eff - entry_eff) / entry_eff if entry_price else 0.0
```

### 5-2. `_close_position()` 매도대금·손익

**찾을 텍스트**
```python
            proceeds = pos.shares * exit_price if pos.shares > 0 else 0.0
            dollar_pnl = pos.shares * (exit_price - entry_price) if pos.shares > 0 else 0.0
```

**바꿀 내용**
```python
            proceeds = pos.shares * exit_eff if pos.shares > 0 else 0.0
            dollar_pnl = pos.shares * (exit_eff - entry_eff) if pos.shares > 0 else 0.0
```

### 5-3. 매수 수량 계산 (2곳)

**찾을 텍스트** — 신규 매수와 교체 매수에서 **2번 나온다**. 두 곳 모두 바꾼다.
```python
shares = allocation / entry_price
```

**각각 바꿀 내용** (앞 들여쓰기는 원래대로 두고, `=` 뒤만 교체)
```python
shares = allocation / (entry_price * (1 + BACKTEST_COST_PER_SIDE))
```

**효과**: 매수·매도마다 편도 0.1%(왕복 0.2%)가 반영된다. 거래 기록의 `entry_price`/`exit_price`는 실제 체결가 그대로 남고, 수익률·현금·손익에만 비용이 들어간다.

---

## 변경 6 — 자본곡선 지표 + meta.json 보강

### 6-1. `_calculate_metrics()`에 자본곡선 지표 추가

`_calculate_metrics()` 함수의 **마지막 줄**을 찾는다.

**찾을 텍스트** (`_calculate_metrics` 함수 안의 것. 이 파일에서 1번만 나온다)
```python
    return summary
```

**이 줄 바로 위에 추가**
```python
    if equity_series is not None and len(equity_series) > 20 and initial_capital > 0:
        eq = equity_series.dropna()
        daily = eq.pct_change().dropna()
        if daily.std() > 0:
            summary["Sharpe_일간"] = round(float(daily.mean() / daily.std() * np.sqrt(252)), 2)
            down = daily[daily < 0].std()
            summary["Sortino_일간"] = round(float(daily.mean() / down * np.sqrt(252)), 2) if down > 0 else None
        years = len(eq) / 252
        if years > 0 and eq.iloc[0] > 0:
            cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
            summary["CAGR"] = round(cagr, 4)
            summary["Calmar"] = round(cagr / abs(mdd), 2) if mdd else None
    if spy_returns is not None and len(spy_returns) > 20 and spy_returns.std() > 0:
        summary["SPY_Sharpe_일간"] = round(float(spy_returns.mean() / spy_returns.std() * np.sqrt(252)), 2)

```

**효과**: 지금까지 본 Sharpe(0.56 ~ 0.77)는 거래 수익률로 계산한 값이다. 이제 **SPY와 같은 기준(일간 수익률)** 으로 비교할 수 있다.

### 6-2. `meta.json`에 재현성 정보 추가

**찾을 텍스트**
```python
            "run_label": run_ts,
```

**바꿀 내용**
```python
            "run_label": run_ts,
            "code_hash": _code_hash(),
            "config_hash": hashlib.md5(
                json.dumps(_config_snapshot(), sort_keys=True, ensure_ascii=False, default=str).encode()
            ).hexdigest()[:8],
            "equity_metrics": {
                k: summary.get(k)
                for k in ("CAGR", "Sharpe_일간", "Sortino_일간", "Calmar", "SPY_Sharpe_일간")
            },
            "cost_per_side": BACKTEST_COST_PER_SIDE,
            "fundamentals_pit_safe": BACKTEST_FUNDAMENTALS_PIT_SAFE,
            "config_full": _config_snapshot(),
```

**효과**: run마다 코드·설정의 지문(hash)과 전체 설정값이 남는다. 04-18_17-00 vs 17-03처럼 기록상 같은데 결과가 다른 일을 추적할 수 있다.

**커밋 메시지** (변경 2 ~ 6 한 번에): `fix: backtest cache key, PIT-safe fundamentals, costs, equity metrics [skip ci]`

---

## 변경 7 — 검증 스크립트 추가 (새 파일)

**만드는 방법**
1. 브랜치 `tier0-integrity`에서 `scripts` 폴더로 이동
2. 오른쪽 위 **Add file → Create new file**
3. 파일명: `verify_backtest_integrity.py`
4. 아래 내용을 붙여넣고 커밋 (`test: add backtest integrity check [skip ci]`)

```python
#!/usr/bin/env python3
"""백테스트 결정성 · 캐시 정합성 · 파라미터 반영 검증."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener import config as cfg
from paper_trading.backtest import run_paper_trading_backtest

KW = dict(period="3y", max_tickers=100, include_fundamentals=False, initial_capital=5000.0, rebalance_every=1)


def run(label: str, **extra):
    r = run_paper_trading_backtest(**KW, **extra)
    s = r["summary"]
    stops = sum(1 for t in r["trades"] if str(t["exit_reason"]).startswith("손절"))
    print(f"[{label}] 거래={s.get('총거래수')} 최종자본={s.get('최종자본')} 손절={stops}")
    return s.get("최종자본"), [(t["ticker"], t["entry_date"], t["exit_date"]) for t in r["trades"]]


def main() -> None:
    a = run("A 캐시없음", no_cache=True)
    b = run("B 캐시사용")
    assert a == b, "캐시 사용 시 결과가 달라짐 → 캐시 정합성 버그"

    orig = copy.deepcopy(cfg.EXIT_PARAMS)
    for p in cfg.EXIT_PARAMS.values():
        p["stop_loss"] = 0.04
    try:
        c = run("C 손절4%")
    finally:
        cfg.EXIT_PARAMS.clear()
        cfg.EXIT_PARAMS.update(orig)
    assert c != a, "손절을 바꿨는데 결과 동일 → 파라미터 미적용 (문제 E)"

    print("OK: 결정성 + 캐시 정합성 + 파라미터 반영 모두 통과")


if __name__ == "__main__":
    main()
```

**검증하는 것**
| 단계 | 기대 결과 | 실패하면 |
|------|-----------|----------|
| A 캐시없음 vs B 캐시사용 | 완전히 같아야 함 | 캐시 버그가 남아 있음 (A) |
| C 손절 4% | A와 달라야 함 | 청산 파라미터가 적용되지 않음 (E) |

- 100종목 × 3년이라 비교적 빨리 끝난다.
- 실행하면 `output/runs/`에 폴더가 3개 더 생긴다. 검증용이니 커밋하지 않아도 된다.

---

## 변경 8 — 피처 분석 스크립트 추가 (새 파일)

**만드는 방법**: 변경 7과 같은 방식으로 `scripts/analyze_trade_features.py` 생성
**커밋 메시지**: `feat: add trade feature IC analysis [skip ci]`

```python
#!/usr/bin/env python3
"""백테스트 거래 로그로 피처별 예측력(IC, 5분위 수익률 차이)을 분석한다.

사용법:
    python scripts/analyze_trade_features.py                       # 기본: 캐시 미사용 run 3개
    python scripts/analyze_trade_features.py 2026-04-26_17-03 ...  # run 지정
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["2026-04-22_00-07", "2026-04-24_02-08", "2026-04-26_17-03"]


def main() -> None:
    runs = sys.argv[1:] or DEFAULT_RUNS
    df = pd.concat(
        [pd.read_csv(ROOT / "output/runs" / r / "backtest_trades_enhanced.csv").assign(run=r) for r in runs],
        ignore_index=True,
    ).drop_duplicates(subset=["ticker", "entry_date"])
    df["strat"] = df["strategy"].astype(str).str.extract(r"(바닥반등|모멘텀|숏스퀴즈)")[0]
    feats = [c for c in df.columns if c.startswith("feat_") and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for strat, g in [("전체", df)] + list(df.groupby("strat")):
        for f in feats:
            x = g[[f, "return_pct"]].dropna()
            if len(x) < 30 or x[f].nunique() < 5:
                continue
            ic, p = spearmanr(x[f], x["return_pct"])
            q = pd.qcut(x[f].rank(method="first"), 5, labels=False)
            by_q = x.groupby(q)["return_pct"].mean()
            rows.append({
                "전략": strat, "피처": f[5:], "n": len(x), "IC": round(ic, 3),
                "p": round(p, 3), "Q5-Q1": round(by_q.iloc[-1] - by_q.iloc[0], 4),
            })

    res = pd.DataFrame(rows)
    res["absIC"] = res["IC"].abs()
    pd.set_option("display.width", 200)
    print(f"고유 거래 {len(df)}건 | |IC| > {2 / len(df) ** 0.5:.2f} 이면 대략 유의")
    for strat, g in res.sort_values("absIC", ascending=False).groupby("전략"):
        print(f"\n=== {strat} ===")
        print(g.drop(columns="absIC").head(15).to_string(index=False))


if __name__ == "__main__":
    main()
```

**결과 읽는 법**
- **IC**: 피처 값과 거래 수익률의 순위 상관. 양수면 "값이 클수록 수익이 좋다".
- **Q5-Q1**: 피처 상위 20% 거래의 평균수익 − 하위 20% 거래의 평균수익.
- **p < 0.05** 이고 `|IC|`가 첫 줄에 표시된 기준보다 크면 의미 있는 신호일 가능성이 높다.
- 확인할 것: 바닥반등이 약한 원인이 되는 피처, `ccs_*` 서브스코어가 실제로 수익과 관련 있는지.
- ⚠️ 이 CSV들은 패치 전 코드로 만든 것이라 문제 A · B의 영향이 섞여 있다. 패치 후 새 기준선 run으로 다시 돌려 비교한다.

---

## 변경 9 — 저점확률 계산 위치 찾기 (문제 D, 코드 수정 없음)

> ✅ 거래 로그 분석으로 **D는 이미 확정**됐다 (396건 전부 `feat_low_prob` 공백). 이 단계는 실거래에서 저점확률을 **어디서 계산하는지** 찾아, 백테스트에 같은 로직을 넣는 패치를 만들기 위한 것이다.

**웹에서 확인**
1. 리포 페이지에서 `/` 키를 눌러 검색창 열기
2. 아래를 입력하고 검색
   ```
   repo:sean-ccho/Project_1 저점확률 path:src
   ```
3. 결과에서 **값을 읽기만 하는 곳**(`row.get("저점확률"...)`)이 아니라 **값을 만드는 곳**(`out["저점확률"] = ...` 등)이 어디인지 확인

**또는 맥 터미널에서**
```bash
grep -rn "저점확률" src/ | grep -v 'get("저점확률"'
```

**판단**
- 값을 만드는 곳이 `main.py` 쪽에만 있고 `paper_trading/backtest.py` → `screener/backtest.py::_compute_ranked_snapshot` 경로에 없다면 → **D 확정**. 백테스트의 바닥반등은 실거래와 다른 전략이다.
- 결과를 공유해 주면 백테스트에 저점확률을 추가하는 패치를 만든다.

---

## 적용 후 맥에서 실행할 것

```bash
cd ~/Desktop/Code/Project_1
git fetch && git checkout tier0-integrity && git pull
source .venv/bin/activate

# 옛 캐시 정리 (새 캐시 키에서는 어차피 쓰이지 않음)
rm -rf data/cache/features data/cache/ic_weights

# 1) 검증
python scripts/verify_backtest_integrity.py

# 2) 기존 run 피처 분석
python scripts/analyze_trade_features.py

# 3) 진짜 기준선 (5년, 캐시 없이, 비용 + 미래정보 차단 적용)
python scripts/run_sp500_backtest.py --period 5y --rebalance 1 --fundamentals --capital 5000 --no-cache
```

> `rm -rf`는 `data/cache/` 아래 캐시 폴더만 지운다. 거래 기록(`data/paper_trading/`)과 run 결과(`output/runs/`)는 건드리지 않는다.

---

## 결과 공유 체크리스트

아래를 복사해서 보내 주면 해석하고 다음 단계(Tier 2 포지션 분산, Tier 4 Optuna)로 넘어간다.

- [ ] `verify_backtest_integrity.py` 출력 전체 (통과 여부와 A/B/C 줄)
- [ ] `analyze_trade_features.py` 출력 전체
- [ ] 저점확률 검색 결과 (변경 9)
- [ ] 새 기준선 run의 `output/runs/<날짜>/meta.json` 중 `benchmark`, `equity_metrics`, `performance` 부분

## 문제가 생기면

| 증상 | 확인할 것 |
|------|-----------|
| `NameError: BACKTEST_COST_PER_SIDE` | 변경 1(config)과 변경 2-2(import)가 둘 다 적용됐는지 |
| `NameError: json` | 변경 2-1이 적용됐는지 |
| `IndentationError` | 붙여넣은 블록의 앞 공백이 주변 줄과 맞는지 |
| `NameError: exit_eff` | 변경 5-1과 5-2가 둘 다 적용됐는지 |
| 검증 A ≠ B | 캐시 문제가 남아 있음. 출력을 그대로 공유 |
| 검증 C = A | 청산 파라미터 미적용 (문제 E). 출력을 그대로 공유 |
