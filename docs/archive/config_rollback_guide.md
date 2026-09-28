# Config 롤백 가이드 — 신호 급증 원인 & 롤백 히스토리

> **사건 요약**: 2026-03-27에 신규 필터들이 대거 추가됐다가 당일 저녁 전부 비활성화됨.
> threshold도 낮아진 상태에서 필터까지 꺼지면서 바닥반등 결과 40-50개 → 100개로 폭증.
> **2026-03-28: `git checkout a799acfb`로 3개 파일 완전 롤백 완료.**

---

## 롤백 히스토리

| 커밋 | 날짜 | 내용 |
|------|------|------|
| `a799acfb` | 2026-03-26 23:18 | **현재 기준점** — 이 상태로 복원됨 |
| `986a74fb` | 2026-03-27 13:36 | 헤지펀드 수준 필터 대거 추가 (OUTPUT_TOP_N, CROSS_SECTIONAL 등) |
| `3c0093aa` | 2026-03-27 20:16 | 하드 패널티 추가 (부채/적자/ROE), 바닥반등 4조건 OR 필터 |
| `c5a96ad7` | 2026-03-27 21:33 | 하드 패널티 완화 (-3.0 → -1.5), threshold 완화 |
| `924d9b1b` | 2026-03-27 | 신규 필터 전부 비활성화 (OUTPUT_TOP_N=9999 등) → **100개 폭증** |
| 롤백 커밋 | 2026-03-28 | `git checkout a799acfb -- config.py signals.py run_full_scan.py` |

---

## 3/27에 추가됐다가 제거된 항목들

### config.py에서 사라진 설정
| 항목 | 3/27 추가값 | 현재 |
|------|------------|------|
| `OUTPUT_TOP_N` | 10 (→ 9999로 비활성화) | **설정 없음** |
| `OUTPUT_MAX_PER_SECTOR` | 3 (→ 9999) | **설정 없음** |
| `OUTPUT_MIN_SCORE` | 6.0 (→ 0.0) | **설정 없음** |
| `CROSS_SECTIONAL_ENABLED` | True (→ False) | **설정 없음** |
| `RISK_ADJUSTED_ENABLED` | True (→ False) | **설정 없음** |
| `CORRELATION_FILTER_ENABLED` | True (→ False) | **설정 없음** |
| `FUND_DANGER_DEBT_TO_EQUITY` | 400.0 | **설정 없음** |
| `FUND_DANGER_PROFIT_MARGIN` | -0.10 | **설정 없음** |
| `FUND_DANGER_ROE` | -0.15 | **설정 없음** |
| `BACKTEST_LOW_PROB_THRESHOLD` | 0.25 | → **0.40** |
| `BACKTEST_REVERSAL_SCORE_MIN` | 2.0 | → **3.0** |
| `BOTTOM_REVERSAL_THRESHOLD` | 7.0 | → **5.0** |
| `ALPHA_FACTOR_STRONG_THRESHOLD` | 0.5 | → **0.3** |
| `VOLUME_BREAKOUT_MULTIPLIER` | 1.8 | → **1.2** |
| `SECTOR_STRENGTH_LOOKBACK` | 60 | → **20** |
| `SECTOR_STRENGTH_THRESHOLD` | 0.02 | → **0.0** |

### signals.py에서 사라진 로직
- Conviction 스코어 (3개+ 팩터 동시 확인 시 +2.0)
- 펀더멘탈 건강 체크 (ROE/마진/성장 +1.0)
- 기관 보유 비율 보너스 (+0.5)
- 부채/적자/ROE 하드 패널티 (-1.5 each)
- Sharpe proxy / Alpha percentile 랭킹 파이프라인
- 상관관계 기반 다변화 필터

### run_full_scan.py에서 사라진 로직
- 바닥 반등 후보 4조건 OR (저점 근처, 최근 급락, 거래량 급증 조건들)
- 현재: 단순 `drop_from_high <= TURNAROUND_MIN_DROP` 1조건만

---

## 현재 기준 threshold (3/26 상태)

```python
BOTTOM_REVERSAL_THRESHOLD = 5.0
VOLUME_BREAKOUT_MULTIPLIER = 1.2
SECTOR_STRENGTH_LOOKBACK = 20
SECTOR_STRENGTH_THRESHOLD = 0.0
ALPHA_FACTOR_STRONG_THRESHOLD = 0.3
ALPHA_FACTOR_WEAK_THRESHOLD = 0.1
BACKTEST_LOW_PROB_THRESHOLD = 0.4
BACKTEST_REVERSAL_SCORE_MIN = 3.0
```
