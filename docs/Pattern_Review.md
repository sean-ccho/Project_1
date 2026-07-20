# Pattern Review

> 차트 패턴 감지 로직을 검토하고 나중에 다시 볼 때 참고하는 문서.
> 패턴별로 섹션을 추가한다.
> 최초 작성: 2026-05-18

---

## 골든크로스 (Golden Cross)

### 1. 완료된 작업 (커밋 완료)

**버그:** DDOG 등에서 주봉/월봉 패턴에 "골든크로스임박"(또는 "골든크로스")이 "정배열(완전)"과 **동시에** 표시되는 모순. 이미 골든크로스를 통과한 상승추세 종목을 신규 교차 직전("임박")으로 오판.

**수정** (`src/screener/patterns.py`):
- **임박 판정 정밀화** — `detect_golden_cross()`: 단기 MA가 장기 MA 아래에 일정 기간(`confirm_window` = 단기 기간) 이상 머문 경우에만 "임박" 인정. 일시적 눌림목 오탐 차단.
- **정배열 시 태그 억제** — `detect_weekly_patterns()` / `detect_monthly_patterns()`: 정배열(완전·눌림목) 감지 시 골든크로스 계열 태그 제거 (정배열형성중은 유지).

**검증:** 141종목 스캔에서 "골든크로스 + 정배열" 모순 8건 → 0건. 점수 영향: 모멘텀_적합도 변화 없음(패턴점수 상한 3.0에 가려짐), 바닥반등_적합도 −0.5~1.25 (거짓 가점 제거).

### 2. 미해결 논의 — 트레이딩뷰 차트와 매칭

**현재 골든크로스 감지 MA (SMA 기준):**

| 타임프레임 | 단기 | 장기 |
|---|---|---|
| 일봉 | SMA 50 | SMA 200 |
| 주봉 | SMA 10주 | SMA 40주 |
| 월봉 | SMA 3개월 | SMA 10개월 |

**차트 (트레이딩뷰 "SeanEMA" Pine script):** EMA 6개 — EMA 5 / 10 / 20 / 50 / 100 / 200.

→ 미스매치 2가지: ① 스크리너 SMA vs 차트 EMA, ② 기간이 다름(스크리너는 캘린더 구간 맞추려 축소).

**핵심 결론 (논의에서 도출):**
- **클래식 50/200으로 매칭하면 안 된다.** 주봉/월봉에서 EMA50/200 (또는 20/50)은 **너무 느려** 실제 전환 신호를 놓친다.
  - **HUM 예시 (월봉):** SMA3/10은 +29% 급반등을 "임박"으로 잘 포착. 반면 EMA20/50은 갭 −17%로 무신호, EMA50/200도 무신호.
- **현재 SMA3/10 같은 빠른 신호 자체는 유효하다** — "교차 전 매수" 목적의 빠른 전환 감지기로 나쁘지 않음.
- 진짜 단점은 신호 품질이 아니라 **SMA 선이 차트에 없어 눈으로 검증이 불가능**한 점.

### 3. 제안 (revisit 시 검토)

EMA로 전환한다면 — **차트에 존재하는 빠른 EMA를 사용. 느린 50/200은 지양:**

| 타임프레임 | 제안 EMA 쌍 | 비고 |
|---|---|---|
| 일봉 | EMA 50 × 200 | 기간 동일, SMA→EMA만. 차트와 일치 |
| 주봉 | EMA 10 × 50 | 차트 기간. 현행 10/40과 속도 유사 |
| 월봉 | EMA 5 × 10 | 차트 기간. 현행 3/10과 속도 유사 |

- 월봉 EMA200은 불가 — 200개월(≈17년) 데이터 필요.
- 변경 범위: `detect_golden_cross()`의 `rolling().mean()` → `ewm()`, 호출부 3곳(일/주/월)의 기간 인자.
- 참고: 정배열 감지는 이미 EMA 사용 중(주봉 EMA10/20/50, 월봉 EMA3/6/12) → 골든크로스도 EMA로 바꾸면 일관성↑.

### 4. 결정 완료 (2026-05-24, 3차에 걸친 반복)

**최종 상태:**

| 타임프레임 | MA | 갭 임계값 | slope 게이트 | close > EMA 필터 |
|---|---|---|---|---|
| **일봉** | **EMA 20 / 50** | 5% | **OFF** | **close > EMA10** |
| 주봉 | SMA 10 / 40 | 5% | ON | OFF |
| 월봉 | SMA 3 / 10 | 5% | ON | OFF |

**배경:** 2026-05-23 스캔에서 일봉 골든크로스/임박이 **355종목 중 0개** 검출. SMA 50/200이 본질적으로 느린데다 일봉만 빠른 쌍 정책에서 벗어나 있었음. 사용자 진입 전략의 핵심이 "골든크로스 임박" 단계라서 검출 부재가 곧 전략 마비를 의미.

**3차에 걸친 변경 경위:**

1. **1차 (EMA 전환)**: 일봉 SMA 50/200 → EMA 20/50.
   - 이유: 차트(TradingView SeanEMA) 매칭 + 빠른 쌍 정책 일관성. 클래식 EMA 50/200은 SMA만큼 느려 배제 (HUM 사례 §2).
   - 결과: 일봉 GCI 0→7개.

2. **2차 (갭 5%)**: 임박 갭 임계값 2% → 5%, 전 타임프레임 적용.
   - 이유: LMT 등 EMA 20/50 갭 4-5% 종목이 2% 안에 들지 않으면서도 명백한 추세 전환 케이스.
   - 결과: 일봉 GCI 7→14개. 주/월봉도 5% 적용 (사용자 결정: 보수적 2% 복귀 옵션 있었으나 5% 유지 선택).

3. **3차 (slope 게이트 OFF, 일봉만)**: LMT처럼 갭 좁혀지나 EMA20 평균값으로는 아직 하락 중인 추세 전환 초입 케이스를 잡기 위해, `short_slope > 0` 안전장치를 일봉 EMA 호출에서만 끔.
   - 이유: EMA는 응답이 빨라 10일 slope lookback이 과보수적. 갭 좁혀짐 조건이 이미 접근을 수학적으로 보증하므로 slope 게이트는 사실상 중복.
   - LMT 5일 슬로프=-2.01, 3일=-1.19로 어느 윈도우 단축으로도 LMT를 노이즈 없이 잡을 수 없어 조건 제거 선택.
   - 결과: 일봉 GCI 14→25개, DCI 10→17개.

4. **4차 (close > EMA10 필터, 일봉만)**: 3차 결과 25개 GCI 중 SCHW/ADBE/AEHL 같이 "갭은 좁혀지나 가격 자체는 단기 EMA 아래에서 횡보/하락 중"인 종목들이 잡힘. close가 단기 EMA(EMA10) 위에 있어야 임박으로 인정하도록 추가 필터 도입.
   - 이유: 실질적 반등 시작 확인. 가격이 EMA10도 못 차지하면 추세 전환 신호로 보기 어려움. AEHL 같은 펌프 후 폭락 종목 자동 차단.
   - 결과: 일봉 GCI 25→19개, DCI 17→11개. LMT 유지 (conf 0.71), SCHW/ADBE/AEHL/DUOL/MA/SE/RTX(아니다 RTX는 EMA10 위라 유지)/TMUS(EMA10 위 유지) 제거.

5. **5차 (이메일 정렬)**: 임박 종목이 27개(일봉 19 + 주봉 + 월봉 합집합)로 늘어나 정렬 위계 정비.
   - `_extract_golden_cross_imminent()` 정렬 우선순위: 다중 TF confirm(★) → 일봉 포함 → conf 내림차순.
   - 표/차트 모두 전부 표시 (사용자 결정 — 분량보다 정보 우선).
   - 다중 TF confirm 행은 ★ 표시 + 행 배경 강조로 시각 위계.

**구현:**
- [src/screener/patterns.py:976](../src/screener/patterns.py#L976) `detect_golden_cross()`에 `ma_type` + `require_short_slope_confirm` + `imminent_close_filter_ema` 3개 인자 추가. 기본값 모두 SMA 시대 값(`"sma"`, `True`, `0`)이라 주봉/월봉 호출부 무변경.
- [src/screener/patterns.py:1085-1110](../src/screener/patterns.py#L1085-L1110) 임박 판정 블록: slope 체크와 close 필터 둘 다 옵션화.
- [src/screener/patterns.py:1170](../src/screener/patterns.py#L1170) 일봉 호출부만 `ma_type="ema", require_short_slope_confirm=False, imminent_close_filter_ema=10` 전달.
- [src/screener/features.py](../src/screener/features.py) FeatureSet에 `daily_golden_cross_conf` / `weekly_golden_cross_conf` / `monthly_golden_cross_conf` 필드 추가, merged_df에 `일봉/주봉/월봉_골든크로스_신뢰도` 컬럼 노출 (정렬용).
- [src/paper_trading/runner.py:138](../src/paper_trading/runner.py#L138) `_extract_golden_cross_imminent()`에 다중 TF 우선 정렬 + `tf_count`/`has_daily`/`max_conf` 필드 추가.
- [src/screener/exporter.py:1287](../src/screener/exporter.py#L1287) 골든크로스 임박 섹션: ★ 표시(다중 TF), 행 배경 강조.
- [tests/test_patterns.py](../tests/test_patterns.py)에 LMT 회귀 가드 + close 필터 차단 가드 2개 추가.

**검증 (실제 데이터 335종목 풀, 2026-05-24):**

| 일봉 신호 | SMA 50/200 (2%) | EMA 20/50 (2%) | EMA 20/50 (5%) | EMA 20/50 (5%, slope OFF) |
|---|---|---|---|---|
| golden_cross | 5 | 6 | 6 | 6 |
| **golden_cross_imminent** | 5 | 7 | 14 | **25** |
| death_cross | (미측정) | 8 | 8 | 8 |
| death_cross_imminent | (미측정) | 7 | 10 | 17 |

- **LMT 검출**: conf=0.71 (`golden_cross_imminent`). 사용자 다중 타임프레임 차트와 일치.
- **임박 라인업 (conf 내림차순)**: BRK-B 0.94, EOSE 0.93, ABBV 0.92, JNJ 0.91, DUOL/GE/ACHR/MA 0.90, HON 0.89, ADBE/SCHW 0.87, GILD 0.86, AMGN 0.85, MNDY/TMUS 0.84, SE/IT 0.82, IBM 0.80, GPK 0.79, RTX/NOW/FIG 0.77, SYK/AEHL 0.76, **LMT 0.71**.
- **데드크로스 임박**: HIMS 0.92, BA 0.92, COP 0.91 등 — 정배열 통과 후 추세 약화 종목.
- confidence가 자동으로 위계 정렬: 갭 작을수록 0.95에 가깝게, LMT처럼 갭 -4.82%는 0.71. 페이퍼 트레이딩에서 conf 기준 정렬하면 자연스럽게 임박도 순으로 표시.

### 5. 남은 항목 (후속)

- 운영 1-2주 관찰 후: 25개 GCI 중 실제 골든크로스로 진행한 비율, 데드캣 바운스 비율 점검. 노이즈 과다하면 `confirm_window` 확대 또는 slope 게이트 부분 복원 검토.
- 주봉/월봉 EMA 전환은 계속 보류 — 현 SMA + 5% 검출이 의미 있으면 유지, 노이즈 발견 시 2% 복귀 또는 EMA 전환 검토.

### 6. 후속 작업 메모

- **2026-05-24**: 일봉 SMA 50/200 → EMA 20/50 전환. §4 참조.
- **2026-05-22 (commit 01c508340)**: 골든크로스 컬럼이 페이퍼 트레이딩 이메일의 4개 테이블(매도/보유/후보/골든크로스)에도 확장 노출 — 정배열 모순 수정의 효과가 PT 알림에서도 적용된다.
- **2026-05-19 (commit f33bc7b5c)**: 페이퍼 트레이딩 이메일에 골든크로스임박 전용 섹션 추가.
