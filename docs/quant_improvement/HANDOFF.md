# HANDOFF — 이 문서 하나로 이어서 진행

> 갱신: 2026-10-09 (연구 단계 마무리) · 리포 `sean-ccho/Project_1` · 이전 문서(zip 이전 가이드, Tier 3 인수인계)를 합친 **단일 인수인계 문서**.
> 새 컴퓨터의 Claude는 **이 문서만 읽고 시작한다.** 배경이 더 필요하면 9절의 문서를 본다.

---

## 0. 30초 요약 (2026-10-09 마무리)

- **연구 결론: 지금 전략(스크리너 + CCS)으로 SPY를 이긴다는 근거가 없다. 종목 선택·구조 연구는 끝났다** (누적 시도 932건, `tier3/TRIAL_LOG.csv`).
  - 3년 백테스트(PIT, 버그 수정 후): Sharpe 1.14 vs SPY 1.08 — 운과 구분 안 됨
  - 10년(2016-01 ~ 2026-10): CCS Sharpe 0.70 · CAGR 14.2% · MDD −43.6% vs SPY 0.89 · 15.2% · −33.7%. CCS 1등 대신 무작위로 골라도 비슷하다 (`tier3/CCS_PLACEBO_10y.md`)
  - 홀드아웃 1회(2025-10-01 ~ 2026-10-07, **이미 사용함**): 모멘텀만 Sharpe 0.91 vs SPY 1.32 → 불통과
  - 구조 워크포워드(종목 수·손절·약세장·비중 등 8개): 표본 외 Sharpe 0.05 vs 현재 0.58 vs SPY 0.85 → 불통과 (`tier3/WFO_STRUCTURE_10y.md`)
  - 실전 PT-1을 $5,000로 환산하면 2026-04-02 이후 +14.6% vs 같은 날 SPY +20.9% (MDD −13.4% vs −4.5%)
- **실전 계좌 (2026-10-09 이후)**: PT-1(기존) · PT-1S(PT-1 규칙, 후보 S&P 500만) · PT-SPY(SPY $5,000 보유, 시트만). PT-2·PT-3은 삭제. PT-1·PT-1S는 $5,000 계좌로도 보여 준다 (메일 섹션 · `페이퍼_계좌`·`페이퍼S_계좌` 탭)
- **PT-1 매매 규칙·수치는 한 달 동안 바꾸지 않았다.** 바뀐 것은 버그 수정·안전장치·표시뿐 (0-1절)
- 다음: 실전 세 계좌를 3~6개월 나란히 지켜본다. 남은 확인·결정은 5절

## 0-1. 실전 페이퍼 트레이딩 — 바뀐 것 총정리 (2026-09-28 ~ 10-09)

git 기록은 2026-09-28부터 있다 (그날 히스토리 정리). 상태: ✅ push 됨(실전 반영) · ⏳ 로컬 커밋(push 전) · 📝 커밋 전(2026-10-09 마무리 시점)

**계좌**

| 날짜 | 변경 | 상태 |
|---|---|---|
| 10-01 | PT-2 골든크로스 스윙 · PT-3 일봉 단타 추가 ($5,000씩, 계좌별 메일·시트 탭) | 10-09 삭제 |
| 10-07 | PT-1S 추가 — PT-1 규칙 그대로, 후보만 S&P 500 (메일 없음, `페이퍼S_*` 탭) | ✅ |
| 10-09 | PT-SPY 추가 — SPY를 $5,000로 사서 계속 보유 (`페이퍼SPY_*` 탭만) · PT-2·PT-3 실행 중단 | ✅ |
| 10-09 | PT-2·PT-3 완전 삭제 (코드·설정·상태 파일·테스트). 시트 탭 6개(`페이퍼2_*`·`페이퍼3_*`)는 사용자가 직접 삭제 | 📝 |
| 10-09 | PT-1 $5,000 계좌 — PT-1 실제 거래를 가상 자본으로 다시 계산 (빈 자리마다 현금 균등 배분, 편도 0.1%). 메일 섹션 + `페이퍼_계좌`·`페이퍼S_계좌` 탭, 매매와 무관 (`pt1_account.py`) | 📝 |

**PT-1 버그 수정·안전장치 (매매 규칙·수치는 그대로)**

| 날짜 | 문제 → 수정 | 상태 |
|---|---|---|
| 10-01 | 주말·휴일·push 때도 매매하고 날짜를 UTC로 써서 묵은 데이터로 체결 (49건 중 6건) → 거래일 = 스냅샷 일봉 날짜, 같은 일봉은 한 번만 (F) · 최신 일봉 스냅샷만 사용 (K) · 장중 미완성 일봉이면 건너뜀 (L) · 헬스체크 날짜 (J) | ✅ |
| 10-01 | 실험 스위치 추가 (CCS v2, H1~H6 등) — 전부 꺼짐, 실전 영향 없음. 이후 R2·구조 실험 스위치도 같은 방식 | ✅ |
| 10-07 | ETN·채권·폐쇄형펀드·우선주·SPAC 이 후보에 섞임 (VXX 매수) → 제외 (10-08 보통주 Preferred Bank 오인 수정) | ✅ |
| 10-07 | 같은 회사 다른 클래스주 중복 매수 (GOOG+GOOGL) → 방지 | ✅ |
| 10-07 | 섹터 이름 불일치(yfinance vs GICS)로 6개 섹터의 매수 신호가 꺼져 있음 → 이름 매핑 | ✅ |
| 10-08 | 어닝 회피 필터가 yfinance 1.x 키 변경으로 꺼져 있음 → 복구 (실적일 epoch 초 → 미국 동부 날짜) | ✅ |
| 10-08 | 바닥반등 MACD 가점 기준이 달러 단위(−0.5달러) → 주가 대비 −1% (CCS 최대 0.02 차이) | ✅ |
| 10-09 | Yahoo 요청 제한으로 NASDAQ/NYSE 종목 정보 885건 전량 실패 (섹터 Unknown·실적일·재무 없음) → 대량 실패면 쉬었다 다시 받기, 끝내 실패하면 전날 스냅샷 값 (`fundamentals_guard.py`) | ⏳ |
| 10-09 | S&P 500 목록이 낡음 (현재 구성종목 29개 빠짐, 없어진 25개 남음) → 위키백과 기준 갱신 | ⏳ |
| 10-09 | S&P 500 목록 매달 1일 자동 갱신 + 바뀐 종목 메일 (20개 넘게 바뀌면 반영 보류·확인 메일, `update-sp500.yml`) | 📝 |

**메일·시트·자동 실행**

| 날짜 | 변경 | 상태 |
|---|---|---|
| 09-30 ~ 10-01 | 차트 스크린샷을 orphan 브랜치로 (저장소 용량) · 메일 수신자는 본인만 | ✅ |
| 10-04 | 새 일봉이 없는 날(주말·휴장)에도 일일 리포트 메일 (매매 없음) | ✅ |
| 10-07 | 시트 거래로그·성과요약에 같은 기간 SPY 수익률·SPY대비 열, PT-1 메일에 "SPY 대비 누적 성과" 표 | ✅ |
| 10-07 | 스크리너 패턴 "차트반전" 추가 (표시만, 점수에 안 들어감) | ✅ |
| 10-08 | 봇 상태 push 가 실행 중 들어온 push 에 거부됨 → pull --rebase 후 3회 재시도 | ✅ |

---

## 1. 새 컴퓨터에서 시작하기

```bash
git clone https://github.com/sean-ccho/Project_1.git && cd Project_1   # 이미 있으면 git pull --ff-only
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
PYTHONPATH=.:src pytest tests -q
```

- 기대 결과: **223 통과, 2 실패** (2026-10-09 마무리 기준). 실패 2건(`tests/test_portfolio_report.py::test_rsi_cell_colors`, `test_vol_cell`)은 리포트 코드가 "N/A" 대신 "—"를 출력해서 생기는 **기존 문제**이고 이번 연구와 무관하다. 고치지 않아도 된다.
- 데이터 파일(parquet)은 git에 없다 → **7절**에서 가져온다.
- 모든 명령은 저장소 루트에서, 앞에 `PYTHONPATH=.:src`를 붙여 실행한다.

데이터를 풀고 나면 아래로 정상 여부를 확인한다 (몇 초).

```bash
PYTHONPATH=.:src python scripts/factor_research.py --set pit    # 팩터 IC 리포트가 다시 나오면 정상
```

---

## 2. Claude에게 — 진행 규칙

- 커밋 메시지는 `type: 한국어 설명` ([CLAUDE.md](../../CLAUDE.md)). **끝에 `[skip ci]`** 를 붙인다. push만으로 실거래 페이퍼 매매·이메일·시트가 돌 수 있다.
- 🛑 = 사용자 확인 후 진행: push, main 머지, 사용자 파일 삭제, 기본 config 값 변경, 되돌리기 어려운 작업.
- **커밋도 허락 후에** — 변경은 커밋하지 않은 채 Source Control 에 먼저 보여 주고 허락받는다 ([CLAUDE.md](../../CLAUDE.md) 2절). push 는 따로 묻는다 (= 실전 반영).
- `data/paper_trading/`의 실거래 상태(positions·trades·state)는 수정·커밋하지 않는다. 봇만 커밋한다.
- **홀드아웃(2025-10-01 ~ 2026-10-07)은 2026-10-08에 한 번 썼다.** 다시 규칙 조정·재시험에 쓰지 않는다. 새 아이디어는 그 뒤 실전(페이퍼) 기록으로 검증한다.
- 채택 기준: 판정이 `채택 후보`(ΔSharpe 95% 구간이 0 초과)인 것만. `운과 구분 안 됨`은 Sharpe가 올라도 채택하지 않는다. 한 번에 하나씩 바꾼다.
- 팩터·합성 검정은 **시도 횟수를 기록**하고 다중검정 보정(Bonferroni)을 적용한다. 결과를 본 뒤 고른 팩터의 전체기간 성과는 낙관적이므로 검증 구간·워크포워드를 같이 본다.
- 백테스트 결과물(`output/runs/` 등)은 커밋하지 않는다. 단 **`output/hypothesis_ab.csv`(A/B 실험 기록표)는 커밋한다** — 시도 횟수 기록이라 다중검정 보정에 필요하다. 연구 결과 요약은 `docs/quant_improvement/tier3/`에 둔다.
- 작업이 끝나면 [WORK_SUMMARY.md](WORK_SUMMARY.md) **9절에 한 줄** 남기고, 이 문서의 0절·5절을 갱신한다.
- 문서와 실제 상태가 다르면 추측하지 말고 묻는다.

---

## 3. 지금까지 한 일

| 단계 | 결과 |
|---|---|
| Tier 0 백테스트 정합성 (`verify_backtest_integrity.py`) | 결정적 검사 전부 통과 |
| PIT 유니버스 (`fetch_sp500_membership.py`) | S&P 500 과거 구성종목 CSV 확보 (`data/universe/sp500_membership.csv`) |
| Baseline (`run_hypothesis_ab.py --only BASE`) | PIT: Sharpe 0.35 / 알파 −4.3% (t=−0.45). PIT 없음: Sharpe 0.81 / 알파 +4.0% → **생존 편향 확인** |
| 새 BASE (2026-10-08, 섹터 버그 수정 후, PIT) | Sharpe 0.83 / CAGR 18.1% / MDD −27.9% / 알파 +4.8% (t=0.44). SPY Sharpe 1.08. R1·H3 채택 없음 |
| 새 BASE (2026-10-08 오후, 어닝 필터·MACD 단위 수정 후) | Sharpe 1.14 (실적일 예정 발표만, 기본) · 1.02 (8-K 전부) / CAGR 25.4% / MDD −21.9% / 알파 t 1.07. SPY 1.08 — 운과 구분 안 됨. R2·H4b·H5·H2 채택 없음 |
| Tier 2 종목 수 (T2-5·10·20) | T2-10 Sharpe 1.09, MDD −11%. 그러나 "운과 구분 안 됨" → **채택 안 함** |
| H1~H6 미세조정 | **건너뜀.** [WORK_SUMMARY.md](WORK_SUMMARY.md) 4-8절 규칙: 알파 t<1이면 미세조정보다 알파 재설계가 먼저 |
| Tier 3-1 연구 패널 | 날짜×종목 피처 + 5/10/20일 선행수익률 (t+1 시가 진입 → t+1+h 시가 청산) |
| Tier 3-2 팩터 IC | 44팩터×3기간=132건. 기준(|t|≥2 & 연도 부호 일관≥67%) 통과 **1건** |
| Tier 3-3 합성 점수 | 시도 150건, 보정 임계 |t|≈3.59. 통과 없음. **Baseline v4 없음** |
| px10y 10년 패널 (`build_price_panel.py`) | 2016-01~2025-09, 571종목, 가격 피처 35개. 팩터 105건 + 합성 18건 재검증 → 통과 없음 (4절 하단) |
| Tier 3-B 신호 연구 (2026-10-07~08) | 스크리너 출력 117 · 조건부 반전 18 · PEAD(대리·실제) 20 · 캔들 확인 18 · SEC 재무 14 · 패널 ML 1 → 후보 0 |
| CCS 플라시보 3년 (2026-10-08) | 실제 CCS = 문턱 통과 후보 중 무작위 선택 분포의 75백분위 → 운과 구분 안 됨 (`tier3/CCS_PLACEBO.md`) |
| 홀드아웃 1회 (2026-10-08, 사전 등록) | 모멘텀만(H2) Sharpe 0.91 vs SPY 1.32 → 불통과. BASE 0.76 |
| CCS 플라시보 10년 (2026-10-09, 2016-01~2026-10) | CCS 0.70 · CAGR 14.2% · MDD −43.6% vs SPY 0.89 · 15.2% · −33.7%. 무작위 중앙 0.61, 40개 중 SPY 넘은 것 0 (`tier3/CCS_PLACEBO_10y.md`) |
| 구조 워크포워드 (2026-10-09) | 종목 수·약세장 처리·비중·손절·트레일링·보유일·바닥반등 on/off, 폴드당 Optuna 60회. 표본 외 2020~2026-10 Sharpe 0.05 vs 현재 0.58 vs SPY 0.85 → 불통과, DSR 0.01 (`tier3/WFO_STRUCTURE_10y.md`) |

실행 결과 원본: `output/hypothesis_ab.csv`(A/B 누적 기록, git에 커밋), `docs/quant_improvement/tier3/*`(팩터·합성 결과, git에 커밋).

---

## 4. Tier 3 결과 상세

패널: PIT 기준 36.1만 행 / 781일 / 493종목 (2022-08-18 ~ 2025-09-29), 그날 S&P 500 구성종목만. t-stat은 h일 간격 비중첩 샘플.

**팩터 개별 (PIT)**
- 평균 IC 절댓값이 대부분 0.02 미만. 추세·골든크로스·모멘텀 계열은 예측력이 없거나 오히려 역방향.
- 방향이 일관된 약한 신호(|t|<2라 채택 불가): `10일고점괴리`·`5일수익률`·`obv_z20` 음(−) = 단기 반전, `최근20일평균거래대금` 양(+) = 큰 종목 우위.
- 레짐별 결론은 못 낸다: 데이터가 거의 상승장이라 bear 표본이 없고 neutral도 20개뿐.

**합성 점수** (`scripts/composite_research.py`)

| 항목 | 결과 |
|---|---|
| 고정 합성 REV / LIQ / VOL 및 조합 (전체기간) | 최고 t≈2.2 (5일, REV+LIQ+VOL). 보정 임계 3.59 미달 |
| 반전(REV) | 학습구간(~2024-08) 5일 t=2.37 → 검증구간(2024-08~) **t=0.14로 소멸** |
| 워크포워드 (과거 데이터로만 팩터 선택, 정직한 OOS) | 5일 +0.89 / 10일 −0.71 / 20일 +1.03 → 무의미. 고른 팩터도 시기마다 완전히 달라짐 = 노이즈 |
| 효과 크기 | Q5 초과수익이 5일 +0.1% 안팎 (거래비용 전) |

**관찰 후보: LIQ(`최근20일평균거래대금`)** — 검증구간(2024-08~) 20일 IC +0.049, t=2.12, 적중률 66%. 하지만 학습구간(2022~2024-08)은 거의 0(t=0.37) → **2024~25 대형주 장세 한정 효과**일 가능성. 결과를 본 뒤 고른 팩터 + 1.1년 표본이라 채택 불가.
- ⚠️ 이 후보를 홀드아웃(2025-10-01~)으로 확인하지 말 것: 홀드아웃은 최종 1회용이다.
- px10y 재검증 결과(아래): `log_dollar_vol_20` 10일 t=+1.79 — 10년에서도 임계 미달. 관찰 후보에서 내린다.

**px10y 10년 패널** (`panel_px10y.parquet`, `FACTOR_REPORT_px10y.md`·`COMPOSITE_REPORT_px10y.md`)

- 108.3만 행 / 2450일 / 571종목 (2016-01-04 ~ 2025-09-30). 일봉 OHLCV에서 직접 계산한 가격·거래량 피처 35개(섹터중립 `_sn` 포함). 피처 캐시를 안 써서 수 분이면 다시 만든다.
- 생존 편향: 가격이 있는 PIT 구성종목 비율이 2016년 79% → 2025년 97% (`tier3/px10y_coverage.csv`). 초기 연도일수록 상장폐지 종목이 빠져 있다. `_sn` 피처의 섹터는 현재 구성종목 기준(`data/universe/sector_map.csv`).
- 팩터 105건(35×3): 최고 `vol_trend` 5일 t=−2.12, 게이트 통과 1건. 단기 반전(`ret_5d`·`ret_10d` 음)은 10년 중 9~10년 부호가 같지만 |t|≈1.1.
- 합성 18건: 최고 REV+LIQ+VOL 5일 Q5 초과 t=1.95. 워크포워드 OOS(2023-08~) IC t = 5일 −0.39 / 10일 +0.02 / 20일 −0.86 → 무의미.
- 2026-10-07 재실행(0-5, 매년 1월 재학습 · OOS 2019~ · split 2021-01, 시도 18건 추가): 최고 REV+LIQ 5일 전체 t_nw=2.57(검증 2021~ 1.62). 워크포워드 OOS t_nw = 5일 0.30 / 10일 1.16 / 20일 0.74 → 통과 없음. 리포트 파일은 이 재실행 결과로 바뀌었다.
- 참고: `COMPOSITE_REPORT_px10y.md`의 "1단계 132건"은 3년 패널 값(`PRIOR_TRIALS` 하드코딩)이다. px10y 실제 팩터 검정은 105건. 누적 시도 수는 0단계 TRIAL_LOG에서 다시 센다.

---

## 5. 다음 할 일 (2026-10-09 마무리 시점)

**연구는 끝났다.** 종목 선택(팩터·합성·스크리너 신호·반전·PEAD·재무·ML)과 구조(종목 수·손절·약세장·비중) 모두 SPY를 이기는 근거가 없었다. 같은 데이터로 규칙을 더 만지면 과적합이다 (CLAUDE.md 4절).

1. **실전 비교 관찰 (3~6개월)**: PT-1 vs PT-1S vs PT-SPY. PT-1 메일의 "SPY 대비 누적 성과"·"$5,000 계좌" 표와 시트 탭으로 본다.
   - 전체 유니버스(NASDAQ/NYSE 소형주 포함)가 값을 하는지는 PT-1 vs PT-1S로 판단한다 (각 30건 이상 쌓인 뒤). 상장폐지 종목까지 든 데이터가 생기기 전에는 전체 유니버스 백테스트를 하지 않는다 (yfinance 생존 편향).
2. **그 뒤 결정**: SPY(지수) 중심으로 바꿀지, PT-1을 계속 둘지. 새 전략 아이디어는 새 데이터가 있을 때만 시험한다.
3. **2026-10-09 마무리 때 남은 확인**
   - 사용자: Source Control 검토 → 커밋 허락 → push (매일 실행 ~02:30 ET 전에, 실행 중에는 push 금지) · 구글 시트 `페이퍼2_*`·`페이퍼3_*` 탭 6개 삭제
   - push 뒤 첫 실행 확인: 실적일이 채워졌는지(`days_to_next_earnings`), NASDAQ 섹터 Unknown 이 없는지, PT-SPY 첫 매수, `페이퍼_계좌`·`페이퍼S_계좌` 탭 생성
4. **알려진 작은 문제**: `test_portfolio_report.py` 2건 실패 (리포트가 "N/A" 대신 "—", 기존 문제) · 보유 중 액면분할이 있으면 백테스트·$5,000 계좌 평가가 틀릴 수 있음 · ETF를 후보에 넣을지 점검 (`docs/TO_DO.md`)

완료된 연구 순서 (참고, [SIGNAL_RESEARCH_PLAN.md](SIGNAL_RESEARCH_PLAN.md)): 0 준비 → 1 스크리너 출력 → 2 조건부 반전 → 3 PEAD·SEC 재무 → 4 패널 ML → 6 백테스트 정비·BASE 재측정 → CCS 플라시보(3년·10년) → 8 홀드아웃 1회 → 구조 워크포워드. 결과는 3·4절과 `tier3/`.

---

## 6. 파일 지도

**스크립트** (`scripts/`)

| 파일 | 용도 |
|---|---|
| `verify_backtest_integrity.py` | Tier 0: 캐시·재현성 검사 |
| `fetch_sp500_membership.py` | PIT 구성종목 CSV 다운로드 |
| `run_hypothesis_ab.py` | A/B 백테스트 (BASE, T2-*, H1~H6), 결과를 `output/hypothesis_ab.csv`에 누적 |
| `optimize_optuna.py`, `analyze_ccs_ic.py` | PT-1 Optuna (구간 나눠 평가), CCS IC |
| `build_research_panel.py` | **Tier 3-1** 패널 생성 → `data/research/panel_{pit,nonpit}.parquet` |
| `build_price_panel.py` | 10년 가격 패널 생성 (yfinance, 2025-09-30에서 다운로드 종료) → `data/research/panel_px10y.parquet`, `data/cache/ohlcv_px10y.parquet`, `data/universe/sector_map.csv`, `tier3/px10y_coverage.csv` |
| `factor_research.py` | **Tier 3-2** 팩터 IC·5분위·레짐 → `tier3/` (`--set pit|nonpit|px10y`) |
| `composite_research.py` | **Tier 3-3** 합성 점수 + 워크포워드 + 다중검정 → `tier3/` (`--set pit|nonpit|px10y`) |
| `research_utils.py` | Tier 3-B 공통: 개발 구간 로드(`load_panel`·`load_ohlcv`), NW t, Bonferroni, `TRIAL_LOG.csv` 기록 |
| `signal_portfolio_sim.py` | 신호 → 겹치는 h일 보유 포트폴리오, SPY·PIT 동일가중 대비 ΔSharpe → `tier3/PORTFOLIO_SIM_*.md` |
| `conditional_research.py` | **Tier 3-B 2단계** 5일 수익률 × 거래량 급증·갭 조건부 반전 (px10y) → `tier3/CONDITIONAL_*` |
| `pead_research.py` | **Tier 3-B 3단계** `--proxy`: 대리 실적 이벤트(갭+거래량) 후 h일 초과수익 → `tier3/PEAD_PROXY_*` · `--real`: SEC 실제 실적일 → `tier3/PEAD_REPORT_px10y.md` |
| `fetch_sec_data.py` → `build_sec_data.py` | SEC EDGAR 실적일(8-K 2.02)·연간 재무 수집·정리 (User-Agent 는 환경변수 `SEC_USER_AGENT`, `--holdout` 이면 홀드아웃까지 든 실적일 파일) |
| `fundamental_research.py`, `fetch_raw_prices.py` | SEC 재무 팩터 14건 (시가총액용 비조정 종가) → `tier3/FUNDAMENTAL_REPORT_px10y.md` |
| `earnings_filter_check.py` | 백테스트 어닝 필터의 기여와 미래 정보 몫 진단 → `tier3/EARNINGS_FILTER_CHECK.md` |
| `ccs_placebo.py` | CCS 1등 vs 문턱 통과 후보 중 무작위 선택 (`--period 13y --start 2016-01-04 --tag 10y` 로 10년) → `tier3/CCS_PLACEBO*.md` |
| `wfo_structure.py` | PT-1 구조 워크포워드 (`--register` → `--fold YYYY` → `--evaluate`) → `tier3/WFO_STRUCTURE_10y.md`, `output/wfo/` |
| `warm_feature_cache.py` | 긴 백테스트의 피처 캐시를 날짜 구간별로 나눠 병렬 예열 |
| `update_sp500_tickers.py` | S&P 500 목록을 위키백과로 갱신 (`--dry-run`). 매달 1일 GitHub Actions `update-sp500.yml` 이 `--notify` 로 실행 |
| `watch_backtest.sh` | 터미널에서 백테스트 진행 상황 보기 |
| `pattern_confirm_research.py` | 반전 캔들(강세잉걸핑·모닝스타·하락추세 도지) 확인 vs 미확인 (px10y) → `tier3/PATTERN_CONFIRM_*` |
| `panel_ml.py` | **Tier 3-B 4단계** LightGBM 워크포워드 (px10y), `--shuffle-check` 대조 → `tier3/ML_*`, `PORTFOLIO_SIM_ml_*` |
| `signal_event_study.py` | **Tier 3-B 1단계** `--build`(신호 재계산) → `--count`(family 등록) → `--analyze` → `tier3/EVENT_*` |

**결과 (git에 있음)**: `docs/quant_improvement/tier3/` — `FACTOR_REPORT_{pit,nonpit,px10y}.md`, `COMPOSITE_REPORT_{pit,px10y}.md`, `factor_ic_*.csv`, `factor_quintile_*.csv`, `factor_regime_*.csv`, `composite_{pit,px10y}.csv`, `px10y_coverage.csv`. **PIT·px10y 파일이 기준**, nonpit은 생존 편향 비교용. 2026-10-07 이후: `TRIAL_LOG.csv`(모든 시도 기록, 932건), `EVENT_STUDY_pit.md`, `CONDITIONAL_REPORT_px10y.md`, `PEAD_*`, `PATTERN_CONFIRM_*`, `ML_REPORT_px10y.md`, `FUNDAMENTAL_REPORT_px10y.md`, `BASELINE_COMPARE.md`, `EARNINGS_FILTER_CHECK.md`, `CCS_PLACEBO.md`·`CCS_PLACEBO_10y.md`, `WFO_STRUCTURE_10y.md`.

**문서**

| 문서 | 언제 보나 |
|---|---|
| 이 문서 | 항상 먼저 |
| [SIGNAL_RESEARCH_PLAN.md](SIGNAL_RESEARCH_PLAN.md) | Tier 3-B 구현 순서 · 코드 골격 · 판정 기준 (완료) |
| [WORK_SUMMARY.md](WORK_SUMMARY.md) | 배경(1절), 만든 것(2절), 연구 명령어(4-8절), 성공·중단 기준(5절), 알려진 제약(6절), 진행 기록(9절) |
| [QUANT_IMPROVEMENT_PLAN.md](QUANT_IMPROVEMENT_PLAN.md) | 설계 근거. Tier 3은 9-5절, 3계좌·CCS v2는 10절 |

---

## 7. 데이터 파일 (parquet) — git에 없음

`.gitignore`의 `data/cache/`, `data/research/`는 용량이 커서 git에서 뺐다. 새 컴퓨터에는 아래 중 **하나**로 준비한다.

### 방법 A (권장): 번들 파일 복사

옛 컴퓨터의 `data/research/project_1_research_data_2026-10-03.zip` (약 180MB) 하나를 AirDrop·외장 디스크로 옮긴다. 새 컴퓨터에서 저장소 루트에:

```bash
unzip -o ~/Downloads/project_1_research_data_2026-10-03.zip -d .
shasum -a 256 -c data/research/SHA256SUMS.txt   # 모두 OK
```

번들 내용 (풀면 이 위치에 들어간다):

| 파일 | 크기 | 용도 |
|---|---|---|
| `data/research/panel_pit.parquet` | 164MB | **연구 패널 (PIT).** `factor_research.py`·`composite_research.py`의 입력. 이것만 있으면 지금까지 결과를 재현·확장할 수 있다 |
| `data/cache/ohlcv_e7842ef0a144.parquet` | 27MB | PIT 유니버스 일봉 가격 (1255일 × 3625종목, SPY 포함). 새 피처를 가격에서 다시 계산하거나 선행수익률을 바꿀 때 쓴다. ⚠️ **2021-10-04 ~ 2026-10-02라 홀드아웃이 들어 있다** → 읽을 때 반드시 `.loc[:"2025-09-30"]`로 자른다 |
| `data/research/SHA256SUMS.txt` | | 체크섬 |

그리고 `data/universe/sp500_membership.csv`(5.3MB)는 **git으로** 온다 (`git pull` 하면 있음).

**px10y (2026-10-03 번들 이후 생성, 번들에 없음)** — 다른 컴퓨터에 보낼 때는 새 번들로 묶거나 `build_price_panel.py`로 다시 만든다 (수 분, yfinance 필요).

| 파일 | 크기 | 용도 |
|---|---|---|
| `data/research/panel_px10y.parquet` | 179MB | 10년 가격 패널 (2016-01-04 ~ 2025-09-30, 571종목). SPY 행 없음 |
| `data/cache/ohlcv_px10y.parquet` | 59MB | 10년 일봉 (2015-01-02 ~ 2025-09-30, 2702일). `build_price_panel.py`가 있으면 재사용. **SPY·섹터 ETF 없음** |
| `data/universe/sector_map.csv` | 15KB | 현재 S&P 500 GICS 섹터 (위키피디아 + config.SECTOR_MAP). 지금은 git에 없음 |

**SEC·10년 백테스트 (2026-10-08~09 생성, 번들·git에 없음)**

| 파일 | 크기 | 용도 |
|---|---|---|
| `data/research/sec_earnings_dates.parquet`, `sec_earnings_dates_holdout.parquet` | 작음 | 백테스트 어닝 필터용 실제 실적일 (`build_sec_data.py`, 뒤 파일은 홀드아웃까지 포함) |
| `data/cache/features/99214d6ca3fe_v1/` | 879MB | 10년(13y, PIT) 피처 캐시. 2026-10-09 config 변경(PT-2·3 설정 삭제, 차트 캡처 켬)으로 코드 해시가 바뀌어 옛 키 `81491a04da3e` 에서 이름만 옮김 (피처 계산과 무관한 설정이라 내용 같음) |
| `data/cache/ohlcv_f57d11b41e4f.parquet` | 79MB | 10년 PIT 일봉 (2013-10 ~ 2026-10). 받은 지 24시간이 지나면 다시 받는다 |

번들에 넣지 않은 것: `panel_nonpit.parquet`(생존 편향 비교용, 필요 없음), 피처 캐시 `data/cache/features/*`(패널에 이미 들어 있음, 합쳐 510MB), 나머지 ohlcv 캐시.

### 방법 B: 새 컴퓨터에서 처음부터 생성

yfinance 네트워크가 필요하고 **수 시간** 걸린다 (첫 실행은 전 종목 피처 계산). yfinance는 과거 데이터를 수정해서 줄 수 있어 **번들과 숫자가 조금 다를 수 있다.**

```bash
PYTHONPATH=.:src python scripts/fetch_sp500_membership.py     # 이미 git에 있으면 생략 가능
PYTHONPATH=.:src python scripts/run_hypothesis_ab.py --only BASE --period 5y --max-tickers 1000 --end 2025-09-30 --pit-universe
# → data/cache/ohlcv_*.parquet, data/cache/features/*_v1/ 생성 (해시는 새로 정해진다)
```

그 뒤 `scripts/build_research_panel.py`의 `SETS["pit"]`에 적힌 해시(`fe5a595b6df7` / `ohlcv_e7842ef0a144.parquet`)를 **새로 생긴 해시로 바꾸고** `--set pit`로 실행한다.

### 새 패널을 만들었을 때
`data/research/`의 parquet은 항상 재생성 가능한 파생물이다. **커밋하지 말고**, 다른 컴퓨터로 보낼 때만 방법 A처럼 zip으로 묶는다. 결과 요약(md·csv)만 `tier3/`에 커밋한다.

---

## 8. 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `ModuleNotFoundError` | import 경로 | 명령 앞에 `PYTHONPATH=.:src` |
| `FileNotFoundError: panel_pit.parquet` | 데이터 번들 미설치 | 7절 방법 A |
| `sp500_membership.csv 없음` | `--pit-universe`인데 CSV 없음 | `git pull` 또는 `scripts/fetch_sp500_membership.py` |
| `git pull`이 divergent로 거부 | 2026-09-30 히스토리 재작성(`filter-repo`) 이전 클론 | 로컬 전용 작업 백업 후 새로 clone |
| 표의 판정이 `데이터 부족` | 구간이 40거래일보다 짧음 | 기간을 늘린다 |
| 그 밖의 운영 문제 (실행 건너뜀, 헬스체크 등) | | [WORK_SUMMARY.md](WORK_SUMMARY.md) 7절 |

주의할 점
- IC 표본이 약 3.1년(781일)이고 연도별 부호 일관성은 표본 4개짜리라 거친 기준이다.
- 레짐은 패널 종목 동일가중 지수의 50/200일선으로 판정한다 (CCS의 레짐 판정과 다를 수 있음).
- 피처는 날짜 t 종가 기준, 수익률은 t+1 시가 진입이라 룩어헤드는 없다.
- PIT 유니버스는 부분 해결: 상장폐지·인수 회사는 yfinance에 가격이 없다 ([WORK_SUMMARY.md](WORK_SUMMARY.md) 6절).

---

## 9. 이 문서로 대체된 것 (2026-10-03 정리)

삭제한 문서는 git 히스토리에 남아 있다 (`git log --diff-filter=D --name-only -- docs/quant_improvement`로 찾는다).

| 삭제 | 이유 |
|---|---|
| 옛 `HANDOFF.md` (zip + AirDrop 이전 가이드) | 이전·머지 완료 (머지 커밋 `3b732a4`, 2026-10-01). 다시 쓸 일 없음 |
| `tier3/HANDOFF_TIER3.md` | 이 문서 3~5절에 합침 |
| `CODE_CHANGES_GUIDE.md` (Tier 0 코드 변경 1~8 수동 적용 가이드) | 코드에 반영·검증 완료 (`_code_hash`, `BACKTEST_COST_PER_SIDE`, `verify_backtest_integrity.py`) |
| `analyze_trades_stdlib.py` (pandas 없는 노트북용 일회성 분석) | 396건 거래 분석은 계획서 2-1절에 결과가 정리됨. 지금은 pandas가 있어 불필요 (같은 종류 분석은 `scripts/analyze_trades.py`). 위치도 `docs/`라 안내된 경로와 달랐음 |
