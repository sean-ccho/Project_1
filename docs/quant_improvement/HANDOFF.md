# HANDOFF — 이 문서 하나로 이어서 진행

> 갱신: 2026-10-03 · 리포 `sean-ccho/Project_1` · 이전 문서(zip 이전 가이드, Tier 3 인수인계)를 합친 **단일 인수인계 문서**.
> 새 컴퓨터의 Claude는 **이 문서만 읽고 시작한다.** 배경이 더 필요하면 9절의 문서를 본다.

---

## 0. 30초 요약

- 3계좌 페이퍼 트레이딩(PT-1·2·3)은 **이미 main에 머지됐고 GitHub Actions로 매일 돈다.** 이전 작업은 끝났다.
- 지금 하는 일은 **백테스트 연구(Tier 3 알파 검증)** 이다.
- **결론(2026-10-05): 현재 전략에는 통계적으로 의미 있는 알파가 없다.**
  - 생존 편향을 빼면(PIT 유니버스) Sharpe 0.35, 알파 −4.3% (t=−0.45). 이전의 +147%는 편향이 만든 착시였다.
  - 종목 수를 늘리면(T2-10) Sharpe 1.09까지 오르지만 부트스트랩 95% 기준에서 "운과 구분 안 됨".
  - 10년 치 장기 데이터 패널(`px10y`) 생성 후 가격 기반 팩터들을 검증했으나, 단일 팩터 및 고정 합성/워크포워드(OOS) 합성 모두 다중검정 보정 임계를 넘지 못함 (과최적화/노이즈 확인).
- **다음 단계(결정 필요)**: 개별 종목 알파 추구를 중단하고 쉬운 방법(지수·모멘텀 ETF 등)을 새로운 기준선 전략으로 채택할지, 현재 페이퍼 봇들을 어떻게 처리할지 사용자와 논의 (5절).

---

## 1. 새 컴퓨터에서 시작하기

```bash
git clone https://github.com/sean-ccho/Project_1.git && cd Project_1   # 이미 있으면 git pull --ff-only
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
PYTHONPATH=.:src pytest tests -q
```

- 기대 결과: **168 통과, 2 실패.** 실패 2건(`tests/test_portfolio_report.py::test_rsi_cell_colors`, `test_vol_cell`)은 리포트 코드가 "N/A" 대신 "—"를 출력해서 생기는 **기존 문제**이고 이번 연구와 무관하다. 고치지 않아도 된다.
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
- `data/paper_trading/`의 실거래 상태(positions·trades·state)는 수정·커밋하지 않는다. 봇만 커밋한다.
- **홀드아웃(2025-10-01 이후)은 최종 후보를 확정한 뒤 딱 한 번만 본다.** 개발 구간 실행에는 모두 `--end 2025-09-30`을 붙인다. (지금 연구 패널도 2025-09-29까지다.)
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
| Tier 2 종목 수 (T2-5·10·20) | T2-10 Sharpe 1.09, MDD −11%. 그러나 "운과 구분 안 됨" → **채택 안 함** |
| H1~H6 미세조정 | **건너뜀.** [WORK_SUMMARY.md](WORK_SUMMARY.md) 4-8절 규칙: 알파 t<1이면 미세조정보다 알파 재설계가 먼저 |
| Tier 3-1 연구 패널 | 날짜×종목 피처 + 5/10/20일 선행수익률 (t+1 시가 진입 → t+1+h 시가 청산) |
| Tier 3-2 팩터 IC | 44팩터×3기간=132건. 기준(|t|≥2 & 연도 부호 일관≥67%) 통과 **1건** |
| Tier 3-3 합성 점수 | 시도 150건, 보정 임계 |t|≈3.59. 통과 없음. **Baseline v4 없음** |

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
- ⚠️ 이 후보를 홀드아웃(2025-10-01~)으로 확인하지 말 것: 홀드아웃은 최종 1회용이다. 더 긴 개발 구간(5절 2번)에서 재검증한다.

---

## 5. 다음 할 일 (결정만 남음)

앞선 검증 과정(10년 치 데이터 패널 생성, 팩터/합성 재검증)에서 의미 있는 알파를 찾지 못하여, 다음 단계에 대한 최종 결정만 남아 있습니다.

1. **🛑 논의 및 결정: 개별 종목 알파 추구 중단 및 새로운 기준선 채택**
   - 현재 방식(S&P 500 내 개별 종목 단기 트레이딩)의 알파 추구 중단을 수용할지 결정.
   - S&P 500 지수 ETF나 모멘텀 ETF를 메인으로 단순 보유하는 전략을 새로운 Baseline으로 채택할지 논의.
2. **🛑 결정: 3계좌 페이퍼 트레이딩 처리 방향**
   - 현재 매일 자동 실행되는 PT-1, PT-2, PT-3 봇들을 어떻게 처리할지(실행 중단, 로직 변경, 관찰용 방치 등) 결정.

---

## 6. 파일 지도

**스크립트** (`scripts/`)

| 파일 | 용도 |
|---|---|
| `verify_backtest_integrity.py` | Tier 0: 캐시·재현성 검사 |
| `fetch_sp500_membership.py` | PIT 구성종목 CSV 다운로드 |
| `run_hypothesis_ab.py` | A/B 백테스트 (BASE, T2-*, H1~H6), 결과를 `output/hypothesis_ab.csv`에 누적 |
| `run_account_backtest.py`, `optimize_optuna.py`, `analyze_ccs_ic.py` | PT-2·3 백테스트, Optuna, CCS IC (아직 안 돌림) |
| `build_research_panel.py` | **Tier 3-1** 패널 생성 → `data/research/panel_{pit,nonpit}.parquet` |
| `factor_research.py` | **Tier 3-2** 팩터 IC·5분위·레짐 → `tier3/` |
| `composite_research.py` | **Tier 3-3** 합성 점수 + 워크포워드 + 다중검정 → `tier3/` |

**결과 (git에 있음)**: `docs/quant_improvement/tier3/` — `FACTOR_REPORT_{pit,nonpit}.md`, `COMPOSITE_REPORT_pit.md`, `factor_ic_*.csv`, `factor_quintile_*.csv`, `factor_regime_*.csv`, `composite_pit.csv` (합계 약 150KB). **PIT 파일이 기준**, nonpit은 생존 편향 비교용.

**문서**

| 문서 | 언제 보나 |
|---|---|
| 이 문서 | 항상 먼저 |
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
| `data/cache/ohlcv_e7842ef0a144.parquet` | 27MB | PIT 유니버스 일봉 가격 (1255일 × 3625종목). 새 피처를 가격에서 다시 계산하거나 선행수익률을 바꿀 때 쓴다 |
| `data/research/SHA256SUMS.txt` | | 체크섬 |

그리고 `data/universe/sp500_membership.csv`(5.3MB)는 **git으로** 온다 (`git pull` 하면 있음).

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
