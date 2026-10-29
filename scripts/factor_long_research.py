#!/usr/bin/env python3
"""100년 팩터 연구 (factor_long_v1) — 10년 검정이 못 잡는 '진짜 우위'를 상장폐지 포함 장기 자료로 본다.

계기: 2026-10-20 사용자 — "페이퍼트레이딩 내가 만든건데 결국 이거보다 더 좋은 전략을 못 찾는 게 말이 되나? 더 좋은 전략 찾아!!"
지금까지 4,387건이 전부 2016~2025 (10년) 한 구간에서 판정됐다. 10년이면 Sharpe 차이의 95% 구간 폭이 ±0.4~0.5 라
(lab_v1 재확인 표), 연 2~3% 쯤의 현실적인 우위는 진짜여도 통과할 수 없다. 이 스크립트는 그 '검정력' 문제를 숫자로 확인하고,
Kenneth French 자료(CRSP 전 종목, 상장폐지 수익 포함 = 생존 편향 없음, 1926~)로 대형주 롱온리 팩터 기울이기가
시장을 이겼는지 60~100년 구간에서 본다.

자료 (Kenneth French 데이터 라이브러리, 월간, 시가총액 가중, 2025-09 에서 자른다):
  미국 25 포트폴리오 (NYSE 5분위 크기 × 5분위 팩터) 중 가장 큰 5분위(ME5 ≈ S&P 500 크기) 줄을 쓴다.
  시장 = Mkt-RF + RF (F-F 3팩터 월간). 해외 = 지역별 6 포트폴리오(2×3, Big = 시가총액 상위 90%) + 지역 3팩터 (USD).

사전 등록 (family factor_long_v1, 7건, 수익 보기 전 고정 — 결과 보고 바꾸지 않는다):
  C1 MOM       ME5 × 직전 12-2개월 수익 상위 5분위 (25_Portfolios_ME_Prior_12_2: BIG HiPRIOR)  비용 연 1.5%
  C2 PROF      ME5 × 영업이익률(OP) 상위 5분위 (25_Portfolios_ME_OP_5x5: BIG HiOP)            비용 연 0.5%
  C3 INV       ME5 × 자산 증가율 하위 5분위 (25_Portfolios_ME_INV_5x5: BIG LoINV)              비용 연 0.5%
  C4 VAL       ME5 × 장부가/시가 상위 5분위 (25_Portfolios_5x5: BIG HiBM)                     비용 연 0.5%
  C5 LOWBETA   ME5 × 베타 하위 5분위 (25_Portfolios_ME_BETA_5x5: BIG LoBETA)                  비용 연 0.5%
  C6 LOWVAR    ME5 × 분산 하위 5분위 (25_Portfolios_ME_VAR_5x5: BIG LoVAR)                    비용 연 0.5%
  C7 MOM+PROF  C1·C2 반반 (매월 다시 맞춤)                                                     비용 연 1.0%
  비용 = 매월 1/12 씩 뺀다 (모멘텀 ≈ 연 250% 편도 회전 × 편도 0.25% × 2 를 넉넉히 올림, 나머지는 연 1회 재구성).
  시장(지수 펀드)은 비용 0.

판정 (전부 '비용 뺀' 수익, Sharpe = 1개월 국채 초과수익 기준, 월간 × √12):
  ① 주 구간 1963-07 ~ 2025-09 (7건 공통): ΔSharpe(후보 − 시장) 블록 부트스트랩(12개월 블록, 10,000회, seed 42)
     p_개선아님 ≤ 0.05/7.
  ② 논문 발표 뒤 구간 ΔSharpe > 0 (점추정): MOM 1993-04 (Jegadeesh·Titman) · PROF 2013-01 (Novy-Marx) ·
     INV 2008-08 (Cooper·Gulen·Schill) · VAL 1992-07 (Fama·French) · LOWBETA 1992-07 (Fama·French '베타는 죽었다') ·
     LOWVAR 2006-02 (Ang·Hodrick·Xing·Zhang) · MOM+PROF 2013-01.
  ③ 해외 (MOM·PROF·INV·VAL·MOM+PROF): 유럽·일본·아시아태평양(일본 제외)·신흥국 4곳 중 3곳 이상에서 ΔSharpe > 0
     (각 지역 자료 시작 ~ 2025-09). LOWBETA·LOWVAR 는 해외 자료가 없어 ③ 없이 판정하고 그렇게 적는다.
  ④ 최근 2016-01 ~ 2025-09: ΔSharpe 95% 상한 > 0 (최근에 뚜렷이 망가지지 않았다).
  ①②③④ 모두 → '장기 우위 확인' (2단계로) · ① 만 → '장기 우위·조건부' · ① 실패 → '탈락'.
  참고(판정 아님): MOM·VAL 1927~ 구간, 10년대별 ΔSharpe, CAPM 알파·NW t, 손익분기 비용, 굴린 10년 창 통과율
  (같은 후보를 10년 창으로 쟀다면 몇 % 의 창에서 z ≥ 2.64 를 넘었을까 = 지금까지 쓴 검정의 검정력).

1단계-B (family factor_long_smid_v1, 7건 — 대형주 7건이 전부 탈락한 걸 본 뒤 2026-10-20 따로 등록):
  같은 7개 후보를 중소형주에서. 미국 = 25 포트폴리오의 ME2·ME3·ME4 줄에서 같은 분위 칸을 1/3씩 (ME1 초소형은 비용 때문에 뺌),
  해외 = 지역 2×3 의 SMALL 칸 (시가총액 하위 10%). 비용은 편도 약 0.45% 를 가정해 연 MOM 3.0% · MOM+PROF 2.0% · 나머지 1.0%.
  비교 기준은 똑같이 미국 전체 시장 (사용자의 대안 = SPY). 판정 ①~④ 도 같다 (p ≤ 0.05/7).
  참고: 같은 크기 구간 평균(ME2~4 × 5분위 15칸 동일 비중) 대비 ΔSharpe = '구간 안에서 팩터가 더했나'.

2단계 (1단계 '확인' 군만, 별도 등록): S&P 500 (그날 구성 종목) 종목 단위 구현 — 월말 신호 → 다음 거래일 시가 체결,
  비용 편도 0.1%. SPY 대비 2016-01~2025-09 ΔSharpe 95% 상한 > 0 (구현이 망가지지 않았다) + 월간 초과수익이 French 짝과
  상관 ≥ 0.5 (같은 것을 사고 있다) → 실전 페이퍼 계좌 제안 (사용자 승인 필요). 2단계는 '고르기'가 아니라 '구현 점검' 이다.
  홀드아웃(2025-10-01~)은 쓰지 않는다.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/factor_long_research.py --fetch     # 받아서 고정 (있으면 해시만 확인)
  PYTHONPATH=.:src .venv/bin/python scripts/factor_long_research.py --inspect   # 표·열·기간만 (수익 안 봄)
  PYTHONPATH=.:src .venv/bin/python scripts/factor_long_research.py --register [--family smid]
  PYTHONPATH=.:src .venv/bin/python scripts/factor_long_research.py --run [--family smid]   # 판정 + 보고서
  PYTHONPATH=.:src .venv/bin/python scripts/factor_long_research.py --report    # 저장된 표로 보고서만 다시 (TRIAL_LOG 안 건드림)
산출: tier3/FACTOR_LONG_REPORT.md · tier3/factor_long{,_smid}_{tests,decades,power,intl}.csv ·
  data/research/factor_long/ (git 제외)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from research_utils import DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, log_trials, total_trials  # noqa: E402

STEP = "factor_long"
DATA_DIR = RESEARCH_DIR / "factor_long"
SUMS = DATA_DIR / "SHA256SUMS.txt"
BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DEV_MONTH = DEV_END.to_period("M")

# 받을 파일: 키 → zip 이름. 표는 '가치(시가총액) 가중 월간' 첫 표, 팩터 파일은 첫 월간 표.
FILES: dict[str, str] = {
    "US_FF3": "F-F_Research_Data_Factors_CSV.zip",
    "US_ME_PRIOR_25": "25_Portfolios_ME_Prior_12_2_CSV.zip",
    "US_ME_OP_25": "25_Portfolios_ME_OP_5x5_CSV.zip",
    "US_ME_INV_25": "25_Portfolios_ME_INV_5x5_CSV.zip",
    "US_ME_BM_25": "25_Portfolios_5x5_CSV.zip",
    "US_ME_BETA_25": "25_Portfolios_ME_BETA_5x5_CSV.zip",
    "US_ME_VAR_25": "25_Portfolios_ME_VAR_5x5_CSV.zip",
}
REGIONS: dict[str, str] = {"EU": "Europe", "JP": "Japan", "AP": "Asia_Pacific_ex_Japan", "EM": "Emerging_Markets"}
REGION_KO = {"EU": "유럽", "JP": "일본", "AP": "아시아태평양(일본 제외)", "EM": "신흥국"}
for _code, _name in REGIONS.items():
    _fac = "Emerging_5_Factors_CSV.zip" if _code == "EM" else f"{_name}_3_Factors_CSV.zip"
    _be = "BE-ME"
    FILES[f"{_code}_FF"] = _fac
    FILES[f"{_code}_ME_PRIOR_6"] = f"{_name}_6_Portfolios_ME_Prior_12_2_CSV.zip"
    FILES[f"{_code}_ME_OP_6"] = f"{_name}_6_Portfolios_ME_OP_CSV.zip"
    FILES[f"{_code}_ME_INV_6"] = f"{_name}_6_Portfolios_ME_INV_CSV.zip"
    FILES[f"{_code}_ME_BM_6"] = f"{_name}_6_Portfolios_ME_{_be}_CSV.zip"

_TITLE_VW = re.compile(r"value\s+weight", re.IGNORECASE)
_MONTH = re.compile(r"\d{6}")

Candidate = tuple[str, list[tuple[str, str, float]], float, str, list[tuple[str, float]]]


@dataclass(frozen=True)
class Family:
    """사전 등록 묶음. candidates: 코드 → (설명, 미국 구성 [(파일 키, 열, 비중)], 연 비용, 발표 뒤 시작 월, 해외 [(팩터, 비중)])."""

    family: str
    label: str
    candidates: dict[str, Candidate]
    intl_col: dict[str, str]
    prefix: str
    segment: list[tuple[str, list[str]]] = field(default_factory=list)   # (파일 키, 열들) = 같은 크기 구간 평균 (참고)


# 1단계 (factor_long_v1, 2026-10-20 등록·실행): 대형주 = ME5 줄
LARGE = Family(
    family="factor_long_v1", label="대형주 (NYSE 크기 5분위 = ME5)", prefix="factor_long",
    candidates={
        "MOM": ("대형주 모멘텀 (12-2개월 수익 상위 20%)", [("US_ME_PRIOR_25", "BIG HiPRIOR", 1.0)], 0.015, "1993-04",
                [("PRIOR", 1.0)]),
        "PROF": ("대형주 고수익성 (영업이익률 상위 20%)", [("US_ME_OP_25", "BIG HiOP", 1.0)], 0.005, "2013-01",
                 [("OP", 1.0)]),
        "INV": ("대형주 보수적 투자 (자산 증가율 하위 20%)", [("US_ME_INV_25", "BIG LoINV", 1.0)], 0.005, "2008-08",
                [("INV", 1.0)]),
        "VAL": ("대형주 가치 (장부가/시가 상위 20%)", [("US_ME_BM_25", "BIG HiBM", 1.0)], 0.005, "1992-07",
                [("BM", 1.0)]),
        "LOWBETA": ("대형주 저베타 (베타 하위 20%)", [("US_ME_BETA_25", "BIG LoBETA", 1.0)], 0.005, "1992-07", []),
        "LOWVAR": ("대형주 저변동 (분산 하위 20%)", [("US_ME_VAR_25", "BIG LoVAR", 1.0)], 0.005, "2006-02", []),
        "MOM+PROF": ("모멘텀 + 고수익성 반반",
                     [("US_ME_PRIOR_25", "BIG HiPRIOR", 0.5), ("US_ME_OP_25", "BIG HiOP", 0.5)], 0.010, "2013-01",
                     [("PRIOR", 0.5), ("OP", 0.5)]),
    },
    intl_col={"PRIOR": "BIG HiPRIOR", "OP": "BIG HiOP", "INV": "BIG LoINV", "BM": "BIG HiBM"},
)


def _rows(key: str, tag: str, q: int, weight: float = 1.0) -> list[tuple[str, str, float]]:
    """ME2·ME3·ME4 줄의 q 분위 칸을 같은 비중으로 (중소형 구간)."""
    return [(key, f"ME{m} {tag}{q}", weight / 3) for m in (2, 3, 4)]


# 2단계 (factor_long_smid_v1, 1단계 결과를 본 뒤 2026-10-20 따로 등록): 중소형주 = ME2·ME3·ME4 줄 평균
# (ME1 = 초소형은 비용이 너무 커 뺀다). 비용은 대형주보다 크게: 편도 약 0.45% 가정.
SMID = Family(
    family="factor_long_smid_v1", label="중소형주 (NYSE 크기 2~4분위 평균)", prefix="factor_long_smid",
    candidates={
        "MOM": ("중소형 모멘텀 (12-2개월 수익 상위 20%)", _rows("US_ME_PRIOR_25", "PRIOR", 5), 0.030, "1993-04",
                [("PRIOR", 1.0)]),
        "PROF": ("중소형 고수익성 (영업이익률 상위 20%)", _rows("US_ME_OP_25", "OP", 5), 0.010, "2013-01",
                 [("OP", 1.0)]),
        "INV": ("중소형 보수적 투자 (자산 증가율 하위 20%)", _rows("US_ME_INV_25", "INV", 1), 0.010, "2008-08",
                [("INV", 1.0)]),
        "VAL": ("중소형 가치 (장부가/시가 상위 20%)", _rows("US_ME_BM_25", "BM", 5), 0.010, "1992-07", [("BM", 1.0)]),
        "LOWBETA": ("중소형 저베타 (베타 하위 20%)", _rows("US_ME_BETA_25", "BETA", 1), 0.010, "1992-07", []),
        "LOWVAR": ("중소형 저변동 (분산 하위 20%)", _rows("US_ME_VAR_25", "VAR", 1), 0.010, "2006-02", []),
        "MOM+PROF": ("중소형 모멘텀 + 고수익성 반반",
                     _rows("US_ME_PRIOR_25", "PRIOR", 5, 0.5) + _rows("US_ME_OP_25", "OP", 5, 0.5), 0.020, "2013-01",
                     [("PRIOR", 0.5), ("OP", 0.5)]),
    },
    intl_col={"PRIOR": "SMALL HiPRIOR", "OP": "SMALL HiOP", "INV": "SMALL LoINV", "BM": "SMALL HiBM"},
    segment=[("US_ME_PRIOR_25", [f"ME{m} PRIOR{q}" for m in (2, 3, 4) for q in range(1, 6)])],
)
FAMILIES = {"large": LARGE, "smid": SMID}
MAIN = (pd.Period("1963-07", "M"), DEV_MONTH)
RECENT = (pd.Period("2016-01", "M"), DEV_MONTH)
LONG_START = pd.Period("1927-01", "M")              # MOM·VAL 참고 구간 시작 (모멘텀 자료 첫 달)
BLOCK, N_BOOT, SEED = 12, 10_000, 42
ALPHA = 0.05
POWER_WINDOW = 120                                   # 굴린 창 = 10년 (지금까지 쓴 개발 구간 길이)
POWER_Z = 2.64                                       # vol_flow_v1 의 Bonferroni z (지금까지 흔히 쓴 문턱)
INTL_NEED = 3
REPORT = TIER3_DIR / "FACTOR_LONG_REPORT.md"
PT1_EQUITY = {"PT-1 지금 규칙 (BASE)": RESEARCH_DIR / "pt1_base_dev_equity.parquet",
              "PT-1 사이징 수정판 (FIX)": RESEARCH_DIR / "sizing_FIX_dev_equity.parquet"}
ETF_OHLCV = RESEARCH_DIR / "etf_ohlcv.parquet"
BLEND_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)

# 2026-10-20 실행 결과를 보고 쓴 쉬운 말 요약 (--report 가 그대로 넣는다)
SUMMARY = [
    "1. **14건 전부 탈락.** 미국 전 종목(상장폐지 포함, CRSP)으로 만든 Kenneth French 포트폴리오로 1963~2025 (62년)를 봤다. "
    "대형주 7개(모멘텀·고수익성·보수적 투자·가치·저베타·저변동·모멘텀+수익성)와 중소형주 7개(같은 것). "
    "비용을 빼면 시장(≈ SPY)보다 Sharpe 가 뚜렷하게 높은 것이 하나도 없다.",
    "2. **대형주 (S&P 500 크기)**: 비용 뒤 시장 대비 Sharpe 차이 −0.04 ~ +0.07. 비용을 0 으로 쳐도 기준을 넘는 게 없다. "
    "가장 유명한 모멘텀도 98년(1927~)으로 보면 +0.05 (p 0.13). → 지금까지 S&P 500 안에서 4,387건이 다 떨어진 건 "
    "검색을 못해서가 아니라 **대형주에는 애초에 잡을 우위가 거의 없기 때문**이다.",
    "3. **중소형주**: 비용 뒤 +0.00 ~ +0.17. 가장 가까운 건 중소형 저변동 (+0.17, p 0.022 — 기준 0.007). "
    "비용을 0 으로 치면 중소형 모멘텀(+0.17)·저변동(+0.25)이 기준을 넘지만, 둘 다 2016~2025 에는 시장보다 나빴다 (비용 0 으로도 −0.20, −0.18).",
    "4. **최근 10년은 시장(시가총액 가중 미국 지수)이 압도했다**: 2016~2025 에 중소형 7개 전부가 시장보다 Sharpe 가 0.24~0.42 낮고, "
    "대형주도 7개 중 6개가 낮다 (고수익성만 +0.11). SPY 는 이 10년 동안 가장 이기기 어려운 기준이었다.",
    "5. **10년 검정의 검정력**: 같은 후보를 10년 창으로 쟀다면 지금까지 쓴 문턱(z 2.64)을 넘는 창은 대형주 0%, 중소형 2~9%. "
    "진짜 우위가 있어도 10년 자료로는 거의 못 잡는다. 지금까지의 '탈락'은 \"없다\"의 증명이 아니라 \"10년으로는 구분할 수 없다\"였다 — "
    "그리고 이번 장기 자료로 보니 우위 자체도 작다 (비용 뒤 CAPM 알파 연 1~3%, 대부분 유의하지 않음).",
    "6. **해외**: 모멘텀은 유럽·신흥국 중소형에서 시장보다 나았다 (유럽 +0.27, p 0.001). 대형주는 해외도 약하다 (신흥국 +0.09 말고는 "
    "+0.06 이하). 일본은 가치(+0.07~+0.10)를 빼면 시장과 비슷하거나 나빴다. 우위가 남은 곳은 해외 중소형뿐 — 대형주일수록 효율적이다.",
    "7. **PT-1 이 SPY 를 못 이기는 구조적 이유**: 3종목만 들고 있어 종목 고유 위험이 크게 더해진다 (아래 표). 그 상태로 SPY 와 같은 "
    "Sharpe 를 내려면 연 5~6% 의 알파가 필요한데, 100년 자료에서 비용 뒤 그만한 알파를 꾸준히 낸 롱온리 전략은 없다.",
    "8. **그래서 PT-1 보다 나은, 근거 있는 전략 = 'SPY 코어 + PT-1 위성 (PT-1 은 20% 이하)'**: 2016~2025 에 PT-1 20% + SPY 80% 는 "
    "Sharpe 0.86 (지금 규칙) / 0.89 (사이징 수정판). PT-1 단독은 0.62 / 0.74, SPY 0.86. 즉 PT-1 단독보다 Sharpe·CAGR 이 모두 높고 SPY 와는 "
    "같거나 조금 낫다 (유의성 검정은 안 함). 지금 규칙은 30% 이상 섞으면 오히려 나빠진다 (0.84 → 0.80, MDD 도 커짐). PT-1 에 진짜 우위가 없어도 20% 위성이면 잃는 게 "
    "거의 없고, 있으면 그만큼 얻는다 — 검증 안 된 전략에 돈을 거는 올바른 방식이다. "
    "(참고 계산, 판정 아님 — 새 우위가 아니라 분산으로 위험을 줄인 것)",
    "9. **Sharadar(유료 상장폐지 자료)**: \"중소형주에 팩터 우위가 있나\" 는 이 자료(상장폐지 포함 전 종목)로 이미 봤다 — 비용 뒤 없음, "
    "최근 10년은 마이너스. 그 목적으로 살 이유는 약하다.",
    "10. **연구 규칙 제안**: 새 아이디어는 먼저 장기 자료(이 French 자료 등)로 본다. 10년짜리 S&P 500 백테스트만으로 '좋다/나쁘다'를 "
    "판정하지 않는다.",
]


# ── 받기 · 읽기 ─────────────────────────────────────────────────────────────────────────────────

def parse_tables(text: str) -> list[tuple[str, pd.DataFrame]]:
    """French CSV 텍스트 → [(표 제목, 월간 표)]. 월간(YYYYMM) 행이 있는 표만, 퍼센트 → 소수, -99.99·-999 → NaN.

    제목 = 머리글(쉼표로 시작하는 줄) 바로 앞의 글자 줄. 팩터 파일은 제목이 없거나 설명 문장이다.
    """
    tables: list[tuple[str, pd.DataFrame]] = []
    title, header, rows = "", None, []

    def _flush() -> None:
        if header is not None and rows:
            df = pd.DataFrame(rows, columns=["month", *header]).set_index("month")
            df.index = pd.PeriodIndex(df.index, freq="M")
            tables.append((title, df.where(df > -99.0) / 100.0))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if line.startswith(","):
            _flush()
            header, rows = parts[1:], []
            continue
        if header is not None and _MONTH.fullmatch(parts[0]) and len(parts) == len(header) + 1:
            rows.append([pd.Period(f"{parts[0][:4]}-{parts[0][4:]}", freq="M")] +
                        [float(x) if x not in ("", "-99.99", "-999") else -999.0 for x in parts[1:]])
            continue
        if not re.fullmatch(r"\d{4}", parts[0]):      # 연간 행이 아닌 글자 줄 = 다음 표 제목
            _flush()
            header, rows = None, []
            title = line
    _flush()
    return tables


def pick_table(tables: list[tuple[str, pd.DataFrame]], factor_file: bool) -> pd.DataFrame:
    """팩터 파일 = 첫 월간 표, 포트폴리오 파일 = 제목에 'Value Weight' 가 든 첫 월간 표."""
    if not tables:
        raise ValueError("월간 표가 없다")
    if factor_file:
        return tables[0][1]
    for title, df in tables:
        if _TITLE_VW.search(title):
            return df
    raise ValueError("가치 가중 월간 표가 없다: " + " | ".join(t for t, _ in tables))


def download(name: str) -> str:
    req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — 고정 URL
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        inner = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return z.read(inner).decode("latin-1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sums() -> dict[str, str]:
    if not SUMS.exists():
        return {}
    out = {}
    for line in SUMS.read_text().splitlines():
        if line.strip():
            h, n = line.split(maxsplit=1)
            out[n.strip()] = h
    return out


def fetch() -> None:
    """없으면 받아서 2025-09 에서 자르고 저장 + SHA256 고정. 이미 고정된 파일은 해시만 확인."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sums = _sums()
    new_lines = []
    for key, name in FILES.items():
        path = DATA_DIR / f"{key}.parquet"
        if path.name in sums:
            if sha256(path) != sums[path.name]:
                raise SystemExit(f"{path.name}: 해시가 다르다 — 고정된 자료가 바뀌었다")
            continue
        df = pick_table(parse_tables(download(name)), key.endswith(("_FF3", "_FF"))).loc[:DEV_MONTH]
        df.to_parquet(path)
        new_lines.append(f"{sha256(path)}  {path.name}")
        print(f"[factor_long] {key:<16} {name:<52} {df.index.min()} ~ {df.index.max()} · 열 {len(df.columns)}",
              flush=True)
    if new_lines:
        with SUMS.open("a") as fh:
            fh.write("\n".join(new_lines) + "\n")
    print(f"[factor_long] 고정 확인 {len(FILES)}개 파일", flush=True)


def load(key: str) -> pd.DataFrame:
    path = DATA_DIR / f"{key}.parquet"
    sums = _sums()
    if path.name not in sums or sha256(path) != sums[path.name]:
        raise SystemExit(f"{path.name}: 고정되지 않았거나 해시가 다르다 (--fetch 먼저)")
    return pd.read_parquet(path)


def inspect() -> None:
    """표·열·기간만 출력 (수익은 계산하지 않는다)."""
    for key in FILES:
        df = load(key)
        print(f"{key:<16} {df.index.min()} ~ {df.index.max()} · 빈 값 {int(df.isna().sum().sum())} · "
              f"{list(df.columns)}")


# ── 등록 ──────────────────────────────────────────────────────────────────────────────────────

REG_MEMO = {
    "factor_long_v1": (
        "Kenneth French 월간 VW, 미국 25 포트폴리오 ME5 줄 (BIG HiPRIOR·HiOP·LoINV·HiBM·LoBETA·LoVAR, MOM+PROF 반반), "
        "비용 연 1.5/0.5/0.5/0.5/0.5/0.5/1.0%."),
    "factor_long_smid_v1": (
        "1단계(대형주) 전부 탈락을 본 뒤 따로 등록. Kenneth French 월간 VW, 미국 25 포트폴리오 ME2·ME3·ME4 줄의 같은 분위 칸 "
        "1/3씩 (PRIOR5·OP5·INV1·BM5·BETA1·VAR1, MOM+PROF 반반), 해외 = 지역 2×3 SMALL 칸, "
        "비용 연 3.0/1.0/1.0/1.0/1.0/1.0/2.0%."),
}


def _registered(fam: Family) -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == fam.family) & (log["판정"] == "등록")).any())


def register(fam: Family) -> None:
    if _registered(fam):
        raise SystemExit(f"{fam.family} 은 이미 등록됐다")
    n = len(fam.candidates)
    memo = (f"{n}건 p≤0.05/{n}: " + ",".join(fam.candidates) + " | " + REG_MEMO[fam.family] +
            f" ① 1963-07~2025-09 ΔSharpe vs 시장 블록 부트스트랩(12개월·1만) p≤0.05/{n} ② 발표 뒤 ΔSharpe>0 "
            "(MOM 1993-04·PROF 2013-01·INV 2008-08·VAL 1992-07·LOWBETA 1992-07·LOWVAR 2006-02·M+P 2013-01) "
            "③ 해외 4곳(유럽·일본·아태·신흥) 중 3+ ΔSharpe>0 (LOWBETA·LOWVAR 없음) ④ 2016-01~2025-09 ΔSharpe 상한>0. "
            "정의 scripts/factor_long_research.py 머리말")
    log_trials(STEP, fam.family, n, [{"test_id": "(등록)", "판정": "등록", "메모": memo}])
    print(f"등록: {fam.family} {n}건 · 누적 시도 {total_trials()}")


# ── 수익 · 통계 ─────────────────────────────────────────────────────────────────────────────────

def market(region: str = "US") -> pd.DataFrame:
    """지역 시장 총수익(mkt)·무위험(rf), 월간 소수."""
    ff = load("US_FF3" if region == "US" else f"{region}_FF")
    return pd.DataFrame({"mkt": ff["Mkt-RF"] + ff["RF"], "rf": ff["RF"]}).dropna()


def candidate(fam: Family, code: str, region: str = "US") -> pd.Series:
    """후보의 비용 뺀 월간 총수익 (여러 칸 조합은 매월 다시 맞춤 = 월 수익의 가중 평균)."""
    _, us, cost, _, intl = fam.candidates[code]
    if region == "US":
        parts = [load(k)[c] * w for k, c, w in us]
    else:
        parts = [load(f"{region}_ME_{f}_6")[fam.intl_col[f]] * w for f, w in intl]
    gross = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
    return (gross - cost / 12.0).dropna().rename(code)


def segment_returns(fam: Family) -> pd.Series | None:
    """같은 크기 구간 평균 (칸 동일 비중, 비용 없음) — '구간 안에서 팩터가 더했나' 참고용."""
    if not fam.segment:
        return None
    parts = [load(k)[cols].mean(axis=1) for k, cols in fam.segment]
    return pd.concat(parts, axis=1).mean(axis=1).rename("seg")


def aligned(fam: Family, code: str, region: str = "US",
            window: tuple[pd.Period, pd.Period] | None = None) -> pd.DataFrame:
    """같은 달로 맞춘 후보·시장 총수익과 무위험 (window 안만)."""
    df = pd.concat([candidate(fam, code, region).rename("a"), market(region)], axis=1).dropna()
    if window is not None:
        df = df.loc[window[0]:window[1]]
    return df


def sharpe_m(ex: np.ndarray) -> float:
    """월간 초과수익 → 연율 Sharpe."""
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(12)) if sd > 0 else 0.0


def boot_dsharpe(a_ex: np.ndarray, b_ex: np.ndarray, *, block: int = BLOCK, n_boot: int = N_BOOT,
                 seed: int = SEED) -> dict[str, float]:
    """같은 달 블록을 통째로 뽑는 paired 부트스트랩 (evaluation.paired_bootstrap 과 같은 방식, 월간·numpy)."""
    n = len(a_ex)
    if n < 2 * block:
        return {}
    rng = np.random.default_rng(seed)
    k = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]

    def _sr(x: np.ndarray) -> np.ndarray:
        s = x[idx]
        return s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(12)

    d = _sr(a_ex) - _sr(b_ex)
    return {"dS": sharpe_m(a_ex) - sharpe_m(b_ex), "lo": float(np.quantile(d, ALPHA / 2)),
            "hi": float(np.quantile(d, 1 - ALPHA / 2)), "p": float((d <= 0).mean())}


def memmel_z(a_ex: np.ndarray, b_ex: np.ndarray) -> float:
    """Sharpe 차이의 해석적 z (Jobson·Korkie, Memmel 2003 보정) — 굴린 창 검정력 계산용 (부트스트랩 대용)."""
    n = len(a_ex)
    sa, sb = a_ex.mean() / a_ex.std(ddof=1), b_ex.mean() / b_ex.std(ddof=1)
    rho = float(np.corrcoef(a_ex, b_ex)[0, 1])
    theta = 2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)
    return float((sa - sb) / np.sqrt(theta / n)) if theta > 0 else float("nan")


def capm(a_ex: np.ndarray, m_ex: np.ndarray, lag: int = 12) -> tuple[float, float, float]:
    """연 알파, 알파의 Newey-West t, 베타."""
    X = np.column_stack([np.ones_like(m_ex), m_ex])
    beta, *_ = np.linalg.lstsq(X, a_ex, rcond=None)
    e = a_ex - X @ beta
    n = len(e)
    xe = X * e[:, None]
    S = xe.T @ xe / n
    for k in range(1, lag + 1):
        g = xe[k:].T @ xe[:-k] / n
        S += (1 - k / (lag + 1)) * (g + g.T)
    Q = np.linalg.inv(X.T @ X / n)
    cov = Q @ S @ Q / n
    return float(beta[0] * 12), float(beta[0] / np.sqrt(cov[0, 0])), float(beta[1])


def perf_m(r: pd.Series) -> dict[str, float]:
    """월간 총수익 → CAGR·변동성·MDD."""
    x = r.to_numpy(dtype=float)
    eq = np.cumprod(1 + x)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return {"CAGR": float(eq[-1] ** (12 / len(x)) - 1), "vol": float(x.std(ddof=1) * np.sqrt(12)),
            "MDD": float((eq / peak - 1).min())}


def breakeven_cost(df: pd.DataFrame, cost: float) -> float:
    """ΔSharpe 가 0 이 되는 연 비용 (비용 전 수익 기준). 사전 가정 비용과 비교용."""
    a_gross = (df["a"] + cost / 12 - df["rf"]).to_numpy()
    m_ex = (df["mkt"] - df["rf"]).to_numpy()
    sr_m = m_ex.mean() / m_ex.std(ddof=1)
    return float(12 * (a_gross.mean() - sr_m * a_gross.std(ddof=1)))


def window_stats(fam: Family, code: str, region: str, window: tuple[pd.Period, pd.Period] | None,
                 boot: bool) -> dict:
    df = aligned(fam, code, region, window)
    a_ex = (df["a"] - df["rf"]).to_numpy()
    m_ex = (df["mkt"] - df["rf"]).to_numpy()
    out = {"code": code, "region": region, "start": str(df.index.min()), "end": str(df.index.max()), "months": len(df),
           "Sharpe": sharpe_m(a_ex), "Sharpe_mkt": sharpe_m(m_ex), "dS": sharpe_m(a_ex) - sharpe_m(m_ex),
           "z_memmel": memmel_z(a_ex, m_ex), "corr": float(np.corrcoef(a_ex, m_ex)[0, 1])}
    alpha, t_alpha, beta = capm(a_ex, m_ex)
    pa, pm = perf_m(df["a"]), perf_m(df["mkt"])
    out.update({"alpha": alpha, "t_alpha": t_alpha, "beta": beta, "CAGR": pa["CAGR"], "CAGR_mkt": pm["CAGR"],
                "vol": pa["vol"], "vol_mkt": pm["vol"], "MDD": pa["MDD"], "MDD_mkt": pm["MDD"],
                "breakeven": breakeven_cost(df, fam.candidates[code][2])})
    if boot:
        b = boot_dsharpe(a_ex, m_ex)
        out.update({"lo": b["lo"], "hi": b["hi"], "p": b["p"]})
    seg = segment_returns(fam) if region == "US" else None
    if seg is not None:
        s = pd.concat([df, seg], axis=1, join="inner")
        out["dS_seg"] = sharpe_m((s["a"] - s["rf"]).to_numpy()) - sharpe_m((s["seg"] - s["rf"]).to_numpy())
        out["Sharpe_seg"] = sharpe_m((s["seg"] - s["rf"]).to_numpy())
    return out


def _start(code: str) -> pd.Period:
    return LONG_START if code in ("MOM", "VAL") else MAIN[0]


def power_table(fam: Family, code: str) -> dict:
    """굴린 10년 창(월 단위로 한 칸씩)마다 Memmel z — 10년 검정이었다면 몇 % 의 창에서 통과했을까."""
    df = aligned(fam, code, "US", (_start(code), DEV_MONTH))
    a_ex = (df["a"] - df["rf"]).to_numpy()
    m_ex = (df["mkt"] - df["rf"]).to_numpy()
    zs, ds = [], []
    for i in range(0, len(df) - POWER_WINDOW + 1):
        a, m = a_ex[i:i + POWER_WINDOW], m_ex[i:i + POWER_WINDOW]
        zs.append(memmel_z(a, m))
        ds.append(sharpe_m(a) - sharpe_m(m))
    zs_a, ds_a = np.array(zs), np.array(ds)
    return {"code": code, "start": str(df.index.min()), "windows": len(zs_a), "z_full": memmel_z(a_ex, m_ex),
            "dS_full": sharpe_m(a_ex) - sharpe_m(m_ex), "share_dS_pos": float((ds_a > 0).mean()),
            "share_z_1645": float((zs_a >= 1.645).mean()), f"share_z_{POWER_Z}": float((zs_a >= POWER_Z).mean()),
            "dS_10y_median": float(np.median(ds_a)), "dS_10y_p10": float(np.quantile(ds_a, 0.1)),
            "dS_10y_p90": float(np.quantile(ds_a, 0.9))}


def decades(fam: Family, code: str) -> dict:
    df = aligned(fam, code, "US", (_start(code), DEV_MONTH))
    out = {"code": code}
    for dec, g in df.groupby(df.index.year // 10 * 10):
        if len(g) >= 24:
            out[f"{dec}s"] = sharpe_m((g["a"] - g["rf"]).to_numpy()) - sharpe_m((g["mkt"] - g["rf"]).to_numpy())
    return out


def judge(n_tests: int, main_row: dict, post: dict, intl_rows: list[dict],
          recent: dict) -> tuple[str, dict[str, bool | None]]:
    g1 = main_row["p"] <= ALPHA / n_tests
    g2 = post["dS"] > 0
    g3 = None if not intl_rows else sum(r["dS"] > 0 for r in intl_rows) >= INTL_NEED
    g4 = recent["hi"] > 0
    gates = {"①": g1, "②": g2, "③": g3, "④": g4}
    if not g1:
        return "탈락", gates
    if g2 and g4 and g3 is not False:
        return "장기 우위 확인", gates
    return "장기 우위·조건부", gates


def run(fam: Family) -> None:
    if not _registered(fam):
        raise SystemExit("등록 먼저 (--register)")
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    if ((log["family"] == fam.family) & (log["판정"] != "등록")).any():
        raise SystemExit(f"{fam.family} 결과가 이미 기록됐다 — 보고서만 다시 쓰려면 --report")
    tests, intl_all, power, dec_rows = [], [], [], []
    for code, (desc, _, cost, pub, intl) in fam.candidates.items():
        main_row = window_stats(fam, code, "US", MAIN, boot=True)
        post = window_stats(fam, code, "US", (pd.Period(pub, "M"), DEV_MONTH), boot=True)
        recent = window_stats(fam, code, "US", RECENT, boot=True)
        long_row = window_stats(fam, code, "US", (LONG_START, DEV_MONTH), boot=True) if code in ("MOM", "VAL") else None
        intl_rows = [window_stats(fam, code, reg, None, boot=True) for reg in REGIONS] if intl else []
        verdict, gates = judge(len(fam.candidates), main_row, post, intl_rows, recent)
        tests.append({"code": code, "설명": desc, "비용": cost, "발표뒤": pub, "판정": verdict,
                      **{f"gate{k}": v for k, v in gates.items()},
                      **{f"main_{k}": v for k, v in main_row.items() if k not in ("code", "region")},
                      **{f"post_{k}": post[k] for k in ("start", "months", "Sharpe", "Sharpe_mkt", "dS", "lo", "hi", "p")},
                      **{f"recent_{k}": recent[k] for k in ("Sharpe", "Sharpe_mkt", "dS", "lo", "hi", "p", "CAGR",
                                                            "CAGR_mkt", "MDD", "MDD_mkt")},
                      **({f"long_{k}": long_row[k] for k in ("start", "months", "Sharpe", "Sharpe_mkt", "dS", "lo", "hi",
                                                              "p", "alpha", "t_alpha", "CAGR", "CAGR_mkt", "MDD",
                                                              "MDD_mkt")} if long_row else {}),
                      "intl_pos": sum(r["dS"] > 0 for r in intl_rows) if intl_rows else None})
        intl_all += intl_rows
        power.append(power_table(fam, code))
        dec_rows.append(decades(fam, code))
        print(f"[factor_long] {fam.family} {code:<9} {verdict}  ΔSharpe {main_row['dS']:+.2f} [{main_row['lo']:+.2f}, "
              f"{main_row['hi']:+.2f}] p {main_row['p']:.4f}", flush=True)
    pd.DataFrame(tests).to_csv(TIER3_DIR / f"{fam.prefix}_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(intl_all).to_csv(TIER3_DIR / f"{fam.prefix}_intl.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(power).to_csv(TIER3_DIR / f"{fam.prefix}_power.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dec_rows).to_csv(TIER3_DIR / f"{fam.prefix}_decades.csv", index=False, encoding="utf-8-sig")
    log_trials(STEP, fam.family, len(fam.candidates), [
        {"test_id": t["code"], "t": f"{t['main_z_memmel']:.2f}", "판정": t["판정"],
         "메모": (f"ΔSharpe {t['main_dS']:+.2f} [{t['main_lo']:+.2f}, {t['main_hi']:+.2f}] p {t['main_p']:.4f} · "
                f"발표뒤 {t['post_dS']:+.2f} · 해외+ {t['intl_pos']} · 최근 {t['recent_dS']:+.2f} "
                f"[{t['recent_lo']:+.2f}, {t['recent_hi']:+.2f}]")} for t in tests])
    print(f"[factor_long] 누적 시도 {total_trials()}", flush=True)


# ── 보고서 ─────────────────────────────────────────────────────────────────────────────────────

def gross_sensitivity(fam: Family) -> pd.DataFrame:
    """비용 0 으로 다시 잰 주 구간·최근 ΔSharpe (참고, 판정 아님)."""
    zero = Family(fam.family, fam.label, {k: (d, us, 0.0, pub, intl) for k, (d, us, _, pub, intl) in
                                          fam.candidates.items()}, fam.intl_col, fam.prefix, fam.segment)
    rows = []
    for code in zero.candidates:
        m = window_stats(zero, code, "US", MAIN, boot=True)
        r = window_stats(zero, code, "US", RECENT, boot=False)
        rows.append({"code": code, "dS": m["dS"], "p": m["p"], "pass": m["p"] <= ALPHA / len(fam.candidates),
                     "recent_dS": r["dS"]})
    return pd.DataFrame(rows)


def pt1_blend() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """PT-1 백테스트(2016-01 ~ 2025-09) 와 SPY 를 섞은 일간 수익 (매일 비중 맞춤 근사, 참고).

    Returns: (섞기 표, PT-1 구조 표: 베타·고유 위험·알파·SPY 와 같은 Sharpe 에 필요한 알파). 파일이 없으면 None.
    """
    if not ETF_OHLCV.exists() or not all(p.exists() for p in PT1_EQUITY.values()):
        return None
    from research_utils import load_ohlcv
    spy = load_ohlcv(ETF_OHLCV)["SPY"]["Close"].pct_change()

    def _stats(r: pd.Series) -> dict[str, float]:
        eq = (1 + r).cumprod()
        return {"Sharpe": float(r.mean() / r.std() * np.sqrt(252)), "vol": float(r.std() * np.sqrt(252)),
                "CAGR": float(eq.iloc[-1] ** (252 / len(r)) - 1), "MDD": float((eq / eq.cummax() - 1).min())}

    blends, struct = [], []
    for name, path in PT1_EQUITY.items():
        pt = pd.read_parquet(path)["equity"].pct_change()
        df = pd.concat({"p": pt, "s": spy.reindex(pt.index)}, axis=1).dropna()
        for w in BLEND_WEIGHTS:
            blends.append({"PT-1": name, "PT-1 비중": w, **_stats(w * df["p"] + (1 - w) * df["s"])})
        beta = float(np.cov(df["p"], df["s"])[0, 1] / df["s"].var())
        idio = float((df["p"] - beta * df["s"]).std() * np.sqrt(252))
        alpha = float((df["p"].mean() - beta * df["s"].mean()) * 252)
        sp, ps = _stats(df["s"]), _stats(df["p"])
        need = sp["Sharpe"] * ps["vol"] - beta * float(df["s"].mean() * 252)
        struct.append({"PT-1": name, "corr": float(df.corr().iloc[0, 1]), "beta": beta, "idio": idio, "alpha": alpha,
                       "alpha_t": alpha / idio * np.sqrt(len(df) / 252), "need": need})
    return pd.DataFrame(blends), pd.DataFrame(struct)


def _pct(v: float, signed: bool = False) -> str:
    return "" if pd.isna(v) else (f"{v:+.1%}" if signed else f"{v:.1%}")


def _ox(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "—"
    return "O" if str(v) in ("True", "1", "1.0") else "X"


def _family_section(fam: Family) -> list[str]:
    t = pd.read_csv(TIER3_DIR / f"{fam.prefix}_tests.csv", encoding="utf-8-sig")
    intl = pd.read_csv(TIER3_DIR / f"{fam.prefix}_intl.csv", encoding="utf-8-sig")
    pw = pd.read_csv(TIER3_DIR / f"{fam.prefix}_power.csv", encoding="utf-8-sig")
    dec = pd.read_csv(TIER3_DIR / f"{fam.prefix}_decades.csv", encoding="utf-8-sig")
    gs = gross_sensitivity(fam)
    seg = "main_dS_seg" in t.columns
    lines = [f"## {fam.label} — family `{fam.family}`", "",
             f"주 구간 1963-07 ~ 2025-09 · 시장 Sharpe {t['main_Sharpe_mkt'].iloc[0]:.2f} · 비용 뺀 뒤 · "
             f"① 기준 p ≤ 0.05/{len(t)} = {0.05 / len(t):.4f}", "",
             "| 후보 | 비용/년 | Sharpe | ΔSharpe [95%] | p | CAPM 알파/년 (t) | 베타 | CAGR (시장) | MDD (시장) | 손익분기 비용 |"
             + (" 구간 평균 대비 Δ |" if seg else "") + " ① | ② 발표 뒤 Δ | ③ 해외 + | ④ 2016~ Δ [95%] | 판정 |",
             "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if seg else "") + "---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['code']} {r['설명']} | {r['비용']:.1%} | {r['main_Sharpe']:.2f} | {r['main_dS']:+.2f} "
            f"[{r['main_lo']:+.2f}, {r['main_hi']:+.2f}] | {r['main_p']:.3f} | {_pct(r['main_alpha'], True)} "
            f"({r['main_t_alpha']:+.1f}) | {r['main_beta']:.2f} | {_pct(r['main_CAGR'])} ({_pct(r['main_CAGR_mkt'])}) | "
            f"{_pct(r['main_MDD'])} ({_pct(r['main_MDD_mkt'])}) | {_pct(r['main_breakeven'])} |"
            + (f" {r['main_dS_seg']:+.2f} |" if seg else "") +
            f" {_ox(r['gate①'])} | {r['post_dS']:+.2f} ({r['post_start']}~) {_ox(r['gate②'])} | "
            f"{'' if pd.isna(r['intl_pos']) else f'{int(r.intl_pos)}/4 '}{_ox(r['gate③'])} | {r['recent_dS']:+.2f} "
            f"[{r['recent_lo']:+.2f}, {r['recent_hi']:+.2f}] {_ox(r['gate④'])} | **{r['판정']}** |")
    if "long_dS" in t.columns:
        lr = t.dropna(subset=["long_dS"])
        lines += ["", "**1927-01 ~ 2025-09 (98년, 참고)**", "",
                  "| 후보 | Sharpe (시장) | ΔSharpe [95%] | p | CAPM 알파/년 (t) | CAGR (시장) | MDD (시장) |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['code']} | {r['long_Sharpe']:.2f} ({r['long_Sharpe_mkt']:.2f}) | {r['long_dS']:+.2f} "
                  f"[{r['long_lo']:+.2f}, {r['long_hi']:+.2f}] | {r['long_p']:.3f} | {_pct(r['long_alpha'], True)} "
                  f"({r['long_t_alpha']:+.1f}) | {_pct(r['long_CAGR'])} ({_pct(r['long_CAGR_mkt'])}) | "
                  f"{_pct(r['long_MDD'])} ({_pct(r['long_MDD_mkt'])}) |" for _, r in lr.iterrows()]
    lines += ["", "**비용 0 으로 다시 잰 값 (참고, 판정 아님)**", "",
              "| 후보 | 1963~2025 ΔSharpe | p | ① 기준 넘음 | 2016~2025 ΔSharpe |", "|---|---|---|---|---|"]
    lines += [f"| {r.code} | {r.dS:+.2f} | {r.p:.4f} | {'O' if r['pass'] else 'X'} | {r.recent_dS:+.2f} |"
              for _, r in gs.iterrows()]
    lines += ["", "**10년 검정이었다면 (굴린 120개월 창, 한 달씩 이동 · Memmel z)**", "",
              "| 후보 | 시작 | 창 수 | 전체 z | 창 중 Δ>0 | z ≥ 1.645 | z ≥ 2.64 | 10년 ΔSharpe 중앙 (10~90%) |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r.code} | {r.start} | {int(r.windows)} | {r.z_full:+.2f} | {r.share_dS_pos:.0%} | "
              f"{r.share_z_1645:.0%} | {r['share_z_2.64']:.0%} | {r.dS_10y_median:+.2f} ({r.dS_10y_p10:+.2f} ~ "
              f"{r.dS_10y_p90:+.2f}) |" for _, r in pw.iterrows()]
    cols = [c for c in dec.columns if c != "code"]
    lines += ["", "**10년대별 ΔSharpe (비용 뒤)**", "", "| 후보 | " + " | ".join(cols) + " |",
              "|---|" + "---|" * len(cols)]
    lines += ["| " + r["code"] + " | " + " | ".join("" if pd.isna(r[c]) else f"{r[c]:+.2f}" for c in cols) + " |"
              for _, r in dec.iterrows()]
    if len(intl):
        lines += ["", "**해외 (지역 시장 대비, 비용 뒤, 각 지역 자료 시작 ~ 2025-09)**", "",
                  "| 후보 | " + " | ".join(REGION_KO[k] for k in REGIONS) + " |", "|---|" + "---|" * len(REGIONS)]
        for code, g in intl.groupby("code", sort=False):
            cells = {r.region: f"{r.dS:+.2f} (p {r.p:.3f})" for r in g.itertuples()}
            lines.append(f"| {code} | " + " | ".join(cells.get(k, "") for k in REGIONS) + " |")
    return lines + [""]


def report() -> None:
    lines = [
        "# 100년 팩터 연구 — 대형주·중소형주 롱온리 팩터가 시장을 이겼나 (factor_long_v1 · factor_long_smid_v1)", "",
        f"- 실행 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 정의·사전 등록은 `scripts/factor_long_research.py` 머리말 · "
        f"누적 시도 {total_trials()}건",
        "- 자료: Kenneth French 데이터 라이브러리 (CRSP 미국 전 종목, 상장폐지 수익 포함 = 생존 편향 없음), 월간 시가총액 가중, "
        "2025-09 에서 자름. 시장 = Mkt-RF + RF. 해외 = 지역별 2×3 포트폴리오 + 지역 팩터 (USD).",
        "- Sharpe = 1개월 국채 초과수익 기준 (월간 × √12). ΔSharpe 95% 구간·p = 12개월 블록 부트스트랩 1만 회. "
        "비용은 사전 가정치를 매월 1/12 씩 뺐다. 홀드아웃(2025-10~)은 쓰지 않았다.", "",
        "## 쉬운 말 요약", "", *SUMMARY, ""]
    for fam in FAMILIES.values():
        if (TIER3_DIR / f"{fam.prefix}_tests.csv").exists():
            lines += _family_section(fam)
    blend = pt1_blend()
    if blend is not None:
        bl, st = blend
        lines += ["## 참고: PT-1 의 구조와 'SPY 코어 + PT-1 위성' (2016-01 ~ 2025-09 백테스트, 판정 아님)", "",
                  "PT-1 백테스트 일간 수익 vs SPY (종가). Sharpe 는 무위험을 빼지 않은 값 (지금까지 PT-1 보고서들과 같은 방식).", "",
                  "| PT-1 | SPY 상관 | 베타 | 종목 고유 위험/년 | 알파/년 (t) | SPY 와 같은 Sharpe 에 필요한 알파 |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['PT-1']} | {r['corr']:.2f} | {r['beta']:.2f} | {_pct(r['idio'])} | {_pct(r['alpha'], True)} "
                  f"({r['alpha_t']:+.1f}) | {_pct(r['need'], True)} |" for _, r in st.iterrows()]
        lines += ["", "| PT-1 | PT-1 비중 (나머지 SPY) | Sharpe | 변동성 | CAGR | MDD |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['PT-1']} | {r['PT-1 비중']:.0%} | {r['Sharpe']:.2f} | {_pct(r['vol'])} | {_pct(r['CAGR'])} | "
                  f"{_pct(r['MDD'])} |" for _, r in bl.iterrows()]
        lines += ["", "- 매일 비중을 맞춘 근사다 (실제로는 월·분기 1회 맞춤). 비중 20~50% 사이는 Sharpe 가 거의 같아 어느 쪽을 골라도 된다.",
                  "- 이 표는 새 우위를 찾은 게 아니다. PT-1 이 SPY 와 덜 겹치는 만큼(상관 0.6) 섞으면 위험이 줄어드는 '분산 효과'다.", ""]
    lines += ["## 한계", "",
              "- French 포트폴리오는 매매 비용 전 수익이다. 비용은 회전율 추정치로 넉넉히 뺐지만 실제 비용은 시대마다 다르다 "
              "(1970~80년대는 더 크고, 2001년 소수점 호가 이후는 더 작다). 비용 0 으로도 대형주는 기준을 넘지 못했다.",
              "- 25 포트폴리오는 NYSE 기준 크기 5분위 × 팩터 5분위 칸이다. 실제 구현(종목 20~100개)은 이 칸과 다르게 움직일 수 있다.",
              "- 한 번에 한 팩터만 본다 (종목 단위 다중 팩터 결합은 이 자료로 볼 수 없다). 하지만 2016~2025 S&P 500 종목 단위 결합 "
              "연구(composite·ML·idea_batch)도 이미 실패했다.",
              "- 시장 Mkt 는 CRSP 전 종목 시가총액 가중이다 (SPY 와 거의 같지만 같지는 않다).",
              "- 1단계-B (중소형)는 1단계 결과를 본 뒤 등록했다. 그래서 따로 family 로 세고 Bonferroni 도 따로 했다."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[factor_long] 보고서: {REPORT.relative_to(ROOT)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--inspect", action="store_true")
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--family", choices=list(FAMILIES), default="large",
                    help="large = 1단계 대형주 (factor_long_v1) · smid = 중소형주 (factor_long_smid_v1)")
    args = ap.parse_args()
    fam = FAMILIES[args.family]
    if args.fetch:
        fetch()
    elif args.inspect:
        inspect()
    elif args.register:
        register(fam)
    elif args.run:
        run(fam)
        report()
    elif args.report:
        report()


if __name__ == "__main__":
    main()
