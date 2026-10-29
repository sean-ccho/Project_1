#!/usr/bin/env python3
"""경기·자금 흐름 따라 섹터 갈아타기 (rotation_v1) — 사용자의 '돈은 흐름을 탄다' 가설을 75~98년 자료로 시험한다.

계기: 2026-10-20 사용자 — "돈은 항상 흐름을 타고 움직이잖아. 경기방어주는 경기가 안 좋을 때 올라가고, 펀드매니저들은 지금
헬스케어를 많이 담고, 경기가 좋을 때는 성장주·테크를 담고… 거의 항상 올라가는 주식들이 있는데 왜 못 찾는 거야?"
이미 본 것: lab_G_v1 (가격 추세로 순환주↔방어주, 탈락), idea_batch_v1 (섹터 ETF 모멘텀 상위 3, 2016~2025, 탈락).
이번에는 ① 가격이 아니라 '실제 경기' 신호(실업률)로 바꾸고 ② 산업 모멘텀을 98년 자료로 본다.

자료 (2025-09 에서 자른다):
  Kenneth French 12 산업 월간 시가총액 가중 (1926-07~, CRSP 전 종목·상장폐지 포함) · 미국 시장 = Mkt-RF + RF (factor_long 고정본)
  FRED UNRATE (실업률, 1948-01~) · FRED USREC (NBER 경기침체 달, 참고용 — 사후에 정해진 날짜라 실전 신호로 못 쓴다)

사전 등록 (family rotation_v1, 3건, 결과 보기 전 고정):
  경기 신호 = Sahm 규칙: 실업률 3개월 평균 − 직전 12개월 동안의 3개월 평균 최저치 ≥ 0.50%p 이면 '침체'.
    실업률은 다음 달 초에 발표되므로, m 월 보유는 m−2 월 실업률까지로 정한다 (m−1 월 말 결정, 발표 지연 1개월).
  방어주 = 필수소비재·통신·유틸리티·헬스케어 4개 산업 동일 비중 (lab_G_v1 과 같은 구분). 테크 = BusEq (IT 장비·소프트웨어).
  R1 경기→방어/시장   침체면 방어주, 아니면 시장 전체. 1950-01 ~ 2025-09.
  R2 경기→방어/테크   침체면 방어주, 아니면 테크(BusEq). 1950-01 ~ 2025-09. (사용자가 말한 그대로)
  R3 돈 몰린 산업     매월 말 12개 산업 중 직전 6개월 수익 상위 3개를 동일 비중으로 한 달 (Moskowitz·Grinblatt 1999). 1927-01 ~ 2025-09.
  매매: 월 1회, 비중은 한 달 동안 가격 따라 움직이고, 바꾼 비중 합 × 편도 0.1% 비용. 비교 기준 = 미국 시장 (비용 0).
판정 (Sharpe = 1개월 국채 초과수익, 월간 × √12):
  ① 주 구간 ΔSharpe(전략 − 시장) 12개월 블록 부트스트랩 1만 회 p ≤ 0.05/3
  ② 2016-01 ~ 2025-09 ΔSharpe 95% 상한 > 0 (최근에 뚜렷이 망가지지 않음)
  ③ R3 만: 논문 발표 뒤 1999-08 ~ ΔSharpe > 0 (R1·R2 는 Sahm 규칙 발표 2019 뒤가 너무 짧아 없음)
  모두 → '통과 후보' (실전 전에 따로 승인) · 아니면 '탈락'.
확인 (family rotation_confirm_v1, 1건 — R3 가 '통과 후보'가 된 걸 본 뒤 2026-10-20 따로 등록):
  C1 실제 섹터 ETF   같은 R3 규칙을 Select Sector SPDR (XLB·XLE·XLF·XLI·XLK·XLP·XLU·XLV·XLY, 생기면 XLRE·XLC 추가)에.
     수정 종가(배당 재투자) 월말 기준, 직전 6개월 수익 상위 3개 동일 비중, 월 1회, 편도 0.1%. 비교 = SPY (같은 방식).
     1999-07 ~ 2025-09 (첫 6개월은 순위 계산용). 판정: ΔSharpe > 0 이고 95% 상한 > 0 → '확인' (실전 페이퍼 계좌 제안),
     아니면 '확인 안 됨'. 26년이라 유의성은 기대하지 않는다 — '살 수 있는 상품으로 해도 시장보다 나빠지지 않았나'를 본다.
  참고: R3 이웃 (직전 3·6·9·12개월 × 상위 2·3·4개), 10년대별 ΔSharpe, 편도 0.5% 비용일 때 (1990년대 전 현실 비용).
참고 (판정 아님, 실전에서는 불가능한 '뒷북' 전략 — 기회가 얼마나 있었나를 본다):
  H1 그해 1등 산업을 그해 처음부터 (완벽한 뒷북) · H2 작년 1등 산업을 올해 · H3 NBER 침체 달을 미리 다 알고 방어주
  침체별 표: 시장·방어주·테크 수익, Sahm 신호가 켜진 달, 시장 바닥과의 시차.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/rotation_research.py --fetch      # FRED·French 12 산업 월간 받아 고정
  PYTHONPATH=.:src .venv/bin/python scripts/rotation_research.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/rotation_research.py --run        # 판정 + 보고서
  PYTHONPATH=.:src .venv/bin/python scripts/rotation_research.py --report     # 저장된 표로 보고서만 다시
  PYTHONPATH=.:src .venv/bin/python scripts/rotation_research.py --fetch-etf  # 섹터 ETF·SPY 월간 (yfinance) 받아 고정
  PYTHONPATH=.:src .venv/bin/python scripts/rotation_research.py --register-confirm
  PYTHONPATH=.:src .venv/bin/python scripts/rotation_research.py --confirm    # C1 판정 + 보고서
산출: tier3/ROTATION_REPORT.md · tier3/rotation_{tests,recessions,hindsight}.csv · data/research/rotation/ (git 제외)
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import factor_long_research as fl  # noqa: E402
from research_utils import DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, log_trials, total_trials  # noqa: E402

STEP = "rotation"
FAMILY = "rotation_v1"
DATA_DIR = RESEARCH_DIR / "rotation"
SUMS = DATA_DIR / "SHA256SUMS.txt"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
IND12_ZIP = "12_Industry_Portfolios_CSV.zip"
DEV_MONTH = DEV_END.to_period("M")
DEFENSIVE = ("NoDur", "Telcm", "Utils", "Hlth")
TECH = "BusEq"
IND_KO = {"NoDur": "필수소비재", "Durbl": "내구재", "Manuf": "제조업", "Enrgy": "에너지", "Chems": "화학",
          "BusEq": "테크(IT장비)", "Telcm": "통신", "Utils": "유틸리티", "Shops": "유통", "Hlth": "헬스케어",
          "Money": "금융", "Other": "기타"}
SAHM_ON = 0.50
PUB_LAG = 2                       # m 월 보유는 m−2 월 실업률까지 (m−1 월 말 결정)
MOM_LOOKBACK, MOM_TOP = 6, 3
COST = 0.001
N_TESTS = 3
WINDOWS = {"R1": ("1950-01", "경기 신호 → 침체면 방어주, 아니면 시장"),
           "R2": ("1950-01", "경기 신호 → 침체면 방어주, 아니면 테크"),
           "R3": ("1927-01", "직전 6개월 수익 상위 3개 산업 (돈 몰린 섹터)")}
RECENT = (pd.Period("2016-01", "M"), DEV_MONTH)
R3_POST = pd.Period("1999-08", "M")
REPORT = TIER3_DIR / "ROTATION_REPORT.md"
FAMILY_CONFIRM = "rotation_confirm_v1"
SECTOR_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC")
ETF_START = "1998-12-01"
CONFIRM_START = pd.Period("1999-07", "M")
NEIGHBORS = [(lb, k) for lb in (3, 6, 9, 12) for k in (2, 3, 4)]
OLD_COST = 0.005

# 2026-10-20 실행 결과를 보고 쓴 쉬운 말 요약 (--report 가 그대로 넣는다)
SUMMARY: list[str] = [
    "1. **경기 신호로 방어주 ↔ 시장/테크 갈아타기(R1·R2)는 실패.** 1950~2025, 실제 실업률(Sahm 규칙)로 침체를 잡으면 "
    "시장 대비 Sharpe 차이 −0.00 / −0.05. 신호는 침체가 시작되고 보통 3~4개월 뒤에 켜지고, 시장 바닥보다 1년 넘게 늦게 꺼진다 "
    "(아래 침체 표). 주가는 경기보다 먼저 움직인다.",
    "2. **'방어주는 경기 나쁠 때 오른다'는 절반만 맞다**: 1950 년 이후 침체 11번 중 방어주가 시장보다 나았던 건 8번이지만, "
    "실제로 오른 건 6번뿐이고 2008년엔 −24% 였다. 시장도 침체 11번 중 5번은 올랐다 (침체 한가운데서 바닥을 찍고 먼저 오른다). "
    "NBER 침체 달을 미리 다 알았다고 쳐도 (불가능) Sharpe 0.62 vs 0.55 정도다.",
    "3. **'돈 몰린 산업 따라가기'(R3)는 98년으로 보면 진짜다**: 12개 산업 중 직전 6개월 수익 상위 3개 → 1927~2025 Sharpe 0.58 vs "
    "시장 0.45 (p 0.0003), 설정을 바꿔도 (직전 3~12개월 × 상위 2~4개) 전부 +0.11 ~ +0.20. 지금까지 4,408건 중 처음으로 장기 기준을 넘었다.",
    "4. **하지만 살 수 있는 상품으로는 시장과 똑같았다**: 같은 규칙을 실제 섹터 ETF(XLK·XLV 등)로 1999~2025 → Sharpe 0.47 vs "
    "SPY 0.47, 연 8.0% vs 8.1% (최대 낙폭만 −45% vs −51%). 2016~2025 는 −0.16. 1990년대 전의 실제 거래비용(편도 0.5%)을 넣으면 "
    "98년 우위도 사라진다 (−0.01). 효과가 있던 시절엔 비싸서 못 했고, 싸진 뒤엔 효과가 없어졌다.",
    "5. **뒷북으로 보면 기회는 늘 있었다**: 매년 그해 1등 산업을 1월부터 샀다면 연 35% (시장 10%). 그런데 작년 1등 산업을 올해 사면 "
    "연 10.5% — 시장과 같다. 작년 1등이 올해 3등 안에 든 해는 29% (아무렇게나 골라도 25%). '항상 오르는 주식이 있다'는 맞다. "
    "문제는 그걸 미리 아는 방법이 없다는 것이다.",
]


# ── 받기 · 읽기 ─────────────────────────────────────────────────────────────────────────────────

def _fetch_fred(sid: str) -> pd.Series:
    req = urllib.request.Request(FRED_URL.format(sid=sid), headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — 고정 URL
        df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
    s = pd.Series(df.iloc[:, 1].astype(float).to_numpy(), index=pd.PeriodIndex(df.iloc[:, 0], freq="M"), name=sid)
    return s.loc[:DEV_MONTH]


def fetch() -> None:
    """FRED UNRATE·USREC 와 French 12 산업 월간을 받아 고정 (이미 고정된 파일은 해시만 확인)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sums = _sums()
    items = {"UNRATE": lambda: _fetch_fred("UNRATE").to_frame(), "USREC": lambda: _fetch_fred("USREC").to_frame(),
             "IND12": lambda: fl.pick_table(fl.parse_tables(fl.download(IND12_ZIP)), factor_file=False).loc[:DEV_MONTH]}
    new = []
    for key, get in items.items():
        path = DATA_DIR / f"{key}.parquet"
        if path.name in sums:
            if fl.sha256(path) != sums[path.name]:
                raise SystemExit(f"{path.name}: 해시가 다르다 — 고정된 자료가 바뀌었다")
            continue
        df = get()
        df.to_parquet(path)
        new.append(f"{fl.sha256(path)}  {path.name}")
        print(f"[rotation] {key}: {df.index.min()} ~ {df.index.max()} · 열 {list(df.columns)}", flush=True)
    if new:
        with SUMS.open("a") as fh:
            fh.write("\n".join(new) + "\n")
    print("[rotation] 고정 확인", flush=True)


def _sums() -> dict[str, str]:
    if not SUMS.exists():
        return {}
    return {n.strip(): h for h, n in (line.split(maxsplit=1) for line in SUMS.read_text().splitlines() if line.strip())}


def load(key: str) -> pd.DataFrame:
    path = DATA_DIR / f"{key}.parquet"
    sums = _sums()
    if path.name not in sums or fl.sha256(path) != sums[path.name]:
        raise SystemExit(f"{path.name}: 고정되지 않았거나 해시가 다르다 (--fetch 먼저)")
    return pd.read_parquet(path)


def assets() -> pd.DataFrame:
    """월간 총수익: 12 산업 + 방어주 바구니(동일 비중, 매월 맞춤) + 시장(MKT) + 무위험(RF)."""
    ind = load("IND12")
    ind.columns = [c.strip() for c in ind.columns]
    mk = fl.market("US")
    out = ind.copy()
    out["DEF"] = ind[list(DEFENSIVE)].mean(axis=1)
    out["MKT"] = mk["mkt"]
    out["RF"] = mk["rf"]
    return out.dropna(subset=["MKT", "RF"])


# ── 신호 ─────────────────────────────────────────────────────────────────────────────────────

def sahm_indicator(u: pd.Series) -> pd.Series:
    """Sahm 규칙 값 = 3개월 평균 실업률 − 직전 12개월 3개월 평균의 최저치 (FRED SAHMCURRENT 와 같은 정의)."""
    u3 = u.rolling(3).mean()
    return u3 - u3.shift(1).rolling(12).min()


def recession_signal(index: pd.PeriodIndex) -> pd.Series:
    """m 월 보유에 쓰는 침체 신호 (True/False). m−2 월 Sahm 값 ≥ 0.50."""
    s = sahm_indicator(load("UNRATE")["UNRATE"])
    lagged = s.shift(PUB_LAG)
    return (lagged.reindex(index) >= SAHM_ON).where(lagged.reindex(index).notna())


def momentum_weights(r: pd.DataFrame, lookback: int = MOM_LOOKBACK, top: int = MOM_TOP,
                     min_assets: int | None = None) -> pd.DataFrame:
    """m 월 행 = 직전 lookback 개월(m−lookback ~ m−1) 수익 상위 top 개 동일 비중.

    순위를 낼 수 있는 자산(그 기간 수익이 다 있는 것)이 min_assets 개 미만이면 그달은 결정하지 않는다 (NaN).
    """
    past = (1 + r).rolling(lookback).apply(np.prod, raw=True).shift(1) - 1
    need = r.shape[1] if min_assets is None else min_assets
    w = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    for m, row in past.iterrows():
        row = row.dropna()
        if len(row) < need:
            w.loc[m] = np.nan
            continue
        w.loc[m, row.nlargest(top).index] = 1.0 / top
    return w


def target_weights(code: str, a: pd.DataFrame, lookback: int = MOM_LOOKBACK, top: int = MOM_TOP) -> pd.DataFrame:
    """m 월 처음 목표 비중 (m 월 행). 정보는 m−1 월 말까지만."""
    cols = list(IND_KO) + ["DEF", "MKT"]
    w = pd.DataFrame(0.0, index=a.index, columns=cols)
    if code in ("R1", "R2", "H3"):
        rec = recession_signal(a.index) if code != "H3" else load("USREC")["USREC"].reindex(a.index) == 1
        good = "MKT" if code in ("R1", "H3") else TECH
        w.loc[rec == True, "DEF"] = 1.0                       # noqa: E712 — NaN 은 결정 안 함
        w.loc[rec == False, good] = 1.0                       # noqa: E712
        w.loc[rec.isna()] = np.nan
    elif code == "R3":
        w[list(IND_KO)] = momentum_weights(a[list(IND_KO)], lookback, top)
        w.loc[w[list(IND_KO)].isna().all(axis=1)] = np.nan
    else:
        raise ValueError(code)
    return w


def simulate(w: pd.DataFrame, a: pd.DataFrame, cost: float = COST) -> pd.DataFrame:
    """월간 매매: 매월 처음 목표 비중으로 맞추고 (바꾼 비중 합 × 비용), 한 달 동안 가격 따라 움직인다."""
    w = w.dropna(how="all")
    r = a.loc[w.index, w.columns].fillna(0.0)
    prev = np.zeros(len(w.columns))
    rows = []
    for m in w.index:
        tgt = w.loc[m].to_numpy()
        turn = float(np.abs(tgt - prev).sum()) if rows else float(np.abs(tgt).sum())
        gross = float(tgt @ r.loc[m].to_numpy())
        net = gross - turn * cost
        rows.append((m, net, turn))
        prev = tgt * (1 + r.loc[m].to_numpy()) / (1 + gross) if (1 + gross) > 0 else tgt
    out = pd.DataFrame(rows, columns=["month", "ret", "turnover"]).set_index("month")
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out


# ── 판정 ─────────────────────────────────────────────────────────────────────────────────────

def stats(ret: pd.Series, a: pd.DataFrame, window: tuple[pd.Period, pd.Period], boot: bool = True) -> dict:
    df = pd.concat([ret.rename("s"), a[["MKT", "RF"]]], axis=1).dropna().loc[window[0]:window[1]]
    s_ex, m_ex = (df["s"] - df["RF"]).to_numpy(), (df["MKT"] - df["RF"]).to_numpy()
    ps, pm = fl.perf_m(df["s"]), fl.perf_m(df["MKT"])
    alpha, t_alpha, beta = fl.capm(s_ex, m_ex)
    out = {"start": str(df.index.min()), "end": str(df.index.max()), "months": len(df),
           "Sharpe": fl.sharpe_m(s_ex), "Sharpe_mkt": fl.sharpe_m(m_ex), "dS": fl.sharpe_m(s_ex) - fl.sharpe_m(m_ex),
           "CAGR": ps["CAGR"], "CAGR_mkt": pm["CAGR"], "MDD": ps["MDD"], "MDD_mkt": pm["MDD"], "vol": ps["vol"],
           "alpha": alpha, "t_alpha": t_alpha, "beta": beta}
    if boot:
        b = fl.boot_dsharpe(s_ex, m_ex)
        out.update({"lo": b["lo"], "hi": b["hi"], "p": b["p"]})
    return out


def _registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def register() -> None:
    if _registered():
        raise SystemExit(f"{FAMILY} 은 이미 등록됐다")
    memo = ("3건 p≤0.05/3: R1 Sahm(3개월 평균 실업률−직전 12개월 최저≥0.5, 발표 지연 1개월)이면 방어주(NoDur·Telcm·Utils·Hlth "
            "동일)·아니면 시장 / R2 침체면 방어주·아니면 테크(BusEq) / R3 12산업 직전 6개월 상위 3 동일 비중. 월 1회·편도 0.1%. "
            "① 주 구간(R1·R2 1950-01~, R3 1927-01~ 2025-09) ΔSharpe vs 시장 12개월 블록 부트스트랩 p≤0.05/3 "
            "② 2016-01~2025-09 상한>0 ③ R3 1999-08~ ΔSharpe>0. 정의 scripts/rotation_research.py 머리말")
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": memo}])
    print(f"등록: {FAMILY} {N_TESTS}건 · 누적 시도 {total_trials()}")


def run() -> None:
    if not _registered():
        raise SystemExit("등록 먼저 (--register)")
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    if ((log["family"] == FAMILY) & (log["판정"] != "등록")).any():
        raise SystemExit(f"{FAMILY} 결과가 이미 기록됐다 — 보고서만 다시 쓰려면 --report")
    a = assets()
    rows = []
    for code, (start, desc) in WINDOWS.items():
        sim = simulate(target_weights(code, a), a)
        main = stats(sim["ret"], a, (pd.Period(start, "M"), DEV_MONTH))
        recent = stats(sim["ret"], a, RECENT)
        post = stats(sim["ret"], a, (R3_POST, DEV_MONTH)) if code == "R3" else None
        w = target_weights(code, a).loc[pd.Period(start, "M"):DEV_MONTH]
        g1, g2 = main["p"] <= 0.05 / N_TESTS, recent["hi"] > 0
        g3 = None if post is None else post["dS"] > 0
        verdict = "통과 후보" if g1 and g2 and g3 is not False else "탈락"
        rows.append({"code": code, "설명": desc, "판정": verdict, "gate①": g1, "gate②": g2, "gate③": g3,
                     **{f"main_{k}": v for k, v in main.items()},
                     **{f"recent_{k}": recent[k] for k in ("Sharpe", "Sharpe_mkt", "dS", "lo", "hi", "p", "CAGR",
                                                           "CAGR_mkt", "MDD", "MDD_mkt")},
                     **({f"post_{k}": post[k] for k in ("start", "dS", "p")} if post else {}),
                     "turnover_yr": float(sim["turnover"].loc[pd.Period(start, "M"):].mean() * 12),
                     "def_share": float((w["DEF"] > 0).mean())})
        print(f"[rotation] {code} {verdict}  ΔSharpe {main['dS']:+.2f} [{main['lo']:+.2f}, {main['hi']:+.2f}] "
              f"p {main['p']:.4f} · 최근 {recent['dS']:+.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(TIER3_DIR / "rotation_tests.csv", index=False, encoding="utf-8-sig")
    hindsight(a).to_csv(TIER3_DIR / "rotation_hindsight.csv", index=False, encoding="utf-8-sig")
    recessions(a).to_csv(TIER3_DIR / "rotation_recessions.csv", index=False, encoding="utf-8-sig")
    log_trials(STEP, FAMILY, N_TESTS, [
        {"test_id": r["code"], "t": f"{r['main_dS']:.3f}", "판정": r["판정"],
         "메모": (f"ΔSharpe {r['main_dS']:+.2f} [{r['main_lo']:+.2f}, {r['main_hi']:+.2f}] p {r['main_p']:.4f} · "
                f"최근 {r['recent_dS']:+.2f} [{r['recent_lo']:+.2f}, {r['recent_hi']:+.2f}]"
                + (f" · 발표뒤 {r['post_dS']:+.2f}" if r["code"] == "R3" else ""))} for _, r in res.iterrows()])
    print(f"[rotation] 누적 시도 {total_trials()}", flush=True)
    report()


# ── 참고: 뒷북 전략 · 침체별 표 ───────────────────────────────────────────────────────────────────

def hindsight(a: pd.DataFrame) -> pd.DataFrame:
    """H1 그해 1등 산업 (완벽한 뒷북) · H2 작년 1등 산업 · H3 NBER 침체 달을 미리 안 방어주 · 시장. 비용 없음."""
    yr = (1 + a[list(IND_KO) + ["MKT", "RF"]]).groupby(a.index.year).prod() - 1
    yr = yr.loc[1927:2024]                                  # 꽉 찬 해만
    best = yr[list(IND_KO)].idxmax(axis=1)
    h1 = pd.Series([yr.loc[y, best[y]] for y in yr.index], index=yr.index)
    h2 = pd.Series([yr.loc[y, best[y - 1]] for y in yr.index[1:]], index=yr.index[1:])
    ranks = yr[list(IND_KO)].rank(axis=1, ascending=False)
    repeat_top3 = float(np.mean([ranks.loc[y, best[y - 1]] <= 3 for y in yr.index[1:]]))

    def _cagr(x: pd.Series) -> float:
        return float((1 + x).prod() ** (1 / len(x)) - 1)

    rows = [{"전략": "H1 그해 1등 산업을 1월부터 (완벽한 뒷북)", "기간": "1927~2024", "CAGR": _cagr(h1),
             "시장 CAGR": _cagr(yr["MKT"]), "메모": "실전 불가능 — 기회가 얼마나 있었나"},
            {"전략": "H2 작년 1등 산업을 올해", "기간": "1928~2024", "CAGR": _cagr(h2), "시장 CAGR": _cagr(yr["MKT"].iloc[1:]),
             "메모": f"작년 1등이 올해 12개 중 3등 안에 든 해 {repeat_top3:.0%} (아무렇게나 골라도 25%)"}]
    sim = simulate(target_weights("H3", a), a)
    st = stats(sim["ret"], a, (pd.Period("1950-01", "M"), DEV_MONTH), boot=False)
    rows.append({"전략": "H3 NBER 침체 달을 미리 다 알고 그동안만 방어주", "기간": "1950-01~2025-09", "CAGR": st["CAGR"],
                 "시장 CAGR": st["CAGR_mkt"],
                 "메모": f"Sharpe {st['Sharpe']:.2f} vs 시장 {st['Sharpe_mkt']:.2f} · MDD {st['MDD']:.0%} vs {st['MDD_mkt']:.0%}"})
    return pd.DataFrame(rows)


def recessions(a: pd.DataFrame) -> pd.DataFrame:
    """1950 이후 NBER 침체마다: 시장·방어주·테크 수익, Sahm 신호(보유 기준)가 켜진 달, 시장 바닥 달."""
    rec = load("USREC")["USREC"].loc["1950-01":DEV_MONTH]
    sig = recession_signal(a.index)
    starts = rec.index[(rec == 1) & (rec.shift(1) == 0)]
    rows = []
    for s in starts:
        e = rec.loc[s:].index[rec.loc[s:] == 0][0] - 1      # 침체 마지막 달
        seg = a.loc[s:e]
        lvl = (1 + a["MKT"]).cumprod()
        win = lvl.loc[s - 6:e + 6]                         # 침체 앞뒤 6개월 안의 시장 최저점
        trough = win.idxmin()
        on = sig.loc[s - 6:e + 24]
        onset = (on == True) & (sig.shift(1).reindex(on.index) == False)     # noqa: E712 — 꺼짐 → 켜짐
        first_on = on.index[onset][0] if onset.any() else None
        last_on = None
        if first_on is not None:
            after = sig.loc[first_on:]
            off = after.index[after == False]                                   # noqa: E712
            last_on = (off[0] - 1) if len(off) else None
        rows.append({"침체": f"{s} ~ {e}", "개월": len(seg),
                     "시장": float((1 + seg["MKT"]).prod() - 1), "방어주": float((1 + seg["DEF"]).prod() - 1),
                     "테크": float((1 + seg[TECH]).prod() - 1),
                     "시장 바닥": str(trough), "Sahm 방어주 시작": str(first_on) if first_on is not None else "",
                     "Sahm 방어주 끝": str(last_on) if last_on is not None else "",
                     "신호 시작 − 침체 시작(개월)": (first_on - s).n if first_on is not None else np.nan,
                     "신호 끝 − 시장 바닥(개월)": (last_on - trough).n if last_on is not None else np.nan})
    return pd.DataFrame(rows)


# ── 확인: 실제 섹터 ETF · R3 이웃 (참고) ────────────────────────────────────────────────────────

def fetch_etf() -> None:
    """Select Sector SPDR + SPY 수정 종가(배당 재투자) → 월말 수익 (1998-12 ~ 2025-09) 고정."""
    import yfinance as yf
    path = DATA_DIR / "ETF_M.parquet"
    sums = _sums()
    if path.name in sums:
        if fl.sha256(path) != sums[path.name]:
            raise SystemExit(f"{path.name}: 해시가 다르다 — 고정된 자료가 바뀌었다")
        print("[rotation] ETF_M 고정 확인", flush=True)
        return
    tickers = list(SECTOR_ETFS) + ["SPY"]
    raw = yf.download(tickers, start=ETF_START, end=str((DEV_END + pd.Timedelta(days=1)).date()), auto_adjust=True,
                      progress=False, threads=True)["Close"]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    month_end = raw.groupby(raw.index.to_period("M")).last()
    ret = month_end.pct_change().loc[:DEV_MONTH]
    missing = [t for t in tickers if t not in ret.columns or ret[t].notna().sum() == 0]
    if missing:
        raise SystemExit(f"받지 못한 티커: {missing}")
    ret.to_parquet(path)
    with SUMS.open("a") as fh:
        fh.write(f"{fl.sha256(path)}  {path.name}\n")
    print(f"[rotation] ETF_M: {ret.index.min()} ~ {ret.index.max()} · 첫 수익 달 " +
          ", ".join(f"{t} {ret[t].first_valid_index()}" for t in tickers), flush=True)


def etf_frame() -> pd.DataFrame:
    """섹터 ETF 월간 수익 + MKT(= SPY, 비교 기준) + RF."""
    e = load("ETF_M")
    out = e[list(SECTOR_ETFS)].copy()
    out["MKT"] = e["SPY"]
    out["RF"] = fl.market("US")["rf"]
    return out.dropna(subset=["MKT", "RF"])


def etf_weights(e: pd.DataFrame) -> pd.DataFrame:
    """R3 규칙을 섹터 ETF 에: 직전 6개월 수익이 있는 ETF 중 상위 3 (처음 9개가 다 있어야 시작)."""
    return momentum_weights(e[list(SECTOR_ETFS)], MOM_LOOKBACK, MOM_TOP, min_assets=9)


def _confirm_registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY_CONFIRM) & (log["판정"] == "등록")).any())


def register_confirm() -> None:
    if _confirm_registered():
        raise SystemExit(f"{FAMILY_CONFIRM} 은 이미 등록됐다")
    memo = ("1건 (R3 통과 후보를 본 뒤 등록): C1 R3 규칙(직전 6개월 상위 3, 월 1회, 편도 0.1%)을 Select Sector SPDR "
            "(9개, XLRE·XLC 는 6개월 쌓인 뒤 추가) 수정 종가 월말 수익에. 1999-07~2025-09, 비교 SPY. "
            "확인 = ΔSharpe>0 · 95% 상한>0 (12개월 블록 부트스트랩 1만). 정의 scripts/rotation_research.py 머리말")
    log_trials(STEP, FAMILY_CONFIRM, 1, [{"test_id": "(등록)", "판정": "등록", "메모": memo}])
    print(f"등록: {FAMILY_CONFIRM} 1건 · 누적 시도 {total_trials()}")


def confirm() -> None:
    if not _confirm_registered():
        raise SystemExit("등록 먼저 (--register-confirm)")
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    if ((log["family"] == FAMILY_CONFIRM) & (log["판정"] != "등록")).any():
        raise SystemExit(f"{FAMILY_CONFIRM} 결과가 이미 기록됐다 — 보고서만 다시 쓰려면 --report")
    e = etf_frame()
    w = etf_weights(e)
    sim = simulate(w, e)
    main = stats(sim["ret"], e, (CONFIRM_START, DEV_MONTH))
    recent = stats(sim["ret"], e, RECENT)
    verdict = "확인" if main["dS"] > 0 and main["hi"] > 0 else "확인 안 됨"
    held = (w.loc[CONFIRM_START:] > 0).mean().sort_values(ascending=False)
    yr = pd.concat([sim["ret"].rename("전략"), e["MKT"].rename("SPY")], axis=1).dropna().loc[CONFIRM_START:]
    yearly = ((1 + yr).groupby(yr.index.year).prod() - 1)
    out = {"code": "C1", "판정": verdict, **{f"main_{k}": v for k, v in main.items()},
           **{f"recent_{k}": recent[k] for k in ("Sharpe", "Sharpe_mkt", "dS", "lo", "hi", "p", "CAGR", "CAGR_mkt")},
           "turnover_yr": float(sim["turnover"].loc[CONFIRM_START:].mean() * 12),
           "보유비율": " · ".join(f"{t} {v:.0%}" for t, v in held.items() if v > 0)}
    pd.DataFrame([out]).to_csv(TIER3_DIR / "rotation_confirm.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(TIER3_DIR / "rotation_confirm_yearly.csv", encoding="utf-8-sig")
    r3_reference(assets()).to_csv(TIER3_DIR / "rotation_r3_neighbors.csv", index=False, encoding="utf-8-sig")
    log_trials(STEP, FAMILY_CONFIRM, 1, [{"test_id": "C1", "t": f"{main['dS']:.3f}", "판정": verdict,
                                          "메모": (f"ΔSharpe {main['dS']:+.2f} [{main['lo']:+.2f}, {main['hi']:+.2f}] "
                                                 f"p {main['p']:.4f} · 2016~ {recent['dS']:+.2f}")}])
    print(f"[rotation] C1 {verdict}  ΔSharpe {main['dS']:+.2f} [{main['lo']:+.2f}, {main['hi']:+.2f}] p {main['p']:.4f} · "
          f"최근 {recent['dS']:+.2f} · 누적 시도 {total_trials()}", flush=True)
    report()


def r3_reference(a: pd.DataFrame) -> pd.DataFrame:
    """R3 이웃 (직전 개월 × 상위 개수) · 옛 비용 — 1927~, 1999-08~, 2016~ ΔSharpe (참고, 부트스트랩 없음)."""
    rows = []
    for spec in [(lb, k, COST) for lb, k in NEIGHBORS] + [(MOM_LOOKBACK, MOM_TOP, OLD_COST)]:
        lb, k, cost = spec
        sim = simulate(target_weights("R3", a, lookback=lb, top=k), a, cost=cost)
        row = {"직전개월": lb, "상위": k, "편도비용": cost}
        for name, win in (("1927~", (pd.Period("1927-01", "M"), DEV_MONTH)), ("1999-08~", (R3_POST, DEV_MONTH)),
                          ("2016~", RECENT)):
            st = stats(sim["ret"], a, win, boot=False)
            row[f"dS {name}"] = st["dS"]
        rows.append(row)
    sim = simulate(target_weights("R3", a), a)
    df = pd.concat([sim["ret"].rename("s"), a[["MKT", "RF"]]], axis=1).dropna()
    for dec, g in df.groupby(df.index.year // 10 * 10):
        rows.append({"직전개월": MOM_LOOKBACK, "상위": MOM_TOP, "편도비용": COST, "10년대": f"{dec}s",
                     "dS 10년대": fl.sharpe_m((g["s"] - g["RF"]).to_numpy()) - fl.sharpe_m((g["MKT"] - g["RF"]).to_numpy())})
    return pd.DataFrame(rows)


# ── 보고서 ─────────────────────────────────────────────────────────────────────────────────────

def _p(v: float, signed: bool = False) -> str:
    return "" if pd.isna(v) else (f"{v:+.1%}" if signed else f"{v:.1%}")


def report() -> None:
    res = pd.read_csv(TIER3_DIR / "rotation_tests.csv", encoding="utf-8-sig")
    hs = pd.read_csv(TIER3_DIR / "rotation_hindsight.csv", encoding="utf-8-sig")
    rc = pd.read_csv(TIER3_DIR / "rotation_recessions.csv", encoding="utf-8-sig")
    lines = [
        "# 경기·자금 흐름 따라 섹터 갈아타기 (rotation_v1)", "",
        f"- 실행 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 정의·사전 등록은 `scripts/rotation_research.py` 머리말 · 누적 시도 {total_trials()}건",
        "- 자료: Kenneth French 12 산업 월간 (CRSP 전 종목, 상장폐지 포함), FRED 실업률(UNRATE)·NBER 침체(USREC). 2025-09 에서 자름.",
        "- 월 1회 매매, 편도 0.1%. Sharpe = 1개월 국채 초과수익 기준. ΔSharpe 95% 구간·p = 12개월 블록 부트스트랩 1만 회.", "",
        "## 쉬운 말 요약", "", *(SUMMARY or ["(실행 뒤 채움)"]), "",
        "## 판정 (p ≤ 0.05/3 = 0.0167)", "",
        "| 전략 | 구간 | Sharpe (시장) | ΔSharpe [95%] · p | CAGR (시장) | MDD (시장) | 방어주 보유 비율 | 회전/년 | "
        "② 2016~ Δ [95%] | ③ 발표 뒤 Δ | 판정 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        post = f"{r['post_dS']:+.2f}" if "post_dS" in r and pd.notna(r.get("post_dS")) else "—"
        lines.append(
            f"| {r['code']} {r['설명']} | {r['main_start']}~ | {r['main_Sharpe']:.2f} ({r['main_Sharpe_mkt']:.2f}) | "
            f"{r['main_dS']:+.2f} [{r['main_lo']:+.2f}, {r['main_hi']:+.2f}] · {r['main_p']:.3f} | "
            f"{_p(r['main_CAGR'])} ({_p(r['main_CAGR_mkt'])}) | {_p(r['main_MDD'])} ({_p(r['main_MDD_mkt'])}) | "
            f"{_p(r['def_share'])} | {r['turnover_yr']:.1f} | {r['recent_dS']:+.2f} [{r['recent_lo']:+.2f}, "
            f"{r['recent_hi']:+.2f}] | {post} | **{r['판정']}** |")
    cf_path = TIER3_DIR / "rotation_confirm.csv"
    if cf_path.exists():
        cf = pd.read_csv(cf_path, encoding="utf-8-sig").iloc[0]
        lines += ["", "## 확인: 실제 섹터 ETF 로 1999~2025 (family `rotation_confirm_v1`)", "",
                  "| 전략 | 구간 | Sharpe (SPY) | ΔSharpe [95%] · p | CAGR (SPY) | MDD (SPY) | 회전/년 | 2016~ Δ [95%] | 판정 |",
                  "|---|---|---|---|---|---|---|---|---|",
                  f"| C1 섹터 ETF 직전 6개월 상위 3 | {cf['main_start']}~ | {cf['main_Sharpe']:.2f} ({cf['main_Sharpe_mkt']:.2f}) | "
                  f"{cf['main_dS']:+.2f} [{cf['main_lo']:+.2f}, {cf['main_hi']:+.2f}] · {cf['main_p']:.3f} | "
                  f"{_p(cf['main_CAGR'])} ({_p(cf['main_CAGR_mkt'])}) | {_p(cf['main_MDD'])} ({_p(cf['main_MDD_mkt'])}) | "
                  f"{cf['turnover_yr']:.1f} | {cf['recent_dS']:+.2f} [{cf['recent_lo']:+.2f}, {cf['recent_hi']:+.2f}] | "
                  f"**{cf['판정']}** |", "", f"- 보유 비율: {cf['보유비율']}"]
        yr_path = TIER3_DIR / "rotation_confirm_yearly.csv"
        if yr_path.exists():
            yr = pd.read_csv(yr_path, encoding="utf-8-sig", index_col=0)
            lines += ["", "| 연도 | " + " | ".join(str(i) for i in yr.index) + " |", "|---|" + "---|" * len(yr),
                      "| 섹터 ETF | " + " | ".join(_p(v, True) for v in yr["전략"]) + " |",
                      "| SPY | " + " | ".join(_p(v, True) for v in yr["SPY"]) + " |"]
    nb_path = TIER3_DIR / "rotation_r3_neighbors.csv"
    if nb_path.exists():
        nb = pd.read_csv(nb_path, encoding="utf-8-sig")
        grid = nb[nb["10년대"].isna()] if "10년대" in nb.columns else nb
        lines += ["", "## 참고: R3 이웃 · 옛 비용 · 10년대별 (판정 아님, 시장 대비 ΔSharpe)", "",
                  "| 직전 개월 | 상위 | 편도 비용 | 1927~2025 | 1999-08~ | 2016~ |", "|---|---|---|---|---|---|"]
        lines += [f"| {int(r['직전개월'])} | {int(r['상위'])} | {r['편도비용']:.1%} | {r['dS 1927~']:+.2f} | "
                  f"{r['dS 1999-08~']:+.2f} | {r['dS 2016~']:+.2f} |" for _, r in grid.iterrows()]
        if "10년대" in nb.columns:
            dec = nb.dropna(subset=["10년대"])
            lines += ["", "| " + " | ".join(dec["10년대"]) + " |", "|" + "---|" * len(dec),
                      "| " + " | ".join(f"{v:+.2f}" for v in dec["dS 10년대"]) + " |"]
    lines += ["", "## 참고: 뒷북으로 보면 (실전에서는 불가능)", "", "| 전략 | 기간 | CAGR | 시장 CAGR | 메모 |", "|---|---|---|---|---|"]
    lines += [f"| {r['전략']} | {r['기간']} | {_p(r['CAGR'])} | {_p(r['시장 CAGR'])} | {r['메모']} |" for _, r in hs.iterrows()]
    lines += ["", "## 참고: 1950 이후 경기침체마다 (NBER 날짜는 몇 달~1년 뒤에야 발표된다)", "",
              "| 침체 | 개월 | 시장 | 방어주 | 테크 | 시장 바닥 | Sahm 신호로 방어주 보유 | 신호 시작 − 침체 시작 | 신호 끝 − 시장 바닥 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in rc.iterrows():
        held = f"{r['Sahm 방어주 시작']} ~ {r['Sahm 방어주 끝']}" if isinstance(r["Sahm 방어주 시작"], str) and r["Sahm 방어주 시작"] else "신호 없음"
        d1 = "" if pd.isna(r["신호 시작 − 침체 시작(개월)"]) else f"{int(r['신호 시작 − 침체 시작(개월)']):+d}개월"
        d2 = "" if pd.isna(r["신호 끝 − 시장 바닥(개월)"]) else f"{int(r['신호 끝 − 시장 바닥(개월)']):+d}개월"
        lines.append(f"| {r['침체']} | {r['개월']} | {_p(r['시장'], True)} | {_p(r['방어주'], True)} | {_p(r['테크'], True)} | "
                     f"{r['시장 바닥']} | {held} | {d1} | {d2} |")
    lines += ["", "## 한계", "",
              "- 실업률은 지금 받을 수 있는 수정치다 (발표 당시 값과 조금 다르다). Sahm 규칙 자체가 2019년에 나온 규칙이라 그 전은 표본 내다.",
              "- 산업 12개는 넓은 묶음이다. 실제 섹터 ETF(1998~)·개별 종목은 이와 다르게 움직일 수 있다.",
              "- 비용은 편도 0.1% (요즘 섹터 ETF 기준). 1990년대 전에는 더 컸다 — 판정을 바꿀 만큼은 아니다 (회전/년 참고)."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rotation] 보고서: {REPORT.relative_to(ROOT)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--fetch", "--register", "--run", "--report", "--fetch-etf", "--register-confirm", "--confirm"):
        g.add_argument(flag, action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    elif args.register:
        register()
    elif args.run:
        run()
    elif args.fetch_etf:
        fetch_etf()
    elif args.register_confirm:
        register_confirm()
    elif args.confirm:
        confirm()
    else:
        report()


if __name__ == "__main__":
    main()
