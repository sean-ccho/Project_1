#!/usr/bin/env python3
"""촉매(사건) 신호 2개 — 사전 등록 시험 (2026-11-16, 한 번만 실행). 예측 모델 v3(파트너 · PM) 설계의 첫 시험.

v3 정의 (2026-11-16 사용자와 합의): v1(주니어) · v2(시니어) 후보 위에서 '무엇을 · 얼마나 · 왜 지금' 을 정하는 PM 단계.
성공 기준은 '더 엄격함' 이 아니라 결과 — v2 Top 10 과 SPY(배당 포함)보다 위험 대비 수익이 높을 것.
'더 싼 순서' 만으로는 SPY 를 못 이겼다 (value_v3_recheck · published_rules_v1) → PM 단계가 더할 후보는 촉매(제값을 받을 사건).
이 시험은 촉매 신호 자체에 효과가 있는지 먼저 본다:

  ACT   행동주의 펀드 13D (Brav · Jiang · Partnoy · Thomas 2008) — ACTIVISTS 목록의 펀드가 처음 낸 SC 13D
        (2024-12 부터 이름이 'SCHEDULE 13D') → 공시일 다음 날이 속한 달의 말에 사서 12개월 보유
  SPIN  스핀오프 (Cusatis · Miles · Woolridge 1993, Greenblatt 1997) — Form 10-12B 로 등록하고 새로 거래를 시작한 회사 →
        거래 시작 다음 달 말에 사서 24개월 보유 (처음 몇 주의 강제 매도는 건너뜀)

사건은 SEC EDGAR 분기 전체 색인(full-index master.gz)에서 찾는다 — 13D 는 한 건이 대상 회사와 낸 쪽 두 줄로 실린다.
상장폐지 · 인수된 회사도 사건 목록엔 남는다(사건 찾기엔 생존 편향 없음). 가격은 지금 상장된 종목만 있다 (Yahoo 는 인수된
종목 기록을 안 주고, 무료 대안 stooq 는 자동 접근을 막는다 — 2026-11-16 확인). 사건의 절반이 그 뒤 인수 · 폐지돼 빠진다.
→ 생존 회사끼리 비교한다 (사용자 11-16: "무료로, 상폐된 애들은 최대한 거르도록"): 판정의 비교 기준은 SPY 가 아니라
   같은 생존 회사 집단 — 같은 달 · 같은 보유 기간 무작위 종목(플라시보)과 그 집단 전체 같은 비중(EW). SPY 대비는 참고.
   남는 편향: 13D 대상은 보유 중 인수(= 프리미엄) 확률이 보통 회사보다 높아 생존만 남기면 좋은 사례를 더 잃는다 (신호에
   불리) — 대상과 S&P 500 보통 회사의 '12개월 안 인수 서식(DEFM14A · SC 14D9 · SC TO-T) 비율' 을 같이 보고한다.
   그래서 통과하면 믿을 만하고, 실패는 데이터 한계일 수 있다.
매달 보유 중인 종목을 같은 비중 (보유 종목이 없는 달은 SPY), 거래비용 편도 0.1%, 2014-06 ~ 2026-10 월간.
대상: 산 날 시가총액 US$10억 이상(value_v3_backtest 와 같은 방법), 금융 · 리츠 제외 (value_model.excluded).
판정 기준은 실행 전에 docs/quant_improvement/tier3/TRIAL_LOG.csv 에 등록 (catalyst_v1). 결과를 보고 고치지 않는다.

  fetch-index  EDGAR 분기 색인 → 필요한 서식만 data/research/value_v3/edgar_index.json.gz (SEC_USER_AGENT 필요)
  coverage     사건 수 · 가격이 있는 비율 · 빠진 대상 중 인수로 보이는 수 (수익률 계산 없음 — 등록 전 확인용)
  run          한 번 실행 → data/research/value_v3/catalyst_results.json · catalyst_report.md

사용: SEC_USER_AGENT="이름 이메일" PYTHONPATH=.:src python scripts/catalyst_backtest.py fetch-index
      PYTHONPATH=.:src python scripts/catalyst_backtest.py coverage | run
연구 전용 — 점수 · 필터 · PT-1 · 사이트와 무관.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import re
import sys
import warnings
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import value_v3_backtest as vb  # noqa: E402
from paper_trading.evaluation import sharpe  # noqa: E402
from web_export.value_model import excluded  # noqa: E402

warnings.filterwarnings("ignore", message="Sorting by default when concatenating")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

OUT = vb.OUT
INDEX_FILE = OUT / "edgar_index.json.gz"
INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/master.gz"
INDEX_FROM = 2012                       # SPIN 24개월 보유가 2014-06 에 걸리려면 2012 년 사건부터
FORMS_13D = ("SC 13D", "SCHEDULE 13D")  # 처음 낸 것만 (/A 정정 제외). 2024-12-18 부터 EDGAR 서식 이름이 바뀜
FORMS_SPIN = ("10-12B", "10-12B/A")
FORMS_EXIT = ("DEFM14A", "SC 14D9", "SC TO-T", "SC14D9C", "15-12B", "25-NSE")   # 빠진 대상 진단용
FORMS_ACQ = ("DEFM14A", "SC 14D9", "SC TO-T", "SC14D9C")

START, END = "2014-05-31", "2026-10-31"  # 다른 시험과 같은 기간 (월간 2014-06 ~ 2026-10)
HOLD = {"ACT": 12, "SPIN": 24}
MCAP_MIN = 1e9
COST = vb.COST
N_PLACEBO = 1000
TESTS = ("ACT", "SPIN")

# 행동주의 펀드 (2013~2025 행동주의 순위표에 꾸준히 나온 곳 · 사전 등록 목록) — EDGAR 이름(대소문자 무시)에 맞춘다.
# 수동형(GAMCO · D. E. Shaw · Glenview · Greenlight 등)은 13D 를 내도 캠페인이 아니라 뺐다. 'THIRD POINT RE(보험사)' 는 제외.
ACTIVISTS: tuple[tuple[str, str], ...] = (
    ("Elliott", r"\bELLIOTT (ASSOCIATES|INTERNATIONAL|INVESTMENT MANAGEMENT|MANAGEMENT CORP|CAPITAL ADVISORS)|\bSINGER,? PAUL"),
    ("Starboard", r"\bSTARBOARD VALUE"),
    ("Icahn", r"\bICAHN\b"),
    ("Pershing Square", r"\bPERSHING SQUARE"),
    ("Trian", r"\bTRIAN (FUND|PARTNERS)|\bPELTZ,? NELSON"),
    ("ValueAct", r"\bVALUEACT\b"),
    ("Third Point", r"\bTHIRD POINT (LLC|PARTNERS|OFFSHORE|ULTRA|MASTER)"),
    ("JANA", r"\bJANA (PARTNERS|MASTER|NIRVANA)"),
    ("Engaged", r"\bENGAGED CAPITAL"),
    ("Sachem Head", r"\bSACHEM HEAD"),
    ("Corvex", r"\bCORVEX\b"),
    ("Land & Buildings", r"\bLAND (&|AND) BUILDINGS"),
    ("Legion", r"\bLEGION PARTNERS"),
    ("Ancora", r"\bANCORA (ADVISORS|HOLDINGS|ALTERNATIVES|CATALYST|MERLIN|BELLATOR)"),
    ("Blue Harbour", r"\bBLUE HARBOU?R GROUP"),
    ("Marcato", r"\bMARCATO\b"),
    ("Relational", r"\bRELATIONAL INVESTORS"),
    ("Barington", r"\bBARINGTON\b"),
    ("Cevian", r"\bCEVIAN\b"),
    ("Mantle Ridge", r"\bMANTLE RIDGE"),
    ("Politan", r"\bPOLITAN\b"),
    ("Irenic", r"\bIRENIC\b"),
    ("Engine", r"\bENGINE CAPITAL"),
    ("Impactive", r"\bIMPACTIVE\b"),
    ("Blackwells", r"\bBLACKWELLS\b"),
    ("FrontFour", r"\bFRONTFOUR\b"),
    ("Voce", r"\bVOCE CAPITAL"),
    ("Scopia", r"\bSCOPIA\b"),
    ("Sandell", r"\bSANDELL\b"),
    ("Biglari", r"\bBIGLARI\b"),
    ("Macellum", r"\bMACELLUM\b"),
    ("Lion Point", r"\bLION POINT\b"),
    ("Clinton", r"\bCLINTON GROUP"),
    ("Inclusive", r"\bINCLUSIVE CAPITAL"),
    ("Carronade", r"\bCARRONADE\b"),
)
_ACT_RE = [(name, re.compile(pat, re.I)) for name, pat in ACTIVISTS]


def activist_of(name: str) -> str | None:
    """EDGAR 이름 → 행동주의 펀드 이름 (목록에 없으면 None)."""
    for label, rx in _ACT_RE:
        if rx.search(name or ""):
            return label
    return None


# ── EDGAR 색인 받기

def parse_master(text: str, keep: Iterable[str]) -> list[list[Any]]:
    """master.idx → [[cik, 이름, 서식, 날짜, 접수번호]] (keep 서식만)."""
    want = set(keep)
    out = []
    for line in text.splitlines():
        if line.count("|") != 4:
            continue
        cik, name, form, d, fn = line.split("|")
        if form in want and cik.isdigit():
            out.append([int(cik), name.strip(), form, d, fn.rsplit("/", 1)[-1].removesuffix(".txt")])
    return out


def cmd_fetch_index(_args: argparse.Namespace) -> int:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        sys.exit("환경변수 SEC_USER_AGENT='이름 이메일' 이 필요합니다 (SEC 규칙)")
    cli = vb.SecClient(ua)
    rows: list[list[Any]] = []
    today = date.today()
    for y in range(INDEX_FROM, today.year + 1):
        for q in range(1, 5):
            if (y, q) > (today.year, (today.month - 1) // 3 + 1):
                break
            st, body = cli.get(INDEX_URL.format(y=y, q=q))
            if st != 200 or not body:
                vb.log(f"  {y}Q{q}: 못 받음 ({st})")
                continue
            got = parse_master(gzip.decompress(body).decode("latin-1"), FORMS_13D + FORMS_SPIN + FORMS_EXIT)
            rows.extend(got)
            vb.log(f"  {y}Q{q}: {len(got)}줄")
    INDEX_FILE.write_bytes(gzip.compress(json.dumps(rows).encode("utf-8")))
    vb.log(f"색인 저장: {len(rows)}줄")
    return 0


# ── 사건 만들기

def act_events(rows: Sequence[Sequence[Any]], known: set[int] | None = None) -> list[dict[str, Any]]:
    """처음 낸 13D 중 낸 쪽이 ACTIVISTS 인 것 → [{cik(대상), name, date, activist, acc}]. 한 접수번호의 줄 중
    목록에 맞는 쪽이 낸 쪽, 나머지가 대상. 나머지가 둘 이상이면(공동 신고) known(우리 회사 목록 CIK)에 있는 하나를 대상으로,
    그래도 하나로 안 정해지면 뺀다. 같은 대상 · 같은 펀드 · 같은 달은 한 번만."""
    by: dict[str, list[Sequence[Any]]] = defaultdict(list)
    for r in rows:
        if r[2] in FORMS_13D:
            by[r[4]].append(r)
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for acc, ents in by.items():
        acts = [(activist_of(e[1]), e) for e in ents]
        hit = [a for a, _ in acts if a]
        subj = {int(e[0]): e for a, e in acts if not a}
        if len(subj) > 1 and known is not None:
            subj = {c: e for c, e in subj.items() if c in known}
        if not hit or len(subj) != 1:
            continue
        s = next(iter(subj.values()))
        d = min(e[3] for e in ents)
        key = (int(s[0]), hit[0], d[:7])
        if key not in out or d < out[key]["date"]:
            out[key] = {"cik": int(s[0]), "name": s[1], "date": d, "activist": hit[0], "acc": acc}
    return sorted(out.values(), key=lambda e: (e["date"], e["cik"]))


def spin_filings(rows: Sequence[Sequence[Any]]) -> dict[int, dict[str, Any]]:
    """Form 10-12B 를 처음 낸 날 (CIK 마다)."""
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        if r[2] in FORMS_SPIN:
            cur = out.get(int(r[0]))
            if cur is None or r[3] < cur["date"]:
                out[int(r[0])] = {"cik": int(r[0]), "name": r[1], "date": r[3]}
    return out


def entry_month_act(filed: str) -> pd.Timestamp:
    """공시일 다음 날이 속한 달의 말 — 그날 종가에 산다 (말일에 낸 공시는 다음 달 말)."""
    return vb.month_end(date.fromisoformat(filed) + timedelta(days=1))


def exits_by_cik(rows: Sequence[Sequence[Any]]) -> dict[int, list[tuple[str, str]]]:
    out: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for r in rows:
        if r[2] in FORMS_EXIT:
            out[int(r[0])].append((r[3], r[2]))
    return out


def acq_within(exits: Mapping[int, Sequence[tuple[str, str]]], cik: int, start: str, months: int) -> bool:
    """start 부터 months 개월 안에 인수 서식(DEFM14A · SC 14D9 · SC TO-T)이 나왔는지."""
    end = (pd.Timestamp(start) + pd.DateOffset(months=months)).date().isoformat()
    return any(f in FORMS_ACQ and start <= d <= end for d, f in exits.get(int(cik), []))


def base_acq_rate(rows: Sequence[Sequence[Any]], uni: "Universe", months: int = 12) -> tuple[int, int]:
    """보통 대형주의 기준 비율 — 해마다 5/31 S&P 500 구성종목(금융 · 리츠 제외) 중 다음 months 개월 안에 인수 서식이 나온 수."""
    exits = exits_by_cik(rows)
    cands, sm = uni.inp["cands"], uni.inp["sector_map"]
    tick2cik = {t: c["cik"] for t, c in cands.items() if c.get("cik")}
    k = n = 0
    for y in range(2013, 2025):
        d = f"{y}-05-31"
        for t in vb.members_on(uni.inp["mem"], d):
            cik = tick2cik.get(t)
            sec, ind = sm.get(t, (None, None))
            if not cik or (sec and excluded({"sector": sec, "industry": ind})):
                continue
            n += 1
            k += acq_within(exits, cik, d, months)
    return k, n


# ── 회사 · 가격 · 그날 시가총액

class Universe:
    """value_v3_backtest 의 회사 목록(지금 상장 미국 보통주 + 옛 S&P 500, SEC 재무 · 가격 있음)과 그날 시가총액."""

    def __init__(self, inp: Mapping[str, Any]) -> None:
        self.inp = inp
        self.px: Mapping[str, pd.DataFrame] = inp["px"]
        facts, cands, scan = inp["facts"], inp["cands"], inp["scan"]
        by_cik: dict[int, dict[str, Any]] = {}
        for t, c in cands.items():
            if not c.get("cik") or not c.get("px") or c["px"] not in self.px:
                continue
            if c["cik"] not in facts and (c.get("alt") or 0) not in facts:
                continue
            cur = by_cik.get(c["cik"])
            mc = (scan.get(c["px"]) or {}).get("mcap_now") or 0.0
            if cur is None or (c["scan"] and mc > cur["mcap_now"]):
                by_cik[c["cik"]] = {"px": c["px"], "mcap_now": mc, "scan": c["scan"],
                                    "alt": c.get("alt") or (cur or {}).get("alt")}
        self.by_cik = by_cik
        self.alt = {int(m["alt"]): cik for cik, m in by_cik.items() if m.get("alt")}
        self.cos = {cik: vb.merge_facts(facts.get(cik), facts.get(m["alt"]) if m.get("alt") else None)
                    for cik, m in by_cik.items()}
        self.now = date.today().isoformat()
        self._mcap: dict[tuple[int, str], float] = {}
        self._pool: dict[str, list[int]] = {}
        self._sec: dict[int, bool] = {}

    def resolve(self, cik: int) -> int | None:
        """사건의 CIK → 목록의 CIK (지주회사 재편으로 바뀐 옛 CIK 포함). 없으면 None (가격 없음)."""
        return cik if cik in self.by_cik else self.alt.get(cik)

    def sector(self, cik: int) -> tuple[str | None, str | None]:
        t = self.by_cik[cik]["px"]
        scan, sm = self.inp["scan"], self.inp["sector_map"]
        return ((scan.get(t) or {}).get("sector") or sm.get(t, (None, None))[0],
                (scan.get(t) or {}).get("industry") or sm.get(t, (None, None))[1])

    def mcap_usd(self, cik: int, m: pd.Timestamp) -> float:
        """그달 말 시가총액 (USD) = 지금 시가총액 × 분할만 반영한 가격비 × 발행주식수 비 — value_v3_backtest 와 같다."""
        key = (cik, str(m.date()))
        if key in self._mcap:
            return self._mcap[key]
        meta = self.by_cik[cik]
        df = self.px[meta["px"]]
        p0, p_now = vb.px_at(df, m), float(df["close"].iloc[-1])
        val = math.nan
        if meta["mcap_now"] and math.isfinite(p0) and math.isfinite(p_now) and p_now > 0:
            ratio, _ = vb.share_ratio(self.cos[cik], df, str(m.date()), self.now)
            val = meta["mcap_now"] * (p0 / p_now) * ratio
        self._mcap[key] = val
        return val

    def eligible(self, cik: int, m: pd.Timestamp) -> bool:
        """산 달 기준: 금융 · 리츠가 아니고, 배당 포함 가격이 있고, 시가총액 US$10억 이상."""
        if cik not in self._sec:
            sec, ind = self.sector(cik)
            self._sec[cik] = bool(sec) and not excluded({"sector": sec, "industry": ind})
        if not self._sec[cik]:
            return False
        if not math.isfinite(vb.px_at(self.px[self.by_cik[cik]["px"]], m, "adj")):
            return False
        mc = self.mcap_usd(cik, m)
        return math.isfinite(mc) and mc >= MCAP_MIN

    def pool(self, m: pd.Timestamp) -> list[int]:
        """그달의 생존 회사 집단 (eligible 전부) — 플라시보 · EW 기준이 여기서 나온다."""
        key = str(pd.Timestamp(m).date())
        if key not in self._pool:
            self._pool[key] = [cik for cik in self.by_cik if self.eligible(cik, pd.Timestamp(m))]
        return self._pool[key]


def build_positions(test: str, rows: Sequence[Sequence[Any]], uni: Universe) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """사건 → 산 포지션 [{cik, ticker, entry, hold, ...}] + 진단(사건 수 · 가격 없음 · 인수로 보임 · 대상 밖)."""
    exits = exits_by_cik(rows)
    start, end = vb.month_end(START), vb.month_end(END)
    diag = {"events": 0, "no_price": 0, "no_price_acq": 0, "no_price_exit": 0, "not_eligible": 0, "used": 0,
            "not_spin": 0, "acq_in": 0, "no_price_acq_in": 0, "by_year": {}}
    out: list[dict[str, Any]] = []
    if test == "ACT":
        known = set(uni.by_cik) | set(uni.alt)
        evs = [dict(e, entry=entry_month_act(e["date"])) for e in act_events(rows, known)]
    else:
        evs = []
        for f in spin_filings(rows).values():
            cik = uni.resolve(f["cik"])
            if cik is None:
                evs.append(dict(f, entry=None))
                continue
            df = uni.px[uni.by_cik[cik]["px"]]
            first = df.index[df["close"].notna()].min()
            if not (vb.month_end(f["date"]) <= first <= vb.month_end(f["date"]) + pd.DateOffset(months=18)):
                diag["not_spin"] += 1  # 서식 전부터 거래되던 회사(장외 → 거래소 이전 등) · 18개월 넘게 상장 안 함
                continue
            evs.append(dict(f, entry=vb.month_end(first + pd.DateOffset(months=1)), first_trade=str(first.date())))
    for e in evs:
        entry = e["entry"]
        last = None if entry is None else entry + pd.DateOffset(months=HOLD[test])
        # 보유 기간이 시험 기간과 겹치는 사건만 (SPIN 은 가격이 없으면 거래 시작 달을 몰라 서식 날짜로 센다)
        ref = vb.month_end(e["date"]) if entry is None else entry
        if (entry is not None and (vb.month_end(last) <= start or entry >= end)) or \
                (entry is None and not (start - pd.DateOffset(months=HOLD[test]) < ref < end)):
            continue
        diag["events"] += 1
        y = e["date"][:4]
        diag["by_year"].setdefault(y, {"events": 0, "used": 0})
        diag["by_year"][y]["events"] += 1
        # 보유 기간 안 인수 서식 (SPIN 은 가격이 없으면 거래 시작을 몰라 서식 날짜부터 보유 + 6개월)
        acq_in = acq_within(exits, e["cik"], e["date"], HOLD[test] + (6 if test == "SPIN" else 0))
        diag["acq_in"] += acq_in
        cik = uni.resolve(e["cik"])
        if cik is None or entry is None:
            diag["no_price"] += 1
            diag["no_price_acq_in"] += acq_in
            later = [f for d, f in exits.get(int(e["cik"]), []) if d >= e["date"]]
            if any(f in FORMS_ACQ for f in later):
                diag["no_price_acq"] += 1
            elif later:
                diag["no_price_exit"] += 1
            continue
        if not uni.eligible(cik, entry):
            diag["not_eligible"] += 1
            continue
        diag["used"] += 1
        diag["by_year"][y]["used"] += 1
        out.append({"cik": cik, "ticker": uni.by_cik[cik]["px"], "entry": entry, "hold": HOLD[test],
                    "date": e["date"], "who": e.get("activist") or e.get("name")})
    return out, diag


# ── 월간 자본곡선 (보유 중인 종목 같은 비중, 매달 다시 맞춤)

def monthly_returns(df: pd.DataFrame | None, months: Sequence[pd.Timestamp]) -> np.ndarray:
    """월말 배당 포함 가격 → 그달 수익률. 가격이 끊기면 그 뒤는 0 (현금화 — value_v3_backtest.period_paths 와 같은 셈)."""
    out = np.zeros(len(months))
    if df is None:
        return out
    prev = None
    for i, m in enumerate(months):
        p = vb.px_at(df, m, "adj")
        if prev is not None and math.isfinite(p):
            out[i] = p / prev - 1
        if math.isfinite(p):
            prev = p          # 끊긴 달은 0 이고, 다시 나오면 마지막 가격부터 잰다
    return out


def window_return(p: Mapping[str, Any], rets: Mapping[str, np.ndarray], months: Sequence[pd.Timestamp]) -> float:
    """한 포지션의 보유 기간 수익률 (시험 기간 안쪽만)."""
    e = pd.Timestamp(p["entry"])
    last = vb.month_end(e + pd.DateOffset(months=int(p["hold"])))
    r = rets[p["ticker"]]
    return float(np.prod([1 + r[i] for i, m in enumerate(months) if e < m <= last]) - 1)


def calendar_portfolio(positions: Sequence[Mapping[str, Any]], rets: Mapping[str, np.ndarray],
                       months: Sequence[pd.Timestamp], spy: np.ndarray, cost: float = COST
                       ) -> tuple[np.ndarray, np.ndarray]:
    """→ (월 수익률, 달마다 보유 종목 수). 포지션은 entry 다음 달부터 hold 개월 수익을 받는다. 같은 종목이 겹치면 한 번만.
    보유 종목이 없는 달은 SPY. 비용 = cost × (목표 비중 − 흘러간 비중의 차 합)."""
    idx = {m: i for i, m in enumerate(months)}
    active: list[set[str]] = [set() for _ in months]
    for p in positions:
        e = pd.Timestamp(p["entry"])
        for k in range(1, int(p["hold"]) + 1):
            m = vb.month_end(e + pd.DateOffset(months=k))
            if m in idx:
                active[idx[m]].add(p["ticker"])
    r = np.zeros(len(months))
    n = np.zeros(len(months), dtype=int)
    prev_w: dict[str, float] = {}
    for i in range(len(months)):
        names = sorted(active[i])
        target = {t: 1.0 / len(names) for t in names} if names else {"SPY": 1.0}
        turn = sum(abs(target.get(t, 0.0) - prev_w.get(t, 0.0)) for t in set(target) | set(prev_w))
        gross = {t: (rets[t][i] if t != "SPY" else spy[i]) for t in target}
        g = sum(w * gross[t] for t, w in target.items())
        r[i] = (1 + g) * (1 - cost * turn) - 1
        n[i] = len(names)
        tot = sum(w * (1 + gross[t]) for t, w in target.items())
        prev_w = {t: w * (1 + gross[t]) / tot for t, w in target.items()} if tot > 0 else {}
    return r, n


# ── 실행

def load_all() -> tuple[Universe, list[list[Any]], list[pd.Timestamp], np.ndarray]:
    inp = vb.load_inputs()
    uni = Universe(inp)
    rows = json.loads(gzip.decompress(INDEX_FILE.read_bytes()))
    spy = inp["px"]["SPY"]
    months = [m for m in spy.index if vb.month_end(START) < m <= vb.month_end(END)]
    spy_r = monthly_returns(spy, [vb.month_end(START)] + months)[1:]
    return uni, rows, months, spy_r


def cmd_coverage(_args: argparse.Namespace) -> int:
    """사건 수 · 가격 있는 비율 · 보유 기간 안 인수 비율만 (수익률 계산 없음)."""
    uni, rows, months, _ = load_all()
    k, n = base_acq_rate(rows, uni, HOLD["ACT"])
    print(f"기준: S&P 500 보통 회사가 12개월 안에 인수 서식을 낸 비율 {k}/{n} = {k / max(n, 1):.1%}")
    for test in TESTS:
        pos, d = build_positions(test, rows, uni)
        print(f"\n[{test}] 사건 {d['events']} · 가격 없음 {d['no_price']} (그중 인수로 보임 {d['no_price_acq']} · 폐지 서식 "
              f"{d['no_price_exit']}) · 대상 밖(금융 · $10억 미만 등) {d['not_eligible']} · 사용 {d['used']}"
              + (f" · 스핀오프 아님(전부터 거래) {d['not_spin']}" if test == "SPIN" else ""))
        print(f"  보유 기간 안 인수 서식: 전체 {d['acq_in']}/{d['events']} = {d['acq_in'] / max(d['events'], 1):.1%} "
              f"(가격 없어 빠진 쪽 {d['no_price_acq_in']})")
        print("  해마다 사건 → 사용:", {y: f"{v['events']}→{v['used']}" for y, v in sorted(d["by_year"].items())})
        if test == "ACT":
            who: dict[str, int] = defaultdict(int)
            for p in pos:
                who[p["who"]] += 1
            print("  펀드별 사용:", dict(sorted(who.items(), key=lambda kv: -kv[1])))
        conc = defaultdict(int)
        for p in pos:
            for k2 in range(1, p["hold"] + 1):
                conc[vb.month_end(pd.Timestamp(p["entry"]) + pd.DateOffset(months=k2))] += 1
        inside = [conc.get(m, 0) for m in months]
        print(f"  달마다 보유 종목 수: 평균 {np.mean(inside):.1f} · 최소 {min(inside)} · 0개인 달 {sum(1 for x in inside if x == 0)}")
    return 0


def summarize(r: np.ndarray, n: np.ndarray, spy: np.ndarray, ew: np.ndarray, months: Sequence[pd.Timestamp],
              placebo: Sequence[float]) -> dict[str, Any]:
    """기간별 통계 · SPY 대비(참고) · 생존 집단 EW 대비(판정) · 플라시보 백분위 · 해마다."""
    cut16 = next(i for i, m in enumerate(months) if m > vb.month_end("2016-05-31"))
    cut25 = next(i for i, m in enumerate(months) if m > vb.month_end("2025-05-31"))
    blk: dict[str, Any] = {"n_avg": float(np.mean(n)), "spy_months": int(np.sum(n == 0))}
    for w, (a, b) in {"full": (0, len(r)), "from2016": (cut16, len(r)), "to2025": (0, cut25)}.items():
        blk[w] = vb.stats_block(list(r[a:b]), list(spy[a:b]))
        blk[w]["vs_spy"] = vb.boot(list(r[a:b]), list(spy[a:b]))
        blk[w]["vs_ew"] = vb.boot(list(r[a:b]), list(ew[a:b]))
    act = blk["full"]["sharpe"]
    sims = sorted(placebo)
    if sims:
        blk["placebo"] = {"pct": 100.0 * sum(1 for x in sims if x < act) / len(sims), "p50": sims[len(sims) // 2],
                          "p95": sims[int(0.95 * len(sims)) - 1]}
    yrs: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for m, x, s, e in zip(months, r, spy, ew):
        yrs[str(m.year)].append((x, s, e))
    blk["years"] = {y: tuple(float(np.prod([1 + v[j] for v in vals]) - 1) for j in range(3)) for y, vals in yrs.items()}
    return blk


def verdict_of(blk: Mapping[str, Any]) -> str:
    """사전 등록 판정 — 생존 회사 집단 기준: EW 대비 ΔSharpe 95% 하한 > 0 + 플라시보 95백분위 이상 → 후보,
    EW 대비 상한 < 0 → 기각, 나머지 운과 구분 안 됨."""
    vs = blk["full"]["vs_ew"]
    if vs and vs["ΔSharpe_하한"] > 0 and blk["placebo"]["pct"] >= 95.0:
        return "후보"
    if vs and vs["ΔSharpe_상한"] < 0:
        return "기각"
    return "운과 구분 안 됨"


def cmd_run(_args: argparse.Namespace, seed: int = 11) -> int:
    uni, rows, months, spy = load_all()
    rng = random.Random(seed)
    start_m = vb.month_end(START)
    grid = [start_m] + list(months)

    def rets_of(t: str, cache: dict[str, np.ndarray]) -> np.ndarray:
        if t not in cache:
            cache[t] = monthly_returns(uni.px.get(t), grid)[1:]
        return cache[t]

    rets: dict[str, np.ndarray] = {}
    # 생존 회사 집단 EW — 달마다 그달 말 대상(시총 $10억 이상 · 금융 제외 · 가격 있음) 전부를 같은 비중으로 다음 달 보유
    ew_pos = []
    for m in [start_m] + list(months[:-1]):
        for cik in uni.pool(m):
            t = uni.by_cik[cik]["px"]
            rets_of(t, rets)
            ew_pos.append({"ticker": t, "entry": m, "hold": 1})
    r_ew, n_ew = calendar_portfolio(ew_pos, rets, months, spy)
    vb.log(f"생존 집단 EW: 평균 {np.mean(n_ew):.0f}종목")
    k, nb = base_acq_rate(rows, uni, HOLD["ACT"])
    res: dict[str, Any] = {"months": [str(m.date()) for m in months], "spy": vb.stats_block(list(spy), list(spy)),
                           "ew": {"n_avg": float(np.mean(n_ew)), "full": vb.stats_block(list(r_ew), list(spy))},
                           "base_acq": {"k": k, "n": nb}, "T": {}}
    res["ew"]["full"]["vs_spy"] = vb.boot(list(r_ew), list(spy))
    for test in TESTS:
        pos, diag = build_positions(test, rows, uni)
        for p in pos:
            rets_of(p["ticker"], rets)
        r, n = calendar_portfolio(pos, rets, months, spy)
        # S&P 500 대상만 (보조) — 산 달에 S&P 500 구성종목이던 사건
        sp_pos = [p for p in pos if p["ticker"] in vb.members_on(uni.inp["mem"], str(pd.Timestamp(p["entry"]).date()))]
        r_sp, n_sp = calendar_portfolio(sp_pos, rets, months, spy)
        # 플라시보: 사건마다 같은 달 · 같은 보유 기간에 생존 집단(그달 $10억 이상 · 금융 제외) 중 무작위 한 종목, 1,000번
        sims = []
        for _ in range(N_PLACEBO):
            fake = []
            for p in pos:
                cik = rng.choice(uni.pool(pd.Timestamp(p["entry"])))
                t = uni.by_cik[cik]["px"]
                rets_of(t, rets)
                fake.append({**p, "ticker": t})
            sims.append(sharpe(list(calendar_portfolio(fake, rets, months, spy)[0]), 12))
        blk = summarize(r, n, spy, r_ew, months, sims)
        blk["diag"] = diag
        blk["sp500"] = {"n_avg": float(np.mean(n_sp)), "full": vb.stats_block(list(r_sp), list(spy))}
        blk["sp500"]["full"]["vs_ew"] = vb.boot(list(r_sp), list(r_ew))
        blk["positions"] = [{**p, "entry": str(pd.Timestamp(p["entry"]).date()), "ret": window_return(p, rets, months)}
                            for p in pos]
        blk["verdict"] = verdict_of(blk)
        res["T"][test] = blk
        vb.log(f"{test}: {blk['verdict']}")
    slim = json.loads(json.dumps(res, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)))
    (OUT / "catalyst_results.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), "utf-8")
    md = report_md(res)
    (OUT / "catalyst_report.md").write_text(md, "utf-8")
    print(md)
    return 0


def report_md(res: Mapping[str, Any]) -> str:
    pct = vb.pct
    sp, ew = res["spy"], res["ew"]["full"]
    names = {"ACT": "행동주의 13D (12개월)", "SPIN": "스핀오프 (24개월)"}

    def ci(vs: Mapping[str, float] | None) -> str:
        vs = vs or {}
        return (f"{vs.get('ΔSharpe', math.nan):+.2f} [{vs.get('ΔSharpe_하한', math.nan):+.2f}, "
                f"{vs.get('ΔSharpe_상한', math.nan):+.2f}]") if vs else "—"

    L = [f"SPY(배당 포함) 2014-06~2026-10: CAGR {pct(sp['cagr'])} · 샤프 {sp['sharpe']:.2f} · MDD {pct(sp['mdd'])}",
         f"생존 집단 EW (지금 상장 · 시총 $10억 이상 · 금융 제외, 평균 {res['ew']['n_avg']:.0f}종목): CAGR {pct(ew['cagr'])} · "
         f"샤프 {ew['sharpe']:.2f} · MDD {pct(ew['mdd'])} · SPY 대비 {ci(ew['vs_spy'])}\n",
         "| 신호 | 평균 보유 종목 | CAGR | 샤프 | MDD(월) | 샤프 차이 vs 생존 EW [95%] (판정) | 플라시보 백분위 (판정) | "
         "샤프 차이 vs SPY [95%] (참고) | 2016~ 샤프 | 2025-05 까지 샤프 | 판정 |",
         "|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|"]
    for t in TESTS:
        b = res["T"][t]
        f = b["full"]
        L.append(f"| {names[t]} | {b['n_avg']:.1f} | {pct(f['cagr'])} | {f['sharpe']:.2f} | {pct(f['mdd'])} | {ci(f['vs_ew'])} | "
                 f"{b['placebo']['pct']:.0f} | {ci(f['vs_spy'])} | {b['from2016']['sharpe']:.2f} | {b['to2025']['sharpe']:.2f} | "
                 f"**{b['verdict']}** |")
    ba = res["base_acq"]
    L.append(f"\n### 사건 · 가격 (빠진 대상)\n\n기준: S&P 500 보통 회사가 12개월 안에 인수 서식을 낸 비율 "
             f"{ba['k']}/{ba['n']} = {ba['k'] / max(ba['n'], 1):.1%}\n")
    for t in TESTS:
        d = res["T"][t]["diag"]
        L.append(f"- {names[t]}: 사건 {d['events']} → 사용 {d['used']} · 가격 없음 {d['no_price']} (인수로 보임 {d['no_price_acq']} · "
                 f"폐지 서식 {d['no_price_exit']}) · 대상 밖 {d['not_eligible']} · 보유 기간 안 인수 서식 {d['acq_in']}/{d['events']} = "
                 f"{d['acq_in'] / max(d['events'], 1):.1%}")
    L.append("\n### 보조 — S&P 500 대상만\n")
    for t in TESTS:
        s = res["T"][t]["sp500"]
        L.append(f"- {names[t]}: 평균 {s['n_avg']:.1f}종목 · CAGR {pct(s['full']['cagr'])} · 샤프 {s['full']['sharpe']:.2f} · "
                 f"생존 EW 대비 {ci(s['full']['vs_ew'])}")
    L.append("\n### 해마다 (신호 · 생존 EW · SPY)\n")
    L.append("| 해 | " + " | ".join(names[t] for t in TESTS) + " | 생존 EW | SPY |")
    L.append("|---|" + "---:|" * (len(TESTS) + 2))
    for y in sorted(res["T"]["ACT"]["years"]):
        a = res["T"]["ACT"]["years"][y]
        L.append(f"| {y} | " + " | ".join(pct(res["T"][t]["years"][y][0], 0) for t in TESTS) +
                 f" | {pct(a[2], 0)} | {pct(a[1], 0)} |")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch-index", "coverage", "run"):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return {"fetch-index": cmd_fetch_index, "coverage": cmd_coverage, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
