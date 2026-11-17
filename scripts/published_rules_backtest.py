#!/usr/bin/env python3
"""공개된 가치 규칙 5개 — 사전 등록 시험 (2026-11-16, 한 번만 실행).

2014년 전에 책 · 논문으로 공개된 규칙을, 우리 데이터에 맞춘 최소 번역만 해서 2014-05 ~ 2026-10 에 그대로 적용한다.
우리가 이 기간 결과를 보고 만든 규칙이 아니라서, 이 기간은 규칙들에게 '처음 보는 구간'이다 (과적합 걱정이 가장 적다).

  MF   마법공식 (Greenblatt 2005) — 자본수익률 EBIT ÷ (순운전자본 + 순유형자산) 순위 + 이익수익률 EBIT ÷ EV 순위의 합이
       작은 순 30개. 금융 · 리츠 · 유틸리티 제외.
  AM   EV/EBIT 하나만 (Loughran · Wellman 2011, Gray · Carlisle 2012 'Quantitative Value') — EBIT ÷ EV 큰 순 30개. 같은 제외.
  PIO  Piotroski F-Score (2000) — 장부가 ÷ 시총 상위 20% 중 F ≥ 8 전부 (10개 미만이면 F 높은 순 · B/M 높은 순으로 10개).
  VMQ  가치 + 모멘텀 + 퀄리티 (Asness · Moskowitz · Pedersen 2013, Asness · Frazzini · Pedersen 2013 QMJ) — 세 점수의
       순위 평균 큰 순 30개.
  SHY  주주환원 수익률 (Faber 2013) — (배당 + 자사주 매입 − 주식 발행 + 부채 순상환) ÷ 시총 큰 순 30개, 최근 해 순이익 ·
       FCF 플러스만.

데이터 · 유니버스(A 그날 S&P 500 · B′ 그날 시총 US$10억 이상) · 수익률(배당 포함 월봉, 1년 보유, 거래비용 0.1%) · 판정은
value_v3_backtest.py (value_v3_recheck) 와 같다 — 그 스크립트의 함수를 그대로 쓴다.
판정 기준은 실행 전에 docs/quant_improvement/tier3/TRIAL_LOG.csv 에 등록했다 (published_rules_v1). 결과를 보고 고치지 않는다.

  build-extra  companyfacts → 추가 항목 (매출총이익 · 계속영업이익) → data/research/value_v3/facts_extra.pkl
  coverage     규칙별로 자료가 있는 회사 수만 센다 (수익률 계산 없음 — 등록 전 확인용)
  run          한 번 실행 → data/research/value_v3/published_results.json · published_report.md

사용: PYTHONPATH=.:src python scripts/published_rules_backtest.py build-extra | coverage | run
연구 전용 — 점수 · 필터 · PT-1 · 사이트와 무관.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import random
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import value_v3_backtest as vb  # noqa: E402
from paper_trading.evaluation import deflated_sharpe, sharpe  # noqa: E402
from web_export.value_model import excluded  # noqa: E402

warnings.filterwarnings("ignore", message="Sorting by default when concatenating")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")  # 베타 계산 경고 (못 내면 1.0 — vb 와 같음)

OUT = vb.OUT
RULES = ("MF", "AM", "PIO", "VMQ", "SHY")
NAMES = {"MF": "마법공식 (Greenblatt)", "AM": "EV/EBIT (Carlisle)", "PIO": "F-Score (Piotroski)",
         "VMQ": "가치+모멘텀+퀄리티 (AQR식)", "SHY": "주주환원 (Faber)", "EW": "같은 비중 전체 (참고)"}
N_PICK = 30            # Greenblatt 20~30 · 나머지도 같은 수
PIO_F_MIN = 8          # Piotroski '높은 F' = 8 · 9
PIO_MIN = 10           # F ≥ 8 이 이보다 적은 해는 F · B/M 순으로 채운다 (1~2종목 묶음 방지)
PIO_BM_Q = 0.20        # 장부가 ÷ 시총 상위 20%
NO_UTIL = frozenset({"Utilities"})   # MF · AM 은 유틸리티도 뺀다 (Greenblatt)
VALUE_MIN, QUAL_MIN = 2, 5           # VMQ — 가치 3개 중 2개 · 퀄리티 8개 중 5개 이상 있어야 점수를 낸다

# 추가로 뽑는 항목 (value_v3_backtest.TAGS 에 없는 것) — 같은 추출 함수(기간 330~390일, 값마다 공시일)를 쓰려고 더해 둔다
EXTRA_TAGS: dict[str, tuple[str, ...]] = {
    "gp": ("GrossProfit",),
    "ni_cont": ("IncomeLossFromContinuingOperations", "ProfitLossFromContinuingOperations"),
}
vb.TAGS.update(EXTRA_TAGS)


# ── 추가 항목 받기 (build-extra)

def _parse_extra(path: str) -> tuple[int, dict[str, Any] | None]:
    cik = int(Path(path).name[3:13])
    try:
        facts = (json.loads(gzip.decompress(Path(path).read_bytes())).get("facts") or {})
        ccy = vb.reporting_ccy(facts)   # build 와 같은 함수 → 같은 통화
        if not ccy:
            return cik, None
        rows = {}
        for row in EXTRA_TAGS:
            got = vb.extract(facts, row, ccy, "flow")
            if got:
                rows[row] = vb.index_rows(got)
        return cik, rows
    except Exception:  # noqa: BLE001 — 깨진 파일은 build 에서도 빠졌다
        return cik, None


def cmd_build_extra(_args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(str(p) for p in vb.CF_DIR.glob("CIK*.json.gz"))
    out: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for cik, rows in ex.map(_parse_extra, files, chunksize=20):
            if rows:
                out[cik] = rows
    (OUT / "facts_extra.pkl").write_bytes(pickle.dumps(out))
    vb.log(f"추가 항목: {len(out)}개 회사 · gp {sum(1 for r in out.values() if 'gp' in r)} · "
           f"ni_cont {sum(1 for r in out.values() if 'ni_cont' in r)}")
    return 0


# ── 그날 시점의 재무 (최근 3개 회계연도)

def _num(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def fund_pit(co: Mapping[str, Any], asof: str) -> dict[str, dict[str, Any]] | None:
    """그날까지 공시된 최근 3개 회계연도(t · p = t−1 · pp = t−2) 항목. 회계연도 고르기는 value_v3_backtest.select_ends
    (연속된 해, 최근 해가 18개월 안, 3년 이상) — 3년 안 되면 None."""
    rows = co["rows"]
    ends = vb.select_ends(rows["rev"], asof)
    if len(ends) < 3:
        return None

    def v(row: str, e: str) -> float | None:
        return _num(vb.pick(rows.get(row), e, asof))

    def year(e: str) -> dict[str, Any]:
        d: dict[str, Any] = {"end": e, "rev": v("rev", e)}
        gp, cogs = v("gp", e), v("cogs", e)
        d["gp"] = gp if gp is not None else (d["rev"] - cogs if d["rev"] is not None and cogs is not None else None)
        nc = v("ni_cont", e)
        d["ni"] = nc if nc is not None else v("ni", e)
        d["ocf"] = v("ocf", e)
        cap = v("capex", e)
        d["fcf"] = d["ocf"] - abs(cap or 0.0) if d["ocf"] is not None else None
        op = v("opinc", e)
        if op is None:
            pt, ie = v("pretax", e), v("intexp", e)
            op = None if pt is None else pt + abs(ie or 0.0)
        d["ebit"] = op
        for k, row in (("ta", "assets"), ("ca", "ca"), ("cl", "cl"), ("eq", "eq"), ("ppe", "ppe"), ("shares", "shares")):
            d[k] = v(row, e)
        c = v("cash_sti", e)
        if c is None:
            c0 = v("cash", e)
            c = None if c0 is None else c0 + (v("sti", e) or 0.0)
        d["cash"] = c
        d["debt"] = vb.total_debt(rows, e, asof)          # None = 차입금 태그가 하나도 없음 → 0 으로 본다
        ln, la, lc = v("ltd_noncur", e), v("ltd_all", e), v("ltd_cur", e)
        d["ltd"] = ln if ln is not None else (la - (lc or 0.0) if la is not None else d["debt"])
        dc = v("debt_cur", e)
        d["cur_debt"] = dc if dc is not None else (lc or 0.0) + (v("stb", e) or 0.0) + (v("cp", e) or 0.0)
        for k in ("div", "rep", "iss", "debt_iss", "debt_rep"):
            x = v(k, e)
            d[k] = abs(x) if x is not None else None
        return d

    return {"t": year(ends[-1]), "p": year(ends[-2]), "pp": year(ends[-3])}


# ── 규칙별 지표 (한 회사)

def ev_of(t: Mapping[str, Any], mcap_fc: float) -> float:
    """EV = 시총 + 차입금 − 현금 (재무제표 통화). 차입금 · 현금 태그가 없으면 0."""
    return mcap_fc + (t["debt"] or 0.0) - (t["cash"] or 0.0)


def mf_metrics(t: Mapping[str, Any], mcap_fc: float) -> dict[str, float] | None:
    """마법공식 — EBIT > 0 · EV > 0 · 유형자본 > 0 일 때만. 순운전자본 = (유동자산 − 현금) − (유동부채 − 단기 차입금),
    0 아래는 0 (Greenblatt: 남는 현금 · 이자 붙는 단기 부채 제외). 유동 구분이 없는 회사는 순운전자본 0."""
    ebit = t["ebit"]
    if ebit is None or ebit <= 0:
        return None
    ev = ev_of(t, mcap_fc)
    if ev <= 0:
        return None
    nwc = 0.0
    if t["ca"] is not None and t["cl"] is not None:
        nwc = max((t["ca"] - (t["cash"] or 0.0)) - (t["cl"] - (t["cur_debt"] or 0.0)), 0.0)
    cap = nwc + max(t["ppe"] or 0.0, 0.0)
    if cap <= 0:
        return None
    return {"ey": ebit / ev, "roc": ebit / cap}


def am_metric(t: Mapping[str, Any], mcap_fc: float) -> float | None:
    """EV/EBIT 의 역수 (EBIT ÷ EV) — EBIT > 0 · EV > 0 일 때만."""
    ebit = t["ebit"]
    if ebit is None or ebit <= 0:
        return None
    ev = ev_of(t, mcap_fc)
    return ebit / ev if ev > 0 else None


PIO_SIGNALS = ("roa", "cfo", "droa", "accrual", "dlever", "dliquid", "eq_offer", "dmargin", "dturn")


PIO_SIG_MIN = 8   # 9개 중 이만큼 계산되면 점수를 낸다 (빠진 신호는 0점) — 매출원가를 따로 안 내는 에너지 · 유틸리티 등이
                  # 높은 B/M 회사의 40~50% 라 '9개 다'를 요구하면 표본이 쏠린다 (등록 전 coverage 확인, 수익률은 안 봄)


def piotroski(f: Mapping[str, Mapping[str, Any]]) -> tuple[int | None, dict[str, bool]]:
    """F-Score 9개 신호 (Piotroski 2000 표 1). PIO_SIG_MIN 개 이상 계산될 때만 점수(빠진 신호 0점), 아니면 None.
    ROA · CFO 는 기초 자산(전년 말) 기준, 부채비율 = 장기 차입금 ÷ 평균 자산 (줄었거나 두 해 모두 0 이면 1),
    유상증자 = 가중평균 희석주식수가 늘지 않았으면 1, 매출총이익률 · 자산회전율은 전년보다 높으면 1."""
    t, p, pp = f["t"], f["p"], f["pp"]
    s: dict[str, bool] = {}

    def pos(x: float | None) -> bool:
        return x is not None and x > 0

    roa_t = t["ni"] / p["ta"] if t["ni"] is not None and pos(p["ta"]) else None
    cfo_t = t["ocf"] / p["ta"] if t["ocf"] is not None and pos(p["ta"]) else None
    roa_p = p["ni"] / pp["ta"] if p["ni"] is not None and pos(pp["ta"]) else None
    if roa_t is not None:
        s["roa"] = roa_t > 0
    if cfo_t is not None:
        s["cfo"] = cfo_t > 0
    if roa_t is not None and roa_p is not None:
        s["droa"] = roa_t > roa_p
    if roa_t is not None and cfo_t is not None:
        s["accrual"] = cfo_t > roa_t
    if pos(t["ta"]) and pos(p["ta"]) and pos(pp["ta"]):
        lt, lp = t["ltd"] or 0.0, p["ltd"] or 0.0
        lev_t, lev_p = lt / ((t["ta"] + p["ta"]) / 2), lp / ((p["ta"] + pp["ta"]) / 2)
        s["dlever"] = lev_t < lev_p or (lt <= 0 and lp <= 0)
    if all(x is not None for x in (t["ca"], p["ca"])) and pos(t["cl"]) and pos(p["cl"]):
        s["dliquid"] = t["ca"] / t["cl"] > p["ca"] / p["cl"]
    if t["shares"] is not None and pos(p["shares"]):
        s["eq_offer"] = t["shares"] <= p["shares"]
    if t["gp"] is not None and p["gp"] is not None and pos(t["rev"]) and pos(p["rev"]):
        s["dmargin"] = t["gp"] / t["rev"] > p["gp"] / p["rev"]
    if pos(t["rev"]) and pos(p["rev"]) and pos(p["ta"]) and pos(pp["ta"]):
        s["dturn"] = t["rev"] / p["ta"] > p["rev"] / pp["ta"]
    return (sum(s.values()) if len(s) >= PIO_SIG_MIN else None), s


def vmq_raw(t: Mapping[str, Any], mcap_fc: float, mom: float | None, beta: float) -> dict[str, float | None]:
    """VMQ 원재료 (클수록 좋게 부호를 맞춤). 가치 = B/P · E/P · CFO/P, 퀄리티(QMJ 수익성 6 + 안전 2) = GPOA · ROE · ROA ·
    CFOA · GMAR · 낮은 발생액 · 낮은 부채비율 · 낮은 베타."""
    def div(a: float | None, b: float | None, pos_b: bool = True) -> float | None:
        if a is None or b is None or (pos_b and b <= 0) or b == 0:
            return None
        return a / b
    ta, eq, ni, ocf = t["ta"], t["eq"], t["ni"], t["ocf"]
    return {
        "bp": div(eq, mcap_fc) if eq is not None and eq > 0 else None,
        "ep": div(ni, mcap_fc), "cfp": div(ocf, mcap_fc), "mom": mom,
        "gpoa": div(t["gp"], ta), "roe": div(ni, eq) if eq is not None and eq > 0 else None, "roa": div(ni, ta),
        "cfoa": div(ocf, ta), "gmar": div(t["gp"], t["rev"]),
        "acc": -(ni - ocf) / ta if ni is not None and ocf is not None and ta is not None and ta > 0 else None,
        "lev": -(t["debt"] or 0.0) / ta if ta is not None and ta > 0 else None,
        "bab": -beta,
    }


def shy_metric(f: Mapping[str, Mapping[str, Any]], mcap_fc: float) -> float | None:
    """주주환원 수익률 (Faber 2013) = (배당 + 자사주 매입 − 주식 발행 + 부채 순상환) ÷ 시총. 부채 순상환은 상환 − 차입
    (현금흐름표), 둘 다 없으면 전년 대비 차입금 감소. 최근 해 순이익 · FCF 가 플러스인 회사만."""
    t, p = f["t"], f["p"]
    if not (t["ni"] is not None and t["ni"] > 0 and t["fcf"] is not None and t["fcf"] > 0) or mcap_fc <= 0:
        return None
    if t["debt_iss"] is not None or t["debt_rep"] is not None:
        pay = (t["debt_rep"] or 0.0) - (t["debt_iss"] or 0.0)
    elif t["debt"] is not None and p["debt"] is not None:
        pay = p["debt"] - t["debt"]
    else:
        pay = 0.0
    return ((t["div"] or 0.0) + (t["rep"] or 0.0) - (t["iss"] or 0.0) + pay) / mcap_fc


# ── 순위 · 고르기 (한 유니버스 · 한 날짜)

def pct_rank(vals: Sequence[float | None]) -> list[float | None]:
    """0~1 백분위 순위 (클수록 1, 같은 값은 평균 순위). None 은 None."""
    s = pd.Series([np.nan if x is None else x for x in vals], dtype=float)
    r = s.rank(pct=True, method="average")
    return [None if math.isnan(x) else float(x) for x in r]


def _mean(xs: Sequence[float | None], need: int) -> float | None:
    ok = [x for x in xs if x is not None]
    return float(np.mean(ok)) if len(ok) >= need else None


def select_rule(rule: str, recs: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """→ (고른 티커, 후보 티커 = 그 규칙의 자료가 있는 회사 — 플라시보도 여기서 뽑는다)."""
    if rule in ("MF", "AM"):
        base = [r for r in recs if r["sector"] not in NO_UTIL]
    else:
        base = list(recs)
    if rule == "MF":
        pool = [r for r in base if r["mf"] is not None]
        ey = pct_rank([r["mf"]["ey"] for r in pool])
        roc = pct_rank([r["mf"]["roc"] for r in pool])
        # 순위 합 (1 = 가장 좋음) — 백분위 합이 클수록 순위 합이 작다. 같으면 이익수익률 높은 쪽
        order = sorted(range(len(pool)), key=lambda i: (-(ey[i] + roc[i]), -pool[i]["mf"]["ey"], pool[i]["ticker"]))
        return [pool[i]["ticker"] for i in order[:N_PICK]], [r["ticker"] for r in pool]
    if rule == "AM":
        pool = [r for r in base if r["am"] is not None]
        pool.sort(key=lambda r: (-r["am"], r["ticker"]))
        return [r["ticker"] for r in pool[:N_PICK]], [r["ticker"] for r in pool]
    if rule == "PIO":
        have = sorted([r for r in base if r["bm"] is not None], key=lambda r: (-r["bm"], r["ticker"]))
        top = have[:max(1, int(round(len(have) * PIO_BM_Q)))]
        pool = [r for r in top if r["f"] is not None]
        pool.sort(key=lambda r: (-r["f"], -r["bm"], r["ticker"]))
        hi = [r for r in pool if r["f"] >= PIO_F_MIN]
        picks = hi if len(hi) >= PIO_MIN else pool[:PIO_MIN]
        return [r["ticker"] for r in picks], [r["ticker"] for r in pool]
    if rule == "VMQ":
        keys_v, keys_q = ("bp", "ep", "cfp"), ("gpoa", "roe", "roa", "cfoa", "gmar", "acc", "lev", "bab")
        ranks = {k: pct_rank([r["vmq"][k] for r in base]) for k in keys_v + keys_q + ("mom",)}
        val = [_mean([ranks[k][i] for k in keys_v], VALUE_MIN) for i in range(len(base))]
        qual = [_mean([ranks[k][i] for k in keys_q], QUAL_MIN) for i in range(len(base))]
        idx = [i for i in range(len(base)) if val[i] is not None and qual[i] is not None and ranks["mom"][i] is not None]
        rv, rq, rm = (pct_rank([x[i] for i in idx]) for x in (val, qual, ranks["mom"]))
        score = {base[i]["ticker"]: (rv[j] + rq[j] + rm[j]) / 3 for j, i in enumerate(idx)}
        pool_t = sorted(score, key=lambda t: (-score[t], t))
        return pool_t[:N_PICK], pool_t
    if rule == "SHY":
        pool = [r for r in base if r["shy"] is not None]
        pool.sort(key=lambda r: (-r["shy"], r["ticker"]))
        return [r["ticker"] for r in pool[:N_PICK]], [r["ticker"] for r in pool]
    raise ValueError(rule)


# ── 한 날짜의 회사 목록 (유니버스 · 시가총액은 value_v3_backtest.run_backtest 와 같은 규칙)

def momentum(df: pd.DataFrame, d0: pd.Timestamp) -> float | None:
    """12-1 개월 수익률 (배당 포함 월말 값, 12개월 전 → 1개월 전)."""
    a = vb.px_at(df, vb.month_end(d0 - pd.DateOffset(months=12)), "adj")
    b = vb.px_at(df, vb.month_end(d0 - pd.DateOffset(months=1)), "adj")
    return b / a - 1 if math.isfinite(a) and math.isfinite(b) else None


def companies_on(inp: Mapping[str, Any], cos: Mapping[int, Any], by_cik: Mapping[int, Any], tick2cik: Mapping[str, int],
                 d0s: str, now: str, flags: dict[str, int]) -> list[dict[str, Any]]:
    px, scan, sm, mem = inp["px"], inp["scan"], inp["sector_map"], inp["mem"]
    spy = px["SPY"]
    d0 = vb.month_end(d0s)
    members = vb.members_on(mem, d0s)
    a_ciks = {tick2cik[t] for t in members if t in tick2cik}
    out: list[dict[str, Any]] = []
    for cik, meta in by_cik.items():
        in_a = cik in a_ciks
        if not (in_a or meta["scan"]):
            continue
        t = meta["px"]
        df = px[t]
        p0 = vb.px_at(df, d0)
        if not (math.isfinite(p0) and math.isfinite(vb.px_at(df, d0, "adj"))):
            continue
        sec = (scan.get(t) or {}).get("sector") or sm.get(t, (None, None))[0]
        ind = (scan.get(t) or {}).get("industry") or sm.get(t, (None, None))[1]
        if not sec or excluded({"sector": sec, "industry": ind}):
            continue
        co = cos[cik]
        mcap_now = meta["mcap_now"]
        p_now = float(df["close"].iloc[-1])
        if not (mcap_now and math.isfinite(p_now) and p_now > 0):
            continue
        ratio, how = vb.share_ratio(co, df, d0s, now)
        mcap_usd = mcap_now * (p0 / p_now) * ratio
        in_b = bool(meta["scan"] and mcap_usd >= 1e9)
        if not (in_a or in_b):
            continue
        ccy = co["ccy"]
        fx = 1.0 if ccy == "USD" else vb.px_at(px.get(f"{ccy}USD=X"), d0)
        if not math.isfinite(fx):
            flags["fx_missing"] = flags.get("fx_missing", 0) + 1
            continue
        f = fund_pit(co, d0s)
        if f is None:
            flags["no_3y"] = flags.get("no_3y", 0) + 1
            continue
        mcap_fc = mcap_usd / fx
        beta = vb.monthly_beta_tr(df, spy, d0)
        fs, sig = piotroski(f)
        t3 = f["t"]
        out.append({"ticker": t, "cik": cik, "in_a": in_a, "in_b": in_b, "sector": sec, "industry": ind,
                    "mcap_usd": mcap_usd, "share_how": how,
                    "mf": mf_metrics(t3, mcap_fc), "am": am_metric(t3, mcap_fc),
                    "bm": t3["eq"] / mcap_fc if t3["eq"] is not None and t3["eq"] > 0 else None,
                    "f": fs, "f_sigs": sig,
                    "vmq": vmq_raw(t3, mcap_fc, momentum(df, d0), beta), "shy": shy_metric(f, mcap_fc)})
    return out


def prepare(inp: Mapping[str, Any]) -> tuple[dict[int, Any], dict[int, Any], dict[str, int]]:
    """CIK 마다 대표 티커 · 옛 CIK 합치기 — value_v3_backtest.run_backtest 와 같은 규칙. 추가 항목(facts_extra)을 붙인다."""
    facts, px, cands, scan = inp["facts"], inp["px"], inp["cands"], inp["scan"]
    extra = pickle.loads((OUT / "facts_extra.pkl").read_bytes())
    for cik, rows in extra.items():
        if cik in facts and rows:
            facts[cik]["rows"].update(rows)
    by_cik: dict[int, dict[str, Any]] = {}
    for t, c in cands.items():
        if not c.get("cik") or not c.get("px") or c["px"] not in px:
            continue
        if c["cik"] not in facts and (c.get("alt") or 0) not in facts:
            continue
        cur = by_cik.get(c["cik"])
        mc = (scan.get(c["px"]) or {}).get("mcap_now") or 0.0
        if cur is None or (c["scan"] and mc > cur["mcap_now"]):
            by_cik[c["cik"]] = {"px": c["px"], "mcap_now": mc, "scan": c["scan"], "alt": c.get("alt") or (cur or {}).get("alt")}
    cos = {cik: vb.merge_facts(facts.get(cik), facts.get(m["alt"]) if m.get("alt") else None) for cik, m in by_cik.items()}
    tick2cik = {t: c["cik"] for t, c in cands.items() if c.get("cik")}
    return by_cik, cos, tick2cik


def cmd_coverage(_args: argparse.Namespace) -> int:
    """규칙별 후보 수만 (수익률 · 고른 종목 수익 없음)."""
    inp = vb.load_inputs()
    by_cik, cos, tick2cik = prepare(inp)
    now = date.today().isoformat()
    flags: dict[str, int] = {}
    print("| 기준일 | 유니버스 | 3년 재무 있음 | MF | AM | PIO 높은 B/M | PIO 9신호 다 | PIO F≥8 | VMQ | SHY | 신호 빠진 이유(B/M 상위) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for d0s in vb.REBAL[:-1]:
        recs = companies_on(inp, cos, by_cik, tick2cik, d0s, now, flags)
        for u in ("a", "b"):
            sub = [r for r in recs if r[f"in_{u}"]]
            n = {rule: len(select_rule(rule, sub)[1]) for rule in RULES}
            have = sorted([r for r in sub if r["bm"] is not None], key=lambda r: -r["bm"])
            top = have[:max(1, int(round(len(have) * PIO_BM_Q)))]
            hi = sum(1 for r in top if r["f"] is not None and r["f"] >= PIO_F_MIN)
            miss: dict[str, int] = {}
            for r in top:
                if r["f"] is None:
                    for k in PIO_SIGNALS:
                        if k not in r["f_sigs"]:
                            miss[k] = miss.get(k, 0) + 1
            print(f"| {d0s} | {u.upper()} | {len(sub)} | {n['MF']} | {n['AM']} | {len(top)} | {n['PIO']} | {hi} | "
                  f"{n['VMQ']} | {n['SHY']} | {', '.join(f'{k} {v}' for k, v in sorted(miss.items(), key=lambda kv: -kv[1]))} |")
    print("표시:", flags)
    return 0


# ── 실행 · 요약

def run_backtest(inp: Mapping[str, Any]) -> dict[str, Any]:
    by_cik, cos, tick2cik = prepare(inp)
    px, spy = inp["px"], inp["px"]["SPY"]
    now = date.today().isoformat()
    vb.log(f"가격 · SEC 둘 다 있는 회사 {len(by_cik)}")
    years: list[dict[str, Any]] = []
    flags: dict[str, int] = {}
    for d0s, d1s in zip(vb.REBAL[:-1], vb.REBAL[1:]):
        d0, d1 = vb.month_end(d0s), vb.month_end(d1s)
        months = [m for m in spy.index if d0 < m <= d1]
        recs = companies_on(inp, cos, by_cik, tick2cik, d0s, now, flags)
        ports: dict[str, dict[str, list[str]]] = {}
        pools: dict[str, dict[str, list[str]]] = {}
        info: dict[str, dict[str, Any]] = {}
        for u in ("A", "B"):
            sub = sorted((r for r in recs if r[f"in_{u.lower()}"]), key=lambda r: r["ticker"])
            ports[u], pools[u] = {}, {}
            for rule in RULES:
                ports[u][rule], pools[u][rule] = select_rule(rule, sub)
            ports[u]["EW"] = pools[u]["EW"] = [r["ticker"] for r in sub]
        for r in recs:
            info[r["ticker"]] = {"sector": r["sector"], "industry": r["industry"], "f": r["f"]}
        all_t = {t for u in pools for lst in pools[u].values() for t in lst} | {"SPY"}
        paths = vb.period_paths(px, all_t, d0, months)
        years.append({"d0": d0s, "d1": d1s, "ports": ports, "pools": pools, "paths": paths, "months": months,
                      "info": {t: info[t] for u in ports for rule in RULES for t in ports[u][rule]}})
        vb.log(f"{d0s}: A {len(pools['A']['EW'])} · B′ {len(pools['B']['EW'])} | 후보 A " +
               " ".join(f"{k} {len(pools['A'][k])}" for k in RULES) + " | B′ " +
               " ".join(f"{k} {len(pools['B'][k])}" for k in RULES) + f" | PIO 고름 A {len(ports['A']['PIO'])} B′ {len(ports['B']['PIO'])}")
    return {"years": years, "flags": flags}


def summarize(bt: Mapping[str, Any], seed: int = 7) -> dict[str, Any]:
    """value_v3_backtest.summarize 와 같은 계산 (규칙만 다름). 판정은 등록한 대로."""
    years = bt["years"]
    paths = [y["paths"] for y in years]
    spy_m, spy_p = vb.run_series([["SPY"]] * len(years), paths, cost=0.0)
    months = [m for y in years for m in y["months"]]
    windows = {"full": (0, len(years)), "from2016": (2, len(years)), "to2025": (0, len(years) - 1)}
    out: dict[str, Any] = {"months": [str(m.date()) for m in months], "spy": {"monthly": spy_m, "period": spy_p}, "U": {}}
    rng = random.Random(seed)
    strats = RULES + ("EW",)
    for u in ("A", "B"):
        res_u: dict[str, Any] = {}
        series_m: dict[str, list[float]] = {}
        for s in strats:
            baskets, picks, spy_years = [], [], 0
            for y in years:
                names = [t for t in y["ports"][u][s] if t in y["paths"]]
                if not names:
                    names, spy_years = ["SPY"], spy_years + 1
                elif s != "EW":
                    picks.extend((y["d0"], t, float(y["paths"][t][-1] - 1)) for t in names)
                baskets.append(names)
            m, p = vb.run_series(baskets, paths)
            series_m[s] = m
            blk: dict[str, Any] = {"baskets": baskets if s != "EW" else [[str(len(b))] for b in baskets], "period": p,
                                   "spy_years": spy_years, "beat_years": sum(1 for a, b in zip(p, spy_p) if a > b),
                                   "n_avg": float(np.mean([len(b) for b in baskets])), "picks_n": len(picks),
                                   "win": float(np.mean([x[2] > 0 for x in picks])) if picks else float("nan"),
                                   "crash": float(np.mean([x[2] <= -0.25 for x in picks])) if picks else float("nan"),
                                   "picks": picks}
            for w, (i0, i1) in windows.items():
                mi0 = sum(len(y["months"]) for y in years[:i0])
                mi1 = sum(len(y["months"]) for y in years[:i1])
                blk[w] = vb.stats_block(m[mi0:mi1], spy_m[mi0:mi1])
                blk[w]["vs_spy"] = vb.boot(m[mi0:mi1], spy_m[mi0:mi1])
            res_u[s] = blk
        for s in RULES:  # 플라시보 — 같은 해 그 규칙의 후보에서 같은 개수 무작위 · 1,000번
            blk = res_u[s]
            sims = []
            for _ in range(vb.N_PLACEBO):
                bk = []
                for y, real in zip(years, blk["baskets"]):
                    pool = [t for t in y["pools"][u][s] if t in y["paths"]]
                    n = 0 if real == ["SPY"] else len(real)
                    bk.append(rng.sample(pool, min(n, len(pool))) if n and pool else ["SPY"])
                sims.append(sharpe(vb.run_series(bk, paths)[0], 12))
            sims.sort()
            act = blk["full"]["sharpe"]
            blk["placebo"] = {"pct": 100.0 * sum(1 for x in sims if x < act) / len(sims),
                              "p50": sims[len(sims) // 2], "p95": sims[int(0.95 * len(sims)) - 1]}
            blk["full"]["vs_ew"] = vb.boot(series_m[s], series_m["EW"])
        trial_sr = [sharpe(series_m[s], 1) for s in RULES]
        for s in RULES:
            r = series_m[s]
            res_u[s]["dsr"] = deflated_sharpe(sharpe(r, 1), trial_sr, len(r), float(pd.Series(r).skew()),
                                              float(pd.Series(r).kurt()) + 3.0)
        out["U"][u] = res_u
    cut = sum(len(y["months"]) for y in years[:2])
    last = len(years[-1]["months"])
    out["spy"].update({"full": vb.stats_block(spy_m, spy_m), "from2016": vb.stats_block(spy_m[cut:], spy_m[cut:]),
                       "to2025": vb.stats_block(spy_m[:-last], spy_m[:-last])})
    verdicts = {}
    for s in RULES:  # 사전 등록 판정
        ok = []
        for u in ("A", "B"):
            b = out["U"][u][s]
            vs = b["full"]["vs_spy"]
            ok.append(bool(vs and vs["ΔSharpe_하한"] > 0 and b["placebo"]["pct"] >= 95.0))
        rej = all((out["U"][u][s]["full"]["vs_spy"] or {}).get("ΔSharpe_상한", 1.0) < 0 for u in ("A", "B"))
        verdicts[s] = "후보" if all(ok) else ("기각" if rej else "운과 구분 안 됨")
    out["verdicts"] = verdicts
    out["flags"] = bt["flags"]
    out["years"] = [{"d0": y["d0"], "d1": y["d1"], "info": y["info"],
                     "ports": {u: {s: y["ports"][u][s] for s in RULES} for u in y["ports"]},
                     "pool_n": {u: {s: len(y["pools"][u][s]) for s in strats} for u in y["pools"]}} for y in years]
    return out


def report_md(res: Mapping[str, Any]) -> str:
    pct = vb.pct
    L = []
    sp = res["spy"]["full"]
    L.append(f"SPY(배당 포함) 2014-06~2026-10: CAGR {pct(sp['cagr'])} · 샤프 {sp['sharpe']:.2f} · MDD {pct(sp['mdd'])}\n")
    for u, title in (("A", "방식 A — 그날 S&P 500 구성종목"), ("B", "방식 B′ — 그날 시가총액 US$10억 이상 (지금 상장 명단)")):
        L.append(f"\n### {title}\n")
        L.append("| 규칙 | 평균 종목 수 | CAGR | 샤프 | MDD(월) | 알파(연) · t | SPY 이긴 해 | 종목 승률 | 폭락(≤−25%) | "
                 "샤프 차이 vs SPY [95%] | 플라시보 백분위 | 2016~ 샤프 | 2025-05 구간 뺀 샤프 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
        for s in RULES + ("EW",):
            b = res["U"][u][s]
            f = b["full"]
            vs = f["vs_spy"] or {}
            ci = (f"{vs.get('ΔSharpe', math.nan):+.2f} [{vs.get('ΔSharpe_하한', math.nan):+.2f}, "
                  f"{vs.get('ΔSharpe_상한', math.nan):+.2f}]") if vs else "—"
            plc = f"{b['placebo']['pct']:.0f}" if "placebo" in b else "—"
            L.append(f"| {NAMES[s]} | {b['n_avg']:.0f} | {pct(f['cagr'])} | {f['sharpe']:.2f} | {pct(f['mdd'])} | "
                     f"{pct(f['alpha'])} · {f['alpha_t']:.2f} | {b['beat_years']}/{len(b['period'])} | {pct(b['win'], 0)} | "
                     f"{pct(b['crash'], 0)} | {ci} | {plc} | {b['from2016']['sharpe']:.2f} | {b['to2025']['sharpe']:.2f} |")
        s16, s25 = res["spy"]["from2016"]["sharpe"], res["spy"]["to2025"]["sharpe"]
        L.append(f"| SPY | 1 | {pct(sp['cagr'])} | {sp['sharpe']:.2f} | {pct(sp['mdd'])} | — | — | — | — | — | — | {s16:.2f} | {s25:.2f} |")
    L.append("\n### 해마다 수익률 (방식 B′)\n")
    yrs = res["years"]
    L.append("| 기간 | " + " | ".join(NAMES[s].split(" (")[0] for s in RULES) + " | 같은 비중 전체 | SPY |")
    L.append("|---|" + "---:|" * (len(RULES) + 2))
    for i, y in enumerate(yrs):
        cells = [pct(res["U"]["B"][s]["period"][i], 0) for s in RULES + ("EW",)]
        L.append(f"| {y['d0'][:7]} → {y['d1'][:7]} | " + " | ".join(cells) + f" | {pct(res['spy']['period'][i], 0)} |")
    L.append("\n### 사전 등록 판정\n")
    for s, v in res["verdicts"].items():
        L.append(f"- {NAMES[s]}: **{v}**")
    return "\n".join(L)


def cmd_run(_args: argparse.Namespace) -> int:
    inp = vb.load_inputs()
    bt = run_backtest(inp)
    res = summarize(bt)
    slim = json.loads(json.dumps(res, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)))
    (OUT / "published_results.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), "utf-8")
    md = report_md(res)
    (OUT / "published_report.md").write_text(md, "utf-8")
    print(md)
    vb.log(f"표시: {res['flags']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build-extra", "coverage", "run"):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    return {"build-extra": cmd_build_extra, "coverage": cmd_coverage, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
