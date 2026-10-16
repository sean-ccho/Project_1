#!/usr/bin/env python3
"""아이디어 묶음 2 (idea_batch_v2) + 발생액 재확인 (acc_confirm_v1): 아직 안 해 본 종목 발굴 신호.

계기: 2026-10-15 사용자 — "B2,3,4 도 테스트해보고" (B1 매출 가속 · B2 애널리스트 · B3 공매도 · B4 발생액 재확인).
틀은 fundamental_research.py 와 같다: 월말 리밸런스, 그날 S&P 500 구성종목(px10y), 다음날 시가 진입 → h 거래일 뒤 시가,
'상위 20% − 그날 신호가 있는 구성종목 평균' (이벤트는 '이벤트 종목 − 같은 구간 구성종목 동일가중').

사전 등록 (결과 계산 전에 --register, TRIAL_LOG):
  idea_batch_v2 (12건, Bonferroni z = 2.87)
    B1 매출 가속 (SEC 분기 매출 YoY, 2016-01 ~ 2025-09)
      REVACC|h21, REVACC|h63   신호 = 최근 분기 YoY − 직전 분기 YoY (두 분기 모두 그날 전에 공시, 최근 분기 말 ≤ 200일 전)
      RUNNER|h21, RUNNER|h63   집합 = YoY ≥ 15% 이고 YoY 가 2분기 연속 커짐 (q > q−1 > q−2). 집합 평균 − YoY 있는 구성종목 평균
    B2 애널리스트 (yfinance upgrades_downgrades, 2019-07 ~ 2025-09 — 그 전 기록이 거의 없음)
      UPG|h21                  이벤트 = 등급 상향(Action up) 난 날, 진입 = 그 달력일 **다음** 거래일 시가 (발표 시각 무관하게 보수적)
      DNG|h63  (피하기)        이벤트 = 등급 하향(Action down). 음(−)이면 피하기 신호
      PTB|h21, PTB|h63         신호 = 직전 90일 목표가 (상향 − 하향) / (상향 + 하향 + 유지), 조치 3건 이상
    B3 공매도 (FINRA 공매도 잔고, 2018-01 ~ 2025-09 — 2017-12-29 부터 공개)
      그날 쓸 수 있는 값 = 결제일 + 12일이 지난 가장 최근 보고 (공개 지연 반영)
      SIR|h21, SIR|h63 (피하기) 신호 = 공매도 잔고 / 발행주식수(SEC 표지, 분할 보정). 상위 20% 가 음(−)이면 피하기 신호
      DTC|h63  (피하기)        신호 = days to cover (FINRA 값)
      DSI|h63  (피하기)        신호 = 공매도 잔고 3개월(6회 보고) 변화율 (분할 보정)
  acc_confirm_v1 (1건, z = 1.96 — 앞 결과 '관찰'의 재확인이라 한 건짜리)
    B4 ACC|h63                 발생액 = −(순이익 − 영업현금흐름) / 자산 (fundamental_research 와 같음).
                               **새 구간 2010-01 ~ 2015-09** (개발 구간 2016~ 와 겹치지 않음), 가격 = yfinance 새로 받은 일봉

판정 (pead_research.grade 와 같음): 후보 = t ≥ z · 비용 후 > 0 · 앞/뒤 반 모두 + · 연도 2/3 이상 +.
  '피하기' 검정은 부호를 뒤집어 같은 기준 (비용 차감 없음 — 안 사는 데는 비용이 없다): 후보 → '피하기 후보'.
한계: yfinance 는 지금 상장된 종목만 → 애널리스트·2010~15 가격에서 상장폐지 종목이 빠진다 (생존 편향).
      2010~15 재무는 px10y 종목(2016 이후 구성종목)만 받아 둔 것 → 2016 전에 빠진 회사는 없다.
      '후보' 라도 곧바로 SPY 를 이긴다는 뜻은 아니다 — 다음 단계는 포트폴리오(PT-1 엔진·달력시간) 검정.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/idea_batch_v2.py --fetch      # 애널리스트·공매도·2010~15 가격 받기 (한 번)
  PYTHONPATH=.:src .venv/bin/python scripts/idea_batch_v2.py --coverage   # 커버리지만 (성과 계산 없음)
  PYTHONPATH=.:src .venv/bin/python scripts/idea_batch_v2.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/idea_batch_v2.py --run
산출: tier3/IDEA_BATCH_V2.md · tier3/idea_batch_v2.csv · data/research/ib2_* (git 제외)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from fundamental_research import _asof, _split_after, latest_fiscal_year  # noqa: E402
from pead_research import grade  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from research_utils import (DEV_END, RESEARCH_DIR, ROUND_TRIP, TIER3_DIR, TRIAL_LOG, bonferroni_z, halves_mean,  # noqa: E402
                            load_ohlcv, load_panel, log_trials, nw_tstat, total_trials, yearly_mean)

FAMILY = "idea_batch_v2"
FAMILY_ACC = "acc_confirm_v1"
STEP = "stage16"
OHLCV_PX10Y = ROOT / "data" / "cache" / "ohlcv_px10y.parquet"
OHLCV_EARLY = RESEARCH_DIR / "ib2_ohlcv_early.parquet"
MEMBERSHIP_CSV = ROOT / "data" / "universe" / "sp500_membership.csv"
ANALYST_PATH = RESEARCH_DIR / "ib2_analyst.parquet"
SHORT_PATH = RESEARCH_DIR / "ib2_short_interest.parquet"
FACTORS_PATH = RESEARCH_DIR / "ib2_factors.parquet"
REV_PATH = RESEARCH_DIR / "rev_yoy_quarterly.parquet"

# (test_id, 신호, h, 종류, 방향, 시작)  종류: xs = 월별 상위 20% · set = 집합 · event = 이벤트. 방향: buy / avoid
TESTS: list[tuple[str, str, int, str, str, str]] = [
    ("REVACC|h21", "REVACC", 21, "xs", "buy", "2016-01-01"),
    ("REVACC|h63", "REVACC", 63, "xs", "buy", "2016-01-01"),
    ("RUNNER|h21", "RUNNER", 21, "set", "buy", "2016-01-01"),
    ("RUNNER|h63", "RUNNER", 63, "set", "buy", "2016-01-01"),
    ("UPG|h21", "UPG", 21, "event", "buy", "2019-07-01"),
    ("DNG|h63", "DNG", 63, "event", "avoid", "2019-07-01"),
    ("PTB|h21", "PTB", 21, "xs", "buy", "2019-07-01"),
    ("PTB|h63", "PTB", 63, "xs", "buy", "2019-07-01"),
    ("SIR|h21", "SIR", 21, "xs", "avoid", "2018-01-01"),
    ("SIR|h63", "SIR", 63, "xs", "avoid", "2018-01-01"),
    ("DTC|h63", "DTC", 63, "xs", "avoid", "2018-01-01"),
    ("DSI|h63", "DSI", 63, "xs", "avoid", "2018-01-01"),
]
N_TESTS = len(TESTS)
ACC_TEST = ("ACC|h63", "ACC", 63, "xs", "buy", "2010-01-01")
EARLY_START, EARLY_REBAL_END, EARLY_PRICE_END = "2009-01-01", "2015-09-30", "2016-01-01"  # 2016 개발 구간과 안 겹치게
TOP_Q = 0.2
MIN_XS = 50          # 월별 단면에 신호 있는 종목이 이보다 적으면 그 달은 건너뜀
MIN_SET = 5          # 집합 종목이 이보다 적으면 그 달은 건너뜀
REV_STALE_DAYS = 200
RUNNER_MIN_YOY = 0.15
PTB_DAYS, PTB_MIN_ACTIONS = 90, 3
SI_PUBLISH_LAG = 12  # 결제일 → 공개까지 (달력일, 보수적)
SI_CHANGE_STEPS = 6  # 6회 보고 ≈ 3개월
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
NAN = float("nan")


def _norm(t: str) -> str:
    return str(t).strip().upper().replace(".", "-").replace("/", "-")


# ============================================================
# 데이터 받기
# ============================================================
def px10y_tickers() -> list[str]:
    o = pd.read_parquet(OHLCV_PX10Y, columns=None)
    close = o.xs("Close", axis=1, level=0)
    return sorted(close.columns[close.notna().any()])


def fetch_analyst(tickers: list[str]) -> pd.DataFrame:
    """yfinance 애널리스트 등급·목표가 변경 기록 (종목마다 한 번)."""
    import yfinance as yf

    parts, fails = [], []
    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        got = None
        for attempt in range(3):
            try:
                got = yf.Ticker(tk).upgrades_downgrades
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  [analyst] {tk} 재시도 {attempt + 1}: {exc}", flush=True)
                time.sleep(5 * (attempt + 1))
        if got is None or got.empty:
            fails.append(tk)
        else:
            g = got.reset_index().rename(columns={"GradeDate": "ts", "Firm": "firm", "ToGrade": "to_grade",
                                                  "FromGrade": "from_grade", "Action": "action",
                                                  "priceTargetAction": "pt_action", "currentPriceTarget": "pt_cur",
                                                  "priorPriceTarget": "pt_prior"})
            g["티커"] = tk
            parts.append(g)
        if i % 25 == 0 or i == len(tickers):
            print(f"[analyst] {i}/{len(tickers)} ({time.time() - t0:.0f}s, 없음 {len(fails)})", flush=True)
        time.sleep(0.4)
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    out.to_parquet(ANALYST_PATH, index=False)
    print(f"[analyst] {out['티커'].nunique()}종목 · {len(out):,}건 · 없음 {len(fails)}: {fails[:20]}", flush=True)
    return out


def _finra(body: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    for attempt in range(4):
        try:
            r = requests.post(FINRA_URL, json=body, headers={"Accept": "application/json"}, timeout=90)
            if r.status_code == 200:
                total = int(r.headers.get("record-total", "0") or 0)
                return (r.json() if r.text.strip() else []), total
            print(f"  [finra] HTTP {r.status_code} 재시도 {attempt + 1}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [finra] {exc} 재시도 {attempt + 1}", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("FINRA 요청 실패")


def fetch_short_interest(tickers: list[str]) -> pd.DataFrame:
    """FINRA 공매도 잔고 — 결제일마다 전 종목을 받아 px10y 종목만 남긴다."""
    rows, _ = _finra({"limit": 5000, "fields": ["settlementDate"],
                      "compareFilters": [{"compareType": "equal", "fieldName": "symbolCode", "fieldValue": "AAPL"},
                                         {"compareType": "gte", "fieldName": "settlementDate", "fieldValue": "2017-01-01"},
                                         {"compareType": "lte", "fieldName": "settlementDate",
                                          "fieldValue": f"{DEV_END:%Y-%m-%d}"}]})
    dates = sorted({r["settlementDate"] for r in rows})
    want = set(tickers)
    fields = ["settlementDate", "symbolCode", "currentShortPositionQuantity", "daysToCoverQuantity",
              "averageDailyVolumeQuantity", "stockSplitFlag"]
    parts = []
    t0 = time.time()
    for i, d in enumerate(dates, 1):
        got, offset, total = [], 0, None
        while total is None or offset < total:
            page, total = _finra({"limit": 5000, "offset": offset, "fields": fields,
                                  "compareFilters": [{"compareType": "equal", "fieldName": "settlementDate",
                                                      "fieldValue": d}]})
            got += page
            offset += 5000
            if not page:
                break
            time.sleep(0.3)
        df = pd.DataFrame(got)
        if not df.empty:
            df["티커"] = df["symbolCode"].map(_norm)
            parts.append(df[df["티커"].isin(want)])
        if i % 10 == 0 or i == len(dates):
            print(f"[short] {i}/{len(dates)} {d} ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True).rename(columns={
        "settlementDate": "settle", "currentShortPositionQuantity": "si", "daysToCoverQuantity": "dtc",
        "averageDailyVolumeQuantity": "adv", "stockSplitFlag": "split_flag"})
    out["settle"] = pd.to_datetime(out["settle"])
    out = out.drop(columns=["symbolCode"]).drop_duplicates(["settle", "티커"])
    out.to_parquet(SHORT_PATH, index=False)
    print(f"[short] {len(dates)}회 · {out['티커'].nunique()}종목 · {len(out):,}행", flush=True)
    return out


def fetch_early_prices() -> pd.DataFrame:
    """2009~2015 S&P 500 구성종목(기간 중 한 번이라도) 일봉 — yfinance (지금 받을 수 있는 종목만)."""
    import yfinance as yf

    mem = Membership(load_membership(MEMBERSHIP_CSV))
    tickers = sorted(_norm(t) for t in mem.union_between("2009-12-31", EARLY_REBAL_END))
    frames = []
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        data = None
        for attempt in range(3):
            try:
                data = yf.download(batch, start=EARLY_START, end=EARLY_PRICE_END, interval="1d", auto_adjust=True,
                                   threads=True, progress=False)
                if data is not None and not data.empty:
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"  재시도 {attempt + 1}: {exc}", flush=True)
            time.sleep(10 * (attempt + 1))
        print(f"[early] {i // 50 + 1}/{(len(tickers) + 49) // 50}", flush=True)
        if data is None or data.empty:
            continue
        if "Close" not in data.columns.get_level_values(0):
            data = data.swaplevel(0, 1, axis=1)
        frames.append(data[["Open", "Close"]])
        time.sleep(3)
    raw = pd.concat(frames, axis=1).sort_index(axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw.columns.names = ["Price", "Ticker"]
    have = raw["Close"].notna().any()
    print(f"[early] 가격 있는 종목 {int(have.sum())} / {len(tickers)}", flush=True)
    raw.to_parquet(OHLCV_EARLY)
    return raw


# ============================================================
# 신호 만들기
# ============================================================
def month_end_members(members: pd.DataFrame) -> pd.DataFrame:
    sessions = pd.DatetimeIndex(sorted(members["date"].unique()))
    month_end = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).max()
    return members[members["date"].isin(set(month_end))][["date", "티커"]].copy()


def forward_returns(rebal: pd.DataFrame, opens: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """다음날 시가 진입 → h 거래일 뒤 시가."""
    x = rebal
    for h in horizons:
        fwd = (opens.shift(-(1 + h)) / opens.shift(-1) - 1).stack().rename(f"fwd_{h}").reset_index()
        fwd.columns = ["date", "티커", f"fwd_{h}"]
        x = x.merge(fwd, on=["date", "티커"], how="left")
    return x


def revenue_signals(rebal: pd.DataFrame) -> pd.DataFrame:
    """(date, 티커) → REVACC, RUNNER, yoy0. 그날 **전에** 공시된 분기만."""
    rev = pd.read_parquet(REV_PATH)
    rev["end"] = pd.to_datetime(rev["end"])
    rev["known"] = pd.to_datetime(rev["known"])
    by = {t: g.sort_values("end") for t, g in rev.groupby("티커")}
    out = []
    for (tk, g) in rebal.groupby("티커"):
        q = by.get(tk)
        if q is None:
            continue
        ends, known, yoy = q["end"].to_numpy(), q["known"].to_numpy(), q["yoy"].to_numpy(float)
        for d in g["date"]:
            dd = np.datetime64(d)
            ok = known < dd
            if not ok.any():
                continue
            e, y = ends[ok], yoy[ok]
            i0 = int(np.argmax(e))  # 공시된 것 중 기말이 가장 최근인 분기
            if (dd - e[i0]) / np.timedelta64(1, "D") > REV_STALE_DAYS:
                continue

            def prev(i: int) -> int | None:
                target = e[i] - np.timedelta64(91, "D")
                gap = np.abs((e - target) / np.timedelta64(1, "D"))
                j = int(np.argmin(gap))
                return j if gap[j] <= 20 and j != i else None

            i1 = prev(i0)
            i2 = prev(i1) if i1 is not None else None
            y0 = y[i0]
            y1 = y[i1] if i1 is not None else NAN
            y2 = y[i2] if i2 is not None else NAN
            out.append({"date": d, "티커": tk, "yoy0": y0, "REVACC": y0 - y1,
                        "RUNNER": float(y0 >= RUNNER_MIN_YOY and y0 > y1 > y2) if np.isfinite(y2) else NAN})
    return pd.DataFrame(out)


def analyst_signals(rebal: pd.DataFrame, an: pd.DataFrame) -> pd.DataFrame:
    """(date, 티커) → PTB = 직전 90일 목표가 (상향 − 하향) / 전체 목표가 조치. 그날 **전** 달력일까지의 기록만."""
    a = an.copy()
    a["day"] = a["ts"].dt.tz_localize(None).dt.normalize() if a["ts"].dt.tz is not None else a["ts"].dt.normalize()
    a = a[a["pt_action"].isin(["Raises", "Lowers", "Maintains"])]
    a["up"] = (a["pt_action"] == "Raises").astype(int)
    a["dn"] = (a["pt_action"] == "Lowers").astype(int)
    by = {t: g.sort_values("day") for t, g in a.groupby("티커")}
    out = []
    for tk, g in rebal.groupby("티커"):
        q = by.get(tk)
        if q is None:
            continue
        days, up, dn = q["day"].to_numpy(), q["up"].to_numpy(), q["dn"].to_numpy()
        for d in g["date"]:
            dd = np.datetime64(d)
            m = (days < dd) & (days >= dd - np.timedelta64(PTB_DAYS, "D"))
            n = int(m.sum())
            if n >= PTB_MIN_ACTIONS:
                out.append({"date": d, "티커": tk, "PTB": (up[m].sum() - dn[m].sum()) / n})
    return pd.DataFrame(out)


def short_signals(rebal: pd.DataFrame, si: pd.DataFrame) -> pd.DataFrame:
    """(date, 티커) → SIR, DTC, DSI. 결제일 + 12일 < 그날 인 가장 최근 보고."""
    splits = pd.read_parquet(RESEARCH_DIR / "splits_px10y.parquet")
    g_after = _split_after(splits)
    s = si.copy()
    s["known"] = s["settle"] + pd.Timedelta(days=SI_PUBLISH_LAG)
    # 지금 분할 기준으로 맞춘 공매도 주식수 (변화율·비율 계산용)
    s["si_adj"] = [v * g_after(t, d) for t, d, v in zip(s["티커"], s["settle"], s["si"].astype(float))]
    s = s.sort_values(["티커", "settle"])
    s["si_adj_prev"] = s.groupby("티커")["si_adj"].shift(SI_CHANGE_STEPS)
    s["settle_prev"] = s.groupby("티커")["settle"].shift(SI_CHANGE_STEPS)
    gap = (s["settle"] - s["settle_prev"]).dt.days
    s["DSI"] = np.where(gap.between(70, 110) & (s["si_adj_prev"] > 0), s["si_adj"] / s["si_adj_prev"] - 1, np.nan)
    x = _asof(rebal, s[["티커", "known", "settle", "si_adj", "dtc", "DSI"]], "date", "known")
    x.loc[~((x["date"] - x["settle"]).dt.days <= 40), ["si_adj", "dtc", "DSI"]] = np.nan  # 오래된 보고는 버림

    sh = pd.read_parquet(RESEARCH_DIR / "sec_shares.parquet")
    sh["shares_adj"] = [v * g_after(t, e) for t, e, v in zip(sh["티커"], sh["end"], sh["shares"])]
    y = _asof(x[["date", "티커"]], sh[["티커", "filed", "shares_adj"]].rename(columns={"filed": "sfiled"}), "date", "sfiled")
    x = x.merge(y[["date", "티커", "shares_adj", "sfiled"]], on=["date", "티커"], how="left")
    x.loc[~((x["date"] - x["sfiled"]).dt.days <= 400), "shares_adj"] = np.nan
    x["SIR"] = (x["si_adj"] / x["shares_adj"]).where(lambda v: (v >= 0) & (v <= 1.5))
    x["DTC"] = x["dtc"].astype(float)
    return x[["date", "티커", "SIR", "DTC", "DSI"]]


def acc_signal(rebal: pd.DataFrame) -> pd.DataFrame:
    """fundamental_research 의 ACC 와 같은 정의 (그날 전에 제출된 가장 최근 회계연도, 550일 넘게 지난 값은 버림)."""
    fa = pd.read_parquet(RESEARCH_DIR / "sec_fund_annual.parquet")
    x = latest_fiscal_year(rebal, fa.drop(columns=["cik"]).rename(columns={"end": "fy_end"}))
    stale = ~((x["date"] - x["fy_end"]).dt.days <= 550)
    acc = -(x["net_income"] - x["cfo"]) / x["assets"]
    x["ACC"] = acc.where(~stale).replace([np.inf, -np.inf], np.nan)
    return x[["date", "티커", "ACC"]]


def event_entries(an: pd.DataFrame, action: str, opens: pd.DataFrame, start: str) -> pd.DataFrame:
    """등급 변경 난 달력일의 **다음** 거래일이 진입일 (같은 종목·같은 날 여러 건은 하나)."""
    a = an[an["action"] == action].copy()
    ts = a["ts"].dt.tz_localize(None) if a["ts"].dt.tz is not None else a["ts"]
    a["day"] = ts.dt.normalize()
    a = a.drop_duplicates(["티커", "day"])
    idx = opens.index
    pos = idx.searchsorted(a["day"].to_numpy(), side="right")  # 그날 이후 첫 거래일
    a["entry_pos"] = pos
    a = a[(a["entry_pos"] < len(idx)) & a["티커"].isin(opens.columns)]
    a = a[idx[a["entry_pos"].to_numpy()] >= pd.Timestamp(start)]
    return a[["티커", "entry_pos"]].reset_index(drop=True)


# ============================================================
# 검정
# ============================================================
def xs_test(x: pd.DataFrame, sig: str, h: int, start: str, end: pd.Timestamp = DEV_END) -> tuple[dict, pd.Series]:
    """월별 (상위 20% − 신호 있는 구성종목 평균)."""
    d = x[(x["date"] >= start) & (x["date"] <= end)][["date", sig, f"fwd_{h}"]].dropna()
    lag = max(1, h // 21)
    rows = []
    for date, g in d.groupby("date"):
        if len(g) < MIN_XS:
            continue
        top = g[g[sig] >= g[sig].quantile(1 - TOP_Q)]
        bot = g[g[sig] <= g[sig].quantile(TOP_Q)]
        rows.append({"date": date, "ex": top[f"fwd_{h}"].mean() - g[f"fwd_{h}"].mean(),
                     "spread": top[f"fwd_{h}"].mean() - bot[f"fwd_{h}"].mean(),
                     "ic": g[sig].corr(g[f"fwd_{h}"], method="spearman"), "n": len(top)})
    return _summ(pd.DataFrame(rows), lag)


def set_test(x: pd.DataFrame, flag: str, base: str, h: int, start: str) -> tuple[dict, pd.Series]:
    """월별 (집합 평균 − base 값이 있는 구성종목 평균)."""
    d = x[(x["date"] >= start) & (x["date"] <= DEV_END)][["date", flag, base, f"fwd_{h}"]].dropna(subset=[base, f"fwd_{h}"])
    lag = max(1, h // 21)
    rows = []
    for date, g in d.groupby("date"):
        s = g[g[flag] == 1.0]
        if len(g) < MIN_XS or len(s) < MIN_SET:
            continue
        rows.append({"date": date, "ex": s[f"fwd_{h}"].mean() - g[f"fwd_{h}"].mean(),
                     "spread": NAN, "ic": NAN, "n": len(s)})
    return _summ(pd.DataFrame(rows), lag)


def _summ(s: pd.DataFrame, lag: int) -> tuple[dict, pd.Series]:
    if s.empty:
        return {}, pd.Series(dtype=float)
    s = s.set_index("date")
    ex = s["ex"]
    yr = yearly_mean(ex)
    first, second = halves_mean(ex)
    return ({"n_obs": int(len(s)), "n_avg": float(s["n"].mean()), "mean_ex": float(ex.mean()),
             "mean_net": float(ex.mean() - ROUND_TRIP), "t_nw": nw_tstat(ex, lag=lag),
             "spread": float(s["spread"].mean()), "t_spread": nw_tstat(s["spread"], lag=lag),
             "ic": float(s["ic"].mean()), "years_pos": int((yr > 0).sum()), "n_years": int(yr.size),
             "first_half": first, "second_half": second, "hit": NAN,
             "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items())}, ex)


def ev_test(entries: pd.DataFrame, opens: pd.DataFrame, member: pd.DataFrame, h: int) -> tuple[dict, pd.Series]:
    """이벤트 종목 h일 수익 − 같은 구간 구성종목 동일가중 → 진입일별 평균 → NW t. 겹치는 재이벤트는 첫 건만."""
    fwd = opens.shift(-h) / opens - 1.0
    ar = fwd.sub(fwd.where(member).mean(axis=1), axis=0)
    keep = []
    for _, g in entries.sort_values("entry_pos").groupby("티커"):
        last = None
        for i, p in zip(g.index, g["entry_pos"]):
            if last is None or p >= last + h:
                keep.append(i)
                last = p
    ep = entries.loc[sorted(keep)]
    col = {t: j for j, t in enumerate(opens.columns)}
    pos = ep["entry_pos"].to_numpy()
    vals = ar.to_numpy()[pos, [col[t] for t in ep["티커"]]] if len(ep) else np.array([])
    ism = member.to_numpy()[pos, [col[t] for t in ep["티커"]]] if len(ep) else np.array([], dtype=bool)
    df = pd.DataFrame({"date": opens.index[pos], "ar": vals})[ism].dropna()
    df = df[df["date"] <= DEV_END]
    if df.empty:
        return {}, pd.Series(dtype=float)
    daily = df.groupby("date")["ar"].mean()
    yr = yearly_mean(daily)
    first, second = halves_mean(daily)
    return ({"n_obs": int(daily.size), "n_avg": float(len(df)), "mean_ex": float(df["ar"].mean()),
             "mean_net": float(df["ar"].mean() - ROUND_TRIP), "t_nw": nw_tstat(daily, lag=h),
             "spread": NAN, "t_spread": NAN, "ic": NAN, "years_pos": int((yr > 0).sum()), "n_years": int(yr.size),
             "first_half": first, "second_half": second, "hit": float((df["ar"] > 0).mean()),
             "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items())}, daily)


def judge(r: dict, z: float, direction: str) -> str:
    """buy = pead_research.grade. avoid = 부호를 뒤집어 같은 기준 (비용 없음)."""
    if not r:
        return "표본 없음"
    if direction == "buy":
        return grade(r, z)
    flipped = {**r, "t_nw": -r["t_nw"], "mean_net": -r["mean_ex"], "first_half": -r["first_half"],
               "second_half": -r["second_half"], "years_pos": r["n_years"] - r["years_pos"]}
    g = grade(flipped, z)
    return {"후보": "피하기 후보", "관찰": "피하기 관찰", "역신호": "역신호(오히려 +)"}.get(g, g)


# ============================================================
# 실행
# ============================================================
def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def _registered(family: str) -> bool:
    log = _log()
    return bool(((log["family"] == family) & (log["판정"] == "등록")).any())


def register() -> None:
    z = bonferroni_z(N_TESTS)
    if not _registered(FAMILY):
        log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": (
            f"B1 매출 가속(REVACC·RUNNER) · B2 애널리스트(UPG·DNG·PTB) · B3 공매도(SIR·DTC·DSI) {N_TESTS}건, "
            f"월말 상위20%/집합/이벤트 vs 구성종목, z={z:.2f}. 정의는 scripts/idea_batch_v2.py 머리말")}])
        print(f"[등록] {FAMILY} {N_TESTS}건 z={z:.2f}")
    if not _registered(FAMILY_ACC):
        log_trials(STEP, FAMILY_ACC, 1, [{"test_id": "(등록)", "판정": "등록", "메모": (
            "fundamental_px10y_v2 ACC|h63 '관찰'의 재확인 1건: 2010-01~2015-09 S&P PIT (yfinance 새 가격), z=1.96")}])
        print(f"[등록] {FAMILY_ACC} 1건 z=1.96")


def build_factors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(개발 구간 표, 2010~15 표)."""
    members = load_panel("px10y")[["date", "티커"]]
    rebal = month_end_members(members)
    opens = load_ohlcv(OHLCV_PX10Y).xs("Open", axis=1, level="Price")
    x = forward_returns(rebal, opens, [21, 63])
    x = x.merge(revenue_signals(rebal), on=["date", "티커"], how="left")
    an = pd.read_parquet(ANALYST_PATH)
    x = x.merge(analyst_signals(rebal, an), on=["date", "티커"], how="left")
    x = x.merge(short_signals(rebal, pd.read_parquet(SHORT_PATH)), on=["date", "티커"], how="left")
    x.to_parquet(FACTORS_PATH, index=False)

    early = load_ohlcv(OHLCV_EARLY)
    eo = early.xs("Open", axis=1, level="Price")
    eo = eo.loc[:, eo.notna().any()]
    mem = Membership(load_membership(MEMBERSHIP_CSV))
    days = eo.index[(eo.index >= "2010-01-01") & (eo.index <= EARLY_REBAL_END)]
    em = pd.DataFrame([(d, t) for d in days for t in mem.on(f"{d:%Y-%m-%d}") if t in eo.columns], columns=["date", "티커"])
    erebal = month_end_members(em)
    e = forward_returns(erebal, eo, [21, 63])
    e = e.merge(acc_signal(erebal), on=["date", "티커"], how="left")
    return x, e


def coverage(x: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    cols = ["REVACC", "RUNNER", "PTB", "SIR", "DTC", "DSI"]
    cov = x.groupby(x["date"].dt.year)[cols].apply(lambda g: g.notna().mean())
    cov["구성종목"] = x.groupby(x["date"].dt.year)["티커"].nunique()
    ecov = e.groupby(e["date"].dt.year).agg(구성종목_가격있음=("티커", "nunique"), ACC=("ACC", lambda s: s.notna().mean()))
    print(cov.round(2).to_string())
    print(ecov.round(2).to_string())
    return cov


def run() -> None:
    if not (_registered(FAMILY) and _registered(FAMILY_ACC)):
        sys.exit("먼저 --register")
    z, z_acc = bonferroni_z(N_TESTS), bonferroni_z(1)
    x, e = build_factors()
    cov = coverage(x, e)
    an = pd.read_parquet(ANALYST_PATH)
    opens = load_ohlcv(OHLCV_PX10Y).xs("Open", axis=1, level="Price")
    mem = Membership(load_membership(MEMBERSHIP_CSV))
    member = pd.DataFrame(False, index=opens.index, columns=opens.columns)
    for d in opens.index:
        cols = [t for t in mem.on(f"{d:%Y-%m-%d}") if t in member.columns]
        member.loc[d, cols] = True

    rows, series = [], {}
    for tid, sig, h, kind, direction, start in TESTS:
        if kind == "xs":
            r, s = xs_test(x, sig, h, start)
        elif kind == "set":
            r, s = set_test(x, sig, "yoy0", h, start)
        else:
            r, s = ev_test(event_entries(an, "up" if sig == "UPG" else "down", opens, start), opens, member, h)
        rows.append({"test_id": tid, "family": FAMILY, "kind": kind, "direction": direction, "start": start, **r,
                     "판정": judge(r, z, direction)})
        series[tid] = s
        print(f"[run] {tid}: t={r.get('t_nw', NAN):+.2f} → {rows[-1]['판정']}", flush=True)
    tid, sig, h, kind, direction, start = ACC_TEST
    r, s = xs_test(e, sig, h, start, pd.Timestamp(EARLY_REBAL_END))
    rows.append({"test_id": tid + " (2010~15)", "family": FAMILY_ACC, "kind": kind, "direction": direction,
                 "start": start, **r, "판정": judge(r, z_acc, direction)})
    print(f"[run] ACC 재확인: t={r.get('t_nw', NAN):+.2f} → {rows[-1]['판정']}", flush=True)
    ref, _ = xs_test(e, "ACC", 21, start, pd.Timestamp(EARLY_REBAL_END))  # 참고 (판정 없음)

    res = pd.DataFrame(rows)
    res.to_csv(TIER3_DIR / "idea_batch_v2.csv", index=False)
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}" if pd.notna(r.get("t_nw")) else "",
                                        "판정": r["판정"]} for _, r in res[res["family"] == FAMILY].iterrows()])
    a = res[res["family"] == FAMILY_ACC].iloc[0]
    log_trials(STEP, FAMILY_ACC, 1, [{"test_id": "ACC|h63", "t": f"{a.t_nw:.2f}", "판정": a["판정"]}])
    write_report(res, cov, ref, z, z_acc, e)


def _pct(v: Any) -> str:
    return f"{v:+.2%}" if v is not None and pd.notna(v) else "–"


def write_report(res: pd.DataFrame, cov: pd.DataFrame, ref: dict, z: float, z_acc: float, e: pd.DataFrame) -> None:
    name = {"REVACC": "매출 가속 (YoY − 직전 분기 YoY) 상위 20%", "RUNNER": "러너형 (YoY ≥ 15% · 2분기 연속 가속)",
            "UPG": "등급 상향 이벤트", "DNG": "등급 하향 이벤트 (피하기)", "PTB": "목표가 상향 비율 상위 20%",
            "SIR": "공매도 비율 상위 20% (피하기)", "DTC": "days to cover 상위 20% (피하기)",
            "DSI": "공매도 3개월 증가율 상위 20% (피하기)", "ACC": "발생액 (이익의 질) 상위 20%"}
    lines = [
        "# 아이디어 묶음 2 — 매출 가속 · 애널리스트 · 공매도 · 발생액 재확인", "",
        f"- 실행 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 정의·사전 등록은 `scripts/idea_batch_v2.py` 머리말",
        f"- idea_batch_v2 **{N_TESTS}건 → z = {z:.2f}** · acc_confirm_v1 1건 → z = {z_acc:.2f} · 누적 시도 {total_trials()}건",
        "- 기준: 그날 S&P 500 구성종목 (px10y), 다음날 시가 진입 → h 거래일 뒤 시가. 비용 후 = 왕복 0.2% 차감 (피하기 검정은 차감 없음)",
        "- 판정: " + " · ".join(f"{k} {v}" for k, v in res["판정"].value_counts().items()), "",
        "| 판정 | test | 내용 | 구간 시작 | 관측 | 평균 초과 | 비용 후 | t_nw | 상위−하위 | 연도+ | 앞/뒤 반 |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        sig = r.test_id.split("|")[0]
        obs = f"{int(r.n_obs)}{'일' if r.kind == 'event' else '달'}" if pd.notna(r.get("n_obs")) else "–"
        lines.append(f"| {r['판정']} | {r.test_id} | {name.get(sig, sig)} | {r.start[:7]} | {obs} | {_pct(r.get('mean_ex'))} | "
                     f"{_pct(r.get('mean_net')) if r.direction == 'buy' else '–'} | {r.get('t_nw', NAN):+.2f} | "
                     f"{_pct(r.get('spread'))} | {int(r.years_pos) if pd.notna(r.get('years_pos')) else '–'}/"
                     f"{int(r.n_years) if pd.notna(r.get('n_years')) else '–'} | "
                     f"{_pct(r.get('first_half'))} / {_pct(r.get('second_half'))} |")
    lines += ["", f"참고 (판정 없음): ACC|h21 2010~15 평균 초과 {_pct(ref.get('mean_ex'))} · t {ref.get('t_nw', NAN):+.2f}", "",
              "## 연도별 평균 초과수익", "", "| test | 연도별 |", "|---|---|"]
    lines += [f"| {r.test_id} | {r.get('yearly', '')} |" for _, r in res.iterrows()]
    lines += ["", "## 커버리지 (그해 구성종목 중 신호가 있는 비율)", "",
              "| 연도 | 구성종목 | REVACC | RUNNER | PTB | SIR | DTC | DSI |", "|---|---|---|---|---|---|---|---|"]
    for y, c in cov.iterrows():
        lines.append(f"| {y} | {int(c['구성종목'])} | " + " | ".join(f"{c[k]:.0%}" for k in
                                                                    ["REVACC", "RUNNER", "PTB", "SIR", "DTC", "DSI"]) + " |")
    ec = e.groupby(e["date"].dt.year).agg(n=("티커", "nunique"), acc=("ACC", lambda s: s.notna().mean()))
    lines += ["", "2010~15 (ACC 재확인): " + " · ".join(f"{y} 가격 있는 구성종목 {int(c.n)} · ACC {c.acc:.0%}"
                                                    for y, c in ec.iterrows())]
    lines += ["", "## 한계", "",
              "- yfinance 는 지금 상장된 종목만 준다 → 애널리스트 기록과 2010~15 가격에서 상장폐지 종목이 빠진다.",
              "- 2010~15 재무는 px10y 종목(2016 이후 구성종목)만 → 2016 전에 지수에서 빠진 회사는 ACC 가 없다 (살아남은 회사끼리 비교).",
              "- 애널리스트 기록은 yfinance 가 2019년부터만 촘촘하다 → B2 는 6년 남짓.",
              "- '후보' 는 단면(종목 간) 차이일 뿐이다. SPY 를 이기는지는 포트폴리오 검정을 따로 해야 한다."]
    (TIER3_DIR / "IDEA_BATCH_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8 + len(res)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", choices=["analyst", "short", "early"], help="--fetch 중 하나만")
    args = ap.parse_args()
    if args.fetch:
        tk = px10y_tickers()
        print(f"===== 데이터 받기 (px10y {len(tk)}종목)", flush=True)
        if args.only in (None, "short"):
            print("===== FINRA 공매도", flush=True)
            fetch_short_interest(tk)
        if args.only in (None, "early"):
            print("===== 2009~15 가격", flush=True)
            fetch_early_prices()
        if args.only in (None, "analyst"):
            print("===== 애널리스트", flush=True)
            fetch_analyst(tk)
    if args.coverage:
        coverage(*build_factors())
    if args.register:
        register()
    if args.run:
        run()


if __name__ == "__main__":
    main()
