#!/usr/bin/env python3
"""SEC 원자료(fetch_sec_data.py) → 연구용 표 3개. 개발 구간(≤ 2025-09-30)에 공개된 것만 남긴다.

1) 실적 발표일 : 8-K Item 2.02 (정정 8-K/A 제외). acceptanceDateTime 은 UTC('Z') → ET 로 바꾼다
   (확인: AMD 장 마감 후 발표가 여름 20:16Z · 겨울 21:16Z = 16:16 ET). 16:00 ET 이후면 다음 거래일이 반응일.
   같은 회사 5일 안에 여러 건이면 첫 건만.
2) 연간 재무   : 10-K(/A) 의 연간(350~380일) 흐름 값과 기말 잔액. (태그, 기말일)마다 **처음 제출된 값**만 (나중 정정 무시)
   → 그 회계연도를 처음 알 수 있던 날 = 항목 중 가장 이른 filed. 그보다 45일 넘게 늦게 처음 나온 항목은 비운다.
3) 발행주식수 : dei:EntityCommonStockSharesOutstanding (10-K·10-Q 표지). 기준일(end)·filed.

사용법: PYTHONPATH=.:src python scripts/build_sec_data.py            (연구용, ≤ 2025-09-30)
        PYTHONPATH=.:src python scripts/build_sec_data.py --holdout  (홀드아웃 1회 확인용 실적일만, 전체 기간)
산출 (git 제외): data/research/sec_earnings_dates.parquet, sec_fund_annual.parquet, sec_shares.parquet
          커버리지 (git): tier3/sec_coverage.csv
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from research_utils import DEV_END, RESEARCH_DIR, TIER3_DIR, load_ohlcv, load_panel

SEC_DIR = RESEARCH_DIR / "sec"
OHLCV_PX10Y = ROOT / "data" / "cache" / "ohlcv_px10y.parquet"
LATE_ITEM_DAYS = 45

# 항목 → 태그 우선순위 (앞이 우선)
FLOW_TAGS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "gross_profit": ["GrossProfit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}
STOCK_TAGS: dict[str, list[str]] = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}


def _load(path: Path) -> dict:
    with gzip.open(path) as f:
        return json.load(f)


def earnings_dates(cik: int, sessions: pd.DatetimeIndex) -> list[dict]:
    """한 회사의 실적 발표 8-K → 반응 거래일."""
    main = SEC_DIR / "submissions" / f"CIK{cik:010d}.json.gz"
    if not main.exists():
        return []
    sub = _load(main)
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        p = SEC_DIR / "submissions" / f"{f['name']}.gz"
        if p.exists():
            blocks.append(_load(p))
    acc = []
    for b in blocks:
        for form, items, ts in zip(b["form"], b["items"], b["acceptanceDateTime"]):
            if form == "8-K" and "2.02" in str(items).split(","):
                acc.append(ts)
    if not acc:
        return []
    rows, last = [], None
    for t, d0 in reaction_sessions(acc, sessions):
        if last is not None and (d0 - last).days <= 5:
            continue
        rows.append({"cik": cik, "accepted_et": t, "event_date": d0})
        last = d0
    return rows


def reaction_sessions(accepted_utc: list[str], sessions: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """UTC 접수 시각 → (ET 시각, 첫 반응 거래일). 16:00 ET 이후면 다음 날부터, 휴장일이면 다음 거래일."""
    et = pd.to_datetime(pd.Series(accepted_utc), utc=True).dt.tz_convert("America/New_York").sort_values()
    out = []
    for t in et:
        day = t.tz_localize(None).normalize() + pd.Timedelta(days=1 if t.hour >= 16 else 0)
        i = sessions.searchsorted(day)
        if i < len(sessions):
            out.append((t.tz_localize(None), sessions[i]))
    return out


def _first_filed(units: list[dict], annual_flow: bool) -> pd.DataFrame:
    """(기말일)마다 처음 제출된 값. annual_flow 면 350~380일 구간만, 아니면 기말 잔액(instant)."""
    df = pd.DataFrame(units)
    if df.empty or "form" not in df:
        return pd.DataFrame()
    df = df[df["form"].isin(["10-K", "10-K/A", "10-KT"])]
    if annual_flow:
        if "start" not in df:
            return pd.DataFrame()
        dur = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
        df = df[dur.between(350, 380)]
    elif "start" in df:
        df = df[df["start"].isna()]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(end=pd.to_datetime(df["end"]), filed=pd.to_datetime(df["filed"]))
    return df.sort_values("filed").drop_duplicates("end")[["end", "filed", "val"]]


def fund_annual(cik: int) -> pd.DataFrame:
    """한 회사의 연간 재무 (end, filed, 항목들). 항목별로 태그 우선순위 적용."""
    p = SEC_DIR / "companyfacts" / f"CIK{cik:010d}.json.gz"
    if not p.exists():
        return pd.DataFrame()
    gaap = _load(p).get("facts", {}).get("us-gaap", {})
    parts = []
    for name, tags in {**FLOW_TAGS, **STOCK_TAGS}.items():
        flow = name in FLOW_TAGS
        got = []
        for tag in tags:
            units = gaap.get(tag, {}).get("units", {}).get("USD", [])
            v = _first_filed(units, flow)
            if not v.empty:
                got.append(v)
        if not got:
            continue
        # 같은 기말일이면 우선순위가 높은 태그의 값 (리스트 앞쪽이 먼저 concat → keep first)
        v = pd.concat(got).drop_duplicates("end", keep="first").rename(columns={"val": name, "filed": f"filed_{name}"})
        parts.append(v.set_index("end"))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1, sort=True)
    filed_cols = [c for c in out.columns if c.startswith("filed_")]
    out["filed"] = out[filed_cols].min(axis=1)  # 그 회계연도 10-K 가 처음 나온 날
    for c in filed_cols:  # 그보다 45일 넘게 늦게 처음 나온 항목은 그 시점엔 몰랐던 값 → 비운다
        out.loc[out[c] > out["filed"] + pd.Timedelta(days=LATE_ITEM_DAYS), c.removeprefix("filed_")] = float("nan")
    out = out.drop(columns=filed_cols).reset_index().rename(columns={"index": "end"})
    out["cik"] = cik
    return out


def shares(cik: int) -> pd.DataFrame:
    """표지 발행주식수 (end=기준일, filed, shares). 클래스가 여럿이면 같은 기준일 값을 합친다."""
    p = SEC_DIR / "companyfacts" / f"CIK{cik:010d}.json.gz"
    if not p.exists():
        return pd.DataFrame()
    units = _load(p).get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    df = pd.DataFrame(units)
    if df.empty:
        return df
    df = df.assign(end=pd.to_datetime(df["end"]), filed=pd.to_datetime(df["filed"]))
    df = df.groupby(["accn", "end", "filed"], as_index=False)["val"].sum()  # 다중 클래스 합
    df = df.sort_values("filed").drop_duplicates("end")[["end", "filed", "val"]].rename(columns={"val": "shares"})
    df["cik"] = cik
    return df


def build_holdout_earnings() -> None:
    """홀드아웃 1회 확인용 실적일 (DEV_END 이후 포함) → sec_earnings_dates_holdout.parquet.

    연구용 파일(sec_earnings_dates.parquet, ≤ DEV_END)은 건드리지 않는다. 백테스트는 BACKTEST_EARNINGS_PATH 로 이 파일을 쓴다.
    반응 거래일(event_date)은 평일 달력으로 근사한다 (백테스트 필터는 접수일 accepted_et 만 쓴다).
    """
    m = pd.read_csv(SEC_DIR / "ticker_cik.csv").dropna(subset=["cik"])
    m["cik"] = m["cik"].astype(int)
    sessions = pd.bdate_range("2014-01-01", pd.Timestamp.today().normalize() + pd.Timedelta(days=120))
    ev = []
    for cik in sorted(m["cik"].unique()):
        ev += earnings_dates(cik, sessions)
    ev = pd.DataFrame(ev).merge(m[["티커", "cik"]], on="cik")
    out = RESEARCH_DIR / "sec_earnings_dates_holdout.parquet"
    ev.to_parquet(out, index=False)
    print(f"[build] 홀드아웃용 실적일 {len(ev):,}건 · 마지막 {ev['accepted_et'].max()} → {out}")


def main() -> None:
    if "--holdout" in sys.argv:
        build_holdout_earnings()
        return
    m = pd.read_csv(SEC_DIR / "ticker_cik.csv").dropna(subset=["cik"])
    m["cik"] = m["cik"].astype(int)
    sessions = load_ohlcv(OHLCV_PX10Y).index
    ev, fa, sh = [], [], []
    for i, cik in enumerate(sorted(m["cik"].unique()), 1):
        ev += earnings_dates(cik, sessions)
        f = fund_annual(cik)
        if not f.empty:
            fa.append(f)
        s = shares(cik)
        if not s.empty:
            sh.append(s)
        if i % 100 == 0:
            print(f"[build] {i}/{m['cik'].nunique()}", flush=True)

    tick = m[["티커", "cik"]]
    ev = pd.DataFrame(ev).merge(tick, on="cik")
    ev = ev[ev["event_date"] <= DEV_END]
    fa = pd.concat(fa, ignore_index=True).merge(tick, on="cik")
    fa = fa[fa["filed"] <= DEV_END]
    sh = pd.concat(sh, ignore_index=True).merge(tick, on="cik")
    sh = sh[sh["filed"] <= DEV_END]
    ev.to_parquet(RESEARCH_DIR / "sec_earnings_dates.parquet", index=False)
    fa.to_parquet(RESEARCH_DIR / "sec_fund_annual.parquet", index=False)
    sh.to_parquet(RESEARCH_DIR / "sec_shares.parquet", index=False)

    # 커버리지: 연도별, 그해 패널 종목 중 실적일·재무가 있는 비율
    panel = load_panel("px10y")[["date", "티커"]]
    panel["year"] = panel["date"].dt.year
    rows = []
    for y, g in panel.groupby("year"):
        names = set(g["티커"])
        e = set(ev.loc[ev["event_date"].dt.year == y, "티커"])
        f = set(fa.loc[fa["end"].dt.year == y - 1, "티커"])
        rows.append({"연도": y, "패널종목": len(names), "실적일_있음": len(names & e) / len(names),
                     "전년_재무_있음": len(names & f) / len(names)})
    cov = pd.DataFrame(rows)
    cov.to_csv(TIER3_DIR / "sec_coverage.csv", index=False, float_format="%.3f")
    print(f"[build] 실적일 {len(ev):,}건 · 연간재무 {len(fa):,}행 · 주식수 {len(sh):,}행")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
