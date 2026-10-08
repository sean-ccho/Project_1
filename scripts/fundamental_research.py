#!/usr/bin/env python3
"""Tier 3-B 추가: 재무 팩터 (SEC XBRL, 시점 기준) — 가격·거래량 밖의 신호를 한 번 본다.

매월 마지막 거래일 d 에 그날 S&P 500 구성종목(px10y 패널)마다, **d 이전에 제출된** 10-K 값만으로 팩터를 만든다.
  BM  = 자본총계 / 시가총액            (가치, 자본 > 0 만)
  EP  = 순이익 / 시가총액              (가치)
  GPA = 매출총이익 / 자산              (수익성, Novy-Marx 2013. 매출총이익 = GrossProfit 또는 매출 − 매출원가)
  AG  = −(자산 / 전년 자산 − 1)         (자산 증가가 낮을수록 좋음, Cooper·Gulen·Schill 2008)
  ACC = −(순이익 − 영업현금흐름) / 자산  (발생액이 낮을수록 좋음, Sloan 1996)
  ISS = −log(발행주식수 / 1년 전)       (순발행이 낮을수록 좋음, 분할 보정)
  COMP = 위 6개 백분위 평균 (4개 이상 있을 때)
시가총액 = 분할만 반영된 종가(raw_close_px10y) × 표지 주식수 × (그 기준일 이후 분할 비율 곱).
수익률: d 다음날 시가 진입 → h 거래일 뒤 시가 청산 (조정 시가, ohlcv_px10y). h ∈ {21, 63}.
검정 14건 (7 신호 × 2 기간, 사전 등록): 상위 20% 평균 − 그날 팩터가 있는 구성종목 평균. 월별 → nw_tstat(lag 1 / 3).
  판정은 1-3과 같다 (비용 = 보유 1회당 왕복 0.2%, 보수적으로 전부 교체 가정).

사용법: PYTHONPATH=.:src python scripts/fundamental_research.py
산출: tier3/FUNDAMENTAL_REPORT_px10y.md, tier3/fundamental_px10y.csv, data/research/fund_factors_px10y.parquet
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from research_utils import (ROUND_TRIP, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, bonferroni_z, halves_mean, load_ohlcv,
                            load_panel, log_trials, nw_tstat, total_trials, yearly_mean)

OHLCV_PX10Y = ROOT / "data" / "cache" / "ohlcv_px10y.parquet"
FAMILY = "fundamental_px10y_v2"  # v1 은 커버리지 버그(늦게 낸 옛 연도 정정을 최신으로 선택)로 무효, 같은 검정 목록 재실행
FACTORS = ["BM", "EP", "GPA", "AG", "ACC", "ISS"]
SIGNALS = FACTORS + ["COMP"]
HORIZONS = {21: 1, 63: 3}  # h → NW lag (월별 표본)
N_TESTS = len(SIGNALS) * len(HORIZONS)
TOP_Q = 0.2
STALE_DAYS = 550  # 마지막 회계연도 말이 이보다 오래되면 버린다


def _split_after(splits: pd.DataFrame) -> callable:
    """(티커, 날짜) → 그 날짜 **이후** 분할 비율의 곱."""
    by = {t: g.sort_values("date") for t, g in splits.groupby("티커")}

    def g(ticker: str, when: pd.Timestamp) -> float:
        s = by.get(ticker)
        if s is None:
            return 1.0
        r = s.loc[s["date"] > when, "ratio"].astype(float)
        return float(r.prod()) if len(r) else 1.0
    return g


def _asof(left: pd.DataFrame, right: pd.DataFrame, on_left: str, on_right: str) -> pd.DataFrame:
    """티커별로 on_left 보다 **엄격히 이전**의 마지막 right 행."""
    l = left.assign(**{on_left: left[on_left].astype("datetime64[ns]")}).sort_values(on_left)
    r = right.assign(**{on_right: right[on_right].astype("datetime64[ns]")}).dropna(subset=[on_right]).sort_values(on_right)
    return pd.merge_asof(l, r, left_on=on_left, right_on=on_right, by="티커", allow_exact_matches=False)


def latest_fiscal_year(rebal: pd.DataFrame, fa: pd.DataFrame) -> pd.DataFrame:
    """(날짜, 티커)마다 그날 **이전에 공개된** 회계연도 중 기말이 가장 최근인 행. 늦게 낸 옛 연도 정정이 최신으로 잡히지 않게."""
    m = rebal.merge(fa, on="티커", how="inner")
    m = m[m["filed"].astype("datetime64[ns]") < m["date"].astype("datetime64[ns]")]
    m = m.sort_values("fy_end").drop_duplicates(["date", "티커"], keep="last")
    return rebal.merge(m.drop(columns=[c for c in rebal.columns if c not in ("date", "티커")]),
                       on=["date", "티커"], how="left")


def build_factors() -> pd.DataFrame:
    """월말 × 종목 팩터 표."""
    members = load_panel("px10y")[["date", "티커"]]
    sessions = pd.DatetimeIndex(sorted(members["date"].unique()))
    month_end = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).max()
    rebal = members[members["date"].isin(set(month_end))].copy()

    fa = pd.read_parquet(RESEARCH_DIR / "sec_fund_annual.parquet")
    fa["gp"] = fa["gross_profit"].fillna(fa["revenue"] - fa["cogs"])
    # 전년 자산 (같은 회사, 기말이 330~400일 전)
    prev = fa[["티커", "end", "assets"]].rename(columns={"end": "prev_end", "assets": "prev_assets"})
    fa = fa.merge(prev, on="티커", how="left")
    lag = (fa["end"] - fa["prev_end"]).dt.days
    fa.loc[~lag.between(330, 400), "prev_assets"] = np.nan
    fa = fa.sort_values("prev_assets", na_position="last").drop_duplicates(["티커", "end"]).drop(columns="prev_end")

    sh = pd.read_parquet(RESEARCH_DIR / "sec_shares.parquet")
    splits = pd.read_parquet(RESEARCH_DIR / "splits_px10y.parquet")
    g_after = _split_after(splits)
    sh["shares_adj"] = [s * g_after(t, e) for t, e, s in zip(sh["티커"], sh["end"], sh["shares"])]  # 현재 분할 기준 주식수

    raw = pd.read_parquet(RESEARCH_DIR / "raw_close_px10y.parquet")
    raw.index = pd.to_datetime(raw.index)
    raw_long = raw.stack().rename("close_sa").reset_index()
    raw_long.columns = ["date", "티커", "close_sa"]
    rebal = rebal.merge(raw_long, on=["date", "티커"], how="left")

    x = latest_fiscal_year(rebal, fa.drop(columns=["cik"]).rename(columns={"end": "fy_end"}))
    stale = ~((x["date"] - x["fy_end"]).dt.days <= STALE_DAYS)
    x.loc[stale, ["revenue", "cogs", "gross_profit", "gp", "net_income", "cfo", "assets", "equity", "prev_assets"]] = np.nan
    s1 = _asof(x[["date", "티커"]], sh[["티커", "filed", "shares_adj"]].rename(columns={"filed": "sfiled"}), "date", "sfiled")
    x = x.merge(s1[["date", "티커", "shares_adj", "sfiled"]], on=["date", "티커"], how="left")
    x.loc[~((x["date"] - x["sfiled"]).dt.days <= 400), "shares_adj"] = np.nan  # 오래된 주식수는 버림 (행은 유지)
    lagged = x[["date", "티커"]].assign(d_prev=x["date"] - pd.Timedelta(days=365))
    s0 = _asof(lagged, sh[["티커", "filed", "shares_adj"]].rename(columns={"filed": "sfiled0", "shares_adj": "shares_prev"}),
               "d_prev", "sfiled0")
    x = x.merge(s0[["date", "티커", "shares_prev", "sfiled0"]], on=["date", "티커"], how="left")
    x.loc[~(((x["date"] - pd.Timedelta(days=365)) - x["sfiled0"]).dt.days <= 400), "shares_prev"] = np.nan

    mcap = x["close_sa"] * x["shares_adj"]
    x["mcap"] = mcap
    x["BM"] = np.where(x["equity"] > 0, x["equity"] / mcap, np.nan)
    x["EP"] = x["net_income"] / mcap
    x["GPA"] = x["gp"] / x["assets"]
    x["AG"] = -(x["assets"] / x["prev_assets"] - 1)
    x["ACC"] = -(x["net_income"] - x["cfo"]) / x["assets"]
    x["ISS"] = -np.log(x["shares_adj"] / x["shares_prev"])
    x = x.replace([np.inf, -np.inf], np.nan)
    x.loc[~(x["mcap"] > 0), ["BM", "EP"]] = np.nan
    pct = x.groupby("date")[FACTORS].rank(pct=True)
    x["COMP"] = pct.mean(axis=1).where(pct.notna().sum(axis=1) >= 4)

    o = load_ohlcv(OHLCV_PX10Y).xs("Open", axis=1, level="Price")
    for h in HORIZONS:
        fwd = (o.shift(-(1 + h)) / o.shift(-1) - 1).stack().rename(f"fwd_{h}").reset_index()
        fwd.columns = ["date", "티커", f"fwd_{h}"]
        x = x.merge(fwd, on=["date", "티커"], how="left")
    keep = ["date", "티커", "mcap", *SIGNALS, *[f"fwd_{h}" for h in HORIZONS]]
    return x[keep]


def test(x: pd.DataFrame, sig: str, h: int) -> tuple[dict, pd.Series]:
    """월별 (상위 20% − 전체) 평균 초과수익 · IC."""
    d = x[["date", sig, f"fwd_{h}"]].dropna()
    rows = []
    for date, g in d.groupby("date"):
        if len(g) < 50:
            continue
        cut = g[sig].quantile(1 - TOP_Q)
        top, bot = g[g[sig] >= cut], g[g[sig] <= g[sig].quantile(TOP_Q)]
        rows.append({"date": date, "ex": top[f"fwd_{h}"].mean() - g[f"fwd_{h}"].mean(),
                     "spread": top[f"fwd_{h}"].mean() - bot[f"fwd_{h}"].mean(),
                     "ic": g[sig].corr(g[f"fwd_{h}"], method="spearman"), "n": len(g)})
    s = pd.DataFrame(rows).set_index("date")
    ex = s["ex"]
    yr = yearly_mean(ex)
    first, second = halves_mean(ex)
    r = {"n_months": int(len(s)), "n_avg": float(s["n"].mean()), "mean_ex": float(ex.mean()),
         "mean_net": float(ex.mean() - ROUND_TRIP), "t_nw": nw_tstat(ex, lag=HORIZONS[h]),
         "spread": float(s["spread"].mean()), "t_spread": nw_tstat(s["spread"], lag=HORIZONS[h]),
         "ic": float(s["ic"].mean()), "t_ic": nw_tstat(s["ic"], lag=HORIZONS[h]),
         "years_pos": int((yr > 0).sum()), "n_years": int(yr.size), "first_half": first, "second_half": second,
         "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items())}
    return r, ex


def grade(r: dict, z: float) -> str:
    """1-3 공통 판정."""
    if not np.isfinite(r.get("t_nw", np.nan)):
        return "표본 없음"
    halves_up = r["first_half"] > 0 and r["second_half"] > 0
    if r["t_nw"] >= z and r["mean_net"] > 0 and halves_up and r["years_pos"] >= math.ceil(r["n_years"] * 2 / 3):
        return "후보"
    if r["t_nw"] <= -z:
        return "역신호"
    if r["t_nw"] >= 2 and r["mean_net"] > 0 and halves_up:
        return "관찰"
    return "탈락"


def _registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def main() -> None:
    z = bonferroni_z(N_TESTS)
    if not _registered():
        log_trials("tier3b-fund", FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록",
                    "메모": f"SEC 10-K 시점 기준 재무 팩터 {SIGNALS} × h{list(HORIZONS)}, 월말 리밸런스, 상위 20% − 구성종목 평균. z={z:.2f}"}])
    x = build_factors()
    x.to_parquet(RESEARCH_DIR / "fund_factors_px10y.parquet", index=False)
    cover = x.groupby(x["date"].dt.year)[SIGNALS].apply(lambda g: g.notna().mean())

    rows = []
    for sig in SIGNALS:
        for h in HORIZONS:
            r, _ = test(x, sig, h)
            rows.append({"test_id": f"{sig}|h{h}", "signal": sig, "h": h, **r, "판정": grade(r, z)})
    res = pd.DataFrame(rows)
    res.to_csv(TIER3_DIR / "fundamental_px10y.csv", index=False)
    log_trials("tier3b-fund", FAMILY, N_TESTS,
               [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}", "판정": r["판정"]} for _, r in res.iterrows()])

    lines = ["# 재무 팩터 (SEC XBRL 시점 기준, px10y)", "",
             f"- 월말 {x['date'].nunique()}회 · {x['date'].min().date()} ~ {x['date'].max().date()} · 그날 S&P 500 구성종목",
             "- 팩터는 그날 **이전에 제출된** 10-K 값만 사용 (기말일마다 처음 제출된 값, 정정 무시). 시가총액 = 분할만 반영된 종가 × 표지 주식수(분할 보정)",
             f"- 검정 = 상위 20% − 그날 팩터가 있는 구성종목 평균, 다음날 시가 진입. **검정 {N_TESTS}건 → z = {z:.2f}** · 누적 시도 {total_trials()}건",
             f"- 비용 후 = 보유 1회당 왕복 {ROUND_TRIP:.1%} 차감 (전부 교체 가정, 보수적)",
             "- 판정: " + " · ".join(f"{k} {v}" for k, v in res["판정"].value_counts().items()), "",
             "| 판정 | test | 월 | 평균 종목 | 상위20% 초과 | 비용 후 | t_nw | 상위−하위 | t | IC | t_IC | 연도+ | 앞/뒤 반 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r['판정']} | {r.test_id} | {r.n_months} | {r.n_avg:.0f} | {r.mean_ex:+.2%} | {r.mean_net:+.2%} | "
                     f"{r.t_nw:+.2f} | {r.spread:+.2%} | {r.t_spread:+.2f} | {r.ic:+.3f} | {r.t_ic:+.2f} | "
                     f"{r.years_pos}/{r.n_years} | {r.first_half:+.2%} / {r.second_half:+.2%} |")
    lines += ["", "## 연도별 평균 초과수익 (상위 20% − 전체)", "", "| test | 연도별 |", "|---|---|"]
    lines += [f"| {r.test_id} | {r.yearly} |" for _, r in res.iterrows()]
    lines += ["", "## 연도별 팩터 커버리지 (구성종목 중 값이 있는 비율)", "",
              "| 연도 | " + " | ".join(SIGNALS) + " |", "|---" * (len(SIGNALS) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(f"{v:.0%}" for v in row) + " |" for y, row in cover.iterrows()]
    (TIER3_DIR / "FUNDAMENTAL_REPORT_px10y.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9 + len(res)]))


if __name__ == "__main__":
    main()
