#!/usr/bin/env python3
"""Tier 3-B 2단계: 조건부 반전 (거래량 · 갭).

질문: 단기 반전(10년 중 10년 같은 부호, |t|≈1.1)이 "거래량·갭이 있는 움직임"과 "없는 움직임"에서 다르게
작동하는가? 뉴스 있는 움직임은 이어지고 뉴스 없는 움직임은 되돌아간다(Chan 2003) → 둘이 섞여 상쇄됐을 수 있다.

데이터: px10y 패널(그날 S&P 500 구성종목, fwd_ret_{h}d = t+1 시가 → t+1+h 시가) + 10년 OHLCV(≤ 2025-09-30).
조건 변수 (날짜 t 종가까지):
  ret_5d_c  = 5일 수익률                           → 날짜별 5분위 (Q1 = 하락 큰 쪽)
  abn_vol   = 최근 5일 평균 거래량 / 직전 60일 평균  → 날짜별 3분위 (V1 = 조용함)
  max_gap_5 = 최근 5일 |시가/전일종가 − 1| 최대     → 3% 기준 갭 없음/있음
두 정렬은 날짜별로 독립 (rank(method="first") 후 qcut).

검정 18건 (사전 등록, h ∈ {5, 10, 20} × 6):
  A1 V1 안에서 Q1−Q5 · A2 V3 안에서 Q1−Q5 · A3 = A1−A2
  B1 갭 없음 안에서 Q1−Q5 · B2 갭 있음 안에서 Q1−Q5 · B3 = B1−B2
롱 온리 번역 (판정 조건에 포함): 롱 셀의 동일가중 대비 초과수익 − 왕복 0.2%.
  롱 셀: A1·A3 = V1∩Q1, A2 = V3∩Q1, B1·B3 = 갭없음∩Q1, B2 = 갭있음∩Q1
판정: 후보 = t_nw ≥ z · 롱 온리 비용 후 > 0 · 연도 2/3 이상 양수 · 앞(2016~2020)/뒤(2021~) 모두 양수
      역신호 = t_nw ≤ −z · 관찰 = t_nw ≥ 2 · 롱 온리 비용 후 > 0 · 앞/뒤 양수 · 나머지 탈락
셀에 종목이 MIN_CELL 미만인 날은 뺀다.

사용법: PYTHONPATH=.:src python scripts/conditional_research.py
산출: tier3/CONDITIONAL_REPORT_px10y.md, tier3/conditional_px10y.csv (후보가 있으면 data/research/events_cond.parquet)
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

from research_utils import (RESEARCH_DIR, ROUND_TRIP, TIER3_DIR, TRIAL_LOG, bonferroni_z, load_ohlcv,
                            load_panel, log_trials, nw_tstat, total_trials, yearly_mean)

OHLCV_PX10Y = ROOT / "data" / "cache" / "ohlcv_px10y.parquet"
HORIZONS = (5, 10, 20)
GAP_TH = 0.03
MIN_CELL = 5
SPLIT_YEAR = 2021  # 앞 2016~2020 / 뒤 2021~2025-09
FAMILY = "conditional_px10y"
TESTS = ["A1", "A2", "A3", "B1", "B2", "B3"]
N_FAMILY = len(TESTS) * len(HORIZONS)
DESC = {
    "A1": "조용한 반전폭 (V1: Q1−Q5)", "A2": "거래량 급증 반전폭 (V3: Q1−Q5)", "A3": "A1 − A2",
    "B1": "갭 없는 반전폭 (갭<3%: Q1−Q5)", "B2": "갭 있는 반전폭 (갭≥3%: Q1−Q5)", "B3": "B1 − B2",
}


def conditioning_vars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """날짜 t 종가 기준: 5일 수익률, 거래량 급증(5일/직전 60일), 최근 5일 최대 갭."""
    close = ohlcv.xs("Close", axis=1, level="Price")
    open_ = ohlcv.xs("Open", axis=1, level="Price")
    vol = ohlcv.xs("Volume", axis=1, level="Price").replace(0, np.nan)
    out = pd.concat({
        "ret_5d_c": (close / close.shift(5) - 1).stack(),
        "abn_vol": (vol.rolling(5).mean() / vol.shift(5).rolling(60).mean()).stack(),
        "max_gap_5": (open_ / close.shift(1) - 1).abs().rolling(5).max().stack(),
    }, axis=1)
    out.index.names = ["date", "티커"]
    return out.reset_index()


def _bucket(s: pd.Series, dates: pd.Series, q: int) -> pd.Series:
    """날짜별 독립 분위 (1..q)."""
    r = s.groupby(dates).rank(method="first")
    n = r.groupby(dates).transform("count")
    return np.ceil(r / n * q).clip(1, q)


def _cell_mean(df: pd.DataFrame, mask: pd.Series, y: str) -> pd.Series:
    g = df.loc[mask].groupby("date")[y]
    m = g.mean()
    return m[g.size() >= MIN_CELL]


def compute(df: pd.DataFrame, h: int) -> dict[str, dict]:
    """h일 기간의 6개 검정: 날짜별 스프레드 시계열 + 롱 셀 초과수익."""
    y = f"fwd_ret_{h}d"
    d = df[df[y].notna()]
    ew = d.groupby("date")[y].mean()
    q1, q5 = d["Q"] == 1, d["Q"] == 5
    v1, v3 = d["V"] == 1, d["V"] == 3
    nogap, gap = d["max_gap_5"] < GAP_TH, d["max_gap_5"] >= GAP_TH
    cells = {
        "V1Q1": _cell_mean(d, v1 & q1, y), "V1Q5": _cell_mean(d, v1 & q5, y),
        "V3Q1": _cell_mean(d, v3 & q1, y), "V3Q5": _cell_mean(d, v3 & q5, y),
        "G0Q1": _cell_mean(d, nogap & q1, y), "G0Q5": _cell_mean(d, nogap & q5, y),
        "G1Q1": _cell_mean(d, gap & q1, y), "G1Q5": _cell_mean(d, gap & q5, y),
    }
    a1 = (cells["V1Q1"] - cells["V1Q5"]).dropna()
    a2 = (cells["V3Q1"] - cells["V3Q5"]).dropna()
    b1 = (cells["G0Q1"] - cells["G0Q5"]).dropna()
    b2 = (cells["G1Q1"] - cells["G1Q5"]).dropna()
    spreads = {"A1": a1, "A2": a2, "A3": (a1 - a2).dropna(), "B1": b1, "B2": b2, "B3": (b1 - b2).dropna()}
    long_cell = {"A1": "V1Q1", "A2": "V3Q1", "A3": "V1Q1", "B1": "G0Q1", "B2": "G1Q1", "B3": "G0Q1"}
    out = {}
    for tid, s in spreads.items():
        lc = long_cell[tid]
        lx = (cells[lc] - ew.reindex(cells[lc].index)).dropna()
        yr = yearly_mean(s)
        idx_year = pd.DatetimeIndex(s.index).year
        out[tid] = {
            "spread": s, "n_days": int(s.size), "mean_spread": float(s.mean()),
            "t_nw": nw_tstat(s, lag=h - 1),
            "years_pos": int((yr > 0).sum()), "n_years": int(yr.size),
            "first_half": float(s[idx_year < SPLIT_YEAR].mean()), "second_half": float(s[idx_year >= SPLIT_YEAR].mean()),
            "long_cell": lc, "long_excess": float(lx.mean()), "long_net": float(lx.mean() - ROUND_TRIP),
            "long_t_nw": nw_tstat(lx, lag=h - 1),
            "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items()),
        }
    return out


def grade(r: dict, z: float) -> str:
    """1~5단계 공통 판정 (비용 후 = 롱 온리 셀 기준)."""
    if not np.isfinite(r.get("t_nw", np.nan)):
        return "표본 없음"
    halves_up = r["first_half"] > 0 and r["second_half"] > 0
    if r["t_nw"] >= z and r["long_net"] > 0 and halves_up and r["years_pos"] >= math.ceil(r["n_years"] * 2 / 3):
        return "후보"
    if r["t_nw"] <= -z:
        return "역신호"
    if r["t_nw"] >= 2 and r["long_net"] > 0 and halves_up:
        return "관찰"
    return "탈락"


def _registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def main() -> None:
    z = bonferroni_z(N_FAMILY)
    if not _registered():  # 검정 목록은 코드에 고정 → 수익률 계산 전에 등록
        log_trials("tier3b-2", FAMILY, N_FAMILY, [{
            "test_id": "(등록)", "판정": "등록",
            "메모": f"A1~A3·B1~B3 × h{list(HORIZONS)}. z={z:.2f}. 롱 셀 A1·A3=V1Q1 A2=V3Q1 B1·B3=G0Q1 B2=G1Q1. "
                    f"갭 기준 {GAP_TH:.0%}, 셀 최소 {MIN_CELL}종목"}])

    panel = load_panel("px10y")
    fwd = [f"fwd_ret_{h}d" for h in HORIZONS]
    cond = conditioning_vars(load_ohlcv(OHLCV_PX10Y))
    df = panel[["date", "티커"] + fwd].merge(cond, on=["date", "티커"], how="left")
    df = df.dropna(subset=["ret_5d_c", "abn_vol", "max_gap_5"])
    df = df[np.isfinite(df["abn_vol"])]
    df["Q"] = _bucket(df["ret_5d_c"], df["date"], 5)
    df["V"] = _bucket(df["abn_vol"], df["date"], 3)
    gap_share = (df["max_gap_5"] >= GAP_TH).groupby(df["date"]).mean()
    print(f"행 {len(df):,} / 날짜 {df['date'].nunique()} / 갭≥{GAP_TH:.0%} 비율 평균 {gap_share.mean():.1%}")

    rows = []
    for h in HORIZONS:
        for tid, r in compute(df, h).items():
            r = {k: v for k, v in r.items() if k != "spread"}
            rows.append({"test_id": f"{tid}|h{h}", "test": tid, "h": h, **r, "판정": grade(r, z)})
    res = pd.DataFrame(rows)
    res.to_csv(TIER3_DIR / "conditional_px10y.csv", index=False)
    log_trials("tier3b-2", FAMILY, N_FAMILY,
               [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}", "판정": r["판정"]} for _, r in res.iterrows()])

    cands = res[res["판정"] == "후보"]
    if not cands.empty:  # 후보 셀 종목 → 0-6 시뮬 입력
        ev = []
        for _, r in cands.iterrows():
            cell = r.long_cell
            m = df["Q"] == 1
            m &= {"V1": df["V"] == 1, "V3": df["V"] == 3, "G0": df["max_gap_5"] < GAP_TH,
                  "G1": df["max_gap_5"] >= GAP_TH}[cell[:2]]
            ev.append(pd.DataFrame({"date": df.loc[m, "date"], "티커": df.loc[m, "티커"],
                                    "event_id": f"{r.test}_{cell}", "score": -df.loc[m, "ret_5d_c"]}))
        pd.concat(ev, ignore_index=True).drop_duplicates(["date", "티커", "event_id"]).to_parquet(
            RESEARCH_DIR / "events_cond.parquet")

    lines = ["# 2단계 조건부 반전 (px10y)", "",
             f"- 기간 {df['date'].min().date()} ~ {df['date'].max().date()} ({df['date'].nunique()}일), "
             f"그날 S&P 500 구성종목, 평균 {df.groupby('date').size().mean():.0f}종목/일",
             "- 스프레드 = 셀 안에서 Q1(5일 하락 큰 쪽) − Q5 의 h일 수익률 차. 양수 = 반전, 음수 = 지속",
             f"- 갭≥{GAP_TH:.0%} 비율 평균 {gap_share.mean():.1%} · 셀 최소 {MIN_CELL}종목",
             f"- **검정 {N_FAMILY}건 → Bonferroni z = {z:.2f}** · 누적 시도 {total_trials()}건",
             f"- 롱 온리 = 롱 셀의 동일가중 대비 초과수익, 비용 후 = 왕복 {ROUND_TRIP:.1%} 차감",
             "- 판정: " + " · ".join(f"{k} {v}" for k, v in res["판정"].value_counts().items()), "",
             "| 판정 | test | 설명 | 일수 | 평균 스프레드 | t_nw | 연도+ | 앞(16~20)/뒤(21~) | 롱 셀 | 롱 초과(비용 전) | 비용 후 | 롱 t_nw |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.sort_values(["h", "test"]).iterrows():
        lines.append(f"| {r['판정']} | {r.test_id} | {DESC[r.test]} | {r.n_days} | {r.mean_spread:+.2%} | {r.t_nw:+.2f} | "
                     f"{r.years_pos}/{r.n_years} | {r.first_half:+.2%} / {r.second_half:+.2%} | {r.long_cell} | "
                     f"{r.long_excess:+.2%} | {r.long_net:+.2%} | {r.long_t_nw:+.2f} |")
    lines += ["", "## 연도별 평균 스프레드", "", "| test | 연도별 |", "|---|---|"]
    lines += [f"| {r.test_id} | {r.yearly} |" for _, r in res.sort_values(["h", "test"]).iterrows()]
    (TIER3_DIR / "CONDITIONAL_REPORT_px10y.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9 + len(res)]))


if __name__ == "__main__":
    main()
