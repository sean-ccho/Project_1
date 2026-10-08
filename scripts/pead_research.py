#!/usr/bin/env python3
"""Tier 3-B 3단계: 실적 이벤트 (PEAD) — 3-1 대리 이벤트.

실적 발표 후 크게 반응한 종목이 몇 주간 같은 방향으로 가는가(PEAD)? 실적일 데이터 없이 "실적처럼 보이는 날"로 먼저 본다.
  이벤트일 t : |open_t / close_{t-1} − 1| ≥ 4% 이고 volume_t ≥ 3 × 직전 20일 평균, t에 S&P 500 구성종목(px10y 패널 행)
  방향       : t일 수익률(close_t / close_{t-1} − 1) − 같은 날 구성종목 동일가중 → 부호로 상승/하락
  진입·청산  : t+1 시가 → t+1+h 시가, h ∈ {5, 10, 20, 40, 60}. 같은 구간 구성종목 동일가중 대비 초과수익
  검정 10건  : {상승 이벤트, 하락 이벤트} × 5 기간. z = bonferroni_z(10) ≈ 2.81. 날짜별 평균 → nw_tstat(lag=h)
  판정 (1-3과 같음): 후보 = t_nw ≥ z · 비용 후 > 0 · 연도 2/3 이상 양수 · 앞/뒤 반 양수 / 역신호 = t_nw ≤ −z
    · 상승 이벤트 후보 = 상승 지속(PEAD) 매수 · 하락 이벤트 후보 = 하락 후 반등 매수
    · 하락 이벤트 역신호 = 하락 지속 → 매수 회피 필터 후보
한계: 실적이 아닌 뉴스(M&A·소송·애널리스트)도 섞인다. 가격은 ≤ 2025-09-30 (t+1+h가 넘는 이벤트는 제외).

사용법: PYTHONPATH=.:src python scripts/pead_research.py --proxy
산출: tier3/PEAD_PROXY_REPORT_px10y.md, tier3/pead_proxy_px10y.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from research_utils import (ROUND_TRIP, TIER3_DIR, TRIAL_LOG, bonferroni_z, halves_mean, load_ohlcv, load_panel,
                            log_trials, nw_tstat, total_trials, yearly_mean)

OHLCV_PX10Y = ROOT / "data" / "cache" / "ohlcv_px10y.parquet"
HORIZONS = (5, 10, 20, 40, 60)
GAP_MIN, VOL_MULT = 0.04, 3.0
PROXY_FAMILY = "pead_proxy_px10y"
N_PROXY = 2 * len(HORIZONS)


def proxy_events(ohlcv: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """대리 실적 이벤트 + 방향 + h일 초과수익 (long: date, 티커, side, ar_{h})."""
    close = ohlcv.xs("Close", axis=1, level="Price")
    open_ = ohlcv.xs("Open", axis=1, level="Price")
    vol = ohlcv.xs("Volume", axis=1, level="Price").replace(0, np.nan)
    gap = (open_ / close.shift(1) - 1).abs()
    vol_mult = vol / vol.shift(1).rolling(20).mean()
    ret1 = close / close.shift(1) - 1

    # 구성종목 마스크 (그날 패널에 있는 종목)
    mem = members.assign(m=True).pivot(index="date", columns="티커", values="m")
    mem = mem.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)

    ew_ret1 = ret1.where(mem).mean(axis=1)
    rel1 = ret1.sub(ew_ret1, axis=0)
    is_event = (gap >= GAP_MIN) & (vol_mult >= VOL_MULT) & mem & rel1.notna()

    ev = is_event.stack()
    ev = ev[ev].index.to_frame(index=False)
    ev.columns = ["date", "티커"]
    ev["rel_ret"] = rel1.stack().reindex(pd.MultiIndex.from_frame(ev)).to_numpy()
    ev["side"] = np.where(ev["rel_ret"] > 0, "up", "down")
    ev["gap"] = gap.stack().reindex(pd.MultiIndex.from_frame(ev[["date", "티커"]])).to_numpy()
    for h in HORIZONS:
        fwd = open_.shift(-(1 + h)) / open_.shift(-1) - 1          # t+1 시가 → t+1+h 시가 (t에 기록)
        ew = fwd.where(mem).mean(axis=1)                            # 같은 구간 구성종목 동일가중
        ar = fwd.sub(ew, axis=0)
        ev[f"ar_{h}"] = ar.stack().reindex(pd.MultiIndex.from_frame(ev[["date", "티커"]])).to_numpy()
    return ev


def stats(ev: pd.DataFrame, h: int) -> dict:
    """날짜별 평균 초과수익 → NW t · 연도별 · 앞/뒤 반."""
    ar = ev[f"ar_{h}"].dropna()
    if ar.empty:
        return {}
    daily = ar.groupby(ev.loc[ar.index, "date"]).mean()
    yr = yearly_mean(daily)
    first, second = halves_mean(daily)
    return {"n_events": int(ar.size), "n_days": int(daily.size), "mean_ar": float(ar.mean()),
            "mean_net": float(ar.mean() - ROUND_TRIP), "median_ar": float(ar.median()),
            "hit": float((ar > 0).mean()), "t_nw": nw_tstat(daily, lag=h),
            "years_pos": int((yr > 0).sum()), "n_years": int(yr.size),
            "first_half": first, "second_half": second,
            "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items())}


def grade(r: dict, z: float) -> str:
    """1-3 공통 판정."""
    if not r or not np.isfinite(r.get("t_nw", np.nan)):
        return "표본 없음"
    halves_up = r["first_half"] > 0 and r["second_half"] > 0
    if r["t_nw"] >= z and r["mean_net"] > 0 and halves_up and r["years_pos"] >= math.ceil(r["n_years"] * 2 / 3):
        return "후보"
    if r["t_nw"] <= -z:
        return "역신호"
    if r["t_nw"] >= 2 and r["mean_net"] > 0 and halves_up:
        return "관찰"
    return "탈락"


def _registered(family: str) -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == family) & (log["판정"] == "등록")).any())


def run_proxy() -> None:
    z = bonferroni_z(N_PROXY)
    if not _registered(PROXY_FAMILY):  # 검정 목록은 코드에 고정 → 수익률 계산 전에 등록
        log_trials("tier3b-3", PROXY_FAMILY, N_PROXY, [{
            "test_id": "(등록)", "판정": "등록",
            "메모": f"대리 실적 이벤트(갭≥{GAP_MIN:.0%} & 거래량≥{VOL_MULT:.0f}×20일) {{up,down}} × h{list(HORIZONS)}. z={z:.2f}"}])

    members = load_panel("px10y")[["date", "티커"]]
    ev = proxy_events(load_ohlcv(OHLCV_PX10Y), members)
    print(f"이벤트 {len(ev):,}건 (상승 {int((ev.side == 'up').sum()):,} · 하락 {int((ev.side == 'down').sum()):,}), "
          f"날짜 {ev['date'].nunique()} · 종목 {ev['티커'].nunique()}")

    rows = []
    for side in ("up", "down"):
        for h in HORIZONS:
            r = stats(ev[ev["side"] == side], h)
            rows.append({"test_id": f"{side}|h{h}", "side": side, "h": h, **r, "판정": grade(r, z)})
    res = pd.DataFrame(rows)
    res.to_csv(TIER3_DIR / "pead_proxy_px10y.csv", index=False)
    log_trials("tier3b-3", PROXY_FAMILY, N_PROXY,
               [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}", "판정": r["판정"]} for _, r in res.iterrows()])

    per_year = ev.groupby([ev["date"].dt.year, "side"]).size().unstack(fill_value=0)
    lines = ["# 3단계 PEAD — 3-1 대리 이벤트 (px10y)", "",
             f"- 이벤트: 시가 갭 ≥ {GAP_MIN:.0%} & 거래량 ≥ {VOL_MULT:.0f}× 직전 20일 평균, 그날 S&P 500 구성종목",
             f"- 방향: 이벤트일 수익률 − 구성종목 동일가중 의 부호. 진입 t+1 시가 → 청산 t+1+h 시가, 동일가중 대비 초과수익",
             f"- 이벤트 {len(ev):,}건 (상승 {int((ev.side == 'up').sum()):,} · 하락 {int((ev.side == 'down').sum()):,}), "
             f"{ev['date'].min().date()} ~ {ev['date'].max().date()}",
             f"- **검정 {N_PROXY}건 → Bonferroni z = {z:.2f}** · 누적 시도 {total_trials()}건 · 비용 후 = 왕복 {ROUND_TRIP:.1%} 차감",
             "- 해석: up 후보 = 상승 지속 매수 · down 후보 = 하락 후 반등 매수 · down 역신호 = 하락 지속(매수 회피)",
             "- 판정: " + " · ".join(f"{k} {v}" for k, v in res["판정"].value_counts().items()), "",
             "| 판정 | test | n | 일수 | 평균 초과(비용 전) | 비용 후 | 중앙값 | 적중률 | t_nw | 연도+ | 앞/뒤 반 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r['판정']} | {r.test_id} | {r.n_events} | {r.n_days} | {r.mean_ar:+.2%} | {r.mean_net:+.2%} | "
                     f"{r.median_ar:+.2%} | {r.hit:.0%} | {r.t_nw:+.2f} | {r.years_pos}/{r.n_years} | "
                     f"{r.first_half:+.2%} / {r.second_half:+.2%} |")
    lines += ["", "## 연도별 평균 초과수익", "", "| test | 연도별 |", "|---|---|"]
    lines += [f"| {r.test_id} | {r.yearly} |" for _, r in res.iterrows()]
    lines += ["", "## 연도별 이벤트 수", "", "| 연도 | 상승 | 하락 |", "|---|---|---|"]
    lines += [f"| {y} | {r.get('up', 0)} | {r.get('down', 0)} |" for y, r in per_year.iterrows()]
    (TIER3_DIR / "PEAD_PROXY_REPORT_px10y.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:11 + len(res)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", action="store_true", help="3-1 대리 이벤트 분석")
    args = ap.parse_args()
    if args.proxy:
        run_proxy()
    else:
        raise SystemExit("--proxy 만 구현됨 (3-2·3-3 실적일 분석은 EDGAR 수집 후)")


if __name__ == "__main__":
    main()
