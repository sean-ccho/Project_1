#!/usr/bin/env python3
"""Tier 3-B 추가 단계: 반전 캔들 패턴 — "확인(컨펌)" 버전 vs 미확인 버전 (px10y 10년).

1단계 이벤트 스터디는 스크리너가 내는 패턴(확인 없음)을 3년 표본으로 봤다. 스크리너의 강세 캔들은 다음 날
확인 없이 신호가 된다 (patterns.py detect_candlestick_patterns). 교과서식 확인 = 다음 날 종가 > 패턴 봉 고가.

패턴 정의는 patterns.py 를 그대로 벡터화했다 (추세 문맥 = get_trend_context: 최근 20봉 변화율 ±7% + 20봉 안 EMA20 기울기).
  강세잉걸핑 : 전일 음봉, 당일 양봉, 당일 시가 < 전일 종가 & 당일 종가 > 전일 시가, 추세 down/sideways
  모닝스타   : 1봉 큰 음봉(몸통 > 범위 50%), 2봉 작은 몸통(< 범위 30%) & 몸통 상단 < 1봉 종가,
               3봉 양봉 & 종가 > 1봉 몸통 중간, 추세 down/sideways
  하락추세 도지 : 몸통/범위 < 0.1, 추세 down (도지는 방향이 없어 하락 추세일 때만 반전 후보로 본다)
이벤트 (그날 S&P 500 구성종목, 같은 종목·이벤트 10거래일 쿨다운):
  U* (미확인) : 이벤트일 = 패턴 완성일 t → t+1 시가 진입
  C* (확인)   : 이벤트일 = t+1, 조건 close_{t+1} > high_t → t+2 시가 진입
검정 18건 (사전 등록): {U1,C1 강세잉걸핑, U2,C2 모닝스타, U3,C3 하락추세 도지} × h {5, 10, 20}. z = bonferroni_z(18).
초과수익 = 진입 시가 → h일 뒤 시가 수익률 − 같은 구간 구성종목 동일가중. 판정은 1-3과 같다 (비용 = 왕복 0.2%).

사용법: PYTHONPATH=.:src python scripts/pattern_confirm_research.py
산출: tier3/PATTERN_CONFIRM_REPORT_px10y.md, tier3/pattern_confirm_px10y.csv
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

from research_utils import (ROUND_TRIP, TIER3_DIR, TRIAL_LOG, bonferroni_z, halves_mean, load_ohlcv, load_panel,
                            log_trials, nw_tstat, total_trials, yearly_mean)

OHLCV_PX10Y = ROOT / "data" / "cache" / "ohlcv_px10y.parquet"
HORIZONS = (5, 10, 20)
TREND_LOOKBACK, TREND_TH = 20, 0.07  # config.PATTERN_TREND_LOOKBACK, get_trend_context 의 7%
COOLDOWN = 10
FAMILY = "pattern_confirm_px10y"
EVENTS = {"U1": "강세잉걸핑 (미확인)", "C1": "강세잉걸핑 + 다음날 확인",
          "U2": "모닝스타 (미확인)", "C2": "모닝스타 + 다음날 확인",
          "U3": "하락추세 도지 (미확인)", "C3": "하락추세 도지 + 다음날 확인"}
N_FAMILY = len(EVENTS) * len(HORIZONS)


def trend_context(close: pd.DataFrame, lookback: int = TREND_LOOKBACK) -> pd.DataFrame:
    """get_trend_context 벡터화: 'up' / 'down' / 'side'. 창 안에서 다시 시작하는 EMA20(adjust=False) 기울기."""
    a = 2.0 / 21.0
    start = close.shift(lookback - 1)
    chg = close / start - 1
    # ema_end = (1-a)^(L-1)·x0 + Σ_{k=1}^{L-1} a(1-a)^(L-1-k)·x_k  (x0 = 창 첫 값)
    ema_end = (1 - a) ** (lookback - 1) * start
    for k in range(1, lookback):
        ema_end = ema_end + a * (1 - a) ** (lookback - 1 - k) * close.shift(lookback - 1 - k)
    slope = ema_end / start - 1
    out = pd.DataFrame("side", index=close.index, columns=close.columns)
    out[(chg > TREND_TH) & (slope > 0)] = "up"
    out[(chg < -TREND_TH) & (slope < 0)] = "down"
    return out.where(start.notna(), "side")


def candle_flags(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """패턴 완성일 t 기준 bool 행렬."""
    trend = trend_context(c)
    not_up = trend.isin(["down", "side"])
    rng, body = h - l, (c - o).abs()
    o1, c1 = o.shift(1), c.shift(1)
    engulf = (c1 < o1) & (c > o) & (o < c1) & (c > o1) & not_up
    o2, c2, h2, l2 = o.shift(2), c.shift(2), h.shift(2), l.shift(2)
    h1, l1 = h.shift(1), l.shift(1)
    first_bear = (c2 < o2) & ((c2 - o2).abs() > (h2 - l2) * 0.5)
    middle_small = (c1 - o1).abs() < (h1 - l1) * 0.3
    middle_gap = np.maximum(o1, c1) < c2
    third_bull = (c > o) & (c > (o2 + c2) / 2)
    morning = first_bear & middle_small & middle_gap & third_bull & not_up
    doji = (rng > 0) & (body / rng.where(rng > 0) < 0.1) & (trend == "down")
    return {"1": engulf.fillna(False), "2": morning.fillna(False), "3": doji.fillna(False)}


def cooldown_mask(flags: pd.DataFrame, n: int = COOLDOWN) -> pd.DataFrame:
    """종목별로 n거래일 안에 한 번만."""
    arr = flags.to_numpy()
    keep = np.zeros_like(arr, dtype=bool)
    for j in range(arr.shape[1]):
        last = -10**9
        for i in np.flatnonzero(arr[:, j]):
            if i - last >= n:
                keep[i, j] = True
                last = i
    return pd.DataFrame(keep, index=flags.index, columns=flags.columns)


def excess_returns(open_: pd.DataFrame, mem: pd.DataFrame, h: int) -> pd.DataFrame:
    """이벤트일 d 기준: d+1 시가 → d+1+h 시가 수익률 − 같은 구간 구성종목 동일가중 (d에 기록)."""
    fwd = open_.shift(-(1 + h)) / open_.shift(-1) - 1
    return fwd.sub(fwd.where(mem).mean(axis=1), axis=0)


def stats(ar: pd.Series, h: int) -> dict:
    """ar: (date, 티커) 인덱스 초과수익. 날짜별 평균 → NW t."""
    ar = ar.dropna()
    if ar.empty:
        return {}
    daily = ar.groupby(level=0).mean()
    yr = yearly_mean(daily)
    first, second = halves_mean(daily)
    return {"n_events": int(ar.size), "n_days": int(daily.size), "mean_ar": float(ar.mean()),
            "mean_net": float(ar.mean() - ROUND_TRIP), "hit": float((ar > 0).mean()), "t_nw": nw_tstat(daily, lag=h),
            "years_pos": int((yr > 0).sum()), "n_years": int(yr.size), "first_half": first, "second_half": second,
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


def _registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def main() -> None:
    z = bonferroni_z(N_FAMILY)
    if not _registered():  # 검정 목록은 코드에 고정 → 수익률 계산 전에 등록
        log_trials("tier3b-3p", FAMILY, N_FAMILY, [{
            "test_id": "(등록)", "판정": "등록",
            "메모": f"{{U1,C1 강세잉걸핑 · U2,C2 모닝스타 · U3,C3 하락추세 도지}} × h{list(HORIZONS)}. "
                    f"확인 = 다음날 종가 > 패턴봉 고가. z={z:.2f}"}])

    ohlcv = load_ohlcv(OHLCV_PX10Y)
    o, h_, l, c = (ohlcv.xs(f, axis=1, level="Price") for f in ("Open", "High", "Low", "Close"))
    members = load_panel("px10y")[["date", "티커"]].assign(m=True)
    mem = members.pivot(index="date", columns="티커", values="m").reindex(index=c.index, columns=c.columns)
    mem = mem.fillna(False).astype(bool)

    pats = candle_flags(o, h_, l, c)
    events: dict[str, pd.DataFrame] = {}
    for k, raw in pats.items():
        confirm = raw.shift(1, fill_value=False) & (c > h_.shift(1))  # t+1 종가 > t 고가 (t+1에 기록)
        events[f"U{k}"] = cooldown_mask(raw & mem)
        events[f"C{k}"] = cooldown_mask(confirm & mem)

    rows = []
    for h in HORIZONS:
        ar = excess_returns(o, mem, h)
        for eid, ev in events.items():
            s = ar.where(ev).stack()
            r = stats(s, h)
            rows.append({"test_id": f"{eid}|h{h}", "event": eid, "h": h, **r, "판정": grade(r, z)})
    res = pd.DataFrame(rows)
    res.to_csv(TIER3_DIR / "pattern_confirm_px10y.csv", index=False)
    log_trials("tier3b-3p", FAMILY, N_FAMILY,
               [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}", "판정": r["판정"]} for _, r in res.iterrows()])

    lines = ["# 반전 캔들 패턴 — 확인 vs 미확인 (px10y)", "",
             f"- 기간 {c.index.min().date()} ~ {c.index.max().date()}, 그날 S&P 500 구성종목, 같은 종목·이벤트 {COOLDOWN}거래일 쿨다운",
             "- 패턴 정의 = patterns.py detect_candlestick_patterns 벡터화 (추세 문맥 포함). 확인 = 다음날 종가 > 패턴봉 고가",
             "- U(미확인): 패턴일 다음날 시가 진입 · C(확인): 확인일 다음날 시가 진입. 초과수익 = 동일가중 대비",
             f"- **검정 {N_FAMILY}건 → Bonferroni z = {z:.2f}** · 누적 시도 {total_trials()}건 · 비용 후 = 왕복 {ROUND_TRIP:.1%} 차감",
             "- 판정: " + " · ".join(f"{k} {v}" for k, v in res["판정"].value_counts().items()), "",
             "| 판정 | test | 이벤트 | n | 일수 | 평균 초과(비용 전) | 비용 후 | 적중률 | t_nw | 연도+ | 앞/뒤 반 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.sort_values(["event", "h"]).iterrows():
        lines.append(f"| {r['판정']} | {r.test_id} | {EVENTS[r.event]} | {r.n_events} | {r.n_days} | {r.mean_ar:+.2%} | "
                     f"{r.mean_net:+.2%} | {r.hit:.0%} | {r.t_nw:+.2f} | {r.years_pos}/{r.n_years} | "
                     f"{r.first_half:+.2%} / {r.second_half:+.2%} |")
    lines += ["", "## 연도별 평균 초과수익", "", "| test | 연도별 |", "|---|---|"]
    lines += [f"| {r.test_id} | {r.yearly} |" for _, r in res.sort_values(["event", "h"]).iterrows()]
    (TIER3_DIR / "PATTERN_CONFIRM_REPORT_px10y.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9 + len(res)]))


if __name__ == "__main__":
    main()
