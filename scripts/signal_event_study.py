#!/usr/bin/env python3
"""Tier 3-B 1단계: 스크리너 출력 이벤트 스터디.

질문: 스크리너의 각 출력(패턴 · 전략구분 · 판단 · buy_signal · 하드필터 통과 · CCS 순위)이 뜬 뒤
h일 동안 같은 날 유니버스(동일가중)보다 더 오르는가?

세 단계로 나눠 실행한다 (사전 등록: 수익률을 보기 전에 검정 목록을 고정한다).
  --build   : 패널 날짜마다 실거래 순서(유동성 → 중립화 → 섹터 강도 → 신호 → CCS)를 재현 → signals_{set}.parquet
  --count   : 이벤트 시작일 수만 센다 (수익률 안 봄) → family 확정 · TRIAL_LOG 등록
  --analyze : family 이벤트의 h일 초과수익 · NW t · 연도별 · 판정

실거래와 다른 점 (고친 규칙 = 6단계에서 실거래·백테스트에 반영할 것):
  - 섹터: 패널 `섹터`는 config.SECTOR_MAP 기준이라 78%가 Unknown → data/universe/sector_map.csv(GICS)로 채우고,
    yfinance 이름은 GICS로 매핑(SECTOR_ALIASES). 남는 Unknown ≈ 3% (실거래 ≈ 2%)
  - 강한 섹터: 섹터 ETF 10년 일봉으로 계산, Unknown 은 통과 (main.py 규칙)
  - 유동성: 최근20일평균거래대금 ≥ LIQUIDITY_DOLLAR_MIN 만 (분위수 기준은 단면 구성에 따라 달라 쓰지 않음)
  - 유니버스: 그날 S&P 500 구성종목 (실거래는 S&P 500 + 나스닥)
  - 보유 종목 없음 가정, 펀더멘털·실적일 열 없음 (해당 하드필터는 기본값으로 통과)

사용법:
  PYTHONPATH=.:src python scripts/signal_event_study.py --set pit --build --limit 20
  PYTHONPATH=.:src python scripts/signal_event_study.py --set pit --build
  PYTHONPATH=.:src python scripts/signal_event_study.py --set pit --count
  PYTHONPATH=.:src python scripts/signal_event_study.py --set pit --analyze
중간 산출(커밋 안 함): data/research/{signals,events}_{set}.parquet, data/research/etf_ohlcv.parquet
결과(커밋): tier3/EVENT_COUNTS_{set}.csv, EVENT_FAMILY_{set}.json, EVENT_STUDY_{set}.md, event_study_{set}.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# src를 맨 앞에: scripts/paper_trading.py 가 src/paper_trading 패키지를 가리지 않게
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator

from factor_research import ic_by_date
from paper_trading.candidate_selector import select_top_candidates
from research_utils import (DEV_END, RESEARCH_DIR, ROUND_TRIP, TIER3_DIR, bonferroni_z, halves_mean,
                            load_panel, log_trials, nw_tstat, total_trials, yearly_mean)
from screener.backtest import _prepare_price_map
from screener.config import LIQUIDITY_DOLLAR_MIN, SECTOR_ETFS
from screener.processing import apply_neutralization
from screener.sector_rotation import get_strong_sectors
from screener.signals import _compute_buy_signal, attach_signals_and_sort

ETF_FILE = RESEARCH_DIR / "etf_ohlcv.parquet"
SECTOR_CSV = ROOT / "data" / "universe" / "sector_map.csv"
ETF_START, ETF_END = "2015-01-01", "2025-10-01"  # yfinance end 배타적 → 2025-09-30까지 (홀드아웃은 받지 않는다)
# yfinance 섹터명 → GICS(SECTOR_ETFS 키). 6단계에서 sector_rotation.py 로 옮긴다
SECTOR_ALIASES = {
    "Technology": "Information Technology", "Healthcare": "Health Care",
    "Financial Services": "Financials", "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples", "Basic Materials": "Materials",
}
REQUIRED = ["시장", "섹터", "close", "ema20", "ema50", "ema200", "macd_hist", "RSI", "volume", "volume_ma20",
            "adx", "obv", "obv_ma20", "obv_mom_5", "반등스코어", "5일수익률", "bollinger_pband", "트렌드점수",
            "거래량Z(20)", "최근20일평균거래대금", "일봉패턴", "주봉패턴", "월봉패턴"]
KEEP = ["date", "티커", "섹터", "최근20일평균거래대금", "in_strong_sector", "buy_signal_raw", "buy_signal",
        "판단", "전략구분", "매수적합도", "바닥반등_적합도", "모멘텀_적합도", "hard_pass", "ccs", "top1", "top3",
        "regime", "spy_bear", "일봉패턴", "주봉패턴", "월봉패턴", "5일수익률", "거래량Z(20)", "RSI"]

HORIZONS = (5, 10, 20)
TOP_JUDGMENTS = {"1. 매수 후보", "1. 저점 반등"}
COOLDOWN, MIN_EVENTS, MIN_DAYS = 10, 300, 100
SCORE_COLS = {"C1_매수적합도": "매수적합도", "C2_바닥반등_적합도": "바닥반등_적합도", "C3_모멘텀_적합도": "모멘텀_적합도"}


# ── build ──────────────────────────────────────────────────────────────

def _etf_map() -> dict[str, pd.DataFrame]:
    """섹터 ETF + SPY 일봉 (2015-01 ~ 2025-09-30). 처음 한 번 받아 data/research 에 보관."""
    if not ETF_FILE.exists():
        import yfinance as yf

        tickers = list(SECTOR_ETFS.values()) + ["SPY"]
        raw = yf.download(tickers, start=ETF_START, end=ETF_END, interval="1d", auto_adjust=True,
                          threads=True, progress=False, group_by="ticker")
        raw = raw.loc[:DEV_END]
        raw.columns.names = ["Ticker", "Price"]
        ETF_FILE.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(ETF_FILE)
        print(f"ETF 저장: {ETF_FILE.name} {raw.shape} {raw.index.min().date()} ~ {raw.index.max().date()}")
    raw = pd.read_parquet(ETF_FILE)
    raw.index = pd.to_datetime(raw.index)
    return _prepare_price_map(raw.loc[:DEV_END])


def _spy_frame(spy: pd.DataFrame) -> pd.DataFrame:
    """SPY 종가·EMA (features.py와 같은 EMAIndicator)."""
    close = spy["Close"].dropna()
    return pd.DataFrame({"close": close, "현재가격": close,
                         "ema50": EMAIndicator(close, window=50).ema_indicator(),
                         "ema200": EMAIndicator(close, window=200).ema_indicator()})


def _sector_series(panel: pd.DataFrame) -> pd.Series:
    """패널 섹터(Unknown 많음)를 sector_map.csv(GICS)로 채우고 yfinance 이름을 GICS로 바꾼다."""
    smap = pd.read_csv(SECTOR_CSV).set_index("ticker")["sector"]
    sec = panel["섹터"].where(panel["섹터"].notna() & (panel["섹터"] != "Unknown"), panel["티커"].map(smap))
    return sec.fillna("Unknown").map(lambda s: SECTOR_ALIASES.get(s, s))


def _strong_flags(sectors: pd.Series, etf_map: dict[str, pd.DataFrame], d: pd.Timestamp) -> pd.Series:
    """실거래 규칙: 강한 섹터 또는 Unknown 이면 True."""
    etf_data = {t: f.loc[:d] for t, f in etf_map.items() if t != "SPY"}
    strong = get_strong_sectors(etf_data, etf_map["SPY"].loc[:d])
    return sectors.isin(strong) | (sectors == "Unknown")


def build(set_name: str, limit: int | None) -> None:
    """패널 → 날짜별 신호 재계산 → signals_{set}.parquet."""
    logging.getLogger().setLevel(logging.WARNING)  # 하드필터 로그가 많다
    panel = load_panel(set_name)
    missing = [c for c in REQUIRED if c not in panel.columns]
    if missing:
        raise SystemExit(f"패널에 필요한 열 없음: {missing}")
    panel["섹터"] = _sector_series(panel)
    print(f"섹터 Unknown 비율: {(panel['섹터'] == 'Unknown').mean():.1%}")

    etf_map = _etf_map()
    spy = _spy_frame(etf_map["SPY"])
    fwd_cols = [c for c in panel.columns if c.startswith("fwd_ret_")]
    frames, t0 = [], time.time()
    dates = sorted(panel["date"].unique())
    for n, (d, day) in enumerate(panel.groupby("date")):
        if limit and n >= limit:
            break
        if d not in spy.index:
            print(f"  {d.date()}: SPY 없음, 건너뜀")
            continue
        day = day[day["최근20일평균거래대금"] >= LIQUIDITY_DOLLAR_MIN].copy()
        spy_row = {"티커": "SPY", "시장": "US", "섹터": "Unknown", "최근20일평균거래대금": 1e12,
                   **spy.loc[d].to_dict()}
        day = pd.concat([day, pd.DataFrame([spy_row])], ignore_index=True)
        neutral = apply_neutralization(day)
        neutral["in_strong_sector"] = _strong_flags(neutral["섹터"], etf_map, d)
        strong = neutral["in_strong_sector"].copy()
        raw_buy, _ = _compute_buy_signal(neutral)
        ranked = attach_signals_and_sort(neutral)  # 정렬돼도 인덱스 라벨은 유지된다
        ranked["in_strong_sector"] = strong.reindex(ranked.index)
        ranked["buy_signal_raw"] = raw_buy.reindex(ranked.index).fillna(False).astype(bool)
        picks, debug = select_top_candidates(ranked, [], k=3)
        scores = debug.get("all_scores", {})
        top = [p["ticker"] for p in picks if p["ticker"] != "SPY"]
        ranked["hard_pass"] = ranked["티커"].isin(list(scores))
        ranked["ccs"] = ranked["티커"].map(lambda t: scores.get(t, {}).get("ccs"))
        ranked["top1"] = ranked["티커"].eq(top[0]) if top else False
        ranked["top3"] = ranked["티커"].isin(top)
        ranked["regime"] = debug.get("regime", "")
        ranked["spy_bear"] = bool(spy.at[d, "close"] < spy.at[d, "ema200"])
        ranked = ranked[ranked["티커"] != "SPY"].assign(date=d)
        frames.append(ranked[[c for c in KEEP + fwd_cols if c in ranked.columns]])
        if (n + 1) % 50 == 0 or n == 0:
            el = time.time() - t0
            print(f"  {n + 1}/{len(dates)} {d.date()} · {el:.0f}s (예상 총 {el / (n + 1) * len(dates) / 60:.0f}분)",
                  flush=True)
    sig = pd.concat(frames, ignore_index=True)
    out = RESEARCH_DIR / f"signals_{set_name}{'_limit' if limit else ''}.parquet"
    sig.to_parquet(out)
    print(f"저장: {out.name} {sig.shape}")
    _print_buy_ratio(sig)


def _buy_ratio(sig: pd.DataFrame) -> pd.DataFrame:
    """날짜별 buy_signal(필터 전·후) 비율과 약세장 여부."""
    g = sig.groupby("date")
    return pd.DataFrame({"n": g.size(), "buy_raw": g["buy_signal_raw"].mean(), "buy": g["buy_signal"].mean(),
                         "spy_bear": g["spy_bear"].first(), "regime": g["regime"].first()})


def _print_buy_ratio(sig: pd.DataFrame) -> None:
    br = _buy_ratio(sig)
    ok = br[~br["spy_bear"]]
    print(f"날짜 {len(br)} (약세장 {int(br['spy_bear'].sum())}) · 평균 buy_raw {br['buy_raw'].mean():.1%} · "
          f"buy {br['buy'].mean():.1%} · 비약세장인데 buy 0인 날 {int((ok['buy'] == 0).sum())}")


# ── count ──────────────────────────────────────────────────────────────

def load_signals(set_name: str) -> pd.DataFrame:
    sig = pd.read_parquet(RESEARCH_DIR / f"signals_{set_name}.parquet")
    sig["date"] = pd.to_datetime(sig["date"])
    return sig[sig["date"] <= DEV_END].reset_index(drop=True)


def event_flags(sig: pd.DataFrame) -> pd.DataFrame:
    """행별 이벤트 상태 (시작일 처리 전). 열 = 이벤트 ID."""
    f = pd.DataFrame(index=sig.index)
    f["S1_buy_raw"] = sig["buy_signal_raw"].astype(bool)
    f["S2_buy"] = sig["buy_signal"].astype(bool)
    f["S3_mom"] = sig["전략구분"].astype(str).str.contains("모멘텀")
    f["S4_bottom"] = sig["전략구분"].astype(str).str.contains("바닥반등")
    f["S5_judge"] = sig["판단"].isin(TOP_JUDGMENTS)
    f["S6_hard"] = sig["hard_pass"].astype(bool)
    f["S7_top1"] = sig["top1"].astype(bool)
    f["S8_top3"] = sig["top3"].astype(bool)
    for col, tag in (("일봉패턴", "PD"), ("주봉패턴", "PW"), ("월봉패턴", "PM")):
        s = sig[col].fillna("").astype(str).str.split(", ").explode().str.strip()
        s = s[(s.str.len() > 0) & (s != "nan")]
        dummies = pd.crosstab(s.index, s).reindex(sig.index, fill_value=0) > 0
        for name in dummies.columns:
            f[f"{tag}_{name}"] = dummies[name]
    return f


def onsets(sig: pd.DataFrame, flag: pd.Series, date_index: dict, cooldown: int = COOLDOWN) -> pd.Series:
    """전 거래일엔 없고 오늘 생긴 이벤트. 같은 종목은 cooldown 거래일 안에 한 번만."""
    df = pd.DataFrame({"t": sig["티커"], "di": sig["date"].map(date_index), "f": flag.astype(bool)})
    df = df.sort_values(["t", "di"])
    prev_f = df.groupby("t")["f"].shift(1).fillna(False).astype(bool)
    prev_di = df.groupby("t")["di"].shift(1)
    start = df["f"] & ~(prev_f & (df["di"] - prev_di == 1))
    keep = pd.Series(False, index=df.index)
    for _, g in df[start].groupby("t"):
        last = -10**9
        for idx, di in zip(g.index, g["di"]):
            if di - last >= cooldown:
                keep[idx] = True
                last = di
    return keep.reindex(sig.index, fill_value=False)


def all_onsets(sig: pd.DataFrame) -> pd.DataFrame:
    flags = event_flags(sig)
    date_index = {d: i for i, d in enumerate(sorted(sig["date"].unique()))}
    return pd.DataFrame({c: onsets(sig, flags[c], date_index) for c in flags.columns})


def count(set_name: str) -> None:
    """이벤트 수만 센다 → EVENT_COUNTS · EVENT_FAMILY · TRIAL_LOG 등록. 수익률은 보지 않는다."""
    fam_path = TIER3_DIR / f"EVENT_FAMILY_{set_name}.json"
    if fam_path.exists():
        raise SystemExit(f"{fam_path.name} 이미 있음 — 결과를 본 뒤 목록을 바꾸려면 새 family 이름으로 등록한다")
    sig = load_signals(set_name)
    ev = all_onsets(sig)
    rows = []
    for c in ev.columns:
        m = ev[c]
        rows.append({"event_id": c, "n_events": int(m.sum()), "n_days": int(sig.loc[m, "date"].nunique()),
                     "n_tickers": int(sig.loc[m, "티커"].nunique())})
    counts = pd.DataFrame(rows).sort_values("n_events", ascending=False)
    counts["in_family"] = (counts["n_events"] >= MIN_EVENTS) & (counts["n_days"] >= MIN_DAYS)
    counts.to_csv(TIER3_DIR / f"EVENT_COUNTS_{set_name}.csv", index=False)

    fam_events = counts.loc[counts["in_family"], "event_id"].tolist()
    tests = [f"{e}|h{h}" for e in fam_events for h in HORIZONS] + \
            [f"{c}|h{h}" for c in SCORE_COLS for h in HORIZONS]
    n_family = len(tests)
    z = bonferroni_z(n_family)
    family = {"family": f"event_study_{set_name}", "n_family": n_family, "z": round(z, 4),
              "events": fam_events, "scores": list(SCORE_COLS), "horizons": list(HORIZONS),
              "cooldown": COOLDOWN, "min_events": MIN_EVENTS, "min_days": MIN_DAYS, "tests": tests}
    fam_path.write_text(json.dumps(family, ensure_ascii=False, indent=2), encoding="utf-8")
    log_trials("tier3b-1", family["family"], n_family,
               [{"test_id": "(등록)", "판정": "등록",
                 "메모": f"이벤트 {len(fam_events)}개×{len(HORIZONS)}기간 + 점수IC {len(SCORE_COLS)}×{len(HORIZONS)}. "
                         f"z={z:.2f}. 제외(표본 부족) {int((~counts['in_family']).sum())}개"}])
    print(counts.to_string(index=False))
    print(f"\nfamily {n_family}건, z={z:.2f}, 누적 시도 {total_trials()}건 → {fam_path.name}")


# ── analyze ────────────────────────────────────────────────────────────

def event_stats(sig: pd.DataFrame, ev: pd.Series, h: int) -> dict:
    """이벤트 후 h일 초과수익 (같은 날 유니버스 동일가중 평균 대비)."""
    y = f"fwd_ret_{h}d"
    bench = sig.groupby("date")[y].transform("mean")
    ar = (sig[y] - bench)[ev].dropna()
    if ar.empty:
        return {}
    daily = ar.groupby(sig.loc[ar.index, "date"]).mean()  # 같은 날 이벤트는 한 묶음
    yr = yearly_mean(daily)
    first, second = halves_mean(daily)
    return {"n_events": int(ar.size), "n_days": int(daily.size),
            "mean_ar": float(ar.mean()), "mean_net": float(ar.mean() - ROUND_TRIP),
            "hit": float((ar > 0).mean()), "t_nw": nw_tstat(daily, lag=h),
            "years_pos": int((yr > 0).sum()), "n_years": int(yr.size),
            "first_half": first, "second_half": second,
            "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items())}


def score_ic_stats(sig: pd.DataFrame, col: str, h: int) -> dict:
    """점수 열의 날짜별 Spearman IC. 판정용 값은 IC로 바꿔 넣는다 (mean_net = 평균 IC)."""
    y = f"fwd_ret_{h}d"
    ic = ic_by_date(pd.to_numeric(sig[col], errors="coerce"), sig[y], sig["date"])
    if ic.empty:
        return {}
    yr = yearly_mean(ic)
    first, second = halves_mean(ic)
    return {"n_events": int(ic.size), "n_days": int(ic.size), "mean_ar": float(ic.mean()),
            "mean_net": float(ic.mean()), "hit": float((ic > 0).mean()), "t_nw": nw_tstat(ic, lag=h - 1),
            "years_pos": int((yr > 0).sum()), "n_years": int(yr.size),
            "first_half": first, "second_half": second,
            "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items())}


def grade(r: dict, z: float) -> str:
    """1~5단계 공통 판정."""
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


def analyze(set_name: str) -> None:
    fam_path = TIER3_DIR / f"EVENT_FAMILY_{set_name}.json"
    if not fam_path.exists():
        raise SystemExit(f"{fam_path.name} 없음 → 먼저 --count 로 family 를 등록한다")
    fam = json.loads(fam_path.read_text(encoding="utf-8"))
    z = float(fam["z"])
    sig = load_signals(set_name)
    ev = all_onsets(sig)

    rows = []
    for e in fam["events"]:
        for h in fam["horizons"]:
            r = event_stats(sig, ev[e], h)
            rows.append({"test_id": f"{e}|h{h}", "kind": "event", "event_id": e, "h": h, **r, "판정": grade(r, z)})
    for c, col in SCORE_COLS.items():
        for h in fam["horizons"]:
            r = score_ic_stats(sig, col, h)
            rows.append({"test_id": f"{c}|h{h}", "kind": "score_ic", "event_id": c, "h": h, **r, "판정": grade(r, z)})
    res = pd.DataFrame(rows)
    res.to_csv(TIER3_DIR / f"event_study_{set_name}.csv", index=False)

    # 0-6 시뮬레이터 입력: family 이벤트 시작일 (S7·S8 은 ccs, 나머지 1)
    ev_rows = []
    for e in fam["events"]:
        m = ev[e]
        score = sig.loc[m, "ccs"].astype(float) if e in ("S7_top1", "S8_top3") else 1.0
        ev_rows.append(pd.DataFrame({"date": sig.loc[m, "date"], "티커": sig.loc[m, "티커"],
                                     "event_id": e, "score": score}))
    pd.concat(ev_rows, ignore_index=True).to_parquet(RESEARCH_DIR / f"events_{set_name}.parquet")

    log_trials("tier3b-1", fam["family"], fam["n_family"],
               [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}" if pd.notna(r.get("t_nw")) else "", "판정": r["판정"]}
                for _, r in res.iterrows()])
    _write_report(set_name, sig, res, fam)


def _write_report(set_name: str, sig: pd.DataFrame, res: pd.DataFrame, fam: dict) -> None:
    br = _buy_ratio(sig)
    counts = res["판정"].value_counts()
    lines = [f"# 1단계 스크리너 출력 이벤트 스터디 ({set_name})", "",
             f"- 기간: {sig['date'].min().date()} ~ {sig['date'].max().date()} ({sig['date'].nunique()}일), "
             f"유니버스: 그날 S&P 500 구성종목 중 유동성 통과 (평균 {sig.groupby('date').size().mean():.0f}종목/일)",
             f"- 초과수익 = 이벤트 종목 h일 수익률 − 같은 날 유니버스 동일가중 평균. 진입 t+1 시가 → 청산 t+1+h 시가",
             f"- 비용 후 = 왕복 {ROUND_TRIP:.1%} 차감. t_nw = 날짜별 평균 초과수익의 Newey-West t (lag=h)",
             f"- 시작일 기준, 같은 종목·이벤트는 {fam['cooldown']}거래일 쿨다운. "
             f"family = 이벤트 ≥{fam['min_events']}건 & ≥{fam['min_days']}일",
             f"- **family {fam['n_family']}건 → Bonferroni 임계 z = {fam['z']:.2f}** · 누적 시도 {total_trials()}건",
             f"- 판정: " + " · ".join(f"{k} {v}" for k, v in counts.items()), "",
             "## 날짜별 buy_signal 비율", "",
             f"- 평균 buy_signal_raw(필터 전) {br['buy_raw'].mean():.1%} · buy_signal(시장·섹터 필터 후) {br['buy'].mean():.1%}",
             f"- 약세장(SPY<EMA200) {int(br['spy_bear'].sum())}일 · 비약세장인데 buy_signal 0인 날 "
             f"{int((br.loc[~br['spy_bear'], 'buy'] == 0).sum())}일",
             f"- 레짐(debug): " + " · ".join(f"{k} {v}" for k, v in br["regime"].value_counts().items()), ""]
    order = {"후보": 0, "역신호": 1, "관찰": 2, "탈락": 3, "표본 없음": 4}
    for kind, title, unit in (("event", "이벤트 (초과수익)", "%"), ("score_ic", "점수 IC (C1~C3)", "ic")):
        sub = res[res["kind"] == kind].copy()
        sub["_o"] = sub["판정"].map(order)
        sub = sub.sort_values(["_o", "t_nw"], ascending=[True, False])
        lines += [f"## {title}", "",
                  "| 판정 | test | n | 일수 | 평균(비용 전) | 비용 후 | 적중률 | t_nw | 연도+ | 앞/뒤 반 | 연도별 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            if pd.isna(r.get("t_nw")):
                lines.append(f"| {r['판정']} | {r.test_id} | | | | | | | | | |")
                continue
            fmt = (lambda v: f"{v:+.2%}") if unit == "%" else (lambda v: f"{v:+.4f}")
            lines.append(f"| {r['판정']} | {r.test_id} | {int(r.n_events)} | {int(r.n_days)} | {fmt(r.mean_ar)} | "
                         f"{fmt(r.mean_net)} | {r.hit:.0%} | {r.t_nw:+.2f} | {int(r.years_pos)}/{int(r.n_years)} | "
                         f"{fmt(r.first_half)} / {fmt(r.second_half)} | {r.yearly} |")
        lines.append("")
    out = TIER3_DIR / f"EVENT_STUDY_{set_name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:14]))
    print(res.groupby("판정").size().to_string())
    print(f"저장: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pit")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="--build 시험용: 앞 N일만")
    args = ap.parse_args()
    if args.build:
        build(args.set, args.limit)
    elif args.count:
        count(args.set)
    else:
        analyze(args.set)


if __name__ == "__main__":
    main()
