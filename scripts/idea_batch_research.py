#!/usr/bin/env python3
"""새 아이디어 묶음 (idea_batch_v1): 아직 시험 안 한 아이디어 7개 → SPY·PIT 동일가중·PT-1 기준선과 비교.

사전 등록 (TRIAL_LOG family idea_batch_v1, 7건 — 결과 계산 전에 --register):
  A1-21 / A1-63  내부자 장내 매수: Form 4 원본, 비파생 P(취득 A), 보고자 임원 또는 이사 (10% 대주주이기만 하면 제외).
                 신호일 = 제출일 → 다음 거래일 시가 진입, 21 / 63거래일 보유. 금액 하한 없음.
  A2             실적 발표 직전 보유: 예상 반응일 E = 약 1년 전 반응일 + 364일 이후 첫 거래일.
                 E−5거래일 시가 진입 → E+2거래일 시가 청산 (7거래일). E 30일 전 ~ 신호일(E−6) 사이에 이미 발표가 있으면 제외.
  B1             섹터 ETF 모멘텀: SPDR 11개, 월말 점수 = 63·126·252일 수익률 평균 (252일 이력 있는 것만),
                 상위 3개 동일가중, 다음 거래일 시가 교체. 첫 교체 전에는 SPY.
  B2             SPY 변동성 조절: 월말 σ = 최근 21일 일간 수익률 표준편차 × √252 → 다음 달 SPY 비중 min(1, 15%/σ), 나머지 현금(수익 0).
  C1             RSI(2) 눌림목 (Connors): 종가 > SMA200 · RSI(2) < 10 → 다음날 시가 진입, 5거래일 보유.
  C2             52주 신고가 + 거래량: 종가 > 직전 252일 최고 종가 · 거래량 ≥ 직전 50일 평균 × 2 → 다음날 시가, 20거래일 보유.
공통: 2016-01-04 ~ 2025-09-30, 그날 S&P 500 구성종목(PIT) ∩ 가격 있는 종목, t 종가 신호 → t+1 시가, 시가→시가, 편도 0.1%.
종목 이벤트(A1·A2·C1·C2) 포트폴리오 = 달력시간 동일가중: 보유 구간 안 종목을 같은 비중(종목당 ≤ 5%), 남는 비중은 SPY.
  → SPY 대비 차이 = 신호로 기울인 효과.
판정 (Bonferroni z = bonferroni_z(7) ≈ 2.69):
  이벤트 통계 = 진입일별 평균 초과수익(같은 구간 PIT 동일가중 대비) → NW t (lag=h) → grade (1~5단계 공통 판정)
  채택 후보 = SPY 대비 verdict '채택 후보' · p_개선아님 ≤ 0.05/7 · (이벤트형) 이벤트 판정 '후보'
  관찰     = 그 밖에 SPY 대비 ΔSharpe 95% 하한 > 0 이거나 이벤트 판정 '후보'/'관찰'
  참고     = PT-1 기준선·PIT 동일가중 대비 ΔSharpe, Deflated Sharpe (판정에 안 씀)
한계: 상장폐지 종목은 가격이 없어 빠진다(생존 편향). 이벤트 통계는 같은 종목의 h일 안 재이벤트를 한 번만 센다.
     보유 중 가격이 끊긴 종목은 그 비중을 수익 0(현금)으로 둔다 (signal_portfolio_sim 과 같음).

사용법:
  PYTHONPATH=.:src python scripts/idea_batch_research.py --register
  PYTHONPATH=.:src python scripts/idea_batch_research.py --run
입력: data/cache/ohlcv_px10y.parquet · data/research/{etf_ohlcv, sec_earnings_dates, insider_trades,
      pt1_base_dev_equity}.parquet (fetch_insider_data.py · pt1_baseline_equity.py 산출)
산출: tier3/IDEA_BATCH_REPORT.md, tier3/idea_batch.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from baseline_compare import timing_returns
from paper_trading.evaluation import (alpha_beta, cagr, deflated_sharpe, max_drawdown, paired_bootstrap, sharpe,
                                      verdict)
from paper_trading.universe import Membership, load_membership
from pead_research import grade  # signal_event_study.grade 와 같은 1~5단계 공통 판정 (스크리너 모듈을 안 불러옴)
from research_utils import (COST_PER_SIDE, RESEARCH_DIR, ROUND_TRIP, TIER3_DIR, TRIAL_LOG, bonferroni_z, halves_mean,
                            load_ohlcv, log_trials, nw_tstat, total_trials, yearly_mean)
from signal_portfolio_sim import equal_weight_pit, open_to_open

FAMILY = "idea_batch_v1"
STEP = "stage10"
START = pd.Timestamp("2016-01-04")
OHLCV_PX10Y = ROOT / "data" / "cache" / "ohlcv_px10y.parquet"
ETF_OHLCV = RESEARCH_DIR / "etf_ohlcv.parquet"
EARNINGS = RESEARCH_DIR / "sec_earnings_dates.parquet"
INSIDER = RESEARCH_DIR / "insider_trades.parquet"
PT1_EQUITY = RESEARCH_DIR / "pt1_base_dev_equity.parquet"
MEMBERSHIP_CSV = ROOT / "data" / "universe" / "sp500_membership.csv"
SECTOR_ETFS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
MAX_WEIGHT = 0.05
TARGET_VOL = 0.15

# test_id → (이름, 종류, 보유 거래일)
TESTS: dict[str, tuple[str, str, int | None]] = {
    "A1-21": ("내부자 장내 매수 (21일)", "event", 21),
    "A1-63": ("내부자 장내 매수 (63일)", "event", 63),
    "A2": ("실적 발표 직전 보유", "event", 7),
    "B1": ("섹터 ETF 모멘텀 상위 3", "etf", None),
    "B2": ("SPY 변동성 조절", "etf", None),
    "C1": ("RSI(2) 눌림목", "event", 5),
    "C2": ("52주 신고가 + 거래량 2배", "event", 20),
}
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
SPEC = ("[2016-01-04~2025-09-30, PIT S&P 500, t 종가 신호→t+1 시가, 시가→시가, 편도 0.1%] "
        "A1-21/A1-63 내부자 장내 매수(Form 4 원본·P 취득·임원/이사, 제출일 다음 거래일 진입, 21/63일, 금액 하한 없음) · "
        "A2 실적 직전(E=약 1년 전 반응일+364일 이후 첫 거래일, E−5 시가→E+2 시가, E−30일~신호일 발표 있으면 제외) · "
        "B1 섹터 ETF 11개 (63·126·252일 평균) 상위 3 월간 · B2 SPY 비중 min(1, 15%/σ21) 월간·현금 0 · "
        "C1 종가>SMA200·RSI(2)<10, 5일 · C2 종가>직전 252일 최고·거래량≥50일 평균×2, 20일. "
        "이벤트형은 달력시간 동일가중(종목당 ≤5%, 나머지 SPY). "
        f"채택 = verdict 채택 후보 · p_개선아님 ≤ 0.05/7 · 이벤트 판정 후보 (z={bonferroni_z(N_TESTS):.2f})")


# ── 데이터 ─────────────────────────────────────────────────────────────────────────────────


def member_mask(index: pd.DatetimeIndex, columns: pd.Index, mem: Membership) -> pd.DataFrame:
    """그날 S&P 500 구성종목(PIT) ∩ 가격 열 (build_price_panel 과 같은 방식)."""
    col_pos = {t: j for j, t in enumerate(columns)}
    rows: dict[int, np.ndarray] = {}
    out = np.zeros((len(index), len(columns)), dtype=bool)
    for i, d in enumerate(index):
        s = mem.on(f"{d:%Y-%m-%d}")
        row = rows.get(id(s))
        if row is None:
            row = np.zeros(len(columns), dtype=bool)
            row[[col_pos[t] for t in s if t in col_pos]] = True
            rows[id(s)] = row
        out[i] = row
    return pd.DataFrame(out, index=index, columns=columns)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """각 달의 마지막 거래일."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


# ── 신호 ───────────────────────────────────────────────────────────────────────────────────


def rsi_wilder(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder RSI (ta 라이브러리와 같은 지수평활 alpha=1/n, adjust=False, min_periods=n). 하락이 없으면 100."""
    diff = close.diff()
    valid = close.notna() & close.shift(1).notna()
    up = diff.clip(lower=0.0).where(valid)
    down = (-diff).clip(lower=0.0).where(valid)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rsi = 100.0 - 100.0 / (1.0 + au / ad)
    return rsi.where(ad > 0, 100.0).where(au.notna() & ad.notna())


def rsi2_pullback_signal(close: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """C1: 종가 > SMA200 그리고 RSI(2) < 10 (그날 구성종목)."""
    sma200 = close.rolling(200, min_periods=200).mean()
    return (close > sma200) & (rsi_wilder(close, 2) < 10) & member


def breakout_volume_signal(close: pd.DataFrame, volume: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """C2: 종가 > 직전 252일 최고 종가 그리고 거래량 ≥ 직전 50일 평균 × 2 (그날 구성종목)."""
    prior_high = close.shift(1).rolling(252, min_periods=252).max()
    vol = volume.replace(0, np.nan)
    vol_avg = vol.shift(1).rolling(50, min_periods=50).mean()
    return (close > prior_high) & (vol >= 2.0 * vol_avg) & member


def signal_entries(signal: pd.DataFrame) -> pd.DataFrame:
    """bool 신호(날짜 × 티커) → 진입 목록 (signal_date, 티커, entry_pos = 다음 거래일 위치)."""
    st = signal.stack(future_stack=True)
    st = st[st.astype(bool)]
    ev = st.index.to_frame(index=False)
    ev.columns = ["signal_date", "티커"]
    ev["entry_pos"] = signal.index.searchsorted(ev["signal_date"], side="right")
    return ev[ev["entry_pos"] < len(signal.index)].reset_index(drop=True)


def insider_entries(trades: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index,
                    mem: Membership) -> pd.DataFrame:
    """A1: 임원·이사 장내 매수 신고 (제출일·종목 단위) → 제출일 다음 거래일 진입. 제출일 구성종목만."""
    od = trades[trades["is_officer"] | trades["is_director"]]
    ev = od.drop_duplicates(["filing_date", "티커"])[["filing_date", "티커"]].rename(columns={"filing_date": "signal_date"})
    ev = ev[ev["티커"].isin(columns)]
    ev = ev[[t in mem.on(f"{d:%Y-%m-%d}") for d, t in zip(ev["signal_date"], ev["티커"])]]
    ev["entry_pos"] = index.searchsorted(ev["signal_date"], side="right")
    ev = ev[ev["entry_pos"] < len(index)]
    return ev.sort_values(["signal_date", "티커"]).reset_index(drop=True)


def earnings_premium_entries(dates: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index,
                             mem: Membership) -> tuple[pd.DataFrame, float]:
    """A2: 예상 반응일 E → E−5 거래일 진입 (7거래일 보유). (진입 목록, 예상일 적중률) 반환.

    E = 같은 종목 과거 반응일 p + 364일 이후 첫 거래일 (p 는 E 보다 약 1년 앞이라 신호일에 이미 알려져 있다).
    E 30일 전 ~ 신호일(E−6) 사이에 실제 반응일이 있으면 이번 분기 발표가 앞당겨진 것 → 제외.
    적중률 = 실제 반응일이 보유 구간 [E−5, E+1] 거래일 안에 든 비율 (진단용).
    """
    rows = []
    hits = 0
    for t, g in dates[dates["티커"].isin(columns)].groupby("티커"):
        actual = np.sort(pd.to_datetime(g["event_date"]).to_numpy())
        for p in actual:
            i_e = int(index.searchsorted(p + np.timedelta64(364, "D"), side="left"))
            if i_e - 6 < 0 or i_e >= len(index):
                continue
            e_day, sig_day = index[i_e], index[i_e - 6]
            lo = np.searchsorted(actual, (e_day - pd.Timedelta(days=30)).to_datetime64(), side="left")
            hi = np.searchsorted(actual, sig_day.to_datetime64(), side="right")
            if hi > lo or t not in mem.on(f"{sig_day:%Y-%m-%d}"):
                continue
            win_lo, win_hi = index[i_e - 5], index[min(i_e + 1, len(index) - 1)]
            a = np.searchsorted(actual, win_lo.to_datetime64(), side="left")
            b = np.searchsorted(actual, win_hi.to_datetime64(), side="right")
            hits += int(b > a)
            rows.append({"signal_date": sig_day, "티커": t, "entry_pos": i_e - 5, "E": e_day})
    ev = pd.DataFrame(rows, columns=["signal_date", "티커", "entry_pos", "E"])
    ev = ev.drop_duplicates(["entry_pos", "티커"]).sort_values(["signal_date", "티커"]).reset_index(drop=True)
    return ev, (hits / len(rows) if rows else float("nan"))


# ── 포트폴리오 ─────────────────────────────────────────────────────────────────────────────


def active_from_entries(entries: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index, hold: int) -> pd.DataFrame:
    """진입 목록 → 보유 표시 [entry, entry + hold). 같은 종목의 재이벤트는 보유 구간을 연장한다."""
    act = np.zeros((len(index), len(columns)), dtype=bool)
    col_pos = {t: j for j, t in enumerate(columns)}
    for p, t in zip(entries["entry_pos"], entries["티커"]):
        j = col_pos.get(t)
        if j is not None and 0 <= p < len(index):
            act[p : p + hold, j] = True
    return pd.DataFrame(act, index=index, columns=columns)


def calendar_time_returns(active: pd.DataFrame, r_oo: pd.DataFrame, tradable: pd.DataFrame, spy_oo: pd.Series,
                          max_weight: float = MAX_WEIGHT, cost: float = COST_PER_SIDE
                          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """달력시간 동일가중: 보유 구간 안 종목(그날 시가 있음)을 같은 비중(≤ max_weight), 남는 비중은 SPY.

    t행 비중은 t 시가 → t+1 시가 수익에 적용. 비용 = 비중 변화(종목·SPY 모두) × 편도 비용.
    Returns: (일간 수익률, 보유 종목 수, 종목 비중 합).
    """
    act = active & tradable
    n = act.sum(axis=1)
    each = (1.0 / n.where(n > 0)).clip(upper=max_weight).fillna(0.0)
    w = act.astype(float).mul(each, axis=0)
    w_spy = 1.0 - w.sum(axis=1)
    gross = (w * r_oo.fillna(0.0)).sum(axis=1) + w_spy * spy_oo.reindex(w.index).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1) + w_spy.diff().abs()
    turnover.iloc[0] = w.iloc[0].sum() + w_spy.iloc[0]
    return gross - turnover * cost, n, w.sum(axis=1)


def sector_momentum_returns(etf: pd.DataFrame, spy_oo: pd.Series, cost: float = COST_PER_SIDE
                            ) -> tuple[pd.Series, pd.DataFrame]:
    """B1: 월말 점수(63·126·252일 수익률 평균) 상위 3개 섹터 ETF, 다음 거래일 시가 교체. 첫 교체 전에는 SPY."""
    names = [t for t in SECTOR_ETFS if t in etf.columns.get_level_values(0)]
    close = pd.DataFrame({t: etf[t]["Close"] for t in names})
    opens = pd.DataFrame({t: etf[t]["Open"] for t in names})
    r_oo = opens.shift(-1) / opens - 1.0
    score = sum(close / close.shift(n) - 1.0 for n in (63, 126, 252)) / 3.0
    idx = close.index
    w = pd.DataFrame(0.0, index=idx, columns=names)
    w_spy = pd.Series(1.0, index=idx)
    for m in month_ends(idx):
        s = score.loc[m].dropna()
        i = idx.get_loc(m) + 1
        if len(s) < 3 or i >= len(idx):
            continue
        w.iloc[i:] = 0.0
        w.iloc[i:, [names.index(t) for t in s.nlargest(3).index]] = 1.0 / 3.0
        w_spy.iloc[i:] = 0.0
    gross = (w * r_oo.fillna(0.0)).sum(axis=1) + w_spy * spy_oo.reindex(idx).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1) + w_spy.diff().abs()
    turnover.iloc[0] = w.iloc[0].sum() + w_spy.iloc[0]
    ret = (gross - turnover * cost).iloc[:-1]  # 마지막 날은 다음 시가가 없다
    return ret, w


def vol_managed_weights(close: pd.Series, target: float = TARGET_VOL) -> pd.Series:
    """B2: 월말 σ(21일, 연율) → 다음 달 SPY 목표 비중 min(1, target/σ). 월말에 정해 다음 월말까지 유지 (t 종가 기준)."""
    sigma = close.pct_change().rolling(21).std() * np.sqrt(252)
    me = month_ends(close.index)
    w = pd.Series(np.nan, index=close.index)
    w.loc[me] = np.minimum(1.0, target / sigma.loc[me])
    return w.ffill().fillna(1.0)


# ── 통계 ───────────────────────────────────────────────────────────────────────────────────


def episodes(entries: pd.DataFrame, hold: int) -> pd.DataFrame:
    """같은 종목의 보유 구간이 겹치는 재이벤트는 첫 건만 (이벤트 통계용)."""
    keep = []
    for _, g in entries.sort_values("entry_pos").groupby("티커"):
        last = None
        for i, p in zip(g.index, g["entry_pos"]):
            if last is None or p >= last + hold:
                keep.append(i)
                last = p
    return entries.loc[sorted(keep)]


def event_stats(entries: pd.DataFrame, opens: pd.DataFrame, member: pd.DataFrame, hold: int) -> dict:
    """진입 시가 → +hold 시가 수익 − 같은 구간 그날 구성종목 동일가중 → 진입일별 평균 → NW t (pead_research.stats 와 같음)."""
    fwd = opens.shift(-hold) / opens - 1.0
    ar = fwd.sub(fwd.where(member).mean(axis=1), axis=0)
    ep = episodes(entries, hold)
    ep = ep[ep["entry_pos"] < len(opens.index)]
    dates = opens.index[ep["entry_pos"].to_numpy()]
    keep = np.asarray(dates >= START)
    ep, dates = ep[keep], dates[keep]
    col_pos = {t: j for j, t in enumerate(opens.columns)}
    vals = ar.to_numpy()[ep["entry_pos"].to_numpy(), [col_pos[t] for t in ep["티커"]]] if len(ep) else np.array([])
    df = pd.DataFrame({"date": dates, "ar": vals}).dropna()
    if df.empty:
        return {}
    daily = df.groupby("date")["ar"].mean()
    yr = yearly_mean(daily)
    first, second = halves_mean(daily)
    return {"n_events": int(len(df)), "n_days": int(daily.size), "mean_ar": float(df["ar"].mean()),
            "mean_net": float(df["ar"].mean() - ROUND_TRIP), "median_ar": float(df["ar"].median()),
            "hit": float((df["ar"] > 0).mean()), "t_nw": nw_tstat(daily, lag=hold),
            "years_pos": int((yr > 0).sum()), "n_years": int(yr.size), "first_half": first, "second_half": second,
            "yearly": ";".join(f"{k}:{v:+.4f}" for k, v in yr.items())}


def compare(s: pd.Series, bench: pd.Series) -> dict:
    """같은 날짜로 맞춘 두 일간 수익률의 paired bootstrap (s = 전략, bench = 기준)."""
    df = pd.concat({"s": s, "b": bench}, axis=1).dropna()
    return paired_bootstrap(df["s"].tolist(), df["b"].tolist())


def final_label(kind: str, v_spy: str, st_spy: dict, ev_grade: str | None) -> str:
    """사전 등록 판정: 채택 후보 / 역신호 / 관찰 / 탈락."""
    bonf_ok = bool(st_spy) and st_spy["p_개선아님"] <= P_BONF
    if v_spy == "채택 후보" and bonf_ok and (kind != "event" or ev_grade == "후보"):
        return "채택 후보"
    if kind == "event" and ev_grade == "역신호":
        return "역신호"
    if (st_spy and st_spy["ΔSharpe_하한"] > 0) or (kind == "event" and ev_grade in ("후보", "관찰")):
        return "관찰"
    return "탈락"


# ── 실행 ───────────────────────────────────────────────────────────────────────────────────


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def register() -> None:
    log = _log()
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": SPEC}])
    print(f"등록: {FAMILY} {N_TESTS}건 · z={bonferroni_z(N_TESTS):.2f} · 누적 시도 {total_trials()}건")


def run() -> None:
    log = _log()
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    already_logged = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    z = bonferroni_z(N_TESTS)

    print("===== 데이터 읽기", flush=True)
    ohlcv = load_ohlcv(OHLCV_PX10Y)
    close = ohlcv.xs("Close", axis=1, level="Price")
    opens = ohlcv.xs("Open", axis=1, level="Price")
    volume = ohlcv.xs("Volume", axis=1, level="Price")
    idx, cols = close.index, close.columns
    r_oo = open_to_open(ohlcv)
    etf = load_ohlcv(ETF_OHLCV)
    spy_oo = (etf["SPY"]["Open"].shift(-1) / etf["SPY"]["Open"] - 1.0).reindex(idx)
    mem = Membership(load_membership(MEMBERSHIP_CSV))
    member = member_mask(idx, cols, mem)
    tradable = opens.notna()
    ew = equal_weight_pit(r_oo, mem)

    print("===== 이벤트 만들기", flush=True)
    trades = pd.read_parquet(INSIDER)
    ins = insider_entries(trades, idx, cols, mem)
    earn, hit_rate = earnings_premium_entries(pd.read_parquet(EARNINGS), idx, cols, mem)
    c1 = signal_entries(rsi2_pullback_signal(close, member))
    c2 = signal_entries(breakout_volume_signal(close, volume, member))
    entries = {"A1-21": ins, "A1-63": ins, "A2": earn, "C1": c1, "C2": c2}
    for k, e in entries.items():
        print(f"[idea] {k}: 진입 {len(e):,}건", flush=True)

    print("===== 포트폴리오", flush=True)
    series: dict[str, pd.Series] = {}
    diag: dict[str, str] = {}
    ev_res: dict[str, dict] = {}
    for i, (tid, (name, kind, hold)) in enumerate(TESTS.items(), 1):
        if kind == "event":
            act = active_from_entries(entries[tid], idx, cols, hold)
            ret, n_names, w_sum = calendar_time_returns(act, r_oo, tradable, spy_oo)
            series[tid] = ret.iloc[:-1]
            sl = slice(START, None)
            diag[tid] = (f"진입 {len(entries[tid]):,}건 · 평균 보유 종목 {n_names.loc[sl].mean():.1f}개 · "
                         f"이벤트 종목 비중 평균 {w_sum.loc[sl].mean():.0%} (나머지 SPY)")
            ev_res[tid] = event_stats(entries[tid], opens, member, hold)
        elif tid == "B1":
            ret, w = sector_momentum_returns(etf, spy_oo)
            series[tid] = ret
            held = (w.loc[START:] > 0).mean().sort_values(ascending=False)
            diag[tid] = "보유 비율 " + " · ".join(f"{t} {v:.0%}" for t, v in held.items() if v > 0)
        else:
            spy = etf["SPY"]
            tw = vol_managed_weights(spy["Close"])
            series[tid] = timing_returns(spy["Open"], spy["Close"], tw)
            twl = tw.loc[START:]
            diag[tid] = f"평균 SPY 비중 {twl.mean():.0%} · 비중 < 100% 인 날 {(twl < 1).mean():.0%} · 최저 {twl.min():.0%}"
        print(f"[idea] {tid} 완료 [{i / N_TESTS:.1%}]", flush=True)

    print("===== 비교·판정", flush=True)
    spy_s = spy_oo.loc[START:].iloc[:-1]
    ew_s = ew.loc[START:].iloc[:-1]
    pt1 = None
    if PT1_EQUITY.exists():
        eq = pd.read_parquet(PT1_EQUITY).iloc[:, 0]
        pt1 = eq.pct_change().dropna().loc[START:]
    srs = {k: v.loc[START:].dropna() for k, v in series.items()}
    daily_sr = [sharpe(v.tolist()) / np.sqrt(252) for v in srs.values()]

    rows = []
    for tid, (name, kind, hold) in TESTS.items():
        s = srs[tid]
        df = pd.concat({"s": s, "SPY": spy_s}, axis=1).dropna()
        a, b = df["s"].tolist(), df["SPY"].tolist()
        st_spy = paired_bootstrap(a, b)
        v_spy = verdict(st_spy, max_drawdown(a), max_drawdown(b))
        st_ew = compare(s, ew_s)
        st_pt1 = compare(s, pt1) if pt1 is not None else {}
        ab = alpha_beta(a, {"SPY": b})
        r = ev_res.get(tid, {})
        ev_grade = grade(r, z) if kind == "event" else None
        x = pd.Series(a)
        dsr = deflated_sharpe(sharpe(a) / np.sqrt(252), daily_sr, len(a), float(x.skew()), float(x.kurt()) + 3.0)
        rows.append({
            "test_id": tid, "이름": name, "종류": kind, "보유일": hold, "시작": df.index.min().date(),
            "끝": df.index.max().date(), "일수": len(df),
            "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a),
            "SPY_Sharpe": sharpe(b), "SPY_CAGR": cagr(b), "SPY_MDD": max_drawdown(b),
            "알파_연": ab.get("알파_연"), "알파_t": ab.get("알파_t"), "베타": ab.get("베타_SPY"),
            "ΔSharpe_SPY": st_spy.get("ΔSharpe"), "하한_SPY": st_spy.get("ΔSharpe_하한"),
            "상한_SPY": st_spy.get("ΔSharpe_상한"), "p_SPY": st_spy.get("p_개선아님"),
            "구간_SPY": f"{st_spy.get('구간_개선')}/{st_spy.get('구간_수')}", "verdict_SPY": v_spy,
            "ΔSharpe_EW": st_ew.get("ΔSharpe"), "하한_EW": st_ew.get("ΔSharpe_하한"),
            "ΔSharpe_PT1": st_pt1.get("ΔSharpe"), "하한_PT1": st_pt1.get("ΔSharpe_하한"),
            "상한_PT1": st_pt1.get("ΔSharpe_상한"), "DSR": dsr,
            "이벤트_판정": ev_grade, **{f"ev_{k}": v for k, v in r.items()},
            "판정": final_label(kind, v_spy, st_spy, ev_grade), "진단": diag.get(tid, ""),
        })
    res = pd.DataFrame(rows)
    res.to_csv(TIER3_DIR / "idea_batch.csv", index=False)

    if already_logged:
        print("결과 행이 이미 TRIAL_LOG 에 있다 — 다시 쓰지 않는다")
    else:
        log_trials(STEP, FAMILY, N_TESTS, [{
            "test_id": r["test_id"], "t": f"{r['ev_t_nw']:.2f}" if pd.notna(r.get("ev_t_nw")) else "",
            "판정": r["판정"],
            "메모": (f"Sharpe {r['Sharpe']:.2f} vs SPY {r['SPY_Sharpe']:.2f}, ΔSharpe {r['ΔSharpe_SPY']:+.2f} "
                     f"[{r['하한_SPY']:+.2f}, {r['상한_SPY']:+.2f}] p {r['p_SPY']:.3f}, CAGR {r['CAGR']:+.1%}, "
                     f"MDD {r['MDD']:.1%}")} for _, r in res.iterrows()])

    write_report(res, srs, spy_s, ew_s, pt1, hit_rate, z)


def _fmt(v: float | None, f: str) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)


def write_report(res: pd.DataFrame, srs: dict[str, pd.Series], spy: pd.Series, ew: pd.Series,
                 pt1: pd.Series | None, hit_rate: float, z: float) -> None:
    """한국어 리포트 → tier3/IDEA_BATCH_REPORT.md."""
    counts = res["판정"].value_counts()
    lines = ["# 새 아이디어 묶음 (idea_batch_v1) — 2016-01 ~ 2025-09", "",
             f"- 사전 등록: TRIAL_LOG family `{FAMILY}`, {N_TESTS}건 → Bonferroni z = {z:.2f} · 누적 시도 {total_trials()}건",
             "- 공통: 그날 S&P 500 구성종목(PIT) ∩ 가격 있는 종목, t 종가 신호 → t+1 시가, 시가→시가, 편도 0.1%",
             "- 이벤트형(A1·A2·C1·C2)은 달력시간 동일가중: 보유 구간 안 종목 같은 비중(종목당 ≤ 5%), 남는 비중 SPY",
             f"- 채택 후보 = SPY 대비 ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내 · p ≤ {P_BONF:.3f} · "
             "(이벤트형) 이벤트 판정 '후보'",
             "- **결과: " + " · ".join(f"{k} {v}건" for k, v in counts.items()) + "**", "",
             "## 한눈에 보기", "",
             "| ID | 아이디어 | Sharpe | CAGR | MDD | SPY 대비 ΔSharpe [95%] | p | PT-1 대비 ΔSharpe [95%] | 이벤트 t | 판정 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    bench = [("SPY 보유", spy), ("PIT 동일가중", ew)] + ([("PT-1 기준선", pt1)] if pt1 is not None else [])
    for label, s in bench:
        r = s.dropna().tolist()
        lines.append(f"| 기준 | {label} | {sharpe(r):.2f} | {cagr(r):+.1%} | {max_drawdown(r):.1%} | | | | | |")
    for _, r in res.iterrows():
        pt1_txt = (f"{r['ΔSharpe_PT1']:+.2f} [{r['하한_PT1']:+.2f}, {r['상한_PT1']:+.2f}]"
                   if pd.notna(r.get("ΔSharpe_PT1")) else "—")
        lines.append(f"| {r['test_id']} | {r['이름']} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | "
                     f"{r['ΔSharpe_SPY']:+.2f} [{r['하한_SPY']:+.2f}, {r['상한_SPY']:+.2f}] | {r['p_SPY']:.3f} | "
                     f"{pt1_txt} | {_fmt(r.get('ev_t_nw'), '+.2f')} | **{r['판정']}** |")
    n_head = len(lines)
    lines += ["", "- 알파·베타 (SPY 회귀, 연):"]
    lines += [f"  - {r['test_id']}: 알파 {_fmt(r['알파_연'], '+.1%')} (t {_fmt(r['알파_t'], '+.2f')}) · "
              f"베타 {_fmt(r['베타'], '.2f')} · 구간 개선 {r['구간_SPY']} · verdict {r['verdict_SPY']} · "
              f"동일가중 대비 ΔSharpe {_fmt(r['ΔSharpe_EW'], '+.2f')} · DSR {_fmt(r['DSR'], '.2f')}"
              for _, r in res.iterrows()]

    ev = res[res["종류"] == "event"]
    lines += ["", "## 이벤트 통계 (같은 구간 PIT 동일가중 대비 초과수익)", "",
              "| test | 이벤트 | 일수 | 평균(비용 전) | 비용 후(왕복 0.2%) | 적중률 | t_nw | 연도+ | 앞/뒤 반 | 이벤트 판정 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ev.iterrows():
        if pd.isna(r.get("ev_n_events")):
            lines.append(f"| {r['test_id']} | 표본 없음 | | | | | | | | |")
            continue
        lines.append(f"| {r['test_id']} | {int(r['ev_n_events']):,} | {int(r['ev_n_days']):,} | {r['ev_mean_ar']:+.2%} | "
                     f"{r['ev_mean_net']:+.2%} | {r['ev_hit']:.0%} | {r['ev_t_nw']:+.2f} | "
                     f"{int(r['ev_years_pos'])}/{int(r['ev_n_years'])} | "
                     f"{r['ev_first_half']:+.2%} / {r['ev_second_half']:+.2%} | {r['이벤트_판정']} |")
    lines += ["", "| test | 연도별 평균 초과수익 |", "|---|---|"]
    lines += [f"| {r['test_id']} | {r.get('ev_yearly', '')} |" for _, r in ev.iterrows()]

    yr_cols = {"SPY": spy, "동일가중": ew, **({"PT-1": pt1} if pt1 is not None else {}), **srs}
    yr = pd.DataFrame({k: (1 + v.dropna()).groupby(v.dropna().index.year).prod() - 1 for k, v in yr_cols.items()})
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(_fmt(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]

    lines += ["", "## 진단", ""]
    lines += [f"- {r['test_id']}: {r['진단']}" for _, r in res.iterrows()]
    lines.append(f"- A2 예상일 적중률 (실제 반응일이 보유 구간 [E−5, E+1] 안): {hit_rate:.0%}")
    lines += ["", "## 한계", "",
              "- 상장폐지·인수된 종목은 yfinance 가격이 없어 빠진다 (2016년 구성종목 가격 커버리지 약 78%).",
              "- B2 의 현금 수익은 0 (보수적 — 실제론 단기채 이자).",
              "- PT-1 기준선은 백테스트 엔진(종가 신호 · 다음날 시가 체결 · 평가 시점 차이)이라 참고용.",
              "- 이벤트 통계는 같은 종목의 보유 구간 안 재이벤트를 한 번만 센다. 포트폴리오는 보유 구간을 연장한다.",
              "- 2016~2025 는 이미 많이 쓴 구간이다. 채택 후보도 실전 그림자 계좌로 3~6개월 확인한 뒤에만 쓴다."]
    out = TIER3_DIR / "IDEA_BATCH_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:n_head]))
    print(f"\n저장: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true", help="결과 계산 전 TRIAL_LOG 사전 등록")
    ap.add_argument("--run", action="store_true", help="백테스트·판정·리포트")
    args = ap.parse_args()
    if args.register:
        register()
    if args.run:
        run()
    if not (args.register or args.run):
        ap.print_help()


if __name__ == "__main__":
    main()
