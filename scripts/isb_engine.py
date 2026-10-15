#!/usr/bin/env python3
"""ISB(Institutional Buy & Sell v9.4) 엔진 파이썬 포팅 — `pinescript/ISB.pine` ENGINE 블록(24~676행)을 봉 단위로 옮겼다.

용도: TradingView 에서 쓰는 ISB 신호를 과거 데이터로 재현해 PT-1 과 비교한다 (scripts/isb_vs_pt1.py).

원칙
  - 입력값 = Pine 기본값. 차트 주기별 Auto 해석:
      일봉  Balanced · Trend Hold + TOP · S2 켬 · 추세 진입 시장 필터 켬 · REV 상위 TF(주봉) 과매도 · REV 시장 투매(60일 −5%)
      주봉  Balanced · Swing(v9.3: T1 +2R 33% 지정가 · Chandelier 4 ATR · TOP) · S2 끔 · 시장 필터 끔 · REV 시장 투매(12주 −5%)
  - 봉 하나 안의 실행 순서는 Pine 과 같다: 지표·레짐 → 피벗·스윙 → 소진·HL → 트리거·등급
      → 직전 봉 신호를 이번 봉 시가에 체결(EXIT → TRIM → BUY) → T1(Swing) → 트레일·seenS2
      → 봉 마감 판단(EXIT → TOP → ENTRY → BREAKDOWN).
  - ta 함수는 TradingView 정의를 따른다: EMA·RMA 는 처음 n 개 SMA 로 시작, RSI·ATR 은 Wilder(RMA),
    OBV 는 종가가 같으면 0, 표준편차는 모집단, percentrank 는 직전 n 개 중 현재값 이하 비율.
  - 가정 (TradingView 대조로 확인): 피벗 고점 = 왼쪽 봉 이상 · 오른쪽 봉 초과 (저점은 반대),
    lowestbars 동점 = 가장 최근 봉.
  - 상위 TF: `lookahead_on` + `[1]` 인 호출(52주 고저 · 주봉 EMA20 이격)만 "직전 완성 주" 값.
    같은 주기의 SPY·VIX·DXY 는 지연 없이 자기 봉으로 계산해 날짜로 붙인다 (빈 날은 앞값).
  - 좀비 필터: 재무 데이터가 없으면 Pine 도 통과 → 여기서도 통과 (보고서 한계에 적음).
데이터: data/cache/ohlcv_f57d11b41e4f.parquet (PT-1 엔진과 같은 캐시) ≤ 2025-09-30.
        TradingView 기본 차트(ADJ 꺼짐 = 분할만 조정, 배당 조정 안 함)와 같게, 캐시의 배당 기록으로 배당 조정을 되돌린 가격으로
        ISB 를 계산한다 (2026-10-12 ABT 대조: 배당 조정 가격으로는 S2 날짜·2024 EXIT 가 TradingView 와 달랐고, 되돌린 가격으로는 일치).
        PT-1 엔진의 손익은 원래 조정 가격 그대로이며, 엔진에 넘기는 주봉 T1 체결가만 그날 계수로 환산한다.
        VIX·DXY = yfinance ^VIX · DX-Y.NYB → data/research/isb_macro_daily.parquet (git 제외).

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/isb_engine.py --build      # 일봉·주봉 이벤트 표 캐시
  PYTHONPATH=.:src .venv/bin/python scripts/isb_engine.py --tv-check   # TradingView 대조용 신호 목록
  PYTHONPATH=.:src .venv/bin/python scripts/isb_engine.py --labels     # 라벨 조합 탐색 캐시 (scripts/isb_label_search.py)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_utils import DEV_END, RESEARCH_DIR  # noqa: E402

PINE_PATH = ROOT / "pinescript" / "ISB.pine"
PRICE_CACHE = ROOT / "data" / "cache" / "ohlcv_f57d11b41e4f.parquet"
MACRO_PATH = RESEARCH_DIR / "isb_macro_daily.parquet"
MACRO_START = "2013-01-01"
TV_CHECK_PATH = RESEARCH_DIR / "isb_tv_check.md"
TV_TICKERS = ("ABT", "CSCO", "AAPL", "NVDA", "JPM")
TV_WINDOWS = {"D": ("2024-01-01", "2025-09-30"), "W": ("2019-01-01", "2025-09-30")}
NAN = float("nan")


# ============================================================
# 설정 (ISB.pine 입력 기본값, Auto 해석)
# ============================================================
@dataclass(frozen=True)
class IsbParams:
    """ISB.pine 입력 기본값. tf = "D"(일봉) 또는 "W"(주봉) — Auto 값은 아래 속성에서 주기별로 정한다."""

    tf: str
    sig_mode: str = "Balanced"
    en_pb: bool = True
    en_bo: bool = True
    en_ea: bool = True
    en_rev: bool = True
    reg_band: float = 0.25
    reg_dead: float = 0.25
    reg_steep: float = 1.0
    base_len: int = 20
    bo_rvol_min: float = 1.5
    exh_look: int = 40
    climax_rvol: float = 2.5
    rev_rsi_max: float = 30.0
    rev_htf_min: float = 2.0
    rev_mkt_pct: float = 5.0
    max_risk_atr: float = 3.0
    trail_atr_mult: float = 4.0
    t1_r: float = 2.0
    trim_pct: float = 33.0
    blow_atr: float = 3.0
    blow_rsi: float = 75.0
    max_thin_pct: float = 30.0
    use_macro: bool = True
    mintick: float = 0.01
    sw_len: int = 3

    @property
    def daily(self) -> bool:
        return self.tf == "D"

    @property
    def reg_len(self) -> int:
        """Regime MA (Auto): 일봉 SMA150(≈30주) · 주봉 SMA30."""
        return 150 if self.daily else 30

    @property
    def reg_k(self) -> int:
        return 20 if self.daily else 5

    @property
    def use_swing(self) -> bool:
        """Exit Mode Auto: 일봉 = Trend Hold + TOP, 주봉 = Swing (v9.3)."""
        return not self.daily

    @property
    def use_top_trim(self) -> bool:
        return True  # 두 모드 모두 TOP 익절 사용 ("Trend Hold" 단독 모드가 아님)

    @property
    def use_s2(self) -> bool:
        return self.daily

    @property
    def use_mkt_filter(self) -> bool:
        return self.daily

    @property
    def need_htf(self) -> bool:
        return self.daily and self.rev_htf_min > 0

    @property
    def use_rev_mkt(self) -> bool:
        """REV Market Washout Filter Auto: 월봉·Aggressive 가 아니면 켬."""
        return self.sig_mode != "Aggressive"

    @property
    def mkt_len(self) -> int:
        return 60 if self.daily else 12

    @property
    def mkt_up_len(self) -> int:
        return 200 if self.daily else 40

    @property
    def min_grade_base(self) -> int:
        return {"Conservative": 3, "Balanced": 2, "Aggressive": 1}[self.sig_mode]

    @property
    def trim_frac(self) -> float:
        return self.trim_pct / 100.0


# ============================================================
# ta 함수 (TradingView 정의)
# ============================================================
def _f(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def shift(x: Any, k: int = 1) -> np.ndarray:
    """Pine x[k] — k 봉 전 값 (앞쪽은 NaN)."""
    x = _f(x)
    out = np.full(x.size, np.nan)
    if 0 <= k < x.size:
        out[k:] = x[: x.size - k]
    return out


def ta_sma(x: Any, n: int) -> np.ndarray:
    """ta.sma — 창 안에 NaN 이 있으면 NaN."""
    return pd.Series(_f(x)).rolling(n, min_periods=n).mean().to_numpy()


def _seeded_ewm(x: Any, n: int, alpha: float) -> np.ndarray:
    """ta.ema/ta.rma 재귀: 직전 값이 없으면 SMA(n) 로 시작, 입력이 NaN 이면 NaN → 다음 봉에서 SMA 로 다시 시작."""
    xs = _f(x).tolist()
    sma = ta_sma(x, n).tolist()
    out = [NAN] * len(xs)
    prev = NAN
    beta = 1.0 - alpha
    for i, xi in enumerate(xs):
        prev = sma[i] if prev != prev else alpha * xi + beta * prev
        out[i] = prev
    return np.array(out, dtype=float)


def ta_ema(x: Any, n: int) -> np.ndarray:
    return _seeded_ewm(x, n, 2.0 / (n + 1))


def ta_rma(x: Any, n: int) -> np.ndarray:
    return _seeded_ewm(x, n, 1.0 / n)


def ta_rsi(x: Any, n: int = 14) -> np.ndarray:
    """ta.rsi — RMA(상승폭)/RMA(하락폭). 하락폭 평균이 0 이고 상승이 있으면 100."""
    x = _f(x)
    ch = x - shift(x, 1)
    up = np.where(np.isnan(ch), np.nan, np.maximum(ch, 0.0))
    dn = np.where(np.isnan(ch), np.nan, np.maximum(-ch, 0.0))
    ru, rd = ta_rma(up, n), ta_rma(dn, n)
    out = np.full(x.size, np.nan)
    ok = ~np.isnan(ru) & ~np.isnan(rd)
    pos = ok & (rd > 0)
    out[pos] = 100.0 - 100.0 / (1.0 + ru[pos] / rd[pos])
    out[ok & (rd == 0) & (ru > 0)] = 100.0
    return out


def ta_tr(h: Any, l: Any, c: Any) -> np.ndarray:
    """ta.tr(true) — 직전 종가가 없으면 고가 − 저가."""
    h, l, c = _f(h), _f(l), _f(c)
    pc = shift(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    miss = np.isnan(pc)
    tr[miss] = (h - l)[miss]
    return tr


def ta_atr(h: Any, l: Any, c: Any, n: int = 14) -> np.ndarray:
    return ta_rma(ta_tr(h, l, c), n)


def ta_stdev(x: Any, n: int) -> np.ndarray:
    """ta.stdev — 모집단 표준편차 (biased)."""
    return pd.Series(_f(x)).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def ta_highest(x: Any, n: int) -> np.ndarray:
    return pd.Series(_f(x)).rolling(n, min_periods=n).max().to_numpy()


def ta_lowest(x: Any, n: int) -> np.ndarray:
    return pd.Series(_f(x)).rolling(n, min_periods=n).min().to_numpy()


def ta_percentrank(x: Any, n: int) -> np.ndarray:
    """ta.percentrank — 직전 n 개 값 중 현재값 이하인 비율 × 100 (직전 n 개가 모두 있어야 계산)."""
    x = _f(x)
    out = np.full(x.size, np.nan)
    if x.size > n:
        win = sliding_window_view(x, n + 1)
        prev, cur = win[:, :n], win[:, n]
        ok = ~np.isnan(prev).any(axis=1) & ~np.isnan(cur)
        val = (prev <= cur[:, None]).sum(axis=1) / n * 100.0
        out[n:] = np.where(ok, val, np.nan)
    return out


def lowest_offset(x: Any, n: int) -> np.ndarray:
    """−ta.lowestbars(x, n) — 최근 n 봉(현재 포함) 중 최저값까지의 거리(0 = 현재 봉). 동점은 가장 최근 봉 (가정)."""
    x = _f(x)
    out = np.full(x.size, np.nan)
    if x.size >= n:
        win = sliding_window_view(x, n)[:, ::-1]
        bad = np.isnan(win).any(axis=1)
        off = np.argmin(np.where(np.isnan(win), np.inf, win), axis=1).astype(float)
        off[bad] = np.nan
        out[n - 1:] = off
    return out


def pivot_high(x: Any, left: int, right: int) -> np.ndarray:
    """ta.pivothigh — right 봉 뒤에 확정. 가운데 값이 왼쪽 봉들 이상 · 오른쪽 봉들 초과 (동점 규칙은 가정)."""
    x = _f(x)
    out = np.full(x.size, np.nan)
    w = left + right + 1
    if x.size >= w:
        win = sliding_window_view(x, w)
        mid = win[:, left]
        ok = ~np.isnan(win).any(axis=1)
        if left:
            ok &= (mid[:, None] >= win[:, :left]).all(axis=1)
        if right:
            ok &= (mid[:, None] > win[:, left + 1:]).all(axis=1)
        out[w - 1:] = np.where(ok, mid, np.nan)
    return out


def pivot_low(x: Any, left: int, right: int) -> np.ndarray:
    """ta.pivotlow — pivot_high 의 반대 (왼쪽 이하 · 오른쪽 미만)."""
    x = _f(x)
    out = np.full(x.size, np.nan)
    w = left + right + 1
    if x.size >= w:
        win = sliding_window_view(x, w)
        mid = win[:, left]
        ok = ~np.isnan(win).any(axis=1)
        if left:
            ok &= (mid[:, None] <= win[:, :left]).all(axis=1)
        if right:
            ok &= (mid[:, None] < win[:, left + 1:]).all(axis=1)
        out[w - 1:] = np.where(ok, mid, np.nan)
    return out


def barssince(cond: Any) -> np.ndarray:
    """ta.barssince — 마지막으로 참이었던 봉까지의 거리 (한 번도 없으면 NaN)."""
    cond = np.asarray(cond, dtype=bool)
    n = cond.size
    idx = np.where(cond, np.arange(n), -1)
    last = np.maximum.accumulate(idx) if n else idx
    return np.where(last >= 0, np.arange(n) - last, np.nan).astype(float)


def valuewhen(cond: Any, src: Any) -> np.ndarray:
    """ta.valuewhen(cond, src, 0) — 마지막으로 참이었던 봉의 src."""
    cond = np.asarray(cond, dtype=bool)
    src = _f(src)
    n = cond.size
    idx = np.where(cond, np.arange(n), -1)
    last = np.maximum.accumulate(idx) if n else idx
    return np.where(last >= 0, src[np.maximum(last, 0)], np.nan)


def crossover(a: Any, level: float) -> np.ndarray:
    a = _f(a)
    return (a > level) & (shift(a, 1) <= level)


def _nmax(a: float, b: float) -> float:
    """Pine math.max — 하나라도 na 면 na."""
    return NAN if a != a or b != b else (a if a > b else b)


def _nmin(a: float, b: float) -> float:
    return NAN if a != a or b != b else (a if a < b else b)


def stop_level(sp: float, k: float, ref: float, atr: float, lg_atr: float, max_risk_atr: float) -> float:
    """f_stopLvl — 구조 저점 − k ATR, 기준가 대비 최대 max_risk ATR · 최소 0.5 ATR. 0 이하면 로그 ATR 로 다시 계산."""
    s = _nmin(_nmax(sp - k * atr, ref - max_risk_atr * atr), ref - 0.5 * atr)
    if s > 0 or lg_atr != lg_atr:
        return s
    return _nmin(_nmax(sp * math.exp(-k * lg_atr), ref * math.exp(-max_risk_atr * lg_atr)), ref * math.exp(-0.5 * lg_atr))


# ============================================================
# 레짐 (Weinstein Stage)
# ============================================================
def f_regime(c: np.ndarray, atr: np.ndarray, length: int, k: int, band: float, dead: float,
             steep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """f_regime — Stage 0(데이터 부족)·1 BASE·2 ADVANCE·3 TOP·4 DECLINE, MA, 기울기(ATR), 하락 둔화."""
    ma = ta_sma(c, length)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(atr > 0, (ma - shift(ma, k)) / atr, np.nan)
    side = np.where(c > ma + band * atr, 1, np.where(c < ma - band * atr, -1, 0))
    dirn = np.where(slope > dead, 1, np.where(slope < -dead, -1, 0))
    stage = np.zeros(c.size, dtype=int)
    st = 0
    sl, sd, dr = slope.tolist(), side.tolist(), dirn.tolist()
    for i in range(c.size):
        if sl[i] != sl[i]:
            st = 0
        elif sd[i] == 1 and dr[i] == 1:
            st = 2
        elif sd[i] == -1 and dr[i] == -1:
            st = 4
        elif st == 2:
            st = 2 if dr[i] == 1 else 3
        elif st == 4:
            st = 4 if dr[i] == -1 else 1
        elif st == 0:
            st = 2 if dr[i] == 1 else 4 if dr[i] == -1 else 1
        stage[i] = st
    slope_k = shift(slope, k)
    decel = (slope > -steep) & (slope > np.where(np.isnan(slope_k), slope, slope_k))
    return stage, ma, slope, decel


# ============================================================
# 데이터
# ============================================================
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """가격 없는 행 제거 + 앞쪽 가짜 평평 봉(거래량 0 · 고가=저가) 제거 — TradingView 에는 그런 봉이 없다."""
    keep = ["Open", "High", "Low", "Close", "Volume"] + (["adj_factor"] if "adj_factor" in df.columns else [])
    df = df[keep].astype(float)
    px = df[["Open", "High", "Low", "Close"]]
    df = df[px.notna().all(axis=1) & (px > 0).all(axis=1)]
    real = (df["Volume"].fillna(0) > 0) & (df["High"] > df["Low"])
    if not real.any():
        return df.iloc[0:0]
    return df.loc[real.idxmax():]


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """일봉 → 주봉 (W-FRI 기간 = TradingView 주봉의 월~금).

    인덱스 = 그 주 마지막 거래일(신호 확정), 'period' = 주 기간, 'first_day' = 그 주 첫 거래일(시가 체결일).
    """
    per = df.index.to_period("W-FRI")
    g = df.groupby(per, sort=True)
    out = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
        "Close": g["Close"].last(), "Volume": g["Volume"].sum(min_count=1),
    })
    days = pd.Series(df.index, index=per).groupby(level=0)
    out["period"] = out.index
    out["first_day"] = pd.DatetimeIndex(days.min().loc[out.index].to_numpy())
    out.index = pd.DatetimeIndex(days.max().loc[out.index].to_numpy())
    return out


def _weekly_last(s: pd.Series) -> pd.Series:
    """일별 시계열 → 주 마지막 값 (인덱스 = 주 기간 시작 시각, 정렬·병합용)."""
    s = s.dropna()
    w = s.groupby(s.index.to_period("W-FRI")).last()
    w.index = w.index.to_timestamp()
    return w


def load_macro(refresh: bool = False) -> pd.DataFrame:
    """VIX·DXY 일봉 종가 (yfinance ^VIX · DX-Y.NYB). 처음 한 번 받아 data/research 에 저장, 이후 파일을 읽는다."""
    if refresh or not MACRO_PATH.exists():
        import yfinance as yf
        end = (DEV_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(["^VIX", "DX-Y.NYB"], start=MACRO_START, end=end, auto_adjust=False,
                          progress=False, threads=False)
        close = raw["Close"]
        df = pd.DataFrame({"vix": close["^VIX"], "dxy": close["DX-Y.NYB"]})
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
        df = df.sort_index()
        if df["vix"].notna().sum() < 2000 or df["dxy"].notna().sum() < 2000:
            raise RuntimeError(f"VIX·DXY 다운로드가 모자람: {df.notna().sum().to_dict()}")
        MACRO_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(MACRO_PATH)
        print(f"[isb] VIX·DXY 저장: {MACRO_PATH.name} ({df.index.min().date()} ~ {df.index.max().date()})", flush=True)
    df = pd.read_parquet(MACRO_PATH)
    df.index = pd.to_datetime(df.index)
    return df.loc[:DEV_END]


@dataclass
class Market:
    """같은 주기 시장 시계열 (각자 자기 봉으로 계산). 인덱스: 일봉 = 날짜, 주봉 = 주 기간 시작 시각."""

    bench: pd.Series
    mkt_up: pd.Series
    vix: pd.Series
    vix_ema: pd.Series
    dxy: pd.Series
    dxy_ema: pd.Series


def build_market(spy: pd.DataFrame, macro: pd.DataFrame, p: IsbParams) -> Market:
    """SPY(벤치마크·시장 필터)·VIX·DXY 를 차트 주기 봉으로 만든다 (request.security 같은 주기, 지연 없음)."""
    vix, dxy = macro["vix"].dropna(), macro["dxy"].dropna()
    if p.daily:
        close = spy["Close"]
        idx = close.index
    else:
        wk = to_weekly(spy)
        close = pd.Series(wk["Close"].to_numpy(), index=pd.PeriodIndex(wk["period"]).to_timestamp())
        idx = close.index
        vix, dxy = _weekly_last(vix), _weekly_last(dxy)
    up = (close.to_numpy() > ta_sma(close, p.mkt_up_len)).astype(float)
    return Market(
        bench=close,
        mkt_up=pd.Series(up, index=idx),
        vix=vix, vix_ema=pd.Series(ta_ema(vix, 20), index=vix.index),
        dxy=dxy, dxy_ema=pd.Series(ta_ema(dxy, 50), index=dxy.index),
    )


def _align(s: pd.Series, keys: pd.DatetimeIndex) -> np.ndarray:
    """날짜 기준 병합: 그 날(또는 그 전 마지막) 값 — request.security gaps_off 와 같다."""
    return s.sort_index().reindex(keys, method="ffill").to_numpy(dtype=float)


# ============================================================
# 엔진
# ============================================================
ENGINE_COLS = ("pos", "rem", "hold_next", "ev_buy", "ev_exit", "ev_top", "ev_topwarn", "ev_break", "ev_t1",
               "t1_level", "t1_fill", "sig_type", "grade", "ev_stop", "exit_why", "btm", "stage")


def prepare_bars(daily: pd.DataFrame, mkt: Market, p: IsbParams) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """차트 주기 봉 + 엔진 입력(시장 시계열 병합, 상위 TF 값)."""
    if p.daily:
        bars = daily
        keys = bars.index
        wk = to_weekly(daily)
        hh = shift(ta_highest(wk["High"], 52), 1)
        ll = shift(ta_lowest(wk["Low"], 52), 1)
        e = ta_ema(wk["Close"], 20)
        a = ta_atr(wk["High"], wk["Low"], wk["Close"], 14)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(a > 0, (e - wk["Low"].to_numpy()) / a, np.nan)
        htf = np.maximum(shift(s, 1), shift(s, 2))
        wpos = pd.Index(wk["period"]).get_indexer(daily.index.to_period("W-FRI"))
        hi_prev, lo_prev, htf_str = hh[wpos], ll[wpos], htf[wpos]
    else:
        bars = to_weekly(daily)
        keys = pd.PeriodIndex(bars["period"]).to_timestamp()
        hi_prev = shift(ta_highest(bars["High"], 52), 1)
        lo_prev = shift(ta_lowest(bars["Low"], 52), 1)
        htf_str = np.full(len(bars), np.nan)
    inp = {
        "bench": _align(mkt.bench, keys), "mkt_up": _align(mkt.mkt_up, keys),
        "vix": _align(mkt.vix, keys), "vix_ema": _align(mkt.vix_ema, keys),
        "dxy": _align(mkt.dxy, keys), "dxy_ema": _align(mkt.dxy_ema, keys),
        "hi_prev": hi_prev, "lo_prev": lo_prev, "htf_str": htf_str,
    }
    return bars, inp


def run_engine(bars: pd.DataFrame, inp: dict[str, np.ndarray], p: IsbParams, ticker: str = "",
               aux_out: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """ISB 엔진 한 종목. Returns (봉별 상태 표, 가상 거래 목록).

    aux_out 을 주면 보유 관리 시뮬레이터(simulate_position)에 필요한 봉별 값을 채운다.
    """
    o, h, l, c, v = (bars[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = c.size
    ar = np.arange(n)
    err = np.errstate(invalid="ignore", divide="ignore")
    err.__enter__()

    # ── CORE SERIES ──
    atr = ta_atr(h, l, c, 14)
    ema20, ema50 = ta_ema(c, 20), ta_ema(c, 50)
    rsi = ta_rsi(c, 14)
    macd = ta_ema(c, 12) - ta_ema(c, 26)
    hist = macd - ta_ema(macd, 9)
    vol_sma = ta_sma(v, 20)
    rvol = np.where(vol_sma > 0, v / vol_sma, np.nan)
    obv = np.nancumsum(np.sign(c - shift(c, 1)) * v)
    obv_ema = ta_ema(obv, 20)
    mfm = np.where(h > l, ((c - l) - (h - c)) / (h - l), 0.0)
    cmf = np.where(vol_sma > 0, ta_sma(mfm * v, 20) / vol_sma, 0.0)
    bb_basis = ta_sma(c, 20)
    bb_width = np.where(bb_basis > 0, 4.0 * ta_stdev(c, 20) / bb_basis, np.nan)
    bb_wpct = ta_percentrank(bb_width, 100)

    # ── CONTEXT ──
    bench = inp["bench"]
    rs_line = np.where(bench > 0, c / bench, np.nan)
    p_rs = rs_line > ta_ema(rs_line, 20)
    hi52 = np.maximum(np.where(np.isnan(inp["hi_prev"]), h, inp["hi_prev"]), h)
    lo52 = np.minimum(np.where(np.isnan(inp["lo_prev"]), l, inp["lo_prev"]), l)
    range52 = np.where(hi52 > lo52, (c - lo52) / (hi52 - lo52) * 100.0, np.nan)
    thin = ((~np.isnan(v)) & (v == 0)) | (h == l)
    thin_pct = np.nan_to_num(ta_sma(thin.astype(float), 50)) * 100.0
    data_ok = thin_pct < p.max_thin_pct
    macro_risk = ((inp["vix"] > inp["vix_ema"] * 1.3).astype(int) + (inp["vix"] > 30).astype(int)
                  + (inp["dxy"] > inp["dxy_ema"]).astype(int))
    min_grade = p.min_grade_base + ((macro_risk >= 2) & p.use_macro).astype(int)
    bench_hi = ta_highest(bench, p.mkt_len)
    mkt_dd = np.where(bench_hi > 0, bench / bench_hi - 1.0, np.nan)
    mkt_up = inp["mkt_up"]
    mkt_gate = (not p.use_mkt_filter) | (np.isnan(mkt_up) | (mkt_up > 0.5))
    htf_ok = (not p.need_htf) | (inp["htf_str"] >= p.rev_htf_min)
    mkt_ok = (not p.use_rev_mkt) | (mkt_dd <= -p.rev_mkt_pct / 100.0)

    # ── E1 REGIME ──
    stage, reg_ma, _slope, reg_decel = f_regime(c, atr, p.reg_len, p.reg_k, p.reg_band, p.reg_dead, p.reg_steep)
    reg_prev = np.r_[0, stage[:-1]] if n else stage

    # ── E2 STRUCTURE (포지션과 무관한 부분) ──
    sw = p.sw_len
    ph, pl = pivot_high(h, sw, sw), pivot_low(l, sw, sw)
    rsi_piv = shift(rsi, sw)
    break_sh = np.zeros(n, dtype=bool)
    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)
    spring = np.zeros(n, dtype=bool)
    sl1_a = np.full(n, np.nan)
    sl1b_a = np.full(n, np.nan)
    sh1 = sh2 = sh1r = sh2r = NAN
    sh1_broken = True
    sl1 = sl2 = sl1r = sl2r = NAN
    sl1_bar = NAN
    PH, PL, RP, C, O, L = ph.tolist(), pl.tolist(), rsi_piv.tolist(), c.tolist(), o.tolist(), l.tolist()
    for i in range(n):
        new_ph, new_pl = PH[i] == PH[i], PL[i] == PL[i]
        if new_ph:
            sh2, sh2r, sh1, sh1r, sh1_broken = sh1, sh1r, PH[i], RP[i], False
        if new_pl:
            sl2, sl2r, sl1, sl1r, sl1_bar = sl1, sl1r, PL[i], RP[i], float(i - sw)
        if (not sh1_broken) and C[i] > sh1:
            break_sh[i] = True
            sh1_broken = True
        bull_div[i] = new_pl and sl1 < sl2 and sl1r > sl2r and sl1r < 40
        bear_div[i] = new_ph and sh1 > sh2 and sh1r < sh2r and sh1r > 65
        spring[i] = L[i] < sl1 and C[i] > sl1 and C[i] > O[i]
        sl1_a[i], sl1b_a[i] = sl1, sl1_bar
    new_pl_a = ~np.isnan(pl)

    tr = ta_tr(h, l, c)
    climax = (rvol >= p.climax_rvol) & (tr >= 1.5 * atr) & (l < shift(ta_lowest(l, 50), 1))
    wash_str = np.where(atr > 0, (ema20 - l) / atr, np.nan)
    washout = (l <= ta_lowest(l, 20)) & ((rsi <= p.rev_rsi_max) | (wash_str >= 2.5))
    exh_event = climax | bull_div | spring | washout
    ev_low = np.where(bull_div, sl1_a, l)
    ev_low_bar = np.where(bull_div, sl1b_a, ar.astype(float))
    lg_c = np.log(c)
    lg_c1 = shift(lg_c, 1)
    lg_tr = np.where(np.isnan(lg_c1), np.log(h) - np.log(l),
                     np.maximum(np.log(h), lg_c1) - np.minimum(np.log(l), lg_c1))
    lg_atr = ta_rma(lg_tr, 14)

    # ── E3 SETUPS · TRIGGERS ──
    pb_touch_bar = valuewhen(l <= ema20 + 0.25 * atr, ar)
    pb_setup = ((ar - pb_touch_bar) <= 4) & (ta_lowest(c - ema50 + 0.5 * atr, 5) >= 0) & (ta_lowest(rsi, 5) < 50)
    pb_trig = pb_setup & (c > shift(h, 1)) & (c > ema20) & (c > o)
    pb_low = ta_lowest(l, 5)
    base_hi = shift(ta_highest(h, p.base_len), 1)
    close_loc = np.where(h > l, (c - l) / (h - l), 0.5)
    low3 = ta_lowest(l, 3)
    bb_tight = shift(ta_lowest(bb_wpct, 10), 1)
    bo_trig = ((c > base_hi) & (shift(c, 1) <= shift(base_hi, 1)) & (rvol >= p.bo_rvol_min)
               & (close_loc >= 0.6) & (c - ema20 <= 2.5 * atr))
    w_lb = np.nan_to_num(lowest_offset(l, 6), nan=0.0).astype(int)
    w_idx = ar - w_lb
    w_is, w_low, w_high = washout[w_idx], l[w_idx], h[w_idx]
    rev_trig = w_is & (w_lb >= 1) & (c > w_high) & (c > shift(h, 1)) & (c > o) & (close_loc >= 0.5)
    rev_sig = rev_trig & htf_ok & mkt_ok

    # ── EVIDENCE PILLARS ──
    macd_rising = (hist > shift(hist, 1)) & (shift(hist, 1) > shift(hist, 2))
    p_mom = macd_rising | (barssince(crossover(rsi, 50)) <= 2)
    p_mom_ea = p_mom | (barssince(bull_div) <= p.exh_look)
    p_flow = (obv > obv_ema) & ((cmf > 0) | (rvol >= 1.5))
    p_val_pb = c <= ema20 + 1.0 * atr
    p_val_bo = bb_tight < 35
    p_val_ea = (range52 < 25) | (c < reg_ma - 2.0 * atr)
    grade_pb = p_mom.astype(int) + p_flow + p_val_pb + p_rs
    grade_bo = p_mom.astype(int) + p_flow + p_val_bo + p_rs
    grade_ea = p_mom_ea.astype(int) + p_flow + p_val_ea + p_rs
    perm_pb = p.en_pb & (stage == 2)
    perm_bo = p.en_bo & ((stage == 1) | (stage == 2)) & (c > reg_ma)
    if p.sig_mode == "Aggressive":
        perm_ea = p.en_ea & (stage != 0)
    elif p.sig_mode == "Balanced":
        perm_ea = p.en_ea & ((stage == 1) | ((stage == 4) & reg_decel))
    else:
        perm_ea = np.zeros(n, dtype=bool)
    perm_rev = (p.en_rev and p.sig_mode != "Conservative") & (stage != 0)
    s2_trig = (stage == 2) & (reg_prev != 2)
    top_cond = ((c - ema20 >= p.blow_atr * atr) & (rsi >= p.blow_rsi)) | bear_div
    err.__exit__(None, None, None)

    # ── E4 POSITION STATE MACHINE (+ 포지션에 묶인 소진 상태) ──
    H, E20, ATR, LGA = h.tolist(), ema20.tolist(), atr.tolist(), lg_atr.tolist()
    ST, RPV = stage.tolist(), reg_prev.tolist()
    EXH, EVL, EVLB = exh_event.tolist(), ev_low.tolist(), ev_low_bar.tolist()
    NPL, SL1, SL1B, BRK = new_pl_a.tolist(), sl1_a.tolist(), sl1b_a.tolist(), break_sh.tolist()
    TOPC, DOK, GATE, MG = top_cond.tolist(), data_ok.tolist(), mkt_gate.tolist(), min_grade.tolist()
    PPB, PBO, PEA, PRV = perm_pb.tolist(), perm_bo.tolist(), perm_ea.tolist(), perm_rev.tolist()
    PBT, PBTB, BOT, REVS, REVT = pb_trig.tolist(), pb_touch_bar.tolist(), bo_trig.tolist(), rev_sig.tolist(), rev_trig.tolist()
    HTFOK, MKTOK, S2T = htf_ok.tolist(), mkt_ok.tolist(), s2_trig.tolist()
    GPB, GBO, GEA = grade_pb.tolist(), grade_bo.tolist(), grade_ea.tolist()
    PBLOW, LOW3, WLOW, WBAR = pb_low.tolist(), low3.tolist(), w_low.tolist(), w_idx.tolist()

    trim_frac, use_swing, use_top = p.trim_frac, p.use_swing, p.use_top_trim
    pos = 0
    pend_buy = pend_trim = pend_exit = False
    sig_type, sig_grade, sig_bar, plan_stop = "", 0, -1, NAN
    entry_px = stop_px = risk_r = t1_px = hi_since = NAN
    entry_bar = -1
    trail_on = t1_done = seen_s2 = False
    trims, rem, realized, trade_cost = 0, 1.0, 0.0, 0.0
    top_armed, last_exit_bar, exit_why = True, -1, ""
    exh_low = exh_low_bar = exh_bar = hl_price = NAN
    btm_last_w = -1
    top_armed_hist: list[bool] = []
    btm_stop_hist: list[float] = []
    trade: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    dates = bars.index                                   # 신호 확정일 (봉 마감)
    fills = pd.DatetimeIndex(bars["first_day"]) if "first_day" in bars.columns else dates  # 시가 체결일
    cols: dict[str, list[Any]] = {k: [] for k in ENGINE_COLS}

    for i in range(n):
        ci, oi, hi_, st, rp = C[i], O[i], H[i], ST[i], RPV[i]
        # E2 소진 → Higher Low (EXIT·EARLY/REV 진입이 초기화하므로 같은 루프)
        if exh_low == exh_low and (ci < exh_low or i - exh_bar > p.exh_look or (st == 2 and rp != 2)):
            exh_low = exh_low_bar = exh_bar = hl_price = NAN
        if hl_price == hl_price and ci < hl_price:
            hl_price = NAN
        if EXH[i]:
            el, elb = EVL[i], EVLB[i]
            if exh_low != exh_low or el < exh_low:
                exh_low, exh_low_bar, hl_price = el, elb, NAN
            exh_bar = i
        if NPL[i] and exh_low == exh_low and SL1B[i] > exh_low_bar and SL1[i] > exh_low:
            hl_price = SL1[i]
        early = hl_price == hl_price and BRK[i] and ci > E20[i]

        ev_b = ev_t1 = ev_top = ev_x = ev_bk = ev_tw = btm = False
        ev_ty, ev_g, ev_sp, t1_lvl, t1_fill = "", 0, NAN, NAN, NAN
        # 1) 직전 봉 신호 → 이번 봉 시가 체결
        if pend_exit and pos == 1:
            ret = realized + rem * (oi / entry_px - 1.0) - trade_cost
            if trade is not None:
                trade.update(exit_date=fills[i], exit_bar=i, exit_px=oi, ret=ret,
                             R=ret * entry_px / risk_r if risk_r > 0 else 0.0, held_bars=i - entry_bar)
                trades.append(trade)
                trade = None
            pos, pend_exit, pend_trim = 0, False, False
        if pend_trim and pos == 1:
            realized += trim_frac * (oi / entry_px - 1.0)
            rem -= trim_frac
            trims += 1
            pend_trim = False
            if trade is not None:
                trade["trims"].append(("TOP", fills[i], oi))
        if pend_buy and pos == 0:
            atr_prev = ATR[i - 1] if i > 0 and ATR[i - 1] == ATR[i - 1] else ATR[i]
            fb = oi - 0.5 * atr_prev
            entry_px, entry_bar = oi, i
            if plan_stop < oi:
                stop_px = plan_stop
            elif fb > 0:
                stop_px = fb
            else:
                lga = LGA[i - 1] if i > 0 and LGA[i - 1] == LGA[i - 1] else LGA[i]
                stop_px = oi * math.exp(-0.5 * lga)
            risk_r = entry_px - stop_px
            t1_px = entry_px + p.t1_r * risk_r
            hi_since = hi_
            trail_on = t1_done = seen_s2 = False
            trims, rem, realized = 0, 1.0, 0.0
            trade_cost = 2.0 * p.mintick / entry_px + 0.002
            pos, pend_buy = 1, False
            trade = {"ticker": ticker, "tf": p.tf, "sig_type": sig_type, "grade": sig_grade,
                     "signal_date": dates[sig_bar] if sig_bar >= 0 else pd.NaT, "entry_date": fills[i], "entry_bar": i,
                     "entry_px": oi, "stop": stop_px, "risk": risk_r, "trims": [],
                     "exit_date": pd.NaT, "exit_bar": -1, "exit_px": NAN, "exit_why": "", "ret": NAN, "R": NAN,
                     "held_bars": -1}
        pos_open, rem_open = pos, rem
        # 2) T1 지정가 부분 익절 (Swing 만, 진입 다음 봉부터, 시가 갭 위면 시가 체결)
        can_trim = rem - trim_frac >= 0.15
        if use_swing and pos == 1 and not t1_done and trims < 2 and can_trim and i > entry_bar and hi_ >= t1_px:
            t1_lvl = t1_px
            t1_fill = oi if oi > t1_px else t1_px
            realized += trim_frac * (t1_fill / entry_px - 1.0)
            rem -= trim_frac
            trims += 1
            t1_done = ev_t1 = True
            if trade is not None:
                trade["trims"].append(("T1", dates[i], t1_fill))
        # 3) Chandelier 트레일 (Swing 만) · Stage 2 확인 기록
        if pos == 1:
            hi_since = hi_since if hi_since >= hi_ else hi_
            if use_swing and not trail_on and hi_since >= entry_px + risk_r:
                trail_on = True
            if trail_on:
                mult = max(p.trail_atr_mult - 1.0, 1.5) if trims > 0 else p.trail_atr_mult
                cand = hi_since - mult * ATR[i]
                stop_px = stop_px if stop_px >= cand else cand
            if st == 2:
                seen_s2 = True
        # 4) EXIT — 종가 기준
        if pos == 1 and not pend_exit:
            stop_hit = ci < stop_px
            trend_brk = st == 4 and rp != 4
            stage_end = (not use_swing) and seen_s2 and st != 2
            if stop_hit or trend_brk or stage_end:
                pend_exit = ev_x = True
                exit_why = ("Trail" if trail_on else "Stop") if stop_hit else ("Trend" if trend_brk else "Stage")
                last_exit_bar = i
                exh_low = exh_low_bar = exh_bar = hl_price = NAN
                if trade is not None:
                    trade["exit_why"] = exit_why
                    trade["exit_signal_date"] = dates[i]
        # 5) TOP — 과열 블로우오프 또는 피벗 약세 다이버전스, 에피소드당 1회 (EMA20 아래 종가에서 재무장)
        if ci < E20[i]:
            top_armed = True
        if TOPC[i] and top_armed:
            if pos == 1 and not pend_exit:
                if use_top and trims < 2 and can_trim:
                    pend_trim = ev_top = True
                elif not use_top:
                    ev_tw = True
                top_armed = False
            elif pos == 0 and (st == 2 or st == 3):
                ev_tw = True
                top_armed = False
        # 6) ENTRY — FLAT 에서만
        if pos == 0 and not pend_buy and DOK[i]:
            go_pb = PPB[i] and PBT[i] and PBTB[i] > last_exit_bar and GPB[i] >= MG[i] and GATE[i]
            go_bo = PBO[i] and BOT[i] and GBO[i] >= MG[i] and GATE[i]
            go_ea = PEA[i] and early and GEA[i] >= MG[i] and GATE[i]
            go_rv = (not (go_pb or go_bo or go_ea)) and PRV[i] and REVS[i] and WBAR[i] >= last_exit_bar
            go_s2 = p.use_s2 and S2T[i] and GATE[i] and not (go_pb or go_bo or go_ea or go_rv)
            if go_pb or go_bo or go_ea or go_rv or go_s2:
                sp = PBLOW[i] if go_pb else LOW3[i] if go_bo else hl_price if go_ea else WLOW[i] if go_rv else PBLOW[i]
                ev_ty = "PB" if go_pb else "BO" if go_bo else "EARLY" if go_ea else "REV" if go_rv else "S2"
                ev_g = GPB[i] if go_pb else GBO[i] if (go_bo or go_s2) else GEA[i]
                ev_sp = stop_level(sp, 0.5 if go_bo else 0.25, ci, ATR[i], LGA[i], p.max_risk_atr)
                plan_stop, sig_type, sig_grade, sig_bar = ev_sp, ev_ty, ev_g, i
                pend_buy = ev_b = True
                if go_ea or go_rv:
                    exh_low = exh_low_bar = exh_bar = hl_price = NAN
        # 7) BREAKDOWN — 보유 없을 때 Stage 2/3 → 4
        if pos == 0 and st == 4 and (rp == 2 or rp == 3):
            ev_bk = True
        # ◇ BOTTOM (표시 전용): REV 패턴은 나왔지만 시장 조건 때문에 매수 보류된 바닥
        if (pos == 0 and not pend_buy and PRV[i] and DOK[i] and REVT[i] and HTFOK[i]
                and WBAR[i] >= last_exit_bar and WBAR[i] != btm_last_w and not MKTOK[i]):
            btm = True
            btm_last_w = WBAR[i]
        top_armed_hist.append(top_armed)
        btm_stop_hist.append(stop_level(WLOW[i], 0.25, ci, ATR[i], LGA[i], p.max_risk_atr) if btm else NAN)

        for k, val in (("pos", pos_open), ("rem", rem_open), ("hold_next", (pos == 1 and not pend_exit) or pend_buy),
                       ("ev_buy", ev_b), ("ev_exit", ev_x), ("ev_top", ev_top), ("ev_topwarn", ev_tw),
                       ("ev_break", ev_bk), ("ev_t1", ev_t1), ("t1_level", t1_lvl), ("t1_fill", t1_fill),
                       ("sig_type", ev_ty), ("grade", ev_g), ("ev_stop", ev_sp),
                       ("exit_why", exit_why if ev_x else ""), ("btm", btm), ("stage", st)):
            cols[k].append(val)

    if trade is not None:  # 기간 끝까지 보유 중 (미청산)
        trades.append(trade)
    if aux_out is not None:
        aux_out.update({"o": o, "h": h, "c": c, "atr": atr, "ema20": ema20, "lg_atr": lg_atr, "top_cond": top_cond,
                        "stage": stage, "reg_prev": reg_prev, "top_armed_end": np.array(top_armed_hist, dtype=bool),
                        "btm_stop": np.array(btm_stop_hist, dtype=float), "fills": fills, "dates": dates})
    out = pd.DataFrame(cols, index=dates)
    if not p.daily:
        out["period"] = bars["period"].to_numpy()
    return out, trades


# ============================================================
# 일봉 날짜 이벤트 표 (오버레이·단독 백테스트 입력)
# ============================================================
EVENT_COLS = ("pos", "rem", "hold_next", "exit_full", "trim_ev", "t1_fill", "t1_sold", "buy_ev", "sig_type", "grade",
              "ev_stop", "stage")


def daily_events(eng: pd.DataFrame, daily: pd.DataFrame, p: IsbParams) -> pd.DataFrame:
    """엔진 결과 → 일봉 날짜 표.

    pos/rem   : 그날(시가 체결 후) ISB 보유 여부 · 남은 비율 — 단독 비중용 (주봉 T1 은 다음 날부터 반영)
    hold_next : 그날 종가 시점에 "다음 시가에 보유" 여부 — 매수 게이트·full 매도용
    exit_full : 그날 종가 확정 전량 매도 신호 (보유 중 EXIT, 미보유 ▼BREAKDOWN) → 다음 시가
    trim_ev   : 그날 종가 확정 33% 트림 신호 (보유 중 TRIM·TOP, 미보유 ◆TOP) → 다음 시가
    t1_fill   : 주봉 T1 장중 체결가 (그날 일봉 고가가 처음 T1 에 닿은 날, max(시가, T1)) — PT-1 엔진 가격(배당 조정) 기준으로 환산
    t1_sold   : 그 T1 체결로 판 비율 (원래 주식 대비 ⅓, 라벨 규칙의 T1 전량 매도면 그때 남은 비율 전부)
    sig_type·grade·ev_stop 은 TradingView 가격 기준 (표시용)
    buy_ev    : 그날 종가 확정 BUY (sig_type · grade · ev_stop) → 다음 시가
    주봉 신호는 그 주 마지막 거래일 종가에 확정된다.
    엔진 표에 t1_full 열(라벨 규칙 rule_frame)이 있고 참이면 그 T1 은 전량 매도 — 그날부터 그 주 남은 날은 미보유.
    """
    exit_full = eng["ev_exit"] | eng["ev_break"]
    trim_ev = eng["ev_top"] | (eng["ev_topwarn"] & (eng["pos"] == 0))
    factor = daily["adj_factor"].to_numpy(dtype=float) if "adj_factor" in daily.columns else np.ones(len(daily))
    if p.daily:
        t1d = eng["t1_fill"].to_numpy(dtype=float) * factor
        return pd.DataFrame({
            "pos": eng["pos"].astype("int8"), "rem": eng["rem"].astype("float32"),
            "hold_next": eng["hold_next"].astype(bool), "exit_full": exit_full.astype(bool),
            "trim_ev": trim_ev.astype(bool), "t1_fill": t1d.astype("float32"),
            "t1_sold": np.where(np.isnan(t1d), np.nan, p.trim_frac).astype("float32"),
            "buy_ev": eng["ev_buy"].astype(bool), "sig_type": eng["sig_type"], "grade": eng["grade"].astype("int8"),
            "ev_stop": eng["ev_stop"].astype("float32"), "stage": eng["stage"].astype("int8"),
        }, index=eng.index)

    wpos = pd.Index(eng["period"]).get_indexer(daily.index.to_period("W-FRI"))
    if (wpos < 0).any():
        raise ValueError("주봉에 없는 일봉 날짜가 있다")
    is_last = daily.index.to_numpy() == eng.index.to_numpy()[wpos]

    def pick(col: pd.Series | np.ndarray) -> np.ndarray:
        return np.asarray(col)[wpos]

    pos_w = pick(eng["pos"]).astype(int)
    rem = pick(eng["rem"]).astype(float)
    hold = np.where(is_last, pick(eng["hold_next"]), pos_w == 1).astype(bool)
    t1_fill = np.full(len(daily), np.nan)
    t1_sold = np.full(len(daily), np.nan)
    full = eng["t1_full"].to_numpy(dtype=bool) if "t1_full" in eng.columns else np.zeros(len(eng), dtype=bool)
    hi, op = daily["High"].to_numpy(), daily["Open"].to_numpy()
    for j in np.flatnonzero(eng["ev_t1"].to_numpy()):
        level = float(eng["t1_level"].iloc[j])
        days = np.flatnonzero(wpos == j)
        hit = days[hi[days] >= level - 1e-12]
        d0 = int(hit[0]) if hit.size else int(days[-1])
        t1_fill[d0] = max(op[d0], level)
        later = days[days > d0]
        if full[j]:
            t1_sold[d0] = rem[d0]
            rem[later] = 0.0
            hold[days[(days >= d0) & ~is_last[days]]] = False   # 그 주 마지막 날은 엔진 표 hold_next (다음 주 재진입 반영)
        else:
            t1_sold[d0] = p.trim_frac
            rem[later] -= p.trim_frac
    return pd.DataFrame({
        "pos": pos_w.astype("int8"), "rem": rem.astype("float32"),
        "hold_next": hold,
        "exit_full": (is_last & pick(exit_full)).astype(bool),
        "trim_ev": (is_last & pick(trim_ev)).astype(bool),
        "t1_fill": (t1_fill * factor).astype("float32"),
        "t1_sold": t1_sold.astype("float32"),
        "buy_ev": (is_last & pick(eng["ev_buy"])).astype(bool),
        "sig_type": np.where(is_last, pick(eng["sig_type"]), ""),
        "grade": np.where(is_last, pick(eng["grade"]), 0).astype("int8"),
        "ev_stop": np.where(is_last, pick(eng["ev_stop"]), np.nan).astype("float32"),
        "stage": pick(eng["stage"]).astype("int8"),
    }, index=daily.index)


# ============================================================
# 유니버스 계산 (병렬) · 캐시
# ============================================================
_W: dict[str, Any] = {}


def _init_worker(mkt: Market, tf: str) -> None:
    _W["mkt"], _W["p"] = mkt, IsbParams(tf)


def compute_ticker(ticker: str, daily: pd.DataFrame, mkt: Market,
                   p: IsbParams) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """한 종목: (엔진 봉별 표, 일봉 이벤트 표, 가상 거래)."""
    bars, inp = prepare_bars(daily, mkt, p)
    eng, trades = run_engine(bars, inp, p, ticker)
    return eng, daily_events(eng, daily, p), trades


def _worker(args: tuple[str, pd.DataFrame]) -> tuple[str, pd.DataFrame, list[dict[str, Any]]]:
    ticker, daily = args
    _, ev, trades = compute_ticker(ticker, daily, _W["mkt"], _W["p"])
    return ticker, ev, trades


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_key(tf: str) -> dict[str, Any]:
    """이벤트 캐시가 유효한지 판단하는 지문 (Pine·엔진 코드·가격·VIX/DXY)."""
    st = PRICE_CACHE.stat()
    return {"tf": tf, "basis": "tv", "pine_sha256": _sha(PINE_PATH), "engine_sha256": _sha(Path(__file__)),
            "price_cache": f"{PRICE_CACHE.name}:{st.st_size}:{int(st.st_mtime)}",
            "macro_sha256": _sha(MACRO_PATH) if MACRO_PATH.exists() else "", "dev_end": str(DEV_END.date())}


def dividend_factors(close_adj: Any, dividends: Any) -> np.ndarray:
    """yfinance 배당 조정 계수 F(t) — 조정 가격 = 실제 가격 × F(t).

    Yahoo 방식: 배당락일 e 이전 가격에 (1 − D_e / 실제 종가_{e−1}) 를 곱한다. 마지막 날(F=1)부터 거꾸로
    F(t) = F(t+1) / (1 + F(t+1)·D(t+1) / 조정 종가(t)).
    """
    c, d = _f(close_adj), np.nan_to_num(_f(dividends))
    out = np.ones(c.size)
    f = 1.0
    for t in range(c.size - 2, -1, -1):
        if d[t + 1] > 0 and c[t] == c[t] and c[t] > 0:
            f = f / (1.0 + f * d[t + 1] / c[t])
        out[t] = f
    return out


def load_daily_prices(basis: str = "tv") -> dict[str, pd.DataFrame]:
    """PT-1 엔진과 같은 가격 캐시 → 종목별 정리된 일봉 ≤ DEV_END (가격 없는 종목 제외).

    basis="tv": TradingView 기본(배당 조정 안 함) — 배당 기록으로 조정을 되돌린 가격 + 'adj_factor'(조정 = 실제 × 계수).
      배당 조정 계수는 캐시 끝(다운로드일)까지의 배당으로 정해지므로 계수 계산에만 전체 파일을 읽고, 결과는 DEV_END 에서 자른다.
      되돌린 가격은 그날 실제로 거래된 가격이라 미래 정보가 없다 (data/research/raw_close_px10y 와 1e-6 이내로 같음).
    basis="adj": 캐시의 분할·배당 조정 가격 그대로 (adj_factor = 1).
    """
    full = pd.read_parquet(PRICE_CACHE)
    full.index = pd.to_datetime(full.index)
    full = full.sort_index()
    out: dict[str, pd.DataFrame] = {}
    for t in full.columns.get_level_values(0).unique():
        x = full[t]
        x = x[x["Close"].notna()].copy()
        if x.empty:
            continue
        if basis == "tv":
            fac = dividend_factors(x["Close"], x["Dividends"] if "Dividends" in x.columns else np.zeros(len(x)))
            for k in ("Open", "High", "Low", "Close"):
                x[k] = x[k] / fac
            x["adj_factor"] = fac
        else:
            x["adj_factor"] = 1.0
        df = clean_ohlcv(x.loc[:DEV_END])
        if len(df) > 0:
            out[str(t)] = df
    return out


def build_events(tf: str, prices: dict[str, pd.DataFrame] | None = None, workers: int | None = None,
                 force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전 종목 ISB → (일봉 이벤트 표 long[date, ticker, ...], 가상 거래 표). 지문이 같으면 캐시를 읽는다."""
    ev_path = RESEARCH_DIR / f"isb_events_{tf}.parquet"
    tr_path = RESEARCH_DIR / f"isb_trades_{tf}.parquet"
    meta_path = RESEARCH_DIR / f"isb_events_{tf}.json"
    load_macro()
    key = cache_key(tf)
    if not force and ev_path.exists() and tr_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == key:
            print(f"[isb] 캐시 사용: {ev_path.name}", flush=True)
            return pd.read_parquet(ev_path), pd.read_parquet(tr_path)
    prices = prices if prices is not None else load_daily_prices()
    p = IsbParams(tf)
    mkt = build_market(prices["SPY"], load_macro(), p)
    tickers = sorted(t for t in prices if t != "SPY")
    print(f"===== ISB {tf} 계산: {len(tickers)}종목", flush=True)
    evs: list[pd.DataFrame] = []
    trades: list[dict[str, Any]] = []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mkt, tf)) as ex:
        for k, (t, ev, tr) in enumerate(ex.map(_worker, ((t, prices[t]) for t in tickers), chunksize=8), 1):
            ev = ev.copy()
            ev.insert(0, "ticker", t)
            evs.append(ev)
            trades.extend(tr)
            if k % 50 == 0 or k == len(tickers):
                print(f"  [isb {tf}] {k}/{len(tickers)} [{k / len(tickers) * 100:5.1f}%]", flush=True)
    events = pd.concat(evs).rename_axis("date").reset_index()
    events["sig_type"] = events["sig_type"].astype(str)
    trades_df = trades_frame(trades)
    events.to_parquet(ev_path, index=False)
    trades_df.to_parquet(tr_path, index=False)
    meta_path.write_text(json.dumps(key, indent=1))
    print(f"[isb] 저장: {ev_path.name} ({len(events):,}행) · {tr_path.name} ({len(trades_df):,}건)", flush=True)
    return events, trades_df


def trades_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    """가상 거래 목록 → 표 (트림은 건수·문자열로)."""
    rows = []
    for t in trades:
        r = {k: v for k, v in t.items() if k != "trims"}
        r["n_trims"] = len(t["trims"])
        r["trims"] = "; ".join(f"{kind} {pd.Timestamp(d).date()} {px:.2f}" for kind, d, px in t["trims"])
        rows.append(r)
    cols = ["ticker", "tf", "sig_type", "grade", "signal_date", "entry_date", "entry_bar", "entry_px", "stop", "risk",
            "exit_signal_date", "exit_date", "exit_bar", "exit_px", "exit_why", "ret", "R", "held_bars", "n_trims", "trims"]
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NaT if "date" in col else np.nan
    return df[cols]


def wide(events: pd.DataFrame, col: str, fill: Any = None) -> pd.DataFrame:
    """long 이벤트 표 → 날짜 × 종목 표."""
    w = events.pivot(index="date", columns="ticker", values=col)
    return w if fill is None else w.fillna(fill)


# ============================================================
# 주봉 BO · BOTTOM 매수 → EXIT 매도 (2026-10-13 사용자 규칙)
# ============================================================
BB_KEY = "WBB"


def simulate_position(aux: dict[str, Any], entry_bar: int, plan_stop: float, top_armed0: bool,
                      p: IsbParams) -> dict[str, Any]:
    """ISB 가 entry_bar 시가에 들어갔다고 보고 보유 관리만 돌린다 (엔진 1~5단계와 같은 순서·식).

    손절 = 계획 손절 (시가가 그 아래면 시가 − 0.5 ATR). Swing = T1 · Chandelier 트레일, Trend Hold = Stage 2 종료.
    내부 트림(T1·TOP)은 트레일 조임에만 쓴다 (사용자 포지션은 트림하지 않음).
    Returns exit_signal_bar (−1 = 기간 끝까지 보유) · exit_bar (체결 봉) · exit_why · trims · entry_px · stop0.
    """
    o, h, c = aux["o"], aux["h"], aux["c"]
    atr, lga, e20 = aux["atr"], aux["lg_atr"], aux["ema20"]
    topc, st, rp = aux["top_cond"], aux["stage"], aux["reg_prev"]
    n, i0 = len(c), entry_bar
    oi = float(o[i0])
    atr_prev = float(atr[i0 - 1]) if i0 > 0 and atr[i0 - 1] == atr[i0 - 1] else float(atr[i0])
    fb = oi - 0.5 * atr_prev
    if plan_stop < oi:
        stop = float(plan_stop)
    elif fb > 0:
        stop = fb
    else:
        g = float(lga[i0 - 1]) if i0 > 0 and lga[i0 - 1] == lga[i0 - 1] else float(lga[i0])
        stop = oi * math.exp(-0.5 * g)
    stop0, risk = stop, oi - stop
    t1 = oi + p.t1_r * risk
    hi_since = float(h[i0])
    trail_on = t1_done = seen_s2 = pend_trim = False
    trims, rem, tf = 0, 1.0, p.trim_frac
    top_armed = bool(top_armed0)
    log: list[tuple[str, int]] = []
    for j in range(i0, n):
        if j > i0 and pend_trim:
            trims, rem, pend_trim = trims + 1, rem - tf, False
            log.append(("TOP", j))
        can_trim = rem - tf >= 0.15
        if p.use_swing and not t1_done and trims < 2 and can_trim and j > i0 and h[j] >= t1:
            trims, rem, t1_done = trims + 1, rem - tf, True
            log.append(("T1", j))
        hj = float(h[j])
        hi_since = hi_since if hi_since >= hj else hj
        if p.use_swing and not trail_on and hi_since >= oi + risk:
            trail_on = True
        if trail_on:
            mult = max(p.trail_atr_mult - 1.0, 1.5) if trims > 0 else p.trail_atr_mult
            cand = hi_since - mult * float(atr[j])
            stop = stop if stop >= cand else cand
        if st[j] == 2:
            seen_s2 = True
        stop_hit = c[j] < stop
        trend_brk = st[j] == 4 and rp[j] != 4
        stage_end = (not p.use_swing) and seen_s2 and st[j] != 2
        if stop_hit or trend_brk or stage_end:
            why = ("Trail" if trail_on else "Stop") if stop_hit else ("Trend" if trend_brk else "Stage")
            return {"exit_signal_bar": j, "exit_bar": j + 1 if j + 1 < n else -1, "exit_why": why,
                    "trims": log, "entry_px": oi, "stop0": stop0, "t1": t1}
        if c[j] < e20[j]:
            top_armed = True
        if topc[j] and top_armed:
            if p.use_top_trim and trims < 2 and can_trim:
                pend_trim = True
            top_armed = False
    return {"exit_signal_bar": -1, "exit_bar": -1, "exit_why": "", "trims": log, "entry_px": oi, "stop0": stop0,
            "t1": t1}


def check_simulator(eng: pd.DataFrame, trades: list[dict[str, Any]], aux: dict[str, Any], p: IsbParams) -> list[str]:
    """엔진이 실제로 낸 거래마다 같은 진입·계획 손절로 simulate_position 을 돌려 EXIT 봉·이유·내부 트림 수가 같은지."""
    bad = []
    for t in trades:
        s = t["entry_bar"] - 1
        sim = simulate_position(aux, t["entry_bar"], float(eng["ev_stop"].iloc[s]), bool(aux["top_armed_end"][s]), p)
        same = sim["exit_bar"] == t["exit_bar"] and len(sim["trims"]) == len(t["trims"])
        if t["exit_bar"] >= 0:
            same = same and sim["exit_why"] == t["exit_why"]
        if not same:
            bad.append(f"{t['ticker']} {t['entry_date']:%Y-%m-%d} {t['sig_type']}: 엔진 exit {t['exit_bar']} "
                       f"{t['exit_why']} 트림 {len(t['trims'])} vs 시뮬 {sim['exit_bar']} {sim['exit_why']} {len(sim['trims'])}")
    return bad


def bo_bottom_bars(eng: pd.DataFrame, trades: list[dict[str, Any]], aux: dict[str, Any], p: IsbParams,
                   ticker: str = "") -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """사용자 규칙: ISB 차트(기본값)의 BUY·BO 또는 ◇ BOTTOM 에서만 사고, EXIT 에서만 전량 판다 (트림 안 함).

    BO     = ISB 가 실제로 든 거래 → 그 거래의 EXIT (차트 라벨 그대로).
    BOTTOM = ISB 는 사지 않는 정보용 표시 → 그 봉에서 REV 로 샀다고 보고 (손절 = 투매 저점 − 0.25 ATR,
             Pine BOTTOM 설명의 'BUY·REV 와 같은 규칙') simulate_position 으로 EXIT 를 계산.
    보유 중 나오는 다른 신호는 무시 (한 종목 한 포지션). Returns (엔진 표와 같은 열의 봉별 표, 사용자 거래 목록).
    """
    n = len(eng)
    std = {t["entry_bar"] - 1: t for t in trades}
    ev_buy, sig, btm = eng["ev_buy"].to_numpy(), eng["sig_type"].to_numpy(), eng["btm"].to_numpy()
    o, fills = aux["o"], aux["fills"]
    pos = np.zeros(n, dtype=int)
    buy = np.zeros(n, dtype=bool)
    ex = np.zeros(n, dtype=bool)
    typ = np.full(n, "", dtype=object)
    why = np.full(n, "", dtype=object)
    stop0 = np.full(n, np.nan)
    out: list[dict[str, Any]] = []
    free_from = 0
    for s in range(n - 1):
        if s < free_from:
            continue
        if ev_buy[s] and sig[s] == "BO" and s in std:
            t = std[s]
            kind, e, xb, xw, st0 = "BO", t["entry_bar"], t["exit_bar"], t["exit_why"], float(eng["ev_stop"].iloc[s])
        elif btm[s]:
            st0 = float(aux["btm_stop"][s])
            sim = simulate_position(aux, s + 1, st0, bool(aux["top_armed_end"][s]), p)
            kind, e, xb, xw = "BOTTOM", s + 1, sim["exit_bar"], sim["exit_why"]
        else:
            continue
        buy[s], typ[s], stop0[s] = True, kind, st0
        end = xb if xb >= 0 else n
        pos[e:end] = 1
        if xb >= 0:
            ex[xb - 1], why[xb - 1] = True, xw
        entry_px = float(o[e])
        exit_px = float(o[xb]) if xb >= 0 else np.nan
        cost = 2.0 * p.mintick / entry_px + 0.002
        out.append({"ticker": ticker, "tf": p.tf, "sig_type": kind, "signal_date": eng.index[s],
                    "entry_date": fills[e], "entry_bar": e, "entry_px": entry_px, "stop": st0,
                    "exit_signal_date": eng.index[xb - 1] if xb >= 0 else pd.NaT,
                    "exit_date": fills[xb] if xb >= 0 else pd.NaT, "exit_bar": xb, "exit_px": exit_px,
                    "exit_why": xw, "ret": exit_px / entry_px - 1.0 - cost if xb >= 0 else np.nan,
                    "held_bars": xb - e if xb >= 0 else -1})
        free_from = end
    hold = np.zeros(n, dtype=bool)
    if n:
        hold[:-1] = pos[1:] == 1
        hold[-1] = pos[-1] == 1 and not ex[-1]
    frame = pd.DataFrame({
        "pos": pos, "rem": 1.0, "hold_next": hold, "ev_buy": buy, "ev_exit": ex, "ev_top": False,
        "ev_topwarn": False, "ev_break": False, "ev_t1": False, "t1_level": np.nan, "t1_fill": np.nan,
        "sig_type": typ, "grade": 0, "ev_stop": stop0, "exit_why": why, "btm": False, "stage": eng["stage"].to_numpy(),
    }, index=eng.index)
    if "period" in eng.columns:
        frame["period"] = eng["period"].to_numpy()
    return frame, out


def compute_bo_bottom(ticker: str, daily: pd.DataFrame, mkt: Market) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """한 종목: 주봉 ISB → BO·BOTTOM 매수 · EXIT 매도 → (일봉 이벤트 표, 사용자 거래)."""
    p = IsbParams("W")
    bars, inp = prepare_bars(daily, mkt, p)
    aux: dict[str, Any] = {}
    eng, trades = run_engine(bars, inp, p, ticker, aux_out=aux)
    frame, user_trades = bo_bottom_bars(eng, trades, aux, p, ticker)
    return daily_events(frame, daily, p), user_trades


def _worker_bb(args: tuple[str, pd.DataFrame]) -> tuple[str, pd.DataFrame, list[dict[str, Any]]]:
    ticker, daily = args
    ev, trades = compute_bo_bottom(ticker, daily, _W["mkt"])
    return ticker, ev, trades


def build_bo_bottom_events(prices: dict[str, pd.DataFrame] | None = None, workers: int | None = None,
                           force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전 종목 주봉 BO·BOTTOM 전략 → (일봉 이벤트 표 long, 사용자 거래 표). 지문이 같으면 캐시."""
    ev_path = RESEARCH_DIR / f"isb_events_{BB_KEY}.parquet"
    tr_path = RESEARCH_DIR / f"isb_trades_{BB_KEY}.parquet"
    meta_path = RESEARCH_DIR / f"isb_events_{BB_KEY}.json"
    load_macro()
    key = {**cache_key("W"), "strategy": "BO+BOTTOM 매수 → EXIT 매도 (트림 없음)"}
    if not force and ev_path.exists() and tr_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == key:
            print(f"[isb] 캐시 사용: {ev_path.name}", flush=True)
            return pd.read_parquet(ev_path), pd.read_parquet(tr_path)
    prices = prices if prices is not None else load_daily_prices()
    mkt = build_market(prices["SPY"], load_macro(), IsbParams("W"))
    tickers = sorted(t for t in prices if t != "SPY")
    print(f"===== ISB 주봉 BO·BOTTOM 계산: {len(tickers)}종목", flush=True)
    evs: list[pd.DataFrame] = []
    trades: list[dict[str, Any]] = []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mkt, "W")) as ex:
        for k, (t, ev, tr) in enumerate(ex.map(_worker_bb, ((t, prices[t]) for t in tickers), chunksize=8), 1):
            ev = ev.copy()
            ev.insert(0, "ticker", t)
            evs.append(ev)
            trades.extend(tr)
            if k % 100 == 0 or k == len(tickers):
                print(f"  [isb BB] {k}/{len(tickers)} [{k / len(tickers) * 100:5.1f}%]", flush=True)
    events = pd.concat(evs).rename_axis("date").reset_index()
    events["sig_type"] = events["sig_type"].astype(str)
    trades_df = pd.DataFrame(trades)
    events.to_parquet(ev_path, index=False)
    trades_df.to_parquet(tr_path, index=False)
    meta_path.write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(f"[isb] 저장: {ev_path.name} ({len(events):,}행) · {tr_path.name} ({len(trades_df):,}건)", flush=True)
    return events, trades_df


# ============================================================
# 라벨 조합 (2026-10-13 사용자: 매수 라벨 · 매도 라벨 반응을 바꿔 가며 — scripts/isb_label_search.py)
# ============================================================
ENTRY_LABELS = {"D": ("PB", "BO", "EARLY", "REV", "S2", "BOTTOM"), "W": ("PB", "BO", "EARLY", "REV", "BOTTOM")}
ACTIONS = ("off", "trim", "full")
LABEL_CACHE_VERSION = 2
_ACT_KO = {"off": "무시", "trim": "⅓", "full": "전량"}


@dataclass(frozen=True)
class LabelRule:
    """차트 라벨을 보고 사고파는 규칙. 라벨 위치는 ISB 기본값 그대로이고, 어느 라벨에 어떻게 반응하는지만 바꾼다.

    entries : 이 라벨이 뜨면 다음 봉 시가에 산다 (BUY·종류, ◇ BOTTOM). BOTTOM 은 REV 로 샀다고 보고
              ISB 보유 관리로 그 뒤 라벨을 계산한다 (isb_w_bo_bottom_v1 과 같음). 보유 중 다른 매수 라벨은 무시.
    exit    : EXIT 라벨에서 전량 매도 (False = 무시 — 그 뒤에도 보유하며 차트에 이어서 뜨는 TOP(◆TOP 포함)·TRIM 라벨에 반응)
    t1      : 주봉 TRIM ⅓ (T1 +2R 지정가) — off 무시 · trim ⅓ 매도 · full 전량 (일봉은 T1 이 없어 off 만)
    top     : TRIM·TOP — off · trim ⅓ · full
    전량 매도 수단(EXIT · T1 전량 · TOP 전량)이 하나도 없는 규칙은 만들지 않는다.
    """

    tf: str
    entries: tuple[str, ...]
    exit: bool
    t1: str
    top: str

    @property
    def key(self) -> str:
        return f"{self.tf}:{'+'.join(self.entries)}:X{int(self.exit)}:T1{self.t1}:TOP{self.top}"

    @classmethod
    def parse(cls, key: str) -> "LabelRule":
        tf, ent, x, t1, top = key.split(":")
        return cls(tf, tuple(ent.split("+")), x == "X1", t1[2:], top[3:])

    def valid(self) -> bool:
        labels = ENTRY_LABELS.get(self.tf, ())
        return (bool(self.entries) and tuple(x for x in labels if x in self.entries) == self.entries
                and self.t1 in ACTIONS and self.top in ACTIONS and (self.tf == "W" or self.t1 == "off")
                and (self.exit or self.t1 == "full" or self.top == "full"))

    def describe(self) -> str:
        sells = [f"EXIT {'전량' if self.exit else '무시'}"]
        if self.tf == "W":
            sells.append(f"TRIM(T1) {_ACT_KO[self.t1]}")
        sells.append(f"TOP {_ACT_KO[self.top]}")
        return f"{'일봉' if self.tf == 'D' else '주봉'} · 매수 {'·'.join(self.entries)} · 매도 " + " · ".join(sells)


def all_rules(tf: str | None = None) -> list[LabelRule]:
    """유효한 규칙 전부 — 일봉 63 × 4 = 252, 주봉 31 × 14 = 434."""
    from itertools import combinations
    out: list[LabelRule] = []
    for f in (tf,) if tf else ("D", "W"):
        labels = ENTRY_LABELS[f]
        for k in range(1, len(labels) + 1):
            for ent in combinations(labels, k):
                for x in (True, False):
                    for t1 in ACTIONS if f == "W" else ("off",):
                        for top in ACTIONS:
                            r = LabelRule(f, ent, x, t1, top)
                            if r.valid():
                                out.append(r)
    return out


def native_rule(tf: str) -> LabelRule:
    """ISB 기본 그대로 (모든 BUY 라벨에서 사고 EXIT 전량 · TRIM ⅓) — 엔진 거래와 같아야 한다."""
    return LabelRule(tf, tuple(x for x in ENTRY_LABELS[tf] if x != "BOTTOM"), True, "trim" if tf == "W" else "off",
                     "trim")


WBB_RULE = LabelRule("W", ("BO", "BOTTOM"), True, "off", "off")   # isb_w_bo_bottom_v1 규칙


def label_signals(eng: pd.DataFrame, aux: dict[str, Any], p: IsbParams) -> list[dict[str, Any]]:
    """차트 매수 라벨(BUY·종류 · ◇ BOTTOM)마다, 그 라벨에서 샀다면 ISB 보유 관리가 낼 라벨 흐름.

    BUY 라벨은 ISB 가 실제로 든 거래와 같다 (check_simulator). BOTTOM 은 REV 규칙 손절(투매 저점 − 0.25 ATR).
    x = EXIT 신호 봉 (−1 = 기간 끝까지 없음), trims = [("T1", 장중 체결 봉) · ("TOP", 시가 체결 봉)], t1 = T1 가격.
    마지막 봉의 라벨은 진입 봉이 없어 흐름 없이 둔다.
    """
    n = len(eng)
    ev_buy, btm = eng["ev_buy"].to_numpy(dtype=bool), eng["btm"].to_numpy(dtype=bool)
    sig, grade = eng["sig_type"].to_numpy(), eng["grade"].to_numpy()
    stops = eng["ev_stop"].to_numpy(dtype=float)
    out: list[dict[str, Any]] = []
    for s in np.flatnonzero(ev_buy | btm):
        s = int(s)
        kind = str(sig[s]) if ev_buy[s] else "BOTTOM"
        stop0 = float(stops[s]) if ev_buy[s] else float(aux["btm_stop"][s])
        item: dict[str, Any] = {"s": s, "kind": kind, "grade": int(grade[s]), "stop": stop0, "x": -1, "why": "",
                                "t1": NAN, "trims": ()}
        if s + 1 < n:
            sim = simulate_position(aux, s + 1, stop0, bool(aux["top_armed_end"][s]), p)
            item.update(x=sim["exit_signal_bar"], why=sim["exit_why"], t1=sim["t1"], trims=tuple(sim["trims"]))
        out.append(item)
    return out


def rule_trades(signals: list[dict[str, Any]], n: int, rule: LabelRule,
                nat: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """규칙대로 고른 사용자 거래 (한 종목 한 포지션, 보유 중 매수 라벨은 무시).

    산 라벨의 흐름(trims · x) 안에서는 그 거래의 라벨에 반응한다. EXIT 를 무시하는 규칙은 그 EXIT 뒤에도 계속 보유하며
    차트에 이어서 뜨는 라벨에 반응한다 — nat["top"] = TOP 신호 봉 (ISB 보유 중 TRIM·TOP · 미보유 ◆TOP),
    nat["t1"] = [(봉, T1 가격)] (뒤이은 ISB 거래의 TRIM ⅓).
    end = 켠 전량 매도 중 가장 이른 것 (봉, 단계, 이유, T1 가격) — EXIT·TOP 은 신호 다음 봉 시가(단계 0), T1 은 그 봉 장중(단계 1).
    part = 그 전의 ⅓ 매도 [(봉, 단계, T1 가격)], ISB canTrim 처럼 최대 2회. end None = 기간 끝까지 보유.
    """
    out: list[dict[str, Any]] = []
    free = 0
    for sg in signals:
        s, x = sg["s"], sg["x"]
        if s < free or sg["kind"] not in rule.entries:
            continue
        t1s = [(j, 1, sg["t1"]) for k, j in sg["trims"] if k == "T1"]
        tops = [(j, 0, NAN) for k, j in sg["trims"] if k == "TOP"]
        if not rule.exit and x >= 0 and nat is not None:
            tops += [(int(b) + 1, 0, NAN) for b in nat["top"] if b > x]
            t1s += [(int(b), 1, float(lv)) for b, lv in nat["t1"] if b > x]
        ends: list[tuple[int, int, str, float]] = []
        if rule.exit and x >= 0:
            ends.append((x + 1, 0, f"EXIT·{sg['why']}", NAN))
        if rule.t1 == "full" and t1s:
            j, ph, lv = min(t1s, key=lambda t: t[:2])
            ends.append((j, ph, "T1", lv))
        if rule.top == "full" and tops:
            ends.append((min(t[0] for t in tops), 0, "TOP", NAN))
        end = min(ends, key=lambda t: t[:2]) if ends else None
        cut = end[:2] if end else (n + 1, 0)
        cand = [t for t in t1s if rule.t1 == "trim"] + [t for t in tops if rule.top == "trim"]
        part = sorted((t for t in cand if t[:2] < cut), key=lambda t: t[:2])[:2]
        out.append({**sg, "e": s + 1, "end": end, "part": part})
        free = end[0] if end else n
    return out


def rule_frame(trades: list[dict[str, Any]], index: pd.DatetimeIndex, stage: np.ndarray, period: np.ndarray | None,
               p: IsbParams) -> pd.DataFrame:
    """사용자 거래 → 엔진 표와 같은 열의 봉별 표 (daily_events 입력). t1_full = 그 봉 T1 장중 체결이 전량 매도."""
    n = len(index)
    pos = np.zeros(n, dtype=np.int8)
    rem = np.ones(n)
    hold = np.zeros(n, dtype=bool)
    ev_buy, ev_exit, ev_top, ev_t1, t1_full = (np.zeros(n, dtype=bool) for _ in range(5))
    t1_level = np.full(n, np.nan)
    ev_stop = np.full(n, np.nan)
    sig = np.full(n, "", dtype=object)
    why = np.full(n, "", dtype=object)
    grade = np.zeros(n, dtype=np.int8)
    tf = p.trim_frac
    for t in trades:
        s, e, end = t["s"], t["e"], t["end"]
        ev_buy[s], sig[s], grade[s], ev_stop[s] = True, t["kind"], t["grade"], t["stop"]
        if e >= n:                                   # 마지막 봉 매수 신호 → 다음 시가 보유 표시만
            hold[s] = True
            continue
        last = n - 1 if end is None else (end[0] - 1 if end[1] == 0 else end[0])   # 시가에 보유한 마지막 봉
        pos[e:last + 1] = 1
        for j, ph, lv in t["part"]:
            if ph == 0:
                ev_top[j - 1] = True
                rem[j:last + 1] -= tf
            else:
                ev_t1[j], t1_level[j] = True, lv
                rem[j + 1:last + 1] -= tf
        if end is not None:
            j, ph, reason, lv = end
            if ph == 0:
                ev_exit[j - 1], why[j - 1] = True, reason
            else:
                ev_t1[j], t1_level[j], t1_full[j] = True, lv, True
    if n:
        hold[:-1] |= pos[1:] == 1
        hold[-1] |= bool(pos[-1] == 1 and not ev_exit[-1] and not t1_full[-1])
    frame = pd.DataFrame({
        "pos": pos, "rem": rem, "hold_next": hold, "ev_buy": ev_buy, "ev_exit": ev_exit, "ev_top": ev_top,
        "ev_topwarn": False, "ev_break": False, "ev_t1": ev_t1, "t1_level": t1_level, "t1_fill": np.nan,
        "t1_full": t1_full, "sig_type": sig, "grade": grade, "ev_stop": ev_stop, "exit_why": why, "btm": False,
        "stage": stage,
    }, index=index)
    if period is not None:
        frame["period"] = period
    return frame


FRAME_CHECK_COLS = ("pos", "hold_next", "ev_buy", "ev_exit", "ev_top", "ev_t1")


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, label: str) -> list[str]:
    """봉별 표 두 개가 같은지 (마지막 봉 제외 — 체결 봉이 없어 흐름이 잘린다). 남은 비율은 보유 봉만, T1 가격은 T1 봉만."""
    if len(a) != len(b) or not a.index.equals(b.index):
        return [f"{label}: 봉 수·날짜가 다름"]
    k = len(a) - 1
    bad = []
    for col in FRAME_CHECK_COLS:
        x, y = a[col].to_numpy()[:k].astype(int), b[col].to_numpy()[:k].astype(int)
        if not np.array_equal(x, y):
            bad.append(f"{label}: {col} 첫 차이 {a.index[int(np.flatnonzero(x != y)[0])]:%Y-%m-%d}")
    held = a["pos"].to_numpy()[:k] == 1
    if not np.allclose(a["rem"].to_numpy(dtype=float)[:k][held], b["rem"].to_numpy(dtype=float)[:k][held]):
        bad.append(f"{label}: 남은 비율 다름")
    t1 = a["ev_t1"].to_numpy(dtype=bool)[:k]
    if not np.allclose(a["t1_level"].to_numpy(dtype=float)[:k][t1], b["t1_level"].to_numpy(dtype=float)[:k][t1]):
        bad.append(f"{label}: T1 가격 다름")
    return bad


def rule_events(item: dict[str, Any], rule: LabelRule, p: IsbParams) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """라벨 캐시 한 종목 + 규칙 → (일봉 이벤트 표, 사용자 거래)."""
    trades = rule_trades(item["signals"], len(item["index"]), rule, item["nat"])
    frame = rule_frame(trades, item["index"], item["stage"], item["period"], p)
    return daily_events(frame, item["daily"], p), trades


def compute_labels(ticker: str, daily: pd.DataFrame, mkt: Market, p: IsbParams) -> dict[str, Any]:
    """한 종목: 엔진 1회 → 매수 라벨별 흐름 + 일봉 매핑 값. 재현 확인: ISB 기본 규칙 = 엔진 표, 주봉 BO·BOTTOM 규칙 = bo_bottom_bars."""
    bars, inp = prepare_bars(daily, mkt, p)
    aux: dict[str, Any] = {}
    eng, trades = run_engine(bars, inp, p, ticker, aux_out=aux)
    t1_bars = np.flatnonzero(eng["ev_t1"].to_numpy(dtype=bool))
    item = {"index": eng.index, "period": eng["period"].to_numpy() if "period" in eng.columns else None,
            "stage": eng["stage"].to_numpy(dtype=np.int8), "signals": label_signals(eng, aux, p),
            "nat": {"top": np.flatnonzero((eng["ev_top"] | eng["ev_topwarn"]).to_numpy(dtype=bool)),
                    "t1": [(int(b), float(eng["t1_level"].iloc[b])) for b in t1_bars]},
            "daily": daily[["Open", "High", "adj_factor"]].astype("float64")}
    n = len(eng)
    native = native_rule(p.tf)
    bad = compare_frames(rule_frame(rule_trades(item["signals"], n, native, item["nat"]), item["index"],
                                    item["stage"], item["period"], p), eng, f"{ticker} {native.key}")
    if not p.daily:
        bb, _ = bo_bottom_bars(eng, trades, aux, p, ticker)
        bad += compare_frames(rule_frame(rule_trades(item["signals"], n, WBB_RULE, item["nat"]), item["index"],
                                         item["stage"], item["period"], p), bb, f"{ticker} {WBB_RULE.key}")
    item["check"] = bad
    return item


def _worker_labels(args: tuple[str, pd.DataFrame]) -> tuple[str, dict[str, Any]]:
    ticker, daily = args
    return ticker, compute_labels(ticker, daily, _W["mkt"], _W["p"])


def build_label_cache(tf: str, prices: dict[str, pd.DataFrame] | None = None, workers: int | None = None,
                      force: bool = False) -> dict[str, dict[str, Any]]:
    """전 종목 매수 라벨별 흐름 캐시 (data/research/isb_labels_{tf}.pkl). 재현 확인이 하나라도 틀리면 멈춘다."""
    path = RESEARCH_DIR / f"isb_labels_{tf}.pkl"
    meta_path = RESEARCH_DIR / f"isb_labels_{tf}.json"
    load_macro()
    key = {**cache_key(tf), "labels": LABEL_CACHE_VERSION}
    if not force and path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == key:
        print(f"[isb] 캐시 사용: {path.name}", flush=True)
        return pd.read_pickle(path)
    prices = prices if prices is not None else load_daily_prices()
    p = IsbParams(tf)
    mkt = build_market(prices["SPY"], load_macro(), p)
    tickers = sorted(t for t in prices if t != "SPY")
    print(f"===== ISB {tf} 라벨 흐름 계산: {len(tickers)}종목", flush=True)
    out: dict[str, dict[str, Any]] = {}
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mkt, tf)) as ex:
        for k, (t, item) in enumerate(ex.map(_worker_labels, ((t, prices[t]) for t in tickers), chunksize=8), 1):
            out[t] = item
            if k % 100 == 0 or k == len(tickers):
                print(f"  [isb 라벨 {tf}] {k}/{len(tickers)} [{k / len(tickers) * 100:5.1f}%]", flush=True)
    bad = [b for item in out.values() for b in item["check"]]
    if bad:
        raise RuntimeError(f"라벨 규칙이 엔진·지난 연구 표를 재현하지 못함 ({len(bad)}건): " + " / ".join(bad[:5]))
    n_sig = sum(len(item["signals"]) for item in out.values())
    print(f"[isb] 라벨 규칙 재현 확인: {len(out)}종목 · 매수 라벨 {n_sig:,}개 — ISB 기본 규칙 = 엔진"
          + (" · BO·BOTTOM 규칙 = 지난 연구" if tf == "W" else ""), flush=True)
    pd.to_pickle(out, path)
    meta_path.write_text(json.dumps(key, indent=1))
    return out


# ============================================================
# TradingView 대조 목록
# ============================================================
_GRADE_ICON = {4: "🏆", 3: "⭐", 2: "⚡"}


def event_lines(eng: pd.DataFrame, p: IsbParams, start: str, end: str) -> list[tuple[str, str]]:
    """엔진 봉별 표 → [(봉 표기, 라벨)] (TradingView 차트에 찍히는 라벨과 같은 순서)."""
    sel = eng.loc[start:end]
    lines: list[tuple[str, str]] = []
    for d, r in sel.iterrows():
        if p.daily:
            when = f"{d:%Y-%m-%d}"
        else:
            monday = pd.Period(r["period"], "W-FRI").start_time + pd.Timedelta(days=2)
            when = f"{monday:%Y-%m-%d} 주 ({d:%m-%d} 마감)"
        labels: list[str] = []
        if r["ev_buy"]:
            g = int(r["grade"])
            grade = "" if r["sig_type"] == "REV" else f" {_GRADE_ICON.get(g, '·')}{g}/4"
            labels.append(f"BUY·{r['sig_type']}{grade} (손절 {r['ev_stop']:.2f})")
        if r["btm"]:
            labels.append("◇ BOTTOM")
        up: list[str] = []
        if r["ev_t1"]:
            up.append(f"TRIM ⅓ (T1 {r['t1_level']:.2f})")
        if r["ev_top"]:
            up.append("TRIM·TOP")
        if r["ev_exit"]:
            up.append(f"EXIT · {r['exit_why']}")
        if r["ev_break"]:
            up = ["▼ BREAKDOWN"]
        if r["ev_topwarn"]:
            up.append("◆ TOP")
        labels.extend(up)
        if labels:
            lines.append((when, " · ".join(labels)))
    return lines


def _state_at(eng: pd.DataFrame, p: IsbParams, start: str) -> str:
    """구간 첫 봉의 ISB 상태 — 이미 보유 중이면 그 매수 신호와 이후 트림을 적는다 (차트에서 찾기 쉽게)."""
    after = eng.loc[start:]
    if after.empty:
        return "데이터 없음"
    first = after.index[0]
    before = eng.loc[:first].iloc[:-1]
    if int(eng.at[first, "pos"]) != 1 or not before["ev_buy"].any():
        return "미보유"
    b = before.index[before["ev_buy"].to_numpy()][-1]
    since = before.loc[b:]
    trims = [f"{'TRIM ⅓' if r['ev_t1'] else 'TRIM·TOP'} {_bar_name(d, r, p)}"
             for d, r in since.iterrows() if r["ev_t1"] or r["ev_top"]]
    g = int(eng.at[b, "grade"])
    typ = eng.at[b, "sig_type"]
    grade = "" if typ == "REV" else f" {_GRADE_ICON.get(g, '·')}{g}/4"
    extra = f", 그 뒤 {' · '.join(trims)}" if trims else ""
    return f"보유 중 — {_bar_name(b, eng.loc[b], p)} BUY·{typ}{grade}{extra}"


def _bar_name(d: pd.Timestamp, r: pd.Series, p: IsbParams) -> str:
    if p.daily:
        return f"{d:%Y-%m-%d}"
    monday = pd.Period(r["period"], "W-FRI").start_time + pd.Timedelta(days=2)
    return f"{monday:%Y-%m-%d} 주"


def tv_check(tickers: tuple[str, ...] = TV_TICKERS, path: Path = TV_CHECK_PATH) -> str:
    """TradingView 대조용 신호 목록 (일봉 2024-01~2025-09, 주봉 2019-01~2025-09) → markdown 저장 + 반환."""
    prices = load_daily_prices()
    macro = load_macro()
    out = [
        "# ISB 신호 대조 목록 (파이썬 포팅 → TradingView)",
        "",
        f"- Pine: `pinescript/ISB.pine` sha256 `{_sha(PINE_PATH)[:12]}` · 입력값 기본 · 데이터 ≤ {DEV_END.date()}",
        "- TradingView **기본 차트(ADJ 배당 조정 꺼짐)** 그대로, 아래 날짜의 봉에 같은 라벨이 있는지 봐 주세요.",
        "- 라벨 위치: BUY·◇ BOTTOM 은 봉 아래, TRIM·EXIT·▼ BREAKDOWN·◆ TOP 은 봉 위. 주봉은 그 주 월요일 날짜의 봉입니다.",
        "- 체결은 다음 봉 시가 (라벨은 신호가 확정된 봉에 찍힘). 손절가는 소수점 둘째 자리까지 비슷하면 됩니다.",
        "",
    ]
    for tf in ("D", "W"):
        p = IsbParams(tf)
        mkt = build_market(prices["SPY"], macro, p)
        start, end = TV_WINDOWS[tf]
        out += [f"## {'일봉' if tf == 'D' else '주봉'} ({start} ~ {end})", ""]
        for t in tickers:
            if t not in prices:
                out += [f"### {t} — 가격 없음", ""]
                continue
            eng, _, _ = compute_ticker(t, prices[t], mkt, p)
            lines = event_lines(eng, p, start, end)
            out += [f"### {t} ({len(lines)}개)", "", f"- 구간 시작 시: {_state_at(eng, p, start)}", "",
                    "| 봉 | 라벨 |", "|---|---|"]
            out += [f"| {w} | {lab} |" for w, lab in lines] or ["| — | 이 기간 신호 없음 |"]
            out.append("")
    text = "\n".join(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="일봉·주봉 이벤트 표 캐시 만들기")
    ap.add_argument("--force", action="store_true", help="캐시가 있어도 다시 계산")
    ap.add_argument("--tv-check", action="store_true", help="TradingView 대조 목록")
    ap.add_argument("--labels", action="store_true", help="라벨 조합 탐색용 매수 라벨 흐름 캐시 (재현 확인 포함)")
    args = ap.parse_args()
    if args.tv_check:
        print(tv_check())
        print(f"\n저장: {TV_CHECK_PATH}")
    if args.build:
        for tf in ("D", "W"):
            build_events(tf, force=args.force)
    if args.labels:
        for tf in ("D", "W"):
            build_label_cache(tf, force=args.force)


if __name__ == "__main__":
    main()
