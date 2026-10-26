#!/usr/bin/env python3
"""주봉·월봉 골든 + 주봉 스퀴즈 상승 발사 (wm_sqz_v1): PT-1 을 하나도 쓰지 않는 새 전략 — 최근 5년, S&P 500 · 나스닥/NYSE.

계기: 2026-10-22 사용자 — "PT-1 을 사용하지 않고 주봉·월봉 골든크로스, 스퀴즈모멘텀 상승 발사만 가지고 SP500, NASDAQ/NYSE
5년 백테스트. 아무 다른 로직 없다고 생각하고 플랜을 짜 봐 — 매도·몇 개·교체·후보 다 새로." 사용자가 고른 것:
5년 통째로 (2025-10~ 의 옛 홀드아웃 1년 포함), 골든크로스 = TradingView 차트의 EMA10·EMA20 선, 스퀴즈 발사는 주봉만.

신호 (끝난 봉만, 분할만 반영한 가격 = TradingView 기본 차트 — screener.ma_cross · screener.squeeze_momentum, 사용자 차트와 일치 확인):
  주봉 골든 = 그 주봉의 EMA10 > EMA20 (Pine ta.ema)            월봉 골든 = 그날 기준 마지막으로 끝난 달의 EMA10 > EMA20
  주봉 발사 = 직전 주 검은 십자(스퀴즈 중) → 이번 주 풀림 + 연두 막대 (모멘텀 > 0 이고 오름) — LazyBear 기본값
후보 = 주봉 발사가 난 주의 마지막 거래일에 주봉 골든 · 월봉 골든 · 거래 가능 최소선(종가 $5 이상, 20일 평균 거래대금 $500만 이상)
순위 = 발사 전에 검은 십자가 이어진 주 수 많은 순 → 모멘텀 크기(막대 값 ÷ 종가) 큰 순 → 티커순
계좌 = 5칸 균등 (그날 종가 기준 자본 ÷ 5), 빈 칸은 그날 모두 채움, 다음 거래일 시가 체결, 편도 0.1%
교체 = 칸이 다 찼을 때 새 후보가 있으면: 보유 중 '직전 끝난 주봉 막대가 약해진'(모멘텀 하락) 종목 중 수익이 가장 낮은 것을 팔고
       1등 후보를 산다 (하루 1번). 그런 종목이 없으면 새 후보는 건너뛴다
매도 (판단은 끝난 봉, 체결은 다음 거래일 시가):
  ① 손절: 일봉 종가 < 진입가 × (1 − 2 × ATR(14)%)  (ATR 은 신호일 값)
  ② 모멘텀 꺾임: 주봉 막대가 2주 연속 약해짐 (val < 직전 val) — 사이트 '힘 빠짐 2봉째 ⚠' 와 같은 기준
  ③ 추세 끝: 주봉 EMA10 < EMA20
  목표가·최대 보유일 없음
기간: 2021-10-22 ~ 2026-10-21 (마지막 끝난 일봉). 성과는 배당 포함 (yfinance Adj Close 비율), SPY·QQQ 도 같은 자료.
사전 등록 (TRIAL_LOG family wm_sqz_v1, 2건 — 결과 계산 전에 --register):
  WS-SP  그날 S&P 500 구성 종목 (sp500_membership)                  vs SPY
  WS-ALL S&P 500 + 나스닥·NYSE 지금 상장 보통주 (사이트 전체분석 목록) vs SPY — 상장폐지 종목이 빠져 결과가 좋게 나올 수 있음
  판정: SPY 이김 = ΔSharpe 95% 하한 > 0 · p ≤ 0.05/2 (paired_bootstrap 1만 · 20일 블록), 관찰 = 하한 > 0 · p 미달, 나머지 탈락
  참고 (판정 없음): 나스닥·NYSE 만 (vs SPY·QQQ), 고르는 순서 무작위 20시드, '발사만'·'골든만', 비용 2배, 연도별,
                    같은 기간과 겹치는 PT-1 (2021-10 ~ 2025-09)

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/wm_sqz_research.py --download     # 가격 받기 (이어받기 됨, 성과 계산 없음)
  PYTHONPATH=.:src .venv/bin/python scripts/wm_sqz_research.py --signals      # 주봉·월봉 신호 표 (성과 계산 없음)
  PYTHONPATH=.:src .venv/bin/python scripts/wm_sqz_research.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/wm_sqz_research.py --run
산출: tier3/WM_SQZ_REPORT.md · tier3/wm_sqz.csv · data/research/wmsqz/ (git 제외)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_utils import RESEARCH_DIR  # noqa: E402

DATA_DIR = RESEARCH_DIR / "wmsqz"
RAW_DIR = DATA_DIR / "raw"
OHLCV_PATH = DATA_DIR / "ohlcv.parquet"
TICKERS_PATH = DATA_DIR / "tickers.json"
HIST_START = "2019-01-01"           # 월봉 EMA20 (20달 + 시작값) · 주봉 스퀴즈 (20주 + 20주) 준비
TEST_START = pd.Timestamp("2021-10-22")
TEST_END = pd.Timestamp("2026-10-21")
MEMBERSHIP = ROOT / "data" / "universe" / "sp500_membership.csv"
CHUNK = 100
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


# 이름만 바뀐 S&P 종목: 구성표의 옛 티커 → 지금 티커 (구성표에서 옛 이름이 빠진 날 새 이름이 들어왔고, 새 이름 가격이
# 2019 년부터 이어짐을 확인 — 2026-10-22). 인수·파산으로 가격이 아예 없는 종목은 빠진다 (생존 편향, 보고서에 개수)
ALIAS = {"ABC": "COR", "ANTM": "ELV", "BLL": "BALL", "FLT": "CPAY", "GPS": "GAP", "NLOK": "GEN", "PEAK": "DOC",
         "PKI": "RVTY", "WLTW": "WTW", "FBHS": "FBIN", "RE": "EG", "VIAC": "PARA", "FI": "FISV", "BK": "BNY", "MMC": "MRSH"}


def yf_symbol(t: str) -> str:
    """S&P 구성표·사이트 목록 티커 → yfinance 기호 (BRK.B → BRK-B, 이름 바뀐 종목은 지금 티커)."""
    s = t.strip().upper().replace(".", "-")
    return ALIAS.get(s, s)


def build_tickers() -> dict[str, list[str]]:
    """S&P 500 = 기간 중 한 번이라도 구성 종목 · 나스닥/NYSE = 사이트 전체분석 목록 (지금 상장) · 기준 ETF."""
    from paper_trading.universe import Membership, load_membership
    mem = Membership(load_membership(MEMBERSHIP))
    sp = sorted({yf_symbol(t) for t in mem.union_between(f"{TEST_START:%Y-%m-%d}", f"{TEST_END:%Y-%m-%d}")})
    raw = subprocess.run(["git", "show", "origin/universe-data:nasdaq.json"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    nq = sorted({yf_symbol(t) for t in json.loads(raw)["items"]} - set(sp))
    return {"sp500": sp, "nasdaq_nyse": nq, "etf": ["SPY", "QQQ"]}


def _chunk_frame(raw: pd.DataFrame, syms: list[str]) -> pd.DataFrame:
    """yf.download 한 묶음 → 긴 표 (date, ticker, 필드들)."""
    out = []
    for s in syms:
        if isinstance(raw.columns, pd.MultiIndex):
            if s not in raw.columns.get_level_values(0):
                continue
            df = raw[s]
        else:
            df = raw
        if not all(f in df.columns for f in FIELDS):
            continue
        df = df[list(FIELDS)].dropna(subset=["Close"])
        if df.empty:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df["ticker"] = s
        out.append(df.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", *FIELDS, "ticker"])


def download(sleep_s: float = 1.0) -> None:
    """묶음마다 받아 raw/ 에 저장 (있으면 건너뜀 = 이어받기). 끝나면 하나로 합친다."""
    import yfinance as yf
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TICKERS_PATH.exists():
        lists = json.loads(TICKERS_PATH.read_text())
    else:
        lists = build_tickers()
        TICKERS_PATH.write_text(json.dumps(lists))
    syms = sorted({s for v in lists.values() for s in v})
    chunks = [syms[i:i + CHUNK] for i in range(0, len(syms), CHUNK)]
    print(f"===== 다운로드: {len(syms):,}종목 ({len(lists['sp500'])} S&P · {len(lists['nasdaq_nyse'])} 나스닥/NYSE) · "
          f"{len(chunks)}묶음 · {HIST_START} ~", flush=True)
    t0 = time.time()
    for k, chunk in enumerate(chunks):
        path = RAW_DIR / f"chunk_{k:03d}.parquet"
        if path.exists():
            continue
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=HIST_START, interval="1d", auto_adjust=False, actions=False,
                                  group_by="ticker", progress=False, threads=True)
                break
            except Exception as exc:  # 네트워크·제한 — 잠깐 쉬고 다시
                print(f"  묶음 {k} 실패 ({exc}) — 다시 {attempt + 1}/3", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"묶음 {k} 를 받지 못했다")
        _chunk_frame(raw, chunk).to_parquet(path, index=False)
        done = k + 1
        el = time.time() - t0
        print(f"  [{done / len(chunks) * 100:5.1f}%] 묶음 {done}/{len(chunks)} · {el / 60:.1f}분", flush=True)
        time.sleep(sleep_s)
    parts = [pd.read_parquet(p) for p in sorted(RAW_DIR.glob("chunk_*.parquet"))]
    allp = pd.concat(parts, ignore_index=True)
    allp.to_parquet(OHLCV_PATH, index=False)
    got = allp["ticker"].nunique()
    print(f"합침: {OHLCV_PATH} · {got:,}/{len(syms):,}종목 · {len(allp):,}행 · {allp['date'].min():%Y-%m-%d} ~ "
          f"{allp['date'].max():%Y-%m-%d}", flush=True)


# ============================================================
# 신호 표 (종목마다 주봉·월봉 → 날짜 × 종목 행렬)
# ============================================================
SIG_PATH = DATA_DIR / "weekly_signals.parquet"
LIQ_PRICE, LIQ_DV = 5.0, 5_000_000.0
ATR_N, ATR_MULT = 14, 2.0


def daily_frame(long: pd.DataFrame, ticker: str) -> pd.DataFrame:
    d = long[long["ticker"] == ticker].set_index("date").sort_index()
    return d[["Open", "High", "Low", "Close"]].astype(float)


def ticker_weekly(args: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """한 종목: 끝난 주봉마다 주봉 골든 · 발사 · 약해짐 · 압축 주 수 · 모멘텀 크기 · (그 주 기준 마지막 끝난 달의) 월봉 골든."""
    from screener.ma_cross import cross_frame
    from screener.squeeze_momentum import squeeze_frame
    ticker, daily = args
    daily = daily[daily["Close"].notna() & (daily[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    if len(daily) < 60:
        return pd.DataFrame()
    as_of = min(daily.index[-1], TEST_END)
    daily = daily.loc[:as_of]
    sq = squeeze_frame(daily, "W", as_of)
    w = cross_frame(daily["Close"], 10, 20, "ema", "W", as_of)
    m = cross_frame(daily["Close"], 10, 20, "ema", "M", as_of)
    wk = pd.DataFrame(index=sq.index)
    wk["fire"] = sq["fire"].astype(bool)
    wk["falling"] = sq["falling"].astype(bool)
    wk["val_pct"] = (sq["val"] / sq["close"]).astype(float)
    on = sq["sqz_on"].to_numpy(bool)
    run = np.zeros(len(on), dtype=np.int16)
    c = 0
    for i, flag in enumerate(on):            # 이번 주 앞까지 검은 십자가 이어진 주 수
        run[i] = c
        c = c + 1 if flag else 0
    wk["sq_run"] = run
    wk["w_golden"] = w["golden"].reindex(wk.index).fillna(False).astype(bool)
    wk["w_gc"] = w["gc"].reindex(wk.index).fillna(False).astype(bool)
    wk["complete"] = sq["complete"] & w["complete"].reindex(wk.index).fillna(False).astype(bool)
    mm = m[m["complete"]]
    pos = mm.index.searchsorted(wk.index, side="right") - 1      # 그 주 마지막 날까지 끝난 달 (달 날짜 = 그 달 마지막 거래일)
    wk["m_golden"] = np.where(pos >= 0, mm["golden"].to_numpy(bool)[np.maximum(pos, 0)], False)
    wk = wk[wk["complete"]].drop(columns="complete")
    wk["ticker"] = ticker
    return wk.rename_axis("date").reset_index()


def build_signals(workers: int = 6, force: bool = False) -> pd.DataFrame:
    if SIG_PATH.exists() and not force:
        return pd.read_parquet(SIG_PATH)
    from concurrent.futures import ProcessPoolExecutor
    long = pd.read_parquet(OHLCV_PATH)
    long["date"] = pd.to_datetime(long["date"])
    groups = [(t, g.set_index("date").sort_index()[["Open", "High", "Low", "Close"]].astype(float))
              for t, g in long.groupby("ticker")]
    print(f"===== 신호: {len(groups):,}종목 (동시 {workers})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(ticker_weekly, groups, chunksize=32))
    sig = pd.concat([x for x in parts if len(x)], ignore_index=True)
    sig.to_parquet(SIG_PATH, index=False)
    print(f"  주봉 행 {len(sig):,} · 발사 {int(sig['fire'].sum()):,} · 발사+주·월 골든 "
          f"{int((sig['fire'] & sig['w_golden'] & sig['m_golden']).sum()):,}", flush=True)
    return sig


# ============================================================
# 시장 행렬 · 시뮬레이터
# ============================================================
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

SLOTS, COST = 5, 0.001
SEEDS = tuple(range(1, 21))


@dataclass
class Mkt:
    dates: pd.DatetimeIndex
    tickers: list[str]
    open: np.ndarray        # 분할만 반영 (체결가)
    close: np.ndarray       # 분할만 반영 (손절 판단)
    adj: np.ndarray         # 배당 포함 종가 (평가)
    factor: np.ndarray      # Adj Close ÷ Close (그날 배당 반영 비율)
    atr: np.ndarray         # ATR(14) ÷ 종가
    dv20: np.ndarray        # 20일 평균 거래대금
    col: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.col = {t: j for j, t in enumerate(self.tickers)}


def load_mkt() -> Mkt:
    long = pd.read_parquet(OHLCV_PATH)
    long["date"] = pd.to_datetime(long["date"])
    piv = {f: long.pivot_table(index="date", columns="ticker", values=f, aggfunc="last").sort_index()
           for f in FIELDS}
    idx = piv["Close"].index
    cols = piv["Close"].columns
    piv = {f: v.reindex(index=idx, columns=cols) for f, v in piv.items()}
    c, h, lo = piv["Close"], piv["High"], piv["Low"]
    prev = c.shift(1)
    tr = np.fmax(h - lo, np.fmax((h - prev).abs(), (lo - prev).abs()))
    atr = tr.rolling(ATR_N, min_periods=ATR_N).mean() / c
    dv = (c * piv["Volume"]).rolling(20, min_periods=15).mean()
    adj = piv["Adj Close"].ffill()
    factor = (piv["Adj Close"] / c).ffill()
    return Mkt(idx, list(cols), piv["Open"].to_numpy(float), c.to_numpy(float), adj.to_numpy(float),
               factor.to_numpy(float), atr.to_numpy(float), dv.to_numpy(float))


@dataclass
class Grid:
    """주봉 신호를 일봉 달력에 놓은 행렬 (그 주 마지막 거래일 칸에만 값)."""

    week_end: np.ndarray
    fire: np.ndarray
    w_golden: np.ndarray
    m_golden: np.ndarray
    w_gc: np.ndarray
    falling: np.ndarray
    sq_run: np.ndarray
    val_pct: np.ndarray


def build_grid(mk: Mkt, sig: pd.DataFrame) -> Grid:
    shape = (len(mk.dates), len(mk.tickers))
    g = Grid(*(np.zeros(shape, dtype=bool) for _ in range(6)), np.zeros(shape, dtype=np.int16),
             np.full(shape, np.nan, dtype=np.float32))
    i = mk.dates.get_indexer(pd.to_datetime(sig["date"]))
    j = np.array([mk.col.get(t, -1) for t in sig["ticker"]])
    ok = (i >= 0) & (j >= 0)
    i, j, x = i[ok], j[ok], sig[ok]
    g.week_end[i, j] = True
    for name in ("fire", "w_golden", "m_golden", "w_gc", "falling"):
        getattr(g, name)[i, j] = x[name].to_numpy(bool)
    g.sq_run[i, j] = x["sq_run"].to_numpy(np.int16)
    g.val_pct[i, j] = x["val_pct"].to_numpy(np.float32)
    return g


def universe_mask(mk: Mkt, which: str, lists: dict[str, list[str]]) -> np.ndarray:
    """날짜 × 종목: 그날 후보가 될 수 있는 종목. SP = 그날 S&P 500 구성 · NQ = 나스닥/NYSE 목록 · ALL = 둘 다."""
    from paper_trading.universe import Membership, load_membership
    m = np.zeros((len(mk.dates), len(mk.tickers)), dtype=bool)
    if which in ("NQ", "ALL"):
        # 사이트 전체분석 나스닥·NYSE 목록 전부 (예전에 S&P 에 있다 빠진 종목도 지금 상장이면 포함)
        raw = subprocess.run(["git", "show", "origin/universe-data:nasdaq.json"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
        nq = [mk.col[s] for s in {yf_symbol(t) for t in json.loads(raw)["items"]} if s in mk.col]
        m[:, nq] = True
    if which in ("SP", "ALL"):
        mem = Membership(load_membership(MEMBERSHIP))
        cache: dict[int, np.ndarray] = {}
        for d, day in enumerate(mk.dates):
            s = mem.on(f"{day:%Y-%m-%d}")
            row = cache.get(id(s))
            if row is None:
                row = np.zeros(len(mk.tickers), dtype=bool)
                row[[mk.col[yf_symbol(t)] for t in s if yf_symbol(t) in mk.col]] = True
                cache[id(s)] = row
            m[d] |= row
    return m


@dataclass
class Pos:
    j: int
    ticker: str
    entry_i: int
    entry_px: float
    units: float            # 배당 포함 가격 기준 수량 (평가 = units × adj)
    stop_px: float
    cost_basis: float
    entry_f: float           # 진입일 배당 반영 비율 (Adj Close ÷ Close)
    weak_last: bool = False  # 마지막 끝난 주봉 막대가 약해졌나 (교체·매도 판단)


def entry_mask(g: Grid, mk: Mkt, uni: np.ndarray, kind: str) -> np.ndarray:
    """그날(주봉 마지막 날) 후보: kind = main(발사+주·월 골든) · fire(발사만) · gold(주봉 골든크로스+월 골든)."""
    with np.errstate(invalid="ignore"):
        liq = (mk.close >= LIQ_PRICE) & (mk.dv20 >= LIQ_DV)
    base = g.week_end & uni & liq
    if kind == "main":
        return base & g.fire & g.w_golden & g.m_golden
    if kind == "fire":
        return base & g.fire
    if kind == "gold":
        return base & g.w_gc & g.m_golden
    raise ValueError(kind)


def simulate(mk: Mkt, g: Grid, cand: np.ndarray, seed: int | None = None, cost: float = COST,
             start: pd.Timestamp = TEST_START, end: pd.Timestamp = TEST_END, capital: float = 10_000.0
             ) -> tuple[pd.Series, list[dict[str, Any]]]:
    """5칸 계좌. 그날 종가(주봉 신호는 그 주 마지막 날 종가)로 판단 → 다음 거래일 시가 체결."""
    import isb_vs_pt1 as iv
    i0 = int(mk.dates.searchsorted(start))
    i1 = int(mk.dates.searchsorted(end, side="right")) - 1
    cash, pos, trades, eq = capital, [], [], {}
    sell_next: list[tuple[Pos, str]] = []
    buy_next: list[tuple[int, float, float]] = []          # (열, 배정 금액, 손절폭)
    for i in range(i0, i1 + 1):
        # 1) 어제 정한 매도 → 매수를 오늘 시가에 (시가가 없으면 하루 미룸)
        later = []
        for p, why in sell_next:
            op, f = mk.open[i, p.j], mk.factor[i, p.j]
            if not (np.isfinite(op) and op > 0 and np.isfinite(f)):
                later.append((p, why))
                continue
            cash += p.units * op * f * (1 - cost)
            trades.append(_trade(p, mk, i, op, f, why, cost))
        sell_next = later
        for j, alloc, stop_pct in buy_next:
            op, f = mk.open[i, j], mk.factor[i, j]
            alloc = min(alloc, cash)
            if not (np.isfinite(op) and op > 0 and np.isfinite(f)) or alloc <= 1e-9:
                continue
            cash -= alloc
            pos.append(Pos(j, mk.tickers[j], i, float(op), alloc / (op * f * (1 + cost)), float(op * (1 - stop_pct)), alloc,
                           float(f)))
        buy_next = []
        # 2) 평가 (배당 포함 종가, 빈 날은 마지막 값)
        equity = cash + sum(p.units * mk.adj[i, p.j] for p in pos if np.isfinite(mk.adj[i, p.j])) \
            + sum(p.units * mk.adj[i, p.j] for p, _ in sell_next if np.isfinite(mk.adj[i, p.j]))
        eq[mk.dates[i]] = equity
        if i == i1:
            for p in pos + [x[0] for x in sell_next]:
                c, f = mk.close[i, p.j], mk.factor[i, p.j]
                if np.isfinite(c) and np.isfinite(f):
                    trades.append(_trade(p, mk, i, c, f, "기간 끝", cost))
            break
        # 3) 매도 판단: 일봉 손절 → (그 주 마지막 날이면) 주봉 추세 끝 · 모멘텀 2주 약해짐
        keep = []
        for p in pos:
            c = mk.close[i, p.j]
            why = None
            if np.isfinite(c) and c < p.stop_px:
                why = "손절"
            elif g.week_end[i, p.j] and i > p.entry_i:
                weak = bool(g.falling[i, p.j])
                if not g.w_golden[i, p.j]:
                    why = "주봉 데드크로스"
                elif weak and p.weak_last:
                    why = "모멘텀 2주 약해짐"
                p.weak_last = weak
            (sell_next.append((p, why)) if why else keep.append(p))
        pos = keep
        # 4) 후보 → 빈 칸 채우기, 다 찼으면 교체 1번
        js = np.flatnonzero(cand[i])
        if not len(js):
            continue
        held = {p.j for p in pos} | {p.j for p, _ in sell_next}
        js = [j for j in js if j not in held and np.isfinite(mk.open[i + 1, j]) and mk.open[i + 1, j] > 0]
        if not js:
            continue
        day = mk.dates[i]
        if seed is None:
            js.sort(key=lambda j: (-int(g.sq_run[i, j]), -float(np.nan_to_num(g.val_pct[i, j], nan=-1e9)), mk.tickers[j]))
        else:
            js.sort(key=lambda j: (iv.stable_uniform("wm_sqz", seed, day, mk.tickers[j]), mk.tickers[j]))
        avail = cash + sum(p.units * mk.adj[i, p.j] * (1 - cost) for p, _ in sell_next if np.isfinite(mk.adj[i, p.j]))
        free = SLOTS - len(pos)
        for j in js:
            if free <= 0:
                break
            alloc = min(avail, equity / SLOTS)
            if alloc <= 1e-9:
                break
            buy_next.append((j, alloc, _stop_pct(mk, i, j)))
            avail -= alloc
            free -= 1
        rest = [j for j in js if j not in {b[0] for b in buy_next}]
        if free <= 0 and rest:
            weak = [p for p in pos if p.weak_last]
            if weak:
                worst = min(weak, key=lambda p: p.units * np.nan_to_num(mk.adj[i, p.j]) / p.cost_basis)
                pos.remove(worst)
                sell_next.append((worst, "교체"))
                value = worst.units * np.nan_to_num(mk.adj[i, worst.j]) * (1 - cost)
                buy_next.append((rest[0], value, _stop_pct(mk, i, rest[0])))
    return pd.Series(eq, name="equity"), trades


def _stop_pct(mk: Mkt, i: int, j: int) -> float:
    a = mk.atr[i, j]
    return float(min(ATR_MULT * a, 0.5)) if np.isfinite(a) and a > 0 else 0.15


def _trade(p: Pos, mk: Mkt, i: int, px: float, f: float, why: str, cost: float = COST) -> dict[str, Any]:
    gross = (px * f) / (p.entry_px * p.entry_f)                     # 배당 포함
    return {"ticker": p.ticker, "entry_date": mk.dates[p.entry_i], "exit_date": mk.dates[i], "days": i - p.entry_i,
            "return_pct": gross * (1 - cost) / (1 + cost) - 1, "reason": why}


# ============================================================
# 등록 · 실행 · 판정 · 보고서
# ============================================================
FAMILY, STEP = "wm_sqz_v1", "wm_sqz"
TESTS = {"WS-SP": ("SP", "그날 S&P 500"), "WS-ALL": ("ALL", "S&P 500 + 나스닥·NYSE (지금 상장)")}
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
N_BOOT, N_BOOT_REF = 10_000, 1_000
REPORT = ROOT / "docs" / "quant_improvement" / "tier3" / "WM_SQZ_REPORT.md"
OUT_CSV = ROOT / "docs" / "quant_improvement" / "tier3" / "wm_sqz.csv"
RUNS_PATH = DATA_DIR / "runs.pkl"


def spec() -> str:
    return (f"[{TEST_START:%Y-%m-%d}~{TEST_END:%Y-%m-%d}, PT-1 안 씀, 분할만 반영 가격·TradingView 계산] 후보 = 주봉 스퀴즈 상승 "
            "발사(직전 주 검은 십자→풀림+연두) · 주봉 EMA10>EMA20 · 마지막 끝난 월봉 EMA10>EMA20 · 종가≥$5 · 20일 거래대금≥$5M; "
            "순위 = 압축 주 수 많은 순 → 막대÷종가 큰 순; 5칸 균등, 빈 칸 모두 채움, 다음날 시가, 편도 0.1%; "
            "교체 = 다 찼을 때 직전 주봉이 약해진 보유 중 수익 가장 낮은 것 ↔ 1등 후보 (하루 1번); "
            "매도 = 일봉 종가 < 진입가×(1−2×ATR14%) · 주봉 EMA10<EMA20 · 주봉 막대 2주 연속 약해짐. "
            f"WS-SP = 그날 S&P 500, WS-ALL = S&P 500 + 나스닥·NYSE 지금 상장 목록. vs SPY(배당 포함): SPY 이김 = ΔSharpe 95% "
            f"하한>0 · p ≤ 0.05/{N_TESTS}; 관찰 = 하한>0 · p 미달. 참고 = 나스닥·NYSE 만, 무작위 20시드, 발사만·골든만, 비용 2배, PT-1 겹치는 기간.")


def _trial_log() -> pd.DataFrame:
    from research_utils import TRIAL_LOG
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def register() -> None:
    from research_utils import log_trials, total_trials
    log = _trial_log()
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"등록: {FAMILY} {N_TESTS}건 · p ≤ {P_BONF:.3f} · 누적 시도 {total_trials()}건")


def _judge(st: dict[str, float]) -> str:
    if st and st["ΔSharpe_하한"] > 0 and st["p_개선아님"] <= P_BONF:
        return "SPY 이김"
    if st and st["ΔSharpe_하한"] > 0:
        return "관찰"
    return "탈락"


def _stats(name: str, eq: pd.Series, tr: list[dict[str, Any]], bench: pd.Series, n_boot: int) -> dict[str, Any]:
    import isb_vs_pt1 as iv
    from paper_trading.evaluation import cagr, max_drawdown, sharpe
    r = eq.pct_change().dropna()
    st, df = iv._cmp(r, bench, n_boot)
    a, b = df["a"].tolist(), df["b"].tolist()
    rets = np.array([t["return_pct"] for t in tr]) if tr else np.array([])
    reasons = pd.Series([t["reason"] for t in tr]).value_counts() if tr else pd.Series(dtype=int)
    held = np.zeros(len(eq))
    for t in tr:
        held[eq.index.searchsorted(t["entry_date"]):eq.index.searchsorted(t["exit_date"])] += 1
    return {"이름": name, "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a), "기준_Sharpe": sharpe(b),
            "기준_CAGR": cagr(b), "기준_MDD": max_drawdown(b), "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"),
            "상한": st.get("ΔSharpe_상한"), "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}",
            "거래수": len(tr), "승률": float((rets > 0).mean()) if len(rets) else np.nan,
            "거래당 평균": float(rets.mean()) if len(rets) else np.nan, "거래당 중앙": float(np.median(rets)) if len(rets) else np.nan,
            "보유일 중앙": float(np.median([t["days"] for t in tr])) if tr else np.nan,
            "청산": " · ".join(f"{k} {v}" for k, v in reasons.items()), "평균 보유 종목": float(held.mean()), "_st": st}


def run() -> None:
    from paper_trading.evaluation import sharpe
    from research_utils import log_trials, total_trials
    log = _trial_log()
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    already = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    lists = json.loads(TICKERS_PATH.read_text())
    sig = build_signals()
    print("===== 시장 행렬", flush=True)
    mk = load_mkt()
    g = build_grid(mk, sig)
    win = (mk.dates >= TEST_START) & (mk.dates <= TEST_END)
    bench = {k: pd.Series(mk.adj[:, mk.col[k]], index=mk.dates)[win].pct_change().dropna() for k in ("SPY", "QQQ")}
    unis = {k: universe_mask(mk, k, lists) for k in ("SP", "ALL", "NQ")}
    rows, runs, yearly = [], {}, {"SPY": bench["SPY"], "QQQ": bench["QQQ"]}

    def one(key: str, uni: str, kind: str, label: str, judged: bool, cost: float = COST, seeds: bool = True,
            bk: str = "SPY") -> dict[str, Any]:
        cand = entry_mask(g, mk, unis[uni], kind)
        eq, tr = simulate(mk, g, cand, cost=cost)
        row = _stats(label, eq, tr, bench[bk], N_BOOT if judged else N_BOOT_REF)
        row.update({"test_id": key, "기준": bk, "판정": _judge(row["_st"]) if judged else "참고",
                    "후보 주당 평균": float(cand[win].sum() / max(1, g.week_end[win].any(axis=1).sum()))})
        if seeds:
            sh = pd.Series([sharpe(simulate(mk, g, cand, seed=s, cost=cost)[0].pct_change().dropna().tolist())
                            for s in SEEDS])
            row.update({"무작위 가운데": float(sh.median()), "무작위 최소": float(sh.min()), "무작위 최대": float(sh.max()),
                        "무작위 기준 넘음": int((sh > row["기준_Sharpe"]).sum())})
        runs[key] = {"equity": eq, "trades": tr}
        yearly[key] = eq.pct_change().dropna()
        rows.append(row)
        print(f"  {key}: 거래 {len(tr)} · 평균 보유 {row['평균 보유 종목']:.1f}종목", flush=True)
        return row

    print("===== 사전 등록 2건 + 참고", flush=True)
    for key, (uni, label) in TESTS.items():
        one(key, uni, "main", label, True)
    one("WS-NQ", "NQ", "main", "나스닥·NYSE 만 (vs SPY)", False)
    one("WS-NQ-QQQ", "NQ", "main", "나스닥·NYSE 만 (vs QQQ)", False, seeds=False, bk="QQQ")
    for key, (uni, label) in TESTS.items():
        one(f"{key}-발사만", uni, "fire", f"{label} · 발사만 (골든 조건 없음)", False, seeds=False)
        one(f"{key}-골든만", uni, "gold", f"{label} · 골든만 (주봉 골든크로스 + 월봉 골든, 발사 없음)", False, seeds=False)
        one(f"{key}-비용2배", uni, "main", f"{label} · 비용 2배", False, cost=2 * COST, seeds=False)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_st"} for r in rows])
    table.to_csv(OUT_CSV, index=False)
    pd.to_pickle(runs, RUNS_PATH)
    if not already:
        log_trials(STEP, FAMILY, N_TESTS, [{
            "test_id": r["test_id"], "t": "", "판정": r["판정"],
            "메모": (f"Sharpe {r['Sharpe']:.2f} vs SPY {r['기준_Sharpe']:.2f}, ΔSharpe {r['ΔSharpe']:+.2f} "
                     f"[{r['하한']:+.2f}, {r['상한']:+.2f}] p {r['p']:.4f}, CAGR {r['CAGR']:+.1%} vs {r['기준_CAGR']:+.1%}, "
                     f"MDD {r['MDD']:.1%}, 거래 {r['거래수']}, 무작위 가운데 {r['무작위 가운데']:.2f}")}
            for r in rows if r["test_id"] in TESTS])
    else:
        print("TRIAL_LOG 에 결과를 쓰지 않는다 (이미 있음)", flush=True)
    write_report(table, yearly, total_trials())


def _pt1_overlap() -> tuple[float, float, float, float] | None:
    """PT-1 BASE (2016-01 ~ 2025-09) 중 이번 기간과 겹치는 2021-10-22 ~ 2025-09-30: PT-1 · SPY Sharpe 와 CAGR."""
    from paper_trading.evaluation import cagr, sharpe
    p = RESEARCH_DIR / "pt1_base_dev_equity.parquet"
    if not p.exists():
        return None
    e = pd.read_parquet(p).iloc[:, 0]
    e.index = pd.to_datetime(e.index)
    r = e.loc[TEST_START:].pct_change().dropna()
    long = pd.read_parquet(OHLCV_PATH, filters=[("ticker", "==", "SPY")])
    spy = long.set_index(pd.to_datetime(long["date"]))["Adj Close"].sort_index().loc[TEST_START:r.index.max()]
    sr = spy.pct_change().dropna()
    return sharpe(r.tolist()), cagr(r.tolist()), sharpe(sr.tolist()), cagr(sr.tolist())


def write_report(table: pd.DataFrame, yearly: dict[str, pd.Series], n_trials: int) -> None:
    import isb_vs_pt1 as iv
    f = iv._f
    t = table.set_index("test_id")
    sp, al = t.loc["WS-SP"], t.loc["WS-ALL"]
    passed = [k for k in TESTS if t.at[k, "판정"] == "SPY 이김"]
    watch = [k for k in TESTS if t.at[k, "판정"] == "관찰"]
    if passed:
        head = f"**결론: SPY 를 이긴다고 볼 수 있는 것이 있다 — {', '.join(passed)}.**"
    else:
        head = ("**결론: 주봉·월봉 골든 + 주봉 스퀴즈 상승 발사로 사면, 최근 5년 동안 SPY 를 이긴다고 볼 수 없다.**"
                + (f" ('관찰': {', '.join(watch)})" if watch else ""))
    pt1 = _pt1_overlap()
    lines = [
        f"# 주봉·월봉 골든 + 주봉 스퀴즈 상승 발사 ({FAMILY}) — {TEST_START:%Y-%m-%d} ~ {TEST_END:%Y-%m-%d}", "", head, "",
        f"- S&P 500: Sharpe {sp['Sharpe']:.2f} · 연 {sp['CAGR']:+.1%} · 최대낙폭 {sp['MDD']:.1%} vs SPY {sp['기준_Sharpe']:.2f} · "
        f"연 {sp['기준_CAGR']:+.1%} · {sp['기준_MDD']:.1%}",
        f"- S&P 500 + 나스닥·NYSE: Sharpe {al['Sharpe']:.2f} · 연 {al['CAGR']:+.1%} · 최대낙폭 {al['MDD']:.1%} "
        "(나스닥·NYSE 는 지금 상장된 종목만이라 결과가 실제보다 좋게 나올 수 있다)",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · {N_TESTS}건 · p ≤ 0.05/{N_TESTS} · 부트스트랩 {N_BOOT:,}회 · "
        f"누적 시도 {n_trials:,}건",
        "- 계기: 사용자 — \"PT-1 을 사용하지 않고 주봉·월봉 골든크로스, 스퀴즈모멘텀 상승 발사만 가지고 S&P 500, NASDAQ/NYSE "
        "5년 백테스트. 매도·몇 개·교체·후보 다 새로\" (2026-10-22). 기간은 사용자 결정으로 최근 5년 통째 "
        "(2025-10 ~ 의 옛 홀드아웃 1년 포함). 스퀴즈는 주봉만.",
        "", "## 규칙", "",
        "| 항목 | 내용 |", "|---|---|",
        "| 후보 | 주봉 스퀴즈 상승 발사 (직전 주 검은 십자 → 이번 주 풀림 + 연두 막대) · 그 주봉 EMA10 > EMA20 · 마지막으로 끝난 월봉 "
        "EMA10 > EMA20 — TradingView 차트의 주황·보라 선과 같은 계산, 끝난 봉만 |",
        f"| 거래 가능 최소선 | 종가 ${LIQ_PRICE:.0f} 이상 · 20일 평균 거래대금 ${LIQ_DV / 1e6:.0f}M 이상 |",
        "| 순위 | 발사 전 검은 십자가 이어진 주 수 많은 순 → 모멘텀 크기 (막대 ÷ 종가) 큰 순 |",
        "| 몇 개 · 얼마나 | 5종목 균등 (자본 ÷ 5), 빈 칸은 그날 모두 채움. 그 주 마지막 거래일 종가로 판단 → 다음 거래일 시가, 편도 0.1% |",
        "| 교체 | 다 찼을 때 새 후보가 있으면, 직전 주봉 막대가 약해진 보유 종목 중 수익이 가장 낮은 것을 팔고 1등 후보를 산다 "
        "(하루 1번). 약해진 종목이 없으면 새 후보는 건너뜀 |",
        f"| 매도 | ① 손절: 일봉 종가 < 진입가 × (1 − {ATR_MULT:g} × ATR14%) ② 주봉 EMA10 < EMA20 ③ 주봉 막대 2주 연속 약해짐. "
        "목표가·최대 보유일 없음 |",
        "| 성과 | 배당 포함 (yfinance Adj Close), SPY · QQQ 도 같은 자료 |",
        "", "## 결과", "",
        "| 검정 | Sharpe | 연수익 | 최대낙폭 | 기준 | 기준 대비 ΔSharpe [95%] | p | 구간 | 판정 |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| SPY 들고 있기 | {sp['기준_Sharpe']:.2f} | {sp['기준_CAGR']:+.1%} | {sp['기준_MDD']:.1%} | | | | | 기준 |"]
    if "WS-NQ-QQQ" in t.index:
        q = t.loc["WS-NQ-QQQ"]
        lines.append(f"| QQQ 들고 있기 | {q['기준_Sharpe']:.2f} | {q['기준_CAGR']:+.1%} | {q['기준_MDD']:.1%} | | | | | 참고 |")
    if pt1:
        lines.append(f"| PT-1 (겹치는 2021-10 ~ 2025-09만) | {pt1[0]:.2f} | {pt1[1]:+.1%} | | SPY 같은 기간 {pt1[2]:.2f} · "
                     f"{pt1[3]:+.1%} | | | | 참고 |")
    for k in ("WS-SP", "WS-ALL", "WS-NQ", "WS-NQ-QQQ"):
        if k not in t.index:
            continue
        x = t.loc[k]
        lines.append(f"| {k} {x['이름']} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | {x['기준']} | "
                     f"{f(x['ΔSharpe'], '+.2f')} [{f(x['하한'], '+.2f')}, {f(x['상한'], '+.2f')}] | {f(x['p'], '.4f')} | "
                     f"{x['구간']} | **{x['판정']}** |")
    lines += ["", "## 어떻게 거래됐나", "",
              "| 검정 | 거래 | 승률 | 거래당 평균 · 중앙 | 보유일 중앙 | 평균 보유 종목 | 후보 (주당) | 청산 이유 |",
              "|---|---|---|---|---|---|---|---|"]
    for k in ("WS-SP", "WS-ALL", "WS-NQ"):
        x = t.loc[k]
        lines.append(f"| {k} | {int(x['거래수'])} | {f(x['승률'], '.0%')} | {f(x['거래당 평균'], '+.2%')} · "
                     f"{f(x['거래당 중앙'], '+.2%')} | {f(x['보유일 중앙'], '.0f')}일 | {x['평균 보유 종목']:.1f} / {SLOTS} | "
                     f"{x['후보 주당 평균']:.1f}개 | {x['청산']} |")
    lines += ["", "## 운의 폭 — 고르는 순서만 무작위로 20번 (판정 없음)", "",
              "| 검정 | 규칙 순 | 무작위 가운데 | 최소 ~ 최대 | 기준 넘은 것 |", "|---|---|---|---|---|"]
    for k in ("WS-SP", "WS-ALL", "WS-NQ"):
        x = t.loc[k]
        lines.append(f"| {k} | {x['Sharpe']:.2f} | {x['무작위 가운데']:.2f} | {x['무작위 최소']:.2f} ~ {x['무작위 최대']:.2f} | "
                     f"{int(x['무작위 기준 넘음'])}/{len(SEEDS)} |")
    lines += ["", "## 어느 쪽이 효과인가 — 조건을 하나씩 (판정 없음)", "",
              "| 검정 | Sharpe | 연수익 | 최대낙폭 | 거래 | SPY 대비 ΔSharpe [95%] |", "|---|---|---|---|---|---|"]
    for k in TESTS:
        for suf in ("", "-발사만", "-골든만", "-비용2배"):
            x = t.loc[k + suf]
            lines.append(f"| {k + suf} {x['이름'] if suf else x['이름'] + ' (본 규칙)'} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | "
                         f"{x['MDD']:.1%} | {int(x['거래수'])} | {f(x['ΔSharpe'], '+.2f')} [{f(x['하한'], '+.2f')}, "
                         f"{f(x['상한'], '+.2f')}] |")
    keys = [k for k in ("WS-SP", "WS-ALL", "WS-NQ", "SPY", "QQQ") if k in yearly]
    yr = pd.DataFrame({k: (1 + yearly[k]).groupby(yearly[k].index.year).prod() - 1 for k in keys})
    lines += ["", "## 연도별 수익 (2021 은 10-22 부터, 2026 은 10-21 까지)", "", "| 연도 | " + " | ".join(yr.columns) + " |",
              "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    lines += ["", "## 한계", "",
              "- 나스닥·NYSE 는 지금 상장된 종목 목록이라, 지난 5년 사이 상장폐지·합병된 종목이 빠졌다 → 그쪽 결과는 실제보다 좋게 나온다. "
              "S&P 500 은 그날 구성 종목으로 봤지만, 그중에도 yfinance 에 가격이 없는 옛 종목은 빠진다.",
              "- 5년은 짧다. Sharpe 차이의 95% 구간이 넓어서 (±0.5 안팎) 큰 차이가 아니면 운과 구분되지 않는다.",
              "- 최근 1년(2025-10 ~)은 예전에 한 번 쓴 홀드아웃이다 — 사용자 결정으로 포함했다.",
              "- 예전 연구에서 골든크로스 48가지 · 스퀴즈 26가지는 각각 다 탈락했다 (MA_CROSS_REPORT · SQZ_REPORT). 둘을 합친 것은 처음.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {REPORT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", action="store_true")
    g.add_argument("--signals", action="store_true")
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
    elif args.signals:
        sig = build_signals(force=True)
        print(sig.groupby(sig["date"].dt.year)[["fire"]].sum().T.to_string())
    elif args.register:
        register()
    else:
        run()


if __name__ == "__main__":
    main()
