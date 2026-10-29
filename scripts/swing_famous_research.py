#!/usr/bin/env python3
"""유명 성장주·방어주 일봉 스윙 (swing_famous_v1) — "저점에 사고 고점에 팔기를 반복하면 그냥 들고 있기보다 나은가".

계기: 2026-10-22 사용자 — "유명한 개별 주식들, 예를 들어 테슬라·메타·마이크로소프트·넷플릭스 같은 성장주들, 그리고 유명한
경기 방어주들 열 개씩 골라서 … 저점이었다 생각했을 때 다시 사고, 고점이야 생각했을 때 다시 팔고, 이걸 계속 반복 … 주봉이나
월봉으로 보면 너무 오랫동안 올라가니까 일봉으로." 같은 날 플랜 승인 ("일단 진행해봐").
이미 본 것: idea_batch_v1 C1 RSI(2) 눌림목 (S&P 500 전체 2016~2025, 탈락) · agent 대형주 단기 반전 (French 1927~, 탈락) ·
ma_cross_v1 골든크로스 48건 · sqz 26건 (모두 탈락). 이번에는 ① 유명 종목 고정 목록 ② 같은 종목 그냥 들고 있기와 비교
③ 40년 일봉 (1986~2025-09).

종목 (결과 보기 전 고정):
  성장 G  TSLA META MSFT NFLX AAPL AMZN NVDA GOOGL AMD ADBE
  방어 D  KO PEP PG CL WMT COST JNJ MRK DUK VZ  (필수소비재·유통·제약·전력·통신)
  대조 C  2010년 한 해 평균 거래대금(종가×거래량) 상위 10 — 2010년 거래일 ≥ 200, ETF 제외, 한 회사 한 종류:
          AAPL BAC C GOOGL XOM MSFT GS JPM INTC CSCO. 2011-01-03 부터만 본다 (결과를 모르고 고른 목록 — 뒤늦은 선택 점검).
규칙 (교과서·TradingView 기본값, 튜닝 없음. 신호 = 끝난 일봉의 분할만 반영 가격 = TradingView 화면 가격):
  A1 RSI(14)            Pine ta.rsi (Wilder RMA)            RSI < 30 → 매수 · RSI > 70 → 매도
  A2 볼린저(20, 2)      Pine ta.bb (모표준편차)              종가 < 하단 → 매수 · 종가 > 상단 → 매도
  A3 BNF 25일 이격도    d = 종가/SMA25 − 1                   d ≤ −2 × (직전 252일 d 표본표준편차) → 매수 · d ≥ 0 → 매도
  A4 Connors RSI(2)     종가 > SMA200 그리고 RSI(2) < 10 → 매수 · 종가 > SMA5 → 매도
  B1 Supertrend(10, 3)  Pine ta.supertrend                   하락→상승 전환(초록) → 매수 · 상승→하락 전환(빨강) → 매도
  상태: 들고 있지 않을 때만 매수 신호, 들고 있을 때만 매도 신호를 본다 (같은 봉에 둘 다면 지금 상태 쪽만). 손절·공매도 없음.
  이웃 3×3 (가운데 = 위 설정, 판정 ⑥용):
    A1 RSI 기간 {10, 14, 20} × 기준선 {25/75, 30/70, 35/65}   A2 기간 {15, 20, 30} × σ {1.5, 2, 2.5}
    A3 이동평균 {20, 25, 30} × 평소 폭 배수 {1.5, 2, 2.5}      A4 RSI(2) 기준 {5, 10, 15} × 매도 이평 {3, 5, 10}
    B1 ATR 기간 {7, 10, 14} × 배수 {2, 3, 4}
매매·회계:
  t 종가 신호 → t+1 시가 체결. 수익 = 총수익(배당 재투자) 시가 → 다음 시가 (시가 × 수정종가/종가, sqz_research 와 같다).
  종목마다 계좌의 1/10 칸. 칸 전부 사거나 전부 판다. 쉬는 칸 = 1개월 국채 (French RF). 비용 = 바뀐 비중 × 편도 0.1%.
  그룹 수익 = 그날 시험 구간 안 종목 칸들의 평균 (매일 같은 비중으로 맞춤 — 비교 기준도 같은 방식).
  시험 구간: 종목마다 자기 282번째 봉 (0부터 센 위치, 약 1년 1개월 — 모든 규칙·이웃 지표가 계산되는 첫날, 그 전날 종가부터
  신호를 본다) ~ 2025-09-29 (그날 시가 → 2025-09-30 시가가 마지막 수익). 대조군은 2011-01-03 부터. 홀드아웃(2025-10-01~) 안 씀.
비교 기준: 같은 종목 그냥 들고 있기 (같은 날짜·같은 칸, 시작 때 한 번 매수 비용). SPY 는 참고 (1993-02~).
사전 등록 (family swing_famous_v1, 10건 = 규칙 5 × 그룹 G·D):
  '채택 후보' = 아래 ①~⑧ 모두.
   ① 전체 구간 ΔSharpe(규칙 − 들고 있기; 1개월 국채 초과수익, 일간 × √252) — 20일 블록 부트스트랩 1만 회: 95% 하한 > 0
      그리고 p(Δ ≤ 0) ≤ 0.05/10
   ② 1986–2007 · 2008–2015 · 2016–2025.09 세 구간 모두 ΔSharpe > 0
   ③ 최대 낙폭이 들고 있기보다 2%p 넘게 깊지 않음
   ④ 10종목 중 7개 이상에서 종목별 Sharpe > 그 종목 들고 있기
   ⑤ 무작위 시점 2,000회 중 규칙 Sharpe 이상이 5% 이하. 무작위 시점 = 모든 종목의 실제 보유 기록을 같은 날 수 k 만큼
      통째로 밀기 (끝은 앞으로 돌림, k = 63일 ~ 그룹에서 가장 짧은 구간 − 63일) — 매매 횟수·보유 기간·종목끼리 겹치는 모양은
      그대로, 날짜만 바뀐다.
   ⑥ 이웃 3×3 중 가운데를 뺀 8개 중 7개 이상 ΔSharpe > 0
   ⑦ 비용 2배(편도 0.2%, 들고 있기도 같은 비용) · 하루 늦은 체결(t+2 시가) 각각 ΔSharpe > 0
   ⑧ 대조군(2011-01~2025-09) 같은 규칙 ΔSharpe > 0
  '관찰' = ① 의 95% 하한 > 0 이지만 나머지 중 하나라도 못 넘음. 그 밖 = '탈락'.
  Sharpe 는 이기고 CAGR 은 들고 있기보다 낮으면 '위험 대비로만 나음' 표시 (판정은 그대로).
  참고 (판정 없음): 비용 편도 0.05% · SPY 대비 (1993-02~) · 연도별 · 종목별 1만 달러 · 거래 통계.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/swing_famous_research.py --count      # 매매 횟수만 (수익 계산 없음)
  PYTHONPATH=.:src .venv/bin/python scripts/swing_famous_research.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/swing_famous_research.py --run        # 판정 (한 번만) + 결과 저장
  PYTHONPATH=.:src .venv/bin/python scripts/swing_famous_research.py --report     # 저장된 결과로 보고서·그림만 다시
산출: tier3/SWING_FAMOUS_REPORT.md · tier3/swing_famous_*.csv · tier3/swing_famous_*.png · data/research/swing_famous/ (git 제외)
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lab_backtest import perf  # noqa: E402
from research_utils import (DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv,  # noqa: E402
                            log_trials, total_trials)

STEP = "swing_famous"
FAMILY = "swing_famous_v1"
SCRIPT = Path(__file__).resolve()
TV_OHLCV = RESEARCH_DIR / "tv_ohlcv_max.parquet"
FRENCH = RESEARCH_DIR / "lab" / "french_daily.parquet"
FROZEN = {TV_OHLCV: RESEARCH_DIR / "SHA256SUMS.txt", FRENCH: RESEARCH_DIR / "lab" / "SHA256SUMS.txt"}
OUT_DIR = RESEARCH_DIR / "swing_famous"
TRADING_DAYS = 252

GROUPS: dict[str, tuple[str, ...]] = {
    "G": ("TSLA", "META", "MSFT", "NFLX", "AAPL", "AMZN", "NVDA", "GOOGL", "AMD", "ADBE"),
    "D": ("KO", "PEP", "PG", "CL", "WMT", "COST", "JNJ", "MRK", "DUK", "VZ"),
    "C": ("AAPL", "BAC", "C", "GOOGL", "XOM", "MSFT", "GS", "JPM", "INTC", "CSCO"),
}
GROUP_KO = {"G": "성장", "D": "방어", "C": "대조"}
TEST_GROUPS = ("G", "D")
CONTROL = "C"
CONTROL_START = pd.Timestamp("2011-01-03")
CONTROL_YEAR = 2010
CONTROL_MIN_DAYS = 200
NOT_STOCKS = {"SPY", "QQQ", "IWM"}                       # ETF
SECOND_CLASS = {"GOOG", "FOX", "NWS", "UA"}               # 한 회사 두 번째 종류 (GOOGL·FOXA·NWSA·UAA 를 남긴다)
WARMUP = 282          # 시험 첫 봉 위치. 가장 긴 지표 = 이격 30일선(29) + 252일 표준편차(251) + 하루 밀기(1) → 281 에서 계산됨
COST = 0.001
COST_STRESS = 0.002
COST_REF = 0.0005
PERIODS: tuple[tuple[str, str, str], ...] = (("1986–2007", "1986-01-01", "2007-12-31"),
                                             ("2008–2015", "2008-01-01", "2015-12-31"),
                                             ("2016–2025.09", "2016-01-01", "2025-09-30"))
N_TESTS = 10
P_MAX = 0.05 / N_TESTS
N_BOOT = 10_000
BLOCK = 20
SEED = 42
N_PLACEBO = 2_000
PLACEBO_MIN_SHIFT = 63
PLACEBO_ALPHA = 0.05
MDD_TOL = 0.02
MIN_STOCK_WINS = 7
MIN_NEIGHBORS = 7
SPY_START = pd.Timestamp("1993-02-01")
GATES = ("①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧")


@dataclass(frozen=True)
class Rule:
    """규칙 하나 — 가운데(본 시험) 설정과 3×3 이웃 (행 값 × 열 값)."""

    code: str
    name: str
    style: str
    buy: str
    sell: str
    row_label: str
    rows: tuple[Any, ...]
    col_label: str
    cols: tuple[Any, ...]

    def params(self, i: int, j: int) -> tuple[float, ...]:
        """이웃 (i 행, j 열) 의 설정 튜플. 열 값이 튜플이면 (RSI 기준선 30/70) 펼친다."""
        col = self.cols[j]
        return (self.rows[i],) + (tuple(col) if isinstance(col, tuple) else (col,))

    @property
    def center(self) -> tuple[float, ...]:
        return self.params(1, 1)

    def grid(self) -> list[tuple[int, int, tuple[float, ...]]]:
        """(i, j, 설정) 9개 — 행 우선."""
        return [(i, j, self.params(i, j)) for i in range(len(self.rows)) for j in range(len(self.cols))]


RULES: tuple[Rule, ...] = (
    Rule("A1", "RSI(14) 30/70", "역추세", "RSI(14) < 30", "RSI(14) > 70",
         "RSI 기간", (10, 14, 20), "기준선", ((25, 75), (30, 70), (35, 65))),
    Rule("A2", "볼린저(20, 2)", "역추세", "종가 < 하단", "종가 > 상단",
         "기간", (15, 20, 30), "σ 배수", (1.5, 2.0, 2.5)),
    Rule("A3", "BNF 25일 이격도", "역추세", "이격 ≤ −2 × 평소 폭", "25일선 회복",
         "이동평균", (20, 25, 30), "평소 폭 배수", (1.5, 2.0, 2.5)),
    Rule("A4", "Connors RSI(2)", "역추세", "200일선 위 + RSI(2) < 10", "종가 > 5일선",
         "RSI(2) 기준", (5, 10, 15), "매도 이평", (3, 5, 10)),
    Rule("B1", "Supertrend(10, 3)", "추세 확인", "초록 전환", "빨강 전환",
         "ATR 기간", (7, 10, 14), "ATR 배수", (2.0, 3.0, 4.0)),
)
RULE_BY_CODE = {r.code: r for r in RULES}


# ── 데이터 ──────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Px:
    """한 종목의 연속된 거래일 가격 — 분할만 반영 OHLC (신호용) + 총수익 시가 (수익용)."""

    ticker: str
    dates: pd.DatetimeIndex
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    open_tr: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen() -> None:
    """고정 파일(tv_ohlcv_max · French 일간)의 SHA256 이 SHA256SUMS 와 같은지."""
    for path, sums in FROZEN.items():
        want = next((ln.split()[0] for ln in sums.read_text().splitlines()
                     if ln.strip() and Path(ln.split(maxsplit=1)[1].strip()).name == path.name), None)
        if want is None or sha256(path) != want:
            raise SystemExit(f"고정 파일 해시가 다르다: {path}")


def load_panel() -> pd.DataFrame:
    """tv_ohlcv_max (분할만 반영 OHLC + 수정종가·거래량) — DEV_END 까지."""
    return load_ohlcv(TV_OHLCV)


def stock_px(panel: pd.DataFrame, ticker: str) -> Px:
    """한 종목 — 첫 거래일부터 빈 날 없이 이어져야 한다 (아니면 오류)."""
    sub = panel.xs(ticker, axis=1, level="Ticker")[["High", "Low", "Close", "Open", "Adj Close"]].astype("float64")
    sub = sub.loc[sub["Close"].first_valid_index():]
    if sub.isna().any().any() or (sub[["Open", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: 빈 날 또는 0 이하 가격")
    return Px(ticker, pd.DatetimeIndex(sub.index), sub["High"].to_numpy(), sub["Low"].to_numpy(),
              sub["Close"].to_numpy(), (sub["Open"] * sub["Adj Close"] / sub["Close"]).to_numpy())


def load_rf(index: pd.DatetimeIndex) -> pd.Series:
    """French 1개월 국채 일간 수익 (없는 날 0)."""
    rf = pd.read_parquet(FRENCH)["rf"]
    rf.index = pd.to_datetime(rf.index)
    return rf.reindex(index).fillna(0.0)


def spy_returns(panel: pd.DataFrame) -> pd.Series:
    """SPY 총수익 시가 → 다음 시가 (t 행)."""
    x = stock_px(panel, "SPY")
    return pd.Series(x.open_tr[1:] / x.open_tr[:-1] - 1.0, index=x.dates[:-1])


def control_ranking(panel: pd.DataFrame, year: int = CONTROL_YEAR) -> pd.Series:
    """그해 평균 거래대금(종가 × 거래량, 둘 다 분할 반영이라 원래 값과 같다) 순위 — ETF·두 번째 종류 제외, 거래일 ≥ 200."""
    c = panel.xs("Close", axis=1, level="Price").loc[str(year)]
    v = panel.xs("Volume", axis=1, level="Price").loc[str(year)]
    dv = (c * v).mean()
    ok = (c.notna().sum() >= CONTROL_MIN_DAYS) & ~c.columns.isin(sorted(NOT_STOCKS | SECOND_CLASS))
    return dv[ok].dropna().sort_values(ascending=False)


# ── 지표 (Pine 과 같은 계산) ─────────────────────────────────────────────────────────────────────

def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ta.rma — 첫 n개 값의 단순평균으로 시작해 (이전 × (n−1) + 현재) / n. NaN 은 앞쪽에만 있어야 한다."""
    v = np.asarray(x, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(v))
    if ok.size < n:
        return out
    s = int(ok[0])
    if np.isnan(v[s:]).any():
        raise ValueError("rma: 중간에 NaN")
    seed = s + n - 1
    out[seed] = v[s:s + n].mean()
    if seed + 1 < v.size:
        a = 1.0 / n
        out[seed + 1:], _ = lfilter([a], [1.0, -(1.0 - a)], v[seed + 1:], zi=[(1.0 - a) * out[seed]])
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).mean().to_numpy()


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    """Pine ta.rsi — 오른 폭·내린 폭의 RMA. 내린 폭 RMA 가 0 이면 100, 오른 폭이 0 이면 0."""
    d = np.diff(close, prepend=np.nan)
    up = np.where(np.isnan(d), np.nan, np.maximum(d, 0.0))
    dn = np.where(np.isnan(d), np.nan, np.maximum(-d, 0.0))
    ru, rd = rma(up, n), rma(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + ru / rd)
    out = np.where(rd == 0, 100.0, np.where(ru == 0, 0.0, out))
    return np.where(np.isnan(ru) | np.isnan(rd), np.nan, out)


def bollinger(close: np.ndarray, n: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Pine ta.bb — (하단, 상단). 표준편차 = ta.stdev 기본 (모표준편차, ddof 0)."""
    s = pd.Series(close)
    mid = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    return (mid - k * sd).to_numpy(), (mid + k * sd).to_numpy()


def deviation(close: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(이격 d = 종가/SMA − 1, 직전 252일 d 표본표준편차 — 오늘 제외)."""
    d = close / sma(close, n) - 1.0
    sd = pd.Series(d).rolling(TRADING_DAYS).std(ddof=1).shift(1).to_numpy()
    return d, sd


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Pine ta.atr — 첫 봉 TR = 고가 − 저가, 그 뒤 max(고−저, |고−전일종가|, |저−전일종가|) 의 RMA."""
    pc = np.concatenate([[np.nan], close[:-1]])
    with np.errstate(invalid="ignore"):
        tr = np.fmax(high - low, np.fmax(np.abs(high - pc), np.abs(low - pc)))
    tr[0] = high[0] - low[0]
    return rma(tr, n)


def supertrend_dir(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int, mult: float) -> np.ndarray:
    """Pine ta.supertrend(mult, n) 의 direction: −1 = 상승 (초록), +1 = 하락 (빨강), ATR 이 없는 봉 = 0.

    Pine 원문 (v5 레퍼런스 pine_supertrend)을 그대로 옮겼다:
      lowerBand := lowerBand > prevLowerBand or close[1] < prevLowerBand ? lowerBand : prevLowerBand
      upperBand := upperBand < prevUpperBand or close[1] > prevUpperBand ? upperBand : prevUpperBand
      na(atr[1]) → 1 · 직전 supertrend 가 윗선이면 close > upperBand ? −1 : 1 · 아니면 close < lowerBand ? 1 : −1
    """
    a = atr(high, low, close, n)
    hl2 = (high + low) / 2.0
    direction = np.zeros(len(close))
    lb_prev = ub_prev = 0.0                      # nz(band[1])
    d_prev = 0
    for i in range(len(close)):
        if np.isnan(a[i]):
            continue
        ub = hl2[i] + mult * a[i]
        lb = hl2[i] - mult * a[i]
        c1 = close[i - 1] if i > 0 else np.nan
        lb = lb if (lb > lb_prev or c1 < lb_prev) else lb_prev
        ub = ub if (ub < ub_prev or c1 > ub_prev) else ub_prev
        if i == 0 or np.isnan(a[i - 1]):
            d = 1
        elif d_prev == 1:
            d = -1 if close[i] > ub else 1
        else:
            d = 1 if close[i] < lb else -1
        direction[i] = d
        lb_prev, ub_prev, d_prev = lb, ub, d
    return direction


def signals(code: str, p: tuple[float, ...], x: Px, memo: dict[tuple, np.ndarray] | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """규칙 code·설정 p 의 (매수 신호, 매도 신호) — 봉 i 종가까지만 본다. NaN 비교는 신호 없음."""
    memo = {} if memo is None else memo

    def get(key: tuple, fn: Any) -> np.ndarray:
        if key not in memo:
            memo[key] = fn()
        return memo[key]

    c = x.close
    with np.errstate(invalid="ignore"):
        if code == "A1":
            n, lo, hi = int(p[0]), p[1], p[2]
            r = get(("rsi", n), lambda: rsi(c, n))
            return r < lo, r > hi
        if code == "A2":
            n, k = int(p[0]), p[1]
            lower, upper = get(("bb", n, k), lambda: np.stack(bollinger(c, n, k)))
            return c < lower, c > upper
        if code == "A3":
            n, k = int(p[0]), p[1]
            d, sd = get(("dev", n), lambda: np.stack(deviation(c, n)))
            return d <= -k * sd, d >= 0.0
        if code == "A4":
            thr, n_exit = p[0], int(p[1])
            r2 = get(("rsi", 2), lambda: rsi(c, 2))
            s200 = get(("sma", 200), lambda: sma(c, 200))
            se = get(("sma", n_exit), lambda: sma(c, n_exit))
            return (c > s200) & (r2 < thr), c > se
        if code == "B1":
            n, m = int(p[0]), p[1]
            d = get(("st", n, m), lambda: supertrend_dir(x.high, x.low, c, n, m))
            prev = np.concatenate([[0.0], d[:-1]])
            return (d == -1) & (prev == 1), (d == 1) & (prev == -1)
    raise ValueError(code)


# ── 매매 ────────────────────────────────────────────────────────────────────────────────────────

def hold_state(entry: np.ndarray, exit_: np.ndarray, start: int) -> np.ndarray:
    """봉 i 종가 뒤 보유 상태 (0/1). start 봉 종가부터 신호를 보고, 그 전은 0.

    들고 있지 않으면 매수 신호만, 들고 있으면 매도 신호만 본다 (같은 봉에 둘 다 있어도 상태는 한 번만 바뀐다).
    """
    st = np.zeros(len(entry), dtype=np.int8)
    p, on = 0, start
    for i in np.flatnonzero(entry[start:] | exit_[start:]) + start:
        if p == 0 and entry[i]:
            p, on = 1, int(i)
        elif p == 1 and exit_[i]:
            st[on:i] = 1
            p = 0
    if p == 1:
        st[on:] = 1
    return st


def first_eval(x: Px, group: str) -> int:
    """시험 첫 봉 위치 — 자기 282번째 봉, 대조군은 2011-01-03 이후 첫 봉과 비교해 늦은 쪽."""
    f = WARMUP
    if group == CONTROL:
        f = max(f, int(x.dates.searchsorted(CONTROL_START)))
    if f >= len(x.dates) - 1:
        raise ValueError(f"{x.ticker}: 시험 구간 없음")
    return f


@dataclass
class Sleeve:
    """한 종목 칸의 시험 구간 일별 기록 (t 행 = t 시가 → t+1 시가)."""

    ticker: str
    dates: pd.DatetimeIndex
    pos: np.ndarray      # 들고 있던 비율 (0/1)
    ret: np.ndarray      # 칸 수익 (비용·국채 포함)
    r: np.ndarray        # 종목 총수익 시가 수익
    rf: np.ndarray       # 그날 국채 수익
    otr: np.ndarray      # 총수익 시가 (구간 첫날 ~ 마지막 날 다음 봉, 길이 = 구간 + 1)

    def series(self) -> pd.Series:
        return pd.Series(self.ret, index=self.dates)


def make_sleeve(x: Px, st: np.ndarray, f: int, rf_s: np.ndarray, cost: float, delay: int = 0) -> Sleeve:
    """상태 → 칸. 보유[t] = 상태[t − 1 − delay] (t−1 종가 신호 → t 시가 체결). 비용 = |보유 변화| × 편도."""
    t = np.arange(f, len(x.close) - 1)
    src = t - 1 - delay
    pos = np.where(src >= 0, st[np.maximum(src, 0)], 0).astype(float)
    r = x.open_tr[t + 1] / x.open_tr[t] - 1.0
    prev = np.concatenate([[0.0], pos[:-1]])
    ret = pos * r + (1.0 - pos) * rf_s[t] - cost * np.abs(pos - prev)
    return Sleeve(x.ticker, x.dates[t], pos, ret, r, rf_s[t], x.open_tr[f:])


def hold_sleeve(x: Px, f: int, rf_s: np.ndarray, cost: float) -> Sleeve:
    """그냥 들고 있기 — 첫날 시가에 사서 끝까지."""
    return make_sleeve(x, np.ones(len(x.close), dtype=np.int8), f, rf_s, cost)


def group_returns(sleeves: list[Sleeve]) -> pd.Series:
    """그날 구간 안 칸들의 평균 수익."""
    return pd.concat({s.ticker: s.series() for s in sleeves}, axis=1, sort=True).mean(axis=1)


def group_exposure(sleeves: list[Sleeve]) -> float:
    """날마다 칸들 보유 비율 평균 → 기간 평균."""
    return float(pd.concat({s.ticker: pd.Series(s.pos, index=s.dates) for s in sleeves}, axis=1,
                           sort=True).mean(axis=1).mean())


def trade_list(s: Sleeve, cost: float) -> pd.DataFrame:
    """칸의 거래 — 진입 시가 → 청산 시가 총수익 (편도 비용 양쪽). 끝까지 들고 있는 거래는 마지막 시가로 평가 (열림)."""
    d = np.diff(np.concatenate([[0.0], s.pos, [0.0]]))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    exit_dates = s.dates.append(pd.DatetimeIndex([pd.NaT]))
    gross = s.otr[b] / s.otr[a]
    return pd.DataFrame({"종목": s.ticker, "진입": s.dates[a], "청산": exit_dates[b],
                         "보유일": b - a, "수익": gross * (1.0 - cost) ** 2 - 1.0, "열림": b == len(s.pos)})


# ── 통계 ────────────────────────────────────────────────────────────────────────────────────────

def sharpe_ex(ex: np.ndarray) -> float:
    """일간 초과수익 → 연율 Sharpe."""
    sd = float(np.std(ex, ddof=1))
    return float(np.mean(ex) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def boot_dsharpe(a_ex: np.ndarray, b_ex: np.ndarray, *, n_boot: int = N_BOOT, block: int = BLOCK,
                 seed: int = SEED, chunk: int = 200) -> dict[str, float]:
    """같은 날짜 블록을 통째로 뽑는 paired 부트스트랩 (evaluation.paired_bootstrap 과 같은 방식, 일간·numpy)."""
    a, b = np.asarray(a_ex, dtype=float), np.asarray(b_ex, dtype=float)
    n = len(a)
    if n != len(b) or n < 2 * block:
        return {}
    rng = np.random.default_rng(seed)
    k = -(-n // block)
    d = np.empty(n_boot)

    def _sr(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)

    for lo in range(0, n_boot, chunk):
        m = min(chunk, n_boot - lo)
        idx = (rng.integers(0, n - block + 1, size=(m, k))[:, :, None] + np.arange(block)).reshape(m, -1)[:, :n]
        d[lo:lo + m] = _sr(a[idx]) - _sr(b[idx])
    return {"dS": sharpe_ex(a) - sharpe_ex(b), "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)), "p": float((d <= 0).mean())}


def placebo_sharpes(sleeves: list[Sleeve], rf: pd.Series, cost: float, *, n_draw: int = N_PLACEBO,
                    seed: int = SEED, shifts: np.ndarray | None = None) -> np.ndarray:
    """무작위 시점 그룹 Sharpe — 모든 칸의 보유 기록을 같은 k 일 밀기 (np.roll). shifts 를 주면 그 k 들로."""
    idx = pd.DatetimeIndex(sorted(set().union(*[set(s.dates) for s in sleeves])))
    where = [(int(idx.get_loc(s.dates[0])), len(s.pos)) for s in sleeves]
    for (m0, n), s in zip(where, sleeves):
        if not idx[m0:m0 + n].equals(s.dates):
            raise ValueError(f"{s.ticker}: 구간이 이어지지 않는다")
    cnt = np.zeros(len(idx))
    for m0, n in where:
        cnt[m0:m0 + n] += 1.0
    rf_u = rf.reindex(idx).fillna(0.0).to_numpy()
    if shifts is None:
        l_min = min(n for _, n in where)
        shifts = np.random.default_rng(seed).integers(PLACEBO_MIN_SHIFT, l_min - PLACEBO_MIN_SHIFT + 1, size=n_draw)
    out = np.empty(len(shifts))
    acc = np.zeros(len(idx))
    for j, k in enumerate(shifts):
        acc[:] = 0.0
        for (m0, n), s in zip(where, sleeves):
            p = np.roll(s.pos, int(k))
            prev = np.concatenate([[0.0], p[:-1]])
            acc[m0:m0 + n] += p * s.r + (1.0 - p) * s.rf - cost * np.abs(p - prev)
        out[j] = sharpe_ex(acc / cnt - rf_u)
    return out


def period_mask(index: pd.DatetimeIndex, a: str, b: str) -> np.ndarray:
    return np.asarray((index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b)))


def stats(ret: pd.Series, rf: pd.Series) -> dict[str, float]:
    """lab_backtest.perf (초과 Sharpe·CAGR·MDD·변동성) + 1만 달러 끝 금액."""
    p = perf(ret, rf)
    p["10k"] = float(10_000 * np.prod(1.0 + ret.to_numpy()))
    return p


# ── 엔진 ────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Universe:
    """시험에 쓰는 모든 종목 가격 · 국채 · SPY."""

    px: dict[str, Px]
    rf: pd.Series
    spy: pd.Series
    index: pd.DatetimeIndex


def load_universe() -> Universe:
    panel = load_panel()
    tickers = sorted({t for g in GROUPS.values() for t in g})
    return Universe({t: stock_px(panel, t) for t in tickers}, load_rf(panel.index), spy_returns(panel), panel.index)


class Engine:
    """(그룹, 종목, 규칙, 설정) 별 상태·칸을 한 번씩만 계산해 둔다."""

    def __init__(self, uni: Universe) -> None:
        self.uni = uni
        self._memo: dict[str, dict[tuple, np.ndarray]] = {t: {} for t in uni.px}
        self._sig: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._rf_s: dict[str, np.ndarray] = {t: uni.rf.reindex(x.dates).fillna(0.0).to_numpy()
                                             for t, x in uni.px.items()}

    def state(self, group: str, ticker: str, code: str, p: tuple[float, ...]) -> np.ndarray:
        key = (ticker, code, p)
        if key not in self._sig:
            self._sig[key] = signals(code, p, self.uni.px[ticker], self._memo[ticker])
        x = self.uni.px[ticker]
        e, xx = self._sig[key]
        return hold_state(e, xx, first_eval(x, group) - 1)

    def sleeves(self, group: str, code: str | None, p: tuple[float, ...] | None, cost: float = COST,
                delay: int = 0) -> list[Sleeve]:
        """그룹 칸들 — code None 이면 그냥 들고 있기."""
        out = []
        for t in GROUPS[group]:
            x = self.uni.px[t]
            f = first_eval(x, group)
            if code is None:
                out.append(hold_sleeve(x, f, self._rf_s[t], cost))
            else:
                out.append(make_sleeve(x, self.state(group, t, code, p), f, self._rf_s[t], cost, delay))
        return out


def counts(engine: Engine) -> pd.DataFrame:
    """가운데 설정의 매매 횟수·보유 비율·보유 기간 (수익은 보지 않는다)."""
    rows = []
    for g in GROUPS:
        for rule in RULES:
            for s in engine.sleeves(g, rule.code, rule.center):
                d = np.diff(np.concatenate([[0.0], s.pos, [0.0]]))
                n_in = int((d == 1).sum())
                holds = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
                yrs = len(s.pos) / TRADING_DAYS
                rows.append({"그룹": GROUP_KO[g], "종목": s.ticker, "규칙": rule.code, "시작": s.dates[0].date(),
                             "년": round(yrs, 2), "매수": n_in, "연간_매수": n_in / yrs, "보유_비율": float(s.pos.mean()),
                             "평균_보유일": float(holds.mean()) if n_in else np.nan})
    return pd.DataFrame(rows)


# ── 모드 ────────────────────────────────────────────────────────────────────────────────────────

def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")


def _registered_sha() -> str | None:
    """등록 줄 메모의 'sha xxxx' (없으면 None)."""
    if not TRIAL_LOG.exists():
        return None
    log = _log()
    rows = log[(log["family"] == FAMILY) & (log["판정"] == "등록")]
    if rows.empty:
        return None
    m = re.search(r"sha ([0-9a-f]{12})", str(rows.iloc[0]["메모"]))
    return m.group(1) if m else ""


def check_control(uni_panel: pd.DataFrame | None = None) -> pd.Series:
    """대조군 목록이 2010 거래대금 상위 10 과 같은지 (다르면 오류)."""
    rank = control_ranking(load_panel() if uni_panel is None else uni_panel)
    top = list(rank.index[:10])
    if sorted(top) != sorted(GROUPS[CONTROL]):
        raise SystemExit(f"대조군 목록이 데이터와 다르다: {top}")
    return rank


def do_count() -> None:
    verify_frozen()
    rank = check_control()
    print("[count] 2010 거래대금 상위 12 ($십억/일):", {k: round(v / 1e9, 2) for k, v in rank.head(12).items()})
    c = counts(Engine(load_universe()))
    c.to_csv(TIER3_DIR / "swing_famous_counts.csv", index=False, encoding="utf-8-sig")
    summ = c.groupby(["그룹", "규칙"], sort=False).agg(연간_매수=("연간_매수", "mean"), 매수_합=("매수", "sum"),
                                                     최소_매수=("매수", "min"), 보유_비율=("보유_비율", "mean"),
                                                     평균_보유일=("평균_보유일", "mean"))
    print(summ.round(2).to_string())
    low = c[c["매수"] < 10]
    print("[count] 매수 10번 미만:", "없음" if low.empty else low[["그룹", "종목", "규칙", "매수"]].to_dict("records"))


def do_register() -> None:
    if _registered_sha() is not None:
        raise SystemExit(f"{FAMILY} 은 이미 등록됐다")
    sha = sha256(SCRIPT)[:12]
    memo = ("10건 p≤0.05/10: 규칙 A1 RSI14<30/>70 · A2 볼린저20,2σ 종가<하단/>상단 · A3 이격(종가/SMA25−1)≤−2×직전252일 "
            "표준편차/≥0 · A4 종가>SMA200·RSI2<10/종가>SMA5 · B1 Supertrend10,3 초록/빨강 전환 × 그룹 성장(TSLA META MSFT "
            "NFLX AAPL AMZN NVDA GOOGL AMD ADBE)·방어(KO PEP PG CL WMT COST JNJ MRK DUK VZ). 끝난 일봉 종가 신호→다음 시가, "
            "종목당 1/10 칸·쉬는 칸 국채, 매수만, 편도 0.1%, 종목별 282번째 봉~2025-09-29. 기준 = 같은 종목 들고 있기. "
            "채택 후보 = ① ΔSharpe 20일 블록 1만 회 하한>0·p≤0.005 ② 1986–2007·2008–15·2016–25.09 모두>0 ③ MDD 2%p 이내 "
            "④ 종목 7/10 ⑤ 무작위 시점(같은 k일 밀기) 2,000회 중 이상 ≤5% ⑥ 이웃 7/8 ⑦ 비용 2배·하루 늦게 >0 "
            f"⑧ 대조(2010 거래대금 상위 10, 2011-01~) >0. 관찰 = ① 하한>0 만. 정의 scripts/swing_famous_research.py (sha {sha})")
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": memo}])
    print(f"등록: {FAMILY} {N_TESTS}건 · sha {sha} · 누적 시도 {total_trials()}")


def _sub_dsharpe(a: pd.Series, b: pd.Series, rf: pd.Series) -> dict[str, float]:
    """세 구간 ΔSharpe (초과수익)."""
    ex_a, ex_b = (a - rf.reindex(a.index)).to_numpy(), (b - rf.reindex(b.index)).to_numpy()
    out = {}
    for name, lo, hi in PERIODS:
        m = period_mask(a.index, lo, hi)
        out[name] = sharpe_ex(ex_a[m]) - sharpe_ex(ex_b[m]) if m.sum() > 2 * BLOCK else float("nan")
    return out


def verdict(g: dict[str, bool], lo: float) -> str:
    if all(g.values()):
        return "채택 후보"
    return "관찰" if lo > 0 else "탈락"


def do_run() -> None:
    sha_reg = _registered_sha()
    if sha_reg is None:
        raise SystemExit("등록 먼저 (--register)")
    if sha_reg != sha256(SCRIPT)[:12]:
        raise SystemExit(f"등록 뒤 스크립트가 바뀌었다 (등록 sha {sha_reg}) — 바꾼 이유를 적고 다시 등록해야 한다")
    log = _log()
    if ((log["family"] == FAMILY) & (log["판정"] != "등록")).any():
        raise SystemExit(f"{FAMILY} 결과가 이미 기록됐다 — 보고서만 다시 쓰려면 --report")
    verify_frozen()
    check_control()
    uni = load_universe()
    eng = Engine(uni)
    rf = uni.rf
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("===== 1/5 이웃 포함 전체 설정", flush=True)

    hold = {g: eng.sleeves(g, None, None) for g in GROUPS}
    hold_ret = {g: group_returns(hold[g]) for g in GROUPS}
    hold_stats = {g: stats(hold_ret[g], rf) for g in GROUPS}
    neighbors, center = [], {}
    for g in GROUPS:
        for rule in RULES:
            for i, j, p in rule.grid():
                sl = eng.sleeves(g, rule.code, p)
                ret = group_returns(sl)
                st = stats(ret, rf)
                col = rule.cols[j]
                col_txt = "/".join(map(str, col)) if isinstance(col, tuple) else col
                neighbors.append({"그룹": GROUP_KO[g], "규칙": rule.code, "행": i, "열": j,
                                  "행값": rule.rows[i], "열값": col_txt,
                                  "가운데": (i, j) == (1, 1), "Sharpe": st["Sharpe"],
                                  "들고있기_Sharpe": hold_stats[g]["Sharpe"], "dS": st["Sharpe"] - hold_stats[g]["Sharpe"],
                                  "CAGR": st["CAGR"], "들고있기_CAGR": hold_stats[g]["CAGR"], "MDD": st["MDD"],
                                  "보유_비율": group_exposure(sl)})
                if (i, j) == (1, 1):
                    center[(g, rule.code)] = (sl, ret, st)
        print(f"  {GROUP_KO[g]} 끝", flush=True)
    nb = pd.DataFrame(neighbors)

    print("===== 2/5 판정 (부트스트랩 · 구간 · 종목 · 스트레스 · 대조)", flush=True)
    hold_stress = {g: group_returns(eng.sleeves(g, None, None, cost=COST_STRESS)) for g in TEST_GROUPS}
    hold_ref = {g: group_returns(eng.sleeves(g, None, None, cost=COST_REF)) for g in TEST_GROUPS}
    spy = uni.spy
    tests, stock_rows, period_rows, trades_all, placebo_cols = [], [], [], [], {}
    for g in GROUPS:
        for rule in RULES:
            sl, ret, st = center[(g, rule.code)]
            hs = hold_stats[g]
            hr = hold_ret[g]
            if not ret.index.equals(hr.index):
                raise ValueError(f"{g}-{rule.code}: 규칙과 들고 있기 날짜가 다르다")
            # 종목별
            wins = 0
            for s, h in zip(sl, hold[g]):
                a, b = stats(s.series(), rf), stats(h.series(), rf)
                tr = trade_list(s, COST)
                closed = tr[~tr["열림"]]
                wins += int(a["Sharpe"] > b["Sharpe"])
                yrs = len(s.pos) / TRADING_DAYS
                stock_rows.append({"그룹": GROUP_KO[g], "종목": s.ticker, "규칙": rule.code,
                                   "시작": s.dates[0].date(), "년": round(yrs, 2),
                                   "Sharpe": a["Sharpe"], "들고있기_Sharpe": b["Sharpe"],
                                   "CAGR": a["CAGR"], "들고있기_CAGR": b["CAGR"], "MDD": a["MDD"], "들고있기_MDD": b["MDD"],
                                   "1만달러": a["10k"], "들고있기_1만달러": b["10k"], "매수": len(tr),
                                   "연간_매수": len(tr) / yrs, "승률": float((closed["수익"] > 0).mean()) if len(closed) else np.nan,
                                   "평균_거래수익": float(closed["수익"].mean()) if len(closed) else np.nan,
                                   "평균_보유일": float(tr["보유일"].mean()) if len(tr) else np.nan,
                                   "최악_거래": float(closed["수익"].min()) if len(closed) else np.nan,
                                   "최고_거래": float(closed["수익"].max()) if len(closed) else np.nan,
                                   "보유_비율": float(s.pos.mean()), "Sharpe_승": a["Sharpe"] > b["Sharpe"]})
                trades_all.append(tr.assign(그룹=GROUP_KO[g], 규칙=rule.code))
            # 구간별 (참고 표 — 대조군 포함)
            for name, lo, hi in PERIODS:
                m = period_mask(ret.index, lo, hi)
                if m.sum() <= 2 * BLOCK:
                    continue
                a, b = stats(ret[m], rf), stats(hr[m], rf)
                period_rows.append({"그룹": GROUP_KO[g], "규칙": rule.code, "구간": name, "Sharpe": a["Sharpe"],
                                    "들고있기_Sharpe": b["Sharpe"], "dS": a["Sharpe"] - b["Sharpe"], "CAGR": a["CAGR"],
                                    "들고있기_CAGR": b["CAGR"], "MDD": a["MDD"], "들고있기_MDD": b["MDD"]})
            if g == CONTROL:
                continue
            ex_a = (ret - rf.reindex(ret.index)).to_numpy()
            ex_b = (hr - rf.reindex(hr.index)).to_numpy()
            bt = boot_dsharpe(ex_a, ex_b)
            sub = _sub_dsharpe(ret, hr, rf)
            ret_x2 = group_returns(eng.sleeves(g, rule.code, rule.center, cost=COST_STRESS))
            ret_lag = group_returns(eng.sleeves(g, rule.code, rule.center, delay=1))
            ret_ref = group_returns(eng.sleeves(g, rule.code, rule.center, cost=COST_REF))
            ds_x2 = stats(ret_x2, rf)["Sharpe"] - stats(hold_stress[g], rf)["Sharpe"]
            ds_lag = stats(ret_lag, rf)["Sharpe"] - hs["Sharpe"]
            ds_ref = stats(ret_ref, rf)["Sharpe"] - stats(hold_ref[g], rf)["Sharpe"]
            ctrl = center[(CONTROL, rule.code)][2]["Sharpe"] - hold_stats[CONTROL]["Sharpe"]
            plc = placebo_sharpes(sl, rf, COST)
            placebo_cols[f"{g}-{rule.code}"] = plc
            p_plc = float((plc >= st["Sharpe"]).mean())
            nbr = nb[(nb["그룹"] == GROUP_KO[g]) & (nb["규칙"] == rule.code) & ~nb["가운데"]]
            n_pos = int((nbr["dS"] > 0).sum())
            sp = spy.reindex(ret.index).dropna()
            sp = sp[sp.index >= SPY_START]
            gate = {"①": bt["lo"] > 0 and bt["p"] <= P_MAX,
                    "②": all(v > 0 for v in sub.values()),
                    "③": st["MDD"] >= hs["MDD"] - MDD_TOL,
                    "④": wins >= MIN_STOCK_WINS,
                    "⑤": p_plc <= PLACEBO_ALPHA,
                    "⑥": n_pos >= MIN_NEIGHBORS,
                    "⑦": ds_x2 > 0 and ds_lag > 0,
                    "⑧": ctrl > 0}
            vd = verdict(gate, bt["lo"])
            tests.append({"test_id": f"{g}-{rule.code}", "그룹": GROUP_KO[g], "규칙": rule.code, "이름": rule.name,
                          "Sharpe": st["Sharpe"], "들고있기_Sharpe": hs["Sharpe"], "dS": bt["dS"], "lo": bt["lo"],
                          "hi": bt["hi"], "p": bt["p"], "CAGR": st["CAGR"], "들고있기_CAGR": hs["CAGR"],
                          "MDD": st["MDD"], "들고있기_MDD": hs["MDD"], "1만달러": st["10k"], "들고있기_1만달러": hs["10k"],
                          "보유_비율": group_exposure(sl),
                          **{f"dS_{k}": v for k, v in sub.items()},
                          "종목_승": wins, "무작위_p": p_plc, "무작위_중앙": float(np.median(plc)),
                          "무작위_95%": float(np.quantile(plc, 0.95)), "이웃_양수": n_pos,
                          "dS_비용2배": ds_x2, "dS_하루늦게": ds_lag, "dS_비용0.05%": ds_ref, "dS_대조": ctrl,
                          "SPY_Sharpe_1993": stats(sp, rf)["Sharpe"] if len(sp) else np.nan,
                          "규칙_Sharpe_1993": stats(ret.loc[sp.index], rf)["Sharpe"] if len(sp) else np.nan,
                          "SPY_CAGR_1993": stats(sp, rf)["CAGR"] if len(sp) else np.nan,
                          "규칙_CAGR_1993": stats(ret.loc[sp.index], rf)["CAGR"] if len(sp) else np.nan,
                          **{f"gate{k}": v for k, v in gate.items()}, "판정": vd,
                          "위험대비만": bool(bt["dS"] > 0 and st["CAGR"] < hs["CAGR"])})
            print(f"  {g}-{rule.code}: Sharpe {st['Sharpe']:.2f} vs {hs['Sharpe']:.2f} · ΔSharpe {bt['dS']:+.2f} "
                  f"[{bt['lo']:+.2f}, {bt['hi']:+.2f}] p {bt['p']:.4f} · 무작위 p {p_plc:.3f} → {vd}", flush=True)

    print("===== 3/5 대조군 표", flush=True)
    control = []
    for rule in RULES:
        sl, ret, st = center[(CONTROL, rule.code)]
        hs = hold_stats[CONTROL]
        control.append({"규칙": rule.code, "Sharpe": st["Sharpe"], "들고있기_Sharpe": hs["Sharpe"],
                        "dS": st["Sharpe"] - hs["Sharpe"], "CAGR": st["CAGR"], "들고있기_CAGR": hs["CAGR"],
                        "MDD": st["MDD"], "들고있기_MDD": hs["MDD"], "보유_비율": group_exposure(sl)})

    print("===== 4/5 저장", flush=True)
    res = pd.DataFrame(tests)
    res.to_csv(TIER3_DIR / "swing_famous_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stock_rows).to_csv(TIER3_DIR / "swing_famous_stocks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(period_rows).to_csv(TIER3_DIR / "swing_famous_periods.csv", index=False, encoding="utf-8-sig")
    nb.to_csv(TIER3_DIR / "swing_famous_neighbors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(control).to_csv(TIER3_DIR / "swing_famous_control.csv", index=False, encoding="utf-8-sig")
    yearly = []
    for g in GROUPS:
        cols = {"들고있기": hold_ret[g], **{r.code: center[(g, r.code)][1] for r in RULES}, "SPY": spy}
        df = pd.concat(cols, axis=1, sort=True).loc[hold_ret[g].index]
        y = (1.0 + df).groupby(df.index.year).prod() - 1.0
        y.loc[:, "SPY"] = y["SPY"].where(df["SPY"].notna().groupby(df.index.year).mean() > 0.9)
        yearly.append(y.assign(그룹=GROUP_KO[g]).rename_axis("연도").reset_index())
    pd.concat(yearly).to_csv(TIER3_DIR / "swing_famous_yearly.csv", index=False, encoding="utf-8-sig")
    daily = pd.concat({(g, k): v for g in GROUPS for k, v in
                       {"들고있기": hold_ret[g], **{r.code: center[(g, r.code)][1] for r in RULES}}.items()}, axis=1,
                      sort=True)
    daily[("SPY", "SPY")] = spy.reindex(daily.index)
    daily.columns = [f"{a}|{b}" for a, b in daily.columns]
    daily.to_parquet(OUT_DIR / "daily.parquet")
    pos = pd.concat({f"{g}|{s.ticker}|{r.code}": pd.Series(s.pos, index=s.dates)
                     for g in GROUPS for r in RULES for s in center[(g, r.code)][0]}, axis=1, sort=True)
    pos.to_parquet(OUT_DIR / "positions.parquet")
    pd.concat(trades_all, ignore_index=True).to_parquet(OUT_DIR / "trades.parquet")
    pd.DataFrame(placebo_cols).to_parquet(OUT_DIR / "placebo.parquet")

    print("===== 5/5 TRIAL_LOG", flush=True)
    rows = []
    for r in tests:
        g_txt = "".join(k for k in GATES if r[f"gate{k}"]) or "없음"
        rows.append({"test_id": r["test_id"], "t": f"{r['dS']:.3f}", "판정": r["판정"],
                     "메모": (f"Sharpe {r['Sharpe']:.2f} vs 들고있기 {r['들고있기_Sharpe']:.2f} · ΔSharpe {r['dS']:+.2f} "
                            f"[{r['lo']:+.2f}, {r['hi']:+.2f}] p {r['p']:.4f} · CAGR {r['CAGR']:.1%} vs "
                            f"{r['들고있기_CAGR']:.1%} · 구간 " + "/".join(f"{r['dS_' + n]:+.2f}" for n, _, _ in PERIODS)
                            + f" · 종목 {r['종목_승']}/10 · 무작위 p {r['무작위_p']:.3f} · 이웃 {r['이웃_양수']}/8 · "
                            f"비용2배 {r['dS_비용2배']:+.2f} · 하루늦게 {r['dS_하루늦게']:+.2f} · 대조 {r['dS_대조']:+.2f} · "
                            f"통과 관문 {g_txt}")})
    log_trials(STEP, FAMILY, N_TESTS, rows)
    done = ", ".join(f"{r['test_id']} {r['판정']}" for r in tests)
    print(f"[swing_famous] {done} · 누적 시도 {total_trials()}", flush=True)


# ── 보고서 (--run 결과를 본 뒤 붙였다 — 판정 계산은 바꾸지 않는다) ─────────────────────────────────

REPORT = TIER3_DIR / "SWING_FAMOUS_REPORT.md"
PNG_EQUITY = TIER3_DIR / "swing_famous_equity.png"
PNG_HEAT = TIER3_DIR / "swing_famous_heatmap.png"
EXAMPLES: tuple[tuple[str, str], ...] = (("G", "TSLA"), ("D", "KO"))
EXAMPLE_WINDOW = ("2019-10-01", "2025-09-30")   # 예시 = 가장 최근 6년 (골라 보이지 않도록 기간을 고정)
RUN_SHA = "f22580af3549"   # --run 을 돌린 스크립트 sha (등록 sha 와 같다). 보고서 코드·concat sort=True 는 그 뒤에 붙였다
BIG_YEARS = (1987, 2000, 2002, 2008, 2009, 2013, 2020, 2022, 2023)

# 그림 색 — dataviz 기본 팔레트 (밝은 화면). 규칙 5개 = 범주 1~5칸 순서 (validate_palette.js: 인접 CVD ΔE ≥ 9.1 통과,
# 밝은 3색은 대비 3:1 미만이라 선 끝 이름표 + 보고서 표를 같이 둔다). 발산 = 빨강 · 회색 · 파랑.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"A1": "#2a78d6", "A2": "#eb6834", "A3": "#1baf7a", "A4": "#eda100", "B1": "#e87ba4"}
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")
FONT = "Apple SD Gothic Neo"

# 2026-10-22 실행 결과를 보고 쓴 쉬운 말 요약 (--report 가 그대로 넣는다)
SUMMARY: list[str] = [
    "1. **10건 모두 탈락.** 저점에 사고 고점에 파는 규칙 4개(RSI(14)·볼린저·BNF 이격도·RSI(2))와 반등을 확인한 뒤 사는 "
    "규칙 1개(Supertrend)를 유명 성장주 10개·방어주 10개에 40년(1986~2025) 일봉으로 돌렸다. 어느 것도 같은 종목을 그냥 "
    "들고 있는 것보다 낫지 않았다. 10건 중 9건은 위험 대비 수익(Sharpe)부터 낮았다.",
    "2. **승률은 높았는데 졌다.** 방어주 RSI(14) 규칙은 거래의 85%, 성장주는 74%가 이익이었다. 그런데 큰 상승장에 빠져 있었다. "
    "성장주를 그냥 들고 있으면 2009년 +142%, 2020년 +105%, 2023년 +102%였고 RSI(14) 규칙은 +36%, +18%, +38%였다. "
    "2008년 폭락을 반쯤 피한 것(−48% → −24%)보다 놓친 상승이 훨씬 컸다.",
    "3. **규칙은 '안 오른 종목'에서만 이겼다.** 150개 조합(종목 × 규칙) 중 1만 달러가 들고 있기보다 많이 남은 건 7개뿐이다: "
    "VZ·INTC(RSI·볼린저), XOM(RSI)처럼 제자리였던 종목과, 오래 폭락했던 AMD(2000~2015년 −97%)·AAPL(2000~2003년 −82%)의 "
    "Supertrend. 크게 오른 종목에서는 "
    "크게 졌다: TSLA 1만 달러가 들고 있기 260만 달러 vs RSI(14) 규칙 11만 달러, NVDA 1,608만 달러 vs 2.3만 달러. "
    "장기 수익은 크게 오른 몇 종목이 거의 다 만든다.",
    "4. **가장 가까웠던 건 성장주 Supertrend.** Sharpe 0.87 vs 0.82, 최대 낙폭 −36% vs −65%로 위험은 줄였지만 연수익은 "
    "20.9% vs 29.5%였다. 차이(+0.06)는 운과 구별되지 않고(p 0.30), 같은 규칙이 방어주(0.25 vs 0.72)와 2010년 유명주(−0.19)"
    "에서는 졌다. 추세 따라가기는 낙폭을 반으로 줄이는 대신 수익도 줄인다 (SPY 로 본 lab_v1 A 와 같은 결론).",
    "5. **RSI(2) 눌림목에는 진짜 신호가 있다 — 너무 작을 뿐.** 매매 횟수·보유 기간이 같은 무작위 매매 2,000번 중 규칙보다 나은 게 "
    "성장 1번, 방어 0번이었다. 하지만 거래당 평균 +0.8%(성장)·+0.3%(방어), 4일 보유, 시장에 있는 시간 11~12%라 연수익이 "
    "9.2%·4.8%로 들고 있기(29.5%·14.2%)의 3분의 1이었다. 비용을 편도 0.05%로 낮춰도 진다.",
    "6. **결과를 모르고 고른 목록에서도 같다.** 2010년 한 해 거래대금 상위 10개(AAPL·BAC·C·GOOGL·XOM·MSFT·GS·JPM·INTC·CSCO)를 "
    "2011~2025에 시험해도 5개 규칙 모두 들고 있기보다 낮았다 (ΔSharpe −0.17 ~ −0.50).",
    "7. **'유명 성장주 들고 있기 연 29.5%'는 미래를 알고 고른 숫자다.** 테슬라·엔비디아가 크게 오른 걸 알고 고른 목록이라서다. "
    "2010년에 그때 유명했던 10개를 골랐다면 연 16.6%였다 (그것도 AAPL·MSFT·GOOGL 덕분). 그래서 이 시험은 SPY 가 아니라 "
    "'같은 종목 들고 있기'와 비교했다.",
]


def _money(x: float) -> str:
    """달러 줄임 — $283.5M · $658K · $9,500."""
    if pd.isna(x):
        return "—"
    if abs(x) >= 1e9:
        return f"${x / 1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e4:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"


def _pct(x: float, sign: bool = False, d: int = 1) -> str:
    """퍼센트 — 천 단위 쉼표, 반올림해 0 이면 부호 없는 0."""
    if pd.isna(x):
        return "—"
    v = round(float(x) * 100, d) + 0.0
    if v == 0:
        return f"{0:.{d}f}%"
    return (format(v, f"+,.{d}f") if sign else format(v, f",.{d}f")).replace("-", "−") + "%"


def _num(x: float, fmt: str = "+.2f") -> str:
    """숫자 — 반올림해 0 이면 부호 없는 0 (−0 을 피한다)."""
    if pd.isna(x):
        return "—"
    out = format(x, fmt)
    if float(out) == 0:
        out = format(0.0, fmt.lstrip("+"))
    return out.replace("-", "−")


def _chart_num(x: float, fmt: str = "+.2f") -> str:
    """그림 글자용 — Apple SD Gothic Neo 에 U+2212 가 없어 ASCII 빼기표를 쓴다."""
    return "—" if pd.isna(x) else format(x, fmt)


def _tick_money(v: float, _: Any = None) -> str:
    """로그 축 눈금 — $10K · $1M · $100M (소수점 없이)."""
    for unit, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= unit:
            q = v / unit
            return f"${q:.0f}{suf}" if abs(q - round(q)) < 1e-9 else f"${q:.1f}{suf}"
    return f"${v:.0f}"


def _mpl() -> Any:
    """Agg 백엔드 + 한글 글꼴 + 차분한 축 (dataviz: 가는 선 · 실선 가는 눈금선 · 글자는 잉크색)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False, "font.size": 9, "text.parse_math": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.titlecolor": INK})
    return plt


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _end_labels(ax: Any, ends: list[tuple[pd.Timestamp, float, str]], gap: float = 0.06) -> None:
    """선 끝 이름표 — 로그 축 위치로 정렬해 겹치면 띄우고, 가는 연결선으로 선 끝과 잇는다."""
    lo, hi = (np.log10(v) for v in ax.get_ylim())
    pts = sorted(((np.log10(v) - lo) / (hi - lo), x, v, txt) for x, v, txt in ends)
    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - 0.98
    if over > 0:
        ys = [y - over for y in ys]
    for y, (_, x, v, txt) in zip(ys, pts):
        ax.annotate(txt, xy=(x, v), xycoords="data", xytext=(1.03, y), textcoords="axes fraction", va="center",
                    ha="left", fontsize=8, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.7, "shrinkA": 0, "shrinkB": 3})


def plot_equity(daily: pd.DataFrame) -> None:
    """성장·방어 그룹 1만 달러 곡선 (로그 축): 그냥 들고 있기 vs 규칙 5개."""
    plt = _mpl()
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    names = {"들고있기": "그냥 들고 있기", **{r.code: f"{r.code} {r.name}" for r in RULES}}
    for ax, g in zip(axes, TEST_GROUPS):
        ends = []
        for key in ["들고있기"] + [r.code for r in RULES]:
            eq = 10_000 * (1.0 + daily[f"{g}|{key}"].dropna()).cumprod()
            hold = key == "들고있기"
            color = INK if hold else SERIES[key]
            ax.plot(eq.index, eq.to_numpy(), color=color, lw=1.9 if hold else 1.2, label=names[key],
                    solid_capstyle="round", solid_joinstyle="round", zorder=3 if hold else 2)
            ax.scatter([eq.index[-1]], [eq.iloc[-1]], s=26, color=color, edgecolor=SURFACE, linewidth=1.4, zorder=4)
            ends.append((eq.index[-1], float(eq.iloc[-1]), f"{'들고 있기' if hold else key} {_money(eq.iloc[-1])}"))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_tick_money))
        _style(ax)
        _end_labels(ax, ends)
        ax.set_title(f"{GROUP_KO[g]} 10종목 — 1만 달러가 어떻게 됐나 (로그 눈금)", loc="left", fontsize=11)
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
    fig.text(0.01, 0.005, "그룹 = 그날 시험 구간 안 종목 칸들의 평균 (매일 같은 비중). 규칙은 쉬는 칸에 국채 이자, 편도 0.1% 비용. "
             "들고 있기 숫자가 큰 건 크게 오른 종목을 알고 고른 목록이기 때문 (보고서 요약 7).", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.06, right=0.9, wspace=0.42, bottom=0.1, top=0.9)
    fig.savefig(PNG_EQUITY, dpi=150)
    plt.close(fig)


def plot_heatmap(nb: pd.DataFrame) -> None:
    """이웃 3×3 ΔSharpe (규칙 − 들고 있기) — 성장·방어 × 규칙 5개, 굵은 테두리 = 본 시험 설정."""
    plt = _mpl()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.8), layout="constrained")
    for gi, g in enumerate(TEST_GROUPS):
        for ri, rule in enumerate(RULES):
            ax = axes[gi, ri]
            sub = nb[(nb["그룹"] == GROUP_KO[g]) & (nb["규칙"] == rule.code)]
            m = np.full((3, 3), np.nan)
            for _, row in sub.iterrows():
                m[int(row["행"]), int(row["열"])] = row["dS"]
            for i in range(3):
                for j in range(3):
                    face = cmap(norm(m[i, j]))
                    r_, g_, b_ = to_rgb(face)
                    ink = "#ffffff" if 0.299 * r_ + 0.587 * g_ + 0.114 * b_ < 0.55 else INK
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor=SURFACE, linewidth=2.0))
                    ax.text(j + 0.5, i + 0.5, _chart_num(m[i, j]), ha="center", va="center", fontsize=9, color=ink)
            ax.add_patch(Rectangle((1.03, 1.03), 0.94, 0.94, fill=False, edgecolor=INK, linewidth=1.6))
            ax.set_xlim(0, 3)
            ax.set_ylim(3, 0)
            cols = ["/".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in rule.cols]
            ax.set_xticks([0.5, 1.5, 2.5], cols)
            ax.set_yticks([0.5, 1.5, 2.5], [str(v) for v in rule.rows])
            ax.set_xlabel(rule.col_label, fontsize=8)
            ax.set_ylabel(rule.row_label, fontsize=8)
            ax.set_title(f"{GROUP_KO[g]} · {rule.code} {rule.name}", loc="left", fontsize=9.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("ΔSharpe = 규칙 Sharpe - 그냥 들고 있기 Sharpe", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("이웃 설정 9칸 — 파랑 = 들고 있기보다 나음 · 빨강 = 못함 · 굵은 테두리 = 본 시험 설정 (통과 기준: 나머지 8칸 중 7칸 파랑)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(PNG_HEAT, dpi=150)
    plt.close(fig)


def example_frame(group: str, ticker: str, panel: pd.DataFrame, pos: pd.DataFrame, rf: pd.Series
                  ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """예시 구간의 (가격 표, 규칙별 매수·매도 날짜와 구간 수익). 수익은 저장된 보유 기록으로 다시 계산 (표시용)."""
    a, b = EXAMPLE_WINDOW
    px = panel.xs(ticker, axis=1, level="Ticker")[["Open", "Close", "Adj Close"]].astype("float64")
    otr = px["Open"] * px["Adj Close"] / px["Close"]
    r = (otr.shift(-1) / otr - 1.0)
    out: dict[str, dict[str, Any]] = {}
    for rule in RULES:
        p = pos[f"{group}|{ticker}|{rule.code}"].dropna()
        prev_all = p.shift(1).fillna(0.0)
        pw, prev = p.loc[a:b], prev_all.loc[a:b]
        rr, rfw = r.reindex(pw.index), rf.reindex(pw.index).fillna(0.0)
        ret = pw * rr + (1.0 - pw) * rfw - COST * (pw - prev).abs()
        out[rule.code] = {"pos": pw, "buys": pw.index[(pw == 1) & (prev == 0)], "sells": pw.index[(pw == 0) & (prev == 1)],
                          "ret": float((1.0 + ret).prod() - 1.0), "hold": float((1.0 + rr).prod() - 1.0)}
    return px.loc[a:b], out


def plot_example(group: str, ticker: str, px: pd.DataFrame, ex: dict[str, dict[str, Any]],
                 full: pd.DataFrame) -> Path:
    """한 종목 일봉 (로그 축) + 규칙별 보유 구간(옅은 색) · 매수 ▲ · 매도 ▼ (둘 다 신호 다음 날 시가).

    full = 그 종목의 종목별 표 줄들 (규칙별 전체 구간 1만 달러) — 예시 구간만 보고 오해하지 않게 제목에 같이 적는다.
    """
    plt = _mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    path = TIER3_DIR / f"swing_famous_example_{ticker}.png"
    f = full.set_index("규칙")
    fig, axes = plt.subplots(len(RULES), 1, figsize=(12.5, 13.5), sharex=True)
    for ax, rule in zip(axes, RULES):
        e = ex[rule.code]
        color = SERIES[rule.code]
        ax.plot(px.index, px["Close"], color=INK2, lw=1.0, zorder=2)
        held = e["pos"].to_numpy() == 1
        idx = e["pos"].index
        k = 0
        while k < len(held):
            if held[k]:
                j = k
                while j + 1 < len(held) and held[j + 1]:
                    j += 1
                ax.axvspan(idx[k], idx[min(j + 1, len(idx) - 1)], color=color, alpha=0.14, lw=0, zorder=1)
                k = j + 1
            else:
                k += 1
        ax.scatter(e["buys"], px["Open"].reindex(e["buys"]), marker="^", s=46, color=color, edgecolor=SURFACE,
                   linewidth=1.2, zorder=4)
        ax.scatter(e["sells"], px["Open"].reindex(e["sells"]), marker="v", s=46, color=INK, edgecolor=SURFACE,
                   linewidth=1.2, zorder=4)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        row = f.loc[rule.code]
        ax.set_title(f"{rule.code} {rule.name} — 이 6년: 규칙 {_chart_num(e['ret'] * 100, '+.0f')}% · 들고 있기 "
                     f"{_chart_num(e['hold'] * 100, '+.0f')}% · 매수 {len(e['buys'])}번   |   전체 {str(row['시작'])[:7]}~: "
                     f"규칙 {_money(row['1만달러'])} · 들고 있기 {_money(row['들고있기_1만달러'])} (1만 달러 시작)",
                     loc="left", fontsize=9.5)
        _style(ax)
    handles = [Line2D([], [], color=INK2, lw=1.0, label="종가 (분할만 반영, 로그 눈금)"),
               Patch(color=MUTED, alpha=0.25, lw=0, label="규칙대로 들고 있던 날"),
               Line2D([], [], marker="^", ls="", color=MUTED, markeredgecolor=SURFACE, markersize=8, label="매수"),
               Line2D([], [], marker="v", ls="", color=INK, markeredgecolor=SURFACE, markersize=8, label="매도")]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.985, 0.985))
    fig.suptitle(f"{ticker} 일봉 {EXAMPLE_WINDOW[0][:7]} ~ {EXAMPLE_WINDOW[1][:7]} (가장 최근 6년) — 예시일 뿐, 판정은 보고서 표",
                 x=0.01, y=0.985, ha="left", fontsize=12, color=INK)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.04, hspace=0.45)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + \
        ["| " + " | ".join(r) + " |" for r in rows]


def write_report() -> None:
    """저장된 표 → SWING_FAMOUS_REPORT.md + 그림 4장 (TRIAL_LOG 는 건드리지 않는다)."""
    rd = lambda name: pd.read_csv(TIER3_DIR / f"swing_famous_{name}.csv")  # noqa: E731
    t, s, per, nb, ctl, yr = rd("tests"), rd("stocks"), rd("periods"), rd("neighbors"), rd("control"), rd("yearly")
    daily = pd.read_parquet(OUT_DIR / "daily.parquet")
    trades = pd.read_parquet(OUT_DIR / "trades.parquet")
    pos = pd.read_parquet(OUT_DIR / "positions.parquet")
    plot_equity(daily)
    plot_heatmap(nb)
    panel = load_panel()
    rf = load_rf(panel.index)
    examples = {}
    for g, tk in EXAMPLES:
        px, ex = example_frame(g, tk, panel, pos, rf)
        plot_example(g, tk, px, ex, s[(s["그룹"] == GROUP_KO[g]) & (s["종목"] == tk)])
        examples[tk] = ex
    name = {r.code: r.name for r in RULES}
    L: list[str] = [
        "# 유명 종목 일봉 스윙 (swing_famous_v1) — 1986-02 ~ 2025-09", "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}`, {N_TESTS}건 (규칙 5 × 성장·방어) → 채택 기준 p ≤ {P_MAX:.3f} · "
        f"누적 시도 {total_trials():,}건",
        "- 질문: 유명 종목을 일봉으로 저점에 사고 고점에 파는 걸 반복하면, **같은 종목을 그냥 들고 있는 것**보다 나은가",
        "- 공통: 끝난 일봉 종가 신호 → 다음 날 시가 체결 · 종목당 계좌 1/10 칸 · 쉬는 칸 = 1개월 국채 · 매수만 · 편도 0.1%",
        f"- 실행 sha `{RUN_SHA}` (등록과 같음) · 정의·사전 등록 전문은 `scripts/swing_famous_research.py` 머리말 · "
        f"보고서 작성 {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        f"- **결과: {len(t)}건 모두 {t['판정'].iloc[0]}**" if t["판정"].nunique() == 1 else
        f"- **결과: " + " · ".join(f"{k} {v}건" for k, v in t["판정"].value_counts().items()) + "**",
        "", "## 쉬운 말 요약", "", *SUMMARY, "",
        "## 한눈에 보기", "",
        "Sharpe = 1개월 국채를 뺀 초과수익 기준 (일간 × √252). ΔSharpe 구간 = 20일 블록 부트스트랩 1만 회 95%.", "",
    ]
    rows = []
    for _, r in t.iterrows():
        rows.append([r["test_id"], f"{r['규칙']} {r['이름']}", f"{r['Sharpe']:.2f} ({r['들고있기_Sharpe']:.2f})",
                     f"{_num(r['dS'])} [{_num(r['lo'])}, {_num(r['hi'])}]", f"{r['p']:.3f}",
                     f"{_pct(r['CAGR'])} ({_pct(r['들고있기_CAGR'])})", f"{_pct(r['MDD'], d=0)} ({_pct(r['들고있기_MDD'], d=0)})",
                     _pct(r["보유_비율"], d=0), f"{_money(r['1만달러'])} ({_money(r['들고있기_1만달러'])})",
                     f"**{r['판정']}**"])
    L += _md_table(["ID", "규칙", "Sharpe (들고 있기)", "ΔSharpe [95%]", "p", "연수익 (들고 있기)", "최대 낙폭 (들고 있기)",
                    "시장에 있던 비율", "1만 달러 → (들고 있기)", "판정"], rows)
    L += ["", "- 그룹 '들고 있기'는 10종목을 매일 같은 비중으로 맞춘 값이다. 1만 달러 숫자가 큰 건 크게 오른 종목을 알고 고른 "
          "목록이라서다 (요약 7). 한 종목씩 사서 그대로 둔 값은 아래 종목별 표.", "",
          "## 관문별 결과 (✓ 통과 · ✗ 못 넘음)", "",
          "① 95% 하한 > 0 · p ≤ 0.005 ② 세 구간 모두 ΔSharpe > 0 (1986–2007 / 2008–2015 / 2016–2025.09) ③ 최대 낙폭 2%p 이내 "
          "④ 종목 7/10 이상 ⑤ 무작위 매매 2,000번 중 규칙 이상 ≤ 5% ⑥ 이웃 8칸 중 7칸 ⑦ 비용 2배 · 하루 늦게 ⑧ 대조군 "
          "(2010년 유명주, 2011~)", ""]
    ok = lambda b: "✓" if bool(b) else "✗"  # noqa: E731
    rows = []
    for _, r in t.iterrows():
        rows.append([r["test_id"], f"{ok(r['gate①'])} p {r['p']:.3f}",
                     f"{ok(r['gate②'])} " + " / ".join(_num(r[f'dS_{n}']) for n, _, _ in PERIODS),
                     f"{ok(r['gate③'])} {_pct(r['MDD'], d=0)} vs {_pct(r['들고있기_MDD'], d=0)}",
                     f"{ok(r['gate④'])} {int(r['종목_승'])}/10", f"{ok(r['gate⑤'])} {r['무작위_p']:.1%}",
                     f"{ok(r['gate⑥'])} {int(r['이웃_양수'])}/8",
                     f"{ok(r['gate⑦'])} {_num(r['dS_비용2배'])} / {_num(r['dS_하루늦게'])}",
                     f"{ok(r['gate⑧'])} {_num(r['dS_대조'])}"])
    L += _md_table(["ID", "① 유의", "② 세 구간", "③ 낙폭", "④ 종목", "⑤ 무작위", "⑥ 이웃", "⑦ 비용 2배 / 하루 늦게",
                    "⑧ 대조군"], rows)
    L += ["", "- ⑤ 는 '매매 횟수·보유 기간·종목끼리 겹치는 모양은 그대로 두고 날짜만 옮긴 가짜 매매' 2,000번 중 규칙보다 "
          "Sharpe 가 높거나 같은 비율이다. 낮을수록 '언제 사고 파는지'에 정보가 있다는 뜻이다. A4 RSI(2) 는 성장·방어 모두 "
          "0.1% 이하로 진짜 신호가 있었지만, ① 들고 있기와의 비교에서 졌다.", "",
          "## 왜 졌나 ① — 승률은 높은데 큰 상승을 놓쳤다", "",
          "거래 통계 (끝난 거래만, 수익 = 진입 시가 → 청산 시가, 배당 포함, 비용 양쪽):", ""]
    closed = trades[~trades["열림"]]
    rows = []
    for g in TEST_GROUPS:
        for rule in RULES:
            c = closed[(closed["그룹"] == GROUP_KO[g]) & (closed["규칙"] == rule.code)]["수익"]
            dd = closed[(closed["그룹"] == GROUP_KO[g]) & (closed["규칙"] == rule.code)]["보유일"]
            rows.append([GROUP_KO[g], f"{rule.code} {rule.name}", f"{len(c):,}", _pct((c > 0).mean(), d=0),
                         _pct(c.mean(), True), _pct(c.median(), True), f"{dd.mean():.0f}일", _pct(c.min(), True, 0)])
    L += _md_table(["그룹", "규칙", "거래", "승률", "평균 거래 수익", "중앙값", "평균 보유", "최악 거래"], rows)
    L += ["", "큰 해만 골라 본 그룹 연수익 (%, 그해 1월 첫 거래일 ~ 12월 마지막 거래일 시가 기준):", ""]
    for g in TEST_GROUPS:
        y = yr[yr["그룹"] == GROUP_KO[g]].set_index("연도")
        rows = [[str(yy)] + [_num(y.loc[yy, c] * 100, "+.0f") for c in ["들고있기"] + [r.code for r in RULES] + ["SPY"]]
                for yy in BIG_YEARS if yy in y.index]
        L += [f"**{GROUP_KO[g]}**", ""] + _md_table(["연도", "들고 있기"] + [r.code for r in RULES] + ["SPY"], rows) + [""]
    L += ["## 왜 졌나 ② — 안 오른 종목에서만 이겼다", "",
          "종목별 1만 달러 → (각 종목 시험 첫날 시가에 사서 2025-09-30 시가까지. **굵게** = 들고 있기보다 많이 남음):", ""]
    for g in GROUPS:
        x = s[s["그룹"] == GROUP_KO[g]]
        rows = []
        for tk in GROUPS[g]:
            xx = x[x["종목"] == tk].set_index("규칙")
            hold = xx["들고있기_1만달러"].iloc[0]
            cells = [f"**{_money(xx.loc[c, '1만달러'])}**" if xx.loc[c, "1만달러"] > hold else _money(xx.loc[c, "1만달러"])
                     for c in [r.code for r in RULES]]
            rows.append([tk, str(xx["시작"].iloc[0]), _money(hold)] + cells)
        wins = x.groupby("규칙")["Sharpe_승"].sum()
        rows.append(["Sharpe 이긴 종목", "", ""] + [f"{int(wins.get(r.code, 0))}/10" for r in RULES])
        title = f"**{GROUP_KO[g]}**" + (" (대조군 — 2011-01-03 부터)" if g == CONTROL else "")
        L += [title, ""] + _md_table(["종목", "시작", "들고 있기"] + [r.code for r in RULES], rows) + [""]
    L += ["## 결과를 모르고 고른 목록 — 대조군 (2010년 거래대금 상위 10, 2011-01 ~ 2025-09)", ""]
    rows = [[f"{r['규칙']} {name[r['규칙']]}", f"{r['Sharpe']:.2f}", f"{r['들고있기_Sharpe']:.2f}", _num(r["dS"]),
             f"{_pct(r['CAGR'])} ({_pct(r['들고있기_CAGR'])})", f"{_pct(r['MDD'], d=0)} ({_pct(r['들고있기_MDD'], d=0)})",
             _pct(r["보유_비율"], d=0)] for _, r in ctl.iterrows()]
    L += _md_table(["규칙", "Sharpe", "들고 있기", "ΔSharpe", "연수익 (들고 있기)", "최대 낙폭 (들고 있기)", "시장에 있던 비율"],
                   rows)
    L += ["", "## 세 구간별 (ΔSharpe = 규칙 − 들고 있기)", ""]
    rows = []
    for g in TEST_GROUPS:
        for rule in RULES:
            x = per[(per["그룹"] == GROUP_KO[g]) & (per["규칙"] == rule.code)].set_index("구간")
            rows.append([GROUP_KO[g], f"{rule.code} {rule.name}"] +
                        [f"{_num(x.loc[n, 'dS'])} ({_pct(x.loc[n, 'CAGR'], d=0)} vs {_pct(x.loc[n, '들고있기_CAGR'], d=0)})"
                         for n, _, _ in PERIODS])
    L += _md_table(["그룹", "규칙"] + [f"{n} ΔSharpe (연수익 규칙 vs 들고 있기)" for n, _, _ in PERIODS], rows)
    L += ["", "## 이웃 설정 히트맵 (판정 ⑥)", "", f"![이웃 설정 히트맵]({PNG_HEAT.name})", "",
          "- 성장 B1 Supertrend 만 8칸 중 5칸이 양수였다 (ATR 배수 3·4 쪽). 기준(7칸)에 못 미쳤고, 같은 규칙이 방어주와 "
          "대조군에서는 졌다. 나머지 9개 규칙은 이웃 72칸 모두 음수 — 설정을 바꿔서 살아나는 규칙은 없었다.", "",
          "## 비용·체결 스트레스 (ΔSharpe, 들고 있기도 같은 비용)", ""]
    rows = [[r["test_id"], _num(r["dS_비용0.05%"]), _num(r["dS"]), _num(r["dS_비용2배"]), _num(r["dS_하루늦게"])]
            for _, r in t.iterrows()]
    L += _md_table(["ID", "편도 0.05% (참고)", "편도 0.1% (본 시험)", "편도 0.2%", "하루 늦은 체결"], rows)
    L += ["", "## SPY 와 비교 (참고, 1993-02 ~ 2025-09, 같은 날짜)", ""]
    rows = [[r["test_id"], f"{r['규칙_Sharpe_1993']:.2f}", f"{r['SPY_Sharpe_1993']:.2f}", _pct(r["규칙_CAGR_1993"]),
             _pct(r["SPY_CAGR_1993"])] for _, r in t.iterrows()]
    L += _md_table(["ID", "규칙 Sharpe", "SPY Sharpe", "규칙 연수익", "SPY 연수익"], rows)
    hi_s = t.loc[t["규칙_Sharpe_1993"] > t["SPY_Sharpe_1993"], "test_id"].tolist()
    hi_c = t.loc[t["규칙_CAGR_1993"] > t["SPY_CAGR_1993"], "test_id"].tolist()
    L += ["", f"- SPY 보다 Sharpe 가 높은 것: {', '.join(hi_s) or '없음'} · 연수익까지 높은 것: {', '.join(hi_c) or '없음'}. "
          "하지만 크게 오른 걸 알고 고른 종목이라서다 — 같은 종목을 그냥 들고 있으면 더 높다. 결과를 모르고 고른 대조군에서는 "
          "규칙 5개 모두 들고 있기에 졌다.", "",
          "## 그림", "", f"![1만 달러 곡선]({PNG_EQUITY.name})", ""]
    for _, tk in EXAMPLES:
        L += [f"![{tk} 예시](swing_famous_example_{tk}.png)", ""]
    L += [f"예시 구간 표 ({EXAMPLE_WINDOW[0][:7]} ~ {EXAMPLE_WINDOW[1][:7]}, 가장 최근 6년 — 그림과 같은 값. 판정은 위 표):", ""]
    rows = [[tk, f"{r.code} {r.name}", str(len(examples[tk][r.code]["buys"])), _pct(examples[tk][r.code]["ret"], True, 0),
             _pct(examples[tk][r.code]["hold"], True, 0)] for _, tk in EXAMPLES for r in RULES]
    L += _md_table(["종목", "규칙", "매수", "규칙 수익", "그냥 들고 있기"], rows)
    L += ["", "## 방법", "",
          "- 가격: `data/research/tv_ohlcv_max.parquet` (분할만 반영 = TradingView 화면 가격, S&P 500 586/586 종가 일치 검증본, "
          "SHA256 고정). 신호는 이 가격, 수익은 배당 재투자 시가 (시가 × 수정종가/종가). 국채 = Kenneth French 일간 RF (고정본).",
          "- 지표는 Pine 과 같은 계산: RSI = ta.rsi (Wilder RMA, 첫 n개 평균에서 시작), 볼린저 = ta.bb (모표준편차), "
          "Supertrend = ta.supertrend 원문을 그대로 옮김. 테스트가 `ta` 라이브러리 RSI·볼린저, Pine 원문 Supertrend 와 같은 값을 확인.",
          "- 상태: 들고 있지 않을 때만 매수 신호, 들고 있을 때만 매도 신호를 본다. t 종가 신호 → t+1 시가 체결. 손절·공매도 없음.",
          "- 시험 구간: 종목마다 자기 282번째 봉(약 1년 1개월 — 모든 지표가 계산되는 첫날)부터. TSLA 2011-08, META 2013-07, "
          "1985년부터 있는 종목은 1986-02. 대조군은 2011-01-03 부터. 홀드아웃(2025-10-01~)은 쓰지 않았다.",
          "- 그룹 수익 = 그날 구간 안 종목 칸들의 평균 (매일 같은 비중). 들고 있기도 같은 방식이라 차이는 '사고판 시점'만이다.",
          "- 무작위 매매 (⑤): 모든 종목의 실제 보유 기록을 같은 k 일(63일 ~ 가장 짧은 구간 − 63일) 통째로 밀어 날짜만 바꿨다 "
          "(끝은 앞으로 돌림). 매매 횟수·보유 기간·종목끼리 겹치는 모양은 그대로다.",
          "- 대조군 목록은 실행 때 데이터로 다시 뽑아 같은지 확인했다 (2010년 평균 종가 × 거래량, 거래일 ≥ 200, ETF·두 번째 주식 "
          "종류 제외). 11위 F, 12위 GE. 2016년 전에 상장폐지된 종목은 이 가격 파일에 없지만, 2010년 거래대금 10위 안 종목은 모두 "
          "살아 있어 영향이 없다.",
          "- 실행 뒤 바꾼 것: 보고서·그림 코드를 붙였고, `pd.concat(..., axis=1)` 에 `sort=True` 를 명시했다 (지금 pandas 기본값과 "
          "같아 숫자는 그대로 — 다시 계산해 저장값과 차이 0 확인).",
          "", "## 다음 단계", "",
          "- 실전(PT-1·시트·이메일)에 바꿀 것은 없다. 채택 후보·관찰이 없어서 그림자 추적이나 새 페이퍼 계좌도 제안하지 않는다.",
          "- 이 결과가 말하는 것: 유명 종목을 고르는 것까지는 쉽지만, 그 종목을 일봉으로 사고팔아서 들고 있기를 이기기는 어렵다. "
          "규칙이 이기는 건 결국 안 오른 종목이고, 어떤 종목이 안 오를지는 미리 모른다.",
          "", "<details><summary>연도별 전체 (그룹 연수익 %)</summary>", ""]
    for g in GROUPS:
        y = yr[yr["그룹"] == GROUP_KO[g]].set_index("연도")
        rows = [[str(yy)] + [_num(y.loc[yy, c] * 100, "+.1f") for c in ["들고있기"] + [r.code for r in RULES] + ["SPY"]]
                for yy in y.index]
        L += [f"**{GROUP_KO[g]}**", ""] + _md_table(["연도", "들고 있기"] + [r.code for r in RULES] + ["SPY"], rows) + [""]
    L += ["</details>", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {REPORT.relative_to(ROOT)} · 그림 {PNG_EQUITY.name}, {PNG_HEAT.name}, "
          + ", ".join(f"swing_famous_example_{tk}.png" for _, tk in EXAMPLES))


def do_report() -> None:
    write_report()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--count", action="store_true")
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.count:
        do_count()
    elif a.register:
        do_register()
    elif a.run:
        do_run()
    else:
        do_report()


if __name__ == "__main__":
    main()
