#!/usr/bin/env python3
"""연구소(lab) 백테스터 — 목표 비중 → 체결·비용·현금·레버리지 회계 (일 단위).

규칙 (결과 보기 전에 고정):
- target 의 t 행 = t 종가까지의 정보로 정한 목표 비중 (자산별, 합이 1을 넘으면 레버리지, 모자라면 현금).
- 결정일은 매매 주기로 정한다: M = 월 마지막 거래일, W = 주 마지막 거래일, D = 매일.
  D 는 목표와 지금 비중의 차이(자산별 최대)가 band 를 넘을 때만 매매한다.
- 체결: close = 결정 다음 거래일 종가 (그다음 날 수익부터 반영), open = 다음 거래일 시가 (시가→시가 수익).
  delay 만큼 더 늦춘다 (스트레스 시험).
- 비용: 매매한 비중 합 × 편도 비용. 현금은 무위험 금리를 받고, 빌린 몫(비중 합 − 1)은 무위험 금리 + borrow_spread 를 낸다.
- drift=True: 다음 결정일까지 비중이 가격 따라 움직인다 (실제 보유와 같다).
  drift=False: 매일 목표 비중으로 유지 (idea_batch 재현용).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_utils import COST_PER_SIDE  # noqa: E402

TRADING_DAYS = 252
BORROW_SPREAD = 0.01          # 빌린 몫의 무위험 금리 위 비용 (2배 ETF 운용보수 + 스왑 비용 근사, 연)
SCHEDULES: dict[str, tuple[str, float]] = {   # 이름 → (결정 주기, 매매 폭) — 느린 것 → 빠른 것 순서
    "M": ("M", 0.0), "W": ("W", 0.0), "D10": ("D", 0.10), "D5": ("D", 0.05), "D2": ("D", 0.02),
    "D0": ("D", 0.0),                           # 달력 효과 전용 (탐색 주기 아님)
}
SCHEDULE_ORDER = ["M", "W", "D10", "D5", "D2"]  # 탐색하는 주기
SCHEDULE_KO = {"M": "월 1회", "W": "주 1회", "D10": "매일(폭 10%p)", "D5": "매일(폭 5%p)", "D2": "매일(폭 2%p)",
               "D0": "매일"}
MIN_TRADE = 1e-12                               # 이보다 작은 비중 차이는 매매하지 않는다 (부동소수 오차)


def decision_mask(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """결정일 표시: M = 그 달 마지막 거래일, W = 그 주(월~일) 마지막 거래일, D = 매일."""
    n = len(index)
    if freq == "D":
        return np.ones(n, dtype=bool)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if freq == "M":
        key = index.year * 12 + index.month
    elif freq == "W":
        key = (index - pd.to_timedelta(index.dayofweek, unit="D")).asi8
    else:
        raise ValueError(f"알 수 없는 주기: {freq}")
    key = np.asarray(key)
    out = np.zeros(n, dtype=bool)
    out[:-1] = key[1:] != key[:-1]
    out[-1] = True
    return out


@dataclass
class SimResult:
    """시뮬레이션 결과. ret = 비용 뺀 일간 수익, weights = 그날 들고 있던 비중(그날 수익 전), turnover = 매매한 비중 합."""

    ret: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    trades: int

    @property
    def exposure(self) -> pd.Series:
        return self.weights.sum(axis=1)


def simulate(target: pd.DataFrame, asset_ret: pd.DataFrame, rf: pd.Series | None = None, *,
             freq: str = "M", band: float = 0.0, cost: float = COST_PER_SIDE, exec_mode: str = "close",
             delay: int = 0, drift: bool = True, borrow_spread: float = BORROW_SPREAD,
             initial: pd.Series | None = None) -> SimResult:
    """목표 비중을 매매 주기에 따라 체결해 일간 수익을 만든다 (모듈 설명의 규칙).

    Args:
        target: 날짜 × 자산 목표 비중. 행 전체가 NaN 이면 그날은 결정하지 않는다. 일부 NaN 은 0.
        asset_ret: 날짜 × 자산 수익률 (close: 종가→종가, t 에 기록 / open: t 시가→t+1 시가, t 에 기록). NaN 은 0.
        rf: 일간 무위험 수익 (없으면 0).
        initial: 첫날부터 들고 있는 비중 (첫날 매수 비용을 낸다).
    """
    if exec_mode not in ("close", "open"):
        raise ValueError(exec_mode)
    idx = asset_ret.index
    cols = list(asset_ret.columns)
    tg = target.reindex(index=idx, columns=cols).to_numpy(dtype=float)
    valid = ~np.all(np.isnan(tg), axis=1)
    tg = np.nan_to_num(tg)
    R = np.nan_to_num(asset_ret.to_numpy(dtype=float))
    T, N = R.shape
    dec = decision_mask(idx, freq) & valid
    rfa = np.zeros(T) if rf is None else rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    spread = borrow_spread / TRADING_DAYS
    shift = (2 if exec_mode == "close" else 1) + int(delay)

    w = np.zeros(N) if initial is None else initial.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    out_ret = np.empty(T)
    out_w = np.empty((T, N))
    out_turn = np.zeros(T)
    pending: dict[int, np.ndarray] = {}
    last_order: np.ndarray | None = None      # 아직 체결 안 된 마지막 주문 (매매 폭 판단용)
    trades = 0
    for i in range(T):
        turn = float(np.abs(w).sum()) if (i == 0 and initial is not None) else 0.0
        order = pending.pop(i, None)
        if order is not None:
            turn = float(np.abs(order - w).sum())
            w = order
            trades += 1
            if not pending:
                last_order = None
        out_w[i] = w
        r_i = R[i]
        s = float(w.sum())
        cash = 1.0 - s
        r_cash = cash * rfa[i] if cash >= 0 else cash * (rfa[i] + spread)
        net = float(w @ r_i) + r_cash - turn * cost
        out_ret[i] = net
        out_turn[i] = turn
        if drift and (1.0 + net) > 0:
            w = w * (1.0 + r_i) / (1.0 + net)
        if dec[i]:
            desired = tg[i]
            ref = last_order if last_order is not None else w
            if float(np.abs(desired - ref).max()) <= max(band, MIN_TRADE):
                continue
            e = i + shift
            if e < T:
                pending[e] = desired.copy()
                last_order = pending[e]
    return SimResult(pd.Series(out_ret, index=idx), pd.DataFrame(out_w, index=idx, columns=cols),
                     pd.Series(out_turn, index=idx), trades)


def run_schedule(target: pd.DataFrame, asset_ret: pd.DataFrame, rf: pd.Series | None, schedule: str,
                 **kw) -> SimResult:
    """SCHEDULES 이름(M·W·D10·D5·D2)으로 simulate."""
    freq, band = SCHEDULES[schedule]
    return simulate(target, asset_ret, rf, freq=freq, band=band, **kw)


# ── 성과 ──────────────────────────────────────────────────────────────────────────────────────

def perf(ret: pd.Series, rf: pd.Series | None = None) -> dict[str, float]:
    """Sharpe(초과·원래), CAGR, MDD, 변동성. 초과 Sharpe = (수익 − 무위험) 평균 / 표준편차 × √252."""
    r = ret.to_numpy(dtype=float)
    n = len(r)
    if n < 2:
        return {"Sharpe": float("nan"), "Sharpe_raw": float("nan"), "CAGR": float("nan"), "MDD": float("nan"),
                "vol": float("nan"), "days": n}
    ex = r - (rf.reindex(ret.index).fillna(0.0).to_numpy(dtype=float) if rf is not None else 0.0)
    sd_ex, sd = ex.std(ddof=1), r.std(ddof=1)
    growth = float(np.prod(1.0 + r))
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return {
        "Sharpe": float(ex.mean() / sd_ex * np.sqrt(TRADING_DAYS)) if sd_ex > 0 else 0.0,
        "Sharpe_raw": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "CAGR": growth ** (TRADING_DAYS / n) - 1.0 if growth > 0 else -1.0,
        "MDD": float(min(0.0, (eq / peak - 1.0).min())),
        "vol": float(sd * np.sqrt(TRADING_DAYS)),
        "days": n,
    }


def score(p: dict[str, float], mdd_free: float = 0.30, penalty: float = 2.0) -> float:
    """옵션 A 점수 = 초과 Sharpe − max(0, |MDD| − 30%) × 2 (wfo_structure·isb_label_search 와 같은 꼴)."""
    return float(p["Sharpe"]) - max(0.0, abs(float(p["MDD"])) - mdd_free) * penalty


# ── 변동성 맞추기 ──────────────────────────────────────────────────────────────────────────────

def vol_overlay(target: pd.DataFrame, asset_ret: pd.DataFrame, sigma_star: float, *, freq: str,
                lookback: int = 63, cap: float = 1.5) -> pd.DataFrame:
    """결정일마다 목표 비중 전체에 k = min(cap, σ*/σ̂) 를 곱한다.

    σ̂ = 그 목표 비중을 최근 lookback 일 자산 수익의 공분산으로 잰 연 변동성 (사전 추정, t 까지의 정보만).
    결정일이 아닌 행은 NaN (결정 안 함).
    """
    idx = asset_ret.index
    cols = list(asset_ret.columns)
    tg = target.reindex(index=idx, columns=cols).to_numpy(dtype=float)
    valid = ~np.all(np.isnan(tg), axis=1)
    tg = np.nan_to_num(tg)
    R = asset_ret.to_numpy(dtype=float)
    dec = decision_mask(idx, freq) & valid
    out = np.full(tg.shape, np.nan)
    for i in np.flatnonzero(dec):
        w = tg[i]
        nz = np.flatnonzero(w != 0)
        if len(nz) == 0:
            out[i] = w
            continue
        if i + 1 < lookback:
            continue
        win = np.nan_to_num(R[i - lookback + 1:i + 1][:, nz])
        cov = np.atleast_2d(np.cov(win, rowvar=False, ddof=1))
        var = float(w[nz] @ cov @ w[nz]) * TRADING_DAYS
        k = cap if var <= 0 else min(cap, sigma_star / np.sqrt(var))
        out[i] = w * k
    return pd.DataFrame(out, index=idx, columns=cols)


def synthetic_2x(r: pd.Series, rf: pd.Series, spread: float = BORROW_SPREAD) -> pd.Series:
    """매일 2배로 다시 맞추는 가상 2배 상품: 2r − (무위험 + spread/252). 실제 2배 ETF 대조용."""
    rfa = rf.reindex(r.index).fillna(0.0)
    return 2.0 * r - (rfa + spread / TRADING_DAYS)
