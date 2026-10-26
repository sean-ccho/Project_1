#!/usr/bin/env python3
"""내계좌 스타일만 백테스트 (my_style_v1): PT-1 엔진·필터·매도 규칙을 하나도 쓰지 않고, 사용자가 직접 사는 방식만으로.

계기: 2026-10-22 사용자 — "다른 거 다 따라하지 말고 내 계좌 스타일로만 백테스트해 볼 수 있어?"
(앞선 my_picks_v1 은 고르는 규칙만 사용자 것이고 계좌·매도는 PT-1 엔진이었다.)
사용자 매수 5건 + 본인 설명에서 옮긴 것 (공개 저장소라 실제 종목은 적지 않는다):
  - 사이트 차트에서 골든크로스(임박 포함) 라벨이 있는 종목 (5건 모두, 일·주·월봉 중 하나 이상), 20일 수익 플러스 (5건 모두)
  - 매출 성장 플러스 (5건 모두), 내부자 매수도 봤다 (5건 중 3건 매수 기록), 헬스체크도 봤다 (55~90 제각각 → 과거 값이 없어 시험 못 함)
  - 차트추천 목록(★ 높은 순)에서 고름, 5종목 균등 (각 약 20%), 며칠 안에 다 채움, 손절가는 ATR×3 (1종목), 아직 판 적 없음
규칙 (그날까지 알 수 있던 값만, 그날 S&P 500 구성 종목):
  라벨·★·20일 수익 = 사이트와 같은 스크리너 라벨을 날짜별로 다시 만든 표 (data/research/my_picks_lists.parquet,
                       my_picks_research 가 기록 — 그날 유동성 통과 종목 중 골든 라벨이 있는 행)
  매출 YoY      = 그날까지 공시된 최근 분기 매출 YoY (SEC companyfacts 처음 제출 값, 분기 말 200일 넘으면 모름 = 불통과)
  내부자 매수   = 그날 포함 90일 안 Form 4 장내 매수(P) 신고자 수 (첫 신고자 기준)
  S1 = 골든 라벨 · 20일 수익 > 0 · 매출 YoY > 0        S2 = S1 + 내부자 매수 1명 이상
  계좌 = 5칸 균등: 빈 칸은 그날 모두 채운다 (그날 종가 기준 자본 ÷ 5, 현금 한도 안). ★ 높은 순 (동점 티커순),
         보유 중·같은 회사 제외. 다음날 시가에 산다. 편도 0.1%
  매도 (그날 종가로 판단 → 다음날 시가):  손절 = 진입가 − ATR(14)% × 3 (신호일 값) 아래 종가 — 세 방식 공통
         X1 = 21거래일 보유 (약 한 달)   X2 = 63거래일 보유 (약 석 달)   X3 = 종가가 EMA50 아래로 내려가면 (기간 제한 없음)
사전 등록 (TRIAL_LOG family my_style_v1, 6건 = S1·S2 × X1·X2·X3 — 결과 계산 전에 --register):
  판정 = SPY(같은 가격 자료, 매수 보유) 대비. SPY 이김 = ΔSharpe 95% 하한 > 0 · p ≤ 0.05/6 (paired_bootstrap 1만·20일 블록),
         관찰 = 하한 > 0 이지만 p 미달, 나머지 탈락
  참고 (판정 없음): PT-1 BASE 대비, 고르는 순서 무작위 20시드 (운의 폭), 거래 수·승률·보유일·평균 보유 종목 수, 연도별
기간 2016-01-04 ~ 2025-09-30 (홀드아웃 안 씀).
한계: S&P 500 밖 종목(사용자 5종목 중 3개)은 생존 편향 없는 자료가 없어 못 넣었다. 헬스체크·'차트가 좋아 보임'은 규칙으로 못 옮김.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/my_style_research.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/my_style_research.py --run
산출: tier3/MY_STYLE_REPORT.md · tier3/my_style.csv · data/research/my_style_* (git 제외)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_engine as isb  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
from paper_trading import candidate_selector as cs  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe  # noqa: E402
from research_utils import DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv, log_trials, total_trials  # noqa: E402

FAMILY, STEP = "my_style_v1", "my_style"
SELECTIONS = {"S1": "골든 라벨 + 20일 상승 + 매출 성장", "S2": "S1 + 내부자 매수 (90일)"}
EXITS = {"X1": "21거래일 보유", "X2": "63거래일 보유", "X3": "종가 < EMA50 이면 매도"}
REF_EXITS = {"X3b": "EMA50 위로 한 번 올라선 뒤 아래로 내려가면 매도"}   # 사후 참고 (사전 등록 아님)
REF_FAMILY = "my_style_followup_v1"
TESTS = tuple(f"{s}-{x}" for s in SELECTIONS for x in EXITS)
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
N_BOOT, N_BOOT_REF = 10_000, 1_000
SLOTS = 5
COST = 0.001
ATR_N, ATR_MULT, EMA_N = 14, 3.0, 50
STOP_FALLBACK = 0.15          # ATR 을 모를 때 손절폭
HOLD = {"X1": 21, "X2": 63}
SEEDS = tuple(range(1, 21))
START = pd.Timestamp("2016-01-04")
LISTS_PATH = RESEARCH_DIR / "my_picks_lists.parquet"
POOL_PATH = RESEARCH_DIR / "my_style_pool.parquet"
RUNS_PATH = RESEARCH_DIR / "my_style_runs.pkl"
BASE_SAVED = RESEARCH_DIR / "pt1_base_dev_equity.parquet"
OUT_CSV = TIER3_DIR / "my_style.csv"
REPORT = TIER3_DIR / "MY_STYLE_REPORT.md"
NAN = float("nan")


def spec() -> str:
    return (f"[2016-01-04~{DEV_END.date()}, 그날 S&P 500, PT-1 엔진 안 씀] S1 = 골든 라벨(일·주·월 하나 이상) · 20일 수익>0 · "
            "최근 분기 매출 YoY>0 (SEC 처음 제출 값, 200일 넘으면 불통과); S2 = S1 + 90일 Form 4 장내 매수 신고자 ≥1. "
            f"계좌 {SLOTS}칸 균등, 빈 칸은 그날 모두 채움, ★ 높은 순, 다음날 시가, 편도 {COST:.1%}. "
            f"매도: 공통 손절 = 진입가 − ATR({ATR_N})%×{ATR_MULT:g} 아래 종가; X1 21일 · X2 63일 보유 · X3 종가<EMA{EMA_N}. "
            f"판정 vs SPY: SPY 이김 = ΔSharpe 95% 하한>0 · p ≤ 0.05/{N_TESTS}; 관찰 = 하한>0 · p 미달. "
            f"참고 = PT-1 BASE 대비, 무작위 순서 {len(SEEDS)}시드, 거래 통계, 연도별.")


# ============================================================
# 자료
# ============================================================
@dataclass
class Market:
    """시뮬레이터가 쓰는 가격 배열 (날짜 × 종목)."""

    dates: pd.DatetimeIndex
    tickers: list[str]
    open: np.ndarray
    close: np.ndarray
    ema: np.ndarray
    atr_pct: np.ndarray
    col: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.col = {t: j for j, t in enumerate(self.tickers)}


def atr_pct(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = ATR_N) -> pd.DataFrame:
    """ATR(n) ÷ 종가 — 참범위의 n일 단순평균."""
    prev = close.shift(1)
    tr = np.fmax(high - low, np.fmax((high - prev).abs(), (low - prev).abs()))   # 첫날(전일 종가 없음)은 고가 − 저가
    return tr.rolling(n, min_periods=n).mean() / close


def load_market() -> Market:
    raw = load_ohlcv(isb.PRICE_CACHE)
    px = {k: raw.xs(k, axis=1, level="Price").sort_index() for k in ("Open", "High", "Low", "Close")}
    close = px["Close"]
    ema = close.ewm(span=EMA_N, adjust=False, min_periods=EMA_N).mean()
    atr = atr_pct(px["High"], px["Low"], close)
    return Market(close.index, list(close.columns), px["Open"].to_numpy(float), close.to_numpy(float),
                  ema.to_numpy(float), atr.to_numpy(float))


def build_pool(force: bool = False) -> pd.DataFrame:
    """골든 라벨 · 20일 상승 행에 매출 YoY · 내부자 매수 신고자 수를 붙인다 (그날까지 알 수 있던 값)."""
    if POOL_PATH.exists() and not force:
        return pd.read_parquet(POOL_PATH)
    from gc_insider_rev import books
    lists = pd.read_parquet(LISTS_PATH)
    lists["date"] = pd.to_datetime(lists["date"])
    x = lists[(lists["tfs"].astype(str) != "") & (lists["ret20"] > 0)].copy()
    cb, rb = books()
    x["rev_yoy"] = [rb.yoy(t, d) for t, d in zip(x["ticker"], x["date"])]
    x["buyers"] = [cb.buyers(t, d) for t, d in zip(x["ticker"], x["date"])]
    x = x.reset_index(drop=True)
    x.to_parquet(POOL_PATH, index=False)
    return x


def eligible(pool: pd.DataFrame, sel: str) -> pd.DataFrame:
    """S1 = 매출 YoY > 0 (골든·20일 상승은 pool 이 이미 걸렀다), S2 = S1 + 내부자 매수 1명 이상."""
    ok = np.nan_to_num(pool["rev_yoy"].to_numpy(float), nan=-1.0) > 0
    if sel == "S2":
        ok &= pool["buyers"].to_numpy() >= 1
    elif sel != "S1":
        raise ValueError(sel)
    return pool[ok]


# ============================================================
# 시뮬레이터 (5칸 균등 계좌)
# ============================================================
@dataclass
class Pos:
    j: int
    ticker: str
    entry_i: int
    entry_px: float
    shares: float
    stop_px: float
    armed: bool = False          # X3b: 산 뒤 종가가 EMA50 위로 한 번 올라섰나


def _group(t: str) -> str:
    return cs._SHARE_CLASS_GROUPS.get(t, t)


def order_key(score: float, ticker: str, seed: int | None, day: pd.Timestamp) -> tuple:
    if seed is not None:
        return (iv.stable_uniform("my_style", seed, day, ticker), ticker)
    return (-(score if score == score else -np.inf), ticker)


def exit_due(rule: str, pos: Pos, i: int, close: float, ema: float) -> str | None:
    """그날 종가로 판단 — 손절이 먼저, 그다음 방식별 매도."""
    if close <= pos.stop_px:
        return "손절"
    held = i - pos.entry_i + 1                      # 진입일 포함 보유 거래일
    if rule in HOLD and held >= HOLD[rule]:
        return f"{HOLD[rule]}일"
    if rule == "X3" and ema == ema and close < ema:
        return "EMA50 이탈"
    if rule == "X3b" and ema == ema:
        if close >= ema:
            pos.armed = True
        elif pos.armed:
            return "EMA50 이탈"
    return None


def simulate(mk: Market, cands: dict[int, list[tuple[float, str]]], rule: str, seed: int | None = None,
             start: pd.Timestamp = START, end: pd.Timestamp = DEV_END, capital: float = 10_000.0
             ) -> tuple[pd.Series, list[dict[str, Any]]]:
    """cands: 날짜 위치 → [(★, 티커)] (그날 규칙 통과). 그날 종가로 판단하고, 다음날 시가에 사고판다 (장부도 그날)."""
    i0 = int(mk.dates.searchsorted(start))
    i1 = int(mk.dates.searchsorted(end, side="right")) - 1
    cash, positions, trades, eq = capital, [], [], {}
    sell_next: list[tuple[Pos, str]] = []
    buy_next: list[tuple[int, str, float, float]] = []      # (열, 티커, 배정 금액, 손절폭)
    for i in range(i0, i1 + 1):
        # 1) 어제 종가로 정한 매도 → 매수를 오늘 시가에
        for p, why in sell_next:
            op = mk.open[i, p.j]
            px = float(op) if np.isfinite(op) and op > 0 else _last_close(mk, i, p)
            cash += p.shares * px * (1 - COST)
            trades.append(_trade(p, mk, i, px, why))
        sell_next = []
        for j, tk, alloc, stop_pct in buy_next:
            op = float(mk.open[i, j])
            alloc = min(alloc, cash)                          # 밤사이 갭으로 판 돈이 모자라면 그만큼만
            if alloc <= 1e-9:
                continue
            cash -= alloc
            positions.append(Pos(j, tk, i, op, alloc / (op * (1 + COST)), op * (1 - stop_pct)))
        buy_next = []
        # 2) 가격이 끊긴 종목은 마지막 종가로 정리, 나머지는 오늘 종가로 평가
        keep = []
        for p in positions:
            if np.isfinite(mk.close[i, p.j]):
                keep.append(p)
                continue
            px = _last_close(mk, i, p)
            cash += p.shares * px * (1 - COST)
            trades.append(_trade(p, mk, i, px, "가격 끊김"))
        positions = keep
        equity = cash + sum(p.shares * float(mk.close[i, p.j]) for p in positions)
        eq[mk.dates[i]] = equity
        if i == i1:                                         # 마지막 날: 그날 종가로 정리
            for p in positions:
                trades.append(_trade(p, mk, i, float(mk.close[i, p.j]), "기간 끝"))
            break
        # 3) 오늘 종가로 매도 판단 → 내일 시가
        keep = []
        for p in positions:
            why = exit_due(rule, p, i, float(mk.close[i, p.j]), float(mk.ema[i, p.j]))
            (sell_next.append((p, why)) if why else keep.append(p))
        positions = keep
        # 4) 빈 칸 → 오늘 규칙 통과 종목을 ★ 순으로 내일 시가에 (팔 종목 대금은 오늘 종가로 어림)
        free = SLOTS - len(positions)
        if free <= 0 or i not in cands:
            continue
        avail = cash + sum(p.shares * float(mk.close[i, p.j]) * (1 - COST) for p, _ in sell_next)
        held = {_group(p.ticker) for p in positions} | {_group(p.ticker) for p, _ in sell_next}
        day = mk.dates[i]
        for score, tk in sorted(cands[i], key=lambda x: order_key(x[0], x[1], seed, day)):
            if free <= 0:
                break
            j = mk.col.get(tk)
            if j is None or _group(tk) in held:
                continue
            op = mk.open[i + 1, j]
            if not (np.isfinite(op) and op > 0):
                continue
            alloc = min(avail, equity / SLOTS)
            if alloc <= 1e-9:
                break
            a = mk.atr_pct[i, j]
            buy_next.append((j, tk, alloc, ATR_MULT * a if np.isfinite(a) and a > 0 else STOP_FALLBACK))
            avail -= alloc
            held.add(_group(tk))
            free -= 1
    return pd.Series(eq, name="equity"), trades


def _last_close(mk: Market, i: int, p: Pos) -> float:
    last = mk.close[:i + 1, p.j][np.isfinite(mk.close[:i + 1, p.j])]
    return float(last[-1]) if last.size else p.entry_px


def _trade(p: Pos, mk: Market, exit_i: int, px: float, why: str) -> dict[str, Any]:
    return {"ticker": p.ticker, "entry_date": mk.dates[p.entry_i], "exit_date": mk.dates[exit_i],
            "entry_price": p.entry_px, "exit_price": px, "return_pct": px * (1 - COST) / (p.entry_px * (1 + COST)) - 1,
            "days": exit_i - p.entry_i, "reason": why}


def candidates(mk: Market, el: pd.DataFrame) -> dict[int, list[tuple[float, str]]]:
    pos = mk.dates.get_indexer(pd.to_datetime(el["date"]))
    out: dict[int, list[tuple[float, str]]] = {}
    for i, sc, tk in zip(pos, el["score"].to_numpy(float), el["ticker"].astype(str)):
        if i >= 0:
            out.setdefault(int(i), []).append((float(sc), tk))
    return out


def avg_positions(eq: pd.Series, trades: list[dict[str, Any]]) -> float:
    """하루 평균 보유 종목 수 (진입일 ≤ d < 청산일)."""
    idx = eq.index
    n = np.zeros(len(idx))
    for t in trades:
        a, b = idx.searchsorted(t["entry_date"]), idx.searchsorted(t["exit_date"])
        n[a:b] += 1
    return float(n.mean())


# ============================================================
# 등록 · 실행 · 판정
# ============================================================
def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def _registered() -> bool:
    log = _log()
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def register() -> None:
    if _registered():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"등록: {FAMILY} {N_TESTS}건 · p ≤ {P_BONF:.4f} · 누적 시도 {total_trials()}건")


def judge(st: dict[str, float]) -> str:
    if st and st["ΔSharpe_하한"] > 0 and st["p_개선아님"] <= P_BONF:
        return "SPY 이김"
    if st and st["ΔSharpe_하한"] > 0:
        return "관찰"
    return "탈락"


def _summarize(tid: str, sel: str, x: str, eq: pd.Series, tr: list[dict[str, Any]], seeds: dict[int, pd.Series],
               spy_r: pd.Series, base_r: pd.Series, per_day: pd.Series, ref: bool) -> tuple[dict[str, Any], str]:
    """한 방식의 표 한 줄과 TRIAL_LOG 메모."""
    r = eq.pct_change().dropna()
    st, df = iv._cmp(r, spy_r, N_BOOT_REF if ref else N_BOOT)
    st_b, _ = iv._cmp(r, base_r, N_BOOT_REF)
    a, b = df["a"].tolist(), df["b"].tolist()
    seed_sh = pd.Series([sharpe(v.pct_change().dropna().tolist()) for v in seeds.values()])
    rets = np.array([t["return_pct"] for t in tr])
    row = {
        "test_id": tid, "고르기": SELECTIONS[sel], "매도": (REF_EXITS if ref else EXITS)[x], "Sharpe": sharpe(a),
        "CAGR": cagr(a), "MDD": max_drawdown(a), "SPY_Sharpe": sharpe(b), "SPY_CAGR": cagr(b),
        "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
        "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}",
        "ΔPT1": st_b.get("ΔSharpe"), "ΔPT1_하한": st_b.get("ΔSharpe_하한"), "ΔPT1_상한": st_b.get("ΔSharpe_상한"),
        "거래수": len(tr), "승률": float((rets > 0).mean()) if len(rets) else NAN,
        "거래당 평균": float(rets.mean()) if len(rets) else NAN, "거래당 중앙": float(np.median(rets)) if len(rets) else NAN,
        "보유일 중앙": float(np.median([t["days"] for t in tr])) if tr else NAN,
        "손절 비율": float(np.mean([t["reason"] == "손절" for t in tr])) if tr else NAN,
        "평균 보유 종목": avg_positions(eq, tr), "후보 하루 평균": float(per_day.mean()), "후보 없는 날": float((per_day == 0).mean()),
        "무작위 가운데": float(seed_sh.median()), "무작위 최소": float(seed_sh.min()), "무작위 최대": float(seed_sh.max()),
        "무작위 SPY 넘음": int((seed_sh > sharpe(b)).sum()), "판정": "참고" if ref else judge(st)}
    memo = (f"Sharpe {sharpe(a):.2f} vs SPY {sharpe(b):.2f}, ΔSharpe {st.get('ΔSharpe'):+.2f} [{st.get('ΔSharpe_하한'):+.2f}, "
            f"{st.get('ΔSharpe_상한'):+.2f}] p {st.get('p_개선아님'):.4f}, CAGR {cagr(a):+.1%}, MDD {max_drawdown(a):.1%}, "
            f"거래 {len(tr)}, 무작위 가운데 {seed_sh.median():.2f}")
    return row, memo


def run() -> None:
    if not _registered():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    log = _log()
    already = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    print("===== 자료: 가격 · 조건표", flush=True)
    mk = load_market()
    pool = build_pool()
    spy_j = mk.col["SPY"]
    spy = pd.Series(mk.close[:, spy_j], index=mk.dates).loc[START:DEV_END]
    spy_r = spy.pct_change().dropna()
    base = pd.read_parquet(BASE_SAVED).iloc[:, 0]
    base.index = pd.to_datetime(base.index)
    base_r = base.pct_change().dropna()
    runs: dict[str, Any] = {}
    rows, log_rows, yearly = [], [], {}
    ref_rows, ref_log = [], []
    for sel in SELECTIONS:
        el = eligible(pool, sel)
        cands = candidates(mk, el)
        per_day = el.groupby("date").size().reindex(mk.dates[(mk.dates >= START) & (mk.dates <= DEV_END)], fill_value=0)
        for x in (*EXITS, *REF_EXITS):
            tid = f"{sel}-{x}"
            ref = x in REF_EXITS
            print(f"===== {tid}{' (사후 참고)' if ref else ''}: {SELECTIONS[sel]} · {(REF_EXITS if ref else EXITS)[x]} "
                  f"(+ 무작위 {len(SEEDS)}시드)", flush=True)
            eq, tr = simulate(mk, cands, x)
            seeds = {s_: simulate(mk, cands, x, seed=s_)[0] for s_ in SEEDS}
            runs[tid] = {"equity": eq, "trades": tr, "seeds": seeds}
            row, memo = _summarize(tid, sel, x, eq, tr, seeds, spy_r, base_r, per_day, ref)
            (ref_rows if ref else rows).append(row)
            (ref_log if ref else log_rows).append({"test_id": tid, "t": "", "판정": row["판정"], "메모": memo})
            yearly[tid] = eq.pct_change().dropna()
            print(f"  {tid}: 거래 {len(tr)} · 평균 보유 {row['평균 보유 종목']:.1f}종목", flush=True)
    pd.to_pickle(runs, RUNS_PATH)
    table, ref_table = pd.DataFrame(rows), pd.DataFrame(ref_rows)
    pd.concat([table, ref_table]).to_csv(OUT_CSV, index=False)
    if not already:
        log_trials(STEP, FAMILY, N_TESTS, log_rows)
    else:
        print("TRIAL_LOG 에 결과를 쓰지 않는다 (이미 있음)", flush=True)
    log = _log()
    if not ((log["family"] == REF_FAMILY)).any():
        log_trials(STEP, REF_FAMILY, len(ref_log), [{**r, "메모": "사후 참고(사전 등록 아님): " + r["메모"]} for r in ref_log])
    yearly["SPY"] = spy_r
    yearly["PT-1"] = base_r
    write_report(table, yearly, runs, ref_table)


def _yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def write_report(table: pd.DataFrame, yearly: dict[str, pd.Series], runs: dict[str, Any],
                 ref: pd.DataFrame | None = None) -> None:
    f = iv._f
    t = table.set_index("test_id")
    spy_sh, spy_cagr = float(table["SPY_Sharpe"].iloc[0]), float(table["SPY_CAGR"].iloc[0])
    base_r = yearly["PT-1"]
    pt1_sh, pt1_cagr = sharpe(base_r.tolist()), cagr(base_r.tolist())
    best = t["Sharpe"].idxmax()
    passed = [x for x in TESTS if t.at[x, "판정"] == "SPY 이김"]
    watch = [x for x in TESTS if t.at[x, "판정"] == "관찰"]
    if passed:
        head = f"**결론: SPY 를 이긴다고 볼 수 있는 방식이 있다 — {', '.join(passed)}.**"
    else:
        head = ("**결론: 내계좌 스타일(골든 라벨 + 20일 상승 + 매출 성장, ★ 높은 순, 5종목 균등)은 매도 방식 셋 · 내부자 조건 유무 "
                "어느 것으로도 SPY 를 이긴다고 볼 수 없다.**" + (f" ('관찰': {', '.join(watch)})" if watch else ""))
    lines = [
        f"# 내계좌 스타일만 백테스트 ({FAMILY}) — 2016-01-04 ~ {DEV_END.date()}", "", head, "",
        f"- 가장 좋았던 것: {best} ({t.at[best, '고르기']} · {t.at[best, '매도']}) Sharpe {t.at[best, 'Sharpe']:.2f} · "
        f"연 {t.at[best, 'CAGR']:+.1%} · 최대낙폭 {t.at[best, 'MDD']:.1%} vs SPY {spy_sh:.2f} · 연 {spy_cagr:+.1%} · "
        f"PT-1 {pt1_sh:.2f} · 연 {pt1_cagr:+.1%}",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · {N_TESTS}건 · p ≤ 0.05/{N_TESTS} · 부트스트랩 {N_BOOT:,}회 · "
        f"누적 시도 {total_trials():,}건",
        "- 계기: 사용자 — \"다른 거 다 따라하지 말고 내 계좌 스타일로만 백테스트해 볼 수 있어?\" (2026-10-22). PT-1 엔진·필터·매도 "
        "규칙은 하나도 안 썼다. 공개 저장소라 실제 종목은 적지 않는다.",
        "- 사용자 방식에서 옮긴 것: 사이트 골든크로스(임박 포함) 라벨 · 20일 수익 플러스 · 매출 성장 플러스 (매수 5건 모두) · "
        "내부자 매수는 봤다 (5건 중 3건) → 넣은 것(S2)과 뺀 것(S1) 둘 다 · 차트추천처럼 ★ 높은 순 · 5종목 균등 · 손절 ATR×3",
        "- 아직 판 적이 없어 매도는 세 가지로: 한 달(21거래일) · 석 달(63거래일) · 추세 이탈(종가 < EMA50, 기간 제한 없음). "
        "셋 다 손절(진입가 − ATR%×3) 먼저",
        "- 그날까지 알 수 있던 값만: 라벨·★·20일 = 그날 스크리너 라벨, 매출 = 그날까지 공시된 최근 분기 YoY (SEC), "
        "내부자 = 90일 안 Form 4 장내 매수. 가격은 배당 포함 (SPY 도 같은 자료)",
        f"- 판정: SPY 대비 ΔSharpe 95% 하한 > 0 · p ≤ {P_BONF:.4f} = SPY 이김, 하한 > 0 이지만 p 미달 = 관찰",
        "", "## 결과", "",
        "| 방식 | Sharpe | 연수익 | 최대낙폭 | SPY 대비 ΔSharpe [95%] | p | 구간 | PT-1 대비 ΔSharpe [95%] | 판정 |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| SPY 들고 있기 | {spy_sh:.2f} | {spy_cagr:+.1%} | | | | | | 기준 |",
        f"| PT-1 (참고) | {pt1_sh:.2f} | {pt1_cagr:+.1%} | {max_drawdown(base_r.tolist()):.1%} | | | | | 참고 |"]
    for tid in TESTS:
        x = t.loc[tid]
        lines.append(f"| {tid} {x['고르기']} · {x['매도']} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | "
                     f"{f(x['ΔSharpe'], '+.2f')} [{f(x['하한'], '+.2f')}, {f(x['상한'], '+.2f')}] | {f(x['p'], '.4f')} | "
                     f"{x['구간']} | {f(x['ΔPT1'], '+.2f')} [{f(x['ΔPT1_하한'], '+.2f')}, {f(x['ΔPT1_상한'], '+.2f')}] | "
                     f"**{x['판정']}** |")
    lines += ["", "## 어떻게 거래됐나", "",
              "| 방식 | 거래 | 승률 | 거래당 평균 · 중앙 | 보유일 중앙 | 손절로 끝남 | 평균 보유 종목 | 후보 하루 평균 · 없는 날 |",
              "|---|---|---|---|---|---|---|---|"]
    for tid in TESTS:
        x = t.loc[tid]
        lines.append(f"| {tid} | {int(x['거래수'])} | {f(x['승률'], '.0%')} | {f(x['거래당 평균'], '+.2%')} · "
                     f"{f(x['거래당 중앙'], '+.2%')} | {f(x['보유일 중앙'], '.0f')}일 | {f(x['손절 비율'], '.0%')} | "
                     f"{x['평균 보유 종목']:.1f} / {SLOTS} | {x['후보 하루 평균']:.1f}개 · {x['후보 없는 날']:.0%} |")
    x3 = [t.at[f"{sel}-X3", "보유일 중앙"] for sel in SELECTIONS]
    lines += ["", f"- 주의: X3 은 산 날 이미 EMA50 아래였던 종목을 바로 다음날 팔아서 보유일 가운데값이 "
              f"{' · '.join(f'{v:.0f}일' for v in x3)}뿐이다 — 추세 매도라기보다 사고 바로 파는 꼴이 됐다 (사전 등록 그대로 둠, "
              "고친 버전은 아래 참고)."]
    if ref is not None and len(ref):
        r_ = ref.set_index("test_id")
        lines += ["", "## 참고: 추세 매도를 고친 버전 — EMA50 위로 한 번 올라선 뒤 내려가면 매도 (사후, 판정 없음)", "",
                  "| 방식 | Sharpe | 연수익 | 최대낙폭 | SPY 대비 ΔSharpe [95%] | 거래 | 승률 | 보유일 중앙 | 무작위 가운데 (최소 ~ 최대) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for tid, x in r_.iterrows():
            lines.append(f"| {tid} {x['고르기']} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | "
                         f"{f(x['ΔSharpe'], '+.2f')} [{f(x['하한'], '+.2f')}, {f(x['상한'], '+.2f')}] | {int(x['거래수'])} | "
                         f"{f(x['승률'], '.0%')} | {f(x['보유일 중앙'], '.0f')}일 | {x['무작위 가운데']:.2f} "
                         f"({x['무작위 최소']:.2f} ~ {x['무작위 최대']:.2f}) |")
    lines += ["", "## 운의 폭 — 같은 규칙에서 고르는 순서만 무작위로 (판정 없음)", "",
              f"- '차트가 좋아 보임'은 규칙으로 옮길 수 없다. 같은 날 후보가 여러 개일 때 ★ 순 대신 무작위로 고른 {len(SEEDS)}번.", "",
              "| 방식 | ★ 순 | 무작위 가운데 | 최소 ~ 최대 | SPY 넘은 것 |", "|---|---|---|---|---|"]
    for tid in TESTS:
        x = t.loc[tid]
        lines.append(f"| {tid} | {x['Sharpe']:.2f} | {x['무작위 가운데']:.2f} | {x['무작위 최소']:.2f} ~ {x['무작위 최대']:.2f} | "
                     f"{int(x['무작위 SPY 넘음'])}/{len(SEEDS)} |")
    yr = pd.DataFrame({k: _yearly(v) for k, v in yearly.items()})
    yr = yr[[*TESTS, "SPY", "PT-1"]]
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    lines += ["", "## 한계", "",
              "- 사용자가 산 종목 여럿은 S&P 500 밖(나스닥·NYSE)이지만, 생존 편향 없는 자료가 S&P 500 뿐이라 그날 S&P 500 구성 종목만 "
              "봤다. 상장폐지 종목은 가격이 없어 빠진다.",
              "- 헬스체크는 과거 시점 값이 없어 넣지 못했다 (yfinance 는 지금 값만 준다). '차트가 좋아 보임'(사람 눈)은 규칙으로 못 옮겨서 "
              "무작위 순서로 운의 폭만 봤다.",
              "- 매도 방식은 사용자가 아직 정하지 않아 세 가지를 가정했다. 실제로 다르게 팔면 결과도 다르다.",
              "- 개발 구간은 지금까지 많은 시도에 쓰였다. 홀드아웃은 쓰지 않았다.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {REPORT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--pool", action="store_true", help="조건표만 만들고 건수만 (성과 계산 없음)")
    args = ap.parse_args()
    if args.register:
        register()
    elif args.pool:
        pool = build_pool()
        for sel in SELECTIONS:
            el = eligible(pool, sel)
            print(sel, SELECTIONS[sel], f"종목-일 {len(el):,} · 하루 평균 {el.groupby('date').size().mean():.1f}")
    else:
        run()


if __name__ == "__main__":
    main()
