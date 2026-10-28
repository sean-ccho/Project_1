#!/usr/bin/env python3
"""ISB vs PT-1 (isb_vs_pt1_v1): TradingView ISB(v9.4) 로 PT-1 종목의 사고팔 시점을 잡거나, ISB 만으로 사면 PT-1 보다 나은가.

계기: 사용자가 PT-1 메일의 '오늘의 후보'(CCS 상위 5종목)를 TradingView ISB(일봉·주봉, 기본값)로 보고 사고판다.

사전 등록 (TRIAL_LOG family isb_vs_pt1_v1, 8건 — 결과 계산 전에 --register):
  D/W = ISB 일봉 / 주봉 (scripts/isb_engine.py: Pine 기본값, 주기별 Auto)
  *-full   매수: [PT-1 실제 선택] + [나머지 top5 를 CCS 순] 중 ISB 가 다음 시가에 보유인 첫 종목 (없으면 그날 안 삼)
           매도: ISB 만 — 다음 시가에 ISB 보유가 아니면 전량, TRIM·TOP·T1 은 원래 주식 수의 33% · 최대 2회.
           PT-1 손절·익절·트레일링·교체는 끔
  *-entry  매수는 full 과 같고 매도는 PT-1 규칙
  *-exit   PT-1 매수·매도 그대로 + ISB 매도 신호 (ISB 보유 중 EXIT · 미보유 ▼BREAKDOWN 전량,
           TRIM·TOP · ◆TOP(Stage 2·3) · 주봉 T1 33%). 같은 날 우선순위: ISB 전량 > PT-1 매도 > ISB 트림
  *-solo   그날 PIT S&P 500 구성 종목(실제 거래된 봉) 전체에 ISB: 보유 종목만 (1/N)×남은 비율, 나머지 현금.
           기준 = 같은 함수에 '전 종목 보유'를 넣은 동일가중 매수보유 (시가→시가, 비중 변화 × 0.1%)
  기간·엔진·설정은 PT-1 기준선과 같다 (validate_upside_walkforward.RUN_KW, 2016-01-04 ~ 2025-09-30, $5,000, 3슬롯).
  홀드아웃(2025-10-01 ~)은 쓰지 않는다. 오버레이는 src/ 를 고치지 않고 paper_trading.backtest 의 이름만 실행 중에 바꿔 끼운다.
판정:
  채택 후보 = 기준 대비 verdict '채택 후보' (ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내)
              · p_개선아님 ≤ 0.05/8  (paired_bootstrap n_boot 10,000)
  관찰     = 그 밖에 ΔSharpe 95% 하한 > 0
  참고     = SPY·PT-1 대비, ISB 단독 3종목 계좌(시드 20), 무작위 게이트 위약(시드 10), ISB 거래 단위 통계 (판정에 안 씀)
확인: BASE 가 저장된 pt1_base_dev_equity 와 같아야 하고, 래퍼를 모두 끼운 통과 모드가 BASE 와 자본곡선·거래까지 같아야 한다.
     둘 중 하나라도 다르면 결과를 쓰지 않고 멈춘다.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/isb_vs_pt1.py --check          # BASE 재현 + 래퍼 투명성 (ISB 성과 계산 안 함)
  PYTHONPATH=.:src .venv/bin/python scripts/isb_vs_pt1.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/isb_vs_pt1.py --run [--workers 3]
산출: tier3/ISB_VS_PT1.md · tier3/isb_vs_pt1.csv · data/research/isb_vs_pt1_runs.parquet (git 제외)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import resource
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_engine as isb  # noqa: E402
import paper_trading.backtest as bt  # noqa: E402
from data.fetch import _ohlcv_cache_path, chart_prices_path, load_chart_prices  # noqa: E402
from paper_trading import candidate_selector as cs  # noqa: E402
from paper_trading.config_override import config_overrides  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, paired_bootstrap, sharpe, verdict  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from research_utils import (COST_PER_SIDE, DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv,  # noqa: E402
                            log_trials, total_trials)
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "isb_vs_pt1_v1"
STEP = "stage12"
OVERLAYS = ("D-full", "W-full", "D-entry", "W-entry", "D-exit", "W-exit")
SOLOS = ("D-solo", "W-solo")
TESTS = OVERLAYS + SOLOS
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
N_BOOT = 10_000
N_BOOT_REF = 1_000
SOLO3_SEEDS = tuple(range(20))
PLACEBO_SEEDS = tuple(range(10))
TOP5 = 5
TRIM_FRAC = 0.33
TRIM_MIN_LEFT = 0.15
TRIM_MAX = 2
TRIM_TAG = "ISB트림·"
SELL_TAG = "ISB매도"
START = pd.Timestamp(RUN_KW["start_date"])
BASE_SAVED = RESEARCH_DIR / "pt1_base_dev_equity.parquet"
ETF_OHLCV = RESEARCH_DIR / "etf_ohlcv.parquet"
PINNED = RESEARCH_DIR / "isb_pit_tickers_13y.json"
SEC_ANNUAL = RESEARCH_DIR / "sec_fund_annual.parquet"
MEMBERSHIP = ROOT / "data" / "universe" / "sp500_membership.csv"
RUNS_PATH = RESEARCH_DIR / "isb_vs_pt1_runs.parquet"
OUT_CSV = TIER3_DIR / "isb_vs_pt1.csv"
REPORT = TIER3_DIR / "ISB_VS_PT1.md"
REASONS = ("ISB매도", "목표가", "시간익절", "트레일링", "손절", "장기보유", "교체", "기간종료")
NAMES = {
    "BASE": "PT-1 그대로 (기준)",
    "D-full": "일봉 ISB로 매수·매도", "W-full": "주봉 ISB로 매수·매도",
    "D-entry": "일봉 ISB 매수 확인만", "W-entry": "주봉 ISB 매수 확인만",
    "D-exit": "일봉 ISB 매도 신호 추가", "W-exit": "주봉 ISB 매도 신호 추가",
    "D-solo": "일봉 ISB 단독 (전 종목)", "W-solo": "주봉 ISB 단독 (전 종목)",
}


def spec() -> str:
    """사전 등록 문구 (TRIAL_LOG 등록 행)."""
    return (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 백테스트 엔진·13y PIT S&P 500·$5,000·3슬롯·매일 리밸런스, "
            f"ISB.pine sha256 {isb._sha(isb.PINE_PATH)[:12]} 기본값 (일봉 Trend Hold+TOP·S2·시장필터 / 주봉 Swing v9.3), "
            "ISB 가격 = TradingView 기본(ADJ 꺼짐, 배당 조정 안 함 · 2026-10-12 사용자 차트 대조), 좀비 필터 통과] *-full = 메일 top5(PT-1 선택 먼저) 중 ISB 다음 시가 보유인 첫 종목 매수(없으면 안 삼) + ISB 매도만"
            "(보유 끝 전량, TRIM·TOP·T1 33%·최대 2회, PT-1 매도·교체 끔); *-entry = 같은 매수 + PT-1 매도; "
            "*-exit = PT-1 그대로 + ISB 매도 신호(EXIT·BREAKDOWN 전량, TRIM·◆TOP·T1 33%, ISB 전량>PT-1>트림); "
            "*-solo = PIT 구성 종목 ISB 보유분 (1/N)×남은 비율·나머지 현금 vs 같은 함수 동일가중 매수보유(시가→시가, 0.1%). "
            f"채택 = 기준 대비 verdict 채택 후보(ΔSharpe 95% 하한>0 · 3구간 중 2+ · MDD 2%p 이내) · p ≤ 0.05/{N_TESTS} "
            f"(부트스트랩 {N_BOOT:,}회). 참고 = SPY·PT-1 대비, 3종목 계좌 시드 {len(SOLO3_SEEDS)}, "
            f"무작위 게이트 위약 시드 {len(PLACEBO_SEEDS)}, ISB 거래 통계.")


# ============================================================
# 순수 판단 함수 (단위 테스트 대상)
# ============================================================
def pick_gated(first: dict[str, Any], top5: list[dict[str, Any]],
               ok: Callable[[str], bool]) -> tuple[dict[str, Any] | None, int, int]:
    """[PT-1 실제 선택] + [나머지 top5 를 CCS 순] 중 ok(ticker) 인 첫 항목. Returns (항목 또는 None, 평가 수, 통과 수)."""
    order = [first] + [s for s in top5[:TOP5] if s["ticker"] != first["ticker"]]
    evaluated = 0
    for item in order:
        evaluated += 1
        if ok(str(item["ticker"])):
            return item, evaluated, 1
    return None, evaluated, 0


def trim_allowed(count: int, shares: float, orig: float) -> bool:
    """ISB canTrim: 최대 2회, 트림 뒤 남는 비율 ≥ 15%."""
    return count < TRIM_MAX and shares - orig * TRIM_FRAC >= orig * TRIM_MIN_LEFT - 1e-9


def sell_decision(mode: str, pt1: Callable[[], tuple[bool, str]], hold_next: bool, exit_full: bool,
                  trim: bool, trim_ok: bool) -> tuple[bool, str]:
    """매도 결정. mode: pt1 | isb(full: 상태 기반) | both(exit 추가: 이벤트 기반, ISB 전량 > PT-1 > ISB 트림)."""
    if mode == "pt1":
        return pt1()
    if mode == "isb":
        if not hold_next:
            return True, f"{SELL_TAG}(보유 끝)"
        return (True, f"{TRIM_TAG}TOP") if trim and trim_ok else (False, "")
    if exit_full:
        return True, f"{SELL_TAG}(EXIT·BREAKDOWN)"
    res = pt1()
    if res[0]:
        return res
    return (True, f"{TRIM_TAG}TOP") if trim and trim_ok else (False, "")


def weighted_returns(w: pd.DataFrame, r: pd.DataFrame, cost: float = COST_PER_SIDE) -> pd.Series:
    """달력시간 포트폴리오: t행 비중 × (t 시가 → t+1 시가) − 비중 변화 × 편도 비용. 남는 비중은 현금(수익 0)."""
    gross = (w * r.fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    return gross - turnover * cost


def judge(stats: dict[str, float], v: str) -> str:
    """사전 등록 판정: 채택 후보 / 관찰 / 탈락."""
    if v == "채택 후보" and stats and stats["p_개선아님"] <= P_BONF:
        return "채택 후보"
    if stats and stats["ΔSharpe_하한"] > 0:
        return "관찰"
    return "탈락"


def stable_uniform(*parts: Any) -> float:
    """시드·날짜·종목으로 정해지는 [0, 1) 난수 (프로세스마다 같다)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) / 2 ** 32


# ============================================================
# ISB 조회 표
# ============================================================
class IsbBook:
    """ISB 일봉 날짜 이벤트(isb_engine.build_events)를 (종목, 날짜) 로 찾는 표."""

    FIELDS = {"hold_next": "bool", "exit_full": "bool", "trim_ev": "bool", "t1_fill": "float",
              "buy_ev": "bool", "grade": "int"}

    def __init__(self, events: pd.DataFrame) -> None:
        ev = events[events["date"] >= START - pd.Timedelta(days=10)]
        self.dates = pd.DatetimeIndex(sorted(ev["date"].unique()))
        self.tickers = sorted(ev["ticker"].unique())
        self.row = {f"{d:%Y-%m-%d}": i for i, d in enumerate(self.dates)}
        self.col = {t: j for j, t in enumerate(self.tickers)}
        self.mat: dict[str, np.ndarray] = {}
        for name, kind in self.FIELDS.items():
            w = ev.pivot(index="date", columns="ticker", values=name).reindex(index=self.dates, columns=self.tickers)
            arr = w.to_numpy(dtype=float, na_value=np.nan)
            self.mat[name] = (np.nan_to_num(arr) > 0.5) if kind == "bool" else (
                np.nan_to_num(arr).astype(int) if kind == "int" else arr)

    def get(self, name: str, ticker: str, day: str, default: Any) -> Any:
        i, j = self.row.get(day), self.col.get(ticker)
        if i is None or j is None:
            return default
        return self.mat[name][i, j]


_BOOKS: dict[str, IsbBook] = {}
_PX: dict[str, Any] = {}


def _book(tf: str) -> IsbBook:
    if tf not in _BOOKS:
        _BOOKS[tf] = IsbBook(pd.read_parquet(RESEARCH_DIR / f"isb_events_{tf}.parquet"))
    return _BOOKS[tf]


def _engine_prices() -> dict[str, Any]:
    """엔진이 읽는 가격 캐시 그대로 (날짜 목록 · 시가 · 저가) — 장중 T1 날짜 추정 확인용."""
    if not _PX:
        raw = pd.read_parquet(isb.PRICE_CACHE)
        dates = [f"{d:%Y-%m-%d}" for d in pd.to_datetime(raw.index)]
        _PX["next"] = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
        _PX["start"] = next(d for d in dates if d >= RUN_KW["start_date"])
        _PX["opens"] = raw.xs("Open", axis=1, level=1)
        _PX["lows"] = raw.xs("Low", axis=1, level=1)
    return _PX


# ============================================================
# 엔진 실행 (paper_trading.backtest 이름 바꿔 끼우기)
# ============================================================
_ORIG = {name: getattr(bt, name) for name in (
    "select_top_candidates", "check_sell_conditions", "_close_position", "intraday_stop_exit",
    "_compute_ranked_snapshot", "_record_equity", "slot_allocation", "fetch_ohlcv", "_pit_universe_tickers",
    "fetch_chart_prices")}
_SNAP: dict[pd.Timestamp, pd.DataFrame] | None = None   # 실행 간 날짜별 ranked 후보표 (enable_snapshot_cache 로 켬)


def enable_snapshot_cache() -> None:
    """같은 프로세스의 다음 실행부터 날짜별 ranked 후보표를 재사용한다 (엔진 시간의 약 80%).

    후보표는 날짜·가격·설정만으로 정해지고 보유 종목과 무관하다. 저장·반환 모두 복사본이라 실행끼리 섞이지 않는다.
    래퍼를 끼운 실행(BASE 제외)에만 적용된다.
    """
    global _SNAP
    if _SNAP is None:
        _SNAP = {}


@dataclass(frozen=True)
class Overlay:
    """엔진 실행 한 번의 설정."""

    key: str
    tf: str | None = None       # ISB 주기 "D"/"W" (None = ISB 안 씀)
    gate: str = "none"           # none | isb | random | probe(투명성: 조회만)
    sells: str = "pt1"           # pt1 | isb(상태 기반) | both(이벤트 기반 추가)
    selector: str = "pt1"        # pt1 | solo(ISB BUY 를 등급순, 동점은 시드 무작위)
    seed: int = 0
    pass_rate: float = float("nan")
    probe: bool = False          # 투명성 확인: 래퍼를 모두 돌리되 결정은 원래대로
    t1_mode: str = ""            # 라벨 규칙(isb_label_search): off · trim(⅓) · full(전량). "" = 아래 기본 판단

    @property
    def t1(self) -> bool:
        """주봉 T1 장중 체결 — ISB 매도를 쓰는 주봉 실행만 (t1_mode 를 주면 그 값)."""
        if self.t1_mode:
            return self.t1_mode != "off" and not self.probe
        return self.tf == "W" and not self.probe and (self.sells != "pt1" or self.selector == "solo")


@dataclass
class RunCtx:
    ov: Overlay
    book: IsbBook | None
    date: pd.Timestamp | None = None
    last_recorded: str | None = None
    gate_eval: int = 0
    gate_pass: int = 0
    gate_days: int = 0
    gate_empty: int = 0
    max_weight: float = 0.0
    intraday_checks: int = 0
    orig_shares: dict[tuple[str, str], float] = field(default_factory=dict)
    trim_count: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    trims: list[dict[str, Any]] = field(default_factory=list)

    def day(self) -> str:
        return f"{self.date:%Y-%m-%d}"

    def current_day(self) -> str:
        """장중 단계(엔진 0단계)의 오늘 = 직전에 자본을 기록한 날의 다음 거래일."""
        px = _engine_prices()
        return px["start"] if self.last_recorded is None else px["next"][self.last_recorded]


@dataclass
class RunResult:
    key: str
    equity: pd.Series
    trades: list[dict[str, Any]]
    summary: dict[str, Any]
    trims: list[dict[str, Any]]
    gate_eval: int
    gate_pass: int
    gate_days: int
    gate_empty: int
    max_weight: float
    intraday_checks: int
    peak_rss_gb: float


def _pinned_tickers() -> list[str]:
    return json.loads(PINNED.read_text())["tickers"]


def ensure_pinned() -> None:
    """오늘의 13y PIT 종목 목록(엔진 순서 그대로)을 저장 — 2026-11-13 이후에도 같은 가격·피처 캐시 키가 나오게."""
    if PINNED.exists():
        return
    mem = Membership(load_membership(MEMBERSHIP))
    tickers = _ORIG["_pit_universe_tickers"](mem, RUN_KW["period"])
    PINNED.write_text(json.dumps({"made": str(date.today()), "period": RUN_KW["period"], "tickers": tickers}))
    print(f"[isb] 종목 목록 고정: {PINNED.name} ({len(tickers)}개)", flush=True)


def _strict_fetch(tickers: list[str], period: str = "1y", force_download: bool = False) -> pd.DataFrame:
    """디스크 가격 캐시만 읽는다. 없으면 다운로드하지 않고 멈춘다 (기준선과 다른 가격이 섞이지 않게)."""
    path = _ohlcv_cache_path(tickers, period)
    if not path.exists():
        raise SystemExit(f"가격 캐시 {path.name} 없음 — 다운로드하면 기준선과 달라지므로 멈춘다")
    return pd.read_parquet(path)


def _strict_chart_fetch(tickers: list[str], *args: Any, **kw: Any) -> dict[str, Any]:
    """얼린 차트 가격 파일만 읽는다 (패턴 · 교차 라벨용, 분할만 반영 · 상장 이후 전체). 없으면 받지 않고 멈춘다.

    파일은 scripts/build_chart_prices.py 가 한 번 만든다 (2026-10-27). 피처를 새로 계산할 때만 불린다.
    """
    path = chart_prices_path(tickers)
    if not path.exists():
        raise SystemExit(f"차트 가격 파일 {path.name} 없음 — scripts/build_chart_prices.py 로 먼저 얼린다")
    return load_chart_prices(path)


def apply_trim(ctx: RunCtx, pos: Any, exit_price: float, exit_date: str, kind: str) -> float:
    """ISB 트림: 그 포지션 원래 주식 수의 33% 를 판다 (포지션은 남김). Returns 매도 대금 (비용 차감, 안 팔면 0)."""
    key = (pos.ticker, pos.entry_date)
    orig = ctx.orig_shares.setdefault(key, pos.shares)
    if not trim_allowed(ctx.trim_count[key], pos.shares, orig):
        return 0.0
    sold = orig * TRIM_FRAC
    pos.shares -= sold
    ctx.trim_count[key] += 1
    ctx.trims.append({"ticker": pos.ticker, "entry_date": pos.entry_date, "date": exit_date, "price": exit_price,
                      "shares": sold, "kind": kind})
    return sold * exit_price * (1 - bt.BACKTEST_COST_PER_SIDE)


def _solo_select(ctx: RunCtx, df: pd.DataFrame, holdings: list[dict[str, Any]],
                 k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """3종목 계좌 선택기: 같은 후보 풀에서 그날 ISB BUY 종목을 등급순 (동점은 시드 무작위)."""
    debug: dict[str, Any] = {"regime": cs.detect_market_regime(df)}
    day, held = ctx.day(), {h["ticker"] for h in holdings}
    rows = []
    for idx, tk in df["티커"].items():
        tk = str(tk)
        if tk == "SPY" or tk in held or not ctx.book.get("buy_ev", tk, day, False):
            continue
        g = int(ctx.book.get("grade", tk, day, 0))
        rows.append((-g, stable_uniform(ctx.ov.seed, day, tk), idx, tk, g))
    rows.sort()
    picks = []
    for _, _, idx, tk, g in rows[:max(1, k)]:
        r = df.loc[idx]
        picks.append({"ticker": tk, "entry_price": float(r.get("close", np.nan)), "strategy": "ISB",
                      "star_rating": "", "ccs_score": float(g), "sector": str(r.get("섹터", "")), "ccs_breakdown": {}})
    return picks, debug


@contextlib.contextmanager
def _patched(ctx: RunCtx) -> Iterator[None]:
    """엔진 이름 바꿔 끼우기. BASE 도 가격(엄격 읽기)·종목 목록(고정)은 같은 방식으로."""
    ov, pinned = ctx.ov, _pinned_tickers()
    bt.fetch_ohlcv = _strict_fetch
    bt.fetch_chart_prices = _strict_chart_fetch
    bt._pit_universe_tickers = lambda membership, period: list(pinned)
    overrides: dict[str, Any] = {}
    if ov.key != "BASE":
        px = _engine_prices()

        def snapshot(*args: Any, **kw: Any) -> pd.DataFrame:
            ctx.date = pd.Timestamp(args[1])
            if _SNAP is None:
                return _ORIG["_compute_ranked_snapshot"](*args, **kw)
            hit = _SNAP.get(ctx.date)
            if hit is None:
                hit = _ORIG["_compute_ranked_snapshot"](*args, **kw)
                _SNAP[ctx.date] = hit.copy()
                return hit
            return hit.copy()

        def record(equity_points: list, positions: list, closes: pd.DataFrame, date_ts: pd.Timestamp,
                   today_str: str, cash: float = 0.0, use_capital: bool = False) -> None:
            ctx.last_recorded = today_str
            return _ORIG["_record_equity"](equity_points, positions, closes, date_ts, today_str,
                                           cash=cash, use_capital=use_capital)

        def alloc(cash: float, equity: float, max_positions: int, empty_slots: int,
                  ranked_df: pd.DataFrame | None, ticker: str) -> float:
            a = _ORIG["slot_allocation"](cash, equity, max_positions, empty_slots, ranked_df, ticker)
            if equity > 0:
                ctx.max_weight = max(ctx.max_weight, a / equity)
            return a

        def gate_ok(ticker: str) -> bool:
            if ov.gate == "random":
                return stable_uniform(ov.seed, ctx.day(), ticker) < ov.pass_rate
            return bool(ctx.book.get("hold_next", ticker, ctx.day(), False))

        def select(df: pd.DataFrame, holdings: list[dict[str, Any]], k: int = 1):
            if ov.selector == "solo":
                return _solo_select(ctx, df, holdings, k)
            picks, debug = _ORIG["select_top_candidates"](df, holdings, k=k)
            if ov.gate == "none" or not picks:
                return picks, debug
            item, n_eval, n_pass = pick_gated(picks[0], debug.get("top5", []), gate_ok)
            ctx.gate_days += 1
            ctx.gate_eval += n_eval
            ctx.gate_pass += n_pass
            if ov.gate == "probe":
                return picks, debug
            if item is None:
                ctx.gate_empty += 1
                return [], debug
            if item is picks[0]:
                return picks, debug
            return [cs._candidate_from_score(item, df.loc[item["idx"]])], debug

        def sell(pos_dict: dict[str, Any], cur: float, today_str: str) -> tuple[bool, str]:
            tk = pos_dict["ticker"]
            key = (tk, pos_dict["entry_date"])
            pt1 = lambda: _ORIG["check_sell_conditions"](pos_dict, cur, today_str)  # noqa: E731
            if ov.sells == "pt1":
                if ov.probe:
                    ctx.book.get("exit_full", tk, today_str, False)
                return pt1()
            b = ctx.book
            shares = float(pos_dict["shares"])
            return sell_decision(
                ov.sells, pt1,
                hold_next=bool(b.get("hold_next", tk, today_str, False)),
                exit_full=bool(b.get("exit_full", tk, today_str, False)),
                trim=bool(b.get("trim_ev", tk, today_str, False)),
                trim_ok=trim_allowed(ctx.trim_count[key], shares, ctx.orig_shares.get(key, shares)))

        def close(positions: list, trades: list, ticker: str, exit_price: float, exit_date: str, reason: str):
            if not reason.startswith(TRIM_TAG):
                return _ORIG["_close_position"](positions, trades, ticker, exit_price, exit_date, reason)
            pos = next((p for p in positions if p.ticker == ticker), None)
            if pos is None:
                return None, 0.0
            return None, apply_trim(ctx, pos, exit_price, exit_date, reason[len(TRIM_TAG):])

        def intraday(pos_dict: dict[str, Any], day_open: float, day_low: float):
            day, tk = ctx.current_day(), pos_dict["ticker"]
            ts = pd.Timestamp(day)
            exp_open, exp_low = float(px["opens"].at[ts, tk]), float(px["lows"].at[ts, tk])
            if not (np.isclose(exp_open, day_open, equal_nan=True) and np.isclose(exp_low, day_low, equal_nan=True)):
                raise RuntimeError(f"장중 날짜 추정이 어긋남: {tk} {day} 시가 {day_open} vs {exp_open}")
            ctx.intraday_checks += 1
            if not ov.t1:
                return None
            key = (tk, pos_dict["entry_date"])
            fill = ctx.book.get("t1_fill", tk, day, np.nan)
            if fill == fill and ov.t1_mode == "full":
                return float(fill), f"{SELL_TAG}(T1)"
            shares = float(pos_dict["shares"])
            if fill == fill and trim_allowed(ctx.trim_count[key], shares, ctx.orig_shares.get(key, shares)):
                return float(fill), f"{TRIM_TAG}T1"
            return None

        bt._compute_ranked_snapshot = snapshot
        bt._record_equity = record
        bt.slot_allocation = alloc
        bt.select_top_candidates = select
        bt.check_sell_conditions = sell
        bt._close_position = close
        bt.intraday_stop_exit = intraday
        if ov.sells == "isb" or ov.selector == "solo":
            overrides["PT1_REPLACE_ENABLED"] = False
        if ov.t1 or ov.probe:
            overrides["PT1_STOP_INTRADAY"] = True
    try:
        with config_overrides(overrides):
            yield
    finally:
        for name, fn in _ORIG.items():
            setattr(bt, name, fn)


def run_overlay(ov: Overlay, quiet: bool = True, book: Any = None) -> RunResult:
    """엔진 한 번 실행 (워커 프로세스에서도 호출). book 을 주면 그 표(get(이름, 종목, 날짜, 기본값))로 ISB 를 찾는다."""
    ctx = RunCtx(ov, book if book is not None else (_book(ov.tf) if ov.tf else None))
    sink = io.StringIO()
    with _patched(ctx), (contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()):
        result = bt.run_paper_trading_backtest(**RUN_KW)
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3  # macOS: 바이트
    return RunResult(ov.key, eq, result["trades"], result["summary"], ctx.trims, ctx.gate_eval, ctx.gate_pass,
                     ctx.gate_days, ctx.gate_empty, ctx.max_weight, ctx.intraday_checks, rss)


def run_many(specs: list[Overlay], workers: int, label: str) -> dict[str, RunResult]:
    """여러 실행을 프로세스 풀로 (워커마다 가격·피처 캐시를 따로 읽는다)."""
    print(f"===== 엔진 {label}: {len(specs)}회 (동시 {workers})", flush=True)
    out: dict[str, RunResult] = {}
    if workers <= 1:
        for k, ov in enumerate(specs, 1):
            out[ov.key] = run_overlay(ov)
            _progress(label, out[ov.key], k, len(specs))
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_overlay, ov): ov.key for ov in specs}
        for k, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            out[res.key] = res
            _progress(label, res, k, len(specs))
    return out


def _progress(label: str, res: RunResult, k: int, n: int) -> None:
    """진행 표시 (성과 숫자는 찍지 않는다 — 확인 단계에서 ISB 결과를 미리 보지 않게)."""
    print(f"  [{label}] {res.key} 완료 · 거래 {len(res.trades)} · 트림 {len(res.trims)} · "
          f"메모리 {res.peak_rss_gb:.1f}GB [{k / n * 100:5.1f}%]", flush=True)


def overlay_specs() -> dict[str, Overlay]:
    out = {"BASE": Overlay("BASE"), "PROBE": Overlay("PROBE", tf="W", gate="probe", sells="pt1", probe=True)}
    for tf in ("D", "W"):
        out[f"{tf}-full"] = Overlay(f"{tf}-full", tf=tf, gate="isb", sells="isb")
        out[f"{tf}-entry"] = Overlay(f"{tf}-entry", tf=tf, gate="isb", sells="pt1")
        out[f"{tf}-exit"] = Overlay(f"{tf}-exit", tf=tf, gate="none", sells="both")
    return out


def solo3_specs() -> list[Overlay]:
    return [Overlay(f"{tf}-solo3-s{s}", tf=tf, selector="solo", sells="isb", seed=s)
            for tf in ("D", "W") for s in SOLO3_SEEDS]


def placebo_specs(rates: dict[str, float]) -> list[Overlay]:
    return [Overlay(f"{tf}-entry-rand-s{s}", tf=tf, gate="random", sells="pt1", seed=s, pass_rate=rates[tf])
            for tf in ("D", "W") for s in PLACEBO_SEEDS]


# ============================================================
# 확인 (BASE 재현 · 래퍼 투명성)
# ============================================================
def _trade_key(t: dict[str, Any]) -> tuple:
    return (t["ticker"], t["entry_date"], t["exit_date"], round(float(t["exit_price"]), 6), t["exit_reason"],
            round(float(t["shares"]), 9))


def check_base(res: RunResult) -> str | None:
    """저장된 기준선과 같은지 (다르면 사유 문자열)."""
    saved = pd.read_parquet(BASE_SAVED).iloc[:, 0]
    saved.index = pd.to_datetime(saved.index)
    eq = res.equity
    common = saved.index.intersection(eq.index)
    max_rel = float((eq.loc[common] / saved.loc[common] - 1).abs().max()) if len(common) else float("inf")
    if len(common) != len(saved) or max_rel > 1e-9:
        return f"BASE 가 저장된 기준선과 다르다 (겹친 날 {len(common)}/{len(saved)}, 최대 차이 {max_rel:.2e})"
    print(f"[isb] BASE = 저장된 기준선 ({len(common)}일, 최대 차이 {max_rel:.1e})", flush=True)
    return None


def check_probe(base: RunResult, probe: RunResult) -> str | None:
    """래퍼를 모두 끼운 통과 모드가 BASE 와 자본곡선·거래 목록까지 같은지."""
    a, b = base.equity, probe.equity
    if not a.index.equals(b.index) or float((a / b - 1).abs().max()) > 1e-12:
        return "통과 모드 자본곡선이 BASE 와 다르다"
    if [_trade_key(t) for t in base.trades] != [_trade_key(t) for t in probe.trades]:
        return "통과 모드 거래 목록이 BASE 와 다르다"
    if probe.gate_eval == 0 or probe.intraday_checks == 0:
        return f"통과 모드에서 래퍼가 안 불렸다 (게이트 {probe.gate_eval}, 장중 {probe.intraday_checks})"
    print(f"[isb] 통과 모드 = BASE (거래 {len(base.trades)}건 동일 · 게이트 조회 {probe.gate_eval} · "
          f"장중 날짜 확인 {probe.intraday_checks})", flush=True)
    return None


def check() -> None:
    """BASE 재현 + 래퍼 투명성만 (ISB 성과는 계산하지 않음)."""
    ensure_pinned()
    isb.build_events("W")
    specs = overlay_specs()
    res = run_many([specs["BASE"], specs["PROBE"]], workers=2, label="확인")
    errors = [e for e in (check_base(res["BASE"]), check_probe(res["BASE"], res["PROBE"])) if e]
    if errors:
        raise SystemExit("확인 실패: " + " / ".join(errors))
    print("확인 통과: BASE 재현 · 래퍼 투명성", flush=True)


# ============================================================
# 단독 (달력시간) · 통계
# ============================================================
def solo_returns(events: pd.DataFrame, raw: pd.DataFrame, mem: Membership) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ISB 단독 vs 동일가중 매수보유 (같은 함수). Returns (ISB 일간 수익, 동일가중 일간 수익, ISB 주식 비중 합)."""
    from idea_batch_research import member_mask
    opens = raw.xs("Open", axis=1, level="Price").sort_index()
    r = opens.shift(-1) / opens - 1.0
    idx = r.loc[START:DEV_END].index[:-1]
    cols = pd.Index(sorted(set(events["ticker"]) & set(r.columns)))
    vol = raw.xs("Volume", axis=1, level="Price").reindex(index=idx, columns=cols)
    hi = raw.xs("High", axis=1, level="Price").reindex(index=idx, columns=cols)
    lo = raw.xs("Low", axis=1, level="Price").reindex(index=idx, columns=cols)
    r = r.reindex(index=idx, columns=cols)
    universe = member_mask(idx, cols, mem) & r.notna() & (vol > 0) & (hi > lo)
    n = universe.sum(axis=1).replace(0, np.nan)
    held = (events.assign(w=events["pos"].astype(float) * events["rem"].astype(float))
            .pivot(index="date", columns="ticker", values="w").reindex(index=idx, columns=cols).fillna(0.0))
    w_isb = held.where(universe, 0.0).div(n, axis=0).fillna(0.0)
    w_ew = universe.astype(float).div(n, axis=0).fillna(0.0)
    return weighted_returns(w_isb, r), weighted_returns(w_ew, r), w_isb.sum(axis=1)


def _cmp(a: pd.Series, b: pd.Series, n_boot: int) -> tuple[dict[str, float], pd.DataFrame]:
    df = pd.concat({"a": a, "b": b}, axis=1, sort=True).dropna()  # 날짜순 (부트스트랩 블록이 연속 거래일이어야 함)
    return paired_bootstrap(df["a"].tolist(), df["b"].tolist(), n_boot=n_boot), df


def blended_trades(res: RunResult) -> list[float]:
    """포지션 단위 수익률 (트림 포함, 비용 포함)."""
    c = bt.BACKTEST_COST_PER_SIDE
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in res.trims:
        by_key[(t["ticker"], t["entry_date"])].append(t)
    out = []
    for t in res.trades:
        trims = by_key.get((t["ticker"], t["entry_date"]), [])
        rem = float(t["shares"])
        orig = rem + sum(x["shares"] for x in trims)
        if orig <= 0:
            out.append(float(t["return_pct"]))
            continue
        proceeds = sum(x["shares"] * x["price"] for x in trims) * (1 - c) + rem * float(t["exit_price"]) * (1 - c)
        out.append(proceeds / (orig * float(t["entry_price"]) * (1 + c)) - 1.0)
    return out


def exit_mix(trades: list[dict[str, Any]]) -> str:
    groups: dict[str, int] = defaultdict(int)
    for t in trades:
        reason = str(t.get("exit_reason") or "")
        groups[next((k for k in REASONS if reason.startswith(k)), "기타")] += 1
    return " · ".join(f"{k} {n}" for k, n in sorted(groups.items(), key=lambda x: -x[1]))


def isb_trade_stats(trades: pd.DataFrame, mem: Membership) -> dict[str, Any]:
    """TradingView 대시보드 방식 거래 통계 (신호일에 S&P 500 구성 종목, 2016-01-04 ~ 개발 끝, 청산된 거래)."""
    t = trades[(trades["signal_date"] >= START) & (trades["signal_date"] <= DEV_END) & (trades["exit_bar"] >= 0)].copy()
    t = t[[tk in mem.on(f"{d:%Y-%m-%d}") for tk, d in zip(t["ticker"], t["signal_date"])]]
    ret = t["ret"]
    gross_loss = -ret[ret <= 0].sum()
    by_type = {k: f"{int((g['ret'] > 0).sum())}/{len(g)}" for k, g in t.groupby("sig_type")}
    return {"n": len(t), "win": float((ret > 0).mean()) if len(t) else np.nan,
            "pf": float(ret[ret > 0].sum() / gross_loss) if gross_loss > 0 else np.nan,
            "exp_r": float(t["R"].mean()) if len(t) else np.nan, "hold": float(t["held_bars"].mean()) if len(t) else np.nan,
            "mean_ret": float(ret.mean()) if len(t) else np.nan, "by_type": by_type}


def zombie_upper_bound(trades: dict[str, pd.DataFrame], mem: Membership) -> dict[str, tuple[int, int]]:
    """좀비 필터 생략 영향 상한: 직전 10-K 가 순이익 ≤ 0 · 영업현금흐름 ≤ 0 인 종목의 ISB 매수 수 (주기별 (해당, 전체))."""
    if not SEC_ANNUAL.exists():
        return {}
    fund = pd.read_parquet(SEC_ANNUAL)
    fund["filed"] = pd.to_datetime(fund["filed"])
    fund = fund.sort_values(["티커", "filed"])
    flags: dict[str, list[tuple[pd.Timestamp, bool]]] = defaultdict(list)
    for r in fund.itertuples():
        flags[r.티커].append((r.filed, bool(r.net_income <= 0 and r.cfo <= 0)))
    out = {}
    for tf, tr in trades.items():
        t = tr[(tr["signal_date"] >= START) & (tr["signal_date"] <= DEV_END)]
        t = t[[tk in mem.on(f"{d:%Y-%m-%d}") for tk, d in zip(t["ticker"], t["signal_date"])]]
        hit = 0
        for tk, d in zip(t["ticker"], t["signal_date"]):
            last = [f for when, f in flags.get(tk, []) if when <= d]
            hit += int(bool(last) and last[-1])
        out[tf] = (hit, len(t))
    return out


def _yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def _f(v: Any, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return format(v, fmt)


# ============================================================
# 등록 · 실행 · 보고서
# ============================================================
def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def register() -> None:
    log = _log()
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"등록: {FAMILY} {N_TESTS}건 · p ≤ {P_BONF:.5f} · 누적 시도 {total_trials()}건")


def run(workers: int) -> None:
    log = _log()
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    already_logged = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    ensure_pinned()
    events, isb_trades = {}, {}
    for tf in ("D", "W"):
        events[tf], isb_trades[tf] = isb.build_events(tf)
    specs = overlay_specs()

    first = run_many([specs["BASE"], specs["PROBE"]], workers=min(2, workers), label="확인")
    errors = [e for e in (check_base(first["BASE"]), check_probe(first["BASE"], first["PROBE"])) if e]
    if errors:
        raise SystemExit("확인 실패 — 결과를 쓰지 않는다: " + " / ".join(errors))
    res = dict(first)
    res.update(run_many([specs[k] for k in OVERLAYS] + solo3_specs(), workers, "오버레이·3종목"))
    rates = {tf: res[f"{tf}-entry"].gate_pass / max(1, res[f"{tf}-entry"].gate_eval) for tf in ("D", "W")}
    print(f"[isb] 게이트 통과율 (후보 1개당): 일봉 {rates['D']:.1%} · 주봉 {rates['W']:.1%}", flush=True)
    res.update(run_many(placebo_specs(rates), workers, "무작위 게이트"))
    pd.DataFrame({k: v.equity for k, v in res.items()}).to_parquet(RUNS_PATH)
    finish(res, events, isb_trades, rates, log_results=not already_logged)


def finish(res: dict[str, RunResult], events: dict[str, pd.DataFrame], isb_trades: dict[str, pd.DataFrame],
           rates: dict[str, float], log_results: bool = True, csv_path: Path = OUT_CSV,
           report_path: Path = REPORT) -> pd.DataFrame:
    """엔진 결과 → 비교·판정 · CSV · TRIAL_LOG · 보고서."""
    print("===== 비교·판정", flush=True)
    mem = Membership(load_membership(MEMBERSHIP))
    base_r = res["BASE"].equity.pct_change().dropna()
    etf = load_ohlcv(ETF_OHLCV)
    spy_c = etf["SPY"]["Close"].pct_change()
    raw = load_ohlcv(isb.PRICE_CACHE)
    spy_open = raw["SPY"]["Open"]
    spy_oo = (spy_open.shift(-1) / spy_open - 1.0)
    rows: list[dict[str, Any]] = []
    base_mdd = max_drawdown(base_r.tolist())
    rows.append({"test_id": "BASE", "이름": NAMES["BASE"], "기준": "", "Sharpe": sharpe(base_r.tolist()),
                 "CAGR": cagr(base_r.tolist()), "MDD": base_mdd, "거래수": len(res["BASE"].trades),
                 "승률": float(np.mean([t["return_pct"] > 0 for t in res["BASE"].trades])),
                 "청산이유": exit_mix(res["BASE"].trades), "판정": "기준"})
    for key in OVERLAYS:
        rr = res[key]
        r = rr.equity.pct_change().dropna()
        st, df = _cmp(r, base_r, N_BOOT)
        a, b = df["a"].tolist(), df["b"].tolist()
        v = verdict(st, max_drawdown(a), max_drawdown(b))
        st_spy, df_spy = _cmp(r, spy_c, N_BOOT_REF)
        blended = blended_trades(rr)
        rows.append({
            "test_id": key, "이름": NAMES[key], "기준": "BASE", "시작": df.index.min().date(), "끝": df.index.max().date(),
            "일수": len(df), "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a), "거래수": len(rr.trades),
            "승률": float(np.mean([x > 0 for x in blended])) if blended else np.nan, "트림": len(rr.trims),
            "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
            "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "", "verdict": v,
            "SPY_Sharpe": sharpe(df_spy["b"].tolist()), "ΔSharpe_SPY": st_spy.get("ΔSharpe"),
            "하한_SPY": st_spy.get("ΔSharpe_하한"), "상한_SPY": st_spy.get("ΔSharpe_상한"),
            "게이트통과율": rr.gate_pass / rr.gate_eval if rr.gate_eval else np.nan,
            "매수없는날": rr.gate_empty / rr.gate_days if rr.gate_days else np.nan,
            "최대비중": rr.max_weight, "청산이유": exit_mix(rr.trades), "판정": judge(st, v),
        })

    solo_series: dict[str, pd.Series] = {}
    ew_r = None
    for tf in ("D", "W"):
        key = f"{tf}-solo"
        r_isb, r_ew, wsum = solo_returns(events[tf], raw, mem)
        ew_r = r_ew
        solo_series[key] = r_isb
        st, df = _cmp(r_isb, r_ew, N_BOOT)
        a, b = df["a"].tolist(), df["b"].tolist()
        v = verdict(st, max_drawdown(a), max_drawdown(b))
        st_spy, df_spy = _cmp(r_isb, spy_oo, N_BOOT_REF)
        rows.append({
            "test_id": key, "이름": NAMES[key], "기준": "동일가중", "시작": df.index.min().date(), "끝": df.index.max().date(),
            "일수": len(df), "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a), "평균비중": float(wsum.mean()),
            "기준_Sharpe": sharpe(b), "기준_CAGR": cagr(b), "기준_MDD": max_drawdown(b),
            "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
            "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "", "verdict": v,
            "SPY_Sharpe": sharpe(df_spy["b"].tolist()), "ΔSharpe_SPY": st_spy.get("ΔSharpe"),
            "하한_SPY": st_spy.get("ΔSharpe_하한"), "상한_SPY": st_spy.get("ΔSharpe_상한"), "판정": judge(st, v),
        })
    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False)

    if not log_results:
        print("TRIAL_LOG 에 결과를 쓰지 않는다 (이미 있거나 점검 실행)")
    else:
        log_trials(STEP, FAMILY, N_TESTS, [{
            "test_id": r["test_id"], "판정": r["판정"],
            "메모": (f"Sharpe {r['Sharpe']:.2f} vs {r['기준']} "
                     f"{(table.iloc[0]['Sharpe'] if r['기준'] == 'BASE' else r.get('기준_Sharpe', np.nan)):.2f}, "
                     f"ΔSharpe {r['ΔSharpe']:+.2f} [{r['하한']:+.2f}, {r['상한']:+.2f}] p {r['p']:.4f}, 구간 {r['구간']}, "
                     f"CAGR {r['CAGR']:+.1%}, MDD {r['MDD']:.1%}")}
            for _, r in table.iterrows() if r["test_id"] in TESTS])

    refs = {
        "solo3": {tf: [(k, v) for k, v in res.items() if k.startswith(f"{tf}-solo3-")] for tf in ("D", "W")},
        "placebo": {tf: [(k, v) for k, v in res.items() if k.startswith(f"{tf}-entry-rand-")] for tf in ("D", "W")},
        "rates": rates,
        "isb_stats": {tf: isb_trade_stats(isb_trades[tf], mem) for tf in ("D", "W")},
        "zombie": zombie_upper_bound(isb_trades, mem),
    }
    series = {"BASE": base_r, **{k: res[k].equity.pct_change().dropna() for k in OVERLAYS}, **solo_series,
              "동일가중": ew_r, "SPY": spy_c.loc[base_r.index.min():base_r.index.max()].dropna()}
    write_report(table, refs, series, base_r, report_path)
    return table


def _solo3_line(tf: str, runs: list[tuple[str, RunResult]], base_r: pd.Series) -> str:
    sh = []
    for _, rr in runs:
        r = rr.equity.pct_change().dropna()
        sh.append((sharpe(r.tolist()), cagr(r.tolist()), max_drawdown(r.tolist())))
    if not sh:
        return f"| {tf} | — | | | | |"
    s = np.array(sh)
    base_sh = sharpe(base_r.tolist())
    return (f"| {'일봉' if tf == 'D' else '주봉'} | {s[:, 0].min():.2f} / {np.median(s[:, 0]):.2f} / {s[:, 0].max():.2f} | "
            f"{np.median(s[:, 1]):+.1%} | {np.median(s[:, 2]):.1%} | {int((s[:, 0] > base_sh).sum())}/{len(s)} |")


def _headline(table: pd.DataFrame) -> list[str]:
    t = table.set_index("test_id")
    ov = t.loc[list(OVERLAYS)]
    adopt = ov.index[ov["판정"] == "채택 후보"].tolist()
    best = ov["ΔSharpe"].astype(float).idxmax()
    if adopt:
        first = f"**결론: ISB 타이밍을 붙인 조합 중 {', '.join(adopt)} 가 PT-1보다 낫다는 채택 기준을 통과했다.**"
    else:
        first = ("**결론: ISB로 PT-1 메일 종목의 사고팔 시점을 잡아도 PT-1보다 나아졌다고 볼 수 없다.** "
                 f"6개 조합 모두 채택 기준에 못 미쳤다. 가장 나은 조합은 {best} {NAMES[best]} "
                 f"(ΔSharpe {t.at[best, 'ΔSharpe']:+.2f}, 95% [{t.at[best, '하한']:+.2f}, {t.at[best, '상한']:+.2f}]).")
    side = ("- 어느 쪽이 차이를 만들었나 (BASE 대비 ΔSharpe): "
            f"매수 확인만 일봉 {t.at['D-entry', 'ΔSharpe']:+.2f} · 주봉 {t.at['W-entry', 'ΔSharpe']:+.2f}, "
            f"매도 신호 추가 일봉 {t.at['D-exit', 'ΔSharpe']:+.2f} · 주봉 {t.at['W-exit', 'ΔSharpe']:+.2f}, "
            f"둘 다 일봉 {t.at['D-full', 'ΔSharpe']:+.2f} · 주봉 {t.at['W-full', 'ΔSharpe']:+.2f}.")
    solo = ("- ISB만으로 전 종목을 사고팔면 (같은 종목을 계속 들고 있는 동일가중 대비): "
            + " · ".join(f"{'일봉' if k == 'D-solo' else '주봉'} Sharpe {t.at[k, 'Sharpe']:.2f} vs {t.at[k, '기준_Sharpe']:.2f}, "
                         f"ΔSharpe {t.at[k, 'ΔSharpe']:+.2f} [{t.at[k, '하한']:+.2f}, {t.at[k, '상한']:+.2f}] → {t.at[k, '판정']}"
                         for k in SOLOS) + ".")
    return [first, "", side, solo]


def write_report(table: pd.DataFrame, refs: dict[str, Any], series: dict[str, pd.Series], base_r: pd.Series,
                 path: Path = REPORT) -> None:
    n_adopt = int((table["판정"] == "채택 후보").sum())
    n_fail = int((table["판정"] == "탈락").sum())
    n_obs = int((table["판정"] == "관찰").sum())
    t = table.set_index("test_id")
    lines = [
        f"# ISB vs PT-1 ({FAMILY}) — {RUN_KW['start_date']} ~ {RUN_KW['end_date']}",
        "",
        *_headline(table),
        "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}`, {N_TESTS}건 · p ≤ 0.05/{N_TESTS} · 부트스트랩 {N_BOOT:,}회 · "
        f"누적 시도 {total_trials()}건",
        "- 계기: 사용자가 PT-1 메일 '오늘의 후보'(상위 5종목)를 TradingView ISB(일봉·주봉, 기본값)로 보고 사고판다",
        f"- ISB: `pinescript/ISB.pine` v9.4 (sha256 `{isb._sha(isb.PINE_PATH)[:12]}`), 파이썬 포팅 `scripts/isb_engine.py`. "
        "일봉 = Trend Hold + TOP · S2 · 시장 필터 / 주봉 = Swing(v9.3). ISB 신호는 TradingView 기본 차트와 같은 배당 조정 안 한 가격으로 "
        "계산 (사용자 차트와 라벨 대조함), PT-1 손익은 배당 포함 조정 가격",
        "- 오버레이 = PT-1 백테스트 엔진 그대로 (같은 날짜·종목·$5,000·3슬롯·비용). 기준 = 같은 실행의 BASE (저장된 기준선과 같은지, "
        "래퍼를 모두 끼운 통과 모드가 BASE 와 거래까지 같은지 확인함)",
        "- 채택 후보 = 기준 대비 ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내 · "
        f"p ≤ {P_BONF:.5f}",
        f"- **결과: 채택 후보 {n_adopt}건 · 관찰 {n_obs}건 · 탈락 {n_fail}건**",
        "",
        "## PT-1 + ISB 타이밍 (PT-1 엔진, BASE 대비)",
        "",
        "| 조합 | Sharpe | CAGR | MDD | 거래 | 승률 | BASE 대비 ΔSharpe [95%] | p | 구간 | SPY 대비 ΔSharpe [95%] | "
        "게이트 통과율 | 매수 없는 날 | 매수 시 최대 비중 | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    spy_sh = t.loc[list(OVERLAYS), "SPY_Sharpe"].astype(float).iloc[0]
    lines.append(f"| SPY 보유 (종가) | {_f(spy_sh, '.2f')} | | | | | | | | | | | | 참고 |")
    b = t.loc["BASE"]
    lines.append(f"| BASE {NAMES['BASE']} | {b['Sharpe']:.2f} | {b['CAGR']:+.1%} | {b['MDD']:.1%} | {int(b['거래수'])} | "
                 f"{b['승률']:.0%} | | | | | | | | 기준 |")
    for key in OVERLAYS:
        r = t.loc[key]
        lines.append(
            f"| {key} {r['이름']} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | {int(r['거래수'])} | "
            f"{_f(r['승률'], '.0%')} | {r['ΔSharpe']:+.2f} [{r['하한']:+.2f}, {r['상한']:+.2f}] | {r['p']:.4f} | {r['구간']} | "
            f"{_f(r['ΔSharpe_SPY'], '+.2f')} [{_f(r['하한_SPY'], '+.2f')}, {_f(r['상한_SPY'], '+.2f')}] | "
            f"{_f(r['게이트통과율'], '.0%')} | {_f(r['매수없는날'], '.0%')} | {_f(r['최대비중'], '.0%')} | **{r['판정']}** |")
    lines += [
        "",
        "- 게이트 통과율 = 메일 상위 5종목을 PT-1 선택부터 차례로 볼 때 ISB가 '다음 시가 보유'인 비율. "
        "매수 없는 날 = 빈 슬롯이 있는데 5종목 모두 ISB 확인이 안 돼 사지 않은 날의 비율",
        "- 승률은 트림까지 합친 포지션 단위 (비용 포함)",
        "",
        "## ISB 단독 (전 종목, 동일가중 매수보유 대비)",
        "",
        "| 변형 | Sharpe | CAGR | MDD | 평균 주식 비중 | 동일가중 Sharpe·CAGR·MDD | 동일가중 대비 ΔSharpe [95%] | p | 구간 | "
        "SPY(시가) 대비 ΔSharpe [95%] | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in SOLOS:
        r = t.loc[key]
        lines.append(
            f"| {key} {r['이름']} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | {r['평균비중']:.0%} | "
            f"{r['기준_Sharpe']:.2f} · {r['기준_CAGR']:+.1%} · {r['기준_MDD']:.1%} | {r['ΔSharpe']:+.2f} "
            f"[{r['하한']:+.2f}, {r['상한']:+.2f}] | {r['p']:.4f} | {r['구간']} | {_f(r['ΔSharpe_SPY'], '+.2f')} "
            f"[{_f(r['하한_SPY'], '+.2f')}, {_f(r['상한_SPY'], '+.2f')}] | **{r['판정']}** |")
    lines += [
        "",
        f"- 참고: PT-1 기준선은 Sharpe {b['Sharpe']:.2f} · CAGR {b['CAGR']:+.1%} · MDD {b['MDD']:.1%}. "
        "단독은 수백 종목에 나눠 담아 3종목인 PT-1보다 Sharpe가 구조적으로 유리하고, 평가 시점(시가 대 종가)도 달라 짝지은 구간은 내지 않는다.",
        "",
        "## 참고: ISB 단독 3종목 계좌 (PT-1과 같은 엔진·후보 풀·$5,000·3슬롯, 시드별 동점 처리)",
        "",
        "| 주기 | Sharpe 최소 / 중앙 / 최대 | CAGR 중앙 | MDD 중앙 | PT-1 Sharpe 를 넘은 시드 |",
        "|---|---|---|---|---|",
        _solo3_line("D", refs["solo3"]["D"], base_r),
        _solo3_line("W", refs["solo3"]["W"], base_r),
        "",
        "## 참고: 무작위 게이트 위약 (매수 확인만, 같은 통과율로 무작위 통과)",
        "",
        "| 주기 | 통과율 | ISB 게이트 Sharpe | 무작위 Sharpe 최소 / 중앙 / 최대 | ISB 보다 높은 무작위 |",
        "|---|---|---|---|---|",
    ]
    for tf in ("D", "W"):
        isb_sh = t.at[f"{tf}-entry", "Sharpe"]
        pl = [sharpe(rr.equity.pct_change().dropna().tolist()) for _, rr in refs["placebo"][tf]]
        if pl:
            lines.append(f"| {'일봉' if tf == 'D' else '주봉'} | {refs['rates'][tf]:.0%} | {isb_sh:.2f} | "
                         f"{min(pl):.2f} / {float(np.median(pl)):.2f} / {max(pl):.2f} | {sum(x >= isb_sh for x in pl)}/{len(pl)} |")
    lines += ["", "## 참고: ISB 거래 단위 통계 (TradingView 대시보드 방식, 신호일 S&P 500 구성 종목, 비용 포함)", "",
              "| 주기 | 거래 | 승률 | PF | 기대값 R | 평균 수익 | 평균 보유(봉) | 신호 종류별 승/건 |", "|---|---|---|---|---|---|---|---|"]
    for tf in ("D", "W"):
        s = refs["isb_stats"][tf]
        types = " · ".join(f"{k} {v}" for k, v in s["by_type"].items())
        lines.append(f"| {'일봉' if tf == 'D' else '주봉'} | {s['n']} | {_f(s['win'], '.0%')} | {_f(s['pf'], '.2f')} | "
                     f"{_f(s['exp_r'], '+.2f')} | {_f(s['mean_ret'], '+.1%')} | {_f(s['hold'], '.0f')} | {types} |")
    lines += ["", "- 거래 단위 통계는 종목별로 따로 산 결과의 평균이다. 동시에 수백 종목을 살 수 없으므로 계좌 수익과 다르다.",
              "", "## 청산 이유 (건수)", "", "| 조합 | 청산 이유 |", "|---|---|"]
    lines += [f"| {k} | {t.at[k, '청산이유']} |" for k in ("BASE", *OVERLAYS)]
    yr = pd.DataFrame({k: _yearly(v) for k, v in series.items() if v is not None})
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(_f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    z = refs["zombie"]
    zline = " · ".join(f"{'일봉' if tf == 'D' else '주봉'} {h}/{n}건" for tf, (h, n) in z.items()) if z else "SEC 자료 없음"
    lines += ["", "## 한계", "",
              "- 개발 구간은 지금까지 많은 시도에 쓰였다. 홀드아웃(2025-10-01 ~)은 이미 한 번 써서 쓰지 않았고, 표본 외 확인이 없다.",
              "- 상장폐지 종목 가격이 없는 생존 편향은 PT-1 기준선과 같다 (2016년 구성 종목의 약 78%만 가격 있음).",
              "- ISB 신호는 TradingView 기본(배당 조정 안 함) 가격으로 계산했다. VIX·DXY 는 yfinance(^VIX·DX-Y.NYB)라 TradingView 와 "
              "미세하게 다를 수 있다. 거래 단위 통계는 TradingView 대시보드처럼 배당을 뺀 가격 수익이다.",
              f"- 좀비(재무 부실) 필터는 재무 자료가 없어 Pine 처럼 '통과'로 뒀다. 영향 상한(직전 10-K 순이익·영업현금흐름 모두 ≤ 0인 종목의 매수): {zline}.",
              "- PT-1 엔진 자체 특성: 약세장 2슬롯에서 현금을 다 쓴 뒤 3슬롯으로 풀리면 현금 0으로 0주 포지션을 산다 (BASE 616건 중 41건). 그 뒤 매수는 남은 현금 전부를 한 종목에 넣어 매수 시 비중이 100%까지 간다. BASE·오버레이 모두 같은 엔진이라 비교는 공정하지만, 기준선의 3종목 분산이 실제로는 자주 2종목이다.",
              "- ISB 의 Auto 결정표는 작성자가 겹치는 기간의 63종목으로 고른 것이라 ISB 에 유리할 수 있다.",
              "- 라이브 PT-1 은 종가 체결이고 NASDAQ 스캔 종목도 산다. 이 백테스트는 PIT S&P 500 · 다음 날 시가 체결이다.",
              "- ISB 확인이 안 되면 메일 상위 5종목 안에서 다음 후보로 넘어간다고 가정했다 (실제 판단과 다를 수 있음).",
              "- 결과가 좋아도 라이브 반영은 사용자 결정이다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="BASE 재현 + 래퍼 투명성 (ISB 성과 계산 안 함)")
    g.add_argument("--register", action="store_true", help="결과 계산 전 사전 등록 (TRIAL_LOG)")
    g.add_argument("--run", action="store_true", help="전체 실행 → 비교·판정·리포트")
    g.add_argument("--tv-check", action="store_true", help="TradingView 대조 목록")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if args.check:
        check()
    elif args.register:
        register()
    elif args.tv_check:
        print(isb.tv_check())
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
