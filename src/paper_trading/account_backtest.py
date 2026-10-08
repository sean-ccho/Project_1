"""PT-2/PT-3 계좌 백테스트.

실거래와 같은 account_engine.process_bar로 과거 일봉을 하루씩 재생한다
(같은 후보 선정, 같은 다음날 시가 체결, 같은 일봉 고가/저가 손절·목표, 같은 비용).
피처 스냅샷은 PT-1 백테스트와 같은 _compute_ranked_snapshot을 써서 디스크 캐시를 공유한다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any, Iterator

import pandas as pd

from data.fetch import fetch_ohlcv
from paper_trading.account_engine import (
    AccountState,
    BarSeries,
    _close_position,
    frames_to_series,
    process_bar,
)
from paper_trading.accounts import get_profile
from paper_trading.backtest import (
    _PROJECT_ROOT,
    _calculate_metrics,
    _code_hash,
    _config_snapshot,
    _git_info,
    _pit_universe_tickers,
    default_universe,
)
from screener.backtest import _compute_ranked_snapshot, _prepare_price_map
from paper_trading.earnings_calendar import EarningsCalendar
from screener.cache import write_cache_meta
from screener import config as _cfg
from screener.config import BACKTEST_COST_PER_SIDE, BACKTEST_FUNDAMENTALS_PIT_SAFE

logger = logging.getLogger(__name__)


class _LazySeries(Mapping):
    """price_map → BarSeries 변환을 처음 접근할 때만 한다 (전 종목을 미리 바꾸면 메모리 낭비)."""

    def __init__(self, price_map: dict[str, pd.DataFrame]):
        self._frames = price_map
        self._cache: dict[str, BarSeries] = {}

    def __getitem__(self, ticker: str) -> BarSeries:
        if ticker not in self._cache:
            if ticker not in self._frames:
                raise KeyError(ticker)
            self._cache[ticker] = frames_to_series({ticker: self._frames[ticker]}).get(ticker) or BarSeries([])
        return self._cache[ticker]

    def __iter__(self) -> Iterator[str]:
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)


def run_account_backtest(
    key: str,
    *,
    period: str = "5y",
    tickers: list[str] | None = None,
    max_tickers: int | None = 100,
    include_fundamentals: bool = False,
    min_history_days: int = 220,
    start_date: str | None = None,
    end_date: str | None = None,
    no_cache: bool = False,
    params_override: dict[str, Any] | None = None,
    save_run: bool = True,
    pit_universe: bool | None = None,
) -> dict[str, Any]:
    """계좌(pt2/pt3) 백테스트.

    Args:
        start_date/end_date: 시뮬레이션 구간 (walk-forward·홀드아웃용). 피처는 그 전 데이터로 계산.
        params_override: 계좌 파라미터 일부를 바꿔서 실행 (Optuna용, config 원본은 건드리지 않음)
        pit_universe: True면 그날의 S&P 500 구성종목만 후보로 (None이면 config.BACKTEST_PIT_UNIVERSE)
    Returns:
        {"trades", "summary", "equity_curve", "state", "run_dir"}
    """
    profile = get_profile(key)
    if params_override:
        profile = replace(profile, params={**profile.params, **params_override})

    membership = None
    if _cfg.BACKTEST_PIT_UNIVERSE if pit_universe is None else pit_universe:
        from paper_trading.universe import Membership, load_membership
        membership = Membership(load_membership(_PROJECT_ROOT / _cfg.SP500_MEMBERSHIP_PATH))
        if tickers is None:
            tickers = _pit_universe_tickers(membership, period)

    universe = [t for t in (tickers or default_universe()) if t != "SPY"]
    if max_tickers is not None:
        universe = universe[:max_tickers]
    tickers = ["SPY"] + universe

    print(f"[{profile.name} 백테스트] 데이터 다운로드: {len(tickers)}종목, {period}")
    raw = fetch_ohlcv(tickers, period=period, force_download=no_cache)
    if raw.empty:
        raise RuntimeError("OHLCV 데이터를 가져오지 못했습니다.")
    dates = list(raw.index)
    if len(dates) <= min_history_days + 5:
        raise ValueError(f"기간이 너무 짧습니다 (필요 {min_history_days + 5}일, 실제 {len(dates)}일)")

    # PT-1 백테스트와 같은 키 → 같은 피처 캐시를 공유
    ohlcv_key = ",".join(sorted(tickers)) + "|" + period + f"|fund={int(include_fundamentals)}|code={_code_hash()}"
    ohlcv_hash = hashlib.md5(ohlcv_key.encode()).hexdigest()[:12]
    write_cache_meta(ohlcv_hash, period, include_fundamentals)

    fundamentals = None
    if include_fundamentals:
        from screener.fundamentals import fetch_fundamental_snapshots

        fundamentals = fetch_fundamental_snapshots(universe, use_cache=not no_cache)
        if fundamentals is not None and BACKTEST_FUNDAMENTALS_PIT_SAFE:
            keep = [c for c in ("티커", "fund_sector") if c in fundamentals.columns]
            fundamentals = fundamentals[keep]

    price_map = _prepare_price_map(raw)
    series = _LazySeries(price_map)

    first_idx = min_history_days
    sim = [
        (i, d) for i, d in enumerate(dates)
        if i >= first_idx
        and (start_date is None or str(d.date()) >= start_date)
        and (end_date is None or str(d.date()) <= end_date)
    ]
    if not sim:
        raise ValueError("시뮬레이션할 날짜가 없습니다 (start_date/end_date 확인)")

    ic_cache: dict[str, Any] = {}
    earnings_cal = EarningsCalendar.load(_PROJECT_ROOT / "data" / "research" / "sec_earnings_dates.parquet")
    state = AccountState.new(float(profile.params["initial_capital"]))
    print(f"[{profile.name} 백테스트] {sim[0][1].date()} ~ {sim[-1][1].date()} ({len(sim)}거래일)")

    for n, (_, d) in enumerate(sim):
        ranked = _compute_ranked_snapshot(
            price_map, d,
            include_fundamentals=include_fundamentals,
            cache={},
            ic_weights_cache=ic_cache,
            ohlcv_hash=ohlcv_hash,
            fundamentals_df=fundamentals,
        )
        if membership is not None and "티커" in ranked.columns:
            ranked = ranked[ranked["티커"].isin(membership.on(str(d.date())) | {"SPY"})]
        ranked = earnings_cal.annotate(ranked, d)  # 어닝 회피 필터용 (실거래와 같게)
        rows = ranked.to_dict("records") if not ranked.empty else []
        process_bar(state, profile, str(d.date()), series, rows=rows)
        if n % 50 == 0:
            print(f"  [{n / len(sim):5.1%}] {d.date()} | 보유 {len(state.positions)} | 거래 {len(state.trades)} | 평가액 ${state.equity:,.0f}")

    # 남은 포지션은 마지막 종가로 정리 (PT-1 백테스트의 '기간종료'와 같은 처리)
    last = str(sim[-1][1].date())
    closing: dict[str, Any] = {"sells": []}
    for pos in list(state.positions):
        bar = series[pos["ticker"]].on(last) if pos["ticker"] in series else None
        if bar:
            _close_position(state, profile, pos["ticker"], float(bar["close"]), last, "기간종료", series, closing)

    equity_curve = pd.Series(
        {pd.Timestamp(h["date"]): h["equity"] for h in state.equity_history}, name="equity_curve"
    )
    spy = raw["SPY"]["Close"].pct_change()
    spy_in_range = spy[(spy.index >= sim[0][1]) & (spy.index <= sim[-1][1])].dropna()
    summary = _calculate_metrics(
        state.trades, spy_in_range,
        initial_capital=state.initial_capital, final_cash=state.cash,
        equity_series=equity_curve if len(equity_curve) > 1 else None,
    )
    summary["시뮬레이션_시작"] = str(sim[0][1].date())
    summary["시뮬레이션_종료"] = last
    if summary.get("총거래수"):
        from paper_trading.benchmarks import benchmark_summary
        summary.update(benchmark_summary(
            equity_curve, raw.xs("Close", level=1, axis=1),
            top_n=int(_cfg.BENCHMARK_MOMENTUM_TOP_N),
            cost_per_side=BACKTEST_COST_PER_SIDE,
            members_on=membership.on if membership is not None else None,
        ))
        summary["PIT_유니버스"] = membership is not None

    run_dir = _save_run(profile, state, summary, period, tickers, sim, include_fundamentals) if save_run else None
    return {"trades": state.trades, "summary": summary, "equity_curve": equity_curve, "state": state, "run_dir": run_dir}


def _save_run(profile: Any, state: AccountState, summary: dict, period: str, tickers: list[str],
              sim: list, include_fundamentals: bool) -> str:
    run_dir = os.path.join("output", "runs", f"{datetime.now():%Y-%m-%d_%H-%M}_{profile.key}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "trades.json"), "w", encoding="utf-8") as f:
        json.dump(state.trades, f, ensure_ascii=False, indent=2, default=str)
    if state.trades:
        keys = list(dict.fromkeys(k for t in state.trades for k in t))
        with open(os.path.join(run_dir, "trades.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(state.trades)
    config_snapshot = _config_snapshot()
    meta = {
        "account": profile.key,
        "version": profile.version,
        "params": profile.params,
        "period": period,
        "include_fundamentals": include_fundamentals,
        "ticker_count": len(tickers),
        "simulation": {"start_date": str(sim[0][1].date()), "end_date": str(sim[-1][1].date()), "trading_days": len(sim)},
        "code_hash": _code_hash(),
        "config_hash": hashlib.md5(json.dumps(config_snapshot, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:8],
        "git": _git_info(),
        "summary": summary,
        "equity_history": state.equity_history,
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"[{profile.name} 백테스트] 저장: {run_dir}/")
    return run_dir
