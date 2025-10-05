"""신규 시그널 기반 백테스트 모듈."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable

import numpy as np
import pandas as pd

from config import TICKERS
from data.fetch import fetch_ohlcv
from features import compute_features_snapshot
from processing import apply_neutralization, liquidity_filter
from signals import attach_signals_and_sort


@dataclass
class SignalTrade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    position_size: float
    entry_support_hits: int
    exit_signal_hits: int


def _prepare_price_map(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {ticker: raw[ticker].dropna(how="all") for ticker in raw.columns.levels[0]}


def _compute_ranked_snapshot(
    price_map: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    *,
    include_fundamentals: bool,
    cache: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    if cutoff in cache:
        return cache[cutoff]

    snapshot: dict[str, pd.DataFrame] = {}
    for ticker, frame in price_map.items():
        history = frame.loc[:cutoff]
        if history.empty:
            continue
        snapshot[ticker] = history

    if not snapshot:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    features = compute_features_snapshot(
        snapshot, include_fundamentals=include_fundamentals
    )
    if features.empty:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    liquid = liquidity_filter(features)
    if liquid.empty:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    neutral = apply_neutralization(liquid)
    ranked = attach_signals_and_sort(neutral)
    cache[cutoff] = ranked
    return ranked


def _lookup(rowset: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if rowset.empty or "티커" not in rowset.columns:
        return {}
    index_df = rowset.assign(티커=rowset["티커"].astype(str)).set_index("티커")
    return index_df.to_dict("index")


def run_backtest(
    tickers: Iterable[str] | None = None,
    *,
    period: str = "1y",
    include_fundamentals: bool = True,
    max_positions: int = 5,
    max_tickers: int | None = 100,
    min_history_days: int = 220,
    rebalance_every: int = 20,
) -> dict[str, Any]:
    if tickers is None:
        tickers = TICKERS
    tickers = list(dict.fromkeys(tickers))
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    if not tickers:
        raise ValueError("백테스트에 사용할 티커가 필요합니다.")

    raw = fetch_ohlcv(tickers, period=period)
    if raw.empty:
        raise RuntimeError("OHLCV 데이터를 가져오지 못했습니다.")

    closes = raw.xs("Close", level=1, axis=1)
    opens = raw.xs("Open", level=1, axis=1)
    dates = closes.index.to_list()
    if len(dates) <= min_history_days + 5:
        raise ValueError("분석 기간이 너무 짧습니다.")

    price_map = _prepare_price_map(raw)
    ranked_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    start_index = min_history_days
    trades: list[SignalTrade] = []
    active_positions: dict[str, dict[str, Any]] = {}
    sell_windows: dict[str, Deque[bool]] = {}

    for idx in range(start_index, len(dates) - 1):
        date = dates[idx]
        next_date = dates[idx + 1]

        ranked = _compute_ranked_snapshot(
            price_map, date, include_fundamentals=include_fundamentals, cache=ranked_cache
        )
        lookup = _lookup(ranked)

        # 1) 기존 포지션 청산 여부 확인(최근 3거래일 중 2회 매도 신호)
        next_active: dict[str, dict[str, Any]] = {}
        for ticker, position in active_positions.items():
            meta = lookup.get(ticker)
            sell_signal = bool(meta.get("sell_signal")) if meta else False
            window = sell_windows.setdefault(ticker, deque(maxlen=3))
            window.append(sell_signal)
            sell_hits = sum(window)

            if sell_hits >= 2:
                exit_price = float(opens.at[next_date, ticker])
                entry_price = position["entry_price"]
                if np.isnan(exit_price) or np.isnan(entry_price) or entry_price == 0:
                    continue
                return_pct = exit_price / entry_price - 1.0
                trades.append(
                    SignalTrade(
                        ticker=ticker,
                        entry_date=position["entry_date"],
                        exit_date=next_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=return_pct,
                        position_size=position.get("position_size", 1.0),
                        entry_support_hits=position.get("support_hits", 0),
                        exit_signal_hits=sell_hits,
                    )
                )
                sell_windows[ticker].clear()
            else:
                next_active[ticker] = position
        active_positions = next_active

        # 2) 신규 진입 – 리밸런스 주기마다 buy_signal 종목 추가
        if (idx - start_index) % max(1, rebalance_every) != 0:
            continue

        candidates = ranked[(ranked.get("buy_signal") == True)].copy()
        if candidates.empty:
            continue

        candidates = candidates.sort_values(
            ["우선순위", "트렌드점수_최종"], ascending=[True, False]
        )

        available_slots = max_positions - len(active_positions)
        if available_slots <= 0:
            continue

        for _, row in candidates.iterrows():
            ticker = str(row["티커"])
            if ticker in active_positions:
                continue

            entry_price = float(opens.at[next_date, ticker])
            if np.isnan(entry_price) or entry_price == 0:
                continue

            support_hits_raw = row.get("support_count", np.nan)
            if pd.notna(support_hits_raw):
                support_hits = int(support_hits_raw)
            else:
                support_hits = sum(
                    [
                        row.get("RSI", np.nan) < 55,
                        row.get("volume", np.nan)
                        > row.get("volume_ma20", np.nan) * 1.2,
                        row.get("adx", np.nan) > 20,
                        (row.get("obv", np.nan) > row.get("obv_ma20", np.nan))
                        or (row.get("obv_mom_5", np.nan) > 0),
                        row.get("atr_pct", np.nan) < row.get("atr_buy_max", np.nan),
                    ]
                )

            active_positions[ticker] = {
                "entry_date": next_date,
                "entry_price": entry_price,
                "position_size": float(row.get("position_size", 1.0)),
                "support_hits": support_hits,
            }
            sell_windows[ticker] = deque(maxlen=3)

            available_slots -= 1
            if available_slots <= 0:
                break

    # 잔여 포지션 강제 청산
    final_date = dates[-1]
    for ticker, position in active_positions.items():
        exit_price = float(closes.at[final_date, ticker])
        entry_price = position["entry_price"]
        if np.isnan(exit_price) or np.isnan(entry_price) or entry_price == 0:
            continue
        return_pct = exit_price / entry_price - 1.0
        trades.append(
            SignalTrade(
                ticker=ticker,
                entry_date=position["entry_date"],
                exit_date=final_date,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=return_pct,
                position_size=position.get("position_size", 1.0),
                entry_support_hits=position.get("support_hits", 0),
                exit_signal_hits=sum(sell_windows.get(ticker, [])),
            )
        )

    if not trades:
        return {"trades": pd.DataFrame(), "summary": {}}

    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
    trades_df.sort_values("exit_date", inplace=True)

    trades_df["weighted_return"] = trades_df["return_pct"] * trades_df["position_size"]
    total_weight = trades_df["position_size"].sum()
    weighted_avg_return = (
        trades_df["weighted_return"].sum() / total_weight if total_weight else np.nan
    )

    wins = trades_df[trades_df["return_pct"] > 0]
    summary = {
        "총거래수": int(len(trades_df)),
        "승률": float(len(wins) / len(trades_df)) if len(trades_df) else np.nan,
        "평균수익률": float(trades_df["return_pct"].mean()),
        "가중평균수익률": float(weighted_avg_return)
        if not np.isnan(weighted_avg_return)
        else np.nan,
        "중앙값수익률": float(trades_df["return_pct"].median()),
    }

    trades_df = trades_df.rename(
        columns={
            "ticker": "티커",
            "entry_date": "진입일",
            "exit_date": "청산일",
            "entry_price": "진입가",
            "exit_price": "청산가",
            "return_pct": "수익률",
            "position_size": "포지션사이즈",
            "entry_support_hits": "진입보조충족",
            "exit_signal_hits": "최근3일매도신호",
        }
    )

    trades_df["수익률"] = trades_df["수익률"].map(lambda x: round(x * 100, 2))
    trades_df["포지션사이즈"] = trades_df["포지션사이즈"].map(lambda x: round(x, 4))

    return {"trades": trades_df, "summary": summary}
