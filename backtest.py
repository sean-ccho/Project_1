"""단순 시그널 기반 백테스트 모듈."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import TICKERS
from data.fetch import fetch_ohlcv
from features import compute_features_snapshot
from processing import apply_neutralization, liquidity_filter
from signals import attach_signals_and_sort


@dataclass
class TradeRecord:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    entry_judgement: str
    entry_recommendation: str
    entry_trend_score_final: float | None
    exit_judgement: str
    exit_recommendation: str
    exit_trend_score_final: float | None


def _prepare_price_map(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {ticker: df[ticker].dropna(how="all") for ticker in df.columns.levels[0]}


def _selected_rows_for_backtest(
    ranked: pd.DataFrame, top_n: int, *, select_bottom: bool = False
) -> pd.DataFrame:
    if select_bottom:
        candidates = ranked.copy()
    else:
        candidates = ranked[ranked["판단"].isin(["매수 후보", "관심 관찰"])].copy()
    if candidates.empty:
        return pd.DataFrame()
    if select_bottom:
        return candidates.tail(top_n)
    return candidates.head(top_n)


def run_backtest(
    tickers: Iterable[str] | None = None,
    *,
    period: str = "3y",
    hold_days: int = 20,
    rebalance_every: int = 20,
    top_n: int = 5,
    include_fundamentals: bool = True,
    max_tickers: int | None = 100,
    select_bottom: bool = False,
) -> dict[str, pd.DataFrame | dict]:
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

    price_map = _prepare_price_map(raw)

    opens = raw.xs("Open", level=1, axis=1)
    closes = raw.xs("Close", level=1, axis=1)
    dates = closes.index.to_list()
    if len(dates) < hold_days + 10:
        raise ValueError("분석 기간이 너무 짧습니다.")

    min_index = 200
    evaluation_indices = range(
        min_index,
        len(dates) - hold_days - 1,
        max(1, rebalance_every),
    )

    active_positions: list[dict[str, Any]] = []
    trades: list[TradeRecord] = []

    ranked_cache: dict[pd.Timestamp, pd.DataFrame] = {}
    signal_lookup_cache: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}

    def compute_ranked(cutoff_date: pd.Timestamp) -> pd.DataFrame:
        if cutoff_date in ranked_cache:
            return ranked_cache[cutoff_date]

        snapshot: dict[str, pd.DataFrame] = {}
        for ticker, frame in price_map.items():
            history = frame.loc[:cutoff_date]
            if history.empty:
                continue
            snapshot[ticker] = history.copy()
        if not snapshot:
            ranked_cache[cutoff_date] = pd.DataFrame()
            signal_lookup_cache[cutoff_date] = {}
            return ranked_cache[cutoff_date]

        features = compute_features_snapshot(
            snapshot, include_fundamentals=include_fundamentals
        )
        if features.empty:
            ranked_cache[cutoff_date] = pd.DataFrame()
            signal_lookup_cache[cutoff_date] = {}
            return ranked_cache[cutoff_date]

        liquid = liquidity_filter(features)
        if liquid.empty:
            ranked_cache[cutoff_date] = pd.DataFrame()
            signal_lookup_cache[cutoff_date] = {}
            return ranked_cache[cutoff_date]

        neutral = apply_neutralization(liquid)
        ranked = attach_signals_and_sort(neutral)
        ranked_cache[cutoff_date] = ranked

        lookup: dict[str, dict[str, Any]] = {}
        if not ranked.empty and "티커" in ranked.columns:
            signals_subset = ranked.assign(
                티커=ranked["티커"].astype(str)
            ).set_index("티커")
            cols = [col for col in ["판단", "추천", "트렌드점수_최종"] if col in signals_subset]
            if cols:
                lookup = signals_subset[cols].to_dict("index")
        signal_lookup_cache[cutoff_date] = lookup
        return ranked

    def exit_signals_for(ticker: str, exit_date: pd.Timestamp) -> dict[str, Any]:
        if exit_date not in closes.index:
            return {}
        exit_idx = closes.index.get_loc(exit_date)
        if isinstance(exit_idx, slice):
            exit_idx = exit_idx.start
        if exit_idx is None:
            return {}
        eval_idx = max(exit_idx - 1, 0)
        eval_date = closes.index[eval_idx]
        compute_ranked(eval_date)
        lookup = signal_lookup_cache.get(eval_date, {})
        return lookup.get(str(ticker), {})

    for idx in evaluation_indices:
        date = dates[idx]
        trade_idx = idx + 1
        exit_idx = idx + 1 + hold_days
        if trade_idx >= len(dates) or exit_idx >= len(dates):
            break

        trade_date = dates[trade_idx]
        exit_date = dates[exit_idx]

        next_positions: list[dict[str, Any]] = []
        for pos in active_positions:
            if trade_date >= pos["exit_date"]:
                exit_price = float(opens.at[pos["exit_date"], pos["ticker"]])
                entry_price = pos["entry_price"]
                if np.isnan(exit_price) or np.isnan(entry_price) or entry_price == 0:
                    continue
                return_pct = exit_price / entry_price - 1.0
                exit_meta = exit_signals_for(pos["ticker"], pos["exit_date"])
                trades.append(
                    TradeRecord(
                        ticker=pos["ticker"],
                        entry_date=pos["entry_date"],
                        exit_date=pos["exit_date"],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=return_pct,
                        entry_judgement=pos.get("entry_judgement", ""),
                        entry_recommendation=pos.get("entry_recommendation", ""),
                        entry_trend_score_final=pos.get("entry_trend_score_final"),
                        exit_judgement=exit_meta.get("판단", ""),
                        exit_recommendation=exit_meta.get("추천", ""),
                        exit_trend_score_final=exit_meta.get("트렌드점수_최종"),
                    )
                )
            else:
                next_positions.append(pos)

        active_positions = next_positions

        ranked = compute_ranked(date)
        selected = _selected_rows_for_backtest(
            ranked, top_n, select_bottom=select_bottom
        )
        if selected.empty:
            continue

        for _, row in selected.iterrows():
            ticker = row["티커"]
            if any(pos["ticker"] == ticker for pos in active_positions):
                continue

            entry_price = float(opens.at[trade_date, ticker])
            exit_price = float(opens.at[exit_date, ticker])
            if np.isnan(entry_price) or np.isnan(exit_price) or entry_price == 0:
                continue

            position = {
                "ticker": ticker,
                "entry_date": trade_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "entry_judgement": row.get("판단", ""),
                "entry_recommendation": row.get("추천", ""),
                "entry_trend_score_final": row.get("트렌드점수_최종"),
            }
            active_positions.append(position)

    for pos in active_positions:
        exit_price = float(opens.at[pos["exit_date"], pos["ticker"]])
        entry_price = pos["entry_price"]
        if np.isnan(exit_price) or np.isnan(entry_price) or entry_price == 0:
            continue
        return_pct = exit_price / entry_price - 1.0
        exit_meta = exit_signals_for(pos["ticker"], pos["exit_date"])
        trades.append(
            TradeRecord(
                ticker=pos["ticker"],
                entry_date=pos["entry_date"],
                exit_date=pos["exit_date"],
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=return_pct,
                entry_judgement=pos.get("entry_judgement", ""),
                entry_recommendation=pos.get("entry_recommendation", ""),
                entry_trend_score_final=pos.get("entry_trend_score_final"),
                exit_judgement=exit_meta.get("판단", ""),
                exit_recommendation=exit_meta.get("추천", ""),
                exit_trend_score_final=exit_meta.get("트렌드점수_최종"),
            )
        )

    if not trades:
        return {"trades": pd.DataFrame(), "summary": {}}

    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
    trades_df.sort_values("exit_date", inplace=True)

    trades_df["return_pct"] = trades_df["return_pct"].astype(float)

    wins = trades_df[trades_df["return_pct"] > 0]

    summary = {
        "총_거래수": int(len(trades_df)),
        "승률": f"{(len(wins) / len(trades_df) * 100):.2f}%" if len(trades_df) else "0.00%",
        "평균_수익률": f"{trades_df['return_pct'].mean() * 100:.2f}%",
        "중앙값_수익률": f"{trades_df['return_pct'].median() * 100:.2f}%",
    }

    trades_df = trades_df.rename(
        columns={
            "ticker": "티커",
            "entry_date": "진입일",
            "exit_date": "청산일",
            "entry_price": "진입가",
            "exit_price": "청산가",
            "return_pct": "수익률",
            "entry_judgement": "판단(진입)",
            "entry_recommendation": "추천(진입)",
            "entry_trend_score_final": "트렌드점수_최종(진입)",
            "exit_judgement": "판단(청산)",
            "exit_recommendation": "추천(청산)",
            "exit_trend_score_final": "트렌드점수_최종(청산)",
        }
    )

    trades_df["진입가"] = trades_df["진입가"].map(lambda x: f"${x:,.2f}")
    trades_df["청산가"] = trades_df["청산가"].map(lambda x: f"${x:,.2f}")
    trades_df["수익률"] = trades_df["수익률"].map(lambda x: f"{x * 100:.2f}%")
    for col in ["트렌드점수_최종(진입)", "트렌드점수_최종(청산)"]:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].map(
                lambda x: f"{x * 100:.2f}%" if pd.notna(x) else ""
            )

    desired_order = [
        "티커",
        "진입일",
        "판단(진입)",
        "추천(진입)",
        "트렌드점수_최종(진입)",
        "청산일",
        "판단(청산)",
        "추천(청산)",
        "트렌드점수_최종(청산)",
        "진입가",
        "청산가",
        "수익률",
    ]
    existing_order = [col for col in desired_order if col in trades_df.columns]
    trades_df = trades_df[existing_order]

    return {"trades": trades_df, "summary": summary}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="간단한 시그널 백테스트 실행")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument(
        "--full-universe",
        action="store_true",
        help="설정 시 등록된 전체 티커를 사용 (시간이 오래 걸릴 수 있음)",
    )
    parser.add_argument(
        "--select-bottom",
        action="store_true",
        help="상위 대신 하위 시그널 후보를 선택해 시뮬레이션",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    max_tickers = None if args.full_universe else args.max_tickers
    result = run_backtest(
        period=args.period,
        hold_days=args.hold_days,
        rebalance_every=args.rebalance_every,
        top_n=args.top_n,
        max_tickers=max_tickers,
        select_bottom=args.select_bottom,
    )
    trades = result["trades"]
    summary = result["summary"]
    if trades.empty:
        print("No trades generated.")
        return
    print("=== Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("\n=== Trades (head) ===")
    print(trades.head())


if __name__ == "__main__":
    main()
