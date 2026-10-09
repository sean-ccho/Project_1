"""PT-2/PT-3 공통 계좌 엔진.

한 일봉(bar_date)을 처리하는 순서 (process_bar):
  1) 전날 예약한 주문을 오늘 시가로 체결 (매도 먼저, 갭 상한을 넘은 매수는 취소)
  2) 보유 종목 손절·목표가를 오늘 고가/저가로 판정
     (갭으로 뚫고 시작하면 시가 체결, 같은 날 둘 다 닿으면 손절 우선 = 보수적)
  3) 장 마감 후: 고점 갱신 → 전략별 청산 신호·손절선 갱신 → 다음날 시가 매도 예약
  4) 평가액(현금 + 종가 평가) 기록
  5) 그날 스냅샷이 있으면 신규 후보를 골라 다음날 시가 매수 예약

핵심 로직은 pandas 없이 동작한다 (실거래와 백테스트가 같은 process_bar를 호출).
pandas가 필요한 부분(일봉 다운로드·스냅샷 로드)은 함수 안에서 import 한다.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from paper_trading.indicators import compute_indicators, is_num, num
from paper_trading.json_store import atomic_json_write, read_json

logger = logging.getLogger(__name__)

INDICATOR_LOOKBACK = 260   # 청산 지표 계산에 쓰는 최근 일봉 수
MAX_CATCHUP_BARS = 10      # 실행이 빠진 날을 따라잡는 최대 일봉 수
EXCLUDED_TICKERS = {"SPY", "QQQ", "IWM"}


# ── 일봉 시리즈 ──────────────────────────────────────────────


class BarSeries:
    """한 종목의 일봉 리스트 (날짜 오름차순, 날짜는 'YYYY-MM-DD')."""

    def __init__(self, bars: list[dict[str, Any]]):
        self.bars = bars
        self.dates = [b["date"] for b in bars]

    def on(self, d: str) -> dict[str, Any] | None:
        i = bisect_left(self.dates, d)
        return self.bars[i] if i < len(self.dates) and self.dates[i] == d else None

    def upto(self, d: str, n: int | None = None) -> list[dict[str, Any]]:
        """d까지(포함)의 일봉. n이 있으면 최근 n개만 — 미래 일봉은 절대 포함하지 않는다."""
        j = bisect_right(self.dates, d)
        return self.bars[max(0, j - n) if n else 0:j]

    def count_after(self, after: str, upto: str) -> int:
        """after 다음 일봉부터 upto까지의 일봉 수 (after 당일 = 0)."""
        return bisect_right(self.dates, upto) - bisect_right(self.dates, after)


def frames_to_series(frames: Mapping[str, Any]) -> dict[str, BarSeries]:
    """{티커: OHLCV DataFrame} → {티커: BarSeries}."""
    out: dict[str, BarSeries] = {}
    for ticker, frame in frames.items():
        if frame is None or getattr(frame, "empty", True):
            continue
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        cols = [frame[c].tolist() for c in ("Open", "High", "Low", "Close")]
        vols = frame["Volume"].tolist() if "Volume" in frame.columns else [0.0] * len(frame)
        out[str(ticker)] = BarSeries([
            {"date": str(ts.date()), "open": float(o), "high": float(h), "low": float(lo),
             "close": float(c), "volume": float(v)}
            for ts, o, h, lo, c, v in zip(frame.index, *cols, vols)
        ])
    return out


def fetch_bar_series(tickers: list[str], period: str = "1y") -> dict[str, BarSeries]:
    """yfinance 일봉을 받아 BarSeries로 변환한다 (실거래용)."""
    from data.fetch import fetch_ohlcv

    raw = fetch_ohlcv(list(tickers), period=period)
    names = raw.columns.get_level_values(0).unique()
    return frames_to_series({t: raw[t] for t in names})


# ── 계좌 상태 ────────────────────────────────────────────────


@dataclass
class AccountState:
    """계좌 1개의 메모리 상태 (실거래는 JSON에 저장, 백테스트는 메모리에만)."""

    positions: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    equity_history: list[dict[str, Any]] = field(default_factory=list)
    cash: float = 0.0
    initial_capital: float = 0.0

    @classmethod
    def new(cls, initial_capital: float) -> "AccountState":
        return cls(cash=initial_capital, initial_capital=initial_capital)

    @property
    def equity(self) -> float:
        """마지막 종가 기준 평가액 (기록이 없으면 현금)."""
        return float(self.equity_history[-1]["equity"]) if self.equity_history else self.cash


def load_account(profile: Any) -> AccountState:
    d = profile.data_dir
    acct = read_json(d / "account.json", None)
    if acct is None:
        state = AccountState.new(float(profile.params["initial_capital"]))
    else:
        state = AccountState(
            cash=float(acct["cash"]),
            initial_capital=float(acct["initial_capital"]),
            pending=acct.get("pending", []),
            equity_history=acct.get("equity_history", []),
        )
    state.positions = read_json(d / "positions.json", [])
    state.trades = read_json(d / "trades.json", [])
    return state


def save_account(profile: Any, state: AccountState) -> None:
    d = profile.data_dir
    atomic_json_write(d / "positions.json", state.positions)
    atomic_json_write(d / "trades.json", state.trades)
    atomic_json_write(d / "account.json", {
        "account": profile.key,
        "version": profile.version,
        "initial_capital": state.initial_capital,
        "cash": round(state.cash, 2),
        "pending": state.pending,
        "equity_history": state.equity_history,
    })


# ── 시장 레짐 ────────────────────────────────────────────────


def regime_from_rows(rows: list[Mapping[str, Any]]) -> str:
    """SPY 행으로 레짐 판단 (PT-1 detect_market_regime과 같은 규칙, 종가 close 우선)."""
    for r in rows:
        if str(r.get("티커", "")).strip() != "SPY":
            continue
        close = num(r, "close")
        if not is_num(close):
            close = num(r, "현재가격")
        e50, e200 = num(r, "ema50"), num(r, "ema200")
        if is_num(close) and is_num(e50) and is_num(e200) and min(close, e50, e200) > 0:
            if close > e50 > e200:
                return "bull"
            if close < e200:
                return "bear"
        return "neutral"
    return "neutral"


# ── 하루 처리 ────────────────────────────────────────────────


def _bar(series: Mapping[str, BarSeries], ticker: str, d: str) -> dict[str, Any] | None:
    s = series.get(ticker)
    return s.on(d) if s else None


def process_bar(
    state: AccountState,
    profile: Any,
    bar_date: str,
    series: Mapping[str, BarSeries],
    rows: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """일봉 하나를 처리한다. rows(그날 스냅샷)가 있으면 신규 매수도 예약한다."""
    result: dict[str, Any] = {
        "date": bar_date, "buys": [], "sells": [], "cancelled": [],
        "orders": [], "candidates": [], "debug": {},
    }
    _execute_pending(state, profile, bar_date, series, result)
    _check_resting_exits(state, profile, bar_date, series, result)
    _after_close(state, profile, bar_date, series, result)
    _record_equity(state, bar_date, series)
    if rows is not None:
        _place_new_orders(state, profile, bar_date, rows, series, result)
    result["equity"] = state.equity
    result["cash"] = round(state.cash, 2)
    result["holdings"] = len(state.positions)
    return result


def _execute_pending(state: AccountState, profile: Any, bar_date: str, series: Mapping[str, BarSeries], result: dict) -> None:
    p = profile.params
    keep: list[dict[str, Any]] = []

    for order in [o for o in state.pending if o["side"] == "SELL"]:
        bar = _bar(series, order["ticker"], bar_date)
        if order["signal_date"] >= bar_date or bar is None:
            keep.append(order)  # 거래정지 등으로 시가가 없으면 다음 일봉에 재시도
            continue
        _close_position(state, profile, order["ticker"], float(bar["open"]), bar_date, order["reason"], series, result)

    target = state.equity / p["max_positions"]
    buys = sorted((o for o in state.pending if o["side"] == "BUY"), key=lambda o: o.get("rank", 0))
    for order in buys:
        if order["signal_date"] >= bar_date:
            keep.append(order)
            continue
        bar = _bar(series, order["ticker"], bar_date)
        reason = None
        if bar is None:
            reason = "시가 없음"
        elif any(pos["ticker"] == order["ticker"] for pos in state.positions):
            reason = "이미 보유"
        elif len(state.positions) >= p["max_positions"]:
            reason = "빈 슬롯 없음"
        else:
            gap = float(bar["open"]) / float(order["signal_close"]) - 1.0
            if gap > p["max_gap_up"]:
                reason = f"갭상승 {gap:+.1%}"
        alloc = min(target, state.cash)
        if reason is None and alloc < target * p["min_alloc_ratio"]:
            reason = "현금 부족"
        if reason:
            result["cancelled"].append({**order, "cancel_reason": reason, "date": bar_date})
            continue
        result["buys"].append(_open_position(state, profile, order, float(bar["open"]), bar_date, alloc))

    state.pending = keep


def _open_position(state: AccountState, profile: Any, order: dict, price: float, bar_date: str, alloc: float) -> dict[str, Any]:
    cost = profile.params["cost_per_side"]
    shares = alloc / (price * (1 + cost))
    state.cash -= shares * price * (1 + cost)
    pos: dict[str, Any] = {
        "ticker": order["ticker"],
        "entry_price": round(price, 4),
        "entry_date": bar_date,
        "shares": round(shares, 6),
        "strategy": order["strategy"],
        "star_rating": order.get("star_rating", ""),
        "ccs_score": order.get("score", 0.0),
        "sector": order.get("sector", ""),
        "setup": order.get("setup", ""),
        "highest_price": round(price, 4),
        "highest_close": round(price, 4),
        "last_close": round(price, 4),
        "bars_held": 0,
        "signal_date": order["signal_date"],
        "signal_close": order["signal_close"],
        "atr_at_signal": order["atr"],
        "version": profile.version,
        **order.get("meta", {}),
    }
    profile.strategy.init_position(pos, order, profile.params)
    state.positions.append(pos)
    return {**pos, "reason": "신규매수", "allocation": round(alloc, 2)}


def _check_resting_exits(state: AccountState, profile: Any, bar_date: str, series: Mapping[str, BarSeries], result: dict) -> None:
    for pos in list(state.positions):
        bar = _bar(series, pos["ticker"], bar_date)
        if bar is None:
            continue
        o, h, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
        stop, target = pos.get("stop_price"), pos.get("target_price")
        kind = pos.get("stop_kind", "손절")
        new_today = pos["entry_date"] == bar_date
        exit_px = reason = None
        if is_num(stop) and not new_today and o <= stop:
            exit_px, reason = o, f"갭{kind}"
        elif is_num(target) and not new_today and o >= target:
            exit_px, reason = o, "갭목표가"
        elif is_num(stop) and lo <= stop:
            exit_px, reason = float(stop), kind
        elif is_num(target) and h >= target:
            exit_px, reason = float(target), "목표가"
        if exit_px is not None:
            _close_position(state, profile, pos["ticker"], exit_px, bar_date, reason, series, result)


def _after_close(state: AccountState, profile: Any, bar_date: str, series: Mapping[str, BarSeries], result: dict) -> None:
    selling = {o["ticker"] for o in state.pending if o["side"] == "SELL"}
    for pos in state.positions:
        s = series.get(pos["ticker"])
        bar = s.on(bar_date) if s else None
        if bar is None:
            continue
        pos["highest_price"] = round(max(float(pos.get("highest_price", pos["entry_price"])), float(bar["high"])), 4)
        pos["highest_close"] = round(max(float(pos.get("highest_close", pos["entry_price"])), float(bar["close"])), 4)
        pos["last_close"] = float(bar["close"])
        pos["last_bar_date"] = bar_date
        pos["bars_held"] = s.count_after(pos["entry_date"], bar_date)
        if pos["ticker"] in selling:
            continue
        ind = compute_indicators(s.upto(bar_date, INDICATOR_LOOKBACK))
        reason = profile.strategy.on_bar_close(pos, ind, bar_date, profile.params)
        if reason:
            order = {
                "side": "SELL", "ticker": pos["ticker"], "signal_date": bar_date,
                "signal_close": float(bar["close"]), "reason": reason,
            }
            state.pending.append(order)
            result["orders"].append(order)


def _record_equity(state: AccountState, bar_date: str, series: Mapping[str, BarSeries]) -> None:
    market_value = 0.0
    for pos in state.positions:
        bar = _bar(series, pos["ticker"], bar_date)
        px = float(bar["close"]) if bar else float(pos.get("last_close", pos["entry_price"]))
        market_value += float(pos["shares"]) * px
    if state.equity_history and state.equity_history[-1]["date"] == bar_date:
        state.equity_history.pop()
    state.equity_history.append({
        "date": bar_date,
        "equity": round(state.cash + market_value, 2),
        "cash": round(state.cash, 2),
        "positions": len(state.positions),
    })


def _place_new_orders(
    state: AccountState,
    profile: Any,
    bar_date: str,
    rows: list[Mapping[str, Any]],
    series: Mapping[str, BarSeries],
    result: dict,
) -> None:
    p = profile.params
    selling = {o["ticker"] for o in state.pending if o["side"] == "SELL"}
    buying = {o["ticker"] for o in state.pending if o["side"] == "BUY"}
    staying = [pos for pos in state.positions if pos["ticker"] not in selling]
    ctx = {
        "bar_date": bar_date,
        "regime": regime_from_rows(rows),
        "exclude": {pos["ticker"] for pos in state.positions} | buying | EXCLUDED_TICKERS,
        "series": series,
    }
    candidates, debug = profile.strategy.select_candidates(rows, ctx, p)
    result["candidates"] = candidates[:10]
    result["debug"] = debug

    slots = min(p["max_positions"] - len(staying) - len(buying), p["max_daily_buys"])
    if slots <= 0:
        debug["skip"] = "빈 슬롯 없음"
        return
    sectors = Counter(pos.get("sector", "") for pos in staying)
    placed = 0
    for c in candidates:
        if placed >= slots:
            break
        if c["sector"] and sectors[c["sector"]] >= p["max_same_sector"]:
            continue
        order = {
            "side": "BUY", "ticker": c["ticker"], "signal_date": bar_date,
            "signal_close": c["close"], "rank": placed, "score": c["score"],
            "strategy": c["strategy"], "star_rating": c.get("star", ""),
            "sector": c["sector"], "atr": c["atr"], "setup": c.get("setup", ""),
            "meta": c.get("meta", {}),
        }
        state.pending.append(order)
        result["orders"].append(order)
        sectors[c["sector"]] += 1
        placed += 1


def _close_position(
    state: AccountState,
    profile: Any,
    ticker: str,
    price: float,
    bar_date: str,
    reason: str,
    series: Mapping[str, BarSeries],
    result: dict,
) -> dict[str, Any] | None:
    idx = next((i for i, pos in enumerate(state.positions) if pos["ticker"] == ticker), None)
    if idx is None:
        return None
    pos = state.positions.pop(idx)
    cost = profile.params["cost_per_side"]
    shares = float(pos["shares"])
    entry = float(pos["entry_price"])
    proceeds = shares * price * (1 - cost)
    state.cash += proceeds
    ret = price * (1 - cost) / (entry * (1 + cost)) - 1.0
    s = series.get(ticker)
    highest = max(float(pos.get("highest_price", entry)), price)
    trade = {
        "ticker": ticker,
        "entry_date": pos["entry_date"],
        "exit_date": bar_date,
        "entry_price": entry,
        "exit_price": round(price, 4),
        "shares": shares,
        "return_pct": round(ret, 4),
        "gross_return_pct": round(price / entry - 1.0, 4),
        "dollar_pnl": round(proceeds - shares * entry * (1 + cost), 2),
        "holding_days": (date.fromisoformat(bar_date) - date.fromisoformat(pos["entry_date"])).days,
        "holding_bars": s.count_after(pos["entry_date"], bar_date) if s else pos.get("bars_held", 0),
        "strategy": pos.get("strategy", ""),
        "setup": pos.get("setup", ""),
        "star_rating": pos.get("star_rating", ""),
        "ccs_score": pos.get("ccs_score", 0.0),
        "sector": pos.get("sector", ""),
        "exit_reason": f"{reason}({ret:+.1%})",
        "highest_price": round(highest, 4),
        "actual_max_upside_pct": round(highest / entry - 1.0, 4),
        "version": pos.get("version", profile.version),
    }
    state.trades.append(trade)
    result["sells"].append(trade)
    state.pending = [o for o in state.pending if not (o["side"] == "SELL" and o["ticker"] == ticker)]
    return trade


# ── 실거래 실행 ──────────────────────────────────────────────


def catchup_dates(calendar: BarSeries, last_processed: str | None, bar_date: str) -> list[str]:
    """마지막 처리일과 bar_date 사이에 빠진 일봉 날짜 (최대 MAX_CATCHUP_BARS개)."""
    if not last_processed:
        return []
    lo = bisect_right(calendar.dates, last_processed)
    hi = bisect_left(calendar.dates, bar_date)
    return calendar.dates[lo:hi][-MAX_CATCHUP_BARS:]


def _print_result(profile: Any, result: dict[str, Any]) -> None:
    tag = f"[{profile.name}]"
    print(
        f"{tag} {result['date']} | 매수 {len(result['buys'])} · 매도 {len(result['sells'])} · "
        f"취소 {len(result['cancelled'])} · 예약 {len(result['orders'])} | "
        f"보유 {result['holdings']} | 평가액 ${result['equity']:,.2f} (현금 ${result['cash']:,.2f})"
    )
    for t in result["sells"]:
        print(f"  SELL {t['ticker']:6s} @ {t['exit_price']:.2f} {t['exit_reason']}")
    for b in result["buys"]:
        print(f"  BUY  {b['ticker']:6s} @ {b['entry_price']:.2f} 손절 {b.get('stop_price') or 0:.2f} ({b['strategy']})")
    for c in result["cancelled"]:
        print(f"  취소 {c['ticker']:6s} {c['cancel_reason']}")
    for o in result["orders"]:
        print(f"  예약 {o['side']:4s} {o['ticker']:6s} (다음 시가) {o.get('reason') or o.get('strategy', '')}")
    debug = result.get("debug") or {}
    if debug:
        print(f"  후보 선정: {debug}")
    for c in result["candidates"][:5]:
        print(f"  후보 {c['ticker']:6s} 점수 {c['score']:.3f} {c['strategy']}")


def _sync_sheets(profile: Any, state: AccountState, result: dict[str, Any]) -> None:
    try:
        from paper_trading.sheet_sync import sync_positions, sync_summary, sync_trade_log

        tabs = profile.worksheets
        for t in result["sells"]:
            sync_trade_log(t, action="SELL", worksheet=tabs["log"])
        for b in result["buys"]:
            sync_trade_log(b, action="BUY", worksheet=tabs["log"])
        prices = {pos["ticker"]: pos.get("last_close", pos["entry_price"]) for pos in state.positions}
        sync_positions(state.positions, prices, worksheet=tabs["positions"])
        sync_summary(state.trades, worksheet=tabs["summary"])
        print(f"[{profile.name}] 구글 시트 동기화 완료")
    except Exception as exc:
        print(f"[{profile.name}] 구글 시트 동기화 실패 (계속 진행): {exc}")


def _send_report_only(profile: Any, bar_date: str | None) -> None:
    """신규 일봉이 없는 날(주말·휴장·재실행): 매매·상태 변경 없이 현재 계좌 일일 리포트만 발송한다."""
    tag = f"[{profile.name}]"
    if not profile.params.get("email", True):
        return
    try:
        from paper_trading.account_email import send_account_email

        state = load_account(profile)
        result = {
            "date": bar_date or "-", "buys": [], "sells": [], "cancelled": [],
            "orders": list(state.pending), "candidates": [], "debug": {},
            "equity": state.equity, "cash": round(state.cash, 2), "holdings": len(state.positions),
        }
        send_account_email(profile, result, state)
    except Exception as exc:
        print(f"{tag} 일일 리포트(매매 없음) 발송 실패 (무시): {exc}")


def run_account_daily(key: str, *, dry_run: bool = False, as_of: str | None = None) -> dict[str, Any] | None:
    """PT-2/PT-3/PT-SPY 하루 실행 (GitHub Actions에서 run_paper_trading.py --account로 호출)."""
    from paper_trading.accounts import get_profile
    from paper_trading.market_date import check_run_guard, load_state, mark_processed, market_today
    from paper_trading.runner import load_and_merge_snapshots, snapshot_bar_date

    profile = get_profile(key)
    tag = f"[{profile.name}]"
    print("=" * 60)
    print(f"{tag} 실행 시작 ({profile.version})" + (" — DRY-RUN" if dry_run else ""))
    print("=" * 60)
    if not profile.enabled:
        print(f"{tag} config에서 비활성화됨 — 건너뜀")
        return None

    merged = load_and_merge_snapshots(universe="all")  # PT1_UNIVERSE는 PT-1 전용
    if merged is None or merged.empty:
        print(f"{tag} 랭킹 스냅샷이 없어 건너뜀")
        return None
    bar_date = as_of or snapshot_bar_date(merged)
    if dry_run:
        bar_date = bar_date or str(market_today())
    else:
        ok, reason = check_run_guard(bar_date, profile.data_dir)
        if not ok:
            print(f"{tag} 건너뜀: {reason}")
            _send_report_only(profile, bar_date)
            return None

    rows = merged.to_dict("records")
    state = load_account(profile)
    pre_ctx = {"bar_date": bar_date, "regime": regime_from_rows(rows), "exclude": set(EXCLUDED_TICKERS)}
    tickers = {"SPY"} | {pos["ticker"] for pos in state.positions} | {o["ticker"] for o in state.pending}
    tickers |= set(profile.strategy.prefilter_tickers(rows, pre_ctx, profile.params))
    print(f"{tag} 거래일 {bar_date} | 일봉 다운로드 {len(tickers)}종목")
    try:
        series = fetch_bar_series(sorted(tickers))
    except Exception as exc:
        # 처리 완료로 기록하지 않으므로 다음 실행에서 빠진 일봉을 따라잡는다
        print(f"{tag} 일봉 다운로드 실패 — 건너뜀: {exc}")
        return None
    if "SPY" not in series or series["SPY"].on(bar_date) is None:
        print(f"{tag} SPY {bar_date} 일봉이 없어 건너뜀 (데이터 지연)")
        return None

    last = load_state(profile.data_dir).get("last_processed_bar_date")
    for d in catchup_dates(series["SPY"], last, bar_date):
        print(f"{tag} 빠진 일봉 따라잡기: {d}")
        process_bar(state, profile, d, series)
    result = process_bar(state, profile, bar_date, series, rows=rows)
    _print_result(profile, result)

    if dry_run:
        print(f"{tag} DRY-RUN — 파일 저장·시트·이메일 생략")
        return result

    save_account(profile, state)
    mark_processed(profile.data_dir, bar_date, account=key, version=profile.version)
    _sync_sheets(profile, state, result)
    if not profile.params.get("email", True):
        return result  # 시트에만 기록하는 계좌 (PT-SPY)
    try:
        from paper_trading.account_email import send_account_email

        send_account_email(profile, result, state)
    except Exception as exc:
        print(f"{tag} 이메일 발송 실패: {exc}")
    return result
