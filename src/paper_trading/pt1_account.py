"""PT-1 $5,000 계좌 — PT-1 실제 거래(같은 종목·날짜·가격)를 가상 자본 $5,000 로 다시 계산한다.

PT-1 은 거래마다 수익률(%)만 기록해서 "실제 돈으로 얼마"가 보이지 않았다. 같은 거래를 $5,000 계좌로 따라간다:
빈 자리(최대 PAPER_TRADING_MAX_POSITIONS 종목)마다 남은 현금을 똑같이 나눠 매수하고, 편도 비용 0.1%
(BACKTEST_COST_PER_SIDE — PT-1 백테스트·PT-SPY 와 같은 회계). 같은 날 같은 돈으로 SPY 를 샀을 때와 비교한다.

매매 판단은 바꾸지 않고 기록·표시만 한다. 매번 거래 기록 전체에서 다시 계산하므로 따로 저장할 상태가 없다.
평가에 쓰는 종가는 수정주가라, 보유 중 액면분할이 있으면 그 종목 평가액이 틀릴 수 있다 (PT-1 자체 기록과 같은 한계).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

import pandas as pd

INITIAL_CAPITAL = 5000.0


@dataclass
class Lot:
    """한 번의 매수~매도 (보유 중이면 매도 없음)."""

    ticker: str
    entry_date: str
    entry_price: float
    exit_date: str | None = None
    exit_price: float | None = None
    shares: float = 0.0
    cost: float = 0.0      # 매수에 쓴 돈 (비용 포함)
    proceeds: float = 0.0  # 매도로 받은 돈 (비용 뺀 뒤)

    @property
    def is_open(self) -> bool:
        return self.exit_date is None


@dataclass
class DollarAccount:
    """$5,000 계좌 계산 결과."""

    initial: float
    cash: float
    lots: list[Lot]
    curve: pd.Series      # 날짜별 평가액 (현금 + 보유 종목 종가)
    spy_curve: pd.Series  # 첫 매수일 시가에 같은 돈으로 SPY 를 샀을 때
    last_close: dict[str, float]

    @property
    def start(self) -> str | None:
        return min((lot.entry_date for lot in self.lots), default=None)

    @property
    def holdings(self) -> list[Lot]:
        return [lot for lot in self.lots if lot.is_open]

    @property
    def closed(self) -> list[Lot]:
        return [lot for lot in self.lots if not lot.is_open]

    def price(self, lot: Lot) -> float:
        """보유 종목 평가 가격 (마지막 종가, 없으면 매수가)."""
        return self.last_close.get(lot.ticker, lot.entry_price)

    @property
    def equity(self) -> float:
        return self.cash + sum(lot.shares * self.price(lot) for lot in self.holdings)


def lots_from_records(trades: Sequence[dict[str, Any]], positions: Sequence[dict[str, Any]]) -> list[Lot]:
    """PT-1 trades.json(청산)·positions.json(보유) → 매수~매도 묶음."""
    lots = [Lot(str(t["ticker"]), str(t["entry_date"]), float(t["entry_price"]),
                str(t["exit_date"]), float(t["exit_price"])) for t in trades]
    lots += [Lot(str(p["ticker"]), str(p["entry_date"]), float(p["entry_price"])) for p in positions]
    return lots


def allocate(lots: list[Lot], *, max_positions: int, initial: float = INITIAL_CAPITAL, cost: float = 0.001) -> float:
    """날짜순으로 사고팔며 lot 마다 주식 수·매수금액·매도금액을 채운다. 남은 현금을 돌려준다.

    같은 날에는 다른 종목 매도 → 매수 → 그날 산 종목의 당일 매도 순서 (판 돈으로 빈 자리를 채운다).
    """
    events = []
    for i, lot in enumerate(lots):
        events.append((lot.entry_date, 1, i))
        if lot.exit_date is not None:
            events.append((lot.exit_date, 2 if lot.exit_date == lot.entry_date else 0, i))
    cash, held = initial, 0
    for _, kind, i in sorted(events):
        lot = lots[i]
        if kind == 1:
            lot.cost = cash / max(1, max_positions - held)
            lot.shares = lot.cost / (lot.entry_price * (1 + cost))
            cash -= lot.cost
            held += 1
        else:
            lot.proceeds = lot.shares * float(lot.exit_price) * (1 - cost)
            cash += lot.proceeds
            held -= 1
    return cash


def daily_equity(lots: Sequence[Lot], initial: float, closes: pd.DataFrame) -> pd.Series:
    """첫 매수일부터 날짜별 평가액 = 현금 + 보유 종목 수 × 그날 종가 (종가가 없으면 마지막 종가, 그것도 없으면 매수가)."""
    if not lots or closes.empty:
        return pd.Series(dtype=float)
    start = pd.Timestamp(min(lot.entry_date for lot in lots))
    days = closes.index[closes.index >= start]
    filled = closes.ffill()
    values = []
    for d in days:
        ds = str(d.date())
        cash, held = initial, 0.0
        for lot in lots:
            if lot.entry_date > ds:
                continue
            cash -= lot.cost
            if lot.exit_date is not None and lot.exit_date <= ds:
                cash += lot.proceeds
                continue
            px = filled.at[d, lot.ticker] if lot.ticker in filled.columns else math.nan
            held += lot.shares * (float(px) if px == px else lot.entry_price)
        values.append(cash + held)
    return pd.Series(values, index=days, name="equity")


def spy_equity(spy_open: pd.Series, spy_close: pd.Series, start: str, initial: float, cost: float = 0.001) -> pd.Series:
    """start 날 시가에 initial 만큼 SPY 를 샀을 때 날짜별 평가액."""
    first = spy_open[spy_open.index >= pd.Timestamp(start)].dropna()
    if first.empty:
        return pd.Series(dtype=float)
    shares = initial / (float(first.iloc[0]) * (1 + cost))
    return (spy_close[spy_close.index >= first.index[0]].ffill() * shares).rename("spy")


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    return float((curve / curve.cummax() - 1).min())


def fetch_price_frame(tickers: Sequence[str], start: str) -> pd.DataFrame:
    """평가용 일봉 (티커, 필드) — 첫 매수일까지 덮는 기간만 받는다."""
    from data.fetch import fetch_ohlcv

    years = max(1, math.ceil((date.today() - date.fromisoformat(start)).days / 365) + 1)
    return fetch_ohlcv(sorted(set(tickers) | {"SPY"}), period=f"{years}y")


def build_dollar_account(
    trades: Sequence[dict[str, Any]],
    positions: Sequence[dict[str, Any]],
    *,
    frame: pd.DataFrame | None = None,
    initial: float = INITIAL_CAPITAL,
    max_positions: int | None = None,
    cost: float | None = None,
) -> DollarAccount:
    """PT-1 거래 기록 → $5,000 계좌. frame(일봉 (티커, 필드))이 없으면 내려받는다."""
    from screener import config as cfg

    max_positions = max_positions or int(cfg.PAPER_TRADING_MAX_POSITIONS)
    cost = cfg.BACKTEST_COST_PER_SIDE if cost is None else cost
    lots = lots_from_records(trades, positions)
    cash = allocate(lots, max_positions=max_positions, initial=initial, cost=cost)
    if not lots:
        empty = pd.Series(dtype=float)
        return DollarAccount(initial, cash, lots, empty, empty, {})
    if frame is None:
        frame = fetch_price_frame([lot.ticker for lot in lots], min(lot.entry_date for lot in lots))
    closes = frame.xs("Close", axis=1, level=1)
    opens = frame.xs("Open", axis=1, level=1)
    curve = daily_equity(lots, initial, closes)
    start = min(lot.entry_date for lot in lots)
    spy = spy_equity(opens["SPY"], closes["SPY"], start, initial, cost) if "SPY" in closes.columns else pd.Series(dtype=float)
    last = closes.ffill().iloc[-1] if not closes.empty else pd.Series(dtype=float)
    last_close = {t: float(v) for t, v in last.items() if v == v}
    return DollarAccount(initial, cash, lots, curve, spy, last_close)


def build_accounts(books: dict[str, tuple[Sequence[dict[str, Any]], Sequence[dict[str, Any]]]]) -> dict[str, DollarAccount]:
    """{계좌 이름: (trades, positions)} → 계좌별 $5,000 계좌 (일봉은 한 번만 받는다)."""
    lots = [lot for trades, positions in books.values() for lot in lots_from_records(trades, positions)]
    frame = fetch_price_frame([lot.ticker for lot in lots], min(lot.entry_date for lot in lots)) if lots else None
    return {label: build_dollar_account(trades, positions, frame=frame) for label, (trades, positions) in books.items()}


# ── 표시 (이메일·시트) ──────────────────────────────────────


def summary(acct: DollarAccount) -> dict[str, Any]:
    """요약 숫자 (평가액·수익률·같은 날 SPY·현금·손익·MDD)."""
    equity = acct.equity
    spy_eq = float(acct.spy_curve.iloc[-1]) if not acct.spy_curve.empty else math.nan
    closed = acct.closed
    return {
        "start": acct.start or "-",
        "equity": equity,
        "ret": equity / acct.initial - 1,
        "spy_equity": spy_eq,
        "spy_ret": spy_eq / acct.initial - 1 if spy_eq == spy_eq else math.nan,
        "cash": acct.cash,
        "holdings": len(acct.holdings),
        "realized": sum(lot.proceeds - lot.cost for lot in closed),
        "unrealized": sum(lot.shares * acct.price(lot) - lot.cost for lot in acct.holdings),
        "n_closed": len(closed),
        "win_rate": sum(lot.proceeds > lot.cost for lot in closed) / len(closed) if closed else math.nan,
        "mdd": max_drawdown(acct.curve),
        "spy_mdd": max_drawdown(acct.spy_curve),
    }


def _usd(v: float) -> str:
    return "—" if v != v else f"${v:,.0f}"


def _pct(v: float) -> str:
    return "—" if v != v else f"{v:+.1%}"


def section_html(accounts: Sequence[tuple[str, DollarAccount]]) -> str:
    """PT-1 메일에 넣는 '$5,000 계좌' 섹션."""
    rows, held = "", ""
    for label, acct in accounts:
        if not acct.lots:
            rows += f"<tr style='text-align:center'><td>{label}</td><td colspan='8'>아직 거래 없음 — 현금 $5,000</td></tr>"
            continue
        s = summary(acct)
        diff = s["ret"] - s["spy_ret"] if s["spy_ret"] == s["spy_ret"] else math.nan
        color = "#27ae60" if diff == diff and diff > 0 else "#c0392b"
        rows += (f"<tr style='text-align:center'><td>{label}</td><td>{s['start']}</td><td><b>{_usd(s['equity'])}</b></td>"
                 f"<td>{_pct(s['ret'])}</td><td>{_usd(s['spy_equity'])} ({_pct(s['spy_ret'])})</td>"
                 f"<td style='color:{color}'><b>{'—' if diff != diff else f'{diff * 100:+.1f}%p'}</b></td>"
                 f"<td>{_usd(s['cash'])}</td><td>{s['holdings']}</td><td>{s['mdd']:.1%} ({s['spy_mdd']:.1%})</td></tr>")
        for lot in acct.holdings:
            value = lot.shares * acct.price(lot)
            held += (f"<tr style='text-align:center'><td>{label}</td><td>{lot.ticker}</td><td>{lot.entry_date}</td>"
                     f"<td>{lot.shares:,.2f}</td><td>{_usd(lot.cost)}</td><td>{_usd(value)}</td>"
                     f"<td>{_pct(value / lot.cost - 1)}</td></tr>")
    held_html = ""
    if held:
        held_html = ("<table border='1' style='border-collapse:collapse;width:80%;font-size:13px;margin-top:6px'>"
                     "<tr style='background:#f4f6f7;text-align:center'><th>계좌</th><th>보유 종목</th><th>매수일</th>"
                     f"<th>주식 수</th><th>매수금액</th><th>평가액</th><th>손익</th></tr>{held}</table>")
    return f"""
<h3 style='color:#34495e'>💵 $5,000 계좌 — PT-1 실제 거래를 실제 돈으로</h3>
<table border='1' style='border-collapse:collapse;width:80%;font-size:14px'>
<tr style='background:#f4f6f7;text-align:center'><th>계좌</th><th>시작</th><th>평가액</th><th>수익률</th><th>같은 날 SPY 샀으면</th><th>SPY 대비</th><th>현금</th><th>보유</th><th>MDD (SPY)</th></tr>
{rows}
</table>
{held_html}
<p style='color:#777;font-size:12px'>PT-1이 실제로 사고판 종목·날짜·가격 그대로, 빈 자리(최대 3종목)마다 남은 현금을 똑같이 나눠 산 것으로 계산합니다 (편도 비용 0.1%). 매매 판단은 바뀌지 않습니다.</p>
"""


def sheet_rows(acct: DollarAccount, label: str) -> list[list[Any]]:
    """계좌 탭 내용: 요약 → 보유 종목 → 청산 거래(최근 순) → 일별 평가액."""
    stamp = datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M EST")
    rows: list[list[Any]] = [[f"{label} $5,000 계좌 — PT-1 실제 거래를 가상 자본으로 계산 "
                              "(빈 자리마다 남은 현금을 똑같이 나눠 매수, 편도 비용 0.1%)", stamp], []]
    if not acct.lots:
        return rows + [["아직 거래 없음 — 현금 $5,000"]]
    s = summary(acct)
    rows += [["시작일", "시작 자본", "평가액", "수익률", "같은 날 SPY 샀으면", "SPY 수익률", "SPY 대비", "현금",
              "보유", "실현 손익", "평가 손익", "청산 거래", "승률", "MDD", "SPY MDD"],
             [s["start"], _usd(acct.initial), _usd(s["equity"]), _pct(s["ret"]), _usd(s["spy_equity"]), _pct(s["spy_ret"]),
              "—" if s["spy_ret"] != s["spy_ret"] else f"{(s['ret'] - s['spy_ret']) * 100:+.1f}%p", _usd(s["cash"]),
              s["holdings"], _usd(s["realized"]), _usd(s["unrealized"]), s["n_closed"],
              "—" if s["win_rate"] != s["win_rate"] else f"{s['win_rate']:.0%}", f"{s['mdd']:.1%}", f"{s['spy_mdd']:.1%}"],
             [], ["보유 종목"], ["종목", "매수일", "매수가", "주식 수", "매수금액", "현재가", "평가액", "손익", "수익률"]]
    for lot in acct.holdings:
        px = acct.price(lot)
        value = lot.shares * px
        rows.append([lot.ticker, lot.entry_date, round(lot.entry_price, 2), round(lot.shares, 4), round(lot.cost, 2),
                     round(px, 2), round(value, 2), round(value - lot.cost, 2), _pct(value / lot.cost - 1)])
    rows += [[], ["청산 거래 (최근 순)"], ["종목", "매수일", "매도일", "주식 수", "매수금액", "매도금액", "손익", "수익률"]]
    for lot in sorted(acct.closed, key=lambda x: (x.exit_date or "", x.entry_date), reverse=True):
        rows.append([lot.ticker, lot.entry_date, lot.exit_date, round(lot.shares, 4), round(lot.cost, 2),
                     round(lot.proceeds, 2), round(lot.proceeds - lot.cost, 2), _pct(lot.proceeds / lot.cost - 1)])
    rows += [[], ["일별 평가액"], ["날짜", "평가액", "같은 날 SPY 샀으면"]]
    spy = acct.spy_curve
    for d, v in acct.curve.items():
        rows.append([str(d.date()), round(float(v), 2), round(float(spy[d]), 2) if d in spy.index else ""])
    return rows


def sync_sheet(acct: DollarAccount, label: str, worksheet: str) -> bool:
    """계좌 탭을 통째로 덮어쓴다 (매번 거래 기록 전체에서 다시 계산하므로)."""
    from paper_trading.sheet_sync import _get_or_create_worksheet, _open_sheet

    sheet = _open_sheet()
    if sheet is None:
        return False
    rows = sheet_rows(acct, label)
    ws = _get_or_create_worksheet(sheet, worksheet, rows=max(1000, len(rows) + 50))
    if ws is None:
        return False
    width = max(len(r) for r in rows)
    if ws.row_count < len(rows) or ws.col_count < width:  # 일별 평가액이 쌓여 탭 크기를 넘기면 늘린다
        ws.resize(rows=max(ws.row_count, len(rows) + 50), cols=max(ws.col_count, width))
    ws.clear()
    ws.update([r + [""] * (width - len(r)) for r in rows], value_input_option="USER_ENTERED")
    return True
