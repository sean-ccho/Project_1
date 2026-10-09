"""PT-SPY: SPY 보유 기준 계좌 — 다른 계좌가 이겨야 할 '그냥 시장을 샀다면'의 실거래 기록.

규칙: 첫 실행에서 SPY 를 다음날 시가에 전액 매수하고 계속 보유한다 (손절·목표가·청산 없음).
가상 자본 $5,000, 편도 비용 0.1% (PT-2·PT-3 와 같은 계좌 엔진·회계). 이메일은 보내지 않고 시트에만 기록한다.

2026-10-09: 10년 백테스트·워크포워드 결과 PT-1 규칙은 어떤 종목 선택·구조로도 SPY 를 넘지 못해
(TRIAL_LOG ccs_placebo_10y · wfo_structure_10y) 지수 기준선을 페이퍼에서도 나란히 기록하기로 했다.
"""

from __future__ import annotations

from typing import Any, Mapping

from paper_trading.indicators import is_num, num

TICKER = "SPY"
PARAMS: dict[str, Any] = {
    "version": "pt-spy-v1",
    "data_dir": "data/paper_trading/pt_spy",
    "initial_capital": 5000.0,
    "max_positions": 1,
    "max_daily_buys": 1,
    "max_same_sector": 1,
    "cost_per_side": 0.001,
    "max_gap_up": 1.0,        # 갭 상승으로 매수 취소하지 않는다
    "min_alloc_ratio": 0.5,
    "email": False,
}
WORKSHEETS = {"log": "페이퍼SPY_거래로그", "positions": "페이퍼SPY_포지션현황", "summary": "페이퍼SPY_성과요약"}


def _spy_close(rows: list[Mapping[str, Any]], ctx: dict[str, Any]) -> float:
    """오늘 SPY 종가 (스냅샷 행 → 없으면 일봉)."""
    for r in rows:
        if str(r.get("티커", "")).strip() == TICKER:
            c = num(r, "close")
            if not (is_num(c) and c > 0):
                c = num(r, "현재가격")
            if is_num(c) and c > 0:
                return float(c)
    s = (ctx.get("series") or {}).get(TICKER)
    bar = s.on(ctx["bar_date"]) if s is not None and ctx.get("bar_date") else None
    return float(bar["close"]) if bar else float("nan")


def prefilter_tickers(rows: list[Mapping[str, Any]], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """SPY 일봉은 계좌 엔진이 항상 받는다."""
    return []


def select_candidates(
    rows: list[Mapping[str, Any]],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """늘 SPY 하나. 이미 보유 중이면 엔진이 빈 슬롯이 없어 주문하지 않는다."""
    close = _spy_close(rows, ctx)
    if not (is_num(close) and close > 0):
        return [], {"skip": "SPY 종가 없음"}
    return [{
        "ticker": TICKER, "score": 1.0, "strategy": "SPY 보유", "setup": "index",
        "sector": "", "star": "", "close": close, "atr": 0.0, "meta": {},
    }], {"regime": ctx.get("regime")}


def init_position(pos: dict[str, Any], order: dict[str, Any], params: dict[str, Any]) -> None:
    """손절·목표가 없음."""
    pos["stop_price"] = None
    pos["target_price"] = None


def on_bar_close(pos: dict[str, Any], ind: dict[str, float], bar_date: str, params: dict[str, Any]) -> str | None:
    """청산하지 않는다."""
    return None
