"""PT-3 일봉 단타 (v0, 최대 10거래일).

기존 스크리너의 일봉 지표를 재료로 1~10일용 셋업을 새로 구성했다.
  셋업 A 눌림목: 종가>EMA50>EMA200, RSI 35~50 또는 볼린저<0.3, 5일 수익<0,
               반전 확인(강세잉걸핑·모닝스타·망치형 캔들, 또는 종가가 전일 고가 돌파)
  셋업 B 돌파:  변동성 압축(ATR%/1년 중앙값≤0.9), 종가가 직전 20일 고가 돌파,
               거래량 1.5배+, ADX≥20, 종가>EMA50, 과열(RSI≥80, 5일 +15%) 제외
청산: 손절 -1.5×ATR, 목표 +2×ATR (일봉 고가/저가로 체결), 고가가 +1×ATR에 닿으면 본절,
      셋업 완료(A: RSI≥60 또는 볼린저≥0.8 / B: 돌파선 - 0.5×ATR 아래 마감), 10거래일 시간 청산
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from paper_trading.indicators import PercentileRanker, is_num, num

REVERSAL_PATTERNS = ("강세잉걸핑", "모닝스타")
RANK_COLUMNS = ("ema_gap_50_200", "20일수익률", "52주포지션", "거래량돌파배수")
SETUP_LABELS = {"pullback": "단타-눌림목", "breakout": "단타-돌파"}


def _close(row: Mapping[str, Any]) -> float:
    c = num(row, "close")
    return c if is_num(c) and c > 0 else num(row, "현재가격")


def _common_reason(row: Mapping[str, Any], params: dict[str, Any]) -> str | None:
    close = _close(row)
    if not is_num(close) or close < params["min_price"]:
        return "저가주"
    if num(row, "최근20일평균거래대금", 0.0) < params["min_dollar_volume"]:
        return "유동성"
    days = num(row, "days_to_next_earnings")
    if is_num(days) and days <= params["earnings_buffer_days"]:
        return "어닝임박"
    if num(row, "atr_value", 0.0) <= 0:
        return "ATR 없음"
    return None


def _pullback_pre(row: Mapping[str, Any], params: dict[str, Any]) -> bool:
    close, e50, e200 = _close(row), num(row, "ema50"), num(row, "ema200")
    if not (is_num(e50) and is_num(e200) and e200 > 0 and close > e50 > e200):
        return False
    rsi, bb = num(row, "RSI"), num(row, "bollinger_pband")
    dipped = (is_num(rsi) and params["pullback_rsi_min"] <= rsi <= params["pullback_rsi_max"]) or (
        is_num(bb) and bb < params["pullback_bb_max"]
    )
    return dipped and num(row, "5일수익률", 0.0) < 0


def _pullback_confirmed(row: Mapping[str, Any], bars: list[dict[str, Any]]) -> bool:
    patterns = str(row.get("일봉패턴", ""))
    if any(p in patterns for p in REVERSAL_PATTERNS) or num(row, "저점반전캔들", 0.0) >= 1:
        return True
    return len(bars) >= 2 and bars[-1]["close"] > bars[-2]["high"]


def _breakout_pre(row: Mapping[str, Any], params: dict[str, Any]) -> bool:
    close, e50, comp = _close(row), num(row, "ema50"), num(row, "변동성압축")
    return (
        is_num(e50) and close > e50
        and is_num(comp) and comp <= params["breakout_compression_max"]
        and num(row, "거래량돌파배수", 0.0) >= params["breakout_volume_min"]
        and num(row, "adx", 0.0) >= params["breakout_adx_min"]
        and num(row, "RSI", 50.0) < params["breakout_rsi_max"]
        and num(row, "5일수익률", 0.0) < params["breakout_ret5_max"]
    )


def _breakout_level(bars: list[dict[str, Any]], params: dict[str, Any]) -> float | None:
    """마지막 종가가 직전 N일 고가를 넘었으면 그 고가(돌파선), 아니면 None."""
    n = params["breakout_days"]
    if len(bars) < n + 1:
        return None
    level = max(b["high"] for b in bars[-n - 1:-1])
    return level if bars[-1]["close"] > level else None


def prefilter_tickers(rows: list[Mapping[str, Any]], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """스냅샷 조건만으로 걸러서 일봉이 필요한 종목만 반환 (실거래 다운로드 최소화)."""
    exclude = ctx.get("exclude", set())
    out = []
    for row in rows:
        ticker = str(row.get("티커", "")).strip()
        if not ticker or ticker in exclude or _common_reason(row, params):
            continue
        if _pullback_pre(row, params) or _breakout_pre(row, params):
            out.append(ticker)
    return out


def select_candidates(
    rows: list[Mapping[str, Any]],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """단타 후보를 점수 내림차순으로 반환한다. ctx['series']의 일봉으로 돌파·반전을 확인한다."""
    regime = ctx.get("regime")
    series = ctx.get("series") or {}
    bar_date = ctx.get("bar_date")
    exclude = ctx.get("exclude", set())
    allow_breakout = not (params["block_breakout_in_bear"] and regime == "bear")
    ranker = PercentileRanker(rows, RANK_COLUMNS)
    w = params["rank_weights"]
    rejections: Counter = Counter()
    out: list[dict[str, Any]] = []

    for row in rows:
        ticker = str(row.get("티커", "")).strip()
        if not ticker or ticker in exclude or _common_reason(row, params):
            continue
        s = series.get(ticker)
        bars = s.upto(bar_date, params["breakout_days"] + 1) if s and bar_date else []
        if bars and bars[-1]["date"] != bar_date:
            bars = []

        setup: str | None = None
        meta: dict[str, Any] = {}
        for name in params["setups"]:
            if name == "breakout" and allow_breakout and _breakout_pre(row, params):
                if bars:
                    level = _breakout_level(bars, params)
                    if level is None:
                        rejections["돌파 미확인"] += 1
                        continue
                    meta = {"breakout_level": round(level, 4)}
                elif num(row, "10일고점괴리", -1.0) < 0:
                    continue
                setup = "breakout"
                break
            if name == "pullback" and _pullback_pre(row, params):
                if not _pullback_confirmed(row, bars):
                    rejections["반전 미확인"] += 1
                    continue
                setup = "pullback"
                break
        if setup is None:
            continue

        score = (
            w["trend"] * ranker.rank("ema_gap_50_200", row)
            + w["rs"] * ranker.rank("20일수익률", row)
            + w["pos52w"] * ranker.rank("52주포지션", row)
            + w["volume"] * ranker.rank("거래량돌파배수", row)
        )
        out.append({
            "ticker": ticker,
            "score": round(score, 4),
            "strategy": SETUP_LABELS[setup],
            "setup": setup,
            "sector": str(row.get("섹터", "")),
            "star": str(row.get("매수적합도_표시", "")),
            "close": float(bars[-1]["close"]) if bars else _close(row),
            "atr": num(row, "atr_value"),
            "meta": meta,
        })

    out.sort(key=lambda c: c["score"], reverse=True)
    debug = {
        "regime": regime,
        "돌파셋업": "허용" if allow_breakout else "약세장 중단",
        "통과": len(out),
        "탈락": dict(rejections),
    }
    return out, debug


def init_position(pos: dict[str, Any], order: dict[str, Any], params: dict[str, Any]) -> None:
    atr = float(order["atr"])
    entry = float(pos["entry_price"])
    pos["stop_price"] = round(entry - params["stop_atr_mult"] * atr, 4)
    pos["target_price"] = round(entry + params["target_atr_mult"] * atr, 4)
    pos["stop_kind"] = "손절"


def on_bar_close(pos: dict[str, Any], ind: dict[str, float], bar_date: str, params: dict[str, Any]) -> str | None:
    """장 마감 후 호출. 청산 사유를 반환하면 다음날 시가에 매도한다. 본절 손절선도 여기서 올린다."""
    entry = float(pos["entry_price"])
    atr = float(pos["atr_at_signal"])
    if (
        float(pos.get("highest_price", entry)) >= entry + params["breakeven_atr_mult"] * atr
        and float(pos.get("stop_price") or 0) < entry
    ):
        pos["stop_price"] = round(entry, 4)
        pos["stop_kind"] = "본절"

    setup = pos.get("setup")
    if setup == "pullback":
        rsi, bb = ind.get("rsi"), ind.get("bb_pband")
        if (is_num(rsi) and rsi >= params["pullback_exit_rsi"]) or (is_num(bb) and bb >= params["pullback_exit_bb"]):
            return "셋업완료(눌림 회복)"
    elif setup == "breakout":
        level, close = pos.get("breakout_level"), ind.get("close")
        if is_num(level) and is_num(close) and close < float(level) - params["breakout_fail_atr"] * atr:
            return "돌파실패"

    if int(pos.get("bars_held", 0)) + 1 >= params["max_hold_bars"]:
        return f"시간청산({params['max_hold_bars']}거래일)"
    return None
