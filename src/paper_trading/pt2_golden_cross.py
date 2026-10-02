"""PT-2 골든크로스 스윙 (v0).

진입: 골든크로스 목록(일봉 EMA20/50, 주봉 SMA10/40, 월봉 SMA3/10) 중
      교차 직후(기본값) 종목을 필터링한 뒤 점수순. 다음날 시가 매수.
청산: 초기 손절(매수가 - 2×ATR) / 추세 이탈(데드크로스, 종가<EMA50 N일 연속) /
      최고 종가 +5% 이후에만 트레일링(최고 종가 - 3×ATR) /
      30일(달력일)부터 매일 추세 체크 → 통과하면 기간 제한 없이 보유, 실패하면 매도.
목표가는 두지 않는다 (추세가 살아 있으면 계속 들고 간다).
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any, Mapping

from paper_trading.golden_cross import extract_golden_cross
from paper_trading.indicators import PercentileRanker, is_num, num

RANK_COLUMNS = ("ema_gap_50_200", "52주포지션", "adx", "거래량돌파배수")


def _entry_type(gap: float) -> str:
    return "crossed" if gap >= 0 else "imminent"


def _close(row: Mapping[str, Any]) -> float:
    c = num(row, "close")
    return c if is_num(c) and c > 0 else num(row, "현재가격")


def _filter_reason(row: Mapping[str, Any] | None, params: dict[str, Any]) -> str | None:
    """필터 통과면 None, 아니면 탈락 사유."""
    if row is None:
        return "스냅샷 없음"
    close = _close(row)
    if not is_num(close) or close < params["min_price"]:
        return "저가주"
    if num(row, "최근20일평균거래대금", 0.0) < params["min_dollar_volume"]:
        return "유동성"
    days = num(row, "days_to_next_earnings")
    if is_num(days) and days <= params["earnings_buffer_days"]:
        return "어닝임박"
    if (
        num(row, "RSI", 50.0) >= params["rsi_max"]
        or num(row, "bollinger_pband", 0.5) >= params["bb_max"]
        or num(row, "5일수익률", 0.0) >= params["ret5_max"]
    ):
        return "과열"
    if params["require_above_ema200"]:
        ema200 = num(row, "ema200", 0.0)
        if ema200 <= 0 or close <= ema200:
            return "200일선 아래"
    if num(row, "atr_value", 0.0) <= 0:
        return "ATR 없음"
    return None


def prefilter_tickers(rows: list[Mapping[str, Any]], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """PT-2 후보 선정은 스냅샷만 쓰므로 추가 일봉이 필요 없다."""
    return []


def select_candidates(
    rows: list[Mapping[str, Any]],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """골든크로스 후보를 점수 내림차순으로 반환한다."""
    debug: dict[str, Any] = {"regime": ctx.get("regime")}
    if params["block_new_in_bear"] and ctx.get("regime") == "bear":
        debug["skip"] = "약세장(SPY<EMA200) 신규 진입 중단"
        return [], debug

    gc_items = extract_golden_cross(rows)
    by_ticker = {str(r.get("티커", "")).strip(): r for r in rows}
    ranker = PercentileRanker(rows, RANK_COLUMNS)
    w = params["rank_weights"]
    exclude = ctx.get("exclude", set())
    rejections: Counter = Counter()
    out: list[dict[str, Any]] = []

    for g in gc_items:
        ticker = g["ticker"]
        if ticker in exclude:
            rejections["보유·제외"] += 1
            continue
        tfs = [
            (tf, gap) for tf, gap in g["tf_gaps"]
            if tf in params["timeframes"] and is_num(gap) and _entry_type(gap) in params["entry_types"]
        ]
        if not tfs:
            rejections["진입유형"] += 1
            continue
        row = by_ticker.get(ticker)
        reason = _filter_reason(row, params)
        if reason:
            rejections[reason] += 1
            continue

        min_gap = min(abs(gap) for _, gap in tfs)
        has_daily = any(tf == "일봉" for tf, _ in tfs)
        score = (
            w["tf"] * len(tfs) / 3
            + w["daily"] * (1.0 if has_daily else 0.0)
            + w["gap"] * (1.0 - min(min_gap / 0.05, 1.0))
            + w["trend"] * ranker.rank("ema_gap_50_200", row)
            + w["pos52w"] * ranker.rank("52주포지션", row)
            + w["adx"] * ranker.rank("adx", row)
            + w["volume"] * ranker.rank("거래량돌파배수", row)
        )
        label = ", ".join(f"{tf}{'직후' if gap >= 0 else '임박'}" for tf, gap in tfs)
        out.append({
            "ticker": ticker,
            "score": round(score, 4),
            "strategy": f"골든크로스({label})",
            "setup": "golden_cross",
            "sector": str(row.get("섹터", "")),
            "star": str(row.get("매수적합도_표시", "")),
            "close": _close(row),
            "atr": num(row, "atr_value"),
            "meta": {"gc_timeframes": [tf for tf, _ in tfs], "gc_min_gap": round(min_gap, 4)},
        })

    out.sort(key=lambda c: c["score"], reverse=True)
    debug.update({"골든크로스": len(gc_items), "통과": len(out), "탈락": dict(rejections)})
    return out, debug


def init_position(pos: dict[str, Any], order: dict[str, Any], params: dict[str, Any]) -> None:
    pos["stop_price"] = round(float(pos["entry_price"]) - params["stop_atr_mult"] * float(order["atr"]), 4)
    pos["stop_kind"] = "손절"
    pos["ema_aligned_seen"] = False
    pos["trend_checks"] = 0


def on_bar_close(pos: dict[str, Any], ind: dict[str, float], bar_date: str, params: dict[str, Any]) -> str | None:
    """장 마감 후 호출. 청산 사유를 반환하면 다음날 시가에 매도한다. 트레일링 손절선도 여기서 올린다."""
    close, ema20, ema50 = ind.get("close"), ind.get("ema20"), ind.get("ema50")

    # 주봉/월봉 교차로 들어와 일봉 정배열이 아닌 종목은 정배열을 한 번 본 뒤부터 데드크로스를 적용
    if is_num(ema20) and is_num(ema50):
        if ema20 >= ema50:
            pos["ema_aligned_seen"] = True
        elif pos.get("ema_aligned_seen"):
            return "추세이탈(데드크로스)"
    streak = int(ind.get("below_ema50_streak") or 0)
    if streak >= params["below_ema50_days"]:
        return f"추세이탈(50일선 아래 {streak}일)"

    days = (date.fromisoformat(bar_date) - date.fromisoformat(pos["entry_date"])).days
    after_hold = days >= params["hold_days"]
    if after_hold:
        adx, ret20 = ind.get("adx"), ind.get("ret_20d")
        strong = (
            all(is_num(x) for x in (close, ema20, ema50, adx, ret20))
            and close > ema20 > ema50
            and adx >= params["extend_adx_min"]
            and ret20 > 0
        )
        if not strong:
            return f"보유 {days}일 추세약화"
        pos["trend_checks"] = int(pos.get("trend_checks", 0)) + 1

    atr = ind.get("atr") if is_num(ind.get("atr")) else pos.get("atr_at_signal")
    entry = float(pos["entry_price"])
    highest_close = float(pos.get("highest_close", entry))
    if is_num(atr) and highest_close >= entry * (1 + params["trail_activate_pct"]):
        mult = params["trail_atr_mult_after_hold"] if after_hold else params["trail_atr_mult"]
        level = highest_close - mult * float(atr)
        if level > float(pos.get("stop_price") or 0):
            pos["stop_price"] = round(level, 4)
            pos["stop_kind"] = "트레일링"
    return None
