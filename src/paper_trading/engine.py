"""페이퍼 트레이딩 매매 의사결정 엔진.

매일 5PM EST 실행 흐름:
1. 보유 종목 현재가 업데이트
2. 매도 판단 (4가지 조건)
3. 후보 선정 (candidate_selector)
4. 매수/교체 판단
5. 포지션 저장 + 로그
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from paper_trading.candidate_selector import select_best_candidate
from paper_trading.portfolio import (
    add_position,
    close_position,
    get_holding_days,
    get_position_return,
    get_worst_position,
    load_positions,
    load_trades,
    save_positions,
    save_trades,
    update_highest_price,
)

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────

from screener.config import (
    PAPER_TRADING_MAX_POSITIONS as MAX_POSITIONS,
    EXIT_PARAMS,
    EXIT_PARAMS_DEFAULT,
    HOLD_WINNERS_ADX_MIN,
    HOLD_WINNERS_BB_PBAND_MAX,
    HOLD_WINNERS_DEFER_FREEZE_DAYS,
    HOLD_WINNERS_MAX_DEFERS,
    HOLD_WINNERS_MIN_CHECKS,
    HOLD_WINNERS_RSI_MAX,
    HOLD_WINNERS_TIGHT_TRAIL,
    HOLD_WINNERS_VOLUME_MULT_MIN,
    HOLD_WINNERS_VOLUME_Z_MIN,
    HIGH_RSI_FLAG_THRESHOLD,
)

CCS_REPLACE_MARGIN = 0.10     # 교체 시 새 후보가 최약 종목 CCS보다 이만큼 높아야 함 (기존 0.05)


# ── Sell Conditions ──────────────────────────────────────────


def _resolve_exit_params(strategy: str) -> dict[str, float]:
    """전략 문자열에서 매칭되는 EXIT_PARAMS 반환, 없으면 DEFAULT."""
    for key, params in EXIT_PARAMS.items():
        if key in strategy:
            return params
    return EXIT_PARAMS_DEFAULT


def check_sell_conditions(
    pos: dict[str, Any],
    current_price: float,
    as_of: str | date | None = None,
) -> tuple[bool, str]:
    """매도 조건 4가지 체크 (전략별 파라미터 적용). (should_sell, reason) 반환."""
    entry_price = pos["entry_price"]
    highest = pos.get("highest_price", entry_price)
    ret = get_position_return(pos, current_price)
    days = get_holding_days(pos, as_of)

    # 전략별 exit 파라미터
    p = _resolve_exit_params(pos.get("strategy", ""))

    # 1. 목표가 도달
    # defer freeze 중이면 시간익절/목표가는 skip (tight trail만 작동)
    last_defer = pos.get("last_defer_date")
    in_freeze = False
    if last_defer and pos.get("defer_count", 0) > 0:
        try:
            from datetime import datetime as _dt
            defer_dt = _dt.strptime(str(last_defer), "%Y-%m-%d").date()
            check_dt = (as_of if isinstance(as_of, date) else
                        _dt.strptime(str(as_of), "%Y-%m-%d").date() if as_of else date.today())
            in_freeze = (check_dt - defer_dt).days < HOLD_WINNERS_DEFER_FREEZE_DAYS
        except Exception:
            in_freeze = False

    if not in_freeze:
        # 1. 목표가 도달
        if ret >= p["profit_target"]:
            return True, f"목표가({ret:+.1%})"

        # 1b. 시간 기반 익절
        if days >= p["time_profit_days"] and ret >= p["time_profit_min"]:
            return True, f"시간익절({days}일,{ret:+.1%})"
    else:
        # freeze 중: 목표가/시간익절 스킵, trailing/손절/장기보유만 체크
        pass

    # 2. 손절
    if ret <= -p["stop_loss"]:
        return True, f"손절({ret:+.1%})"

    # 3. 트레일링 스탑 (defer 시 override 우선 적용)
    # 고점이 매수가를 넘은 적이 있을 때만 활성화 — 한 번도 안 오른 종목은 손절(-stop_loss)에 맡김
    if highest > entry_price:
        drawdown = (current_price - highest) / highest
        trailing_threshold = pos.get("trailing_stop_override") or p["trailing_stop"]
        if drawdown <= -trailing_threshold:
            return True, f"트레일링({drawdown:+.1%})"

    # 4. 장기 보유 + 저조한 수익
    if days > p["max_holding_days"] and ret < p["stale_min_return"]:
        return True, f"장기보유({days}일,{ret:+.1%})"

    return False, ""


# ── Hold-Winners Re-Evaluation ──────────────────────────────


def _get_row_for_ticker(ranked_df: pd.DataFrame | None, ticker: str) -> pd.Series | None:
    """ranked_df에서 ticker에 해당하는 row 반환. 없으면 None."""
    if ranked_df is None or ranked_df.empty:
        return None
    if "티커" in ranked_df.columns:
        mask = ranked_df["티커"] == ticker
    else:
        mask = ranked_df.index == ticker
    rows = ranked_df[mask] if hasattr(mask, "any") else ranked_df.loc[[ticker]] if ticker in ranked_df.index else None
    if rows is None or rows.empty:
        return None
    return rows.iloc[0]


def _safe_num(row: pd.Series, key: str, default: float = 0.0) -> float:
    """row에서 숫자 값 안전 추출. NaN/None은 default."""
    if row is None:
        return default
    val = row.get(key)
    if val is None:
        return default
    try:
        val = float(val)
    except (TypeError, ValueError):
        return default
    if pd.isna(val):
        return default
    return val


def _should_defer_sell(
    pos: dict[str, Any],
    row: pd.Series | None,
    current_price: float,
) -> tuple[bool, dict[str, Any]]:
    """목표가/시간익절 도달 시 모멘텀 재평가. 강하면 defer.

    Returns: (defer, debug_dict)
    - row=None이거나 필수 필드 누락 시 fail-safe로 (False, {...})
    - defer_count >= MAX_DEFERS면 무조건 (False, ...)
    """
    debug: dict[str, Any] = {"ticker": pos.get("ticker"), "current_price": current_price}

    defer_count = int(pos.get("defer_count", 0))
    if defer_count >= HOLD_WINNERS_MAX_DEFERS:
        debug["reason"] = f"max_defers_reached ({defer_count}/{HOLD_WINNERS_MAX_DEFERS})"
        return False, debug

    if row is None:
        debug["reason"] = "no_row_data"
        return False, debug

    rsi = _safe_num(row, "RSI", -1)
    adx = _safe_num(row, "adx", -1)
    vol_mult = _safe_num(row, "거래량돌파배수", 0)
    vol_z = _safe_num(row, "거래량Z(20)", 0)
    ret_5d = _safe_num(row, "5일수익률", -1)
    bb_pband = _safe_num(row, "bollinger_pband", 1.0)

    debug.update({
        "RSI": rsi, "adx": adx, "vol_mult": vol_mult,
        "vol_z": vol_z, "ret_5d": ret_5d, "bb_pband": bb_pband,
        "defer_count": defer_count,
    })

    checks = {
        # RSI_MIN 제거: 과매수(입질 장벽) 취패드마즈 차단만 함. RSI가 낙을수록 상승 여지 더 많음.
        "rsi_not_overbought": rsi < HOLD_WINNERS_RSI_MAX,
        "volume_strong": vol_mult >= HOLD_WINNERS_VOLUME_MULT_MIN or vol_z >= HOLD_WINNERS_VOLUME_Z_MIN,
        "ret_5d_positive": ret_5d > 0,
        "adx_trending": adx >= HOLD_WINNERS_ADX_MIN,
        "bb_not_peaked": bb_pband < HOLD_WINNERS_BB_PBAND_MAX,
    }
    debug["checks"] = checks

    # 5개 조건 중 MIN_CHECKS개 이상 통과 시 defer (기존 all() → 다수결)
    passed = sum(1 for v in checks.values() if v)
    if passed >= HOLD_WINNERS_MIN_CHECKS:
        debug["reason"] = f"checks_passed_{passed}/{len(checks)}"
        return True, debug

    failed = [k for k, v in checks.items() if not v]
    debug["reason"] = f"failed: {', '.join(failed)}"
    # 실패 이유를 명시적으로 로그에 남김 (디버깅용)
    ticker = pos.get("ticker", "?")
    logger.info(
        f"[PaperTrading] DEFER SKIP {ticker}: {', '.join(failed)} "
        f"(RSI={rsi:.1f}, ADX={adx:.1f}, vol_mult={vol_mult:.2f}, "
        f"vol_z={vol_z:.2f}, ret_5d={ret_5d:+.2%}, bb_pband={bb_pband:.2f})"
    )
    return False, debug


def _activate_tight_trail(
    pos: dict[str, Any],
    current_price: float,
    today_str: str,
    debug: dict[str, Any],
) -> None:
    """defer 발동 시 포지션 tight-trail 모드로 전환.
    - trailing_stop_override = HOLD_WINNERS_TIGHT_TRAIL
    - highest_price를 현재가로 리셋 (신규 고점 기준 trailing)
    - defer_count 증가
    - last_defer_date / defer_debug 기록
    """
    pos["trailing_stop_override"] = HOLD_WINNERS_TIGHT_TRAIL
    pos["highest_price"] = max(current_price, pos.get("entry_price", current_price))
    pos["defer_count"] = int(pos.get("defer_count", 0)) + 1
    pos["last_defer_date"] = today_str
    pos["defer_debug"] = debug


def _build_high_rsi_rationale(row: pd.Series | None) -> str:
    """RSI>70 매수에 대한 한국어 정당화 설명."""
    if row is None:
        return ""
    rsi = _safe_num(row, "RSI", 0)
    adx = _safe_num(row, "adx", 0)
    vol_mult = _safe_num(row, "거래량돌파배수", 0)
    pos_52w = _safe_num(row, "52주포지션", 0)
    mom_fit = _safe_num(row, "모멘텀_적합도", 0)

    parts = [f"RSI {rsi:.1f}는 모멘텀 전략 허용 구간(<75)."]
    details: list[str] = []
    if adx > 0:
        details.append(f"ADX={adx:.0f}" + ("(강추세)" if adx >= 25 else ""))
    if vol_mult > 0:
        details.append(f"거래량돌파 {vol_mult:.1f}x")
    if pos_52w > 0:
        details.append(f"52주포지션 {pos_52w*100:.0f}%ile")
    if mom_fit > 0:
        details.append(f"모멘텀_적합도 {mom_fit:.1f}/10")
    if details:
        parts.append(", ".join(details) + " — 과매수 구간에서도 추세 지속 근거 충분.")
    return " ".join(parts)


# ── Replace Logic ────────────────────────────────────────────


def should_replace(candidate_ccs: float, worst_ccs: float) -> bool:
    """보유 3개일 때 교체 여부 판단.
    새 후보의 CCS가 최약 종목보다 충분히 높아야 교체."""
    return candidate_ccs > worst_ccs + CCS_REPLACE_MARGIN


# ── Daily Trading Logic ──────────────────────────────────────


def run_daily_trading(
    ranked_df: pd.DataFrame,
    data_dir: Path | None = None,
    today: str | date | None = None,
    prices: dict[str, float] | None = None,
) -> dict[str, Any]:
    """전체 일일 페이퍼 트레이딩 로직.

    Args:
        ranked_df: 스크리너가 정렬한 최종 DataFrame
        data_dir: 데이터 저장 경로 (None이면 기본 경로)
        today: 오늘 날짜 (테스트용)
        prices: 현재가 dict (None이면 yfinance로 조회)

    Returns:
        실행 결과 요약 dict (sells, buys, holdings 등)
    """
    if today is None:
        today = date.today()
    today_str = str(today)

    positions = load_positions(data_dir)
    trades = load_trades(data_dir)
    result: dict[str, Any] = {"date": today_str, "sells": [], "buys": [], "skipped": None}

    # ── 1. 현재가 조회 ──
    held_tickers = [p["ticker"] for p in positions]
    if prices is None and held_tickers:
        from data.fetch import fetch_latest_prices
        prices = fetch_latest_prices(held_tickers)
    elif prices is None:
        prices = {}

    # 고점 업데이트
    update_highest_price(positions, prices)

    # ── 2. 매도 판단 (Hold-Winners 재평가 포함) ──
    sells_to_execute = []
    defers_applied: list[dict[str, Any]] = []
    defers_skipped: list[dict[str, Any]] = []
    for pos in positions:
        ticker = pos["ticker"]
        current_price = prices.get(ticker, pos["entry_price"])
        should_sell, reason = check_sell_conditions(pos, current_price, today)
        if not should_sell:
            continue

        # 수익성 매도(목표가/시간익절)만 재평가 대상 — 손절/트레일링/장기보유는 즉시 매도
        profit_driven = reason.startswith("목표가") or reason.startswith("시간익절")
        if profit_driven:
            row = _get_row_for_ticker(ranked_df, ticker)
            defer, debug = _should_defer_sell(pos, row, current_price)
            if defer:
                _activate_tight_trail(pos, current_price, today_str, debug)
                defers_applied.append({
                    "ticker": ticker,
                    "trigger_reason": reason,
                    "defer_count": pos["defer_count"],
                    "current_price": current_price,
                    "current_return": (current_price - pos["entry_price"]) / pos["entry_price"],
                    "debug": debug,
                })
                logger.info(
                    f"[PaperTrading] DEFER {ticker} @ {current_price:.2f} "
                    f"({reason} → tight trail {HOLD_WINNERS_TIGHT_TRAIL*100:.1f}%, "
                    f"defer {pos['defer_count']}/{HOLD_WINNERS_MAX_DEFERS})"
                )
                continue
            else:
                defers_skipped.append({
                    "ticker": ticker,
                    "trigger_reason": reason,
                    "current_price": current_price,
                    "current_return": (current_price - pos["entry_price"]) / pos["entry_price"],
                    "debug": debug,
                })

        sells_to_execute.append((ticker, current_price, reason))

    result["defers"] = defers_applied
    result["defers_skipped"] = defers_skipped

    for ticker, price, reason in sells_to_execute:
        trade = close_position(
            positions, trades,
            ticker=ticker, exit_price=price, exit_date=today_str, reason=reason,
        )
        if trade:
            result["sells"].append(trade)
            logger.info(f"[PaperTrading] SELL {ticker} @ {price:.2f} ({reason})")

    # ── 3. 후보 선정 ──
    candidate, debug = select_best_candidate(ranked_df, positions)
    result["selection_debug"] = debug

    # 약세장 최대 포지션 축소
    regime = debug.get("regime", "neutral")
    effective_max_positions = MAX_POSITIONS if regime != "bear" else max(1, MAX_POSITIONS - 1)

    if candidate is None:
        logger.info("[PaperTrading] 오늘 매수 후보 없음")
        result["skipped"] = "후보_없음"
        save_positions(positions, data_dir)
        save_trades(trades, data_dir)
        result["holdings"] = len(positions)
        return result

    # ── 4. 매수/교체 판단 ──
    # 매수 전에 upside 예측 + RSI/근거 조회
    cand_row = _get_row_for_ticker(ranked_df, candidate["ticker"])
    entry_rsi = _safe_num(cand_row, "RSI", 0.0)
    high_rsi_flag = entry_rsi > HIGH_RSI_FLAG_THRESHOLD
    high_rsi_rationale = _build_high_rsi_rationale(cand_row) if high_rsi_flag else ""

    try:
        from paper_trading.upside_model import predict_upside
        upside = predict_upside(
            candidate["strategy"], candidate["ccs_score"], candidate["star_rating"],
        )
    except Exception as exc:
        logger.warning(f"[PaperTrading] upside predict failed: {exc}")
        upside = {"n": 0}

    entry_price = candidate["entry_price"]
    upside_payload = {
        "entry_rsi": entry_rsi,
        "high_rsi_flag": high_rsi_flag,
        "high_rsi_rationale": high_rsi_rationale,
        "upside_p25": upside.get("p25", 0.0),
        "upside_p50": upside.get("p50", 0.0),
        "upside_p75": upside.get("p75", 0.0),
        "upside_p90": upside.get("p90", 0.0),
        "upside_price_p25": entry_price * (1 + upside.get("p25", 0.0)),
        "upside_price_p50": entry_price * (1 + upside.get("p50", 0.0)),
        "upside_price_p75": entry_price * (1 + upside.get("p75", 0.0)),
        "upside_price_p90": entry_price * (1 + upside.get("p90", 0.0)),
        "upside_n": upside.get("n", 0),
        "upside_days_to_peak": upside.get("mean_days_to_peak", 0.0),
        "upside_level": upside.get("level", ""),
    }

    position_upside_kwargs = dict(
        entry_rsi=entry_rsi,
        predicted_upside_p25=upside.get("p25", 0.0),
        predicted_upside_p50=upside.get("p50", 0.0),
        predicted_upside_p75=upside.get("p75", 0.0),
        predicted_upside_p90=upside.get("p90", 0.0),
        upside_sample_size=upside.get("n", 0),
        upside_days_to_peak=upside.get("mean_days_to_peak", 0.0),
        upside_level=upside.get("level", ""),
        upside_bucket_key=upside.get("bucket_key", ""),
    )

    if len(positions) < effective_max_positions:
        # 빈 슬롯 있음 → 바로 매수
        add_position(
            positions,
            ticker=candidate["ticker"],
            entry_price=entry_price,
            entry_date=today_str,
            strategy=candidate["strategy"],
            star_rating=candidate["star_rating"],
            ccs_score=candidate["ccs_score"],
            sector=candidate["sector"],
            **position_upside_kwargs,
        )
        buy_entry = {
            "ticker": candidate["ticker"],
            "entry_date": today_str,
            "entry_price": entry_price,
            "sector": candidate["sector"],
            "strategy": candidate["strategy"],
            "star_rating": candidate["star_rating"],
            "ccs_score": candidate["ccs_score"],
            "reason": "신규매수",
            "vol_ratio": candidate.get("vol_ratio", 0),
            "vol_ma20": candidate.get("vol_ma20", 0),
            **upside_payload,
        }
        result["buys"].append(buy_entry)
        logger.info(
            f"[PaperTrading] BUY {candidate['ticker']} @ {entry_price:.2f} "
            f"(CCS={candidate['ccs_score']:.4f}, RSI={entry_rsi:.0f}, "
            f"upside P50={upside.get('p50', 0)*100:+.0f}% P75={upside.get('p75', 0)*100:+.0f}%)"
        )
    elif len(positions) >= effective_max_positions:
        # 풀슬롯 → 교체 검토
        worst = get_worst_position(positions, prices, ranked_df)
        if worst and should_replace(candidate["ccs_score"], worst.get("ccs_score", 0)):
            worst_ticker = worst["ticker"]
            worst_price = prices.get(worst_ticker, worst["entry_price"])

            # 최약 종목 매도
            trade = close_position(
                positions, trades,
                ticker=worst_ticker, exit_price=worst_price,
                exit_date=today_str, reason=f"교체→{candidate['ticker']}",
            )
            if trade:
                result["sells"].append(trade)

            # 새 종목 매수
            add_position(
                positions,
                ticker=candidate["ticker"],
                entry_price=entry_price,
                entry_date=today_str,
                strategy=candidate["strategy"],
                star_rating=candidate["star_rating"],
                ccs_score=candidate["ccs_score"],
                sector=candidate["sector"],
                **position_upside_kwargs,
            )
            buy_entry = {
                "ticker": candidate["ticker"],
                "entry_date": today_str,
                "entry_price": entry_price,
                "sector": candidate["sector"],
                "strategy": candidate["strategy"],
                "star_rating": candidate["star_rating"],
                "ccs_score": candidate["ccs_score"],
                "reason": f"교체({worst_ticker}→{candidate['ticker']})",
                "vol_ratio": candidate.get("vol_ratio", 0),
                "vol_ma20": candidate.get("vol_ma20", 0),
                **upside_payload,
            }
            result["buys"].append(buy_entry)
            logger.info(
                f"[PaperTrading] REPLACE {worst_ticker} → {candidate['ticker']} "
                f"@ {entry_price:.2f}"
            )
        else:
            result["skipped"] = "교체_미달"
            logger.info("[PaperTrading] 교체 기준 미달 — 매수 안 함")

    # ── 5. 저장 ──
    save_positions(positions, data_dir)
    save_trades(trades, data_dir)
    result["holdings"] = len(positions)

    return result
