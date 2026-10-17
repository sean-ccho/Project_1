"""페이퍼 트레이딩 구글 시트 3탭 동기화.

기존 exporter.py의 _open_sheet() 패턴 재활용.
- [페이퍼_거래로그] — 매수/매도 기록. PT-1·1S·1N 은 매일 통째로 다시 쓴다 ($5,000 계좌 기준 주식 수·금액·손익, 2026-10-16).
  PT-SPY 는 예전처럼 한 줄씩 append (sync_trade_log)
- [페이퍼_포지션현황] — 현재 보유 종목 덮어쓰기
- [페이퍼_성과요약] — 누적 성과 집계 덮어쓰기

매도 거래마다 "같은 기간 SPY를 들고 있었으면" 수익률을 붙인다 (SPY동기간%, SPY대비%p).
엣지가 없는 전략은 거래 수익이 좋아 보여도 SPY대비가 0 근처로 나온다.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 워크시트 이름
from screener.config import (
    PAPER_TRADING_WORKSHEET_LOG as WORKSHEET_LOG,
    PAPER_TRADING_WORKSHEET_POSITIONS as WORKSHEET_POSITIONS,
    PAPER_TRADING_WORKSHEET_SUMMARY as WORKSHEET_SUMMARY,
)

WORKSHEET_BACKTEST = "백테스트_결과"

_SPY_CLOSE = None  # 실행당 한 번만 받는다 (계좌들이 공유)


def _spy_close():
    """SPY 종가 시계열 (실패하면 빈 Series)."""
    global _SPY_CLOSE
    if _SPY_CLOSE is None:
        from paper_trading.benchmarks import load_spy_close
        _SPY_CLOSE = load_spy_close()
    return _SPY_CLOSE


def _spy_return(trade: dict[str, Any]) -> float | None:
    """거래와 같은 기간(매수일 종가 → 매도일 종가) SPY 수익률."""
    from paper_trading.benchmarks import spy_window_return
    return spy_window_return(_spy_close(), trade.get("entry_date", ""), trade.get("exit_date", ""))


def _open_sheet():
    """exporter.py와 동일한 인증 로직으로 스프레드시트 핸들 반환."""
    from screener.exporter import _open_sheet as _exporter_open_sheet
    return _exporter_open_sheet()


def _get_or_create_worksheet(sheet, name: str, rows: int = 1000, cols: int = 20):
    """워크시트가 없으면 생성."""
    try:
        import gspread
        try:
            return sheet.worksheet(name)
        except gspread.exceptions.WorksheetNotFound:
            return sheet.add_worksheet(title=name, rows=str(rows), cols=str(cols))
    except Exception as exc:
        logger.error(f"[SheetSync] 워크시트 '{name}' 접근 실패: {exc}")
        return None


# ── [페이퍼_거래로그] — $5,000 계좌 기준으로 통째로 다시 쓰기 (PT-1·1S·1N) ─────

DOLLAR_LOG_HEADERS = [
    "날짜", "종목", "섹터", "액션", "주식 수", "가격", "금액", "손익($)", "수익률", "보유일", "거래 후 현금",
    "전략", "별점", "CCS점수", "사유", "SPY동기간%", "SPY대비%p",
]


def _pct_text(v: float | None) -> str:
    return "" if v is None or v != v else f"{v:+.1%}"


def dollar_log_events(trades: list[dict[str, Any]], positions: list[dict[str, Any]], *,
                      max_positions: int | None = None, spy_fn=None) -> list[dict[str, Any]]:
    """거래로그 줄을 숫자로 — $5,000 계좌(pt1_account)와 같은 배분으로 주식 수·금액·손익·거래 후 현금.

    줄 순서 = 계좌 배분 순서 (날짜 → 같은 날은 다른 종목 매도 · 매수 · 당일 매도). 수익률 = 매도금액 ÷ 매수금액 − 1
    (편도 비용 0.1% 포함, $5,000 계좌 탭과 같은 값). vs_spy = 그 수익률 − 같은 기간 SPY. 시트 탭과 웹 거래로그가 같이 쓴다.
    """
    from paper_trading.pt1_account import allocation_events, dollar_lots

    spy_fn = spy_fn or _spy_return
    recs = list(trades) + list(positions)
    lots = dollar_lots(trades, positions, max_positions=max_positions)
    out: list[dict[str, Any]] = []
    for day, side, i in allocation_events(lots):
        lot, t = lots[i], recs[i]
        ev = {"date": day, "side": side, "ticker": lot.ticker, "sector": t.get("sector", "") or "",
              "shares": int(lot.shares), "strategy": t.get("strategy", "") or "", "star": t.get("star_rating", "") or "",
              "ccs": t.get("ccs_score", t.get("ccs")), "entry_date": lot.entry_date, "exit_date": lot.exit_date}
        if side == "buy":
            ev.update(price=lot.entry_price, amount=lot.cost, pnl=None, ret=None, days=None, cash_after=lot.cash_after_buy,
                      reason="신규매수" if lot.shares > 0 else "0주 — 자리 몫으로 1주도 못 삼", spy=None, vs_spy=None)
        else:
            ret = lot.proceeds / lot.cost - 1 if lot.cost else None
            spy = spy_fn(t)
            ev.update(price=float(lot.exit_price), amount=lot.proceeds, pnl=lot.proceeds - lot.cost, ret=ret,
                      days=t.get("holding_days"), cash_after=lot.cash_after_sell, reason=t.get("exit_reason", t.get("reason", "")),
                      spy=spy, vs_spy=ret - spy if ret is not None and spy is not None else None)
        out.append(ev)
    return out


def dollar_log_rows(trades: list[dict[str, Any]], positions: list[dict[str, Any]], *,
                    max_positions: int | None = None, spy_fn=None) -> list[list[Any]]:
    """거래로그 탭 전체 (머리글 포함) — dollar_log_events 를 시트 글자로."""
    rows: list[list[Any]] = [list(DOLLAR_LOG_HEADERS)]
    for e in dollar_log_events(trades, positions, max_positions=max_positions, spy_fn=spy_fn):
        sell = e["side"] == "sell"
        rows.append([
            e["date"], e["ticker"], e["sector"], "🔴 SELL" if sell else "🟢 BUY", e["shares"], round(e["price"], 2),
            round(e["amount"], 2), round(e["pnl"], 2) if sell else "", _pct_text(e["ret"]) if sell else "",
            (e["days"] if e["days"] is not None else "") if sell else "", round(e["cash_after"], 2),
            e["strategy"], e["star"], e["ccs"] if e["ccs"] is not None else "", e["reason"],
            _pct_text(e["spy"]) if sell else "", f"{e['vs_spy'] * 100:+.1f}" if e["vs_spy"] is not None else "",
        ])
    return rows


def rebuild_trade_log(trades: list[dict[str, Any]], positions: list[dict[str, Any]],
                      worksheet: str = WORKSHEET_LOG) -> bool:
    """거래로그 탭을 $5,000 계좌 기준으로 통째로 다시 쓴다 (같은 탭 — 시트 주소(gid)는 그대로).

    예전 열 모양의 숫자 서식(%)이 새 열에 남지 않게 서식도 지운다.
    """
    sheet = _open_sheet()
    if sheet is None:
        return False
    ws = _get_or_create_worksheet(sheet, worksheet)
    if ws is None:
        return False
    try:
        rows = dollar_log_rows(trades, positions)
        width = len(DOLLAR_LOG_HEADERS)
        if ws.row_count < len(rows) + 10 or ws.col_count < width:
            ws.resize(rows=max(ws.row_count, len(rows) + 50), cols=max(ws.col_count, width))
        ws.clear()
        sheet.batch_update({"requests": [{"updateCells": {"range": {"sheetId": ws.id}, "fields": "userEnteredFormat"}}]})
        ws.update(rows, value_input_option="USER_ENTERED")
        logger.info(f"[SheetSync] 거래로그 다시 쓰기: {worksheet} {len(rows) - 1}줄")
        return True
    except Exception as exc:
        logger.error(f"[SheetSync] 거래로그 다시 쓰기 실패: {exc}")
        return False


# ── [페이퍼_거래로그] — append only (PT-SPY) ──────────────────────────


_LOG_HEADERS = [
    "날짜", "종목", "섹터", "액션",
    "매수가", "매도가", "수익률%", "보유일",
    "전략", "별점", "CCS점수", "사유",
    "SPY동기간%", "SPY대비%p",
]


def _fmt_price(val) -> str:
    """float 가격을 소수점 2자리 문자열로 변환."""
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return str(val) if val else ""


def _norm_date(val: Any) -> str:
    """시트 날짜 셀을 YYYY-MM-DD 로. USER_ENTERED 로 날짜가 된 셀은 표시 형식이 달라질 수 있다. 모르는 형식은 원문."""
    text = str(val or "").strip()
    for fmt in ("%Y-%m-%d", "%Y. %m. %d", "%Y.%m.%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text


def _log_key(row: list[Any]) -> tuple[str, str, str]:
    """거래로그 행의 (날짜, 종목, BUY/SELL)."""
    return _norm_date(row[0]), str(row[1]).strip(), "BUY" if "BUY" in str(row[3]) else "SELL"


def _already_logged(existing: list[list[Any]], row: list[Any]) -> bool:
    """같은 (날짜, 종목, BUY/SELL) 행이 이미 있으면 True — 같은 일봉을 다시 처리해도 행이 겹치지 않게."""
    key = _log_key(row)
    return any(len(r) >= 4 and _log_key(r) == key for r in existing)


def sync_trade_log(trade: dict[str, Any], action: str = "SELL", worksheet: str = WORKSHEET_LOG) -> bool:
    """거래 기록 1건을 거래로그 탭(기본 [페이퍼_거래로그])에 append."""
    sheet = _open_sheet()
    if sheet is None:
        return False

    ws = _get_or_create_worksheet(sheet, worksheet)
    if ws is None:
        return False

    try:
        # 헤더 검사: 첫 행이 비어있거나 헤더가 안맞으면 삽입 (gspread 6 은 빈 시트에 [[]] 를 돌려준다)
        existing = ws.get_all_values()
        if not existing or not existing[0] or existing[0][0] != "날짜":
            ws.insert_row(_LOG_HEADERS, index=1, value_input_option="USER_ENTERED")
        elif len([c for c in existing[0] if c]) < len(_LOG_HEADERS):
            ws.update([_LOG_HEADERS], "A1", value_input_option="USER_ENTERED")  # 새 열(SPY) 헤더 보강

        if action == "BUY":
            row = [
                trade.get("entry_date", str(date.today())),  # 날짜
                trade.get("ticker", ""),                      # 종목
                trade.get("sector", ""),                      # 섹터
                "🟢 BUY",                                     # 액션
                _fmt_price(trade.get("entry_price", trade.get("price"))),  # 매수가
                "",                                           # 매도가 (없음)
                "",                                           # 수익률% (매도 시 채움)
                "",                                           # 보유일 (매도 시 채움)
                trade.get("strategy", ""),                    # 전략
                trade.get("star_rating", ""),                  # 별점
                trade.get("ccs_score", trade.get("ccs", "")), # CCS
                trade.get("reason", "신규매수"),               # 사유
                "", "",                                       # SPY동기간%, SPY대비%p (매도 시 채움)
            ]
        else:  # SELL
            ret = trade.get("return_pct", 0)
            try:
                ret_str = f"{float(ret):+.1%}"
            except (TypeError, ValueError):
                ret_str = ""
            spy = _spy_return(trade)
            try:
                spy_str = f"{spy:+.1%}" if spy is not None else ""
                diff_str = f"{(float(ret) - spy) * 100:+.1f}" if spy is not None else ""
            except (TypeError, ValueError):
                spy_str = diff_str = ""
            row = [
                trade.get("exit_date", str(date.today())),   # 날짜
                trade.get("ticker", ""),                      # 종목
                trade.get("sector", ""),                      # 섹터
                "🔴 SELL",                                    # 액션
                _fmt_price(trade.get("entry_price")),         # 매수가
                _fmt_price(trade.get("exit_price")),          # 매도가
                ret_str,                                      # 수익률%
                trade.get("holding_days", ""),               # 보유일
                trade.get("strategy", ""),                    # 전략
                trade.get("star_rating", ""),                  # 별점
                trade.get("ccs_score", ""),                   # CCS
                trade.get("exit_reason", trade.get("reason", "")),  # 사유
                spy_str,                                      # SPY동기간%
                diff_str,                                     # SPY대비%p
            ]

        if _already_logged(existing, row):
            logger.info(f"[SheetSync] 거래로그 이미 있음 — 건너뜀: {action} {trade.get('ticker', '')} {row[0]}")
            return True
        ws.append_row(row, value_input_option="USER_ENTERED")
        logger.info(f"[SheetSync] 거래로그 append: {action} {trade.get('ticker', '')}")
        return True
    except Exception as exc:
        logger.error(f"[SheetSync] 거래로그 append 실패: {exc}")
        return False


# ── [페이퍼_포지션현황] — 매일 덮어씀 ───────────────────────


_POS_HEADERS = [
    "종목", "섹터", "매수일", "매수가", "현재가", "수익률%",
    "고점", "고점대비%", "보유일", "전략", "별점", "CCS점수",
]


def sync_positions(
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    worksheet: str = WORKSHEET_POSITIONS,
) -> bool:
    """현재 보유 종목을 포지션 탭(기본 [페이퍼_포지션현황])에 덮어쓰기."""
    sheet = _open_sheet()
    if sheet is None:
        return False

    ws = _get_or_create_worksheet(sheet, worksheet)
    if ws is None:
        return False

    try:
        est = timezone(timedelta(hours=-5), name="EST")
        timestamp = datetime.now(est).strftime("%Y-%m-%d %H:%M:%S EST")

        rows = [_POS_HEADERS + [timestamp]]
        for pos in positions:
            ticker = pos["ticker"]
            entry_price = pos["entry_price"]
            current_price = prices.get(ticker, entry_price)
            highest = pos.get("highest_price", entry_price)
            ret = (current_price - entry_price) / entry_price if entry_price else 0
            drawdown = (current_price - highest) / highest if highest else 0

            entry_date = pos.get("entry_date", "")
            if entry_date:
                days = (date.today() - datetime.strptime(str(entry_date), "%Y-%m-%d").date()).days
            else:
                days = 0

            rows.append([
                ticker,
                pos.get("sector", "Unknown"),
                entry_date,
                f"{entry_price:.2f}",
                f"{current_price:.2f}",
                f"{ret:+.1%}",
                f"{highest:.2f}",
                f"{drawdown:+.1%}",
                days,
                pos.get("strategy", ""),
                pos.get("star_rating", ""),
                pos.get("ccs_score", ""),
            ])

        if not positions:
            rows.append(["보유 종목 없음"] + [""] * (len(_POS_HEADERS) - 1))

        ws.clear()
        ws.update(rows, value_input_option="USER_ENTERED")
        from screener.exporter import _resize_worksheet_to_data
        _resize_worksheet_to_data(ws, rows)

        logger.info(f"[SheetSync] 포지션현황 업데이트: {len(positions)}개 종목")
        return True
    except Exception as exc:
        logger.error(f"[SheetSync] 포지션현황 업데이트 실패: {exc}")
        return False


# ── [페이퍼_성과요약] — 매일 덮어씀 ─────────────────────────


def dollar_trades(trades: list[dict[str, Any]], positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """청산 거래의 return_pct 를 $5,000 계좌 기준(비용 포함 매도금액 ÷ 매수금액 − 1)으로 바꾼 사본. 0주 거래는 뺀다
    ($5,000 계좌 요약과 같게 — 1주도 못 산 자리는 거래로 세지 않는다)."""
    from paper_trading.pt1_account import dollar_lots

    lots = dollar_lots(trades, positions)
    return [{**t, "return_pct": lot.proceeds / lot.cost - 1} for t, lot in zip(trades, lots) if lot.shares > 0 and lot.cost]


def sync_summary(trades: list[dict[str, Any]], worksheet: str = WORKSHEET_SUMMARY,
                 positions: list[dict[str, Any]] | None = None) -> bool:
    """전체 거래 히스토리 기반 성과 집계를 성과요약 탭(기본 [페이퍼_성과요약])에 덮어쓰기.

    positions 를 주면 (PT-1·1S·1N) 수익률을 $5,000 계좌 기준으로 계산한다 — 거래로그 탭과 같은 숫자.
    """
    if positions is not None:
        trades = dollar_trades(trades, positions)
    sheet = _open_sheet()
    if sheet is None:
        return False

    ws = _get_or_create_worksheet(sheet, worksheet)
    if ws is None:
        return False

    try:
        est = timezone(timedelta(hours=-5), name="EST")
        timestamp = datetime.now(est).strftime("%Y-%m-%d %H:%M:%S EST")
        headers = ["구분", "총거래", "승률", "평균수익", "최대수익", "최대손실", "누적수익",
                   "SPY동기간평균", "SPY대비평균", "SPY이긴비율", timestamp]
        spy_by_id = {id(t): _spy_return(t) for t in trades}

        rows = [headers]

        def _calc_stats(label: str, subset: list[dict]) -> list:
            if not subset:
                return [label, 0, "—", "—", "—", "—", "—", "—", "—", "—"]
            returns = [t.get("return_pct", 0) for t in subset]
            wins = sum(1 for r in returns if r > 0)
            total = len(returns)
            avg_ret = sum(returns) / total
            max_ret = max(returns)
            min_ret = min(returns)
            cumulative = 1.0
            for r in returns:
                cumulative *= (1 + r)
            cumulative -= 1.0
            pairs = [(t.get("return_pct", 0), spy_by_id.get(id(t))) for t in subset]
            pairs = [(r, b) for r, b in pairs if b is not None]
            if pairs:
                spy_avg = sum(b for _, b in pairs) / len(pairs)
                diff_avg = sum(r - b for r, b in pairs) / len(pairs)
                beat = sum(1 for r, b in pairs if r > b) / len(pairs)
                # '+0.1%p' 는 시트가 수식으로 읽어 #ERROR! 가 된다 (숫자가 아니라서) → ' 로 글자 고정 (2026-10-16)
                spy_cols = [f"{spy_avg:+.1%}", f"'{diff_avg * 100:+.1f}%p", f"{beat:.0%}"]
            else:
                spy_cols = ["—", "—", "—"]
            return [
                label,
                total,
                f"{wins/total:.0%}" if total else "—",
                f"{avg_ret:+.1%}",
                f"{max_ret:+.1%}",
                f"{min_ret:+.1%}",
                f"{cumulative:+.1%}",
                *spy_cols,
            ]

        # 전체
        rows.append(_calc_stats("전체", trades))

        # 전략별
        strategy_groups: dict[str, list] = {}
        for t in trades:
            s = t.get("strategy", "기타")
            # 전략구분 문자열에서 핵심 키워드 추출
            if "바닥반등" in s:
                key = "바닥반등"
            elif "모멘텀" in s:
                key = "모멘텀"
            else:
                key = "기타"
            strategy_groups.setdefault(key, []).append(t)

        _fixed = ["바닥반등", "모멘텀", "기타"]
        for label in _fixed + sorted(k for k in strategy_groups if k not in _fixed):
            if label in strategy_groups:
                rows.append(_calc_stats(label, strategy_groups[label]))

        # 별점별
        star_groups: dict[str, list] = {}
        for t in trades:
            star = str(t.get("star_rating", ""))
            if "★★★★★" in star:
                key = "★5"
            elif "★★★★" in star:
                key = "★4"
            else:
                key = "기타별점"
            star_groups.setdefault(key, []).append(t)

        for label in ["★5", "★4", "기타별점"]:
            if label in star_groups:
                rows.append(_calc_stats(label, star_groups[label]))

        ws.clear()
        ws.update(rows, value_input_option="USER_ENTERED")
        from screener.exporter import _resize_worksheet_to_data
        _resize_worksheet_to_data(ws, rows)

        logger.info(f"[SheetSync] 성과요약 업데이트: {len(trades)}건 거래")
        return True
    except Exception as exc:
        logger.error(f"[SheetSync] 성과요약 업데이트 실패: {exc}")
        return False


# ── 백테스트 결과 동기화 ─────────────────────────────────────


def sync_backtest_result(
    summary: dict[str, Any],
    trades: list[dict[str, Any]],
) -> bool:
    """백테스트 결과를 [백테스트_결과] 탭에 기록 (없으면 생성, 있으면 덮어쓰기)."""
    sheet = _open_sheet()
    if sheet is None:
        return False

    ws = _get_or_create_worksheet(sheet, WORKSHEET_BACKTEST, rows=2000, cols=15)
    if ws is None:
        return False

    try:
        rows: list[list[str]] = []

        # ── 1. 요약 섹션 ─────────────────────────────────────
        rows.append(["=== 백테스트 요약 ==="])
        rows.append(["기간", f"{summary.get('시뮬레이션_시작', '')} ~ {summary.get('시뮬레이션_종료', '')}"])
        rows.append(["총거래수", str(summary.get("총거래수", 0))])
        rows.append(["승률", f"{summary.get('승률', 0):.1%}"])
        rows.append(["평균수익률", f"{summary.get('평균수익률', 0):+.2%}"])
        rows.append(["평균승리", f"{summary.get('평균승리', 0):+.2%}"])
        rows.append(["평균손실", f"{summary.get('평균손실', 0):+.2%}"])
        rows.append(["승패비율", f"{summary.get('승패비율', 0):.2f}"])
        rows.append(["평균보유일", f"{summary.get('평균보유일', 0):.1f}일"])
        rows.append(["Sharpe Ratio", f"{summary.get('Sharpe', 0):.2f}"])
        rows.append(["MDD", f"{summary.get('MDD', 0):.1%}"])
        if "SPY수익률" in summary:
            rows.append(["SPY수익률", f"{summary['SPY수익률']:+.2%}"])
            rows.append(["전략총수익률", f"{summary.get('전략총수익률', 0):+.2%}"])
            rows.append(["SPY초과수익", f"{summary.get('SPY초과수익', 0):+.2%}"])
        rows.append([])

        # ── 2. 전략별 성과 ───────────────────────────────────
        rows.append(["=== 전략별 성과 ==="])
        rows.append(["전략", "건수", "승률", "평균수익률"])
        strat_perf = summary.get("전략별", {})
        for strat, perf in strat_perf.items():
            rows.append([
                strat,
                str(perf.get("건수", 0)),
                f"{perf.get('승률', 0):.1%}",
                f"{perf.get('평균수익률', 0):+.2%}",
            ])
        rows.append([])

        # ── 3. 매도사유 분포 ─────────────────────────────────
        rows.append(["=== 매도사유 분포 ==="])
        rows.append(["사유", "건수", "비율"])
        exit_dist = summary.get("매도사유", {})
        total_exits = sum(exit_dist.values()) or 1
        for reason, cnt in sorted(exit_dist.items(), key=lambda x: -x[1]):
            rows.append([reason, str(cnt), f"{cnt/total_exits:.1%}"])
        rows.append([])

        # ── 4. 전체 거래 로그 ────────────────────────────────
        rows.append(["=== 전체 거래 로그 ==="])
        rows.append([
            "매수일", "매도일", "종목", "섹터", "전략", "별점",
            "매수가", "매도가", "수익률%", "보유일", "CCS점수", "매도사유",
        ])
        for t in sorted(trades, key=lambda x: x.get("entry_date", "")):
            ret_pct = t.get("return_pct", 0)
            rows.append([
                t.get("entry_date", ""),
                t.get("exit_date", ""),
                t.get("ticker", ""),
                t.get("sector", ""),
                t.get("strategy", "").replace("📉 ", "").replace("📈 ", "").replace("🔄 ", ""),
                t.get("star_rating", ""),
                _fmt_price(t.get("entry_price")),
                _fmt_price(t.get("exit_price")),
                f"{ret_pct*100:+.2f}",
                str(t.get("holding_days", 0)),
                str(round(t.get("ccs_score", 0), 4)),
                t.get("exit_reason", ""),
            ])

        ws.clear()
        ws.update(rows, value_input_option="USER_ENTERED")
        logger.info(f"[SheetSync] [{WORKSHEET_BACKTEST}] 업데이트 완료 ({len(trades)}건)")
        return True

    except Exception as exc:
        logger.error(f"[SheetSync] 백테스트 결과 업데이트 실패: {exc}")
        return False


# ── 통합 동기화 ──────────────────────────────────────────────


def sync_all(
    result: dict[str, Any],
    positions: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    prices: dict[str, float],
    tabs: dict[str, str] | None = None,
) -> None:
    """engine.run_daily_trading() 결과를 구글 시트 3탭에 동기화. tabs = {"log","positions","summary"} 탭 이름 (없으면 PT-1 기본)."""
    tabs = tabs or {"log": WORKSHEET_LOG, "positions": WORKSHEET_POSITIONS, "summary": WORKSHEET_SUMMARY}
    # 거래로그: 거래 기록 전체에서 $5,000 계좌 기준으로 다시 쓴다 (매일 같은 결과 — 중복·빈 칸 없음)
    rebuild_trade_log(trades, positions, worksheet=tabs["log"])

    # 포지션현황
    sync_positions(positions, prices, worksheet=tabs["positions"])

    # 성과요약 ($5,000 계좌 기준)
    sync_summary(trades, worksheet=tabs["summary"], positions=positions)
