"""페이퍼 트레이딩 구글 시트 3탭 동기화.

기존 exporter.py의 _open_sheet() 패턴 재활용.
- [페이퍼_거래로그] — 매수/매도 기록 append
- [페이퍼_포지션현황] — 현재 보유 종목 덮어쓰기
- [페이퍼_성과요약] — 누적 성과 집계 덮어쓰기
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


# ── [페이퍼_거래로그] — append only ──────────────────────────


_LOG_HEADERS = [
    "날짜", "종목", "섹터", "액션",
    "매수가", "매도가", "수익률%", "보유일",
    "전략", "별점", "CCS점수", "사유",
]


def _fmt_price(val) -> str:
    """float 가격을 소수점 2자리 문자열로 변환."""
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return str(val) if val else ""


def sync_trade_log(trade: dict[str, Any], action: str = "SELL") -> bool:
    """거래 기록 1건을 [페이퍼_거래로그]에 append."""
    sheet = _open_sheet()
    if sheet is None:
        return False

    ws = _get_or_create_worksheet(sheet, WORKSHEET_LOG)
    if ws is None:
        return False

    try:
        # 헤더 검사: 첫 행이 비어있거나 헤더가 안맞으면 삽입
        existing = ws.get_all_values()
        if not existing or existing[0][0] != "날짜":
            ws.insert_row(_LOG_HEADERS, index=1, value_input_option="USER_ENTERED")

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
            ]
        else:  # SELL
            ret = trade.get("return_pct", 0)
            try:
                ret_str = f"{float(ret):+.1%}"
            except (TypeError, ValueError):
                ret_str = ""
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
            ]

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
) -> bool:
    """현재 보유 종목을 [페이퍼_포지션현황]에 덮어쓰기."""
    sheet = _open_sheet()
    if sheet is None:
        return False

    ws = _get_or_create_worksheet(sheet, WORKSHEET_POSITIONS)
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


def sync_summary(trades: list[dict[str, Any]]) -> bool:
    """전체 거래 히스토리 기반 성과 집계를 [페이퍼_성과요약]에 덮어쓰기."""
    sheet = _open_sheet()
    if sheet is None:
        return False

    ws = _get_or_create_worksheet(sheet, WORKSHEET_SUMMARY)
    if ws is None:
        return False

    try:
        est = timezone(timedelta(hours=-5), name="EST")
        timestamp = datetime.now(est).strftime("%Y-%m-%d %H:%M:%S EST")
        headers = ["구분", "총거래", "승률", "평균수익", "최대수익", "최대손실", "누적수익", timestamp]

        rows = [headers]

        def _calc_stats(label: str, subset: list[dict]) -> list:
            if not subset:
                return [label, 0, "—", "—", "—", "—", "—"]
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
            return [
                label,
                total,
                f"{wins/total:.0%}" if total else "—",
                f"{avg_ret:+.1%}",
                f"{max_ret:+.1%}",
                f"{min_ret:+.1%}",
                f"{cumulative:+.1%}",
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

        for label in ["바닥반등", "모멘텀", "기타"]:
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
) -> None:
    """engine.run_daily_trading() 결과를 구글 시트 3탭에 동기화."""
    # 거래로그: 매도 기록
    for sell in result.get("sells", []):
        sync_trade_log(sell, action="SELL")

    # 거래로그: 매수 기록
    for buy in result.get("buys", []):
        sync_trade_log(buy, action="BUY")

    # 포지션현황
    sync_positions(positions, prices)

    # 성과요약
    sync_summary(trades)
