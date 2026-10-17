"""개인 웹사이트(별도 리포)가 읽는 dashboard.json 을 만든다.

매일 CI 가 PT·스크리너 실행 뒤 이 모듈을 돌려 결과를 dashboard-data 브랜치에 덮어쓴다.
웹사이트는 raw.githubusercontent.com 의 그 파일만 읽으므로, 이 리포와 웹사이트는 서로의 배포에 영향을 주지 않는다.
표시 전용 — 매매 판단·상태 파일은 건드리지 않는다.

사용:
    PYTHONPATH=.:src python -m web_export.dashboard --out "$RUNNER_TEMP/web/dashboard.json"
    # 로컬: 차트 폴더가 없으면 원격 스크린샷 브랜치(git fetch 해 둔 origin/screenshots-*) 목록을 대신 읽는다
    PYTHONPATH=.:src python -m web_export.dashboard --out /tmp/dashboard.json --charts-from-git
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd

SCHEMA_VERSION = 1
TIMEFRAMES = ("1H", "4H", "Daily", "Weekly", "Monthly")
EXIT_REASON_KEYS = ("목표가", "시간익절", "트레일링", "손절", "장기보유", "교체")
CANDIDATE_LIMIT = 30
ET = ZoneInfo("America/Toronto")

# (시장 키, 표시 이름, 랭킹 스냅샷 파일, 차트 폴더 config 이름, 스크린샷 브랜치)
MARKETS = (
    ("sp500", "S&P 500", "sp500_ranked.parquet", "CHARTS_OUTPUT_DIR", "screenshots-sp500"),
    ("nasdaq", "나스닥·NYSE", "nasdaq_ranked.parquet", "CHARTS_SIGNALS2_OUTPUT_DIR", "screenshots-nasdaq"),
)

_CHART_FILE_RE = re.compile(r"_(?P<tf>1H|4H|Daily|Weekly|Monthly)_(?P<stamp>\d{8}_\d{6})\.(?:png|webp)$")
# 전체차트 (charts/capture_remaining.py) — 차트분석에 없는 분석 시트 종목 캡처
ALL_CHARTS = ("charts/screenshots_all", "screenshots-all")


# ── 값 정리 ──────────────────────────────────────────────────


def _num(v: Any, digits: int | None = None) -> float | None:
    """NaN·None·숫자 아님 → None, 아니면 float (digits 가 있으면 반올림)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits) if digits is not None else f


def _text(v: Any) -> str | None:
    """빈 값·NaN·NaT 은 None, 아니면 앞뒤 공백을 뗀 문자열."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s or None


def _label(v: Any) -> str | None:
    """'1. 매수 후보' → '매수 후보', '📈 모멘텀' → '모멘텀', '—' → None."""
    s = _text(v)
    if not s or s in {"—", "-"}:
        return None
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"^[^\w]+", "", s)  # 앞의 이모지·기호
    return s or None


def _patterns(v: Any) -> list[str]:
    """'이중천장, 골든크로스' → ['이중천장', '골든크로스']."""
    s = _text(v)
    return [p.strip() for p in s.split(",") if p.strip()] if s else []


def parse_score(display: Any) -> float | None:
    """'★★★★★ (11.5)' → 11.5."""
    m = re.search(r"\(([-\d.]+)\)", str(display or ""))
    return _num(m.group(1)) if m else None


def parse_health(text: Any) -> dict[str, Any] | None:
    """'100 (A+) · 매출+48% 마진+25% 이익+86%' → {score, grade, detail}."""
    s = _text(text)
    if not s:
        return None
    m = re.match(r"\s*(\d+)\s*\(([^)]+)\)\s*(?:·\s*(.*))?$", s)
    if not m:
        return {"score": None, "grade": None, "detail": s}
    return {"score": int(m.group(1)), "grade": m.group(2), "detail": _text(m.group(3))}


# ── 스크리너 스냅샷 ──────────────────────────────────────────


def load_ranked(data_dir: Path) -> pd.DataFrame:
    """S&P 500·나스닥 랭킹 스냅샷을 합친다 (market 컬럼 추가, 티커가 겹치면 앞의 것)."""
    frames = []
    for key, _label_name, fname, _cfg_name, _branch in MARKETS:
        path = data_dir / fname
        if path.exists():
            df = pd.read_parquet(path)
            df["market"] = key
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("티커")


# (웹 키, 스냅샷 컬럼, 반올림 자리) — yfinance 값 그대로: 성장률·마진·ROE·비중은 0.165 = 16.5%,
# 부채비율은 % (32 = 0.32배), 유동비율은 배, 현금흐름·시총은 달러
FUNDAMENTAL_FIELDS = (
    ("revenue_growth", "fund_revenue_growth", 4),
    ("earnings_growth", "fund_earnings_growth", 4),
    ("gross_margin", "fund_gross_margin", 4),
    ("profit_margin", "fund_profit_margin", 4),
    ("roe", "fund_roe", 4),
    ("debt_to_equity", "fund_debt_to_equity", 2),
    ("current_ratio", "fund_current_ratio", 2),
    ("free_cashflow", "fund_free_cashflow", 0),
    ("operating_cashflow", "fund_operating_cashflow", 0),
    ("market_cap", "fund_market_cap", 0),
    ("dividend_yield", "dividend_yield", 4),
    ("short_pct_float", "fund_short_pct_float", 4),
    ("institutional_pct", "fund_institutional_holders_pct", 4),
)


def fundamentals(row: pd.Series) -> dict[str, float | None]:
    """스냅샷의 재무 컬럼(fund_*) → 웹 키. 없는 값은 None."""
    return {key: _num(row.get(col), digits) for key, col, digits in FUNDAMENTAL_FIELDS}


def _sec_flags(v: Any) -> list[dict[str, Any]]:
    """스냅샷 sec_flags 열(JSON 글자) → 목록. 비었거나 깨졌으면 []."""
    s = _text(v)
    if not s or s == "nan":
        return []
    try:
        out = json.loads(s)
    except ValueError:
        return []
    return out if isinstance(out, list) else []


def _short(row: pd.Series) -> dict[str, Any] | None:
    pct, dtc = _num(row.get("short_pct"), 4), _num(row.get("short_dtc"), 2)
    if pct is None and dtc is None:
        return None
    high = str(row.get("short_high")) == "True"
    return {"pct": pct, "dtc": dtc, "change": _num(row.get("short_change"), 4), "settle": _text(row.get("short_settle")),
            "high": high}


def squeeze_block(data_dir: Path | None = None) -> dict[str, Any]:
    """숏스퀴즈 감시 목록 (🔥 진행 가능 · 대기). 파일이 없으면 빈 목록."""
    from screener.short_squeeze import load_watch

    w = load_watch(str((data_dir or Path("data/paper_trading")) / "squeeze_watch.parquet"))
    rows = [{"ticker": r.ticker, "name": _text(r.name), "status": r.status, "price": _num(r.price, 2),
             "short_pct": _num(r.short_pct, 4), "short_basis": _text(r.short_basis), "dtc": _num(r.dtc, 2),
             "si_change": _num(r.si_change, 4), "ret_5d": _num(r.ret_5d, 4), "vol_ratio": _num(r.vol_ratio, 2)}
            for r in w.itertuples()]
    return {"settle": _text(w["settle"].iloc[0]) if len(w) else None, "items": rows}


def ticker_info(row: pd.Series) -> dict[str, Any]:
    """랭킹 스냅샷 한 줄 → 웹에 보여 줄 요약."""
    earnings = _text(row.get("next_earnings_date"))
    days = _num(row.get("days_to_next_earnings"))
    return {
        "company": _text(row.get("회사")),
        "market": _text(row.get("market")),
        "sector": _text(row.get("섹터")),
        "price": _num(row.get("현재가격"), 2),
        "strategy": _label(row.get("전략구분")),
        "score": parse_score(row.get("매수적합도_표시")),
        "verdict": _label(row.get("판단")),
        "action": _label(row.get("추천")),
        "health": parse_health(row.get("[헬스체크]")),
        "market_cap": _text(row.get("[시총]")),
        "rsi": _num(row.get("RSI"), 1),
        "ret_5d": _num(row.get("5일수익률"), 4),
        "ret_20d": _num(row.get("20일수익률"), 4),
        "earnings_date": earnings[:10] if earnings else None,
        "days_to_earnings": int(days) if days is not None else None,
        "patterns": {
            "daily": _patterns(row.get("일봉패턴")),
            "weekly": _patterns(row.get("주봉패턴")),
            "monthly": _patterns(row.get("월봉패턴")),
        },
        "vol_ratio": _num(row.get("거래량돌파배수"), 2),  # 오늘 거래량 ÷ 20일 평균
        "ma_gap": {  # 단기 MA 가 장기 MA 보다 몇 % 위(+)·아래(−)인지 — 골든·데드크로스까지 거리
            "daily": _num(row.get("ema_gap_20_50"), 4),
            "weekly": _num(row.get("주봉_MA갭"), 4),
            "monthly": _num(row.get("월봉_MA갭"), 4),
        },
        "golden": None,  # PT 골든크로스 목록에 있으면 build_dashboard 가 채운다
        "position_52w": _num(row.get("52주포지션"), 4),  # 52주 저점 0 ~ 고점 1
        "atr_pct": _num(row.get("ATR%"), 4),
        "rel_market": _num(row.get("시장상대강도"), 4),
        "rel_sector": _num(row.get("섹터상대강도"), 4),
        "fundamentals": fundamentals(row),
        "insider": None,  # build_dashboard 가 후보·보유·차트 종목에만 채운다
        "sec_flags": _sec_flags(row.get("sec_flags")),  # SEC 공시 경고 (sec_filings.Flag 목록, 없으면 [])
        "short": _short(row),  # 공매도 (FINRA 잔고·되사기 + 유통주식 대비 비중), 없으면 None
        "squeeze": None,  # build_dashboard 가 숏스퀴즈 감시 목록으로 채운다 ('fire' · 'fuel')
    }


def insider_info(tickers: list[str], *, cache_only: bool = True) -> dict[str, dict[str, Any]]:
    """openinsider 90일 내부자 거래 요약.

    cache_only: 같은 실행에서 스크리너·PT 가 받아 둔 24시간 캐시가 있는 종목만 (네트워크 없음 — openinsider 가
    멈춰도 종목당 30초씩 기다리지 않는다). 캐시가 없는 종목은 빠진다 (웹은 '정보 없음').
    signal: 'cluster_buy'(여러 명이 매수) · 'buy' · 'sell' · None(거래 없음). 조회가 통째로 실패하면 빈 dict.
    """
    from screener.insider import fetch_insider_snapshots, has_fresh_cache

    if cache_only:
        tickers = [t for t in tickers if has_fresh_cache(t)]
    if not tickers:
        return {}
    try:
        df = fetch_insider_snapshots(tickers)
    except Exception as exc:  # noqa: BLE001 — 표시용이라 실패해도 나머지는 내보낸다
        print(f"[web_export] 내부자 거래 조회 실패 (건너뜀): {exc}")
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        buys = int(_num(r.get("insider_buy_count")) or 0)
        sells = int(_num(r.get("insider_sell_count")) or 0)
        text = _text(r.get("내부자(90일)")) or ""
        signal = "cluster_buy" if "클러스터" in text else "buy" if buys else "sell" if sells else None
        out[str(r["티커"])] = {
            "buys": buys,
            "sells": sells,
            "net_value": _num(r.get("insider_net_value"), 0),
            "last_date": _text(r.get("insider_last_trade_date")),
            "signal": signal,
        }
    return out


def build_candidates(ranked: pd.DataFrame, infos: dict[str, dict[str, Any]], held: dict[str, int],
                     limit: int = CANDIDATE_LIMIT) -> list[dict[str, Any]]:
    """'1. 매수 후보' 판정 종목을 매수 적합도 순으로 (두 시장 합쳐서)."""
    if ranked.empty or "판단" not in ranked.columns:
        return []
    tickers = ranked.loc[ranked["판단"].astype(str).str.startswith("1"), "티커"].astype(str)
    rows = [{"ticker": t, **infos[t], "held": t in held} for t in tickers]
    rows.sort(key=lambda c: -(c["score"] or 0.0))
    return rows[:limit]


_TF_KEYS = {"일봉": "daily", "주봉": "weekly", "월봉": "monthly"}


def golden_cross_lists(ranked: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """시장별 PT 골든크로스 목록 (PT-1 이메일 '골든크로스임박' 섹션과 같은 규칙·순서).

    timeframes: [{tf, gap}] — gap < 0 이면 교차 임박(단기 MA 가 아래, 5% 이내), ≥ 0 이면 방금 교차(0~5%).
    """
    from paper_trading.golden_cross import extract_golden_cross

    out: dict[str, list[dict[str, Any]]] = {}
    for key, *_rest in MARKETS:
        part = ranked[ranked["market"] == key] if "market" in ranked.columns else ranked.iloc[0:0]
        out[key] = [{
            "ticker": g["ticker"],
            "timeframes": [{"tf": _TF_KEYS.get(label, label), "gap": _num(gap, 4)} for label, gap in g["tf_gaps"]],
        } for g in extract_golden_cross(part)]
    return out


# ── 계좌 ─────────────────────────────────────────────────────


def _exit_reason_key(reason: str) -> str:
    return next((k for k in EXIT_REASON_KEYS if reason.startswith(k)), "기타")


def trade_log(trades: list[dict[str, Any]], spy_close: pd.Series | None = None,
              positions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """웹 '거래로그' — 청산 거래 전부 (최근 청산 먼저), $5,000 계좌 기준 (구글 시트 거래로그 탭과 같은 숫자).

    return = 매도금액 ÷ 매수금액 − 1 (편도 비용 0.1% 포함), shares·cost·proceeds·pnl = $5,000 계좌의 주식 수·금액.
    """
    events = sheet_log(trades, positions or [], spy_close)
    by_exit = {(e["ticker"], e["entry_date"], e["exit_date"]): e for e in events if e["side"] == "sell"}
    buys = {(e["ticker"], e["entry_date"]): e for e in events if e["side"] == "buy"}
    out = []
    for t in sorted(trades, key=lambda x: (str(x.get("exit_date") or ""), str(x.get("entry_date") or "")), reverse=True):
        e = by_exit.get((t.get("ticker"), str(t.get("entry_date")), str(t.get("exit_date"))), {})
        b = buys.get((t.get("ticker"), str(t.get("entry_date"))), {})
        out.append({
            "ticker": t.get("ticker"),
            "sector": _text(t.get("sector")),
            "strategy": _label(t.get("strategy")),
            "star": _text(t.get("star_rating")),
            "ccs": _num(t.get("ccs_score"), 4),
            "entry_date": t.get("entry_date"),
            "exit_date": t.get("exit_date"),
            "entry_price": _num(t.get("entry_price"), 4),
            "exit_price": _num(t.get("exit_price"), 4),
            "return": _num(e.get("ret"), 4),
            "holding_days": t.get("holding_days"),
            "exit_reason": t.get("exit_reason"),
            "spy_return": _num(e.get("spy"), 4),
            "vs_spy": _num(e.get("vs_spy"), 4),
            "shares": e.get("shares"),
            "cost": _num(b.get("amount"), 2),
            "proceeds": _num(e.get("amount"), 2),
            "pnl": _num(e.get("pnl"), 2),
            "cash_after": _num(e.get("cash_after"), 2),
        })
    return out


def sheet_log(trades: list[dict[str, Any]], positions: list[dict[str, Any]],
              spy_close: pd.Series | None = None) -> list[dict[str, Any]]:
    """시트 거래로그 탭과 같은 줄 (매수·매도, 계좌 배분 순서) — sheet_sync.dollar_log_events 그대로."""
    from paper_trading.benchmarks import spy_window_return
    from paper_trading.sheet_sync import dollar_log_events

    def spy_fn(t: dict[str, Any]) -> float | None:
        if spy_close is None or spy_close.empty or not t.get("entry_date") or not t.get("exit_date"):
            return None
        return spy_window_return(spy_close, str(t["entry_date"]), str(t["exit_date"]))

    try:
        return dollar_log_events(list(trades), list(positions), spy_fn=spy_fn)
    except Exception as exc:  # noqa: BLE001 — 표시용
        print(f"[web_export] 거래로그 계산 실패 (건너뜀): {exc}")
        return []


def _sheet_log_json(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = ("date", "side", "ticker", "sector", "shares", "price", "amount", "pnl", "ret", "days", "cash_after",
            "strategy", "star", "ccs", "reason", "spy", "vs_spy")
    out = []
    for e in events:
        row = {k: e.get(k) for k in keep}
        for k in ("price", "amount", "pnl", "cash_after"):
            row[k] = _num(row[k], 2)
        for k in ("ret", "spy", "vs_spy", "ccs"):
            row[k] = _num(row[k], 4)
        row["strategy"] = _label(row["strategy"])
        out.append(row)
    return out


def trade_summary(trades: list[dict[str, Any]], recent: int = 10) -> dict[str, Any]:
    """청산 기록 → 건수·승률·평균·청산 이유별 성적·거래별 기록(청산일 순)·최근 청산."""
    trades = sorted(trades, key=lambda t: (str(t.get("exit_date") or ""), str(t.get("entry_date") or "")))
    rets = [float(t.get("return_pct") or 0.0) for t in trades]
    groups: dict[str, list[float]] = {}
    for t, r in zip(trades, rets):
        groups.setdefault(_exit_reason_key(str(t.get("exit_reason") or "")), []).append(r)
    wins = sum(r > 0 for r in rets)
    return {
        "count": len(rets),
        "wins": wins,
        "win_rate": wins / len(rets) if rets else None,
        "avg_return": sum(rets) / len(rets) if rets else None,
        "first_entry": min((str(t["entry_date"]) for t in trades if t.get("entry_date")), default=None),
        "by_reason": [{"reason": k, "count": len(groups[k]), "avg_return": sum(groups[k]) / len(groups[k])}
                      for k in EXIT_REASON_KEYS + ("기타",) if k in groups],
        "history": [{"ticker": t.get("ticker"), "exit_date": t.get("exit_date"), "return": round(r, 4),
                     "exit_reason": t.get("exit_reason")} for t, r in zip(trades, rets)],
        "recent": [{
            "ticker": t.get("ticker"),
            "entry_date": t.get("entry_date"),
            "exit_date": t.get("exit_date"),
            "return": _num(t.get("return_pct"), 4),
            "exit_reason": t.get("exit_reason"),
            "strategy": _label(t.get("strategy")),
            "holding_days": t.get("holding_days"),
        } for t in trades[-recent:][::-1]],
    }


def _dollar_accounts(books: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]) -> dict[str, Any]:
    """$5,000 계좌 재계산 (일봉을 내려받는다). 실패하면 빈 dict — 웹은 그 부분만 숨긴다."""
    try:
        from paper_trading.pt1_account import build_accounts

        return build_accounts(books)
    except Exception as exc:  # noqa: BLE001 — 표시용이라 나머지는 계속
        print(f"[web_export] $5,000 계좌 계산 실패 (건너뜀): {exc}")
        return {}


def _dollar_block(acct: Any) -> dict[str, Any] | None:
    """DollarAccount → 요약 숫자 + 날짜별 평가액 곡선 (같은 날 SPY 를 샀을 때와 함께)."""
    if acct is None or not acct.lots:
        return None
    from paper_trading.pt1_account import summary

    s = summary(acct)
    spy = acct.spy_curve
    curve = [{"date": ts.strftime("%Y-%m-%d"), "equity": _num(v, 2),
              "spy": _num(spy.get(ts), 2) if not spy.empty else None}
             for ts, v in acct.curve.items()]
    return {
        "initial": _num(acct.initial, 2),
        "start": s["start"],
        "equity": _num(s["equity"], 2),
        "return": _num(s["ret"], 4),
        "spy_equity": _num(s["spy_equity"], 2),
        "spy_return": _num(s["spy_ret"], 4),
        "cash": _num(s["cash"], 2),
        "realized": _num(s["realized"], 2),
        "unrealized": _num(s["unrealized"], 2),
        "mdd": _num(s["mdd"], 4),
        "spy_mdd": _num(s["spy_mdd"], 4),
        "curve": curve,
    }


def build_account(key: str, cfg: dict[str, Any], positions: list[dict[str, Any]], trades: list[dict[str, Any]],
                  dollar: Any, infos: dict[str, dict[str, Any]], spy_close: pd.Series | None = None) -> dict[str, Any]:
    """계좌 1개 — 보유 종목(현재가·매도선·SPY 대비) + 청산 성적 + $5,000 계좌."""
    from paper_trading.engine import exit_levels
    from paper_trading.portfolio import get_holding_days
    from paper_trading.pt1_account import spy_since

    lots = {(lot.ticker, lot.entry_date): lot for lot in dollar.holdings} if dollar is not None else {}
    rows = []
    for p in positions:
        ticker = str(p["ticker"])
        info = infos.get(ticker, {})
        entry = float(p["entry_price"])
        lot = lots.get((ticker, str(p.get("entry_date"))))
        price = _num(dollar.price(lot)) if lot is not None else info.get("price")
        # 엔진은 매도 판단 전에 고점을 현재가로 올린다 — 메일 매도 기준 셀과 같은 순서로 계산
        highest = max(float(p.get("highest_price") or entry), price or 0.0)
        lv = exit_levels({**p, "highest_price": highest})
        nxt = lv.get("next")
        rows.append({
            "ticker": ticker,
            "company": info.get("company"),
            "sector": _text(p.get("sector")) or info.get("sector"),
            "strategy": _label(p.get("strategy")),
            "star_rating": _text(p.get("star_rating")),
            "entry_date": p.get("entry_date"),
            "entry_price": _num(entry, 4),
            "highest_price": _num(highest, 4),
            "price": _num(price, 4),
            "return": _num(price / entry - 1, 4) if price else None,
            "days_held": get_holding_days(p),
            "shares": _num(lot.shares) if lot is not None else None,
            "value": _num(lot.shares * price, 2) if lot is not None and price else None,
            "spy_return": _num(spy_since(dollar, str(p["entry_date"])), 4) if dollar is not None else None,
            "defer_count": int(p.get("defer_count") or 0),
            "exit": {
                "up": _num(lv.get("up"), 4),
                "up_tag": lv.get("up_tag") or None,
                "down": _num(lv.get("down"), 4),
                "down_tag": lv.get("down_tag") or None,
                "next": {"day": int(nxt["day"]), "side": nxt["side"], "price": _num(nxt["price"], 4),
                         "tag": nxt["tag"]} if nxt else None,
            },
            "health": info.get("health"),
            "market_cap": info.get("market_cap"),
        })
    return {
        "key": key,
        "label": cfg["label"],
        "positions": rows,
        "trades": trade_summary(trades),
        "log": trade_log(trades, spy_close, positions),
        "sheet_log": _sheet_log_json(sheet_log(trades, positions, spy_close)),
        "dollar": _dollar_block(dollar),
    }


# ── 차트 ─────────────────────────────────────────────────────


def list_chart_files(chart_dir: str, branch: str, *, from_git: bool = False) -> dict[str, dict[str, str]]:
    """{티커: {타임프레임: 저장소 안 경로}} — 같은 타임프레임 파일이 여럿이면 가장 최근 것."""
    prefix = chart_dir.rstrip("/") + "/"
    if from_git:
        out = subprocess.run(["git", "ls-tree", "-r", "--name-only", f"origin/{branch}"],
                             capture_output=True, text=True, check=True).stdout.splitlines()
        paths = [p for p in out if p.startswith(prefix)]
    else:
        root = Path(chart_dir)
        paths = [p.as_posix() for p in root.glob("*/*") if p.suffix in (".png", ".webp")] if root.exists() else []
    found: dict[str, dict[str, str]] = {}
    for path in sorted(paths):  # 이름에 시각이 들어 있어 정렬하면 최근 파일이 마지막
        parts = path.split("/")
        m = _CHART_FILE_RE.search(parts[-1])
        if m and len(parts) >= 2:
            found.setdefault(parts[-2], {})[m["tf"]] = path
    return found


def build_all_charts(ranked: pd.DataFrame, held: dict[str, int], repo: str, *,
                     from_git: bool = False) -> list[dict[str, Any]]:
    """웹 '전체차트' — 두 분석 시트의 모든 종목 (매수적합도 순). 이미지는 차트분석(PNG) 먼저, 없으면 전체차트(WebP).

    종목 정보는 stocks 맵에서 찾는다 (크기를 줄이려고 여기엔 안 넣는다). 아직 캡처가 없으면 images 가 빈 dict.
    """
    from screener import config as cfg

    sources = [(getattr(cfg, cfg_name), branch) for _k, _l, _f, cfg_name, branch in MARKETS] + [ALL_CHARTS]
    urls: dict[str, dict[str, str]] = {}
    stamps: list[str] = []
    for chart_dir, branch in sources:
        try:
            files = list_chart_files(chart_dir, branch, from_git=from_git)
        except subprocess.CalledProcessError:  # 브랜치가 아직 없음 (첫 실행 전)
            files = {}
        at = _captured_at(files)
        if at:
            stamps.append(at)
        for ticker, tfs in files.items():
            if ticker not in urls:
                urls[ticker] = {tf: f"https://raw.githubusercontent.com/{repo}/{branch}/{quote(tfs[tf])}"
                                for tf in TIMEFRAMES if tf in tfs}
    if ranked.empty or "market" not in ranked.columns:
        return []
    score = pd.to_numeric(ranked.get("매수적합도", pd.Series(0.0, index=ranked.index)), errors="coerce").fillna(0.0)
    r = ranked.assign(_s=score).sort_values("_s", ascending=False)
    out = []
    for key, label, *_rest in MARKETS:
        tickers = [str(t) for t in r.loc[r["market"] == key, "티커"] if str(t) not in ("SPY", "QQQ", "IWM")]
        out.append({"key": key, "label": label, "captured_at": max(stamps) if stamps else None,
                    "items": [{"ticker": t, "held": t in held, "images": urls.get(t, {})} for t in dict.fromkeys(tickers)]})
    return out


def _captured_at(files: dict[str, dict[str, str]]) -> str | None:
    """가장 최근 캡처 시각 (파일 이름의 YYYYMMDD_HHMMSS, CI 러너 시계 = UTC)."""
    stamps = [m["stamp"] for tfs in files.values() for path in tfs.values()
              if (m := _CHART_FILE_RE.search(path))]
    if not stamps:
        return None
    dt = datetime.strptime(max(stamps), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def build_charts(infos: dict[str, dict[str, Any]], held: dict[str, int], repo: str, *,
                 from_git: bool = False) -> list[dict[str, Any]]:
    """시장별 차트 목록 — 구글 시트 '차트분석' 탭과 같은 종목, 같은 순서 (점수 없는 고정 종목 → 보유 → 점수순)."""
    from screener import config as cfg

    markets = []
    for key, label, _fname, cfg_name, branch in MARKETS:
        files = list_chart_files(getattr(cfg, cfg_name), branch, from_git=from_git)
        items = []
        for ticker, tfs in files.items():
            info = infos.get(ticker)
            items.append({
                "ticker": ticker,
                "held": ticker in held,
                "info": info,
                "images": {tf: f"https://raw.githubusercontent.com/{repo}/{branch}/{quote(tfs[tf])}"
                           for tf in TIMEFRAMES if tf in tfs},
            })

        def _order(it: dict[str, Any]) -> tuple[int, int, float, str]:
            group = 0 if it["info"] is None else 1 if it["held"] else 2
            score = (it["info"] or {}).get("score") or 0.0
            return group, held.get(it["ticker"], 0), -score, it["ticker"]

        items.sort(key=_order)
        markets.append({"key": key, "label": label, "captured_at": _captured_at(files), "items": items})
    return markets


# ── 전체 ─────────────────────────────────────────────────────


def benchmark_block(period: str = "6y", close: pd.Series | None = None) -> dict[str, Any] | None:
    """SPY 일봉 종가 (배당 조정) — 웹 내계좌(판 거래·보유 중)가 '같은 기간 SPY 수익률'을 계산할 때 쓴다. 실패하면 None.
    close 를 주면 다시 받지 않는다 (build_dashboard 가 계좌 거래로그용으로 받은 것)."""
    from paper_trading.benchmarks import load_spy_close

    close = load_spy_close(period) if close is None else close
    if close.empty:
        print("[web_export] SPY 종가 없음 (benchmark 건너뜀)")
        return None
    if getattr(close.index, "tz", None) is not None:
        close = close.tz_localize(None)
    return {"ticker": "SPY", "closes": [[ts.strftime("%Y-%m-%d"), _num(v, 4)] for ts, v in close.items()
                                        if _num(v, 4) is not None]}


def build_dashboard(data_dir: Path | None = None, *, charts_from_git: bool = False,
                    with_prices: bool = True, with_insider: bool = True) -> dict[str, Any]:
    """dashboard.json 내용 전체."""
    from paper_trading.portfolio import load_positions, load_trades
    from paper_trading.runner import PT1_ACCOUNTS
    from screener import config as cfg

    data_dir = data_dir or Path(cfg.PAPER_TRADING_DATA_DIR)
    ranked = load_ranked(data_dir)
    infos = {str(r["티커"]): ticker_info(r) for _, r in ranked.iterrows()}
    bar_date = None
    if not ranked.empty and "_bar_date" in ranked.columns:
        bar_date = _text(ranked["_bar_date"].dropna().astype(str).max())

    books: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for key, acfg in PT1_ACCOUNTS.items():
        acct_dir = data_dir / acfg["subdir"] if acfg["subdir"] else data_dir
        books[key] = (load_trades(acct_dir), load_positions(acct_dir))
    dollars = _dollar_accounts(books) if with_prices else {}

    spy_close = pd.Series(dtype=float)
    if with_prices:
        from paper_trading.benchmarks import load_spy_close
        spy_close = load_spy_close("6y")
    accounts = [build_account(key, PT1_ACCOUNTS[key], positions, trades, dollars.get(key), infos, spy_close)
                for key, (trades, positions) in books.items()]
    held: dict[str, int] = {}
    for acct in accounts:
        for p in acct["positions"]:
            held.setdefault(p["ticker"], len(held) + 1)
    # 보유 종목 현재가·평가액은 $5,000 계좌가 받은 마지막 종가 기준 (스크리너 일봉보다 하루 늦을 수 있다)
    curve_dates = [a["dollar"]["curve"][-1]["date"] for a in accounts if a["dollar"] and a["dollar"]["curve"]]
    golden = golden_cross_lists(ranked)
    for items in golden.values():
        for rank, g in enumerate(items):
            if g["ticker"] in infos:
                infos[g["ticker"]]["golden"] = {"rank": rank, "timeframes": g["timeframes"]}
    # 숏스퀴즈 상태는 후보를 만들기 전에 채운다 (후보는 infos 를 복사한다)
    squeeze = squeeze_block(data_dir)
    status = {q["ticker"]: q["status"] for q in squeeze["items"]}
    for t, info in infos.items():
        info["squeeze"] = status.get(t.replace(".", "-"))
    candidates = build_candidates(ranked, infos, held)
    charts = build_charts(infos, held, cfg.GITHUB_REPO_NAME, from_git=charts_from_git)

    # 웹 종목 상세 페이지용 — 후보·보유·차트·골든크로스에 나오는 종목만 (차트 항목의 info 는 같은 dict 라 함께 채워진다)
    all_charts = build_all_charts(ranked, held, cfg.GITHUB_REPO_NAME, from_git=charts_from_git)
    shown = ([c["ticker"] for c in candidates] + list(held) + [it["ticker"] for m in charts for it in m["items"]]
             + [g["ticker"] for items in golden.values() for g in items]
             + [it["ticker"] for m in all_charts for it in m["items"]])
    stocks = {t: infos[t] for t in dict.fromkeys(shown) if t in infos}
    if with_insider and stocks:
        for t, rec in insider_info(list(stocks)).items():
            if t in stocks:
                stocks[t]["insider"] = rec
        for c in candidates:
            c["insider"] = infos[c["ticker"]]["insider"]

    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "bar_date": bar_date,
        "price_date": max(curve_dates) if curve_dates else bar_date,
        "repo": cfg.GITHUB_REPO_NAME,
        "max_positions": int(cfg.PAPER_TRADING_MAX_POSITIONS),
        "accounts": accounts,
        "candidates": candidates,
        "charts": charts,
        "all_charts": all_charts,
        "golden_cross": golden,
        "stocks": stocks,
        "benchmark": benchmark_block(close=spy_close) if with_prices else None,
        "squeeze": squeeze,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI — dashboard.json 을 --out 경로에 쓴다."""
    parser = argparse.ArgumentParser(description="개인 웹사이트용 dashboard.json 만들기")
    parser.add_argument("--out", required=True, type=Path, help="쓸 파일 경로")
    parser.add_argument("--charts-from-git", action="store_true",
                        help="로컬 차트 폴더 대신 origin/screenshots-* 브랜치 파일 목록을 읽는다 (로컬 테스트용)")
    parser.add_argument("--no-prices", action="store_true",
                        help="$5,000 계좌 재계산(일봉 다운로드)을 건너뛴다")
    parser.add_argument("--no-insider", action="store_true",
                        help="내부자 거래 요약을 넣지 않는다 (기본은 이미 받아 둔 캐시만 읽는다)")
    args = parser.parse_args(argv)

    data = build_dashboard(charts_from_git=args.charts_from_git, with_prices=not args.no_prices,
                           with_insider=not args.no_insider)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                        encoding="utf-8")
    n_charts = sum(len(m["items"]) for m in data["charts"])
    n_pos = sum(len(a["positions"]) for a in data["accounts"])
    n_insider = sum(1 for s in data["stocks"].values() if s["insider"] is not None)
    print(f"[web_export] {args.out} — 일봉 {data['bar_date']}, 보유 {n_pos}, 후보 {len(data['candidates'])}, "
          f"차트 {n_charts}종목, 종목 정보 {len(data['stocks'])} (내부자 {n_insider}) "
          f"({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
