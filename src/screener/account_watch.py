#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""내계좌 장중 30분 체크 (account-watch).

장중(09:30~16:15 ET, 평일)에 30분마다 '내계좌'·'관심종목' 탭을 **읽기만** 하고 Yahoo 일봉(오늘 봉 = 장중 값)으로:
  1) 매번 ntfy 로 요약 푸시 — 종목별 현재가 · 오늘 ±% · 매수가 대비 ±%, 오늘 손익·평가 손익
  2) 새 알림이 있을 때만 ntfy(높은 우선순위) + Gmail 메일
     🛑 손절가 아래 (현재가 또는 장중 저가) · ⚠️ 손절가 3% 이내 · 🎯 목표가 (현재가 또는 장중 고가)
     🚀/📉 전날 종가 대비 ±5% · 👀 관심종목 목표가 (조건 이하/이상)
같은 알림은 하루 한 번만 보낸다 (state.json — CI 에서는 고아 브랜치 account-watch 에 보관).
실행 신호는 Mac mini LaunchAgent 가 30분마다 workflow_dispatch 로 보낸다 (GitHub 예약 실행은 몇 시간씩 늦어서 안 씀).
휴장일은 SPY 오늘 봉이 없는 것으로 알아채고 조용히 끝낸다. 시트에는 아무것도 쓰지 않는다 (밤 리포트가 쓴다).

사용:
    python src/screener/account_watch.py --state state.json           # 장 시간 밖이면 바로 끝
    python src/screener/account_watch.py --dry-run --force --tickers NOW,PEP   # 시트 없이 시세·요약만 출력
"""

from __future__ import annotations

import argparse
import os
import re
import smtplib
import sys
from dataclasses import dataclass
from datetime import date, datetime, time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf

from paper_trading.json_store import atomic_json_write, read_json
from paper_trading.market_date import market_now
from screener.config import (
    EMAIL_PASSWORD,
    EMAIL_SENDER,
    MY_ACCOUNT_EMAIL,
    MY_ACCOUNT_TICKER_COLUMN,
    MY_ACCOUNT_WORKSHEET,
    SMTP_PORT,
    SMTP_SERVER,
)
from screener.portfolio_report import STOP_NEAR_PCT, WATCHLIST_WORKSHEET, _open_sheet, _to_float, is_closed

MARKET_OPEN = time(9, 30)
WINDOW_END = time(16, 15)        # 16:07 예약 실행이 마감 요약이 되도록 마감 뒤 15분까지
MOVE_ALERT_PCT = 0.05            # 전날 종가 대비 ±5% 넘으면 급등락 알림
NTFY_URL = "https://ntfy.sh"
ACCOUNT_URL = "https://seancho.vercel.app/quant/account"
CAD_EXCHANGES = ("TSX", "TSXV", "CSE")
_CAD_SUFFIX = re.compile(r"\.(V|TO|CN|NE)$")

_ICON = {"stop": "🛑", "near": "⚠️", "target": "🎯", "up": "🚀", "down": "📉", "watch": "👀"}


# ---------------------------------------------------------------------------
# 시간 · 심볼
# ---------------------------------------------------------------------------

def in_market_window(now: datetime) -> bool:
    """평일 09:30 ≤ 토론토 시각 < 16:15 이면 True (휴장일은 시세로 따로 판정)."""
    return now.weekday() < 5 and MARKET_OPEN <= now.time() < WINDOW_END


def _yf_symbol(ticker: str) -> str:
    """시트 티커 → Yahoo 심볼. 캐나다 접미사(.V/.TO/.CN/.NE)는 두고 클래스 주식 점은 하이픈 (BRK.B → BRK-B)."""
    t = ticker.strip().upper()
    return t if _CAD_SUFFIX.search(t) else t.replace(".", "-")


def is_cad(ticker: str, exchange: str = "") -> bool:
    """캐나다 상장(C$)인지 — 웹 AccountView 와 같은 판정 (거래소 칸 또는 티커 접미사)."""
    return exchange.strip().upper() in CAD_EXCHANGES or bool(_CAD_SUFFIX.search(ticker.strip().upper()))


def _norm_date(v: Any) -> Optional[date]:
    """시트 날짜 글자 → date ('2026-10-21', '2026. 10. 21', '2026/10/21', '10/21/2026'). 못 읽으면 None."""
    s = str(v or "").strip()
    m = re.match(r"^(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
    else:
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", s)
        if not m:
            return None
        mo, d, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 시트 읽기 (쓰지 않는다)
# ---------------------------------------------------------------------------

def _rows_from_values(values: list[list[str]], ticker_col: str = "티커") -> list[dict[str, str]]:
    """get_all_values() 결과 → 티커가 있는 줄 {머리글: 값}. '샘플' 줄은 뺀다."""
    if not values or not values[0]:
        return []
    header = [str(h).strip() for h in values[0]]
    if ticker_col not in header:
        return []
    out = []
    for row in values[1:]:
        rec = {h: (str(row[i]).strip() if i < len(row) else "") for i, h in enumerate(header) if h}
        ticker = rec.get(ticker_col, "").upper()
        if ticker and ticker != "샘플":
            rec[ticker_col] = ticker
            out.append(rec)
    return out


def read_open_rows(sheet) -> list[dict[str, str]]:
    """'내계좌' 탭에서 보유 중(매도일 없음)인 줄만 읽는다."""
    values = sheet.worksheet(MY_ACCOUNT_WORKSHEET).get_all_values()
    return [r for r in _rows_from_values(values, MY_ACCOUNT_TICKER_COLUMN) if not is_closed(r)]


def read_watchlist(sheet) -> list[dict[str, str]]:
    """'관심종목' 탭 (티커·목표가·조건·메모). 탭이 없으면 []."""
    try:
        values = sheet.worksheet(WATCHLIST_WORKSHEET).get_all_values()
    except Exception:
        return []
    return _rows_from_values(values)


# ---------------------------------------------------------------------------
# 시세
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    """한 종목의 장중 시세 (Yahoo 일봉의 마지막 봉)."""

    price: float
    prev_close: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    bar_date: date
    stale: bool  # 오늘 봉이 없음 (오늘 거래 없음 · 휴장) → price 는 마지막 종가

    @property
    def day_pct(self) -> Optional[float]:
        """전날 종가 대비 오늘 등락률. 오늘 봉이 없으면 None."""
        if self.stale or not self.prev_close:
            return None
        return self.price / self.prev_close - 1


def _num(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if pd.notna(x) else None


def quotes_from_frame(df: pd.DataFrame, symbols: dict[str, str], today: date) -> dict[str, Quote]:
    """yf.download(group_by="ticker") 결과 → {시트 티커: Quote}. 데이터가 없는 종목은 빠진다.

    Args:
        symbols: {시트 티커: Yahoo 심볼}
    """
    out: dict[str, Quote] = {}
    if df is None or df.empty:
        return out
    if not isinstance(df.columns, pd.MultiIndex):  # 심볼 하나뿐일 때 대비
        df = pd.concat({next(iter(symbols.values())): df}, axis=1)
    have = set(df.columns.get_level_values(0))
    for ticker, sym in symbols.items():
        if sym not in have:
            continue
        sub = df[sym].dropna(subset=["Close"])
        if sub.empty:
            continue
        last = sub.iloc[-1]
        bar_date = pd.Timestamp(sub.index[-1]).date()
        stale = bar_date != today
        out[ticker] = Quote(
            price=float(last["Close"]),
            prev_close=_num(sub["Close"].iloc[-2]) if len(sub) >= 2 else None,
            day_high=None if stale else _num(last.get("High")),
            day_low=None if stale else _num(last.get("Low")),
            bar_date=bar_date,
            stale=stale,
        )
    return out


def fetch_quotes(tickers: list[str], today: date) -> dict[str, Quote]:
    """보유·관심 종목 + SPY 를 Yahoo 일봉 한 번에 받는다 (auto_adjust=False — 시트에 적은 실제 가격과 비교)."""
    symbols = {t: _yf_symbol(t) for t in dict.fromkeys([*tickers, "SPY"])}
    try:
        df = yf.download(list(dict.fromkeys(symbols.values())), period="5d", interval="1d", auto_adjust=False,
                         group_by="ticker", progress=False, threads=True)
    except Exception as exc:
        print(f"[계좌감시] 시세 받기 실패: {exc}")
        return {}
    return quotes_from_frame(df, symbols, today)


# ---------------------------------------------------------------------------
# 포지션 · 알림 · 요약
# ---------------------------------------------------------------------------

def _money(x: float, cad: bool = False) -> str:
    cur = "C$" if cad else "$"
    return f"{cur}{x:,.2f}" if abs(x) >= 1 else f"{cur}{x:.4g}"


def _signed_money(x: float, cad: bool = False) -> str:
    cur = "C$" if cad else "$"
    return f"{'+' if x >= 0 else '-'}{cur}{abs(x):,.0f}"


def positions(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """보유 줄 → 티커별 묶음 {qty, buy(수량 가중 평균), cad, stops, targets, rows}. 시트 순서 유지."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = r.get("티커", "").upper()
        p = out.setdefault(t, {"qty": 0.0, "cost": 0.0, "buys": [], "cad": False,
                               "stops": set(), "targets": set(), "rows": []})
        p["rows"].append(r)
        p["cad"] = p["cad"] or is_cad(t, r.get("거래소", ""))
        qty, buy = _to_float(r.get("수량")), _to_float(r.get("매수가"))
        if buy:
            p["buys"].append(buy)
            if qty:
                p["qty"] += qty
                p["cost"] += qty * buy
        stop, target = _to_float(r.get("손절가")), _to_float(r.get("목표가"))
        if stop and stop > 0:
            p["stops"].add(stop)
        if target and target > 0:
            p["targets"].add(target)
    for p in out.values():
        p["buy"] = p["cost"] / p["qty"] if p["qty"] else (sum(p["buys"]) / len(p["buys"]) if p["buys"] else None)
    return out


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:+.1%}"


def _context(q: Quote, buy: Optional[float], cad: bool) -> str:
    """'현재 $812.40 · 오늘 +1.2% · 매수 대비 +6.5%' — 모든 알림·요약 줄에 붙는 꼬리."""
    parts = [f"현재 {_money(q.price, cad)}", "오늘 거래 없음" if q.stale else f"오늘 {_pct(q.day_pct)}"]
    if buy:
        parts.append(f"매수 대비 {q.price / buy - 1:+.1%}")
    return " · ".join(parts)


def build_watch_alerts(rows: list[dict[str, str]], watch: list[dict[str, str]],
                       quotes: dict[str, Quote]) -> list[dict[str, str]]:
    """보유·관심 종목 알림. 같은 티커·같은 수준은 한 번만. key 에 수준이 들어가 손절가를 바꾸면 새 알림이 된다."""
    alerts: list[dict[str, str]] = []

    def add(kind: str, ticker: str, key: str, text: str) -> None:
        alerts.append({"kind": kind, "ticker": ticker, "key": key, "text": f"{_ICON[kind]} {text}"})

    for t, p in positions(rows).items():
        q = quotes.get(t)
        if q is None:
            continue
        cad, ctx = p["cad"], _context(q, p["buy"], p["cad"])
        for stop in sorted(p["stops"], reverse=True):
            if q.price <= stop:
                add("stop", t, f"stop:{t}:{stop:g}", f"{t} 손절가 {_money(stop, cad)} 아래로 내려왔어요 — {ctx}")
            elif q.day_low is not None and q.day_low <= stop:
                add("stop", t, f"stop:{t}:{stop:g}",
                    f"{t} 장중 저가 {_money(q.day_low, cad)} — 손절가 {_money(stop, cad)} 아래까지 내려갔다 올라왔어요 — {ctx}")
            elif q.price <= stop * (1 + STOP_NEAR_PCT):
                add("near", t, f"near:{t}:{stop:g}",
                    f"{t} 손절가 {_money(stop, cad)} 까지 {q.price / stop - 1:.1%} 남았어요 — {ctx}")
        for target in sorted(p["targets"]):
            if q.price >= target or (q.day_high is not None and q.day_high >= target):
                how = "" if q.price >= target else f" (장중 고가 {_money(q.day_high, cad)})"
                add("target", t, f"target:{t}:{target:g}",
                    f"{t} 목표가 {_money(target, cad)} 도달{how} — {ctx}. 일부 익절하거나 손절가를 올릴 때예요")
        pct = q.day_pct
        if pct is not None and pct >= MOVE_ALERT_PCT:
            add("up", t, f"up:{t}", f"{t} 오늘 {pct:+.1%} 급등 — {ctx}")
        elif pct is not None and pct <= -MOVE_ALERT_PCT:
            add("down", t, f"down:{t}", f"{t} 오늘 {pct:+.1%} 급락 — {ctx}")

    seen: set[str] = set()
    for w in watch:
        t, target = w.get("티커", "").upper(), _to_float(w.get("목표가"))
        q = quotes.get(t)
        if q is None or not target:
            continue
        above = str(w.get("조건", "")).strip() in ("이상", "above", ">=")  # update_watchlist 와 같은 판정
        hit = (q.price >= target or (q.day_high or 0) >= target) if above else \
              (q.price <= target or (q.day_low is not None and q.day_low <= target))
        cond = "이상" if above else "이하"
        key = f"watch:{t}:{target:g}:{cond}"
        if hit and key not in seen:
            seen.add(key)
            cad = is_cad(t)
            now_hit = q.price >= target if above else q.price <= target
            how = "" if now_hit else \
                f" (장중 {'고가' if above else '저가'} {_money(q.day_high if above else q.day_low, cad)})"
            memo = f" · 메모: {w['메모']}" if w.get("메모") else ""
            add("watch", t, key, f"관심종목 {t} 목표가 {_money(target, cad)} {cond} 도달{how} — "
                                 f"{_context(q, None, cad)}{memo}")
    return alerts


def build_summary(rows: list[dict[str, str]], quotes: dict[str, Quote], now: datetime) -> tuple[str, str]:
    """(제목, 본문) — 종목별 한 줄 + 오늘 손익·평가 손익 합계 (USD · C$ 따로)."""
    spy = quotes.get("SPY")
    spy_txt = f" · SPY {spy.day_pct:+.2%}" if spy and spy.day_pct is not None else ""
    title = f"내계좌 {now:%H:%M}{spy_txt}"
    lines: list[str] = []
    day = {False: 0.0, True: 0.0}
    unreal = {False: 0.0, True: 0.0}
    has = {False: False, True: False}
    today = now.date()
    for t, p in positions(rows).items():
        q = quotes.get(t)
        if q is None:
            lines.append(f"{t} 시세 없음")
            continue
        lines.append(f"{t} {_context(q, p['buy'], p['cad']).removeprefix('현재 ')}")
        for r in p["rows"]:
            qty, buy = _to_float(r.get("수량")), _to_float(r.get("매수가"))
            if not qty:
                continue
            has[p["cad"]] = True
            if buy:
                unreal[p["cad"]] += qty * (q.price - buy)
            base = buy if (buy and _norm_date(r.get("매수일")) == today) else (None if q.stale else q.prev_close)
            if base:
                day[p["cad"]] += qty * (q.price - base)
    for cad in (False, True):
        if has[cad]:
            lines.append(f"오늘 {_signed_money(day[cad], cad)} · 평가 손익 {_signed_money(unreal[cad], cad)}"
                         f" ({'CAD' if cad else 'USD'})")
    return title, "\n".join(lines)


# ---------------------------------------------------------------------------
# 상태 (같은 알림 하루 한 번)
# ---------------------------------------------------------------------------

def load_state(path: Optional[Path], today: date) -> dict[str, Any]:
    """{"date", "sent": [key…]}. 날짜가 바뀌었거나 파일이 없으면 빈 상태."""
    state = read_json(path, {}) if path else {}
    if state.get("date") != today.isoformat():
        return {"date": today.isoformat(), "sent": []}
    return {"date": state["date"], "sent": list(state.get("sent", []))}


def save_state(path: Path, state: dict[str, Any]) -> None:
    atomic_json_write(path, state)


# ---------------------------------------------------------------------------
# 보내기
# ---------------------------------------------------------------------------

def send_ntfy(topic: str, title: str, body: str, priority: int = 3, tags: Optional[list[str]] = None) -> bool:
    """ntfy.sh 로 폰 푸시 (JSON publish — 한글 제목을 헤더에 넣지 않으려고)."""
    try:
        resp = requests.post(NTFY_URL, json={"topic": topic, "title": title, "message": body,
                                             "priority": priority, "tags": tags or [], "click": ACCOUNT_URL},
                             timeout=15)
        resp.raise_for_status()
        return True
    except Exception as exc:
        print(f"[계좌감시] ❌ ntfy 실패: {exc}")
        return False


def _email_html(alerts: list[dict[str, str]], rows: list[dict[str, str]], quotes: dict[str, Quote]) -> str:
    items = "".join(f"<li style='margin:4px 0'>{a['text']}</li>" for a in alerts)
    th = "".join(f"<th style='padding:4px 8px;text-align:right;border-bottom:1px solid #d1d5db'>{h}</th>"
                 for h in ("티커", "현재가", "오늘", "매수 대비", "손절까지", "목표까지"))
    body = ""
    for t, p in positions(rows).items():
        q = quotes.get(t)
        if q is None:
            continue
        stop = max(p["stops"]) if p["stops"] else None
        target = min(p["targets"]) if p["targets"] else None
        cells = [t, _money(q.price, p["cad"]), "거래 없음" if q.stale else _pct(q.day_pct),
                 _pct(q.price / p["buy"] - 1) if p["buy"] else "—",
                 _pct(q.price / stop - 1) if stop else "—", _pct(target / q.price - 1) if target else "—"]
        body += "<tr>" + "".join(f"<td style='padding:4px 8px;text-align:right'>{c}</td>" for c in cells) + "</tr>"
    return (f"<div style='font-family:-apple-system,Segoe UI,sans-serif;font-size:14px'>"
            f"<div style='background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:12px 16px'>"
            f"<b>장중 알림</b><ul style='margin:8px 0 0;padding-left:18px'>{items}</ul></div>"
            f"<table style='border-collapse:collapse;margin-top:16px'><tr>{th}</tr>{body}</table>"
            f"<p style='color:#6b7280;font-size:12px'>30분마다 Yahoo 시세로 확인 · 같은 알림은 하루 한 번 · "
            f"<a href='{ACCOUNT_URL}'>웹 내계좌</a></p></div>")


def send_alert_email(alerts: list[dict[str, str]], rows: list[dict[str, str]], quotes: dict[str, Quote],
                     now: datetime) -> bool:
    """새 알림이 있을 때만 부른다 — 알림 목록 + 보유 표."""
    if not EMAIL_PASSWORD:
        print("[계좌감시] EMAIL_PASSWORD 미설정 — 메일 건너뜀")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = MY_ACCOUNT_EMAIL
        msg["Subject"] = f"⚠️ 장중 알림 {len(alerts)}건 · 내계좌 [{now:%H:%M} {now.tzname()}]"
        msg.attach(MIMEText(_email_html(alerts, rows, quotes), "html", "utf-8"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"[계좌감시] ✅ 메일 발송 → {MY_ACCOUNT_EMAIL}")
        return True
    except Exception as exc:
        print(f"[계좌감시] ❌ 메일 실패: {exc}")
        return False


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def run(state_path: Optional[Path] = None, force: bool = False, dry_run: bool = False,
        tickers_override: Optional[list[str]] = None, now: Optional[datetime] = None) -> int:
    """한 번 체크. 반환값은 종료 코드 (시트를 못 읽으면 1, 나머지는 0)."""
    now = now or market_now()
    if not force and not in_market_window(now):
        print(f"[계좌감시] 장 시간 밖 ({now:%a %H:%M %Z}) — 건너뜀")
        return 0
    today = now.date()

    if tickers_override:
        rows, watch = [{"티커": t.strip().upper()} for t in tickers_override if t.strip()], []
    else:
        sheet = _open_sheet()
        if sheet is None:
            print("[계좌감시] ❌ 시트를 열지 못함")
            return 1
        try:
            rows, watch = read_open_rows(sheet), read_watchlist(sheet)
        except Exception as exc:
            print(f"[계좌감시] ❌ 시트 읽기 실패: {exc}")
            return 1
    tickers = list(dict.fromkeys([r["티커"] for r in rows] + [w["티커"] for w in watch]))
    print(f"[계좌감시] 보유 {len({r['티커'] for r in rows})} · 관심 {len(watch)}: {tickers}")
    if not tickers:
        return 0

    quotes = fetch_quotes(tickers, today)
    spy = quotes.get("SPY")
    if not force and (spy is None or spy.stale):
        print("[계좌감시] SPY 오늘 봉 없음 (휴장 또는 시세 실패) — 건너뜀")
        return 0
    missing = [t for t in tickers if t not in quotes]
    if missing:
        print(f"[계좌감시] 시세 없음: {missing}")

    alerts = build_watch_alerts(rows, watch, quotes)
    state = load_state(state_path, today)
    sent = set(state["sent"])
    new = [a for a in alerts if a["key"] not in sent]
    title, summary = build_summary(rows, quotes, now)
    print(f"[계좌감시] 알림 {len(alerts)}건 (새 알림 {len(new)}건)")
    for a in alerts:
        print(f"  {'NEW ' if a in new else 'sent'} {a['text']}")
    print(f"--- {title}\n{summary}")
    if dry_run:
        print("[계좌감시] --dry-run — 보내지 않음")
        return 0

    topic = os.getenv("NTFY_TOPIC", "").strip()
    pushed = False
    if topic:
        if new:
            pushed = send_ntfy(topic, f"⚠️ 알림 {len(new)}건 · {title}",
                               "\n".join(a["text"] for a in new) + "\n\n" + summary, priority=4, tags=["warning"])
        else:
            pushed = send_ntfy(topic, title, summary, priority=2)
        print(f"[계좌감시] ntfy {'✅' if pushed else '❌'}")
    else:
        print("[계좌감시] NTFY_TOPIC 미설정 — 푸시 건너뜀")
    mailed = send_alert_email(new, rows, quotes, now) if new else False

    if new and (pushed or mailed):
        state["sent"] = sorted(sent | {a["key"] for a in new})
        if state_path:
            save_state(state_path, state)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="내계좌 장중 30분 체크 (ntfy 요약 + 새 알림 메일)")
    ap.add_argument("--state", type=Path, default=None, help="같은 알림 하루 한 번용 state.json (없으면 기억하지 않음)")
    ap.add_argument("--force", action="store_true", help="장 시간·휴장 확인을 건너뛴다 (수동 실행·테스트)")
    ap.add_argument("--dry-run", action="store_true", help="보내지 않고 상태도 쓰지 않는다 — 출력만")
    ap.add_argument("--tickers", default="", help="시트 대신 이 티커들 (쉼표) — 시세·요약 확인용")
    args = ap.parse_args(argv)
    return run(args.state, force=args.force, dry_run=args.dry_run,
               tickers_override=[t for t in args.tickers.split(",") if t.strip()] or None)


if __name__ == "__main__":
    sys.exit(main())
