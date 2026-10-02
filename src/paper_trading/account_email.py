"""PT-2/PT-3 계좌별 일일 이메일 (계좌마다 1통).

PT-1 이메일(screener.exporter.send_paper_trading_email)은 그대로 두고 따로 보낸다.
"""

from __future__ import annotations

import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from paper_trading.indicators import is_num

_TABLE = "border-collapse:collapse;font-size:13px;margin:6px 0 14px 0"
_TH = "background:#f2f2f2;border:1px solid #ddd;padding:4px 8px;text-align:left"
_TD = "border:1px solid #ddd;padding:4px 8px"


def _esc(v: Any) -> str:
    return html.escape(str(v if v is not None else ""))


def _price(v: Any) -> str:
    return f"${float(v):,.2f}" if is_num(v) else "-"


def _pct(v: Any) -> str:
    if not is_num(v):
        return "-"
    color = "#27ae60" if float(v) >= 0 else "#e74c3c"
    return f"<span style='color:{color};font-weight:bold'>{float(v):+.1%}</span>"


def _table(headers: list[str], rows: list[list[str]], empty: str = "없음") -> str:
    if not rows:
        return f"<p style='color:#999'>{_esc(empty)}</p>"
    head = "".join(f"<th style='{_TH}'>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td style='{_TD}'>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table style='{_TABLE}'><tr>{head}</tr>{body}</table>"


def _max_drawdown(history: list[dict[str, Any]]) -> float:
    peak, mdd = 0.0, 0.0
    for h in history:
        eq = float(h["equity"])
        peak = max(peak, eq)
        if peak > 0:
            mdd = min(mdd, eq / peak - 1.0)
    return mdd


def build_account_email(profile: Any, result: dict[str, Any], state: Any) -> tuple[str, str]:
    """(제목, HTML 본문)."""
    d = result["date"]
    active = result["buys"] or result["sells"] or result["orders"]
    subject = f"[{profile.name}] {d} {'매매 알림' if active else '일일 리포트'}"

    trades = state.trades
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    avg = sum(t["return_pct"] for t in trades) / len(trades) if trades else None
    summary = _table(["평가액", "누적수익(자본)", "현금", "보유", "청산 거래", "승률", "평균수익(비용 반영)", "MDD"], [[
        _price(state.equity),
        _pct(state.equity / state.initial_capital - 1.0 if state.initial_capital else None),
        _price(state.cash),
        f"{len(state.positions)} / {profile.params['max_positions']}",
        str(len(trades)),
        f"{wins / len(trades):.0%}" if trades else "-",
        _pct(avg),
        _pct(_max_drawdown(state.equity_history)),
    ]])

    buys = _table(["종목", "매수가(시가)", "수량", "손절가", "목표가", "전략"], [[
        _esc(b["ticker"]), _price(b["entry_price"]), f"{float(b['shares']):.3f}",
        _price(b.get("stop_price")), _price(b.get("target_price")), _esc(b["strategy"]),
    ] for b in result["buys"]])

    sells = _table(["종목", "매수일", "매수가", "매도가", "수익률", "보유(일/거래일)", "사유"], [[
        _esc(t["ticker"]), _esc(t["entry_date"]), _price(t["entry_price"]), _price(t["exit_price"]),
        _pct(t["return_pct"]), f"{t['holding_days']} / {t['holding_bars']}", _esc(t["exit_reason"]),
    ] for t in result["sells"]])

    orders = _table(["구분", "종목", "신호 종가", "점수", "전략/사유"], [[
        "매수" if o["side"] == "BUY" else "매도", _esc(o["ticker"]), _price(o.get("signal_close")),
        f"{o['score']:.3f}" if o["side"] == "BUY" else "-",
        _esc(o.get("strategy") if o["side"] == "BUY" else o.get("reason")),
    ] for o in result["orders"]], empty="예약 없음")

    cancelled = _table(["종목", "취소 사유"], [
        [_esc(c["ticker"]), _esc(c["cancel_reason"])] for c in result["cancelled"]
    ])

    holdings = _table(["종목", "매수일", "매수가", "종가", "수익률", "손절가", "목표가", "보유 거래일", "전략"], [[
        _esc(p["ticker"]), _esc(p["entry_date"]), _price(p["entry_price"]), _price(p.get("last_close")),
        _pct(float(p.get("last_close", p["entry_price"])) / float(p["entry_price"]) - 1.0),
        f"{_price(p.get('stop_price'))} ({_esc(p.get('stop_kind', ''))})", _price(p.get("target_price")),
        str(p.get("bars_held", 0)), _esc(p["strategy"]),
    ] for p in state.positions], empty="보유 종목 없음")

    candidates = _table(["종목", "점수", "전략", "섹터", "별점"], [[
        _esc(c["ticker"]), f"{c['score']:.3f}", _esc(c["strategy"]), _esc(c["sector"]), _esc(c.get("star", "")),
    ] for c in result["candidates"]], empty="조건을 만족한 후보 없음")

    debug = result.get("debug") or {}
    debug_html = f"<p style='color:#666;font-size:12px'>선정 과정: {_esc(debug)}</p>" if debug else ""

    body = f"""
<h2 style='margin-bottom:4px'>{_esc(profile.name)} — {_esc(d)}</h2>
<p style='color:#666;margin-top:0'>버전 {_esc(profile.version)} · 신호는 장 마감 기준, 체결은 다음날 시가 · 편도 비용 {profile.params['cost_per_side']:.2%} 반영</p>
<h3>계좌 요약</h3>{summary}
<h3>오늘 시가 매수</h3>{buys}
<h3>오늘 매도</h3>{sells}
<h3>다음날 시가 예약</h3>{orders}
<h3>취소된 주문</h3>{cancelled}
<h3>보유 종목</h3>{holdings}
<h3>후보 (점수순)</h3>{candidates}
{debug_html}
<p style='color:#999;font-size:12px'>본 메일은 시스템에 의해 자동으로 발송되었습니다.</p>
"""
    return subject, body


def send_account_email(profile: Any, result: dict[str, Any], state: Any) -> bool:
    """계좌 이메일 1통 발송. 변동도 보유도 없으면 생략."""
    from screener.config import (
        EMAIL_ENABLED, EMAIL_PASSWORD, EMAIL_RECIPIENTS, EMAIL_SENDER, SMTP_PORT, SMTP_SERVER,
    )

    if not EMAIL_ENABLED:
        return False
    if not (result["buys"] or result["sells"] or result["orders"] or state.positions):
        print(f"[{profile.name}] 변동·보유 없음 — 이메일 생략")
        return False
    if not EMAIL_PASSWORD:
        print(f"[{profile.name}] EMAIL_PASSWORD 없음 — 이메일 생략")
        return False

    subject, body = build_account_email(profile, result, state)
    recipients = profile.params.get("email_recipients") or EMAIL_RECIPIENTS
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_SENDER
    msg["Bcc"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
    print(f"[{profile.name}] 이메일 발송 완료 → {len(recipients)}명")
    return True
