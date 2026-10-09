#!/usr/bin/env python3
"""S&P 500 현재 구성종목을 위키백과에서 받아 src/data/sp500_tickers.py 를 다시 쓴다.

매달 1일 GitHub Actions(.github/workflows/update-sp500.yml)가 --notify 로 돌린다. 바뀐 종목은
data/universe/sp500_changes.csv 에 쌓고 메일로 알린다. 한 번에 너무 많이 바뀌면(MAX_CHANGES 초과) 위키백과 표가
깨졌거나 훼손된 것으로 보고 반영하지 않고 확인 메일만 보낸다.

2026-10-09: 손으로 한 번 긁어 둔 목록이 낡아서 현재 구성종목 29개가 빠지고 빠진 지 오래된 25개가 남아 있었다
(가격도 없는 BK·EA·EQR 등). 실거래 스크리너와 PT-1S 가 새 편입 종목(VRT·CVNA·RDDT·SNDK 등)을 보지 못했다.

사용법: .venv/bin/python scripts/update_sp500_tickers.py [--dry-run] [--notify] [--force]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import smtplib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "data" / "sp500_tickers.py"
CHANGE_LOG = ROOT / "data" / "universe" / "sp500_changes.csv"
POSITION_FILES = {
    "PT-1": ROOT / "data" / "paper_trading" / "positions.json",
    "PT-1S": ROOT / "data" / "paper_trading" / "pt1s" / "positions.json",
}
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
COUNT_RANGE = (495, 510)  # S&P 500 은 2주식 회사 때문에 503개 안팎 — 벗어나면 표 구조가 바뀐 것
MAX_CHANGES = 20          # 한 달 변경(추가+제거)은 보통 0~10개 — 이보다 많으면 반영하지 않고 확인을 받는다


def fetch_table() -> pd.DataFrame:
    """위키백과 구성종목 표: 티커(yfinance 표기, BRK.B → BRK-B)·회사·섹터·편입일."""
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (Project_1 ticker update)"}, timeout=30)
    r.raise_for_status()
    raw = pd.read_html(io.StringIO(r.text))[0]
    table = pd.DataFrame({
        "티커": raw["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False),
        "회사": raw.get("Security", ""),
        "섹터": raw.get("GICS Sector", ""),
        "편입일": raw.get("Date added", ""),
    })
    return table[table["티커"] != ""].drop_duplicates("티커").reset_index(drop=True)


@dataclass
class Plan:
    """이번 갱신에서 바뀌는 것."""

    tickers: list[str]                                          # 새 목록 (위키백과 표 순서)
    added: list[dict[str, Any]] = field(default_factory=list)  # 추가 종목 행 (티커·회사·섹터·편입일)
    removed: list[str] = field(default_factory=list)
    problem: str = ""                                           # 비어 있지 않으면 반영하지 않는다

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


def plan_update(old: list[str], table: pd.DataFrame) -> Plan:
    """기존 목록과 위키백과 표를 비교한다. 변경이 너무 많으면 problem 에 이유를 적는다."""
    plan = Plan(tickers=table["티커"].tolist(),
                added=table[~table["티커"].isin(old)].to_dict("records"),
                removed=sorted(set(old) - set(table["티커"])))
    n_changes = len(plan.added) + len(plan.removed)
    if n_changes > MAX_CHANGES:
        plan.problem = f"변경 {n_changes}개 (보통 한 달 0~10개, 기준 {MAX_CHANGES}개) — 표가 깨졌거나 훼손됐을 수 있음"
    return plan


def render(tickers: list[str], today: str) -> str:
    """sp500_tickers.py 본문 (받은 날짜를 docstring 에 남긴다)."""
    body = "".join(f'    "{t}",\n' for t in tickers)
    return (f'"""S&P 500 ticker universe scraped from Wikipedia ({today}, scripts/update_sp500_tickers.py)."""\n\n'
            f"SP500_TICKERS = [\n{body}]\n")


def append_change_log(plan: Plan, today: str, path: Path = CHANGE_LOG) -> None:
    """바뀐 종목을 변경 기록 CSV 에 덧붙인다."""
    rows = [{"날짜": today, "구분": "추가", "티커": a["티커"], "회사": a.get("회사", ""), "섹터": a.get("섹터", "")}
            for a in plan.added]
    rows += [{"날짜": today, "구분": "제거", "티커": t, "회사": "", "섹터": ""} for t in plan.removed]
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def held_tickers() -> dict[str, list[str]]:
    """계좌별 지금 보유 종목 (상태 파일이 없으면 빈 목록)."""
    held: dict[str, list[str]] = {}
    for account, path in POSITION_FILES.items():
        try:
            positions = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            positions = []
        held[account] = [str(p.get("ticker", "")) for p in positions if isinstance(p, dict)]
    return held


def change_message(plan: Plan, today: str, held: dict[str, list[str]]) -> tuple[str, str]:
    """반영한 변경을 알리는 메일 제목·본문."""
    lines = [f"S&P 500 구성종목이 바뀌어 목록을 갱신했습니다 (위키백과 기준, {today}).", ""]
    if plan.added:
        lines.append(f"■ 추가 {len(plan.added)}개 — 다음 실행부터 S&P 스크리너·PT-1S 후보에 들어갑니다")
        lines += [f"  - {a['티커']}  {a.get('회사', '')} ({a.get('섹터', '')}, 편입 {a.get('편입일', '')})"
                  for a in plan.added]
        lines.append("")
    if plan.removed:
        lines.append(f"■ 제거 {len(plan.removed)}개 — 후보에서 빠집니다")
        lines += [f"  - {t}" for t in plan.removed]
        lines.append("")
    gone = [f"{account}: {t}" for account, tickers in held.items() for t in tickers if t in plan.removed]
    if gone:
        lines.append("■ 보유 중인 종목이 빠졌습니다: " + ", ".join(gone))
        lines.append("  이미 산 종목은 손절·목표가 등 원래 매도 규칙대로 계속 관리됩니다.")
        lines.append("")
    lines.append(f"전체 기록: {CHANGE_LOG.relative_to(ROOT)}")
    subject = f"[S&P 500 목록] {today} 갱신: 추가 {len(plan.added)} · 제거 {len(plan.removed)}"
    return subject, "\n".join(lines)


def problem_message(plan: Plan, today: str) -> tuple[str, str]:
    """반영을 막은 이유를 알리는 메일 제목·본문."""
    body = "\n".join([
        f"S&P 500 목록 월간 갱신을 반영하지 않았습니다 ({today}) — 확인이 필요합니다.",
        "",
        f"이유: {plan.problem}",
        f"위키백과 기준 추가 {len(plan.added)}개 · 제거 {len(plan.removed)}개",
        "",
        "직접 확인한 뒤 맞으면 이렇게 반영합니다:",
        "  .venv/bin/python scripts/update_sp500_tickers.py --dry-run   (바뀌는 종목 보기)",
        "  .venv/bin/python scripts/update_sp500_tickers.py --force     (반영)",
    ])
    return f"⚠️ [S&P 500 목록] {today} 갱신 보류 — 확인 필요", body


def send_email(subject: str, body: str) -> None:
    """프로젝트 알림 메일 (EMAIL_PASSWORD 가 없으면 건너뛴다). 받는 사람은 screener.config 설정."""
    password = os.environ.get("EMAIL_PASSWORD", "")
    if not password:
        print("EMAIL_PASSWORD 없음 — 메일 생략")
        return
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from screener.config import EMAIL_RECIPIENTS, EMAIL_SENDER, SMTP_PORT, SMTP_SERVER

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_SENDER  # 발신자만 표시하고 실제 수신자는 BCC (exporter 메일과 같게)
    msg["Bcc"] = ", ".join(EMAIL_RECIPIENTS)
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, password)
        server.send_message(msg)
    print(f"메일 발송: {subject}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="바뀌는 종목만 보여 주고 파일은 쓰지 않는다")
    ap.add_argument("--notify", action="store_true", help="반영했거나 반영을 막았으면 메일로 알린다")
    ap.add_argument("--force", action="store_true", help="변경이 많아도 반영한다 (직접 확인했을 때만)")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from data.sp500_tickers import SP500_TICKERS as old

    today = datetime.now(ZoneInfo("America/Toronto")).date().isoformat()
    table = fetch_table()
    if not COUNT_RANGE[0] <= len(table) <= COUNT_RANGE[1]:
        raise SystemExit(f"구성종목이 {len(table)}개 (정상 {COUNT_RANGE[0]}~{COUNT_RANGE[1]}개) — "
                         "위키백과 표 구조가 바뀐 듯해 쓰지 않음")
    plan = plan_update(list(old), table)
    print(f"현재 {len(plan.tickers)}개 (기존 {len(old)}개)")
    print(f"추가 {len(plan.added)}: {[a['티커'] for a in plan.added]}")
    print(f"제거 {len(plan.removed)}: {plan.removed}")
    if not plan.changed:
        print("변경 없음")
        return
    if plan.problem and not args.force:
        print(f"⚠️ 반영 안 함: {plan.problem}")
        if args.notify:
            send_email(*problem_message(plan, today))
        return
    if args.dry_run:
        return
    TARGET.write_text(render(plan.tickers, today), encoding="utf-8")
    append_change_log(plan, today)
    print(f"→ {TARGET.relative_to(ROOT)} 갱신, 변경 기록 {CHANGE_LOG.relative_to(ROOT)}")
    if args.notify:
        send_email(*change_message(plan, today, held_tickers()))


if __name__ == "__main__":
    main()
