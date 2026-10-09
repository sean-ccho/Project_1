#!/usr/bin/env python3
"""S&P 500 현재 구성종목을 위키백과에서 받아 src/data/sp500_tickers.py 를 다시 쓴다.

손으로 한 번 긁어 둔 목록이 낡아서 (2026-10-09: 현재 구성종목 29개가 빠지고, 빠진 지 오래된 25개가 남음 —
인수·상장폐지로 가격도 없는 BK·EA·EQR 등) 실거래 스크리너와 PT-1S 가 새 편입 종목(VRT·CVNA·RDDT·SNDK 등)을
보지 못했다. 분기마다, 또는 편입·편출 소식이 있을 때 돌린다.

사용법: .venv/bin/python scripts/update_sp500_tickers.py [--dry-run]
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "data" / "sp500_tickers.py"
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MIN_COUNT = 495  # 표 구조가 바뀌어 일부만 읽히면 쓰지 않는다


def fetch_current() -> list[str]:
    """위키백과 구성종목 표 순서 그대로, yfinance 표기 (BRK.B → BRK-B)."""
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (Project_1 ticker update)"}, timeout=30)
    r.raise_for_status()
    table = pd.read_html(io.StringIO(r.text))[0]
    tickers = [str(s).strip().upper().replace(".", "-") for s in table["Symbol"]]
    return list(dict.fromkeys(t for t in tickers if t))


def render(tickers: list[str]) -> str:
    """sp500_tickers.py 본문 (받은 날짜를 docstring 에 남긴다)."""
    today = datetime.now(ZoneInfo("America/Toronto")).date().isoformat()
    body = "".join(f'    "{t}",\n' for t in tickers)
    return (f'"""S&P 500 ticker universe scraped from Wikipedia ({today}, scripts/update_sp500_tickers.py)."""\n\n'
            f"SP500_TICKERS = [\n{body}]\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="바뀌는 종목만 보여 주고 파일은 쓰지 않는다")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from data.sp500_tickers import SP500_TICKERS as old

    new = fetch_current()
    if len(new) < MIN_COUNT:
        raise SystemExit(f"구성종목이 {len(new)}개뿐 — 위키백과 표 구조가 바뀐 듯해 쓰지 않음")
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    print(f"현재 {len(new)}개 (기존 {len(old)}개)")
    print(f"추가 {len(added)}: {added}")
    print(f"제거 {len(removed)}: {removed}")
    if not args.dry_run:
        TARGET.write_text(render(new), encoding="utf-8")
        print(f"→ {TARGET.relative_to(ROOT)} 갱신")


if __name__ == "__main__":
    main()
