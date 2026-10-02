#!/usr/bin/env python3
"""S&P 500 과거 구성종목 CSV 다운로드 (생존 편향 완화용, --pit-universe).

출처: github.com/fja05680/sp500 — "S&P 500 Historical Components & Changes" (date,tickers).
저장: config.SP500_MEMBERSHIP_PATH (기본 data/universe/sp500_membership.csv)

사용법:
  PYTHONPATH=.:src python scripts/fetch_sp500_membership.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.universe import Membership, load_membership
from screener import config as cfg

URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def main() -> None:
    dest = ROOT / cfg.SP500_MEMBERSHIP_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"다운로드: {URL}")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        data = resp.read()
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)

    rows = load_membership(tmp)
    if len(rows) < 100 or len(rows[-1][1]) < 450:
        tmp.unlink()
        raise SystemExit(f"형식이 예상과 다름 (행 {len(rows)}개) — 저장하지 않음")
    tmp.replace(dest)

    m = Membership(rows)
    last_date, last_set = rows[-1]
    print(f"저장: {dest} ({rows[0][0]} ~ {last_date}, {len(rows)}개 변경 시점)")
    print(f"최근 구성종목 {len(last_set)}개, 최근 5년 누적 {len(m.union_between(str(int(last_date[:4]) - 5) + last_date[4:], last_date))}개")


if __name__ == "__main__":
    main()
