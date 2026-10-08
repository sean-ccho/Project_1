#!/usr/bin/env python3
"""SEC EDGAR 원자료 수집: 실적 발표 8-K(Item 2.02) 제출 기록 + XBRL 재무(companyfacts).

대상: px10y 패널 종목 (현재 SEC 티커 목록에 있는 것만 CIK 매핑 — 상장폐지 종목은 커버리지로 보고).
SEC 규칙: User-Agent에 이름·이메일, 초당 10회 이하. 이메일은 코드에 쓰지 않고 환경변수로 받는다.

사용법:
  SEC_USER_AGENT="이름 you@example.com" PYTHONPATH=.:src python scripts/fetch_sec_data.py
산출 (git 제외):
  data/research/sec/submissions/CIK##########.json.gz (+ 과거 파일)
  data/research/sec/companyfacts/CIK##########.json.gz
  data/research/sec/ticker_cik.csv
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import requests

from research_utils import RESEARCH_DIR

SEC_DIR = RESEARCH_DIR / "sec"
SUB_URL = "https://data.sec.gov/submissions/"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
MIN_INTERVAL = 0.12  # 초당 10회 이하

_last_call = 0.0


def _get(url: str, ua: str) -> bytes | None:
    """SEC GET (속도 제한·재시도). 404 는 None."""
    global _last_call
    for attempt in range(4):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
        if r.status_code == 404:
            return None
        if r.status_code in (429, 503):
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"SEC 요청 실패: {url}")


def _save(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(content)


def panel_tickers() -> list[str]:
    """px10y 패널 종목 목록."""
    p = pd.read_parquet(RESEARCH_DIR / "panel_px10y.parquet", columns=["티커"])
    return sorted(p["티커"].unique())


def ticker_cik_map(ua: str) -> pd.DataFrame:
    """SEC 현재 티커 → CIK. yfinance 표기(BRK-B)와 SEC 표기(BRK-B)를 같게 맞춘다."""
    data = json.loads(_get(TICKERS_URL, ua))
    rows = [{"티커": v["ticker"].upper().replace(".", "-"), "cik": int(v["cik_str"]), "name": v["title"]}
            for v in data.values()]
    return pd.DataFrame(rows).drop_duplicates("티커")


def fetch_company(cik: int, ua: str) -> None:
    """한 회사의 submissions(+과거 파일)·companyfacts 를 받아 둔다 (이미 있으면 건너뜀)."""
    sub_path = SEC_DIR / "submissions" / f"CIK{cik:010d}.json.gz"
    if not sub_path.exists():
        raw = _get(f"{SUB_URL}CIK{cik:010d}.json", ua)
        if raw is not None:
            for f in json.loads(raw)["filings"].get("files", []):
                extra = SEC_DIR / "submissions" / f"{f['name']}.gz"
                if not extra.exists():
                    more = _get(SUB_URL + f["name"], ua)
                    if more is not None:
                        _save(extra, more)
            _save(sub_path, raw)
    facts_path = SEC_DIR / "companyfacts" / f"CIK{cik:010d}.json.gz"
    if not facts_path.exists():
        raw = _get(f"{FACTS_URL}CIK{cik:010d}.json", ua)
        if raw is not None:
            _save(facts_path, raw)


def main() -> None:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        sys.exit("환경변수 SEC_USER_AGENT='이름 이메일' 이 필요합니다 (SEC 규칙)")
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    tickers = panel_tickers()
    cmap = ticker_cik_map(ua)
    m = pd.DataFrame({"티커": tickers}).merge(cmap, on="티커", how="left")
    m.to_csv(SEC_DIR / "ticker_cik.csv", index=False)
    found = m.dropna(subset=["cik"])
    print(f"[SEC] 패널 {len(tickers)}종목 중 CIK 매핑 {len(found)} ({len(found) / len(tickers):.0%})")
    ciks = sorted({int(c) for c in found["cik"]})
    t0 = time.time()
    for i, cik in enumerate(ciks, 1):
        try:
            fetch_company(cik, ua)
        except Exception as e:  # 한 회사 실패로 전체를 멈추지 않는다
            print(f"[SEC] CIK {cik} 실패: {e}")
        if i % 25 == 0 or i == len(ciks):
            print(f"[SEC] {i}/{len(ciks)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
