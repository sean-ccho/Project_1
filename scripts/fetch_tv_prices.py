#!/usr/bin/env python3
"""TradingView 기본 차트와 같은 가격 (분할만 반영, 배당 미반영) — 상장 첫날부터 2025-09-30 까지.

TradingView 기본 차트(ADJ 꺼짐)의 이동평균은 yfinance auto_adjust=False 의 Close(분할만 반영)로 계산한 값과
1센트까지 같다 (2026-10-16 HL·T·UNH 일/주/월 범례 대조). EMA 는 첫 봉부터 이어지므로 1985-01 부터 받는다
(연구에 쓰는 가장 긴 선 = 일봉 200 · 주봉 40 · 월봉 20 → 2006 년이면 시작점 영향이 사라진다. 월봉 EMA200 처럼
더 긴 선을 TV 와 맞출 때는 상장 첫날부터 받아야 한다 — tv_legend_check.py 는 따로 period="max").

- 종목: 2006-01 ~ 2025-09 사이 한 번이라도 S&P 500 이었던 종목 (data/universe/sp500_membership.csv) + SPY
  (2016~ 본 구간 + 2006~2015 확인 구간). 상장폐지 등으로 야후에 가격이 없는 종목은 빠진다 → 커버리지 표.
- 컬럼: Open High Low Close(분할만 반영) · Adj Close(배당 재투자) · Volume. 총수익 시가 = Open × Adj Close / Close.
- 홀드아웃(2025-10-01~)은 받지 않는다 (end 미포함).
- float32 로 저장한다 (계산할 때 float64 로 바꿔 쓴다).
- 저장 후 SHA256 을 data/research/SHA256SUMS.txt 에 덧붙여 고정한다. 이후 연구는 이 파일만 읽는다.

사용법: PYTHONPATH=.:src .venv/bin/python scripts/fetch_tv_prices.py [--check]
산출 (git 제외): data/research/tv_ohlcv_max.parquet, docs/quant_improvement/tier3/tv_prices_coverage.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.universe import Membership, load_membership  # noqa: E402

RESEARCH_DIR = ROOT / "data" / "research"
OUT = RESEARCH_DIR / "tv_ohlcv_max.parquet"
SUMS = RESEARCH_DIR / "SHA256SUMS.txt"
COVERAGE = ROOT / "docs" / "quant_improvement" / "tier3" / "tv_prices_coverage.csv"
START, END = "1985-01-01", "2025-10-01"  # END 미포함 → 2025-09-30 까지
UNIVERSE_START, UNIVERSE_END = "2006-01-01", "2025-09-30"
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
BATCH = 100


def sha256(path: Path) -> str:
    """파일 SHA256 (16진)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def universe() -> list[str]:
    """2006-01 ~ 2025-09 S&P 500 구성 종목 합집합 + SPY."""
    mem = Membership(load_membership(ROOT / "data" / "universe" / "sp500_membership.csv"))
    return sorted(mem.union_between(UNIVERSE_START, UNIVERSE_END) | {"SPY"})


def download(tickers: list[str]) -> pd.DataFrame:
    """배치로 받아 (Price, Ticker) 2단 컬럼 하나로 합친다. 빈 종목은 빠진다."""
    parts = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        d = yf.download(batch, start=START, end=END, interval="1d", auto_adjust=False,
                        group_by="column", progress=False, threads=True)
        d = d.loc[:, d.columns.get_level_values(0).isin(FIELDS)].astype("float32")  # 메모리 절반 (가격 7자리면 충분)
        parts.append(d)
        print(f"[tv] {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)
    out = pd.concat(parts, axis=1).sort_index()
    out.columns = out.columns.set_names(["Price", "Ticker"])
    close = out["Close"]
    keep = close.columns[close.notna().any()]
    out = out.loc[:, out.columns.get_level_values("Ticker").isin(keep)]
    return out.loc[:"2025-09-30"]


def coverage(tickers: list[str], px: pd.DataFrame) -> pd.DataFrame:
    """종목별 첫·마지막 봉과 봉 수 (없으면 빈 칸)."""
    close = px["Close"]
    rows = []
    for t in tickers:
        s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        rows.append({"티커": t, "있음": not s.empty,
                     "첫봉": s.index.min().date() if not s.empty else None,
                     "마지막봉": s.index.max().date() if not s.empty else None, "봉수": int(s.size)})
    return pd.DataFrame(rows)


def check() -> None:
    """고정된 해시와 파일이 같은지만 본다."""
    want = {line.split()[1]: line.split()[0] for line in SUMS.read_text().splitlines() if line.strip()}
    key = str(OUT.relative_to(ROOT))
    if key not in want:
        raise SystemExit(f"{key}: SHA256SUMS 에 없다 — 먼저 받기")
    if sha256(OUT) != want[key]:
        raise SystemExit(f"{key}: 해시가 다르다 — 파일이 바뀌었다")
    print(f"{key}: 해시 일치")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="해시만 확인")
    args = ap.parse_args()
    if args.check:
        check()
        return
    if OUT.exists():
        raise SystemExit(f"{OUT} 가 이미 있다 — 다시 받으려면 직접 지우고 SHA256SUMS 줄도 지울 것")
    tickers = universe()
    print(f"[tv] 종목 {len(tickers)}개, {START} ~ 2025-09-30, auto_adjust=False")
    px = download(tickers)
    px.to_parquet(OUT)
    cov = coverage(tickers, px)
    cov.to_csv(COVERAGE, index=False, encoding="utf-8-sig")
    with SUMS.open("a") as fh:
        fh.write(f"{sha256(OUT)}  {OUT.relative_to(ROOT)}\n")
    print(f"[tv] 저장 {OUT.relative_to(ROOT)} · 가격 있는 종목 {int(cov['있음'].sum())}/{len(cov)} · "
          f"{px.index.min().date()} ~ {px.index.max().date()}")


if __name__ == "__main__":
    main()
