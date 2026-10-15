#!/usr/bin/env python3
"""연구소(lab) 데이터 — 장기 지수·펀드·ETF 시세와 Kenneth French 일간 자료를 받아 고정한다.

- 받는 것: yfinance 수정주가(배당 재투자) OHLCV, Kenneth French 일간 3팩터(미국 시장 총수익·1개월 국채 금리)
- 2025-09-30(DEV_END)에서 자르고 data/research/lab/ 에 저장한 뒤 SHA256 으로 고정한다. 이후 연구는 이 파일만 읽는다.
- 자산군은 대용 펀드 → ETF 로 이어 붙인다 (CHAINS). 이어 붙인 날, 겹치는 구간의 상관은 coverage 표에 남긴다.
- 원자재 지수 ^SPGSCI 는 롤 비용이 빠진 현물 지수라(GSG 보다 연 +4.6%p) 대용으로 쓰지 않는다 → 원자재는 DBC(2006~)부터.

사용:
  .venv/bin/python scripts/lab_data.py --build    # 없으면 내려받고, 있으면 해시만 확인
  .venv/bin/python scripts/lab_data.py --check    # 해시 확인 + 자산별 시작일·이어붙이기 표
  .venv/bin/python scripts/lab_data.py --build-industries   # Kenneth French 12 산업 일간 (lab_sector_rotation)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_utils import DEV_END, RESEARCH_DIR  # noqa: E402

LAB_DIR = RESEARCH_DIR / "lab"
PRICES = LAB_DIR / "prices.parquet"
FRENCH = LAB_DIR / "french_daily.parquet"
SUMS = LAB_DIR / "SHA256SUMS.txt"
COVERAGE = LAB_DIR / "coverage.csv"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FRENCH_IND = LAB_DIR / "french_ind12_daily.parquet"
IND12_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip"
IND12 = ("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")

Source = str | tuple[tuple[str, float], ...]   # 티커 하나 또는 (티커, 비중) 묶음(매일 다시 맞춘 혼합)

# 자산군 = [(출처, 이 출처로 바꾸는 날)] — 첫 출처는 생긴 날부터. 바꾸는 날은 ETF 상장 1달 뒤 첫 거래일 무렵 (사전 고정)
CHAINS: dict[str, list[tuple[Source, str | None]]] = {
    "US": [("VFINX", None), ("SPY", "1993-03-01")],
    "DEV": [((("VEURX", 0.6), ("VPACX", 0.4)), None), ("EFA", "2001-10-01")],
    "EM": [("VEIEX", None), ("EEM", "2003-06-02")],
    "LTT": [("VUSTX", None), ("TLT", "2002-09-03")],
    "ITT": [("VFITX", None), ("IEF", "2002-09-03")],
    "REIT": [("VGSIX", None), ("VNQ", "2004-11-01")],
    "GOLD": [("GC=F", None), ("GLD", "2005-01-03")],
    "CMDTY": [("DBC", None)],
}
ASSET_NAMES = {"US": "미국 주식", "DEV": "선진국 주식", "EM": "신흥국 주식", "LTT": "미국 장기국채",
               "ITT": "미국 중기국채", "REIT": "미국 리츠", "GOLD": "금", "CMDTY": "원자재"}
INTL = {"^N225": "일본 닛케이225", "^FTSE": "영국 FTSE100", "^GDAXI": "독일 DAX", "^HSI": "홍콩 항셍",
        "^KS11": "한국 KOSPI", "^GSPTSE": "캐나다 TSX"}
LEV2X = {"SSO": "SPY", "UBT": "TLT", "UGL": "GLD"}   # 2배 ETF → 기초 ETF (레버리지 비용 가정 대조용)


def chain_tickers() -> list[str]:
    """CHAINS 에 나오는 모든 티커."""
    out: list[str] = []
    for chain in CHAINS.values():
        for src, _ in chain:
            out += [src] if isinstance(src, str) else [t for t, _ in src]
    return out


def all_tickers() -> list[str]:
    """내려받을 티커 전부 (중복 없이, 순서 유지)."""
    return list(dict.fromkeys(chain_tickers() + list(INTL) + list(LEV2X) + list(LEV2X.values())))


# ── 내려받기 ──────────────────────────────────────────────────────────────────────────────────

def download_prices(tickers: list[str]) -> pd.DataFrame:
    """yfinance 수정주가 OHLCV (컬럼 MultiIndex: 티커 × 항목). DEV_END 에서 자른다."""
    import yfinance as yf
    raw = yf.download(tickers, period="max", auto_adjust=True, group_by="ticker", progress=False, threads=True)
    frames = {}
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            raise SystemExit(f"{t}: 시세를 받지 못했다")
        f = raw[t][["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        if f["Close"].notna().sum() == 0:
            raise SystemExit(f"{t}: 종가가 비어 있다")
        frames[t] = f
    out = pd.concat(frames, axis=1, sort=True)
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.columns.names = ["ticker", "field"]
    return out.sort_index().loc[:DEV_END]


def parse_french(text: str) -> pd.DataFrame:
    """Kenneth French 일간 3팩터 CSV 텍스트 → mkt(시장 총수익), rf (소수, 일간)."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and re.fullmatch(r"\d{8}", parts[0]):
            mkt_rf, rf = float(parts[1]), float(parts[4])
            rows.append((pd.Timestamp(parts[0]), (mkt_rf + rf) / 100.0, rf / 100.0))
    if not rows:
        raise ValueError("Kenneth French 파일에서 일간 행을 찾지 못했다")
    return pd.DataFrame(rows, columns=["date", "mkt", "rf"]).set_index("date").sort_index()


def parse_french_industries(text: str) -> pd.DataFrame:
    """Kenneth French 12 산업 일간 CSV → 시가총액 가중 수익률만 (소수, 일간). -99.99·-999 는 NaN."""
    rows, started = [], False
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 13 and re.fullmatch(r"\d{8}", parts[0]):
            started = True
            vals = [float(x) for x in parts[1:13]]
            rows.append([pd.Timestamp(parts[0])] + [np.nan if v <= -99.99 else v / 100.0 for v in vals])
        elif started:
            break                                   # 첫 표(가치 가중)가 끝나면 멈춘다 — 다음 표는 동일 가중
    if not rows:
        raise ValueError("12 산업 파일에서 일간 행을 찾지 못했다")
    return pd.DataFrame(rows, columns=["date", *IND12]).set_index("date").sort_index()


def download_french_industries() -> pd.DataFrame:
    """Kenneth French 12 산업 일간 (메모리에서 바로 읽는다)."""
    req = urllib.request.Request(IND12_URL, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — 고정 URL
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        text = z.read(name).decode("latin-1")
    return parse_french_industries(text).loc[:DEV_END]


def build_industries() -> None:
    """12 산업 파일이 없으면 받아서 고정 (SHA256SUMS 에 한 줄 덧붙임), 있으면 해시만 확인."""
    sums = _sums()
    if FRENCH_IND.name in sums:
        verify(FRENCH_IND)
        print(f"[lab] {FRENCH_IND.name} 해시 확인 통과", flush=True)
        return
    if FRENCH_IND.exists():
        raise SystemExit(f"{FRENCH_IND.name} 은 있는데 해시가 없다 — 지우지 말고 원인부터 확인")
    ind = download_french_industries()
    ind.to_parquet(FRENCH_IND)
    with SUMS.open("a") as fh:
        fh.write(f"{sha256(FRENCH_IND)}  {FRENCH_IND.name}\n")
    print(f"[lab] {FRENCH_IND.name}: {ind.index.min().date()} ~ {ind.index.max().date()} · 빈 값 {int(ind.isna().sum().sum())}",
          flush=True)


def load_french_industries() -> pd.DataFrame:
    verify(FRENCH_IND)
    return pd.read_parquet(FRENCH_IND)


def download_french() -> pd.DataFrame:
    """Kenneth French 데이터 라이브러리 일간 3팩터 (메모리에서 바로 읽는다, 디스크에 풀지 않음)."""
    req = urllib.request.Request(FRENCH_URL, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — 고정 URL
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        text = z.read(name).decode("latin-1")
    return parse_french(text).loc[:DEV_END]


# ── 고정 · 읽기 ─────────────────────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sums() -> dict[str, str]:
    if not SUMS.exists():
        return {}
    out = {}
    for line in SUMS.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def verify(path: Path) -> None:
    """고정한 해시와 다르면 멈춘다 (연구 도중 데이터가 바뀌지 않게)."""
    want = _sums().get(path.name)
    if want is None:
        raise SystemExit(f"{path.name}: SHA256SUMS 에 없다 — lab_data.py --build 먼저")
    if sha256(path) != want:
        raise SystemExit(f"{path.name}: 해시가 고정 값과 다르다 — 데이터가 바뀌었다")


def load_prices() -> pd.DataFrame:
    verify(PRICES)
    return pd.read_parquet(PRICES)


def load_french() -> pd.DataFrame:
    verify(FRENCH)
    return pd.read_parquet(FRENCH)


# ── 수익률 ────────────────────────────────────────────────────────────────────────────────────

def us_calendar(french: pd.DataFrame) -> pd.DatetimeIndex:
    """미국 거래일 = Kenneth French 일간 날짜 (CRSP, 1926~)."""
    return pd.DatetimeIndex(french.index[french.index <= DEV_END])


def close_returns(prices: pd.DataFrame, ticker: str, calendar: pd.DatetimeIndex | None = None) -> pd.Series:
    """수정 종가 → 종가→종가 수익률. calendar 를 주면 그 날짜로 맞추고, 빠진 날은 0 (다음 날에 몰아서 반영됨)."""
    c = prices[ticker]["Close"].dropna()
    c = c[c > 0]
    r = c.pct_change(fill_method=None).iloc[1:]
    if calendar is None:
        return r
    first = r.index.min()
    out = r.reindex(calendar)
    out[(out.index >= first) & out.isna()] = 0.0
    return out


def open_returns(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """수정 시가 → 시가→다음 시가 수익률 (t 시가 → t+1 시가, t 에 기록 — idea_batch 와 같은 규칙)."""
    o = prices[ticker]["Open"].dropna()
    o = o[o > 0]
    return (o.shift(-1) / o - 1.0).iloc[:-1]


def _source_returns(prices: pd.DataFrame, src: Source, calendar: pd.DatetimeIndex) -> pd.Series:
    if isinstance(src, str):
        return close_returns(prices, src, calendar)
    parts = [close_returns(prices, t, calendar) * w for t, w in src]
    return pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))


def chain_returns(prices: pd.DataFrame, asset: str, calendar: pd.DatetimeIndex) -> pd.Series:
    """자산군 수익률: 출처를 바꾸는 날 전까지는 앞 출처, 그날부터 다음 출처."""
    chain = CHAINS[asset]
    out = pd.Series(np.nan, index=calendar, name=asset)
    for k, (src, _) in enumerate(chain):
        lo = pd.Timestamp(chain[k][1]) if chain[k][1] else calendar.min()
        hi = pd.Timestamp(chain[k + 1][1]) if k + 1 < len(chain) else None
        r = _source_returns(prices, src, calendar)
        mask = (calendar >= lo) & ((calendar < hi) if hi is not None else True)
        out[mask] = r[mask]
    return out


def asset_returns(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """모든 자산군의 일간 총수익률 (생기기 전은 NaN)."""
    return pd.DataFrame({a: chain_returns(prices, a, calendar) for a in CHAINS})


@dataclass
class SpliceStat:
    asset: str
    proxy: str
    etf: str
    switch: str
    n: int
    corr: float
    ann_diff: float


def splice_stats(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> list[SpliceStat]:
    """이어 붙인 자리마다 대용 ↔ ETF 의 겹치는 구간 (ETF 상장 ~ 2025-09) 일간 상관·연 수익 차이."""
    out = []
    for asset, chain in CHAINS.items():
        for k in range(1, len(chain)):
            proxy_src, etf = chain[k - 1][0], chain[k][0]
            a = _source_returns(prices, proxy_src, calendar)
            b = close_returns(prices, str(etf), calendar)
            x = pd.concat([a, b], axis=1).dropna()
            ann = lambda s: float((1 + s).prod() ** (252 / len(s)) - 1)  # noqa: E731
            name = proxy_src if isinstance(proxy_src, str) else "+".join(f"{w:.0%} {t}" for t, w in proxy_src)
            out.append(SpliceStat(asset, name, str(etf), str(chain[k][1]), len(x),
                                  float(x.iloc[:, 0].corr(x.iloc[:, 1])), ann(x.iloc[:, 0]) - ann(x.iloc[:, 1])))
    return out


def coverage_table(prices: pd.DataFrame, french: pd.DataFrame) -> pd.DataFrame:
    """자산군별 시작일·출처·이어 붙인 날·겹치는 구간 상관."""
    cal = us_calendar(french)
    rets = asset_returns(prices, cal)
    stats = {s.asset: s for s in splice_stats(prices, cal)}
    rows = []
    for a in CHAINS:
        r = rets[a].dropna()
        srcs = " → ".join((s if isinstance(s, str) else "+".join(f"{w:.0%} {t}" for t, w in s))
                          + (f" ({d})" if d else "") for s, d in CHAINS[a])
        st = stats.get(a)
        rows.append({"자산": a, "이름": ASSET_NAMES[a], "시작": r.index.min().date(), "끝": r.index.max().date(),
                     "출처": srcs, "겹친 구간 상관": round(st.corr, 3) if st else None,
                     "겹친 구간 연수익 차(대용−ETF)": round(st.ann_diff, 4) if st else None,
                     "빈 날(0 처리)": int((rets[a].loc[r.index.min():] == 0).sum())})
    for t, name in INTL.items():
        c = prices[t]["Close"].dropna()
        rows.append({"자산": t, "이름": name, "시작": c.index.min().date(), "끝": c.index.max().date(),
                     "출처": "yfinance 가격 지수(배당 제외, DAX 는 총수익)"})
    rows.append({"자산": "RF·MKT", "이름": "1개월 국채 금리 · 미국 시장 총수익", "시작": french.index.min().date(),
                 "끝": french.index.max().date(), "출처": "Kenneth French 일간 3팩터"})
    return pd.DataFrame(rows)


# ── 실행 ──────────────────────────────────────────────────────────────────────────────────────

def build(refresh: bool = False) -> None:
    """없으면 내려받아 고정, 있으면 해시만 확인한다."""
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    sums = _sums()
    if refresh or not PRICES.exists() or not FRENCH.exists() or not sums:
        if sums and not refresh:
            raise SystemExit("SHA256SUMS 는 있는데 파일이 없다 — 지우지 말고 원인부터 확인")
        print("===== 시세 내려받기 (yfinance)", flush=True)
        prices = download_prices(all_tickers())
        prices.to_parquet(PRICES)
        print(f"[lab] {PRICES.relative_to(RESEARCH_DIR.parent.parent)}: {prices.shape}", flush=True)
        print("===== Kenneth French 일간 3팩터", flush=True)
        french = download_french()
        french.to_parquet(FRENCH)
        print(f"[lab] {FRENCH.name}: {french.index.min().date()} ~ {french.index.max().date()}", flush=True)
        SUMS.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in (PRICES, FRENCH)))
    check()


def check() -> None:
    prices, french = load_prices(), load_french()
    cov = coverage_table(prices, french)
    cov.to_csv(COVERAGE, index=False)
    print("===== 자산별 시작일 · 이어붙이기", flush=True)
    print(cov.to_string(index=False), flush=True)
    cal = us_calendar(french)
    spy = prices["SPY"]["Close"].dropna().index
    miss = spy.difference(cal)
    extra = cal[(cal >= spy.min())].difference(spy)
    print(f"[lab] 달력 대조: SPY 에만 있는 날 {len(miss)}, Kenneth French 에만 있는 날 {len(extra)}", flush=True)
    print("[lab] 해시 확인 통과", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--build-industries", action="store_true", help="Kenneth French 12 산업 일간을 받아 고정")
    ap.add_argument("--refresh", action="store_true", help="고정한 파일을 다시 받는다 (연구 결과가 바뀔 수 있음)")
    a = ap.parse_args()
    if a.build:
        build(a.refresh)
    elif a.build_industries:
        build_industries()
    else:
        check()


if __name__ == "__main__":
    main()
