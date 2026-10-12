#!/usr/bin/env python3
"""SEC 내부자 거래 데이터(Form 3/4/5 Insider Transactions Data Sets) → 연구용 장내 매수 기록표.

출처: https://www.sec.gov/dera/data/form-345 — 분기별 zip ({year}q{q}_form345.zip), 탭 구분 UTF-8.
표 (readme: https://www.sec.gov/files/insider_transactions_readme.pdf):
  SUBMISSION      ACCESSION_NUMBER(키) · FILING_DATE(DD-MON-YYYY) · DOCUMENT_TYPE(3, 3/A, 4, 4/A, 5, 5/A)
                  · ISSUERCIK · ISSUERTRADINGSYMBOL
  REPORTINGOWNER  ACCESSION_NUMBER + RPTOWNERCIK(키) · RPTOWNER_RELATIONSHIP(OFFICER, DIRECTOR, TENPERCENTOWNER, OTHER)
  NONDERIV_TRANS  ACCESSION_NUMBER + NONDERIV_TRANS_SK(키) · TRANS_DATE · TRANS_CODE · TRANS_SHARES
                  · TRANS_PRICEPERSHARE · TRANS_ACQUIRED_DISP_CD
남기는 것: Form 4 원본(4/A 제외)의 비파생 매수 거래(코드 P, 취득 A). 제출일 ≤ 2025-09-30 (개발 구간).
보고자 관계(임원·이사·10% 대주주)는 열로 남기고, 거르기는 idea_batch_research.py 에서 한다.
티커: 발행사 CIK → 연구 패널 티커(sec/ticker_cik.csv, fetch_sec_data.py 산출)를 먼저 쓰고, 없으면 신고서의 거래 기호.

SEC 규칙: User-Agent에 이름·이메일, 초당 10회 이하 (fetch_sec_data._get 재사용). 이메일은 코드에 쓰지 않고 환경변수로 받는다.

사용법:
  SEC_USER_AGENT="이름 you@example.com" PYTHONPATH=.:src python scripts/fetch_insider_data.py
산출 (git 제외):
  data/research/sec/insider/{year}q{q}_form345.zip (원본)
  data/research/insider_trades.parquet
"""

from __future__ import annotations

import csv
import io
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from fetch_sec_data import _get
from research_utils import DEV_END, RESEARCH_DIR

INSIDER_DIR = RESEARCH_DIR / "sec" / "insider"
OUT = RESEARCH_DIR / "insider_trades.parquet"
TICKER_CIK = RESEARCH_DIR / "sec" / "ticker_cik.csv"
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
FIRST_Q, LAST_Q = "2015q1", "2025q3"

SUB_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]
OWNER_COLS = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"]
TRANS_COLS = ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
              "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"]


def quarters(first: str = FIRST_Q, last: str = LAST_Q) -> list[str]:
    """'2015q1' ~ '2025q3' 분기 이름 목록."""
    y, q = int(first[:4]), int(first[-1])
    ly, lq = int(last[:4]), int(last[-1])
    out = []
    while (y, q) <= (ly, lq):
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def download(q: str, ua: str) -> Path:
    """분기 zip 을 받아 둔다 (이미 있으면 건너뜀)."""
    path = INSIDER_DIR / f"{q}_form345.zip"
    if path.exists() and path.stat().st_size > 0:
        return path
    raw = _get(URL.format(q=q), ua)
    if raw is None:
        raise FileNotFoundError(f"SEC 에 {q} 파일 없음: {URL.format(q=q)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return path


def _read_table(zf: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    """zip 안의 탭 구분 표 하나 (파일 이름 대소문자·확장자 무관)."""
    member = next((m for m in zf.namelist() if Path(m).stem.upper() == name), None)
    if member is None:
        raise KeyError(f"zip 안에 {name} 없음: {zf.namelist()}")
    with zf.open(member) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), sep="\t", dtype=str,
                           usecols=lambda c: c in cols, quoting=csv.QUOTE_NONE, keep_default_na=False)


def owner_flags(owners: pd.DataFrame) -> pd.DataFrame:
    """신고서(접수번호)별 보고자 관계 플래그. 공동 신고면 한 명이라도 해당하면 True."""
    rel = owners["RPTOWNER_RELATIONSHIP"].fillna("").str.upper().str.replace(r"[\s_]", "", regex=True)
    flags = pd.DataFrame({
        "ACCESSION_NUMBER": owners["ACCESSION_NUMBER"],
        "is_director": rel.str.contains("DIRECTOR"),
        "is_officer": rel.str.contains("OFFICER"),
        "is_ten_pct": rel.str.contains("TENPERCENT"),
    })
    g = flags.groupby("ACCESSION_NUMBER")
    out = g[["is_director", "is_officer", "is_ten_pct"]].any()
    out["n_owners"] = g.size()
    return out.reset_index()


def parse_tables(sub: pd.DataFrame, owners: pd.DataFrame, trans: pd.DataFrame) -> pd.DataFrame:
    """세 표 → Form 4 원본의 장내 매수(P, 취득 A) 거래 행."""
    sub = sub[sub["DOCUMENT_TYPE"].str.strip() == "4"]
    trans = trans[(trans["TRANS_CODE"].str.strip().str.upper() == "P")
                  & (trans["TRANS_ACQUIRED_DISP_CD"].str.strip().str.upper() == "A")]
    df = trans.merge(sub, on="ACCESSION_NUMBER", how="inner").merge(owner_flags(owners), on="ACCESSION_NUMBER",
                                                                    how="left")
    out = pd.DataFrame({
        "accession": df["ACCESSION_NUMBER"],
        "filing_date": pd.to_datetime(df["FILING_DATE"].str.strip(), format="%d-%b-%Y", errors="coerce"),
        "trans_date": pd.to_datetime(df["TRANS_DATE"].str.strip(), format="%d-%b-%Y", errors="coerce"),
        "issuer_cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce"),
        "symbol": df["ISSUERTRADINGSYMBOL"].fillna("").str.strip(),
        "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce"),
        "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
        "is_director": df["is_director"].fillna(False).astype(bool),
        "is_officer": df["is_officer"].fillna(False).astype(bool),
        "is_ten_pct": df["is_ten_pct"].fillna(False).astype(bool),
        "n_owners": df["n_owners"].fillna(0).astype(int),
    })
    out["value"] = out["shares"] * out["price"]
    return out.dropna(subset=["filing_date", "issuer_cik"]).reset_index(drop=True)


def read_quarter(path: Path) -> pd.DataFrame:
    """분기 zip → 장내 매수 거래 행."""
    with zipfile.ZipFile(path) as zf:
        sub = _read_table(zf, "SUBMISSION", SUB_COLS)
        owners = _read_table(zf, "REPORTINGOWNER", OWNER_COLS)
        trans = _read_table(zf, "NONDERIV_TRANS", TRANS_COLS)
    return parse_tables(sub, owners, trans)


def normalize_symbol(s: str) -> str:
    """신고서 거래 기호 → yfinance 표기 (BRK.B → BRK-B). 비거나 이상하면 ''."""
    s = s.strip().upper().replace(".", "-").replace(" ", "")
    return s if s and s.replace("-", "").isalnum() else ""


def map_tickers(df: pd.DataFrame, cik_map: pd.DataFrame) -> pd.DataFrame:
    """발행사 CIK → 패널 티커 (같은 CIK에 여러 티커면 신고서 기호와 같은 것, 없으면 첫 번째).

    CIK 가 패널에 없으면 신고서 기호를 쓴다. 단 그 기호가 CIK 를 아는 다른 패널 종목의 티커면
    (다른 회사 신고서의 잘못 적힌 기호) 버린다.
    """
    by_cik: dict[int, list[str]] = {}
    cik_of: dict[str, int] = {}
    for t, c in zip(cik_map["티커"], cik_map["cik"]):
        if pd.notna(c):
            by_cik.setdefault(int(c), []).append(str(t))
            cik_of[str(t)] = int(c)
    sym = df["symbol"].map(normalize_symbol)
    tickers, source = [], []
    for cik, s in zip(df["issuer_cik"].astype(int), sym):
        cands = by_cik.get(cik)
        if cands:
            tickers.append(s if s in cands else cands[0])
            source.append("cik")
        elif s and s in cik_of:
            tickers.append("")
            source.append("conflict")
        else:
            tickers.append(s)
            source.append("symbol" if s else "none")
    out = df.copy()
    out["티커"] = tickers
    out["ticker_source"] = source
    return out[out["티커"] != ""].reset_index(drop=True)


def main() -> None:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        sys.exit("환경변수 SEC_USER_AGENT='이름 이메일' 이 필요합니다 (SEC 규칙)")
    qs = quarters()
    frames = []
    for i, q in enumerate(qs, 1):
        path = download(q, ua)
        part = read_quarter(path)
        frames.append(part)
        print(f"[insider] {q}: 매수 거래 {len(part):,}행 ({path.stat().st_size / 1e6:.0f}MB) [{i / len(qs):.1%}]", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["filing_date"] <= DEV_END]
    df = map_tickers(df, pd.read_csv(TICKER_CIK))
    df = df.drop_duplicates(subset=["accession", "trans_date", "shares", "price", "티커"]).reset_index(drop=True)
    df.to_parquet(OUT)

    od = df[df["is_officer"] | df["is_director"]]
    print(f"\n저장: {OUT} — {len(df):,}행 · 임원/이사 {len(od):,}행 · 티커 {df['티커'].nunique():,}개")
    print("티커 출처:", df["ticker_source"].value_counts().to_dict())
    print("\n연도별 임원/이사 매수 신고 (제출일·종목 단위):")
    days = od.drop_duplicates(["filing_date", "티커"])
    print(days.groupby(days["filing_date"].dt.year).size().to_string())
    jpm = od[(od["티커"] == "JPM") & (od["filing_date"].between("2016-02-01", "2016-02-29"))]
    print("\n확인: JPM 2016-02 임원/이사 매수", "있음" if len(jpm) else "없음 ⚠️")
    if len(jpm):
        print(jpm[["filing_date", "trans_date", "shares", "price", "value"]].to_string(index=False))
    if np.isfinite(df["value"]).any():
        print(f"\n매수 금액 중앙값 ${od['value'].median():,.0f} · 상위 1% ${od['value'].quantile(0.99):,.0f}")


if __name__ == "__main__":
    main()
