"""ETF 전체 보유 종목 — SEC N-PORT (2026-10-30, 표시 전용: ETF 성격(경기순환/방어 × 내수/글로벌) · 내 계좌 쏠림).

Yahoo 는 상위 10개만 줘서 SPY · VOO 가 '글로벌 순환'(대형 기술주 10개)으로만 나왔다. 미국 ETF(1940법 펀드)는
분기마다 SEC 에 전체 보유 종목(N-PORT)을 낸다 — 분기 말 기준, 60일 뒤 공개라 2~4달 늦지만 성격처럼 천천히 바뀌는 값엔 충분.

- ETF → series: https://www.sec.gov/files/company_tickers_mf.json (symbol → seriesId)
- series → 최신 NPORT-P: browse-edgar (CIK 자리에 seriesId) → 공시 목록 → primary_doc.xml (서식 입힌 xsl 판은 빼고)
- 보유 종목 → 티커: N-PORT 엔 티커가 없다 (CUSIP · ISIN · 이름). 이름을 다듬어 SEC company_tickers.json 이름과 맞추고,
  안 맞으면 전체분석 종목 이름과 맞춘다. 같은 회사 여러 종류(GOOGL · GOOG)는 성격이 같아서 한 티커면 된다.
- 주식(assetCat EC · EP)만 남긴다. 채권 · 파생 · 현금은 성격이 없어서 뺀다.
- SPY · DIA 같은 UIT, 캐나다 상장, 금 · 가상자산 신탁은 N-PORT 가 없다 — screener.etf_scan 이 같은 지수 쌍둥이(VOO 등)로 메운다.

매달 etf-holdings.yml 이 부른다 (이전 파일과 공시 번호가 같은 ETF 는 다시 받지 않는다):
    PYTHONPATH=.:src python -m data.etf_holdings --prev prev.csv --out etf_holdings.csv
SEC 규칙: User-Agent 에 연락처(SEC_USER_AGENT), 초당 10회 이하 — screener.sec_filings.SecClient 를 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MF_URL = "https://www.sec.gov/files/company_tickers_mf.json"
CO_URL = "https://www.sec.gov/files/company_tickers.json"
SERIES_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={sid}&type=NPORT-P&dateb=&owner=include"
              "&count=3&output=atom")
EQUITY_CATS = {"EC", "EP"}
FIELDS = ["etf", "accession", "rep_date", "ticker", "name", "cusip", "weight", "country"]

# 이름 맞추기 — 회사 형태 · 'The' · 주식 종류 같은 말은 뺀다
_DROP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COS", "COMPANY", "COMPANIES", "LTD", "LIMITED", "PLC",
         "LLC", "LP", "THE", "SA", "NV", "AG", "SE", "CLASS", "CL", "ORD", "SHS", "COMMON", "STOCK", "NEW", "DE", "MD"}
# 이름이 공시와 목록에서 다르게 남은 회사 (지주회사 바꿈 등) — 다듬은 이름 → 티커
ALIASES = {"EXXON MOBIL": "XOM", "BLACKROCK FUNDING": "BLK"}


def norm_name(name: str) -> str:
    """'Coca-Cola Co/The' · 'COCA COLA CO' → 'COCA COLA'. '/DE/' 같은 주 표시 · 문장부호 · 회사 형태는 뺀다."""
    s = re.sub(r"/[A-Z]{2}/?$", " ", name.upper().strip())        # 'APPLE INC /CA/' 'XYZ CORP/DE'
    s = s.replace("&", " AND ").replace("'", "")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return " ".join(w for w in s.split() if w not in _DROP and len(w) > 1)   # 'N V' · 'CLASS A' 의 한 글자도 뺀다


@dataclass
class Holding:
    name: str
    cusip: str
    weight: float      # 0~1 (N-PORT pctVal 은 %)
    country: str


def parse_nport(xml: str) -> tuple[str | None, list[Holding]]:
    """primary_doc.xml → (기준일, 주식 보유 종목). 정규식으로 — 문서가 10MB 넘기도 해서 가볍게."""
    rep = re.search(r"<repPdDate>([^<]+)</repPdDate>", xml)
    out = []
    for block in re.findall(r"<invstOrSec>(.*?)</invstOrSec>", xml, re.S):
        cat = re.search(r"<assetCat>([^<]+)</assetCat>", block)
        pct = re.search(r"<pctVal>([^<]+)</pctVal>", block)
        if not cat or cat.group(1).strip() not in EQUITY_CATS or not pct:
            continue
        try:
            w = float(pct.group(1)) / 100
        except ValueError:
            continue
        if w <= 0:
            continue
        name = re.search(r"<name>([^<]*)</name>", block)
        cusip = re.search(r"<cusip>([^<]*)</cusip>", block)
        country = re.search(r"<invCountry>([^<]*)</invCountry>", block)
        out.append(Holding(name=_unescape(name.group(1)) if name else "", cusip=cusip.group(1).strip() if cusip else "",
                           weight=w, country=country.group(1).strip() if country else ""))
    return (rep.group(1).strip() if rep else None), out


def _unescape(s: str) -> str:
    return s.replace("&amp;", "&").replace("&apos;", "'").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")


class NameMatcher:
    """다듬은 이름 → 티커. SEC 회사 목록이 먼저, 그다음 전체분석 종목 이름. known 에 있는 티커를 고른다."""

    def __init__(self, sec_rows: list[dict[str, Any]], universe_names: dict[str, str], known: set[str]) -> None:
        self.by_name: dict[str, str] = {}
        by_cik: dict[int, list[str]] = {}
        for r in sec_rows:
            by_cik.setdefault(int(r["cik_str"]), []).append(str(r["ticker"]).upper().replace(".", "-"))
        for r in sec_rows:
            key = norm_name(str(r["title"]))
            if not key or key in self.by_name:
                continue
            tickers = by_cik[int(r["cik_str"])]
            pick = next((t for t in tickers if t in known), None)
            if pick:
                self.by_name[key] = pick
        for t, name in universe_names.items():
            key = norm_name(name)
            if key and key not in self.by_name and t in known:
                self.by_name[key] = t

    def ticker(self, name: str) -> str | None:
        key = norm_name(name)
        if not key:
            return None
        if key in ALIASES:
            return ALIASES[key]
        if key in self.by_name:
            return self.by_name[key]
        nospace = key.replace(" ", "")   # 'META PLATFORMS' ↔ 'METAPLATFORMS' 같은 띄어쓰기 차이
        for k, t in self._nospace().items():
            if k == nospace:
                return t
        return None

    def _nospace(self) -> dict[str, str]:
        if not hasattr(self, "_ns"):
            self._ns = {k.replace(" ", ""): t for k, t in self.by_name.items()}
        return self._ns


def latest_nport(client: Any, series_id: str) -> tuple[str, str] | None:
    """series → (공시 번호, primary_doc.xml 주소). 없으면 None."""
    atom = client.get_bytes(SERIES_URL.format(sid=series_id)).decode("utf-8", "replace")
    href = re.search(r"<filing-href>([^<]+)</filing-href>", atom)
    if not href:
        return None
    idx = client.get_bytes(href.group(1)).decode("utf-8", "replace")
    xml = next((u for u in re.findall(r'href="(/Archives/[^"]+primary_doc\.xml)"', idx) if "xsl" not in u), None)
    acc = re.search(r"/(\d{10}-\d{2}-\d{6})-index", href.group(1))
    if not xml or not acc:
        return None
    return acc.group(1), "https://www.sec.gov" + xml


def load_universe(base: str) -> tuple[dict[str, str], set[str]]:
    """전체분석 종목 → (티커 → 회사 이름, 성격을 매길 수 있는 티커(업종 있음))."""
    names: dict[str, str] = {}
    known: set[str] = set()
    for key in ("sp500", "nasdaq"):
        with urllib.request.urlopen(f"{base}/{key}.json.gz", timeout=60) as r:
            for t, v in json.loads(gzip.decompress(r.read())).get("items", {}).items():
                t = t.upper().replace(".", "-")
                if v.get("company"):
                    names.setdefault(t, v["company"])
                if v.get("industry"):
                    known.add(t)
    return names, known


def read_prev(path: Path | None) -> dict[str, list[dict[str, str]]]:
    if not path or not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["etf"], []).append(row)
    return out


def build(etfs: list[str], client: Any, matcher: NameMatcher, prev: dict[str, list[dict[str, str]]],
          budget_s: float = 3600) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """ETF 마다 최신 N-PORT → 행. 공시 번호가 이전과 같으면 이전 행을 그대로 (티커만 다시 맞춘다). → (행, ETF별 상태)."""
    mf = client.get_json(MF_URL) or {}
    series = {str(r[3]).upper(): str(r[1]) for r in mf.get("data", []) if r[1]}
    rows: list[dict[str, Any]] = []
    status: dict[str, str] = {}
    t0 = time.time()
    for etf in etfs:
        sid = series.get(etf)
        if not sid:
            status[etf] = "no_series"
            continue
        if time.time() - t0 > budget_s:
            rows += prev.get(etf, [])
            status[etf] = "budget"
            continue
        try:
            got = latest_nport(client, sid)
        except Exception as exc:  # noqa: BLE001 — 한 ETF 실패는 이전 행으로
            print(f"[ETF 보유] {etf}: 목록 실패 — {exc}")
            got = None
        if not got:
            rows += prev.get(etf, [])
            status[etf] = "no_filing"
            continue
        acc, url = got
        old = prev.get(etf, [])
        if old and old[0].get("accession") == acc:
            for r in old:
                r["ticker"] = matcher.ticker(r["name"]) or ""
            rows += old
            status[etf] = "same"
            continue
        try:
            rep, held = parse_nport(client.get_bytes(url).decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            print(f"[ETF 보유] {etf}: 문서 실패 — {exc}")
            rows += old
            status[etf] = "error"
            continue
        for h in held:
            rows.append({"etf": etf, "accession": acc, "rep_date": rep or "", "ticker": matcher.ticker(h.name) or "",
                         "name": h.name, "cusip": h.cusip, "weight": f"{h.weight:.6f}", "country": h.country})
        status[etf] = "new"
    return rows, status


def coverage(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """ETF → (주식 비중 합, 티커를 찾은 비중 합)."""
    out: dict[str, list[float]] = {}
    for r in rows:
        c = out.setdefault(r["etf"], [0.0, 0.0])
        c[0] += float(r["weight"])
        if r["ticker"]:
            c[1] += float(r["weight"])
    return {k: (v[0], v[1]) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    from data.etf_universe import etf_tickers
    from data.geo_revenue import UNIVERSE_BASE
    from screener.sec_filings import SecClient

    ap = argparse.ArgumentParser(description="ETF 전체 보유 종목 (SEC N-PORT) → etf_holdings.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prev", type=Path)
    ap.add_argument("--only", nargs="*", help="이 ETF 만 (확인용)")
    ap.add_argument("--budget-min", type=float, default=60)
    args = ap.parse_args(argv)

    client = SecClient()
    if not client.ua:
        print("[ETF 보유] SEC_USER_AGENT 가 없어 건너뜀")
        return 0
    names, known = load_universe(UNIVERSE_BASE)
    sec_rows = list((client.get_json(CO_URL) or {}).values())
    matcher = NameMatcher(sec_rows, names, known)
    etfs = [t.upper() for t in (args.only or etf_tickers())]
    rows, status = build(etfs, client, matcher, read_prev(args.prev), budget_s=args.budget_min * 60)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    cov = coverage(rows)
    print(f"[ETF 보유] {len(rows)}행 · ETF {len(cov)}개 · 상태 {counts}")
    for etf, (eq, hit) in sorted(cov.items()):
        print(f"  {etf}: 주식 {eq:.0%} · 티커 찾음 {hit / eq if eq else 0:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
