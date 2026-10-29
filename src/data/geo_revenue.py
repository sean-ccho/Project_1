"""회사별 미국 매출 비중 — SEC 10-K XBRL 의 지역별 매출 (2026-10-29, 표시 전용: 경기순환/방어 × 내수/글로벌 구분).

매달 geo-revenue.yml 이 S&P 500 + 나스닥/NYSE 전체분석 종목을 받아 고아 브랜치 geo-data 에 geo_revenue.csv 로 올린다.
밤 실행(run-screener · universe-scan)은 그 파일을 GEO_REVENUE_FILE 로 받고 data.industry_style 이 읽는다 (없으면 업종 기준 추정).
점수 · PT-1 에는 안 쓴다.

한 회사:
  1. submissions JSON → 가장 최근 10-K (10-K 가 없고 20-F · 40-F 를 내면 해외 기업)
  2. 그 공시 폴더의 XBRL 인스턴스 (*_htm.xml, 옛 공시는 *-YYYYMMDD.xml)
  3. 회계연도(약 1년, 끝 = 보고 기준일) 매출 개념마다: 차원 없는 합계 + 축 하나짜리 값
     - 지역 축(StatementGeographicalAxis)의 미국 멤버(country:US · UnitedStates · Domestic …) → 미국 ÷ 합계
     - 미국 멤버가 없고 해외 합계 멤버(NonUs · Foreign · International …)만 있으면 1 − 해외 ÷ 합계 (Citi 처럼)
     - 지역 축에 없으면 사업부 축(StatementBusinessSegmentsAxis)에서 이름이 미국 · 해외인 사업부 (MCD 'U.S.' 처럼)
     - 'North America' · 'Americas' 는 캐나다 · 중남미가 섞여 쓰지 않는다
     - 값이 없으면 지역 집중 퍼센트(ConcentrationRiskPercentage1 — TSN 95% · MCHP 해외 75%), 단 지역 값과 맞을 때만
  4. 0 ≤ 비중 ≤ 1.02 만 받는다 (1 을 조금 넘는 건 반올림 · 내부거래 → 1)
비중을 못 구한 회사: 지역(북미 · 유럽 …)별로만 낸 회사는 'geo_regions' (해외 사업이 크다 — CL · PM · LVS),
지역별 매출을 XBRL 로 아예 안 낸 회사는 'none' (해외 매출이 작아 글로만 쓰는 경우가 대부분 — INTU · UNH · 은행 · 통신),
매출 숫자 자체가 없으면 'no_revenue'. 해석(내수 · 글로벌)은 data.industry_style 이 한다.

    SEC_USER_AGENT="이름 이메일" PYTHONPATH=.:src python -m data.geo_revenue --out geo_revenue.csv [--prev 이전.csv] [--limit 20]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

UNIVERSE_BASE = "https://raw.githubusercontent.com/sean-ccho/Project_1/universe-data"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"
FILE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{name}"

# 매출 개념 — 앞에 있을수록 먼저 본다 (같은 공시 안에서 합계와 지역 값을 같은 개념으로 맞춘다)
REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense", "RegulatedAndUnregulatedOperatingRevenue",
    "OperatingLeasesIncomeStatementLeaseRevenue", "RealEstateRevenueNet", "InterestAndDividendIncomeOperating",
    "NetSalesOfReportableSegmentsAndAllOther",     # 2024 사업부 공시 기준 (GLW — 사업부 매출을 나라별로)
)
GEO_AXIS = "StatementGeographicalAxis"
SEGMENT_AXIS = "StatementBusinessSegmentsAxis"
# 지역 값에 같이 붙어도 무시하는 축 · 멤버 — '지역 집중' 공시(QCOM · TXN)와 '보고 지역 구성요소'(DHR)는 합계 대비 지역 값이다
IGNORABLE_DIMS = {
    ("ConsolidationItemsAxis", "ReportableGeographicalComponentsMember"),
    ("ConsolidationItemsAxis", "OperatingSegmentsMember"),             # 사업부 매출 (EXPD 처럼 사업부가 나라)
    ("ConcentrationRiskByTypeAxis", "GeographicConcentrationRiskMember"),
}
IGNORABLE_AXES = {"ConcentrationRiskByBenchmarkAxis"}   # 집중의 기준(매출) — 위 '지역 집중'과 같이만 나온다
PCT_CONCEPT = "ConcentrationRiskPercentage1"            # 매출 대비 미국 비율을 퍼센트로만 낸 회사
FOREIGN_FORMS = ("20-F", "40-F")
# 멤버 이름에서 빼고 보는 낱말 (UnitedStatesSegmentMember → united states)
NOISE = {"member", "segment", "segments", "market", "markets", "operations", "operating", "reportable", "region",
         "regions", "geographic", "geographical", "geography", "country", "countries", "area", "areas", "total"}
US_SETS = [{"us"}, {"usa"}, {"u", "s"}, {"united", "states"}, {"domestic"}, {"united", "states", "of", "america"}]
FOREIGN_SETS = [{"non", "us"}, {"nonus"}, {"foreign"}, {"international"}, {"outside", "us"},
                {"outside", "united", "states"}, {"non", "united", "states"}, {"outside", "the", "united", "states"},
                {"all", "foreign"}, {"other", "than", "us"}]
# 확실히 미국 밖인 지역 이름 낱말 (country:XX 는 XX ≠ US 면 해외) — 'North America' · 'Americas' 는 캐나다가 섞여 안 넣는다
FOREIGN_WORDS = {"europe", "european", "eurasia", "asia", "asian", "pacific", "apac", "emea", "latin", "latam", "africa",
                 "middle", "china", "chinese", "japan", "canada", "mexico", "brazil", "india", "korea", "taiwan",
                 "germany", "kingdom", "uk", "france", "australia", "singapore", "macau", "macao", "international",
                 "foreign", "nonus", "rest", "world", "overseas"}
SHARE_MAX = 1.02
DUR_MIN, DUR_MAX = 300, 400            # 회계연도 기간 (52/53주 포함)


@dataclass
class GeoRow:
    ticker: str
    cik: int | None = None
    form: str = ""                     # 10-K · 20-F · 40-F · '' (공시 없음)
    fy_end: str = ""
    accession: str = ""
    concept: str = ""
    member: str = ""
    us_rev: float | None = None
    total_rev: float | None = None
    us_share: float | None = None
    foreign_min: float | None = None   # 확실한 해외 매출 ÷ 합계 (해외 비중의 하한: 나라는 합, 지역은 가장 큰 하나) — 비중을 못 구했을 때
    # geo_us · geo_foreign · segment_us · segment_foreign · geo_pct (비중 있음) / geo_regions (지역별만) · none (지역 공시 없음)
    # · no_revenue · foreign_filer (20-F · 40-F) · no_filing · error
    method: str = ""
    fetched: str = ""


def member_kind(qname: str) -> str | None:
    """멤버 QName → 'us' · 'foreign' · None. country:US 는 미국, NorthAmerica · Americas 는 None."""
    local = qname.split(":")[-1]
    if qname.lower() == "country:us":
        return "us"
    words = [w.lower() for w in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", local)]
    rest = {w for w in words if w not in NOISE}
    if not rest:
        return None
    if rest in US_SETS:
        return "us"
    if rest in FOREIGN_SETS:
        return "foreign"
    return None


def is_foreign_region(qname: str) -> bool:
    """미국 밖으로 확실한 지역 멤버인가 — country:CN · EuropeMember · NonNorthAmerica · AmericasExcludingUnitedStates …"""
    head, _, local = qname.rpartition(":")
    if head == "country":
        return local.upper() != "US"
    kind = member_kind(qname)
    if kind:
        return kind == "foreign"
    words = [w.lower() for w in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", local)]
    if "excluding" in words or "ex" in words:          # 'Americas excluding United States' — 미국을 뺀 것
        cut = words.index("excluding" if "excluding" in words else "ex")
        after = set(words[cut + 1:])
        return bool(after & {"us", "usa"} or {"united", "states"} <= after)
    if {"us", "usa", "domestic"} & set(words) or {"united", "states"} <= set(words):
        return False                                    # 'United States and Canada' — 미국이 섞임
    if words and words[0] in ("non", "outside"):
        return True                                     # NonNorthAmerica
    return bool(FOREIGN_WORDS & set(words))


def _num(text: str | None) -> float | None:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def parse_instance(xml: bytes, period_end: str) -> dict[str, Any]:
    """XBRL 인스턴스 → {method, concept, member, us_rev, total_rev, us_share}. 찾지 못하면 method 'none'.

    합계가 같은 개념에 없으면(GOOGL — 지역 값은 계약 매출, 합계는 Revenues) 다른 매출 개념의 가장 큰 합계,
    그것도 없고 미국 · 해외가 다 있으면 둘의 합. 값이 없고 퍼센트(ConcentrationRiskPercentage1)만 있으면 그것 (geo_pct).
    """
    root = ET.fromstring(xml)
    contexts: dict[str, tuple[tuple[str, str], ...]] = {}   # 회계연도 기간 context id → 차원들 (축, 멤버)
    for ctx in root.iter():
        if not ctx.tag.endswith("}context"):
            continue
        per = ctx.find("{*}period")
        start, end = (per.find("{*}startDate"), per.find("{*}endDate")) if per is not None else (None, None)
        if start is None or end is None or (end.text or "").strip() != period_end:
            continue
        try:
            days = (date.fromisoformat(end.text.strip()) - date.fromisoformat(start.text.strip())).days
        except ValueError:
            continue
        dims = tuple((m.get("dimension", "").split(":")[-1], (m.text or "").strip())
                     for m in ctx.iter() if m.tag.endswith("}explicitMember"))
        has_typed = any(m.tag.endswith("}typedMember") for m in ctx.iter())
        if DUR_MIN <= days <= DUR_MAX and not has_typed:
            contexts[ctx.get("id")] = dims
    totals: dict[str, float] = {}
    pct: float | None = None
    regions: dict[str, None] = {}                                       # 매출에 붙은 지역 멤버 (어떤 조합이든)
    by_axis: dict[tuple[str, str], list[tuple[str, float]]] = {}        # (개념, 축) → [(멤버, 값)]
    for el in root:
        concept = el.tag.split("}")[-1]
        if concept == PCT_CONCEPT and el.get("contextRef") in contexts:
            d = _core_dims(contexts[el.get("contextRef")], pct=True)
            kind = member_kind(d[0][1]) if d and len(d) == 1 and d[0][0] == GEO_AXIS else None
            v = _num(el.text)
            if kind and v is not None and 0 <= v <= 1 and pct is None:
                pct = v if kind == "us" else 1 - v          # 해외 퍼센트(MCHP NonUs 75%) → 미국 = 1 − 해외
            continue
        if concept not in REVENUE_CONCEPTS or el.get("contextRef") not in contexts:
            continue
        v = _num(el.text)
        if v is None:
            continue
        raw = contexts[el.get("contextRef")]
        for axis, m in raw:
            if axis == GEO_AXIS and is_foreign_region(m):
                regions.setdefault(m.split(":")[-1], None)
        dims = _core_dims(raw)
        if dims is None:
            continue
        if not dims:
            totals.setdefault(concept, v)
        elif len(dims) == 1 and dims[0][0] in (GEO_AXIS, SEGMENT_AXIS):
            by_axis.setdefault((concept, dims[0][0]), []).append((dims[0][1], v))
    for axis, tag in ((GEO_AXIS, "geo"), (SEGMENT_AXIS, "segment")):
        for concept in REVENUE_CONCEPTS:
            members = by_axis.get((concept, axis))
            if not members:
                continue
            us = next(((m, v) for m, v in members if member_kind(m) == "us"), None)
            foreign = next(((m, v) for m, v in members if member_kind(m) == "foreign"), None)
            total = totals.get(concept) or max(totals.values(), default=None)
            if total is None and us and foreign:
                total = us[1] + foreign[1]
            if not total or total <= 0:
                continue
            if us:
                share, member, method, us_rev = us[1] / total, us[0], f"{tag}_us", us[1]
            elif foreign:
                share, member, method, us_rev = 1 - foreign[1] / total, foreign[0], f"{tag}_foreign", total - foreign[1]
            else:
                continue
            if 0 <= share <= SHARE_MAX:
                return {"method": method, "concept": concept, "member": member, "us_rev": us_rev,
                        "total_rev": total, "us_share": min(share, 1.0)}
    total = max(totals.values(), default=None)
    foreign_min = None
    if total and total > 0:
        # 축 하나짜리 값에서 확실한 해외의 합. 멤버를 다 더해 합계를 넘으면(Europe ⊃ GB 처럼 겹침) 나라끼리의 합과
        # 가장 큰 지역 하나 중 큰 쪽만 (보수적으로)
        best = 0.0
        for (concept, _axis), members in by_axis.items():
            base = totals.get(concept) or total
            foreign = [(m, v) for m, v in members if is_foreign_region(m) and v > 0]
            if not foreign:
                continue
            if sum(v for _m, v in members if v > 0) <= base * 1.05:
                got = sum(v for _m, v in foreign)
            else:
                got = max(sum(v for m, v in foreign if m.startswith("country:")),
                          max((v for m, v in foreign if not m.startswith("country:")), default=0.0))
            best = max(best, got / base)
        foreign_min = min(best, 1.0) if best > 0 else None
    # 퍼센트는 기준이 애매한 회사가 있다 (CRM 'US 93%' = 미주 안 비중) → 지역 값과 맞을 때만 (미국 ≤ 1 − 확실한 해외 + 5%p)
    if pct is not None and (foreign_min is None or pct <= 1 - foreign_min + 0.05):
        return {"method": "geo_pct", "concept": PCT_CONCEPT, "member": "country:US", "us_rev": None,
                "total_rev": total, "us_share": pct}
    for members in by_axis.values():
        for m, _v in members:
            if is_foreign_region(m):
                regions.setdefault(m.split(":")[-1], None)
    empty = {"concept": "", "us_rev": None, "total_rev": total, "us_share": None, "foreign_min": foreign_min}
    if regions:     # 미국 밖 지역 · 사업부는 있는데 미국만 따로는 없음 (CL · PM · LVS · KLAC)
        return {**empty, "method": "geo_regions", "member": " · ".join(list(regions)[:4])}
    # 지역별 매출 공시가 아예 없음 — 해외 매출이 작아 글로만 쓰는 경우가 대부분 (INTU · UNH · 은행 · 통신)
    return {**empty, "method": "none" if totals else "no_revenue", "member": ""}


def _core_dims(dims: tuple[tuple[str, str], ...], pct: bool = False) -> tuple[tuple[str, str], ...] | None:
    """무시해도 되는 축(지역 집중 · 보고 지역 구성요소)을 뺀 차원. '지역 집중'인데 기준이 매출이 아니면 None.

    pct=True: 퍼센트 공시 — 기준 축이 매출(…Revenue… · SalesRevenueNet…) 이어야 쓴다.
    """
    out = []
    for axis, member in dims:
        if (axis, member.split(":")[-1]) in IGNORABLE_DIMS:
            continue
        if axis in IGNORABLE_AXES:
            if "Revenue" not in member and "Sales" not in member:
                return None
            continue
        out.append((axis, member))
    if pct and len(out) == len(dims):          # 집중 공시가 아닌 퍼센트(세율 등)는 안 쓴다
        return None
    return tuple(out)


def _instance_name(names: list[str]) -> str | None:
    """공시 폴더 파일 목록에서 XBRL 인스턴스."""
    got = next((n for n in names if n.endswith("_htm.xml")), None)
    return got or next((n for n in names if re.fullmatch(r"[a-z0-9]+-\d{8}\.xml", n)), None)


def fetch_company(client: Any, ticker: str, cik: int | None, prev: GeoRow | None = None) -> GeoRow:
    """한 회사. prev 와 최신 10-K 번호가 같으면 인스턴스를 다시 받지 않는다."""
    from screener.sec_filings import SUBMISSIONS_URL

    today = datetime.now(timezone.utc).date().isoformat()
    row = GeoRow(ticker=ticker, cik=cik, fetched=today)
    if cik is None:
        row.method = "no_filing"
        return row
    sub = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    if not sub:
        row.method = "error"
        return row
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    i = next((k for k, f in enumerate(forms) if f == "10-K"), None)
    if i is None:
        foreign = next((f for f in forms if f in FOREIGN_FORMS), None)
        row.form, row.method = (foreign, "foreign_filer") if foreign else ("", "no_filing")
        return row
    row.form = "10-K"
    row.accession = recent["accessionNumber"][i]
    row.fy_end = recent.get("reportDate", [""] * len(forms))[i] or ""
    if prev and prev.accession == row.accession and prev.method not in ("error", ""):
        return GeoRow(**{**asdict(prev), "fetched": prev.fetched or today})
    acc = row.accession.replace("-", "")
    idx = client.get_json(INDEX_URL.format(cik=cik, acc=acc)) or {}
    name = _instance_name([x.get("name", "") for x in idx.get("directory", {}).get("item", [])])
    xml = client.get_bytes(FILE_URL.format(cik=cik, acc=acc, name=name)) if name else b""
    if not xml or not row.fy_end:
        row.method = "error"
        return row
    try:
        got = parse_instance(xml, row.fy_end)
    except ET.ParseError:
        row.method = "error"
        return row
    for k, v in got.items():
        setattr(row, k, v)
    return row


def read_csv(path: Path | str | None) -> dict[str, GeoRow]:
    """geo_revenue.csv → 티커 → GeoRow. 파일이 없으면 빈 dict."""
    if not path or not Path(path).exists():
        return {}
    out: dict[str, GeoRow] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            kw: dict[str, Any] = {k: r.get(k, "") for k in (fl.name for fl in fields(GeoRow))}
            kw["cik"] = int(kw["cik"]) if str(kw["cik"]).strip() else None
            for k in ("us_rev", "total_rev", "us_share", "foreign_min"):
                kw[k] = _num(kw[k]) if str(kw[k]).strip() else None
            out[kw["ticker"]] = GeoRow(**kw)
    return out


def write_csv(path: Path, rows: Iterable[GeoRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(GeoRow)])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x.ticker):
            d = asdict(r)
            for k in ("us_share", "foreign_min"):
                d[k] = "" if d[k] is None else round(d[k], 4)
            w.writerow(d)


def load_geo_revenue(path: str | None = None) -> dict[str, GeoRow]:
    """밤 실행용 — GEO_REVENUE_FILE(워크플로가 geo-data 브랜치에서 받아 둔 파일)만 읽는다. 네트워크 안 씀, 없으면 {}."""
    return read_csv(path or os.environ.get("GEO_REVENUE_FILE"))


def universe_tickers(base: str = UNIVERSE_BASE) -> list[str]:
    """전체분석 종목 (S&P 500 먼저, 그다음 나스닥/NYSE) — universe-data 브랜치의 시장 파일에서."""
    out: dict[str, None] = {}
    for key in ("sp500", "nasdaq"):
        with urllib.request.urlopen(f"{base}/{key}.json.gz", timeout=60) as r:
            data = json.loads(gzip.decompress(r.read()))
        for t in data.get("items", {}):
            out.setdefault(t, None)
    return list(out)


def main(argv: list[str] | None = None) -> int:
    from screener.sec_filings import SecClient, _norm

    ap = argparse.ArgumentParser(description="SEC 10-K 미국 매출 비중 (geo_revenue.csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prev", type=Path, help="이전 geo_revenue.csv — 10-K 가 그대로인 회사는 다시 안 받는다")
    ap.add_argument("--tickers", nargs="*", help="이 티커만 (없으면 전체분석 S&P 500 + 나스닥/NYSE)")
    ap.add_argument("--sp500-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=300, help="시간 한도 (분) — 넘으면 남은 회사는 이전 값 유지")
    args = ap.parse_args(argv)

    client = SecClient()
    if not client.enabled:
        print("[미국 매출] SEC_USER_AGENT 없음 — 건너뜀")
        return 1
    if args.tickers:
        tickers = [_norm(t) for t in args.tickers]
    else:
        tickers = universe_tickers()
        if args.sp500_only:
            with urllib.request.urlopen(f"{UNIVERSE_BASE}/sp500.json.gz", timeout=60) as r:
                tickers = list(json.loads(gzip.decompress(r.read())).get("items", {}))
    if args.limit:
        tickers = tickers[:args.limit]
    prev = read_csv(args.prev)
    ciks = client.cik_map()
    start = time.monotonic()
    rows: dict[str, GeoRow] = dict(prev)
    for n, t in enumerate(tickers, 1):
        if time.monotonic() - start > args.budget_min * 60:
            print(f"[미국 매출] 시간 한도 — {len(tickers) - n + 1}개는 이전 값 유지")
            break
        try:
            rows[t] = fetch_company(client, t, ciks.get(_norm(t)), prev.get(t))
        except Exception as exc:  # noqa: BLE001 — 한 회사 실패로 멈추지 않는다
            print(f"[미국 매출] {t} 실패: {exc}")
            rows[t] = prev.get(t) or GeoRow(ticker=t, method="error")
        if n % 100 == 0:
            print(f"[미국 매출] {n}/{len(tickers)} ({time.monotonic() - start:.0f}초)")
    write_csv(args.out, rows.values())
    counts: dict[str, int] = {}
    for r in rows.values():
        counts[r.method] = counts.get(r.method, 0) + 1
    print(f"[미국 매출] {len(rows)}개 → {args.out} · " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f" · {time.monotonic() - start:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
