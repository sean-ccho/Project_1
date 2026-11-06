"""국회의원 주식 거래 신고 (STOCK Act PTR) — 오늘 탭 '최근 국회의원 거래 신고' · 종목 카드 · 목록 칩 (2026-11-06, 표시 전용).

매매 신호로 쓰지 않는다 (점수 · 필터 · PT-1 무관):
- 신고는 거래 뒤 최대 45일 늦고, 금액은 구간으로만 나온다.
- 신고 뒤 따라 사면 시장과 비슷했다 — NANC ETF(민주당 의원 거래를 신고 뒤 따라 삼) 2023-02-07~2026-11-06 +92.5%,
  SPY +88.5% (샤프 1.12 · 1.21). Belmont 외(2022): 2012–2020 평균은 무작위 수준.
- 지도부만 앞섰다는 연구(Wei & Zhou, NBER w34524 — 지도부 20명, 거래일 기준)는 섀도로만 1년 기록한다 (shadow_track CONG_*).
법: 5 U.S.C. §13107(c) — 신고서를 상업 목적으로 얻거나 쓰면 불법 (뉴스 매체 예외). 무료 개인 사이트에 보여 주는 것만.

자료 (2026-11-06 확인):
- 하원 Clerk 연도별 색인 zip (XML — FilingType P = PTR) + PTR PDF. 전자 신고(DocID 2…)는 pypdf 로 글자를 뽑아 거래 줄을
  나누고, 손으로 쓴 스캔 신고(8 · 9…)는 원문 링크만 남긴다.
- 상원 eFD: 사이트 이용 동의(신고서 사용 제한 확인)를 사람이 해야 해서 아직 안 받는다 — 사용자 결정 대기.
- 정당 · 지도부: unitedstates/congress-legislators (공개 데이터) — 하원 선거구(주 + 번호)로 맞춘다. 지도부 = 그 자료의
  당 지도부 직함(의장 · 원내대표 · 원내총무 · 당 회의 의장 등)이 있는 의원.

    PYTHONPATH=.:src python -m data.congress_trades --out-dir "$RUNNER_TEMP/universe" --prev-dir "$RUNNER_TEMP/prev"
출력: congress.json (오늘 탭 — 최근 RECENT_DAYS 일 신고), congress_tickers.json (종목별 HISTORY_DAYS 일),
      snapshot/congress_state.json.gz (이미 읽은 신고 — 다음 밤엔 새 신고만 받는다)
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

HOUSE_ZIP = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
HOUSE_PDF = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc}.pdf"
LEGIS_URL = "https://unitedstates.github.io/congress-legislators/legislators-current.json"
UA = {"User-Agent": "Mozilla/5.0 (personal research; congress_trades)"}
STATE_NAME = "congress_state.json.gz"
RECENT_DAYS = 30          # 오늘 탭 목록 — 신고일 기준
HISTORY_DAYS = 365        # 종목 카드 · 상태 파일에 남기는 기간
SUMMARY_DAYS = 90         # 목록 칩 · 필터 (매수 · 매도 수)
MAX_NEW_DOCS = 1200       # 한 번에 새로 받는 PDF 수 (처음 한 번은 작년 치까지)
BUDGET_SEC = 900
PAUSE_SEC = 0.25          # PDF 사이 쉬는 시간 (사이트에 부담 주지 않게)
BIG_MIN = 50_001          # '큰 금액' = 구간 하한 $50,001 이상

OWNERS = {"SP": "spouse", "JT": "joint", "DC": "child"}
TX_TYPES = {"P": "buy", "S": "sell", "S (partial)": "sell", "E": "exchange"}

# ── 하원 색인 ────────────────────────────────────────────────────────────────


def parse_house_index(raw_zip: bytes) -> list[dict[str, Any]]:
    """연도별 zip → PTR 신고 목록 [{doc, last, first, state, district, filed(ISO), scanned}] (신고일 순)."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xml"))
        root = ET.fromstring(z.read(name))
    out = []
    for m in root:
        row = {c.tag: (c.text or "").strip() for c in m}
        if row.get("FilingType") != "P" or not row.get("DocID"):
            continue
        try:
            filed = datetime.strptime(row.get("FilingDate", ""), "%m/%d/%Y").date().isoformat()
        except ValueError:
            continue
        sd = row.get("StateDst", "")
        m2 = re.fullmatch(r"([A-Z]{2})(\d{1,2})", sd)
        out.append({"doc": row["DocID"], "last": row.get("Last", ""), "first": row.get("First", ""),
                    "state": m2.group(1) if m2 else sd[:2], "district": int(m2.group(2)) if m2 else None,
                    "filed": filed, "year": int(row.get("Year") or filed[:4]),
                    "scanned": not row["DocID"].startswith("2")})
    return sorted(out, key=lambda r: (r["filed"], r["doc"]))


# ── PTR PDF ──────────────────────────────────────────────────────────────────

# 이름표 글자가 NUL 로 나온다 ('F\x00\x00 S\x00\x00: New') — NUL 을 지우면 'F S: New' · 'S O: 계좌' · 'D: 설명'
_META_LINE = re.compile(r"^\s*[A-Z](?:\s+[A-Z])?\s*:\s")
_HEADER_LINES = re.compile(
    r"^\s*(?:Filing ID #\d+|ID Owner Asset Transaction|Type|Date Notification|Date|Amount Cap\.|Gains >|\$200\?)\s*$")
_TABLE_START = re.compile(r"^\s*ID Owner Asset Transaction")
_TX = re.compile(
    r"(?:(?P<owner>\b(?:SP|JT|DC))\s+)?(?P<asset>[^\[\]]*?)\((?P<ticker>[A-Z][A-Z0-9.\-]{0,9})\)\s*\[(?P<atype>[A-Z]{2})\]\s*"
    r"(?P<type>S \(partial\)|P|S|E)\s+(?P<tdate>\d{2}/\d{2}/\d{4})\s*(?P<ndate>\d{2}/\d{2}/\d{4})\s*"
    r"(?P<amount>Over \$[\d,]+|\$[\d,]+\s*-\s*\$[\d,]+|\$[\d,]+)")
_AMOUNT = re.compile(r"\$([\d,]+)")


def pdf_text(raw: bytes) -> str:
    """PDF → 글자 (페이지 순). 스캔 이미지면 거의 빈 글자."""
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_amount(text: str) -> tuple[int | None, int | None]:
    """'$1,001 - $15,000' → (1001, 15000), 'Over $50,000,000' → (50000001, None)."""
    nums = [int(x.replace(",", "")) for x in _AMOUNT.findall(text)]
    if not nums:
        return None, None
    if text.strip().lower().startswith("over"):
        return nums[0] + 1, None
    return nums[0], nums[1] if len(nums) > 1 else nums[0]


def _iso(mdy: str) -> str | None:
    try:
        return datetime.strptime(mdy, "%m/%d/%Y").date().isoformat()
    except ValueError:
        return None


def parse_ptr_text(text: str) -> list[dict[str, Any]]:
    """PTR 글자 → 거래 [{owner, asset, ticker, atype, type, partial, date, notified, low, high, amount}].
    티커가 없는 자산(채권 · 비상장 펀드 등)은 뺀다. 표 머리 앞(제목 · 이름 · 선거구)과 페이지 머리줄 · 계좌 · 설명 줄을
    지운 뒤 한 줄로 이어 거래 꼴을 찾는다. 소유자(SP 배우자 · JT 공동 · DC 자녀)는 이름 앞 두 글자."""
    lines = text.replace("\x00", "").splitlines()
    start = next((i for i, ln in enumerate(lines) if _TABLE_START.match(ln)), -1)
    lines = [ln for ln in lines[start + 1:] if not _META_LINE.match(ln) and not _HEADER_LINES.match(ln)]
    flat = re.sub(r"\s+", " ", " ".join(lines))
    out = []
    for m in _TX.finditer(flat):
        # 앞 거래가 티커 없는 자산이었으면 그 줄이 이름 앞에 붙는다 — 마지막 금액 · 날짜 뒤만
        asset = re.split(r"\$[\d,]+(?:\s*-\s*\$[\d,]+)?|\d{2}/\d{2}/\d{4}", m.group("asset"))[-1].strip(" -,")
        owner = m.group("owner") or ""
        lead = re.match(r"^(SP|JT|DC)\s+(.*)$", asset)
        if lead:
            owner, asset = lead.group(1), lead.group(2)
        low, high = parse_amount(m.group("amount"))
        out.append({
            "owner": OWNERS.get(owner, "self"),
            "asset": asset[:120],
            "ticker": m.group("ticker").replace(".", "-"),
            "atype": m.group("atype"),
            "type": TX_TYPES[m.group("type")],
            "partial": m.group("type") == "S (partial)",
            "date": _iso(m.group("tdate")),
            "notified": _iso(m.group("ndate")),
            "low": low,
            "high": high,
            "amount": re.sub(r"\s+", " ", m.group("amount")).strip(),
        })
    return out


# ── 의원 정보 ────────────────────────────────────────────────────────────────


def parse_legislators(raw: bytes | str) -> dict[tuple[str, int], dict[str, Any]]:
    """legislators-current.json → {(주, 선거구): {name, party, leader(직함 | None), bioguide}} (하원 현직만)."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for p in json.loads(raw):
        term = (p.get("terms") or [{}])[-1]
        if term.get("type") != "rep":
            continue
        roles = [r.get("title") for r in p.get("leadership_roles", [])
                 if r.get("chamber") == "house" and not r.get("end")]
        out[(term.get("state", ""), int(term.get("district") or 0))] = {
            "name": (p.get("name") or {}).get("official_full"),
            "party": {"Democrat": "D", "Republican": "R", "Independent": "I"}.get(term.get("party", ""), term.get("party")),
            "leader": roles[0] if roles else None,
            "bioguide": (p.get("id") or {}).get("bioguide"),
        }
    return out


# ── 받기 · 쌓기 ──────────────────────────────────────────────────────────────


def load_state(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"docs": {}}
    try:
        return json.loads(gzip.decompress(path.read_bytes()))
    except Exception:  # noqa: BLE001 — 깨진 상태는 처음부터
        return {"docs": {}}


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))


def update(state: dict[str, Any], today: date, get: Callable[[str], bytes], *, years: Iterable[int] | None = None,
           max_new: int = MAX_NEW_DOCS, budget_sec: float = BUDGET_SEC, pause: float = PAUSE_SEC,
           log: Callable[[str], None] = print) -> dict[str, int]:
    """색인을 받아 아직 안 읽은 신고(HISTORY_DAYS 안)만 PDF 를 받아 읽는다. 오래된 신고는 지운다 → 개수 요약."""
    docs: dict[str, Any] = state.setdefault("docs", {})
    cutoff = (today - timedelta(days=HISTORY_DAYS)).isoformat()
    years = list(years or sorted({today.year, (today - timedelta(days=HISTORY_DAYS)).year}))
    filings: list[dict[str, Any]] = []
    for y in years:
        try:
            filings += parse_house_index(get(HOUSE_ZIP.format(year=y)))
        except Exception as exc:  # noqa: BLE001
            log(f"[의회] {y} 색인 실패: {exc}")
    filings = list({f["doc"]: f for f in filings}.values())       # 같은 신고가 두 해 색인에 있어도 한 번
    want = [f for f in filings if f["filed"] >= cutoff and f["doc"] not in docs]
    want.sort(key=lambda f: f["filed"], reverse=True)            # 최근 신고부터
    t0 = time.monotonic()
    got = scanned = failed = 0
    for f in want[:max_new]:
        if time.monotonic() - t0 > budget_sec:
            break
        entry = {k: f[k] for k in ("last", "first", "state", "district", "filed", "year")}
        if f["scanned"]:
            docs[f["doc"]] = {**entry, "status": "scanned", "tx": []}
            scanned += 1
            continue
        try:
            text = pdf_text(get(HOUSE_PDF.format(year=f["year"], doc=f["doc"])))
            tx = parse_ptr_text(text)
            docs[f["doc"]] = {**entry, "status": "ok" if tx or len(text) > 200 else "scanned", "tx": tx}
            got += 1
        except Exception as exc:  # noqa: BLE001 — 다음 밤에 다시
            failed += 1
            log(f"[의회] {f['doc']} 실패: {str(exc)[:120]}")
        time.sleep(pause)
    for k in [k for k, v in docs.items() if v.get("filed", "") < cutoff]:
        del docs[k]
    left = max(0, len(want) - got - scanned - failed)
    log(f"[의회] 하원 PTR 색인 {len(filings)}건 · 새로 읽음 {got} · 스캔본 {scanned} · 실패 {failed} · 남음 {left} · 보관 {len(docs)}")
    return {"index": len(filings), "read": got, "scanned": scanned, "failed": failed, "left": left, "kept": len(docs)}


def _tx_rows(state: Mapping[str, Any], members: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    """상태 → 거래 한 줄씩 (의원 정보 · 신고까지 걸린 날 · 원문 링크)."""
    rows = []
    for doc, d in state.get("docs", {}).items():
        mem = members.get((d.get("state", ""), int(d.get("district") or 0)), {})
        name = mem.get("name") or f"{d.get('first', '')} {d.get('last', '')}".strip()
        base = {"doc": doc, "filed": d["filed"], "member": name, "party": mem.get("party"), "state": d.get("state"),
                "district": d.get("district"), "leader": mem.get("leader"), "chamber": "house",
                "url": HOUSE_PDF.format(year=d.get("year") or d["filed"][:4], doc=doc)}
        for tx in d.get("tx", []):
            lag = None
            if tx.get("date"):
                lag = (date.fromisoformat(d["filed"]) - date.fromisoformat(tx["date"])).days
            rows.append({**base, **tx, "lag": lag})
    return sorted(rows, key=lambda r: (r["filed"], r.get("high") or r.get("low") or 0), reverse=True)


def build_outputs(state: Mapping[str, Any], members: Mapping[tuple[str, int], Mapping[str, Any]],
                  today: date) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """→ (congress.json, {티커: 12개월 거래}, {티커: 90일 요약 — 목록 칩})."""
    rows = _tx_rows(state, members)
    recent_from = (today - timedelta(days=RECENT_DAYS)).isoformat()
    summary_from = (today - timedelta(days=SUMMARY_DAYS)).isoformat()
    scanned = sorted(({"doc": k, "filed": v["filed"], "member": f"{v.get('first', '')} {v.get('last', '')}".strip(),
                       "url": HOUSE_PDF.format(year=v.get("year") or v["filed"][:4], doc=k)}
                      for k, v in state.get("docs", {}).items()
                      if v.get("status") == "scanned" and v["filed"] >= recent_from),
                     key=lambda x: x["filed"], reverse=True)
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append({k: r[k] for k in (
            "filed", "date", "lag", "member", "party", "state", "leader", "owner", "type", "partial", "atype",
            "low", "high", "amount", "url")})
    summary: dict[str, dict[str, Any]] = {}
    for t, txs in by_ticker.items():
        near = [x for x in txs if x["filed"] >= summary_from]
        if not near:
            continue
        summary[t] = {"buys": sum(1 for x in near if x["type"] == "buy"),
                      "sells": sum(1 for x in near if x["type"] == "sell"),
                      "members": len({x["member"] for x in near}), "last": max(x["filed"] for x in near),
                      "leader": any(x["leader"] for x in near),
                      "big_buy": any(x["type"] == "buy" and (x["low"] or 0) >= BIG_MIN for x in near)}
    recent = [r for r in rows if r["filed"] >= recent_from]
    payload = {
        "schema": 1,
        "generated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "as_of": today.isoformat(),
        "source": "house",
        "days": RECENT_DAYS,
        "counts": {"filings": len(state.get("docs", {})), "tx_365d": len(rows), "tx_recent": len(recent),
                   "tickers_90d": len(summary)},
        "recent": recent[:400],
        "scanned": scanned[:50],
    }
    return payload, by_ticker, summary


def _get(url: str, timeout: int = 40) -> bytes:
    import requests

    last: Exception | None = None
    for _ in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise RuntimeError(f"{url}: {last}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="하원 PTR 거래 신고 → congress.json (표시 전용)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prev-dir", type=Path, help="지난 밤 snapshot/congress_state.json.gz 가 있는 폴더")
    ap.add_argument("--max-new", type=int, default=MAX_NEW_DOCS)
    ap.add_argument("--budget", type=float, default=BUDGET_SEC)
    args = ap.parse_args(argv)

    today = datetime.now(ZoneInfo("America/New_York")).date()
    state = load_state(args.prev_dir / STATE_NAME if args.prev_dir else None)
    stats = update(state, today, _get, max_new=args.max_new, budget_sec=args.budget)
    try:
        members = parse_legislators(_get(LEGIS_URL))
    except Exception as exc:  # noqa: BLE001 — 정당 · 지도부 없이도 싣는다
        print(f"[의회] 의원 정보 실패 (정당 · 지도부 없이): {exc}")
        members = {}
    payload, by_ticker, summary = build_outputs(state, members, today)
    payload["counts"].update({k: stats[k] for k in ("read", "scanned", "failed", "left")})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "congress.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                                                encoding="utf-8")
    (args.out_dir / "congress_tickers.json").write_text(
        json.dumps({"as_of": today.isoformat(), "tickers": by_ticker, "summary": summary}, ensure_ascii=False,
                   separators=(",", ":")), encoding="utf-8")
    save_state(args.out_dir / "snapshot" / STATE_NAME, state)
    leaders = sorted({r["member"] for r in payload["recent"] if r.get("leader")})
    print(f"[의회] 최근 {RECENT_DAYS}일 거래 {payload['counts']['tx_recent']}건 · 종목 {len(by_ticker)} · "
          f"90일 칩 {len(summary)}종목 · 지도부 {leaders or '없음'}")
    return 0 if payload["counts"]["filings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
