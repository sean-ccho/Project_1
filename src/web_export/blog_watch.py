"""저자 블로그 새 글 · 커버리지 변경 감지 (2026-11-11 · 표시 전용).

사이트 내계좌 › 예측 모델 위쪽 1~8절은 고정 글(블로그 글 44개를 읽고 정리, 숫자는 2026-11-05 기준)이라 저자(Baoro
Research)가 새 글을 올리거나 커버리지(종목 · 의견 · 목표가)를 바꿔도 저절로 바뀌지 않는다. 여기서는 그걸 알아채기만 한다.

- 일주일에 한 번(CHECK_EVERY_DAYS) 블로그 글 사이트맵 + 커버리지 페이지를 받는다 (요청 2개 + 새 글마다 제목 1개).
  실패하면 다음 밤에 다시 받는다. 사이 밤에는 받지 않고, 전에 받은 원자료로 목록만 다시 만든다.
- 정리 안 된 새 글(ANALYZED_POSTS 에 없음) · 정리한 뒤 고친 글(사이트맵 lastmod 가 더 늦음) · 커버리지 차이
  (value_model.AUTHOR_COVERAGE 대비 추가 · 빠짐 · 목표가 · 의견)를 blog_watch.json 에 남긴다.
- 처음 보는 소식만 ntfy 로 한 번 알린다 (NTFY_TOPIC 이 있을 때). 누르면 예측 모델 페이지.
정리(분석)는 사람이 한다 — 끝나면 ANALYZED_POSTS · value_model.AUTHOR_COVERAGE 를 고치면 다음 밤부터 목록에서 빠진다.

결과는 universe-data 맨 위 blog_watch.json — 사이트가 읽고, 다음 밤 실행의 기준이 된다 (universe-scan.yml).
점수 · 필터 · PT-1 · 시트와 무관.

사용: PYTHONPATH=.:src python -m web_export.blog_watch --out-dir DIR [--prev-dir DIR] [--force] [--no-notify]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from web_export.value_model import AUTHOR_AS_OF, AUTHOR_COVERAGE

OUT_NAME = "blog_watch.json"
SCHEMA = 1
BLOG = "https://www.baorocapital.com"
SITEMAP_URL = f"{BLOG}/blog-posts-sitemap.xml"  # Wix 블로그 글 사이트맵 — 글마다 주소 · 마지막 수정일
COVERAGE_URL = f"{BLOG}/general-7"  # 사이트 메뉴 'Coverage' — 종목마다 거래소: 티커 · 의견 · 목표가
POST_URL = f"{BLOG}/post/"
PAGE_URL = "https://seancho.vercel.app/quant/model"  # ntfy 를 누르면 여는 곳
NTFY_URL = "https://ntfy.sh"
CHECK_EVERY_DAYS = 7    # 저자 사이트는 일주일에 한 번만 받는다
TITLE_FETCH_MAX = 5     # 한 번에 제목을 받을 새 글 수 (나머지는 다음 확인 때)
NOTIFY_LINES = 8        # ntfy 한 건에 넣을 줄 수
NOTIFIED_KEEP = 500     # 알린 소식 키를 이만큼만 기억
TIMEOUT = 20
UA = "Mozilla/5.0 (compatible; blog-watch/1.0)"
ET = ZoneInfo("America/Toronto")
BASE_RATING = "OVERWEIGHT"  # 2026-11-05 커버리지 12종목 전부 — AUTHOR_COVERAGE 종목에 'rating' 이 있으면 그걸 쓴다

# 사이트 1~8절에 정리한 글 = 2026-11-11 사이트맵의 44개 (슬러그: 그때 lastmod). 새 글을 정리하면 여기에 더한다.
ANALYZED_AS_OF = "2026-11-11"
ANALYZED_POSTS: dict[str, str] = {
    "coeur-announces-acquisition-of-silvercrest-to-create-leading-global-silver-company": "2024-10-09",
    "byline-bancorp-inc-undervalued-stock-presents-opportunity-post-acquisition-announcement": "2024-10-10",
    "bigbear-ai-awarded-5-year-production-contract-valued-at-165-million-to-deliver-the-u-s-army-s-glob": "2024-10-14",
    "evaluating-market-impact-of-asnd-s-topline-data-on-bmrn-a-historical-and-comparative-analysis": "2024-10-15",
    "rani-therapeutics-evaluating-market-reaction-to-capital-offering-and-dilution-effects": "2024-10-15",
    "b-riley-financial-analyzing-the-potential-for-significant-shareholder-gains": "2024-10-17",
    "maximus-nyse-mms-analyzing-the-impact-of-the-irs-blanket-purchase-agreement-and-revenue-potentia": "2024-10-18",
    "merger-analysis-zeta-global-s-acquisition-of-liveintent-and-its-implications": "2024-10-21",
    "equity-offering-impact-atlantic-union-bankshares-analysis-and-recommendation": "2024-10-22",
    "macrogenics-and-tersera-a-strategic-acquisition-with-significant-risks": "2024-10-23",
    "assessing-the-impact-of-eyepoint-pharmaceutical-s-recent-equity-issuance": "2024-10-29",
    "unlocking-value-a-buy-recommendation-for-joby-aviation-post-secondary-offering-disclosure": "2024-10-30",
    "assessing-mid-penn-bancorp-price-targets-and-growth-opportunities-post-acquisition-and-equity-offer": "2024-11-01",
    "farmland-partners-to-sell-289-million-of-farmland-to-farmland-reserve-inc": "2024-11-04",
    "sinking-prospects-cadiz-inc-faces-financial-and-dilution-challenges": "2024-11-04",
    "assessing-the-potential-price-floor-for-briacell-therapeutics-a-dilution-and-return-on-equity-analy": "2024-12-28",
    "evaluating-the-akya-qtrx-merger-risk-reward-and-implied-upside": "2025-02-24",
    "march-2025-market---operational-update": "2025-04-01",
    "newsletter-updates": "2025-04-01",
    "emergency-update-call-for-caution-for-u-s-equities": "2025-09-16",
    "october-bottom-up-ierp-a-series-backtest-result": "2025-10-02",
    "market-update-october-2025": "2025-11-01",
    "november-bottom-up": "2025-11-06",
    "market-update-november-2025": "2025-12-02",
    "robert-half-inc": "2026-02-27",
    "factset-research-systems-inc": "2026-02-28",
    "cgi-inc": "2026-03-05",
    "continuing-market-update": "2026-03-11",
    "feb-2026-market-analysis": "2026-03-11",
    "market-update": "2026-03-11",
    "market-update-06-2025": "2026-03-11",
    "market-update-august-2025": "2026-03-11",
    "market-update-september-2025": "2026-03-11",
    "market-update-april-2026": "2026-04-02",
    "market-update-may-2026": "2026-06-02",
    "market-update-june-2026": "2026-07-02",
    "pembina-pipeline-corporation": "2026-07-09",
    "red-chris-block-cave-imperial-metals": "2026-07-13",
    "market-upate-sept-2026": "2026-10-01",
    "nyse-acn-fair-value-update": "2026-10-01",
    "nyse-fds-fair-value-update": "2026-10-01",
    "tsx-engh-fair-value-update": "2026-10-01",
    "fiserv-inc": "2026-10-05",
    "market-upate-oct-2026": "2026-11-02",
}

Getter = Callable[[str], str]
Sender = Callable[[str, str, str], bool]


def today_et() -> date:
    """온타리오 날짜."""
    return datetime.now(ET).date()


def get_text(url: str) -> str:
    """GET → 본문 글자. 실패하면 예외."""
    import requests  # noqa: PLC0415

    resp = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _short(exc: BaseException) -> str:
    return (str(exc).splitlines() or [type(exc).__name__])[0][:160]


# ── 사이트맵 · 글 제목

_URL_RE = re.compile(r"<url>(.*?)</url>", re.S)
_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")  # <image:loc> 은 안 잡힌다
_LASTMOD_RE = re.compile(r"<lastmod>\s*([^<\s]+)\s*</lastmod>")


def parse_sitemap(xml: str) -> dict[str, str]:
    """블로그 글 사이트맵 → {슬러그: 마지막 수정일 YYYY-MM-DD}. /post/ 가 아닌 주소는 뺀다."""
    out: dict[str, str] = {}
    for block in _URL_RE.findall(xml):
        loc = _LOC_RE.search(block)
        if not loc or "/post/" not in loc.group(1):
            continue
        slug = loc.group(1).split("/post/", 1)[1].strip("/")
        lastmod = _LASTMOD_RE.search(block)
        if slug:
            out[slug] = lastmod.group(1)[:10] if lastmod else ""
    return out


def post_title(page: str) -> str | None:
    """글 페이지 → 제목 (og:title, 없으면 <title>). 뒤에 붙는 ' | Baoro …' 는 뗀다."""
    raw = None
    for tag in re.findall(r"<meta\b[^>]*>", page, re.I):
        if re.search(r"""property=["']og:title["']""", tag, re.I):
            m = re.search(r"""content=["']([^"']*)["']""", tag, re.I)
            raw = m.group(1) if m else None
            break
    if raw is None:
        m = re.search(r"<title[^>]*>(.*?)</title>", page, re.I | re.S)
        raw = m.group(1) if m else None
    if not raw:
        return None
    title = re.split(r"\s+[|\-–]\s+Baoro\b", html.unescape(raw).strip(), maxsplit=1, flags=re.I)[0].strip()
    return title or None


def slug_title(slug: str) -> str:
    """제목을 못 받았을 때 — 슬러그를 글자로."""
    words = re.sub(r"-+", " ", slug).strip()
    return words[:1].upper() + words[1:]


# ── 커버리지 페이지

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None) | {0xA0: " "}
_EXCH_RE = re.compile(r"^([A-Z]{2,12})\s*:\s*([A-Z][A-Z0-9.\-]{0,9})$")
_PRICE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def page_lines(page: str) -> list[str]:
    """HTML → 보이는 글 줄 (스크립트 · 스타일 · 태그 · 폭 없는 공백 제거)."""
    text = re.sub(r"<(script|style)\b.*?</\1\s*>", " ", page, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", text)).translate(_INVISIBLE)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def our_ticker(exch: str, symbol: str) -> str:
    """저자 표기(거래소: 티커) → 우리 티커. TSX: GIB.A → GIB-A.TO, TSXV → .V, 미국은 점 → 줄표 (BRK.B → BRK-B)."""
    sym = symbol.replace(".", "-")
    if exch in ("TSX", "TSE"):
        return f"{sym}.TO"
    if exch in ("TSXV", "CVE"):
        return f"{sym}.V"
    return sym


def parse_coverage(page: str) -> list[dict[str, Any]]:
    """커버리지 페이지 → [{ticker, exch, symbol, rating, target}] (페이지 순서).
    'NYSE: RHI' 같은 줄부터 다음 거래소 줄 전까지가 한 종목 — 'RECOMMENDATION :' 뒤 첫 글 = 의견,
    'TARGET PRICE:' 뒤 첫 $ 금액 = 목표가. 의견 · 목표가가 둘 다 없는 줄은 종목 카드가 아니라고 보고 뺀다."""
    lines = page_lines(page)
    heads = [(i, m) for i, ln in enumerate(lines) if (m := _EXCH_RE.match(ln))]
    out: list[dict[str, Any]] = []
    for k, (i, m) in enumerate(heads):
        block = lines[i + 1:heads[k + 1][0] if k + 1 < len(heads) else len(lines)]
        rating: str | None = None
        target: float | None = None
        for j, ln in enumerate(block):
            up = ln.upper()
            if rating is None and up.startswith("RECOMMENDATION"):
                rest = ln.split(":", 1)[1].strip() if ":" in ln else ""
                rating = rest or next((b for b in block[j + 1:j + 3] if not b.upper().startswith("TARGET")), None)
            elif target is None and up.startswith("TARGET"):
                pm = _PRICE_RE.search(" ".join(block[j:j + 3]))
                if pm:
                    target = float(pm.group(1).replace(",", ""))
        if rating is None and target is None:
            continue
        exch, symbol = m.group(1), m.group(2)
        out.append({"ticker": our_ticker(exch, symbol), "exch": exch, "symbol": symbol,
                    "rating": rating.upper() if rating else None, "target": target})
    return out


def coverage_diff(page_items: Iterable[Mapping[str, Any]],
                  ours: Iterable[Mapping[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """지금 저자 페이지 vs 우리가 코드에 넣어 둔 커버리지(기본 value_model.AUTHOR_COVERAGE) → 추가 · 빠짐 · 목표가 · 의견 차이."""
    ours_by = {str(c["ticker"]): c for c in (AUTHOR_COVERAGE if ours is None else ours)}
    now_by = {str(c["ticker"]): c for c in page_items}
    added = [dict(c) for t, c in now_by.items() if t not in ours_by]
    removed = [{"ticker": t, "target": c.get("target")} for t, c in ours_by.items() if t not in now_by]
    target: list[dict[str, Any]] = []
    rating: list[dict[str, Any]] = []
    for t, c in now_by.items():
        o = ours_by.get(t)
        if o is None:
            continue
        if c.get("target") is not None and abs(float(c["target"]) - float(o["target"])) >= 0.005:
            target.append({"ticker": t, "ours": o["target"], "now": c["target"]})
        base = str(o.get("rating") or BASE_RATING).upper()
        if c.get("rating") and c["rating"] != base:
            rating.append({"ticker": t, "before": base, "now": c["rating"]})
    return {"added": added, "removed": removed, "target": target, "rating": rating}


# ── 목록 · 알림

def post_lists(posts: Mapping[str, str], titles: Mapping[str, str], seen: Mapping[str, str],
               analyzed: Mapping[str, str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(정리 안 된 새 글, 정리한 뒤 고친 글) — 둘 다 최근 수정일 순. analyzed 기본 = ANALYZED_POSTS."""
    analyzed = ANALYZED_POSTS if analyzed is None else analyzed

    def row(slug: str) -> dict[str, Any]:
        return {"slug": slug, "url": POST_URL + slug, "title": titles.get(slug) or slug_title(slug),
                "lastmod": posts[slug], "seen": seen.get(slug)}

    new = [row(s) for s in posts if s not in analyzed]
    edited = [row(s) for s in posts if s in analyzed and posts[s] and posts[s] > analyzed[s]]
    key = lambda r: (r["lastmod"] or "", r["slug"])  # noqa: E731
    return sorted(new, key=key, reverse=True), sorted(edited, key=key, reverse=True)


def _money(x: Any) -> str:
    return f"${float(x):,.2f}" if isinstance(x, (int, float)) else "—"


def news_items(out: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(알림 키, 한 줄) — 키가 같으면 같은 소식 (한 번만 알린다)."""
    items: list[tuple[str, str]] = []
    for p in out.get("new_posts") or []:
        items.append((f"post:{p['slug']}", f"새 글 ({p['lastmod'][5:]}) {p['title']}"))
    for p in out.get("edited_posts") or []:
        items.append((f"edit:{p['slug']}:{p['lastmod']}", f"고친 글 ({p['lastmod'][5:]}) {p['title']}"))
    diff = out.get("coverage_diff") or {}
    for c in diff.get("added", []):
        items.append((f"cov:add:{c['ticker']}", f"커버리지 추가 {c['ticker']} · {c.get('rating') or '의견 없음'}"
                                                 f" · 목표가 {_money(c.get('target'))}"))
    for c in diff.get("removed", []):
        items.append((f"cov:rm:{c['ticker']}", f"커버리지에서 빠짐 {c['ticker']}"))
    for c in diff.get("target", []):
        items.append((f"cov:tgt:{c['ticker']}:{c['now']}", f"목표가 {c['ticker']} {_money(c['ours'])} → {_money(c['now'])}"))
    for c in diff.get("rating", []):
        items.append((f"cov:rate:{c['ticker']}:{c['now']}", f"의견 {c['ticker']} {c['before']} → {c['now']}"))
    return items


def send_ntfy(topic: str, title: str, body: str) -> bool:
    """ntfy.sh 로 폰 푸시 (JSON — 한글 제목). 누르면 예측 모델 페이지."""
    import requests  # noqa: PLC0415

    try:
        resp = requests.post(NTFY_URL, json={"topic": topic, "title": title, "message": body, "priority": 3,
                                             "tags": ["newspaper"], "click": PAGE_URL}, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[블로그 감지] ntfy 실패: {_short(exc)}")
        return False


def notify(out: dict[str, Any], topic: str, send: Sender | None = None) -> int:
    """아직 안 알린 소식만 한 번에 보낸다. 보낸 소식 수 (주제가 없거나 실패하면 0 — 다음 밤에 다시)."""
    send = send or send_ntfy
    done = set(out.get("notified") or [])
    fresh = [(k, line) for k, line in news_items(out) if k not in done]
    if not fresh or not topic:
        return 0
    lines = [line for _, line in fresh[:NOTIFY_LINES]]
    if len(fresh) > NOTIFY_LINES:
        lines.append(f"… 외 {len(fresh) - NOTIFY_LINES}건")
    lines.append("→ 예측 모델 페이지에서 확인 · 정리는 Claude 에게")
    if not send(topic, f"Baoro 블로그 — 새 소식 {len(fresh)}건", "\n".join(lines)):
        return 0
    out["notified"] = (list(out.get("notified") or []) + [k for k, _ in fresh])[-NOTIFIED_KEEP:]
    return len(fresh)


# ── 실행

def run(prev: Mapping[str, Any] | None, today: date, *, get: Getter | None = None, force: bool = False) -> dict[str, Any]:
    """전 결과 + 오늘 → 새 결과. 확인할 때가 되면(마지막 성공 + CHECK_EVERY_DAYS, 처음, force) 받아 오고,
    아니면 전에 받은 원자료(posts · titles · coverage)로 목록만 다시 만든다 (ANALYZED_POSTS 를 고친 게 바로 반영)."""
    get = get or get_text
    prev = dict(prev or {})
    posts: dict[str, str] = dict(prev.get("posts") or {})
    titles: dict[str, str] = dict(prev.get("titles") or {})
    seen: dict[str, str] = dict(prev.get("seen") or {})
    coverage: list[dict[str, Any]] = list(prev.get("coverage") or [])
    checked: str | None = prev.get("checked")
    attempted: str | None = prev.get("attempted")
    errors: list[str] = list(prev.get("errors") or [])
    due = force or not checked or (today - date.fromisoformat(checked)).days >= CHECK_EVERY_DAYS
    if due:
        errors, ok, day = [], True, today.isoformat()
        try:
            got = parse_sitemap(get(SITEMAP_URL))
            if not got:
                raise ValueError("사이트맵에서 글을 못 찾음 (형식이 바뀌었나)")
            posts = got
            for slug in posts:
                seen.setdefault(slug, day)
        except Exception as exc:  # noqa: BLE001
            ok = False
            errors.append(f"사이트맵: {_short(exc)}")
        try:
            got_cov = parse_coverage(get(COVERAGE_URL))
            if not got_cov:
                raise ValueError("커버리지 종목을 못 찾음 (페이지가 바뀌었나)")
            coverage = got_cov
        except Exception as exc:  # noqa: BLE001
            ok = False
            errors.append(f"커버리지: {_short(exc)}")
        new, edited = post_lists(posts, titles, seen)
        for p in [p for p in new + edited if p["slug"] not in titles][:TITLE_FETCH_MAX]:
            try:
                title = post_title(get(p["url"]))
                if title:
                    titles[p["slug"]] = title
            except Exception as exc:  # noqa: BLE001 — 제목은 없어도 된다 (슬러그로 보여 줌)
                errors.append(f"글 제목 {p['slug'][:40]}: {_short(exc)}")
        attempted = day
        if ok:
            checked = day
    new, edited = post_lists(posts, titles, seen)
    return {
        "schema": SCHEMA,
        "checked": checked,  # 마지막으로 사이트맵 · 커버리지를 둘 다 받은 날 (온타리오)
        "attempted": attempted,
        "next": (date.fromisoformat(checked) + timedelta(days=CHECK_EVERY_DAYS)).isoformat() if checked else None,
        "every_days": CHECK_EVERY_DAYS,
        "analyzed_as_of": ANALYZED_AS_OF,
        "analyzed_posts": len(ANALYZED_POSTS),
        "coverage_as_of": AUTHOR_AS_OF,
        "posts_total": len(posts),
        "new_posts": new,
        "edited_posts": edited,
        "coverage_diff": coverage_diff(coverage) if coverage else None,
        "errors": errors,
        "blog_url": f"{BLOG}/blog",
        "coverage_url": COVERAGE_URL,
        # 원자료 — 다음 밤 기준 (사이트는 안 씀)
        "posts": posts,
        "titles": titles,
        "seen": seen,
        "coverage": coverage,
        "notified": list(prev.get("notified") or []),
    }


def load_prev(*paths: Path | None) -> dict[str, Any]:
    """앞 경로부터 처음 읽히는 전 결과 (없으면 빈 것)."""
    for p in paths:
        if p and p.exists():
            try:
                data = json.loads(p.read_text("utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError) as exc:
                print(f"[블로그 감지] 전 결과 읽기 실패 ({p}): {_short(exc)}")
    return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="저자 블로그 새 글 · 커버리지 변경 감지 (예측 모델 · 표시 전용)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prev-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help=f"{CHECK_EVERY_DAYS}일이 안 됐어도 지금 확인")
    ap.add_argument("--no-notify", action="store_true", help="ntfy 안 보냄")
    args = ap.parse_args(argv)

    out_path = args.out_dir / OUT_NAME
    prev = load_prev(out_path, args.prev_dir / OUT_NAME if args.prev_dir else None)
    out = run(prev, today_et(), force=args.force)
    sent = 0 if args.no_notify else notify(out, os.getenv("NTFY_TOPIC", "").strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    diff = out["coverage_diff"] or {}
    n_cov = sum(len(v) for v in diff.values())
    print(f"[블로그 감지] 확인 {out['checked'] or '아직'} (시도 {out['attempted'] or '-'}) · 글 {out['posts_total']}개 · "
          f"새 글 {len(out['new_posts'])} · 고친 글 {len(out['edited_posts'])} · 커버리지 차이 {n_cov} · "
          f"알림 {sent} · 다음 {out['next'] or '-'}")
    for e in out["errors"]:
        print(f"[블로그 감지] ⚠️ {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
