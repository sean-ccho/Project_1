"""web_export.blog_watch — 저자 블로그 새 글 · 커버리지 변경 감지 (네트워크 없이 가짜 페이지로)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from web_export import blog_watch as bw
from web_export.value_model import AUTHOR_COVERAGE

# 2026-11-05 저자 커버리지 페이지 표기 (거래소: 티커) — 'NAS: RCI' · 'TSX:III' 처럼 실제 페이지 그대로
EXCH = {"RHI": "NYSE: RHI", "IT": "NYSE: IT", "FDS": "NYSE: FDS", "NICE": "NASDAQ: NICE", "SJ.TO": "TSX: SJ",
        "EFOR": "NYSE: EFOR", "RCI": "NAS: RCI", "ENGH.TO": "TSX: ENGH", "GIB-A.TO": "TSX: GIB.A",
        "PPL.TO": "TSX: PPL", "ACN": "NYSE: ACN", "III.TO": "TSX:III"}


def coverage_page(cards: list[tuple[str, str, float]]) -> str:
    """Wix 커버리지 페이지 모양 — 줄마다 태그, 폭 없는 공백, 스크립트 섞임."""
    parts = ["<html><head><title>Coverage | Baoro Capital</title><script>var x = 'NYSE: FAKE';</script></head><body>",
             "<p>BAORO RESEARCH</p><p>Coverage</p><h2>RECOMMENDATIONS</h2>"]
    for head, rating, target in cards:
        parts.append(f"<div><h3>{head}</h3><p>RECOMMENDATION :</p><p>{rating}</p><p>​</p>"
                     f"<p>TARGET PRICE:</p><p>${target:,.2f}​</p><p>​</p></div>")
    parts.append("<p>bottom of page</p></body></html>")
    return "".join(parts)


def base_cards() -> list[tuple[str, str, float]]:
    return [(EXCH[c["ticker"]], "OVERWEIGHT", float(c["target"])) for c in AUTHOR_COVERAGE]


def sitemap(posts: dict[str, str], extra_urls: tuple[str, ...] = ()) -> str:
    urls = "".join(f"<url><loc>{bw.POST_URL}{s}</loc><lastmod>{lm}</lastmod>"
                   f"<image:image><image:loc>https://static.wixstatic.com/media/x.png</image:loc></image:image></url>"
                   for s, lm in posts.items())
    urls += "".join(f"<url><loc>{u}</loc><lastmod>2026-01-01</lastmod></url>" for u in extra_urls)
    return f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'


class FakeWeb:
    """주소 → 글. 없는 주소 · fail 에 든 주소는 예외. 부른 주소를 센다."""

    def __init__(self, pages: dict[str, str], fail: tuple[str, ...] = ()) -> None:
        self.pages, self.fail, self.calls = pages, fail, []

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        if url in self.fail or url not in self.pages:
            raise RuntimeError(f"HTTP 503 {url}")
        return self.pages[url]


NEW_SLUG = "market-update-nov-2026"


def web_with_news() -> FakeWeb:
    posts = dict(bw.ANALYZED_POSTS)
    posts[NEW_SLUG] = "2026-12-01"
    posts["fiserv-inc"] = "2026-11-20"  # 정리한 뒤 고친 글
    return FakeWeb({
        bw.SITEMAP_URL: sitemap(posts, extra_urls=("https://www.baorocapital.com/about",)),
        bw.COVERAGE_URL: coverage_page(base_cards()),
        bw.POST_URL + NEW_SLUG: '<html><head><meta content="Market Update — Nov 2026 | Baoro Capital" '
                                'property="og:title"></head></html>',
        bw.POST_URL + "fiserv-inc": "<html><head><title>Fiserv Inc. | Baoro Capital</title></head></html>",
    })


def test_parse_sitemap_posts_only() -> None:
    got = bw.parse_sitemap(sitemap({"a-post": "2026-01-02T10:00:00Z", "b": "2025-12-31"}, ("https://x.com/about",)))
    assert got == {"a-post": "2026-01-02", "b": "2025-12-31"}


def test_parse_coverage_matches_hardcoded_list() -> None:
    items = bw.parse_coverage(coverage_page(base_cards()))
    assert [c["ticker"] for c in items] == [c["ticker"] for c in AUTHOR_COVERAGE]
    assert all(c["rating"] == "OVERWEIGHT" for c in items)
    assert [c["target"] for c in items] == [c["target"] for c in AUTHOR_COVERAGE]
    assert bw.coverage_diff(items) == {"added": [], "removed": [], "target": [], "rating": []}


def test_our_ticker() -> None:
    assert bw.our_ticker("TSX", "GIB.A") == "GIB-A.TO"
    assert bw.our_ticker("TSXV", "ABC") == "ABC.V"
    assert bw.our_ticker("NYSE", "BRK.B") == "BRK-B"
    assert bw.our_ticker("NAS", "RCI") == "RCI"


def test_coverage_diff_changes() -> None:
    cards = [c for c in base_cards() if c[0] != "NYSE: RHI"]  # RHI 빠짐
    cards = [(h, "NEUTRAL", t) if h == "NYSE: IT" else (h, r, 400.0) if h == "NYSE: FDS" else (h, r, t)
             for h, r, t in cards]
    cards.append(("NYSE: XYZ", "OVERWEIGHT", 1234.5))  # 새 종목
    diff = bw.coverage_diff(bw.parse_coverage(coverage_page(cards)))
    assert [c["ticker"] for c in diff["added"]] == ["XYZ"] and diff["added"][0]["target"] == 1234.5
    assert diff["removed"] == [{"ticker": "RHI", "target": 57.90}]
    assert diff["target"] == [{"ticker": "FDS", "ours": 423.0, "now": 400.0}]
    assert diff["rating"] == [{"ticker": "IT", "before": "OVERWEIGHT", "now": "NEUTRAL"}]


def test_non_card_lines_are_ignored() -> None:
    page = "<p>NOTE: ABC</p><p>just text</p>" + coverage_page(base_cards()[:1])
    assert [c["ticker"] for c in bw.parse_coverage(page)] == ["RHI"]


def test_post_title_variants() -> None:
    assert bw.post_title('<meta property="og:title" content="A &amp; B | Baoro Capital">') == "A & B"
    assert bw.post_title('<meta content="Only Title" property="og:title"/>') == "Only Title"
    assert bw.post_title("<title>Fiserv Inc. - Baoro Research</title>") == "Fiserv Inc."
    assert bw.post_title("<html></html>") is None
    assert bw.slug_title("market-upate-oct-2026") == "Market upate oct 2026"


def test_first_run_finds_news_and_weekly_gate() -> None:
    web = web_with_news()
    out = bw.run({}, date(2026, 12, 2), get=web)
    assert out["checked"] == "2026-12-02" and out["next"] == "2026-12-09" and out["errors"] == []
    assert [p["slug"] for p in out["new_posts"]] == [NEW_SLUG]
    assert out["new_posts"][0]["title"] == "Market Update — Nov 2026"
    assert out["new_posts"][0]["url"] == bw.POST_URL + NEW_SLUG
    assert [(p["slug"], p["title"]) for p in out["edited_posts"]] == [("fiserv-inc", "Fiserv Inc.")]
    assert out["posts_total"] == len(bw.ANALYZED_POSTS) + 1  # /about 은 글이 아님
    assert out["coverage_diff"] == {"added": [], "removed": [], "target": [], "rating": []}

    # 사흘 뒤 — 받지 않는다 (부르면 실패하는 가짜로 확인), 목록은 그대로
    idle = FakeWeb({}, fail=(bw.SITEMAP_URL, bw.COVERAGE_URL))
    out2 = bw.run(json.loads(json.dumps(out)), date(2026, 12, 5), get=idle)
    assert idle.calls == [] and out2["checked"] == "2026-12-02"
    assert [p["slug"] for p in out2["new_posts"]] == [NEW_SLUG]

    # 7일 뒤 — 다시 받는다 (제목은 이미 있어 글 페이지는 안 부른다)
    web.calls.clear()
    out3 = bw.run(out2, date(2026, 12, 9), get=web)
    assert web.calls == [bw.SITEMAP_URL, bw.COVERAGE_URL] and out3["checked"] == "2026-12-09"


def test_failure_keeps_previous_and_retries_next_night() -> None:
    out = bw.run({}, date(2026, 12, 2), get=web_with_news())
    broken = web_with_news()
    broken.fail = (bw.COVERAGE_URL,)
    out2 = bw.run(out, date(2026, 12, 10), get=broken)
    assert out2["checked"] == "2026-12-02" and out2["attempted"] == "2026-12-10"  # 성공일 그대로 → 다음 밤 다시
    assert len(out2["errors"]) == 1 and out2["errors"][0].startswith("커버리지:")
    assert out2["coverage"] == out["coverage"]  # 전에 받은 커버리지 유지
    assert bw.run(out2, date(2026, 12, 11), get=web_with_news())["checked"] == "2026-12-11"


def test_unreadable_coverage_is_error_not_all_removed() -> None:
    out = bw.run({}, date(2026, 12, 2), get=web_with_news())
    changed = web_with_news()
    changed.pages[bw.COVERAGE_URL] = "<html><body><p>Coming soon</p></body></html>"  # 구조가 바뀐 페이지
    out2 = bw.run(out, date(2026, 12, 9), get=changed)
    assert any("커버리지 종목을 못 찾음" in e for e in out2["errors"])
    assert out2["coverage_diff"]["removed"] == []


def test_analyzed_list_update_applies_without_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    out = bw.run({}, date(2026, 12, 2), get=web_with_news())
    monkeypatch.setattr(bw, "ANALYZED_POSTS", {**bw.ANALYZED_POSTS, NEW_SLUG: "2026-12-01", "fiserv-inc": "2026-11-20"})
    idle = FakeWeb({})
    out2 = bw.run(out, date(2026, 12, 3), get=idle)
    assert idle.calls == [] and out2["new_posts"] == [] and out2["edited_posts"] == []


def test_notify_once_and_retry_on_failure() -> None:
    out = bw.run({}, date(2026, 12, 2), get=web_with_news())
    sent: list[tuple[str, str, str]] = []

    def ok(topic: str, title: str, body: str) -> bool:
        sent.append((topic, title, body))
        return True

    assert bw.notify(out, "", send=ok) == 0 and sent == []  # 주제 없음
    assert bw.notify(out, "t", send=lambda *_: False) == 0 and out["notified"] == []  # 실패 → 다음에 다시
    assert bw.notify(out, "t", send=ok) == 2
    assert sent[0][1] == "Baoro 블로그 — 새 소식 2건"
    assert "새 글 (12-01) Market Update — Nov 2026" in sent[0][2] and "고친 글 (11-20) Fiserv Inc." in sent[0][2]
    assert bw.notify(out, "t", send=ok) == 0 and len(sent) == 1  # 같은 소식은 한 번만


def test_main_writes_json_and_reads_prev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    web = web_with_news()
    monkeypatch.setattr(bw, "get_text", web)
    monkeypatch.setattr(bw, "today_et", lambda: date(2026, 12, 2))
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    prev_dir, out_dir = tmp_path / "prev", tmp_path / "out"
    assert bw.main(["--out-dir", str(prev_dir)]) == 0
    first = json.loads((prev_dir / bw.OUT_NAME).read_text("utf-8"))
    assert first["schema"] == bw.SCHEMA and first["checked"] == "2026-12-02"
    web.calls.clear()
    assert bw.main(["--out-dir", str(out_dir), "--prev-dir", str(prev_dir)]) == 0  # 같은 날 — 전 결과를 이어 받는다
    assert web.calls == []
    again = json.loads((out_dir / bw.OUT_NAME).read_text("utf-8"))
    assert again["new_posts"] == first["new_posts"] and again["seen"] == first["seen"]


def test_analyzed_posts_baseline() -> None:
    assert len(bw.ANALYZED_POSTS) == 44
    assert all(len(v) == 10 and v <= bw.ANALYZED_AS_OF for v in bw.ANALYZED_POSTS.values())
