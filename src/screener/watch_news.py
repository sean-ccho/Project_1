"""내계좌 30분 체크용 종목 헤드라인 (표시 전용) — 구글 뉴스 RSS, 키 없음.

사이트 오늘 페이지(personal-site lib/news.ts)의 종목 뉴스 규칙을 파이썬으로 옮긴 것: 회사 이름 검색 3일 →
그 종목 기사만 → 잡음(13F 지분·ETF 편입·시세 페이지·집단소송 광고) 빼고 → 같은 사건 합치기 → 매체 등급·최신순.
알림에 '왜 움직였나' 한 줄을 붙이는 용도라서 영문 원문 제목 그대로 쓴다 (번역은 사이트에만 있다).
실패하면 빈 목록 — 호출한 쪽은 그냥 계속한다 (구글이 서버 IP 를 막거나 동의 페이지를 줄 수 있다).
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote_plus

import requests

RSS_URL = "https://news.google.com/rss/search?q={q}&{edition}"
EDITIONS = {"us": "hl=en-US&gl=US&ceid=US:en", "ca": "hl=en-CA&gl=CA&ceid=CA:en"}
TIMEOUT_SEC = 8
FRESH_HOURS = 24
_UA = {"User-Agent": "Mozilla/5.0 (compatible; account-watch)"}

# 자동 생성 기사만 내는 매체 (MarketBeat 계열 13F 기사, 독일어 모음 사이트, 자동 밸류에이션 글)
BLOCK_SOURCE = re.compile(
    r"marketbeat|ad hoc news|ad-hoc-news|stock traders daily|etf daily news|defense world|ticker report|"
    r"american banking news|lincolnian|wkrb|zolmax|modern readers|cerbat gem|daily political|markets daily|"
    r"transcript daily|the am reporter|watchlist news|simply wall st|kalkine", re.I)
BLOCK_TITLE = re.compile("|".join([
    # 13F: '… Asset Management LLC Purchases New Stake in …' (운용사 이름이 앞에 있을 때만 — 'Nvidia takes stake in Intel' 은 남긴다)
    r"\b(LLC|L\.L\.C|LP|Management|Advisors?|Advisory|Capital|Partners|Wealth|Investments?|Securities|Asset|Fund|Trust|Bank|Group)\b"
    r"[^.]{0,60}\b(purchases|acquires|buys|sells|trims|raises|lowers|boosts|cuts|increases|reduces|grows|adds to|initiates|"
    r"establishes|takes|has|holds)\b[^.]{0,30}\b(stake|position|holdings) in\b",
    r"\b(buys|sells|acquires|purchases)\s+[\d,.]+\s+shares of\b",
    r"\b(shares|stake|position) (bought|sold|acquired|purchased) by\b",
    # ETF 자금 흐름
    r"\bETFs? (add|hold|own|buy)\b",
    r"\b(buys|sells|adds|trims)\s+\$[\d.,]+\s*(million|billion|thousand|[kmb])?\s+(more|worth)\b",
    # 시세·예측 페이지, 공시 서식 자동 기사
    r"stock price, news, quote|quote (and|&) history|stock quote\b|\bstock (forecasts?|price predictions?)$|\boptions? chain\b",
    r"^form [0-9a-z-]+ .*\bfor:",
    # 로펌 집단소송 광고
    r"class action|investor alert|shareholder alert|\bdeadline\b|investigation on behalf|securities fraud|"
    r"\b(rosen|pomerantz|levi & korsinsky|bragar|faruqi|kessler topaz|bronstein|glancy|schall|gross law|portnoy|"
    r"robbins geller|johnson fistel)\b",
]), re.I)

# 매체 등급 — 같은 사건이면 높은 쪽을 대표로, 목록도 높은 쪽 먼저
TIER1 = re.compile(r"reuters|bloomberg|wsj|wall street journal|cnbc|barron|financial times|associated press|ap news|"
                   r"marketwatch|investor's business|new york times|fox business|cnn|axios|business insider|"
                   r"globe and mail|financial post|bnn", re.I)
TIER2 = re.compile(r"yahoo finance|seeking alpha|benzinga|investing\.com|motley fool|zacks|tipranks|nasdaq|morningstar|"
                   r"forbes|barchart|thestreet|investopedia|kiplinger|stockstory|business wire|pr newswire|globenewswire|"
                   r"newsfile|accesswire|investing news network|proactive|mining\.com|canadian manufacturing", re.I)
STOP = set("the a an to of in on for and or as at by with after from its is are be this that stock stocks shares inc corp "
           "why what how here new says said will".split())
LINK_WORDS = {"of", "and", "&", "for", "the", "de"}
_SUFFIX = re.compile(r"[,\s]+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|l\.p|n\.v|s\.a|ag|se|class [a-c])\.?$", re.I)


@dataclass
class Headline:
    """기사 한 건 (제목은 매체 꼬리를 뗀 영문 원문)."""

    title: str
    source: str
    url: str
    time: Optional[datetime]

    def age_text(self, now: Optional[datetime] = None) -> str:
        """'3시간 전' · '어제' 같은 짧은 시각 (모르면 빈 글자)."""
        if self.time is None:
            return ""
        hours = ((now or datetime.now(timezone.utc)) - self.time).total_seconds() / 3600
        if hours < 1:
            return "방금"
        return f"{int(hours)}시간 전" if hours < 24 else f"{int(hours // 24)}일 전"


def _tier(source: str) -> int:
    return 0 if TIER1.search(source) else 1 if TIER2.search(source) else 2


def _words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$%.]+", title.lower()) if len(w) > 2 and w not in STOP}


def _same_event(a: set[str], b: set[str]) -> bool:
    """겹치는 낱말 3개 이상이고 짧은 쪽의 절반 이상."""
    n = len(a & b)
    return n >= 3 and bool(a) and bool(b) and n / min(len(a), len(b)) >= 0.5


def clean_name(name: Optional[str]) -> Optional[str]:
    """'Example, Inc.' → 'Example', 'The Foo Corporation (Class A)' → 'Foo'. 비면 None."""
    if not name:
        return None
    s = re.sub(r"\s*\([^)]*\)", "", name)
    s = re.sub(r"^the\s+", "", s, flags=re.I).strip()
    for _ in range(2):
        s = _SUFFIX.sub("", s).strip()
    return s or None


def search_name(clean: str) -> str:
    """세 낱말 이상이면 앞 두 낱말 (긴 정식 이름은 기사 제목에 잘 안 나온다). 'Bank of X' 는 세 낱말."""
    w = clean.split()
    if len(w) <= 2:
        return clean
    return " ".join(w[: 3 if w[1].lower() in LINK_WORDS else 2])


def keyword(clean: Optional[str]) -> Optional[str]:
    """제목에서 그 회사를 찾을 낱말 — 첫 낱말이 짧으면('Old Dominion' 의 'old') 두 낱말."""
    if not clean:
        return None
    w = clean.split()
    return (w[0] if len(w[0]) >= 4 else " ".join(w[:2])).lower()


def build_query(name: Optional[str], ticker: str) -> str:
    """구글 뉴스 검색어 (기간 제한 제외). 이름이 티커처럼 짧은 대문자면 법인 꼬리까지 붙은 전체 이름으로."""
    base = ticker.upper().split(".")[0]
    clean = clean_name(name)
    if not clean:
        return f"{base} stock"
    if re.fullmatch(r"[A-Z0-9&.-]{1,4}", clean) and name:
        return f'"{name}"'
    return f'"{search_name(clean)}"'


def _plain(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(s or ""))).strip()


def parse_rss(xml_text: str) -> list[Headline]:
    """RSS 글자 → 기사 목록. RSS 가 아니면 빈 목록. 링크는 구글 뉴스 주소만 받는다 (피드 글자가 그대로 링크가 되니까)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[Headline] = []
    for item in root.iter("item"):
        source = _plain(item.findtext("source") or "")
        title = _plain(item.findtext("title") or "")
        if source and title.endswith(f" - {source}"):
            title = title[: -(len(source) + 3)].strip()
        url = (item.findtext("link") or "").strip()
        try:
            when = parsedate_to_datetime(item.findtext("pubDate") or "")
            when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            when = None
        if title and url.startswith("https://news.google.com/"):
            out.append(Headline(title, source, url, when))
    return out


def pick(items: list[Headline], ticker: str, name: Optional[str], n: int = 2,
         now: Optional[datetime] = None) -> list[Headline]:
    """그 종목 기사만 → 잡음 빼기 → 같은 사건 합치기 → 최근 하루 먼저, 그 안에서 매체 등급·최신순."""
    base = ticker.upper().split(".")[0]
    ticker_re = re.compile(rf"\b{re.escape(base)}\b")
    key = keyword(clean_name(name))
    now = now or datetime.now(timezone.utc)
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    kept = [x for x in items
            if (ticker_re.search(x.title) or (key is not None and key in x.title.lower()))
            and not BLOCK_SOURCE.search(x.source) and not BLOCK_TITLE.search(x.title)]
    kept.sort(key=lambda x: (_tier(x.source), -(x.time or epoch).timestamp()))
    groups: list[tuple[Headline, set[str]]] = []
    for x in kept:
        w = _words(x.title)
        if not any(g.title == x.title or _same_event(gw, w) for g, gw in groups):
            groups.append((x, w))
    fresh = lambda x: 0 if x.time and now - x.time <= timedelta(hours=FRESH_HOURS) else 1  # noqa: E731
    leads = sorted((g for g, _ in groups), key=lambda x: (fresh(x), _tier(x.source), -(x.time or epoch).timestamp()))
    return leads[:n]


def fetch_rss(query: str, cad: bool = False, session: Optional[requests.Session] = None) -> list[Headline]:
    """구글 뉴스 RSS 한 번. 실패·차단·RSS 아님이면 로그 한 줄 + 빈 목록 (검색어는 로그에 남기지 않는다)."""
    url = RSS_URL.format(q=quote_plus(query), edition=EDITIONS["ca" if cad else "us"])
    try:
        resp = (session or requests).get(url, headers=_UA, timeout=TIMEOUT_SEC)
    except requests.RequestException as exc:
        print(f"[뉴스] 요청 실패: {type(exc).__name__}")
        return []
    if resp.status_code != 200 or "<rss" not in resp.text:
        print(f"[뉴스] 응답 {resp.status_code} — RSS 아님 (차단·동의 페이지일 수 있음)")
        return []
    return parse_rss(resp.text)


def headlines(name: Optional[str], ticker: str, cad: bool = False, n: int = 2,
              session: Optional[requests.Session] = None) -> list[Headline]:
    """한 종목의 최근 3일 헤드라인 최대 n건. 절대 throw 하지 않는다."""
    try:
        items = fetch_rss(f"{build_query(name, ticker)} when:3d", cad, session)
        return pick(items, ticker, name, n)
    except Exception as exc:  # noqa: BLE001 — 표시용이라 실패해도 계속
        print(f"[뉴스] {ticker} 실패: {type(exc).__name__}")
        return []
