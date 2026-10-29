"""SEC 공시 경고 (표시 전용) — 회사가 SEC 에 낸 공시 종류·8-K 항목으로 위험 신호를 뽑는다.

헬스체크 숫자(매출·마진)는 yfinance 를 그대로 쓴다: 2026-10-16 후보 348종목 비교에서 SEC XBRL 로 자동 계산한 매출이
크게 틀린 종목(은행 수수료만 '매출' 태그, 태그 변경으로 2년 묵은 값 등)이 있었다. SEC 의 강점은 숫자가 아니라
'무슨 공시가 언제 나왔나' — yfinance 에 없고, 나오는 즉시 공식 기록이다.

경고 (최근 공시 기준, 심각도 순):
  🔴 재무제표 재작성 8-K 4.02 (180일) · 상장폐지 경고 8-K 3.01 (180일) · 파산 신청 8-K 1.03 (365일) · 보고서 지연 NT 10-Q/10-K (90일)
     8-K 3.01 은 자진 상장 이전·폐지(다른 거래소로 옮김, 워런트 상장 종료 등)에도 쓴다 → 항목 제목을 뺀 본문에 요건 미달 말이 없고
     이전·자진 말만 있으면 🟠 '상장 이전(자진)'으로 내린다 (주주에게 중요할 수 있어 노랑이 아니라 주황)
  🟠 감사인 교체 8-K 4.01 (180일) · 자금 조달 424B1/424B4/424B5·S-1·S-3·F-1·F-3·8-K 3.02 (30일, 증자·채권 모두)
  🟡 임원 변동 8-K 5.02 (30일) · 내부자 매도 예정 Form 144 (30일) — 큰 회사는 늘 있어서 메일·시트에는 안 쓰고 웹에만
매매 판단(CCS)에는 쓰지 않는다. 경고가 수익률을 예측하는지는 검증 전 — '알고 사자' 용도.

SEC 규칙: 요청 User-Agent 에 연락처 (환경변수 SEC_USER_AGENT, CI 는 GitHub Secret), 초당 10회 이하.
SEC_USER_AGENT 가 없으면 아무것도 받지 않고 빈 결과를 돌려준다.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
# 8-K 3.01 본문 판별 — 요건 미달 말이 하나라도 있으면 경고, 없고 이전 말만 있으면 거래소 이전
DEFICIENCY_WORDS = ("deficien", "not in compliance", "noncompliance", "non-compliance", "failure to satisfy",
                    "minimum bid", "notification letter", "hearings panel", "no longer complies", "regain compliance")
TRANSFER_WORDS = ("transfer the listing", "transfer of listing", "transfer its listing", "transfer of the listing",
                  "to transfer", "dual list", "nyse texas", "voluntar", "warrant")
# 8-K 3.01 항목 제목 — 두 쪽 말이 다 들어 있어서 판별 전에 지운다
ITEM_301_TITLE = "notice of delisting or failure to satisfy a continued listing rule or standard; transfer of listing"
MIN_INTERVAL = 0.13          # 초당 10회 이하 (여유 두고 ~7.7회)
TIME_BUDGET_SEC = 240        # 한 번 실행에서 공시 받는 데 쓰는 최대 시간 — 넘으면 남은 종목은 건너뛴다
SEVERITY_ORDER = {"red": 0, "orange": 1, "yellow": 2}
ICON = {"red": "🔴", "orange": "🟠", "yellow": "🟡"}
OFFERING_FORMS = {"424B1", "424B4", "424B5", "S-1", "S-3", "F-1", "F-3"}
LATE_FORMS = {"NT 10-K", "NT 10-Q", "NT 20-F", "NT 10-K/A", "NT 10-Q/A"}


@dataclass(frozen=True)
class Rule:
    code: str
    label: str
    severity: str
    days: int
    match: Callable[[str, set[str]], bool]  # (form, 8-K 항목들) → 해당?


def _item(code: str) -> Callable[[str, set[str]], bool]:
    return lambda form, items: form in ("8-K", "8-K/A") and code in items


RULES: tuple[Rule, ...] = (
    Rule("restatement", "재무제표 재작성", "red", 180, _item("4.02")),
    Rule("delisting", "상장폐지 경고", "red", 180, _item("3.01")),
    Rule("bankruptcy", "파산 신청", "red", 365, _item("1.03")),
    Rule("late_filing", "보고서 지연", "red", 90, lambda form, items: form in LATE_FORMS),
    Rule("auditor", "감사인 교체", "orange", 180, _item("4.01")),
    Rule("offering", "자금 조달", "orange", 30,
         lambda form, items: form in OFFERING_FORMS or (form in ("8-K", "8-K/A") and "3.02" in items)),
    Rule("officer", "임원 변동", "yellow", 30, _item("5.02")),
    Rule("insider_sale_plan", "내부자 매도 예정", "yellow", 30, lambda form, items: form == "144"),
)


@dataclass
class Flag:
    code: str
    label: str
    severity: str
    date: str     # 가장 최근 공시일 YYYY-MM-DD
    form: str     # 그 공시 양식
    count: int    # 기간 안 건수
    url: str      # 가장 최근 공시 폴더 (SEC)


TRANSFER_FLAG = ("listing_transfer", "상장 이전(자진)", "orange")


def classify_301(text: str) -> str:
    """8-K 3.01 본문 → 'deficiency' | 'transfer' | 'unknown' (모르면 경고로 둔다)."""
    t = " ".join(re.sub(r"<[^>]+>", " ", html.unescape(text)).lower().split()).replace(ITEM_301_TITLE, " ")
    if any(w in t for w in DEFICIENCY_WORDS):
        return "deficiency"
    if any(w in t for w in TRANSFER_WORDS):
        return "transfer"
    return "unknown"


def flags_from_recent(recent: dict[str, list[Any]], cik: int, today: date,
                      transfer_accs: frozenset[str] = frozenset()) -> list[Flag]:
    """submissions JSON 의 filings.recent → 경고 목록 (심각도 → 최근 순). transfer_accs = 거래소 이전으로 판별한 3.01 공시."""
    dates = recent.get("filingDate", [])
    forms = recent.get("form", [])
    items_col = recent.get("items", [""] * len(dates))
    accs = recent.get("accessionNumber", [""] * len(dates))
    found: dict[str, Flag] = {}
    for d, form, items, acc in zip(dates, forms, items_col, accs):
        try:
            filed = date.fromisoformat(str(d)[:10])
        except ValueError:
            continue
        age = (today - filed).days
        if age < 0:
            continue
        item_set = {x.strip() for x in str(items or "").split(",") if x.strip()}
        for rule in RULES:
            if age > rule.days or not rule.match(str(form), item_set):
                continue
            url = FILING_URL.format(cik=cik, acc=str(acc).replace("-", "")) if acc else ""
            code, label, severity = rule.code, rule.label, rule.severity
            if code == "delisting" and acc in transfer_accs:
                code, label, severity = TRANSFER_FLAG
            cur = found.get(code)
            if cur is None:
                found[code] = Flag(code, label, severity, filed.isoformat(), str(form), 1, url)
            else:
                cur.count += 1
                if filed.isoformat() > cur.date:
                    cur.date, cur.form, cur.url = filed.isoformat(), str(form), url
    return sorted(found.values(), key=lambda f: (SEVERITY_ORDER[f.severity], -int(f.date.replace("-", ""))))


def format_flags(flags: Iterable[Flag | dict[str, Any]], min_severity: str = "orange") -> str:
    """'🔴 상장폐지 경고 10/02 · 🟠 자금 조달 9/28 (3건)' — min_severity 보다 약한 경고는 뺀다."""
    limit = SEVERITY_ORDER[min_severity]
    parts = []
    for f in flags:
        f = f if isinstance(f, dict) else asdict(f)
        if SEVERITY_ORDER[f["severity"]] > limit:
            continue
        m, d = int(f["date"][5:7]), int(f["date"][8:10])
        cnt = f" ({f['count']}건)" if f["count"] > 1 else ""
        parts.append(f"{ICON[f['severity']]} {f['label']} {m}/{d}{cnt}")
    return " · ".join(parts)


def _norm(t: Any) -> str:
    return str(t).strip().upper().replace(".", "-").replace("/", "-")


class SecClient:
    """User-Agent·속도 제한을 지키는 아주 작은 SEC 클라이언트 (실행 중 메모리 캐시만, 디스크에 안 쓴다)."""

    def __init__(self, user_agent: str | None = None, session: requests.Session | None = None) -> None:
        self.ua = (user_agent if user_agent is not None else os.environ.get("SEC_USER_AGENT", "")).strip()
        self.session = session or requests.Session()
        self._last = 0.0
        self._cik: dict[str, int] | None = None

    @property
    def enabled(self) -> bool:
        return "@" in self.ua

    def _get_json(self, url: str) -> Any | None:
        wait = MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            self._last = time.monotonic()
            try:
                r = self.session.get(url, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate"}, timeout=30)
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(2.0 * (attempt + 1))  # 429·5xx → 잠깐 쉬고 다시
        return None

    def cik_map(self) -> dict[str, int]:
        if self._cik is None:
            data = self._get_json(TICKERS_URL) or {}
            self._cik = {_norm(v["ticker"]): int(v["cik_str"]) for v in data.values()}
        return self._cik

    def get_json(self, url: str) -> Any | None:
        """JSON 하나 (속도 제한 · 재시도). 없거나 실패면 None."""
        return self._get_json(url)

    def get_bytes(self, url: str) -> bytes:
        """큰 파일 (XBRL 인스턴스 등) — 속도 제한 지킴, 한 번 다시 시도, 실패하면 b''."""
        for attempt in range(2):
            wait = MIN_INTERVAL - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return b""
            time.sleep(2.0 * (attempt + 1))
        return b""

    def _get_text(self, url: str) -> str:
        wait = MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        try:
            r = self.session.get(url, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate"}, timeout=30)
            return r.text if r.status_code == 200 else ""
        except requests.RequestException:
            return ""

    def _transfers(self, recent: dict[str, list[Any]], cik: int, today: date) -> frozenset[str]:
        """기간 안 8-K 3.01 공시 중 본문이 거래소 이전뿐인 것의 접수번호."""
        days = next(r.days for r in RULES if r.code == "delisting")
        out = set()
        rows = zip(recent.get("filingDate", []), recent.get("form", []), recent.get("items", []),
                   recent.get("accessionNumber", []), recent.get("primaryDocument", []))
        for d, form, items, acc, doc in rows:
            if form not in ("8-K", "8-K/A") or "3.01" not in str(items) or not doc:
                continue
            try:
                if (today - date.fromisoformat(str(d)[:10])).days > days:
                    continue
            except ValueError:
                continue
            text = self._get_text(DOC_URL.format(cik=cik, acc=str(acc).replace("-", ""), doc=doc))
            if text and classify_301(text) == "transfer":
                out.add(acc)
        return frozenset(out)

    def flags(self, ticker: str, today: date) -> list[Flag] | None:
        """한 종목 경고. SEC 에 없는 종목(캐나다 등)이면 None."""
        cik = self.cik_map().get(_norm(ticker))
        if cik is None:
            return None
        sub = self._get_json(SUBMISSIONS_URL.format(cik=cik))
        if not sub:
            return None
        recent = sub.get("filings", {}).get("recent", {})
        return flags_from_recent(recent, cik, today, self._transfers(recent, cik, today))


def fetch_flags(tickers: Iterable[str], today: date | None = None, client: SecClient | None = None,
                budget_sec: float = TIME_BUDGET_SEC) -> dict[str, list[Flag]]:
    """티커별 경고 (SEC 에 없는 종목은 빠진다). SEC_USER_AGENT 가 없거나 SEC 가 안 되면 빈 dict — 호출한 쪽은 그냥 계속."""
    client = client or SecClient()
    if not client.enabled:
        print("[SEC] SEC_USER_AGENT 없음 — 공시 경고 건너뜀")
        return {}
    today = today or datetime.now(timezone(timedelta(hours=-4))).date()
    out: dict[str, list[Flag]] = {}
    start = time.monotonic()
    todo = list(dict.fromkeys(_norm(t) for t in tickers if str(t).strip()))
    try:
        client.cik_map()
    except Exception as exc:  # noqa: BLE001 — 표시용이라 실패해도 계속
        print(f"[SEC] 티커 목록 실패 — 공시 경고 건너뜀: {exc}")
        return {}
    for i, t in enumerate(todo):
        if time.monotonic() - start > budget_sec:
            print(f"[SEC] 시간 한도 {budget_sec:.0f}초 — {len(todo) - i}종목 건너뜀")
            break
        try:
            got = client.flags(t, today)
        except Exception:  # noqa: BLE001
            got = None
        if got is not None:
            out[t] = got
    n_red = sum(1 for v in out.values() if any(f.severity == "red" for f in v))
    n_orange = sum(1 for v in out.values() if any(f.severity == "orange" for f in v))
    print(f"[SEC] 공시 경고: {len(out)}/{len(todo)}종목 확인 · 🔴 {n_red} · 🟠 {n_orange} ({time.monotonic() - start:.0f}초)")
    return out


def flags_json(flags: list[Flag]) -> str:
    return json.dumps([asdict(f) for f in flags], ensure_ascii=False)
