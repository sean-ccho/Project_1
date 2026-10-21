"""내계좌 30분 체크용 새 SEC 공시 (표시 전용) — 보유·관심 종목이 8-K 등을 내면 바로 알린다.

sec_filings.py 는 '최근 N일 경고 상태'(밤 리포트·웹용)를 뽑고, 여기는 '아직 알리지 않은 새 공시'를 뽑는다.
SecClient(속도 제한·User-Agent·티커→CIK)를 그대로 쓴다. SEC_USER_AGENT 가 없으면 아무것도 받지 않는다.

알리는 것 (심각도): 🔴 재무제표 재작성 4.02 · 상장 요건 3.01 · 파산 1.03 · 보고서 지연 NT
  🟠 실적 발표 2.02 · 중요 계약 1.01/1.02 · 구조조정·손상 2.05/2.06 · 증자 3.02 / 424B · 감사인 교체 4.01
  🟡 임원 변동 5.02
안 알리는 것: 주총 결과 5.07 · Reg FD 7.01 · 기타 8.01 · 첨부 9.01 (단독이면 잡음). 매매 판단에는 쓰지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Optional

from screener.sec_filings import FILING_URL, SUBMISSIONS_URL, SecClient, _norm

SINCE_DAYS = 1           # 접수일이 이 일수 안(어제부터)이고 아직 안 알린 공시 — 어제 마감 후 낸 실적도 아침 첫 실행에 잡힌다
TIME_BUDGET_SEC = 60     # 한 번 실행에서 공시 받는 데 쓰는 최대 시간
SEVERITY_ORDER = {"red": 0, "orange": 1, "yellow": 2}
ICON = {"red": "🔴", "orange": "🟠", "yellow": "🟡"}

# 8-K 항목 → (이름, 심각도)
ITEMS: dict[str, tuple[str, str]] = {
    "4.02": ("재무제표 재작성", "red"),
    "3.01": ("상장 요건 통보", "red"),
    "1.03": ("파산 신청", "red"),
    "2.02": ("실적 발표", "orange"),
    "1.01": ("중요 계약 체결", "orange"),
    "1.02": ("중요 계약 해지", "orange"),
    "2.05": ("구조조정", "orange"),
    "2.06": ("자산 손상", "orange"),
    "3.02": ("증자(비등록 지분)", "orange"),
    "4.01": ("감사인 교체", "orange"),
    "5.02": ("임원 변동", "yellow"),
}
FORMS: dict[str, tuple[str, str]] = {
    "424B1": ("증자(공모)", "orange"),
    "424B4": ("증자(공모)", "orange"),
    "424B5": ("증자(공모)", "orange"),
    "NT 10-K": ("보고서 지연", "red"),
    "NT 10-Q": ("보고서 지연", "red"),
}


@dataclass(frozen=True)
class Filing:
    """아직 알리지 않은 공시 한 건. items 는 8-K 항목 중 알릴 것만."""

    ticker: str
    form: str
    items: tuple[str, ...]
    label: str       # '실적 발표' 또는 '실적 발표 · 임원 변동'
    severity: str    # 가장 센 쪽
    acc: str         # 접수번호 (알림 중복 방지 키)
    filed: date
    url: str

    @property
    def is_earnings(self) -> bool:
        return "2.02" in self.items

    @property
    def key(self) -> str:
        return f"sec:{self.ticker}:{self.acc}"


def classify(form: str, items: str) -> Optional[tuple[tuple[str, ...], str, str]]:
    """(양식, 8-K 항목 글자) → (알릴 항목들, 이름, 심각도). 알릴 게 없으면 None."""
    f = form.strip().upper()
    if f in ("8-K", "8-K/A"):
        hit = [c for c in dict.fromkeys(x.strip() for x in str(items or "").split(",")) if c in ITEMS]
        if not hit:
            return None
        sev = min((ITEMS[c][1] for c in hit), key=SEVERITY_ORDER.__getitem__)
        return tuple(hit), " · ".join(dict.fromkeys(ITEMS[c][0] for c in hit)), sev
    base = f[:-2] if f.endswith("/A") else f
    if base in FORMS:
        label, sev = FORMS[base]
        return (), label, sev
    return None


def filings_from_recent(ticker: str, recent: dict[str, list[Any]], cik: int, today: date,
                        seen: Iterable[str] = ()) -> list[Filing]:
    """submissions JSON 의 filings.recent → 새 공시 (심각도 → 최근 순). seen = 이미 알린 접수번호."""
    seen_set = set(seen)
    since = today - timedelta(days=SINCE_DAYS)
    dates, forms = recent.get("filingDate", []), recent.get("form", [])
    items_col = recent.get("items", [""] * len(dates))
    accs = recent.get("accessionNumber", [""] * len(dates))
    out: list[Filing] = []
    for d, form, items, acc in zip(dates, forms, items_col, accs):
        try:
            filed = date.fromisoformat(str(d)[:10])
        except ValueError:
            continue
        if filed < since or filed > today or not acc or acc in seen_set:
            continue
        got = classify(str(form), str(items))
        if got is None:
            continue
        codes, label, sev = got
        url = FILING_URL.format(cik=cik, acc=str(acc).replace("-", ""))
        out.append(Filing(ticker, str(form), codes, label, sev, str(acc), filed, url))
    out.sort(key=lambda f: (SEVERITY_ORDER[f.severity], -f.filed.toordinal()))
    return out


def fetch_new_filings(tickers: Iterable[str], today: date, seen: Iterable[str] = (),
                      client: Optional[SecClient] = None, budget_sec: float = TIME_BUDGET_SEC) -> list[Filing]:
    """티커들의 새 공시. SEC_USER_AGENT 가 없거나 SEC 가 안 되면 빈 목록 — 호출한 쪽은 그냥 계속한다. throw 하지 않는다."""
    client = client or SecClient()
    if not client.enabled:
        print("[SEC] SEC_USER_AGENT 없음 — 새 공시 확인 건너뜀")
        return []
    seen = set(seen)
    out: list[Filing] = []
    start = time.monotonic()
    try:
        cik_map = client.cik_map()
        todo = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip() and str(t).strip().upper() != "SPY"))
        checked = 0
        for t in todo:
            if time.monotonic() - start > budget_sec:
                print(f"[SEC] 시간 한도 {budget_sec:.0f}초 — {len(todo) - checked}종목 건너뜀")
                break
            cik = cik_map.get(_norm(t))
            if cik is None:  # 캐나다 등 SEC 에 없는 종목
                continue
            sub = client._get_json(SUBMISSIONS_URL.format(cik=cik))  # noqa: SLF001 — 같은 패키지의 작은 클라이언트
            checked += 1
            if sub:
                out += filings_from_recent(t, sub.get("filings", {}).get("recent", {}), cik, today, seen)
        print(f"[SEC] 새 공시 확인: {checked}/{len(todo)}종목 · {len(out)}건 ({time.monotonic() - start:.0f}초)")
    except Exception as exc:  # noqa: BLE001 — 표시용이라 실패해도 계속
        print(f"[SEC] 새 공시 확인 실패: {type(exc).__name__}: {exc}")
    out.sort(key=lambda f: (SEVERITY_ORDER[f.severity], -f.filed.toordinal()))
    return out
