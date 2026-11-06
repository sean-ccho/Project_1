"""주요 일정 — 오늘 탭 '이번 주 일정' · '다가오는 큰 일정' · 시트 [경제] (macro.json events, 2026-11-06, 표시 전용).

점수 · 필터 · PT-1 에 쓰지 않는다. screener.macro_scan.build 가 밤마다 부른다.

일정 한 건 = {date, time(ET, 없으면 None), key, kind, country(US · CA), tier(★ 1~3), ko, en, note_ko, note_en, impact, src}
- kind: rate(금리) · data(경제지표) · policy(정치 · 정책) · market(옵션 만기 · 지수 재구성) · holiday(휴장 · 조기 폐장)
- tier: 3 = 시장 전체를 흔드는 날, 2 = 자주 흔드는 날, 1 = 알아 두면 좋은 날
- impact = '평소 대비 변동': 그 일정 날(선거는 다음 거래일) 지수 하루 수익률 절대값 평균 ÷ 같은 기간 모든 날 평균.
  미국 일정은 SPY, 캐나다 일정은 XIC.TO. 지난 일정이 MIN_IMPACT_N 번보다 적으면 None.
  (2021-01~2026-11 확인: FOMC 결정일 0.96% ÷ 평소 0.75% = 1.3배 — 매매 신호가 아니라 '이날은 크게 움직이곤 했다'는 참고)

자료 (2026-11-06 확인):
- FOMC · BoC: data.rate_calendar (+ 밤마다 공식 페이지에서 새 해). FOMC 14:00, 의사록 = 결정 21일 뒤 14:00, BoC 09:45 ET
- 미국 지표 날짜: FRED 발표 달력 HTML (키 없이 올해 · 내년 — 시각은 미국 중부라 ET 로 +1시간). 과거 날짜(영향 계산)는
  FRED API 만 주는데 키가 필요하다 — 환경변수 FRED_API_KEY 가 있을 때만 CPI · 고용 등 배지를 붙인다. BLS 는 로봇을 막는다(403)
- 캐나다 지표: StatCan 주요 지표 일정 JSON (2012년부터 앞으로 몇 달까지 — 영향 계산까지 키 없이)
- 옵션 만기 · 쿼드러플 위칭: 셋째 금요일 (NYSE 휴장이면 전 거래일), 3 · 6 · 9 · 12월 = S&P 500 분기 리밸런싱 적용
- 휴장 · 조기 폐장 · 정치 · 정책 · 러셀: 공식 발표에서 옮긴 목록 (항목마다 출처). 목록이 내년을 덮지 못하면 errors 에 경고
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

# key → (kind, country, tier, 한국어, 영어, 시각 ET)
SPECS: dict[str, tuple[str, str, int, str, str, str | None]] = {
    "fomc": ("rate", "US", 3, "FOMC 금리 결정", "FOMC rate decision", "14:00"),
    "fomc_minutes": ("rate", "US", 2, "FOMC 의사록", "FOMC minutes", "14:00"),
    "boc": ("rate", "CA", 3, "BoC 금리 결정", "Bank of Canada decision", "09:45"),
    "cpi": ("data", "US", 3, "미국 CPI (소비자물가)", "US CPI", "08:30"),
    "jobs": ("data", "US", 3, "미국 고용보고서", "US jobs report", "08:30"),
    "pce": ("data", "US", 2, "미국 PCE 물가 · 소득 · 지출", "US PCE · income & spending", "08:30"),
    "gdp": ("data", "US", 2, "미국 GDP", "US GDP", "08:30"),
    "retail": ("data", "US", 2, "미국 소매판매", "US retail sales", "08:30"),
    "ppi": ("data", "US", 1, "미국 PPI (생산자물가)", "US PPI", "08:30"),
    "ca_cpi": ("data", "CA", 2, "캐나다 CPI", "Canada CPI", "08:30"),
    "ca_jobs": ("data", "CA", 2, "캐나다 고용 (LFS)", "Canada jobs (LFS)", "08:30"),
    "opex": ("market", "US", 1, "월간 옵션 만기", "Monthly options expiry", None),
    "quad": ("market", "US", 2, "쿼드러플 위칭 · S&P 500 분기 리밸런싱", "Quad witching · S&P 500 rebalance", None),
    "russell": ("market", "US", 2, "러셀 지수 재구성", "Russell reconstitution", None),
    "holiday_us": ("holiday", "US", 1, "미국 휴장", "US market closed", None),
    "early_us": ("holiday", "US", 1, "미국 조기 폐장", "US early close", "13:00"),
    "holiday_ca": ("holiday", "CA", 1, "캐나다 휴장", "Canada market closed", None),
    "early_ca": ("holiday", "CA", 1, "캐나다 조기 폐장", "Canada early close", "13:00"),
    "election_us": ("policy", "US", 3, "미국 선거", "US election", None),
    "election_ca": ("policy", "CA", 3, "캐나다 총선", "Canada federal election", None),
    "funding_us": ("policy", "US", 3, "미국 정부 예산 기한 (셧다운 위험)", "US government funding deadline", None),
}

FRED_RELEASES = {"cpi": 10, "jobs": 50, "pce": 54, "gdp": 53, "retail": 9, "ppi": 46}
FRED_CAL_URL = "https://fred.stlouisfed.org/releases/calendar?rid={rid}&y={year}"
FRED_API_URL = ("https://api.stlouisfed.org/fred/release/dates?release_id={rid}&realtime_start=1990-01-01"
                "&realtime_end=9999-12-31&include_release_dates_with_no_data=false&sort_order=asc&limit=10000"
                "&file_type=json&api_key={key}")
STATCAN_URL = "https://www150.statcan.gc.ca/n1/dai-quo/ssi/homepage/schedule-key_indicators-eng.json"
STATCAN_PAGE = "https://www150.statcan.gc.ca/n1/dai-quo/cal1-eng.htm"
STATCAN_TITLES = {"Consumer Price Index": "ca_cpi", "Labour Force Survey": "ca_jobs"}
FOMC_SRC = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BOC_SRC = "https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/"
MINUTES_LAG_DAYS = 21          # 연준은 결정 3주 뒤 수요일 14:00 에 의사록을 낸다

PAST_DAYS = 7                  # 지난 일정도 이만큼은 남긴다 (오늘 · 어제 칸)
AHEAD_DAYS = 400               # 앞으로 이만큼 (정치 일정은 POLICY_AHEAD_YEARS)
POLICY_AHEAD_YEARS = 5
MIN_IMPACT_N = 8               # 지난 일정이 이보다 적으면 '평소 대비 변동'을 내지 않는다
IMPACT_START = "1993-01-01"    # SPY 상장(1993-01) — 캐나다 XIC.TO 는 2001년부터

# NYSE — ICE 보도자료 2025-12-23 'NYSE Group Announces 2026, 2027 and 2028 Holiday and Early Closings Calendar'
NYSE_SRC = ("https://ir.theice.com/press/news-details/2025/"
            "NYSE-Group-Announces-2026-2027-and-2028-Holiday-and-Early-Closings-Calendar/default.aspx")
NYSE_CLOSED: list[tuple[str, str, str]] = [
    ("2026-01-01", "새해 첫날", "New Year's Day"), ("2026-01-19", "마틴 루서 킹 데이", "Martin Luther King Jr. Day"),
    ("2026-02-16", "대통령의 날", "Washington's Birthday"), ("2026-04-03", "성금요일", "Good Friday"),
    ("2026-05-25", "메모리얼 데이", "Memorial Day"), ("2026-06-19", "준틴스", "Juneteenth"),
    ("2026-07-03", "독립기념일 (대체)", "Independence Day (observed)"), ("2026-09-07", "노동절", "Labor Day"),
    ("2026-11-26", "추수감사절", "Thanksgiving Day"), ("2026-12-25", "크리스마스", "Christmas Day"),
    ("2027-01-01", "새해 첫날", "New Year's Day"), ("2027-01-18", "마틴 루서 킹 데이", "Martin Luther King Jr. Day"),
    ("2027-02-15", "대통령의 날", "Washington's Birthday"), ("2027-03-26", "성금요일", "Good Friday"),
    ("2027-05-31", "메모리얼 데이", "Memorial Day"), ("2027-06-18", "준틴스 (대체)", "Juneteenth (observed)"),
    ("2027-07-05", "독립기념일 (대체)", "Independence Day (observed)"), ("2027-09-06", "노동절", "Labor Day"),
    ("2027-11-25", "추수감사절", "Thanksgiving Day"), ("2027-12-24", "크리스마스 (대체)", "Christmas Day (observed)"),
    # 2028 — 새해 첫날은 토요일이라 쉬는 날이 없다 (보도자료 각주)
    ("2028-01-17", "마틴 루서 킹 데이", "Martin Luther King Jr. Day"), ("2028-02-21", "대통령의 날", "Washington's Birthday"),
    ("2028-04-14", "성금요일", "Good Friday"), ("2028-05-29", "메모리얼 데이", "Memorial Day"),
    ("2028-06-19", "준틴스", "Juneteenth"), ("2028-07-04", "독립기념일", "Independence Day"),
    ("2028-09-04", "노동절", "Labor Day"), ("2028-11-23", "추수감사절", "Thanksgiving Day"),
    ("2028-12-25", "크리스마스", "Christmas Day"),
]
NYSE_EARLY: list[tuple[str, str, str]] = [
    ("2026-11-27", "추수감사절 다음 날", "Day after Thanksgiving"), ("2026-12-24", "크리스마스 이브", "Christmas Eve"),
    ("2027-11-26", "추수감사절 다음 날", "Day after Thanksgiving"),
    ("2028-07-03", "독립기념일 전날", "Day before Independence Day"), ("2028-11-24", "추수감사절 다음 날", "Day after Thanksgiving"),
]
NYSE_LAST_YEAR = 2028
# TSX · TSXV — TMX 거래 달력 (tsx.com, 2026-11-06 확인 · 2028 은 아직 없음)
TSX_SRC = "https://www.tsx.com/en/trading/calendars-and-trading-hours/calendar"
TSX_CLOSED: list[tuple[str, str, str]] = [
    ("2026-01-01", "새해 첫날", "New Year's Day"), ("2026-02-16", "가족의 날", "Family Day"),
    ("2026-04-03", "성금요일", "Good Friday"), ("2026-05-18", "빅토리아 데이", "Victoria Day"),
    ("2026-07-01", "캐나다 데이", "Canada Day"), ("2026-08-03", "시민의 날", "Civic Holiday"),
    ("2026-09-07", "노동절", "Labour Day"), ("2026-10-12", "추수감사절", "Thanksgiving Day"),
    ("2026-12-25", "크리스마스", "Christmas Day"), ("2026-12-28", "박싱 데이 (대체)", "Boxing Day (in lieu)"),
    ("2027-01-01", "새해 첫날", "New Year's Day"), ("2027-02-15", "가족의 날", "Family Day"),
    ("2027-03-26", "성금요일", "Good Friday"), ("2027-05-24", "빅토리아 데이", "Victoria Day"),
    ("2027-07-01", "캐나다 데이", "Canada Day"), ("2027-08-02", "시민의 날", "Civic Holiday"),
    ("2027-09-06", "노동절", "Labour Day"), ("2027-10-11", "추수감사절", "Thanksgiving Day"),
    ("2027-12-27", "크리스마스 (대체)", "Christmas Day (in lieu)"), ("2027-12-28", "박싱 데이 (대체)", "Boxing Day (in lieu)"),
]
TSX_EARLY: list[tuple[str, str, str]] = [
    ("2026-12-24", "크리스마스 이브", "Christmas Eve"), ("2027-12-24", "크리스마스 이브", "Christmas Eve"),
]
TSX_LAST_YEAR = 2027

# 정치 · 정책 · 지수 — 날짜를 확인한 것만 (확인 못 한 것은 넣지 않는다)
FUNDING_SRC = ("https://www.naco.org/news/congress-passes-short-term-funding-bill-keeping-federal-government-funded-"
               "through-dec-11-and")
RUSSELL_SRC = ("https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2026/"
               "ftse-russell-announces-december-2026-russell-us-indexes-reconstitution-schedule")
MANUAL_EVENTS: list[dict[str, str]] = [
    {"date": "2026-12-11", "key": "funding_us",
     "note_ko": "임시 예산(CR, 9-2 서명)이 끝나는 날 — 새 예산이나 연장이 없으면 정부 일부가 멈춘다",
     "note_en": "Stopgap funding (CR signed Sep 2) runs out — parts of the government shut down without a new bill",
     "src": FUNDING_SRC},
    {"date": "2026-12-11", "key": "russell",
     "note_ko": "6월에 이어 올해 두 번째 — 장 마감 뒤 적용, 12-14 월요일부터 새 구성",
     "note_en": "Second of the year (now twice yearly) — effective after the close, new indexes from Mon Dec 14",
     "src": RUSSELL_SRC},
]
# 미국 연방 선거 = 짝수 해 11월 첫 월요일 다음 화요일 (2 U.S.C. §7), 결과는 밤에 나와 다음 거래일에 반영
US_ELECTION_SRC = "https://www.law.cornell.edu/uscode/text/2/7"
# 캐나다 총선 — 고정일 = 지난 총선 4년째 해 10월 셋째 월요일 (Canada Elections Act 56.1, 더 일찍 치를 수 있다)
CA_ELECTION_SRC = "https://laws-lois.justice.gc.ca/eng/acts/E-2.01/"
CA_ELECTIONS_PAST = ["2004-06-28", "2006-01-23", "2008-10-14", "2011-05-02", "2015-10-19", "2019-10-21",
                     "2021-09-20", "2025-04-28"]
CA_ELECTION_NEXT = "2029-10-15"

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
     "December"], start=1)}
_MONTHS_KO = {v: f"{v}월" for v in _MONTHS.values()}


# ── 날짜 규칙 ────────────────────────────────────────────────────────────────

def third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def us_election_day(year: int) -> date:
    """11월 첫 월요일 다음 화요일."""
    d = date(year, 11, 1)
    first_monday = d + timedelta(days=(0 - d.weekday()) % 7)
    return first_monday + timedelta(days=1)


def us_election_days(start_year: int, end_year: int) -> list[date]:
    return [us_election_day(y) for y in range(start_year + start_year % 2, end_year + 1, 2)]


def nyse_closed() -> set[date]:
    return {date.fromisoformat(d) for d, _k, _e in NYSE_CLOSED}


def opex_days(start: date, end: date, is_open: Callable[[date], bool]) -> list[tuple[date, bool]]:
    """[start, end] 의 옵션 만기일 (셋째 금요일, 휴장이면 전 거래일) → [(날짜, 쿼드러플 위칭인가)]."""
    out: list[tuple[date, bool]] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        d = third_friday(y, m)
        for _ in range(5):
            if is_open(d):
                break
            d -= timedelta(days=1)
        if start <= d <= end:
            out.append((d, m in (3, 6, 9, 12)))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ── 받은 자료 읽기 ───────────────────────────────────────────────────────────

_FRED_DATE = re.compile(r'font-weight:\s*bold;">\s*(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\s+([A-Za-z]+)\s+'
                        r'(\d{1,2}),\s+(\d{4})')
_FRED_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([ap]m)")


def parse_fred_calendar(page: str, year: int) -> list[tuple[date, str | None]]:
    """FRED 발표 달력 HTML → [(날짜, 'HH:MM' ET | None)]. 날짜 줄마다 그 아래 첫 시각 (다음 날짜 줄 전까지) — 페이지 시각은
    미국 중부라 +1시간. 발표가 끝난 줄에는 'Updated' 표시가 붙어도 같은 줄이다. 그 해 날짜만 (없는 해를 물으면 FRED 가
    다른 해를 보여 주기도 해서)."""
    marks = list(_FRED_DATE.finditer(page))
    out: list[tuple[date, str | None]] = []
    for i, m in enumerate(marks):
        mon, day, yr = m.groups()
        if int(yr) != year or mon not in _MONTHS:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(page)
        t = _FRED_TIME.search(page, m.end(), end)
        tm = None
        if t:
            h = int(t.group(1)) % 12 + (12 if t.group(3) == "pm" else 0) + 1
            tm = f"{h:02d}:{t.group(2)}"
        out.append((date(int(yr), _MONTHS[mon], int(day)), tm))
    return sorted(set(out), key=lambda x: (x[0], x[1] or ""))


def parse_fred_api_dates(raw: bytes | str) -> list[date]:
    """FRED API release/dates JSON → 발표일 목록."""
    d = json.loads(raw)
    return sorted({date.fromisoformat(x["date"]) for x in d.get("release_dates", []) if x.get("date")})


def parse_statcan(raw: bytes | str) -> dict[str, list[tuple[date, str]]]:
    """StatCan 주요 지표 일정 JSON → {key: [(날짜, 대상 기간 'October 2026')]}."""
    out: dict[str, list[tuple[date, str]]] = {k: [] for k in STATCAN_TITLES.values()}
    for e in json.loads(raw):
        key = STATCAN_TITLES.get(str(e.get("title", "")).strip())
        day = str(e.get("date", ""))[:10]
        if key and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            out[key].append((date.fromisoformat(day), str(e.get("description", "")).strip()))
    return {k: sorted(set(v)) for k, v in out.items()}


def period_ko(desc: str) -> str:
    """'October 2026' → '2026년 10월분' (모르는 꼴은 그대로)."""
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", desc.strip())
    if not m or m.group(1) not in _MONTHS:
        return desc
    return f"{m.group(2)}년 {_MONTHS_KO[_MONTHS[m.group(1)]]}분"


# ── 평소 대비 변동 ───────────────────────────────────────────────────────────

def impact(days: Iterable[date], close: pd.Series | None, next_day: bool = False,
           min_n: int = MIN_IMPACT_N) -> dict[str, Any] | None:
    """일정 날 |하루 수익률| 평균 ÷ 같은 기간(첫 일정 날부터) 모든 날 평균. next_day = 다음 거래일 (밤에 결과가 나오는 선거).
    그날 장이 안 열렸으면 그 일정은 뺀다. 지난 일정이 min_n 번보다 적으면 None."""
    if close is None:
        return None
    s = close.dropna()
    s = s[~s.index.duplicated(keep="last")]
    r = s.pct_change().abs().dropna()
    if r.empty:
        return None
    idx = pd.DatetimeIndex(r.index).normalize()
    r.index = idx
    picked: list[pd.Timestamp] = []
    for d in sorted(set(days)):
        ts = pd.Timestamp(d)
        if ts < idx[0]:
            continue                      # 가격이 시작하기 전 일정 (첫날로 몰리지 않게)
        pos = idx.searchsorted(ts, side="right" if next_day else "left")
        if pos >= len(idx) or (not next_day and idx[pos] != ts):
            continue
        picked.append(idx[pos])
    picked = sorted(set(picked))
    if len(picked) < min_n:
        return None
    base = float(r[r.index >= picked[0]].mean())
    ev = float(r.loc[picked].mean())
    if not base > 0:
        return None
    return {"x": round(ev / base, 2), "n": len(picked), "since": picked[0].year, "avg": round(ev, 4),
            "base": round(base, 4)}


# ── 모으기 ───────────────────────────────────────────────────────────────────

def _event(day: date, key: str, *, time: str | None = None, ko: str | None = None, en: str | None = None,
           note_ko: str = "", note_en: str = "", src: str | None = None) -> dict[str, Any]:
    kind, country, tier, k_ko, k_en, k_time = SPECS[key]
    return {"date": day.isoformat(), "time": time if time is not None else k_time, "key": key, "kind": kind,
            "country": country, "tier": tier, "ko": ko or k_ko, "en": en or k_en, "note_ko": note_ko,
            "note_en": note_en, "impact": None, "src": src}


def build_events(today: date, fomc: list[tuple[str, str, bool]], boc: list[tuple[str, bool]], *,
                 get: Callable[[str], bytes] | None = None, prices: pd.DataFrame | None = None,
                 fred_key: str | None = None, errors: list[str] | None = None) -> list[dict[str, Any]]:
    """오늘 기준 −PAST_DAYS ~ +AHEAD_DAYS 일정 (정치 일정은 POLICY_AHEAD_YEARS 년) → 날짜 · 시각 · 중요도 순 목록.

    fomc = [(회의 시작일, 결정일, 경제 전망)], boc = [(발표일, MPR)] — macro_scan 이 공식 페이지까지 합친 목록.
    get(url) → bytes (없으면 받는 일정 없이 목록 · 규칙만), prices = SPY · XIC.TO 종가 (평소 대비 변동, 없으면 None).
    하나를 못 받아도 나머지는 싣고 errors 에 적는다.
    """
    errors = errors if errors is not None else []
    lo, hi = today - timedelta(days=PAST_DAYS), today + timedelta(days=AHEAD_DAYS)
    policy_hi = date(today.year + POLICY_AHEAD_YEARS, 12, 31)
    spy = prices["SPY"] if prices is not None and "SPY" in prices else None
    xic = prices["XIC.TO"] if prices is not None and "XIC.TO" in prices else None
    hist: dict[str, list[date]] = {}            # 영향 계산용 지난 날짜
    events: list[dict[str, Any]] = []

    def add(ev: dict[str, Any], until: date = hi) -> None:
        if lo.isoformat() <= ev["date"] <= until.isoformat():
            events.append(ev)

    # 금리
    for _start, day, sep in fomc:
        d = date.fromisoformat(day)
        add(_event(d, "fomc", note_ko="경제 전망(점도표) 같이" if sep else "",
                   note_en="with projections (dot plot)" if sep else "", src=FOMC_SRC))
        add(_event(d + timedelta(days=MINUTES_LAG_DAYS), "fomc_minutes",
                   note_ko=f"{d.month}/{d.day} 회의", note_en=f"{d:%b} {d.day} meeting", src=FOMC_SRC))
    hist["fomc"] = [date.fromisoformat(d) for _s, d, _p in fomc]
    hist["fomc_minutes"] = [date.fromisoformat(d) + timedelta(days=MINUTES_LAG_DAYS) for _s, d, _p in fomc]
    for day, mpr in boc:
        add(_event(date.fromisoformat(day), "boc", note_ko="통화정책보고서(MPR) 같이" if mpr else "",
                   note_en="with Monetary Policy Report" if mpr else "", src=BOC_SRC))
    hist["boc"] = [date.fromisoformat(d) for d, _m in boc]

    # 미국 지표 — FRED 달력 (올해 · 내년)
    if get is not None:
        for key, rid in FRED_RELEASES.items():
            got: list[tuple[date, str | None]] = []
            for year in (today.year, today.year + 1):
                try:
                    got += parse_fred_calendar(get(FRED_CAL_URL.format(rid=rid, year=year)).decode("utf-8", "replace"),
                                               year)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"FRED 달력 {key} {year}: {str(exc)[:120]}")
            if not any(d >= today for d, _t in got):
                errors.append(f"FRED 달력 {key}: 앞으로의 발표일이 없음")
            for d, t in got:
                add(_event(d, key, time=t, src=FRED_CAL_URL.format(rid=rid, year=d.year)))
            if fred_key:
                try:
                    hist[key] = parse_fred_api_dates(get(FRED_API_URL.format(rid=rid, key=fred_key)))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"FRED API {key}: {str(exc)[:120]}")
        # 캐나다 지표 — StatCan
        try:
            sc = parse_statcan(get(STATCAN_URL))
            for key, rows in sc.items():
                for d, desc in rows:
                    add(_event(d, key, note_ko=period_ko(desc), note_en=desc, src=STATCAN_PAGE))
                hist[key] = [d for d, _x in rows]
                if not any(d >= today for d, _x in rows):
                    errors.append(f"StatCan {key}: 앞으로의 발표일이 없음")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"StatCan 일정: {str(exc)[:120]}")

    # 옵션 만기 · 쿼드러플 위칭 (앞으로는 NYSE 휴장 목록, 과거는 SPY 가 연 날)
    closed = nyse_closed()
    for d, quad in opex_days(lo, hi, lambda x: x.weekday() < 5 and x not in closed):
        if quad:
            dec = d.month == 12
            add(_event(d, "quad", note_ko="나스닥100 연말 재구성도 이날 장 마감 뒤 적용" if dec else "",
                       note_en="Nasdaq-100 annual reconstitution also takes effect after this close" if dec else ""))
        else:
            add(_event(d, "opex"))
    if spy is not None and len(spy.dropna()):
        open_days = {ts.date() for ts in pd.DatetimeIndex(spy.dropna().index)}
        past = opex_days(date(2000, 1, 1), today - timedelta(days=1), lambda x: x in open_days)
        hist["opex"] = [d for d, q in past if not q]
        hist["quad"] = [d for d, q in past if q]

    # 휴장 · 조기 폐장
    for rows, key, src in ((NYSE_CLOSED, "holiday_us", NYSE_SRC), (NYSE_EARLY, "early_us", NYSE_SRC),
                           (TSX_CLOSED, "holiday_ca", TSX_SRC), (TSX_EARLY, "early_ca", TSX_SRC)):
        _kind, _c, _t, k_ko, k_en, _tm = SPECS[key]
        for day, ko, en in rows:
            add(_event(date.fromisoformat(day), key, ko=f"{k_ko} — {ko}", en=f"{k_en} — {en}", src=src))
    for last, name in ((NYSE_LAST_YEAR, "NYSE"), (TSX_LAST_YEAR, "TSX")):
        if last < today.year + 1:
            errors.append(f"{name} 휴장 목록에 {today.year + 1}년이 없음 — data.event_calendar 를 새 해로 고쳐야 함")

    # 정치 · 정책 · 지수 재구성
    for m in MANUAL_EVENTS:
        add(_event(date.fromisoformat(m["date"]), m["key"], note_ko=m["note_ko"], note_en=m["note_en"], src=m["src"]),
            until=policy_hi)
    for d in us_election_days(today.year, policy_hi.year):
        pres = d.year % 4 == 0
        add(_event(d, "election_us", ko="미국 대선 · 의회 선거" if pres else "미국 중간선거",
                   en="US presidential & congressional election" if pres else "US midterm election",
                   note_ko="결과는 밤에 나와 다음 거래일에 반영", note_en="Results land overnight — the next session reacts",
                   src=US_ELECTION_SRC), until=policy_hi)
    hist["election_us"] = us_election_days(1994, today.year)
    add(_event(date.fromisoformat(CA_ELECTION_NEXT), "election_ca", note_ko="법정 고정일 — 더 일찍 치를 수 있다",
               note_en="Fixed date — can be called earlier", src=CA_ELECTION_SRC), until=policy_hi)
    hist["election_ca"] = [date.fromisoformat(d) for d in CA_ELECTIONS_PAST]

    # 평소 대비 변동 — 키마다 한 번 (지난 날짜만)
    for key in {e["key"] for e in events}:
        days = [d for d in hist.get(key, []) if d < today]
        if not days:
            continue
        close = xic if SPECS[key][1] == "CA" else spy
        imp = impact(days, close, next_day=key.startswith("election"))
        if imp:
            for e in events:
                if e["key"] == key:
                    e["impact"] = imp

    order = {"rate": 0, "data": 1, "policy": 2, "market": 3, "holiday": 4}
    events.sort(key=lambda e: (e["date"], -e["tier"], e["time"] or "99:99", order[e["kind"]], e["key"]))
    return events


def impact_text_ko(imp: Mapping[str, Any] | None) -> str:
    """시트용 — '평소의 1.3배 (47번, 2021년부터)'."""
    if not imp:
        return ""
    return f"평소의 {imp['x']:.1f}배 ({imp['n']}번, {imp['since']}년부터)"
