"""FOMC · Bank of Canada 금리 결정 일정 (2026-10-30, 오늘 탭 '금리 결정' · 시트 [경제], 표시 전용).

- 아래 목록은 공식 페이지에서 그대로 옮긴 것 (2026-10-30 확인):
  FOMC = federalreserve.gov/monetarypolicy/fomccalendars.htm (2021–2027, '*' = 경제 전망(SEP) 같이 발표, notation vote 제외)
  BoC = bankofcanada.ca 의 'Schedule for 2026 · 2027' + 2021–2025 는 공식 보도자료 주소(fad-press-release-YYYY-MM-DD)가
  있는 날 (연 8번, 1 · 4 · 7 · 10월은 통화정책보고서(MPR) 같이 발표)
- 밤 실행은 두 페이지를 다시 읽어 새 해 일정을 더한다 (parse_fomc_page · parse_boc_page). 못 읽으면 이 목록만 쓴다.
- 결정(인상 · 동결 · 인하)은 날짜가 아니라 금리 숫자로 판단한다 — FRED DFEDTARU/DFEDTARL · BoC Valet V39079 (screener.macro_scan).
"""

from __future__ import annotations

import html
import re
from datetime import date

# (회의 시작일, 결정 발표일, 경제 전망(SEP) 같이 발표)
FOMC_MEETINGS: list[tuple[str, str, bool]] = [
    ("2021-01-26", "2021-01-27", False), ("2021-03-16", "2021-03-17", True), ("2021-04-27", "2021-04-28", False),
    ("2021-06-15", "2021-06-16", True), ("2021-07-27", "2021-07-28", False), ("2021-09-21", "2021-09-22", True),
    ("2021-11-02", "2021-11-03", False), ("2021-12-14", "2021-12-15", True),
    ("2022-01-25", "2022-01-26", False), ("2022-03-15", "2022-03-16", True), ("2022-05-03", "2022-05-04", False),
    ("2022-06-14", "2022-06-15", True), ("2022-07-26", "2022-07-27", False), ("2022-09-20", "2022-09-21", True),
    ("2022-11-01", "2022-11-02", False), ("2022-12-13", "2022-12-14", True),
    ("2023-01-31", "2023-02-01", False), ("2023-03-21", "2023-03-22", True), ("2023-05-02", "2023-05-03", False),
    ("2023-06-13", "2023-06-14", True), ("2023-07-25", "2023-07-26", False), ("2023-09-19", "2023-09-20", True),
    ("2023-10-31", "2023-11-01", False), ("2023-12-12", "2023-12-13", True),
    ("2024-01-30", "2024-01-31", False), ("2024-03-19", "2024-03-20", True), ("2024-04-30", "2024-05-01", False),
    ("2024-06-11", "2024-06-12", True), ("2024-07-30", "2024-07-31", False), ("2024-09-17", "2024-09-18", True),
    ("2024-11-06", "2024-11-07", False), ("2024-12-17", "2024-12-18", True),
    ("2025-01-28", "2025-01-29", False), ("2025-03-18", "2025-03-19", True), ("2025-05-06", "2025-05-07", False),
    ("2025-06-17", "2025-06-18", True), ("2025-07-29", "2025-07-30", False), ("2025-09-16", "2025-09-17", True),
    ("2025-10-28", "2025-10-29", False), ("2025-12-09", "2025-12-10", True),
    ("2026-01-27", "2026-01-28", False), ("2026-03-17", "2026-03-18", True), ("2026-04-28", "2026-04-29", False),
    ("2026-06-16", "2026-06-17", True), ("2026-07-28", "2026-07-29", False), ("2026-09-15", "2026-09-16", True),
    ("2026-10-27", "2026-10-28", False), ("2026-12-08", "2026-12-09", True),
    ("2027-01-26", "2027-01-27", False), ("2027-03-16", "2027-03-17", True), ("2027-04-27", "2027-04-28", False),
    ("2027-06-08", "2027-06-09", True), ("2027-07-27", "2027-07-28", False), ("2027-09-14", "2027-09-15", True),
    ("2027-10-26", "2027-10-27", False), ("2027-12-07", "2027-12-08", True),
]

# (발표일, 통화정책보고서(MPR) 같이 발표)
BOC_ANNOUNCEMENTS: list[tuple[str, bool]] = [
    ("2021-01-20", True), ("2021-03-10", False), ("2021-04-21", True), ("2021-06-09", False), ("2021-07-14", True),
    ("2021-09-08", False), ("2021-10-27", True), ("2021-12-08", False),
    ("2022-01-26", True), ("2022-03-02", False), ("2022-04-13", True), ("2022-06-01", False), ("2022-07-13", True),
    ("2022-09-07", False), ("2022-10-26", True), ("2022-12-07", False),
    ("2023-01-25", True), ("2023-03-08", False), ("2023-04-12", True), ("2023-06-07", False), ("2023-07-12", True),
    ("2023-09-06", False), ("2023-10-25", True), ("2023-12-06", False),
    ("2024-01-24", True), ("2024-03-06", False), ("2024-04-10", True), ("2024-06-05", False), ("2024-07-24", True),
    ("2024-09-04", False), ("2024-10-23", True), ("2024-12-11", False),
    ("2025-01-29", True), ("2025-03-12", False), ("2025-04-16", True), ("2025-06-04", False), ("2025-07-30", True),
    ("2025-09-17", False), ("2025-10-29", True), ("2025-12-10", False),
    ("2026-01-28", True), ("2026-03-18", False), ("2026-04-29", True), ("2026-06-10", False), ("2026-07-15", True),
    ("2026-09-02", False), ("2026-10-28", True), ("2026-12-09", False),
    ("2027-01-27", True), ("2027-03-03", False), ("2027-04-28", True), ("2027-06-02", False), ("2027-07-21", True),
    ("2027-09-08", False), ("2027-10-27", True), ("2027-12-08", False),
]

FOMC_PAGE = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BOC_PAGE = "https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/"

_MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                       "September", "October", "November", "December"], 1)}
_ABBR = {name[:3]: i for name, i in _MONTHS.items()}


def fomc_url(decision_day: str) -> str:
    """결정 발표일의 FOMC 성명 (연준 보도자료)."""
    return f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{decision_day.replace('-', '')}a.htm"


def boc_url(day: str) -> str:
    """발표일의 BoC 금리 결정 보도자료."""
    return f"https://www.bankofcanada.ca/{day[:4]}/{day[5:7]}/fad-press-release-{day}/"


def parse_fomc_page(page: str) -> list[tuple[str, str, bool]]:
    """연준 일정 페이지 → (시작일, 발표일, SEP). notation vote 는 빼고, 임시 회의(하루)는 시작일 = 발표일."""
    out: list[tuple[str, str, bool]] = []
    for m in re.finditer(r"<h4><a[^>]*>(\d{4}) FOMC Meetings</a></h4>(.*?)(?=<h4><a|$)", page, re.S):
        yr = int(m.group(1))
        rows = re.findall(r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<",
                          m.group(2), re.S)
        for mon, day in rows:
            mon, day = mon.strip(), html.unescape(day).strip()
            nums = re.findall(r"\d+", day)
            if "notation" in day.lower() or not nums:
                continue
            sep = "*" in day
            try:
                if "/" in mon:   # 'Apr/May' '30-1' — 달이 바뀌는 회의
                    a, b = mon.split("/")
                    out.append((date(yr, _ABBR[a.strip()[:3]], int(nums[0])).isoformat(),
                                date(yr, _ABBR[b.strip()[:3]], int(nums[-1])).isoformat(), sep))
                else:
                    mo = _MONTHS[mon]
                    out.append((date(yr, mo, int(nums[0])).isoformat(), date(yr, mo, int(nums[-1])).isoformat(), sep))
            except (KeyError, ValueError):
                continue
    return sorted(set(out))


def parse_boc_page(page: str) -> list[tuple[str, bool]]:
    """BoC 기준금리 페이지의 'Schedule for YYYY' 표 → (발표일, MPR)."""
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))
    out: list[tuple[str, bool]] = []
    for m in re.finditer(r"Schedule for (\d{4}) (.*?)(?=Schedule for \d{4}|See Blackout|$)", text):
        yr = int(m.group(1))
        pat = r"(" + "|".join(_MONTHS) + r") (\d{1,2}) Interest rate announcement( and Monetary Policy Report)?"
        for mon, day, mpr in re.findall(pat, m.group(2)):
            try:
                out.append((date(yr, _MONTHS[mon], int(day)).isoformat(), bool(mpr)))
            except ValueError:
                continue
    return sorted(set(out))


def merge_fomc(scraped: list[tuple[str, str, bool]]) -> list[tuple[str, str, bool]]:
    """내장 목록 + 페이지에서 새로 읽은 일정 (발표일 기준으로 겹치면 내장 목록이 먼저)."""
    have = {d for _s, d, _x in FOMC_MEETINGS}
    return sorted(FOMC_MEETINGS + [m for m in scraped if m[1] not in have and m[1] >= FOMC_MEETINGS[0][1]],
                  key=lambda m: m[1])


def merge_boc(scraped: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    have = {d for d, _m in BOC_ANNOUNCEMENTS}
    return sorted(BOC_ANNOUNCEMENTS + [a for a in scraped if a[0] not in have and a[0] >= BOC_ANNOUNCEMENTS[0][0]])
