"""실적 발표일 기억 — 실적을 낸 회사의 재무 · 분기 실적을 발표 뒤 다시 받게 하는 공용 도구 (2026-11-17, 표시 전용).

캐시(예측 모델 value_fund · 종목 상세 detail)의 종목 줄에 지난 실적일(last_report) · 다음 실적일(next_report)을 남긴다.
- 미국 국내 회사: 전체분석 목록의 SEC 실적 공시일(earnings_move.last — 8-K 실적 공시 시각)이 지난 실적일.
- 외국 회사(6-K 를 내는 VIPS · DOX 같은 곳) · 캐나다 종목: SEC 날짜가 없어서, 목록의 다음 실적일(Yahoo 예정일)을 밤마다
  next_report 로 적어 두고 그날이 지나면 지난 실적일로 본다. 둘 다 없으면 각 캐시의 정해진 주기로만 다시 받는다.
캐시가 종목을 다시 받을 때 이 두 칸은 그대로 옮겨야 한다 (받은 재무가 아니라 밤마다 적은 값).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import date, timedelta
from typing import Any

KEYS = ("last_report", "next_report")


def day(v: Any) -> str | None:
    """'2026-11-18' · '2026-11-18T…' → '2026-11-18'. 날짜 모양이 아니면 None."""
    s = str(v or "")[:10]
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return None


def item_dates(it: Mapping[str, Any] | None) -> dict[str, str | None]:
    """전체분석 · 캐나다 목록의 종목 한 줄 → {"last_report": SEC 지난 실적일, "next_report": Yahoo 다음 실적일}."""
    it = it or {}
    last = (it.get("earnings_move") or {}).get("last") or []
    sec = day(last[-1][0]) if last and isinstance(last[-1], (list, tuple)) and last[-1] else None
    return {"last_report": sec, "next_report": day(it.get("earnings_date"))}


def note(cache: Mapping[str, dict[str, Any]], src: Mapping[str, Mapping[str, Any]], today: date,
         key: Callable[[str], str] = lambda t: t) -> int:
    """캐시 줄마다 실적일을 적는다 → 지난 실적일이 새로 바뀐 종목 수. src = {key(티커): item_dates 모양}.
    지난 실적일 = SEC 날짜 · 적어 둔 날 · 목록의 다음 실적일 중 오늘까지 온 가장 늦은 날. 다음 실적일 = 오늘 뒤의 예정일."""
    today_s = today.isoformat()
    changed = 0
    for t, ent in cache.items():
        sc = src.get(key(t)) or {}
        cands = (ent.get("last_report"), sc.get("last_report"), ent.get("next_report"), sc.get("next_report"))
        last = max((c for c in cands if c and c <= today_s), default=None)
        nxt = next((c for c in (sc.get("next_report"), ent.get("next_report")) if c and c > today_s), None)
        if last and last != ent.get("last_report"):
            ent["last_report"] = last
            changed += 1
        if nxt:
            ent["next_report"] = nxt
        else:
            ent.pop("next_report", None)
    return changed


def due(ent: Mapping[str, Any], today: date, checks: Iterable[int]) -> str | None:
    """지난 실적 발표 뒤 다시 받을 날(발표일 + checks 일) 중 오늘까지 온 가장 늦은 날. 그 전에 받았으면 다시 받을 때."""
    rep = day(ent.get("last_report"))
    if not rep:
        return None
    base = date.fromisoformat(rep)
    days = sorted(base + timedelta(days=k) for k in checks if base + timedelta(days=k) <= today)
    return days[-1].isoformat() if days else None


def carry(old: Mapping[str, Any] | None) -> dict[str, Any]:
    """다시 받은 줄에 옮길 실적일 칸."""
    return {k: old[k] for k in KEYS if old and old.get(k)}
