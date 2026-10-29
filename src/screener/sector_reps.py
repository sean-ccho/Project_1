"""섹터별 대표 기업 — 글로벌 매출 중심 / 내수 매출 중심 × 섹터마다 시가총액 상위 (2026-10-29, 시트 [섹터 대표] · 표시 전용).

사용자 시트(내수 · 글로벌 매출 중심 섹터별 미국 대표 기업)를 매일 자동으로. 종목은 S&P 500, 내수 · 글로벌은 data.industry_style
(SEC 10-K 미국 매출 비중, 미국 ≥ 70% = 내수). 사이트 /quant/sectors 도 같은 규칙으로 sp500.json 에서 만든다.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from data.industry_style import CYCLE_KO

SHEET_TAB = "[섹터 대표]"
TOP_N = 4
# Yahoo 섹터 → 시트 이름 (사용자 시트 순서)
SECTORS: list[tuple[str, str]] = [
    ("Communication Services", "Communication (커뮤니케이션)"),
    ("Consumer Cyclical", "Consumer Discretionary (임의소비재)"),
    ("Consumer Defensive", "Consumer Staples (필수소비재)"),
    ("Energy", "Energy (에너지)"),
    ("Financial Services", "Financials (금융)"),
    ("Healthcare", "Healthcare (헬스케어)"),
    ("Industrials", "Industrials (산업재)"),
    ("Technology", "IT (기술)"),
    ("Basic Materials", "Materials (소재)"),
    ("Real Estate", "Real Estate (부동산)"),
    ("Utilities", "Utilities (유틸리티)"),
]


def sector_reps(items: Mapping[str, Mapping[str, Any]], top: int = TOP_N) -> list[dict[str, Any]]:
    """S&P 500 웹 정보(item_info) → 지역(글로벌 먼저) × 섹터 × 시가총액 상위 top. 성격 · 시총이 없는 종목은 빠진다."""
    order = {s: i for i, (s, _label) in enumerate(SECTORS)}
    rows = []
    for t, info in items.items():
        style = info.get("style") or {}
        cap = (info.get("fundamentals") or {}).get("market_cap")
        if style.get("region") not in ("global", "domestic") or info.get("sector") not in order or not cap:
            continue
        tv = str(info.get("tv") or "")
        rows.append({"region": style["region"], "sector": info["sector"], "ticker": t, "company": info.get("company"),
                     "exchange": tv.split(":")[0] if ":" in tv else "", "price": info.get("price"), "market_cap": cap,
                     "us_share": style.get("us_share"), "src": style.get("src"), "cycle": style.get("cycle"),
                     "ret_20d": info.get("ret_20d")})
    rows.sort(key=lambda r: (r["region"] != "global", order[r["sector"]], -r["market_cap"]))
    out, seen = [], {}
    for r in rows:
        k = (r["region"], r["sector"])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= top:
            out.append(r)
    return out


def sector_reps_frame(items: Mapping[str, Mapping[str, Any]], top: int = TOP_N) -> pd.DataFrame:
    """시트 [섹터 대표] — 구분 · 섹터 · 회사명 · 거래소 · 티커 · 가격 · 미국 매출 비중 · 경기 성격 · 1달 수익."""
    label = dict(SECTORS)
    rows = []
    for r in sector_reps(items, top):
        us = r["us_share"]
        rows.append({
            "구분": "글로벌 매출 중심" if r["region"] == "global" else "내수 매출 중심",
            "섹터": label[r["sector"]], "회사명": r["company"] or "", "거래소": r["exchange"], "티커": r["ticker"],
            "가격": r["price"],
            "미국 매출 비중": f"{us:.0%}" if us is not None else ("추정" if r["src"] in ("industry", "sec_none") else ""),
            "경기 성격": CYCLE_KO.get(r["cycle"] or "", ""),
            "1달 수익": "" if r["ret_20d"] is None else f"{r['ret_20d']:+.1%}",
        })
    return pd.DataFrame(rows, columns=["구분", "섹터", "회사명", "거래소", "티커", "가격", "미국 매출 비중", "경기 성격", "1달 수익"])

