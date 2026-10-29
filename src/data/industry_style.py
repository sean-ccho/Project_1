"""경기순환/방어 × 내수/글로벌 — 종목 성격 4칸 (2026-10-29, 표시 전용: 사이트 배지 · 필터 · 섹터 지도, 시트 칸).

- 경기 성격(순환 · 방어): Yahoo 세부 업종(약 145개)으로 — 사용자 표(경기순환주 · 경기방어주 × 내수 · 글로벌) 그대로.
  표에 없는 업종(폐기물 · 인력 · 교육 등)은 GICS 묶음으로 골랐다.
- 매출 지역(내수 · 글로벌): 회사별 SEC 10-K 미국 매출 비중(data.geo_revenue) — 미국 ≥ 70% 면 내수.
  비중을 못 구하면 순서대로: 해외 지역별로만 공시 → 확실한 해외가 30% 이상이면 글로벌, 아니면 내수 (크기를 모르면 업종 기본값 —
  PLD · SPG 처럼 유럽 · 아시아가 사업부와 섞여만 나오는 회사) · 지역별 매출 공시가 아예 없음 → 내수 (해외 매출이 작아 글로만
  쓰는 경우가 대부분) · 20-F/40-F 해외 기업 → 글로벌 · 그 밖(매출 없음 · 공시 없음) → 업종 기본값.
점수 · PT-1 에는 안 쓴다. 우리 시험(rotation_v1)에서 경기 국면에 따라 순환 ↔ 방어를 갈아타는 규칙은 SPY 를 못 이겼다 — 이해 · 분산용.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from data.geo_revenue import GeoRow, load_geo_revenue

DOMESTIC_MIN = 0.70          # 미국 매출이 이 이상이면 내수
FOREIGN_GLOBAL = 0.30        # 지역별로만 공시한 회사 — 확실한 해외가 이 이상이면 글로벌

DC, GC, DD, GD = ("cyclical", "domestic"), ("cyclical", "global"), ("defensive", "domestic"), ("defensive", "global")

# Yahoo 세부 업종 → (경기 성격, 매출 지역 기본값). 사이트 lib/i18n.ts INDUSTRIES_KO 와 같은 목록 (테스트가 확인)
INDUSTRY_STYLE: dict[str, tuple[str, str]] = {
    # 소재 — 원자재는 글로벌 순환. 건축 자재(골재 · 시멘트)는 산업 인프라 쪽 내수 순환
    "Agricultural Inputs": GC, "Building Materials": DC, "Chemicals": GC, "Specialty Chemicals": GC,
    "Lumber & Wood Production": GC, "Paper & Paper Products": GC, "Aluminum": GC, "Copper": GC,
    "Other Industrial Metals & Mining": GC, "Gold": GC, "Silver": GC, "Other Precious Metals & Mining": GC,
    "Coking Coal": GC, "Steel": GC,
    # 커뮤니케이션 — 방송 · 미디어는 글로벌 순환, 인터넷은 글로벌 방어, 통신은 내수 방어
    "Advertising Agencies": GC, "Broadcasting": GC, "Electronic Gaming & Multimedia": GC, "Entertainment": GC,
    "Internet Content & Information": GD, "Publishing": GC, "Telecom Services": DD,
    # 경기소비재 — 자동차 · 내구재 · 의류 제조는 글로벌 순환, 소매(의류 · 쇼핑 · 호텔) · 주택 건설은 내수 순환,
    # 소비 서비스(외식 · 여행 · 카지노 · 레저)는 글로벌 방어
    "Auto & Truck Dealerships": DC, "Auto Manufacturers": GC, "Auto Parts": GC, "Recreational Vehicles": GC,
    "Furnishings, Fixtures & Appliances": GC, "Residential Construction": DC, "Textile Manufacturing": GC,
    "Apparel Manufacturing": GC, "Footwear & Accessories": GC, "Packaging & Containers": GC, "Personal Services": DD,
    "Restaurants": GD, "Apparel Retail": DC, "Department Stores": DC, "Home Improvement Retail": DC,
    "Luxury Goods": GC, "Internet Retail": DC, "Specialty Retail": DC, "Gambling": GD, "Leisure": GD,
    "Lodging": DC, "Resorts & Casinos": GD, "Travel Services": GD,
    # 필수소비재 — 식품 소매 · 가공식품 · 주류는 내수 방어, 음료 · 과자 · 담배 · 생활용품은 글로벌 방어
    "Beverages - Brewers": DD, "Beverages - Wineries & Distilleries": DD, "Beverages - Non-Alcoholic": GD,
    "Confectioners": GD, "Farm Products": DD, "Household & Personal Products": GD, "Packaged Foods": DD,
    "Education & Training Services": DD, "Discount Stores": DD, "Food Distribution": DD, "Grocery Stores": DD,
    "Tobacco": GD,
    # 에너지 — 글로벌 순환
    "Oil & Gas Drilling": GC, "Oil & Gas E&P": GC, "Oil & Gas Integrated": GC, "Oil & Gas Midstream": GC,
    "Oil & Gas Refining & Marketing": GC, "Oil & Gas Equipment & Services": GC, "Thermal Coal": GC, "Uranium": GC,
    # 금융 — 은행 · 보험 · 복합 금융은 내수 순환
    "Banks - Diversified": DC, "Banks - Regional": DC, "Mortgage Finance": DC, "Capital Markets": DC,
    "Financial Data & Stock Exchanges": DC, "Insurance - Life": DC, "Insurance - Property & Casualty": DC,
    "Insurance - Reinsurance": DC, "Insurance - Specialty": DC, "Insurance - Diversified": DC, "Insurance Brokers": DC,
    "Shell Companies": DC, "Financial Conglomerates": DC, "Credit Services": DC, "Asset Management": DC,
    # 헬스케어 — 제약 · 바이오는 글로벌 방어, 나머지(보험 · 병원 · 기기 · 유통)는 내수 방어
    "Biotechnology": GD, "Drug Manufacturers - General": GD, "Drug Manufacturers - Specialty & Generic": GD,
    "Healthcare Plans": DD, "Medical Care Facilities": DD, "Pharmaceutical Retailers": DD,
    "Health Information Services": DD, "Medical Devices": DD, "Medical Instruments & Supplies": DD,
    "Diagnostics & Research": DD, "Medical Distribution": DD,
    # 산업재 — 자본재는 글로벌 순환, 운송 · 인프라 · 기업 서비스는 내수 순환
    "Aerospace & Defense": GC, "Specialty Business Services": DC, "Consulting Services": DC,
    "Rental & Leasing Services": DC, "Security & Protection Services": DC, "Staffing & Employment Services": DC,
    "Conglomerates": GC, "Engineering & Construction": DC, "Infrastructure Operations": DC,
    "Building Products & Equipment": DC, "Farm & Heavy Construction Machinery": GC, "Industrial Distribution": DC,
    "Business Equipment & Supplies": GC, "Specialty Industrial Machinery": GC, "Metal Fabrication": GC,
    "Pollution & Treatment Controls": GC, "Tools & Accessories": GC, "Electrical Equipment & Parts": GC,
    "Airports & Air Services": DC, "Airlines": DC, "Railroads": DC, "Marine Shipping": DC, "Trucking": DC,
    "Integrated Freight & Logistics": DC, "Waste Management": DC,
    # 부동산 — 내수 순환
    "Real Estate - Development": DC, "Real Estate Services": DC, "Real Estate - Diversified": DC,
    "REIT - Healthcare Facilities": DC, "REIT - Hotel & Motel": DC, "REIT - Industrial": DC, "REIT - Office": DC,
    "REIT - Residential": DC, "REIT - Retail": DC, "REIT - Mortgage": DC, "REIT - Specialty": DC,
    "REIT - Diversified": DC,
    # 기술 — 하드웨어 · 반도체 · IT 서비스는 글로벌 순환, 소프트웨어는 글로벌 방어
    "Information Technology Services": GC, "Software - Application": GD, "Software - Infrastructure": GD,
    "Communication Equipment": GC, "Computer Hardware": GC, "Consumer Electronics": GC, "Electronic Components": GC,
    "Electronics & Computer Distribution": GC, "Scientific & Technical Instruments": GC,
    "Semiconductor Equipment & Materials": GC, "Semiconductors": GC, "Solar": GC,
    # 유틸리티 — 내수 방어
    "Utilities - Independent Power Producers": DD, "Utilities - Renewable": DD, "Utilities - Regulated Water": DD,
    "Utilities - Regulated Electric": DD, "Utilities - Regulated Gas": DD, "Utilities - Diversified": DD,
}

CYCLE_KO = {"cyclical": "경기순환", "defensive": "경기방어"}
REGION_KO = {"domestic": "내수", "global": "글로벌"}


def region_from_geo(geo: GeoRow | None) -> tuple[str | None, str | None]:
    """SEC 결과 → (매출 지역, 근거). 근거: sec (비중) · sec_regions (지역별만) · sec_none (지역 공시 없음) · foreign (해외 기업)."""
    if geo is None:
        return None, None
    if geo.us_share is not None:
        return ("domestic" if geo.us_share >= DOMESTIC_MIN else "global"), "sec"
    if geo.method == "geo_regions" and geo.foreign_min is not None:
        return ("global" if geo.foreign_min >= FOREIGN_GLOBAL else "domestic"), "sec_regions"
    if geo.method == "none":
        return "domestic", "sec_none"
    if geo.method == "foreign_filer":
        return "global", "foreign"
    return None, None


def stock_style(industry: str | None, geo: GeoRow | None) -> dict[str, Any] | None:
    """종목 하나의 4칸 성격. 업종도 SEC 도 없으면 None.

    → {cycle, region, us_share, foreign_min, src, fy} — src 'industry' 면 매출 지역은 업종 기본값(추정).
    """
    base = INDUSTRY_STYLE.get(industry or "")
    region, src = region_from_geo(geo)
    if region is None and base:
        region, src = base[1], "industry"
    cycle = base[0] if base else None
    if cycle is None and region is None:
        return None
    sec = src in ("sec", "sec_regions", "sec_none")
    return {
        "cycle": cycle,
        "region": region,
        "us_share": round(geo.us_share, 4) if sec and geo and geo.us_share is not None else None,
        "foreign_min": round(geo.foreign_min, 4) if sec and geo and geo.foreign_min is not None else None,
        "src": src,
        "fy": (geo.fy_end or None) if sec and geo else None,
    }


@lru_cache(maxsize=1)
def _geo() -> dict[str, GeoRow]:
    return load_geo_revenue()


def style_for(ticker: str, industry: str | None) -> dict[str, Any] | None:
    """밤 실행용 — GEO_REVENUE_FILE 을 한 번만 읽어 둔다 (없으면 업종 기준)."""
    t = str(ticker).strip().upper().replace(".", "-")
    return stock_style(industry, _geo().get(t))


def style_text(style: dict[str, Any] | None) -> tuple[str, str, str]:
    """시트 칸 (경기 성격, 매출 지역, 미국 매출 비중) — '경기방어' · '내수 (추정)' · '82%'."""
    if not style:
        return "", "", ""
    cycle = CYCLE_KO.get(style.get("cycle") or "", "")
    region = REGION_KO.get(style.get("region") or "", "")
    if region and style.get("src") in ("industry", "sec_none"):
        region += " (추정)"
    elif region and style.get("src") == "foreign":
        region += " (해외 기업)"
    us = style.get("us_share")
    return cycle, region, f"{us:.0%}" if us is not None else ""
