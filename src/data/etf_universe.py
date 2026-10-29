"""ETF 탭(사이트 /quant/etf · 시트 [ETF])에서 보는 ETF 목록 — 묶음별로 유명하고 운용자산이 큰 것 (2026-10-28).

한 ETF 는 한 묶음에만 넣는다. 캐나다 상장은 야후 형식(.TO). 레버리지 · 인버스는 넣지 않는다 (오래 들고 갈 상품이 아님).
목록을 바꾸면 사이트 · 시트가 다음 밤 실행부터 따라간다.
"""

from __future__ import annotations

# (키, 한국어 이름, 영어 이름, 티커들)
ETF_GROUPS: list[tuple[str, str, str, list[str]]] = [
    ("us_market", "미국 시장 전체", "US broad market",
     ["SPY", "VOO", "IVV", "VTI", "QQQ", "QQQM", "DIA", "RSP", "IWM", "IJH", "IJR", "VUG", "VTV", "SCHG"]),
    ("us_sector", "미국 섹터", "US sectors",
     ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]),
    ("industry", "산업 · 테마", "Industries & themes",
     ["SMH", "SOXX", "IGV", "CIBR", "XBI", "IBB", "KRE", "KBE", "ITA", "XAR", "ITB", "XHB", "XRT", "XOP", "OIH",
      "GDX", "GDXJ", "URA", "TAN", "ICLN", "LIT", "COPX", "PAVE", "BOTZ", "ARKK", "JETS"]),
    ("dividend_factor", "배당 · 팩터", "Dividend & factor",
     ["SCHD", "VYM", "VIG", "DGRO", "NOBL", "HDV", "DVY", "MTUM", "QUAL", "USMV", "SPLV", "VLUE", "COWZ"]),
    ("international", "해외", "International",
     ["VXUS", "EFA", "VEA", "IEFA", "EEM", "VWO", "IEMG", "EWJ", "EWY", "MCHI", "FXI", "INDA", "EWT", "EWZ", "EWG",
      "EWU", "EWC", "EWA"]),
    ("bonds", "채권", "Bonds",
     ["SGOV", "BIL", "SHY", "IEI", "IEF", "TLT", "AGG", "BND", "TIP", "LQD", "HYG", "JNK", "EMB", "MUB"]),
    ("commodities", "금 · 원자재", "Gold & commodities",
     ["GLD", "IAU", "GLDM", "SLV", "DBC", "PDBC", "USO", "UNG", "DBA", "CPER"]),
    ("real_estate", "부동산", "Real estate", ["VNQ", "IYR", "SCHH", "REET"]),
    ("canada", "캐나다 상장 (TSX)", "Canada-listed (TSX)",
     ["XIU.TO", "XIC.TO", "VFV.TO", "ZSP.TO", "XUS.TO", "XEQT.TO", "VEQT.TO", "XGRO.TO", "VGRO.TO", "ZQQ.TO",
      "XQQ.TO", "ZEB.TO", "XEI.TO", "ZDV.TO", "VDY.TO", "ZAG.TO", "XBB.TO", "ZRE.TO", "XGD.TO", "HXT.TO"]),
    ("crypto", "가상자산", "Crypto",
     ["IBIT", "FBTC", "ARKB", "BITB", "GBTC", "ETHA", "FETH", "ETHE", "BITO",
      "BTCC-B.TO", "FBTC.TO", "EBIT.TO", "ETHH.TO"]),
]

# 같은 · 비슷한 것을 담는 ETF 묶음 (2026-10-29) — (키, 한국어, 영어, 종류, 티커들). 한 ETF 는 한 묶음에만.
# same = 같은 지수 · 같은 실물 (보수만 다름), similar = 지수는 달라도 담는 것이 거의 같음. 사이트 상세 · 표에 '같은 것' 으로 보인다.
ETF_TWINS: list[tuple[str, str, str, str, list[str]]] = [
    ("sp500", "S&P 500", "S&P 500", "same", ["SPY", "VOO", "IVV"]),
    ("nasdaq100", "나스닥 100", "Nasdaq-100", "same", ["QQQ", "QQQM"]),
    ("gold", "금 실물", "Physical gold", "same", ["GLD", "IAU", "GLDM"]),
    ("bitcoin", "비트코인 현물 (미국)", "Spot bitcoin (US)", "same", ["IBIT", "FBTC", "ARKB", "BITB", "GBTC"]),
    ("ether", "이더리움 현물 (미국)", "Spot ether (US)", "same", ["ETHA", "FETH", "ETHE"]),
    ("bitcoin_ca", "비트코인 현물 (캐나다)", "Spot bitcoin (Canada)", "same", ["BTCC-B.TO", "FBTC.TO", "EBIT.TO"]),
    ("ca_sp500", "S&P 500 (캐나다 상장)", "S&P 500 (Canada-listed)", "same", ["VFV.TO", "ZSP.TO", "XUS.TO"]),
    ("tsx60", "TSX 60", "TSX 60", "same", ["XIU.TO", "HXT.TO"]),
    ("ca_nasdaq100", "나스닥 100 (캐나다, 환헤지)", "Nasdaq-100 (Canada, hedged)", "same", ["ZQQ.TO", "XQQ.TO"]),
    ("us_growth", "미국 대형 성장주", "US large growth", "similar", ["VUG", "SCHG"]),
    ("semis", "반도체", "Semiconductors", "similar", ["SMH", "SOXX"]),
    ("biotech", "바이오", "Biotech", "similar", ["XBI", "IBB"]),
    ("banks", "미국 은행", "US banks", "similar", ["KRE", "KBE"]),
    ("aero_defense", "항공우주 · 방산", "Aerospace & defense", "similar", ["ITA", "XAR"]),
    ("homebuilders", "주택 건설", "Homebuilders", "similar", ["ITB", "XHB"]),
    ("gold_miners", "금광 회사", "Gold miners", "similar", ["GDX", "GDXJ"]),
    ("div_growth", "배당 성장", "Dividend growth", "similar", ["VIG", "DGRO", "NOBL"]),
    ("min_vol", "저변동", "Low volatility", "similar", ["USMV", "SPLV"]),
    ("developed", "선진국 (미국 밖)", "Developed ex-US", "similar", ["EFA", "VEA", "IEFA"]),
    ("emerging", "신흥국", "Emerging markets", "similar", ["EEM", "VWO", "IEMG"]),
    ("china", "중국", "China", "similar", ["MCHI", "FXI"]),
    ("tbills", "초단기 국채", "T-bills", "similar", ["SGOV", "BIL"]),
    ("us_agg", "미국 채권 전체", "US aggregate bonds", "similar", ["AGG", "BND"]),
    ("high_yield", "하이일드 채권", "High-yield bonds", "similar", ["HYG", "JNK"]),
    ("broad_commod", "원자재 바스켓", "Broad commodities", "similar", ["DBC", "PDBC"]),
    ("us_reit", "미국 리츠", "US REITs", "similar", ["VNQ", "IYR", "SCHH"]),
    ("ca_all_equity", "캐나다 올인원 (주식 100%)", "Canada all-in-one (100% equity)", "similar", ["XEQT.TO", "VEQT.TO"]),
    ("ca_growth", "캐나다 올인원 (주식 80%)", "Canada all-in-one (80% equity)", "similar", ["XGRO.TO", "VGRO.TO"]),
    ("ca_agg", "캐나다 채권 전체", "Canada aggregate bonds", "similar", ["ZAG.TO", "XBB.TO"]),
    ("ca_dividend", "캐나다 배당", "Canada dividend", "similar", ["XEI.TO", "ZDV.TO", "VDY.TO"]),
]

# '역할별 기본 1개' (2026-10-29) — (키, 한국어, 영어, 후보). 후보 중 운용자산이 기준 이상인 것에서 보수가 가장 싼 것을 매일 고른다.
# 후보는 같은 것을 담는 ETF 만 (다른 지수끼리 보수로 고르면 의미가 없어서 배당 같은 역할은 넣지 않았다).
ETF_ROLES: list[tuple[str, str, str, list[str]]] = [
    ("sp500", "미국 대표 500 (S&P 500)", "US 500 (S&P 500)", ["SPY", "VOO", "IVV"]),
    ("total_us", "미국 주식 전체", "Total US stock market", ["VTI"]),
    ("nasdaq100", "기술주 많이 (나스닥 100)", "Tech-heavy (Nasdaq-100)", ["QQQ", "QQQM"]),
    ("ex_us", "미국 밖 주식 전체", "All stocks outside the US", ["VXUS"]),
    ("us_bonds", "미국 채권 전체", "US bonds (aggregate)", ["AGG", "BND"]),
    ("cash", "현금 대신 (초단기 국채)", "Cash alternative (T-bills)", ["SGOV", "BIL"]),
    ("gold", "금", "Gold", ["GLD", "IAU", "GLDM"]),
    ("bitcoin", "비트코인 (미국 상장)", "Bitcoin (US-listed)", ["IBIT", "FBTC", "ARKB", "BITB", "GBTC"]),
    ("ca_world", "캐나다 계좌 · 전 세계 주식 한 개로", "Canadian account · whole world in one", ["XEQT.TO", "VEQT.TO"]),
    ("ca_sp500", "캐나다 계좌 · S&P 500 (C$)", "Canadian account · S&P 500 (C$)", ["VFV.TO", "ZSP.TO", "XUS.TO"]),
]
ROLE_MIN_AUM = 1e9                 # 운용자산 10억 (그 ETF 통화) — 작으면 문 닫을 위험 · 사고팔 때 손해
ROLE_FEE_TIE = 0.00005             # 보수 차이 0.005%p 미만은 같다고 보고 운용자산 큰 쪽

# '요즘 강한 ETF (참고)' 를 고르는 묶음 — 시장 전체 · 채권 · 가상자산 · 캐나다는 빼고 섹터 · 테마 · 나라 · 원자재 · 부동산만
MOMENTUM_GROUPS = ("us_sector", "industry", "international", "commodities", "real_estate")
MOMENTUM_TOP = 5

# 12개월 추세(무위험 초과 수익 > 0)를 볼 때 무위험 대신 쓰는 단기 국채 ETF
RISK_FREE_ETF = "BIL"
BENCHMARK_ETF = "SPY"


def etf_tickers() -> list[str]:
    """모든 묶음의 티커 (순서 유지, 중복 없음)."""
    seen: dict[str, None] = {}
    for _key, _ko, _en, tickers in ETF_GROUPS:
        for t in tickers:
            seen.setdefault(t, None)
    return list(seen)


def etf_group_of() -> dict[str, str]:
    """티커 → 묶음 키."""
    return {t: key for key, _ko, _en, tickers in ETF_GROUPS for t in tickers}


def etf_twin_of() -> dict[str, str]:
    """티커 → 같은 것 묶음 키 (묶음이 없는 ETF 는 빠짐)."""
    return {t: key for key, _ko, _en, _kind, tickers in ETF_TWINS for t in tickers}
