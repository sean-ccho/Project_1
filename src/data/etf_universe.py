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
