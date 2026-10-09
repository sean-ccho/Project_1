"""Nasdaq 100 ticker universe."""

# 상장폐지/인수합병된 종목 제거: ANSS, SGEN, SPLK, GGFS, ATVI, EA(2026-10 가격 없음, S&P 500 에서도 빠짐)
NASDAQ_TICKERS = [
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP",
    "COST", "CSCO", "TMUS", "CMCSA", "ADBE", "NFLX", "TXN", "AMD", "QCOM", "AMGN",
    "HON", "INTU", "INTC", "SBUX", "GILD", "AMAT", "ADP", "BKNG", "ISRG", "VRTX",
    "MDLZ", "REGN", "ADI", "PYPL", "KLAC", "LRCX", "PANW", "SNPS", "MU", "CHTR",
    "CDNS", "MAR", "CSX", "ORLY", "MELI", "MNST", "ASML", "CTAS", "ODFL", "NXPI",
    "PCAR", "FTNT", "KDP", "MCHP", "ABNB", "PAYX", "ROST", "IDXX", "PDD", "WDAY",
    "MRVL", "EXC", "AEP", "LULU", "AZN", "BIIB", "DXCM", "BKR", "KHC",
    "MRNA", "FAST", "CEG", "TEAM", "VRSK", "CPRT", "XEL", "DDOG",
    "GEHC", "ALGN", "DLTR", "EBAY", "SIRI", "ZM", "ZS", "LCID", "CRWD",
    "MTCH", "SWKS", "DOCU", "OKTA", "NTES", "JD", "BIDU", "FISV"
]
