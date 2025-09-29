"""전략 전역 설정 모음.

프로젝트 전반에서 공유하는 티커 목록, 섹터 매핑, 가중치, 임계치 등을 한 곳에서 관리하여
파일이 나뉘어 있어도 수정 포인트를 명확히 하기 위한 모듈이다."""

from typing import Dict

from data.sp500_tickers import SP500_TICKERS

# 분석 대상 종목 풀: S&P 500 편입 종목(유니버스 관리 파일은 data/sp500_tickers.py 참고).
TICKERS = SP500_TICKERS.copy()
COMPANY_NAME_MAP = {}

# 섹터 매핑: 섹터 중립화 시 Unknown 여부를 판단하는 기준이 된다.
SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "GOOGL": "Communication Services",
    "AMD": "Information Technology",
    "NFLX": "Communication Services",
    "INTC": "Information Technology",
    "SHOP.TO": "Information Technology",
    "NTR.TO": "Materials",
    "BNS.TO": "Financials",
    "BMO.TO": "Financials",
    "SU.TO": "Energy",
    "ENB.TO": "Energy",
    "CNQ.TO": "Energy",
}

# 트렌드 점수 계산에 사용되는 가중치.
WEIGHTS = {
    "ret5": 0.30,   # 5일 수익률 기반 단기 모멘텀
    "vol": 0.30,    # 거래량 급증 정도
    "break": 0.25,  # 52주 고저 대비 위치(돌파 정도)
    "vola": 0.05,   # ATR% 변동성 (높을수록 감점)
    "rsi": 0.10,    # RSI 기반 스무딩 점수
}

# 시장별(US/CA) 거래대금 분위수를 이용한 유동성 컷 비율.
LIQUIDITY_QUANTILE = 0.40  # 하위 40% 제거 = 상위 60% 유지
LIQUIDITY_WHITELIST = ["GRRR", "COIN", "NBM.V"]  # 필수 포함 종목(저유동성이라도 유지)

for ticker in LIQUIDITY_WHITELIST:
    if ticker not in TICKERS:
        TICKERS.append(ticker)

for _extra in LIQUIDITY_WHITELIST:
    if _extra not in TICKERS:
        TICKERS.append(_extra)

# Google Sheets 연결 설정(기본 비활성화)
GOOGLE_SHEETS_ENABLED = True
GOOGLE_SHEETS_CREDENTIALS_PATH: str | None = "gspread-service-account.json"
GOOGLE_SHEETS_SPREADSHEET_ID: str | None = "1VxzTPfDvRX0UGJZcNnRsqTRex4gqmjv4P-1I_VcuXaw"
GOOGLE_SHEETS_WORKSHEET = "Signals"

# 시그널 판정 임계치.
BUY_SCORE_THRESHOLD = 0.15
BUY_POS_THRESHOLD = 0.70
BUY_RSI_MIN = 55
BUY_RSI_MAX = 75
WATCH_SCORE_THRESHOLD = 0.05
WATCH_POS_THRESHOLD = 0.50
OVERBOUGHT_RSI = 80

# 추가 기술적 지표 임계치
MACD_BUY_HIST_THRESHOLD = 0.5
MACD_SELL_HIST_THRESHOLD = -0.5
ADX_TREND_THRESHOLD = 20
ADX_WEAK_THRESHOLD = 15
STOCH_OVERBOUGHT = 85
STOCH_MIN_BUY = 40
OBV_Z_BUY_THRESHOLD = 0.0
CMF_BUY_THRESHOLD = 0.0
BOLLINGER_BREAKOUT_PBAND = 0.85
BOLLINGER_OVERBOUGHT_PBAND = 0.98

# 시그널 우선순위 매핑(테이블 정렬 순서를 통제).
SIGNAL_PRIORITY = {
    "매수 후보": 0,
    "관심 관찰": 1,
    "관망 과열": 2,
    "관망 약세": 3,
}

# 퍼센티지로 표현할 지표 목록(콘솔 및 CSV 출력 시 사용).
PERCENT_COLUMNS = [
    "트렌드점수_최종",
    "트렌드점수",
    "5일수익률",
    "1일수익률",
    "52주포지션",
    "ATR%",
    "ema_gap_20_50",
    "ema_gap_50_200",
    "ema_gap_20_200",
    "bollinger_pband",
    "keltner_pband",
    "accdist_slope_5",
    "dividend_yield",
]

# 실행 때마다 백업(타임스탬프) CSV를 추가로 만들 것인지 여부.
EXPORT_WITH_BACKUP = False

TECH_COLUMN_LABELS = {
    "macd": "MACD",
    "macd_signal": "MACD 시그널",
    "macd_hist": "MACD 히스토그램",
    "stoch_k": "스토캐스틱%K",
    "stoch_d": "스토캐스틱%D",
    "roc_10": "ROC(10)",
    "adx": "ADX",
    "adx_pos": "+DI",
    "adx_neg": "-DI",
    "ema_gap_20_50": "EMA20-50갭",
    "ema_gap_50_200": "EMA50-200갭",
    "ema_gap_20_200": "EMA20-200갭",
    "bollinger_pband": "볼린저 P밴드",
    "bollinger_width": "볼린저 폭",
    "keltner_pband": "켈트너 P밴드",
    "keltner_width": "켈트너 폭",
    "obv_z20": "OBV Z20",
    "cmf_20": "CMF(20)",
    "accdist_slope_5": "A/D 기울기(5)",
    "annual_dividend": "연간 배당",
    "dividend_yield": "배당수익률",
    "최근20일평균거래대금": "최근20일평균거래대금(M)",
}
