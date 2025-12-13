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
    "ret5": 0.45,        # 5일 수익률 기반 단기 모멘텀 가중 강화
    "vol": 0.20,         # 거래량 급증 정도
    "break": 0.30,       # 52주 고저 대비 위치(돌파 정도)
    "vola": -0.15,       # ATR% 변동성 (높을수록 감점)
    "rsi": 0.10,         # RSI 기반 스무딩 점수
    "accel": 0.12,       # 단기 대비 중기 모멘텀 가속도
    "vola_penalty": -0.18,  # ATR% 비선형 패널티
}

# 시장별(US/CA) 거래대금 분위수를 이용한 유동성 컷 비율.
LIQUIDITY_QUANTILE = 0.25  # 하위 25% 제거 = 상위 75% 유지
LIQUIDITY_WHITELIST = []  # 필수 포함 종목(저유동성이라도 유지)
LIQUIDITY_DOLLAR_MIN = 5_000_000  # 최근 20일 평균 거래대금 최소선
LIQUIDITY_TURNOVER_MIN = 0.0  # 거래대금/시가총액(또는 유통시가) 회전율 하한(0=미사용)

for ticker in LIQUIDITY_WHITELIST:
    if ticker not in TICKERS:
        TICKERS.append(ticker)

# Google Sheets 연결 설정(기본 비활성화)
GOOGLE_SHEETS_ENABLED = True
GOOGLE_SHEETS_CREDENTIALS_PATH: str | None = "gspread-service-account.json"
GOOGLE_SHEETS_SPREADSHEET_ID: str | None = "1VxzTPfDvRX0UGJZcNnRsqTRex4gqmjv4P-1I_VcuXaw"
GOOGLE_SHEETS_SIGNALS_WORKSHEET = "Signals"
GOOGLE_SHEETS_PORTFOLIO_WORKSHEET: str | None = "보유주식"
GOOGLE_SHEETS_PORTFOLIO_TICKER_COLUMN = "티커"


# 백테스트 자동 실행 설정
BACKTEST_ENABLED = False
BACKTEST_WORKSHEET_NAME = "백테스트"
BACKTEST_RUNS = [
    {
        "label": "Core_1y",
        "period": "1y",
        "max_positions": 6,
        "rebalance_every": 5,
        "max_tickers": None,
        "min_history_days": 220,
        "include_fundamentals": True,
    }
]

# 시그널 판정 임계치.
BUY_SCORE_THRESHOLD = 0.20
BUY_POS_THRESHOLD = 0.75
BUY_RSI_MIN = 55
BUY_RSI_MAX = 72
WATCH_SCORE_THRESHOLD = 0.06
WATCH_POS_THRESHOLD = 0.52
OVERBOUGHT_RSI = 78

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
BOLLINGER_OVERSOLD_PBAND = 0.10

STOCH_OVERSOLD = 20
RSI_OVERSOLD = 35

BOTTOM_POS_THRESHOLD = 0.35
BOTTOM_TREND_SCORE = -0.02
BOTTOM_VOLUME_Z = -0.3
MACD_BOTTOM_THRESHOLD = -0.3

# 저점 탐색 보조 지표 임계치
FUND_HEALTH_ROE_MIN = 0.08
FUND_HEALTH_REVENUE_GROWTH_MIN = 0.03
FUND_HEALTH_PROFIT_MARGIN_MIN = 0.05
FUND_HEALTH_EARNINGS_GROWTH_MIN = 0.02
FUND_HEALTH_DEBT_TO_EQUITY_MAX = 200.0
FUND_HEALTH_CURRENT_RATIO_MIN = 1.0

REL_STRENGTH_LOOKBACK = 20
REL_STRENGTH_SECTOR_BUFFER = -0.05
REL_STRENGTH_MARKET_BUFFER = -0.05

VOLUME_STABILITY_RATIO_MIN = 0.6
LONG_TERM_SLOPE_LOOKBACK = 20
LONG_TERM_SLOPE_MIN = -0.03
EMA200_DISTANCE_MIN = -0.25

GAP_DOWN_EXTREME = -0.04
HAMMER_LOWER_SHADOW_MIN = 0.5
HAMMER_UPPER_SHADOW_MAX = 0.25
INTRADAY_RECOVERY_MIN = 0.04

EARNINGS_EVENT_WINDOW_DAYS = 9
EARNINGS_SOON_DAYS = 7
EXTREME_LOW_LOOKBACK = 10
EXTREME_HIGH_LOOKBACK = 10

# 극단 구간(저점/고점) 라벨링 및 모델링 설정
EXTREME_LABEL_LOOKAHEAD = 10
EXTREME_LOW_MIN_RETURN = 0.07
EXTREME_HIGH_MAX_DRAWDOWN = -0.07
EXTREME_HISTORY_STEP = 3
EXTREME_MODEL_ENABLED = True
EXTREME_MODEL_TRAIN_PERIOD = "2y"
EXTREME_MODEL_MIN_SAMPLES = 400
EXTREME_MODEL_REG_C = 1.5
EXTREME_MODEL_FEATURES = [
    "RSI",
    "bollinger_pband",
    "10일저점괴리",
    "10일고점괴리",
    "거래량돌파배수",
    "변동성압축",
    "장중반등률",
    "accdist_slope_5",
    "obv_z20",
    "cmf_20",
    "ema_gap_20_50",
    "ema_gap_50_200",
    "52주포지션",
    "5일수익률",
    "20일수익률",
    "시장상대강도",
    "섹터상대강도",
]

# --- Simplified signal configuration (indicator set restricted to EMA/RSI/MACD/Volume/ADX/OBV/ATR) ---
VOLUME_ROLLING_WINDOW = 20
VOLUME_BREAKOUT_MULTIPLIER = 1.2

OBV_ROLLING_WINDOW = 20
OBV_MOMENTUM_LOOKBACK = 5

ATR_MEDIAN_LOOKBACK = 252
ATR_BUY_THRESHOLD_MULTIPLIER = 1.5
ATR_SELL_THRESHOLD_MULTIPLIER = 1.8

RSI_BUY_MAX = 55
RSI_SELL_MIN = 70

ADX_BUY_MIN = 20
ADX_SELL_MAX = 18

RISK_PER_TRADE = 0.01
ATR_POSITION_MULTIPLE = 2.5
DEFAULT_EQUITY = 100_000

# 판단·추천 강도를 위한 세부 임계치 조정
BUY_POSITIVE_MIN = 6
BUY_NEGATIVE_MAX = 1
WATCH_POSITIVE_MIN = 4
WATCH_NEGATIVE_MAX = 2
BOTTOM_NEGATIVE_TOLERANCE = 3
BUY_REL_STRENGTH_MIN = 0.02
SECTOR_REL_STRENGTH_MIN = -0.01

# 변동성 패널티 구간
VOLATILITY_PENALTY_START = 0.025
VOLATILITY_PENALTY_END = 0.08

# 시그널 우선순위 매핑(테이블 정렬 순서를 통제).
SIGNAL_PRIORITY = {
    "매수 후보": 0,
    "저점 관찰": 1,
    "관심 관찰": 2,
    "관망 과열": 3,
    "관망 약세": 4,
}

JUDGEMENT_DISPLAY = {
    "매수 후보": "1. 매수 후보",
    "저점 관찰": "2. 저점 관찰",
    "관심 관찰": "3. 관심 관찰",
    "관망 과열": "4. 관망 과열",
    "관망 약세": "5. 관망 약세",
}

RECOMMENDATION_DISPLAY = {
    "적극 매수": "1. 즉시 진입",
    "분할 매수": "2. 분할 매수",
    "저점 분할 매수": "2. 분할 매수",
    "조건 확인 후 매수": "3. 조건 확인",
    "저점 매수 대기": "3. 조건 확인",
    "조건 충족 시 매수": "3. 조건 확인",
    "반등 모니터링": "4. 모니터링/보유",
    "추가 관찰": "4. 모니터링/보유",
    "관망/보유": "4. 모니터링/보유",
    "차익 실현 고려": "5. 차익/관망",
    "고평가 관망": "5. 차익/관망",
}

# 퍼센티지로 표현할 지표 목록(콘솔 및 CSV 출력 시 사용).
PERCENT_COLUMNS = [
    "트렌드점수_최종",
    "트렌드점수",
    "5일수익률",
    "20일수익률",
    "52주포지션",
    "ATR%",
    "ema_gap_20_50",
    "ema_gap_50_200",
    "bollinger_pband",
    "accdist_slope_5",
    "dividend_yield",
    "close_to_ema200_pct",
    "거래대금안정비",
    "시장상대강도",
    "섹터상대강도",
    "장중반등률",
    "10일저점괴리",
    "갭하락률",
    "atr_med_252",
    "atr_buy_max",
    "atr_sell_max",
]

# 실행 때마다 백업(타임스탬프) CSV를 추가로 만들 것인지 여부.
EXPORT_WITH_BACKUP = False

TECH_COLUMN_LABELS = {
    "macd_hist": "MACD 히스토그램",
    "stoch_k": "스토캐스틱%K",
    "roc_10": "ROC(10)",
    "adx": "ADX",
    "ema_gap_20_50": "EMA20-50갭",
    "ema_gap_50_200": "EMA50-200갭",
    "bollinger_pband": "볼린저 P밴드",
    "obv_z20": "OBV Z20",
    "cmf_20": "CMF(20)",
    "accdist_slope_5": "A/D 기울기(5)",
    "annual_dividend": "연간 배당",
    "dividend_yield": "배당수익률",
    "최근20일평균거래대금": "최근20일평균거래대금(M)",
    "ema20": "EMA20",
    "ema50": "EMA50",
    "volume": "거래량(최근)",
    "volume_ma20": "거래량MA20",
    "obv": "OBV",
    "obv_ma20": "OBV MA20",
    "obv_mom_5": "OBV 모멘텀(5)",
    "obv_mom_ratio": "OBV 모멘텀(20)",
    "atr_pct": "ATR%",
    "거래량돌파배수": "거래량돌파배수",
    "거래량Z(20)": "거래량Z(20)",
    "변동성압축": "변동성압축",
    "10일고점괴리": "10일고점괴리",
    "atr_med_252": "ATR% 중앙값(1y)",
    "atr_buy_max": "ATR 매수 한도",
    "atr_sell_max": "ATR 매도 한도",
    "atr_value": "ATR(가격기준)",
    "stop_dist": "스탑 거리",
    "position_size": "권장 수량",
    "buy_signal": "매수 신호",
    "sell_signal": "매도 신호",
    "buy_signal_text": "매수 신호",
    "sell_signal_text": "매도 신호",
    "buy_support_count": "매수 보조",
    "sell_support_count": "매도 보조",
    "close": "종가",
    "저점확률": "저점확률",
    "고점확률": "고점확률",
    "극점편차": "극점편차",
}

# 후처리 및 출력 단계에서 공통으로 사용하는 컬럼 정의.
NEUTRALIZE_COLUMNS = ["트렌드점수", "5일수익률", "거래량Z(20)"]

DISPLAY_COLUMNS = [
    # "판단",
    # "추천",
    # "회사",
    # "긍정",
    # "저점",
    # "저점강도",
    # "저점근거",
    # "저점점수",
    # "저점건강",
    # "상대강도",
    # "이벤트주의",
    # "경계",
    # "티커",
    # "현재가격",
    # "우선순위",
    # "트렌드점수_최종",
    # "트렌드점수",
    # "RSI",
    # "macd",
    # "annual_dividend",
    # "dividend_yield",
    # "5일수익률",
    # "1일수익률",
    # "52주포지션",
    # "거래량Z(20)",
    # "ATR%",
    # "macd_signal",
    # "macd_hist",
    # "stoch_k",
    # "stoch_d",
    # "roc_10",
    # "adx",
    # "adx_pos",
    # "adx_neg",
    # "ema_gap_20_50",
    # "ema_gap_50_200",
    # "ema_gap_20_200",
    # "bollinger_pband",
    # "bollinger_width",
    # "keltner_pband",
    # "keltner_width",
    # "obv_z20",
    # "cmf_20",
    # "accdist_slope_5",
    # "최근20일평균거래대금",
]

EXPORT_COLUMNS = [
    "티커",
    "회사",
    "현재가격",
    "최근뉴스",
    "우선순위",
    "판단",
    "추천",
    "트렌드점수_최종",
    "buy_signal_text",
    "sell_signal_text",
    "buy_support_count",
    "sell_support_count",
    "position_size",
    "stop_dist",
    "RSI",
    "거래량Z(20)",
    "저점확률",
    "고점확률",
    "극점편차",
    # "macd_hist",
    # "ema20",
    # "ema50",
    # "atr_pct",
    # "atr_buy_max",
    # "atr_sell_max",
    # "volume",
    # "volume_ma20",
    # "obv",
    # "obv_ma20",
    # "obv_mom_5",
    # "obv_mom_ratio",
    "dividend_yield",
]
