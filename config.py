"""전략 전역 설정 모음.

프로젝트 전반에서 공유하는 티커 목록, 섹터 매핑, 가중치, 임계치 등을 한 곳에서 관리하여
파일이 나뉘어 있어도 수정 포인트를 명확히 하기 위한 모듈이다."""

import os
from typing import Dict

from data.sp500_tickers import SP500_TICKERS
from data.nasdaq_tickers import NASDAQ_TICKERS

# 분석 대상 종목 풀: S&P 500 + Nasdaq 100 + Market Indices
INDICES = ["SPY", "QQQ", "IWM"]
TICKERS = list(set(SP500_TICKERS + NASDAQ_TICKERS + INDICES))
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
GOOGLE_SHEETS_SIGNALS_ENABLED = False  # Signals 워크시트 업데이트 on/off
GOOGLE_SHEETS_PORTFOLIO_WORKSHEET: str | None = "차트분석"
GOOGLE_SHEETS_PORTFOLIO_ENABLED = True  # 보유주식 워크시트 업데이트 on/off
GOOGLE_SHEETS_PORTFOLIO_TICKER_COLUMN = "티커"

# TradingView 차트 캡처 설정
CHARTS_ENABLED = True  # 차트 캡처 기능 on/off
CHARTS_TIMEFRAMES = ["1H", "4H", "Daily", "Weekly", "Monthly"]  # 캡처할 타임프레임 목록
CHARTS_MIN_SCORE = 0.0  # 차트 캡처 최소 매수적합도 (0.0 = 모든 종목)
CHARTS_OUTPUT_DIR = "charts/screenshots"  # 차트 저장 디렉토리

# TradingView 티커 매핑 (시스템 티커 -> TradingView 티커)
# 예: {"NBM.V": "NBM", "TICKER.TO": "TICKER"}
TRADINGVIEW_TICKER_MAP = {
    "NBM.V": "NBM",
}

# 커스텀 지표(TEMA 등)를 쓰고 싶다면, TradingView에서 차트를 저장하고 그 ID를 입력하세요.
# https://www.tradingview.com/chart/FeHr4sS7/ -> "FeHr4sS7"
TRADINGVIEW_CHART_ID = "Ct9Py1WO"  # None이면 기본값(TEMA 9) 사용

# Google Drive 이미지 호스팅 설정
DRIVE_UPLOAD_ENABLED = False  # Drive 업로드 on/off
DRIVE_FOLDER_NAME = "Stock_Chart_Screenshots"
DRIVE_FOLDER_ID = "1KbDahjFYZF0bPDwIAJ5soBmbOhxFznrU"
DRIVE_SHARE_EMAIL = "chunghwan14@gmail.com"

# GitHub 이미지 호스팅 설정 (Public Repo인 경우 사용)
GITHUB_UPLOAD_ENABLED = True
GITHUB_REPO_NAME = "sean-ccho/Project_1"  # "username/repo_name" 형식
GITHUB_BRANCH_NAME = "main"


# 이메일 알림 설정
EMAIL_ENABLED = True  # 알림 사용 여부
EMAIL_SENDER = "chunghwan14@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")  # Gmail의 경우 '앱 비밀번호' 사용 권장
EMAIL_RECIPIENT = "chunghwan14@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SCORE_THRESHOLD = 4.0  # 알림을 보낼 매수적합도 최소 점수



# 백테스트 자동 실행 설정
BACKTEST_ENABLED = False
BACKTEST_WORKSHEET_NAME = "백테스트"
BACKTEST_RUNS = [
    {
        "label": "Core_1y",
        "period": "1y",
        "max_positions": 8,
        "rebalance_every": 10,
        "max_tickers": None,
        "min_history_days": 220,
        "include_fundamentals": True,
    }
]

# 백테스트 진입 조건 (Entry Score 계산)
BACKTEST_ENTRY_SCORE_MIN = 4.0          # 최소 진입 스코어 (엄격하게 조정)
BACKTEST_LOW_PROB_THRESHOLD = 0.4       # 저점확률 임계치
BACKTEST_REVERSAL_SCORE_MIN = 3.0       # 반등스코어 최소값
BACKTEST_BUY_SIGNAL_WEIGHT = 2.0        # buy_signal 가중치
BACKTEST_LOW_PROB_WEIGHT = 2.0          # 저점확률 가중치 (강화)
BACKTEST_REVERSAL_WEIGHT = 2.0          # 반등스코어 가중치 (강화)
BACKTEST_PATTERN_WEIGHT = 1.0           # 상승 패턴 가중치
BACKTEST_SECTOR_WEIGHT = 0.5            # 강한 섹터 가중치
BACKTEST_TREND_SCORE_WEIGHT = 1.0       # 트렌드점수 가중치 (대체 조건)

# 백테스트 청산 조건
BACKTEST_PROFIT_TARGET = 0.15           # 15% 수익 목표 (보수적)
BACKTEST_TRAILING_STOP = 0.08           # 고점 대비 8% 하락 시 청산 (타이트하게)
BACKTEST_STOP_LOSS = 0.07               # 7% 손절 (위험 관리)
BACKTEST_HIGH_PROB_THRESHOLD = 0.70     # 고점확률 청산 임계치
BACKTEST_MIN_HOLDING_DAYS = 3           # 최소 보유일 (3일로 단축)
BACKTEST_SELL_SIGNAL_ENABLED = False    # 매도신호 청산 비활성화

# 백테스트 리스크 관리
BACKTEST_MAX_DAILY_LOSS = 0.03          # 일일 포트폴리오 최대 손실 3%
BACKTEST_MAX_POSITION_LOSS = 0.10       # 개별 종목 최대 손실 10%
BACKTEST_USE_ATR_SIZING = True          # ATR 기반 포지션 사이징 사용
BACKTEST_BENCHMARK = "SPY"              # 비교 대상 벤치마크
BACKTEST_INITIAL_CASH = 100_000         # 초기 자본금

# 시그널 판정 임계치.
BUY_SCORE_THRESHOLD = 0.20
BUY_POS_THRESHOLD = 0.75
BUY_RSI_MIN = 55
BUY_RSI_MAX = 72
WATCH_SCORE_THRESHOLD = 0.06
WATCH_POS_THRESHOLD = 0.52
OVERBOUGHT_RSI = 78

# 시장 상황 필터 및 전략 모드 설정
MARKET_FILTER_ENABLED = True  # SPY 200일 이평선 하회 시 매수 제한
STRATEGY_MODE = "AGGRESSIVE"  # "STANDARD", "AGGRESSIVE"

# 섹터 로테이션 설정
SECTOR_ROTATION_ENABLED = True  # 강한 섹터 종목만 매수
SECTOR_STRENGTH_LOOKBACK = 20   # 상대 강도 계산 기간 (거래일)
SECTOR_STRENGTH_THRESHOLD = 0.0  # SPY 대비 초과 수익률 기준 (0 = SPY보다 높으면 강함)

# 차트 패턴 인식 설정
PATTERN_LOOKBACK_DAYS = 60          # 패턴 탐색 기간 (거래일)
PATTERN_MIN_TOUCHES = 2             # 추세선 최소 터치 횟수
PATTERN_CONVERGENCE_THRESHOLD = 0.02  # 수렴 판정 기준 (가격 대비 %) - 더 엄격하게
PEAK_PROMINENCE = 0.025             # 고점/저점 탐지 민감도 (가격 대비 %) - 조정됨

# 패턴 필터링 추가 설정
PATTERN_TREND_LOOKBACK = 20         # 트렌드 판단 기간
PATTERN_MIN_R2 = 0.5                # 추세선 최소 R² 값
PATTERN_BREAKOUT_CONFIRM = 0.02     # 돌파 확인 기준 (2%)

# 섹터 ETF 매핑 (섹터명 → ETF 티커)
SECTOR_ETFS: Dict[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

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

# 저점 반등 스코어 설정
BOTTOM_RSI_OVERSOLD = 30              # RSI 과매도 기준
BOTTOM_RSI_RECOVERY = 35              # RSI 반등 확인 기준
BOTTOM_RSI_LOOKBACK = 5               # RSI 반등 확인 기간 (일)
BOTTOM_VOLUME_SURGE_MULT = 1.5        # 거래량 급증 배수
BOTTOM_SUPPORT_TOLERANCE = 0.03       # 지지선 허용 오차 (3%)
BOTTOM_MACD_DIV_LOOKBACK = 10         # MACD 다이버전스 확인 기간
BOTTOM_REVERSAL_THRESHOLD = 5.0       # 반등 점수 기준점

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
    "저점 반등": 1,
    "저점 관찰": 2,
    "관심 관찰": 3,
    "관망 과열": 4,
    "관망 약세": 5,
}

JUDGEMENT_DISPLAY = {
    "매수 후보": "1. 매수 후보",
    "저점 반등": "1. 저점 반등",
    "저점 관찰": "2. 저점 관찰",
    "관심 관찰": "3. 관심 관찰",
    "관망 과열": "4. 관망 과열",
    "관망 약세": "5. 관망 약세",
}

RECOMMENDATION_DISPLAY = {
    "적극 매수": "1. 즉시 진입",
    "반등 매수 고려": "1. 반등 매수",
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
    "패턴_삼각형": "삼각형 패턴",
    "패턴_쐐기": "쐐기 패턴",
    "패턴_더블": "더블바텀/탑",
    "패턴_헤드숄더": "헤드앤숄더",
    "패턴_컵핸들": "컵위드핸들",
    "패턴_캔들": "캔들스틱",
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
    "섹터",
    "매수적합도_표시",  # Entry Score - 매수 적합도 (★ 표시)
    # TradingView 차트 컬럼들
    "차트_1H",
    "차트_4H",
    "차트_Daily",
    "차트_Weekly",
    "차트_Monthly",
    # 기존 컬럼들
    "섹터강도",
    "판단",
    "추천",
    "트렌드점수_최종",
    "buy_signal_text",
    "sell_signal_text",
    # "buy_support_count",  # 매수 보조
    # "sell_support_count",  # 매도 보조
    # "position_size",  # 권장 수량
    # "stop_dist",  # 스탑 거리
    "RSI",
    "거래량Z(20)",
    # "저점확률",
    # "고점확률",
    # "극점편차",
    "패턴_삼각형",
    "패턴_쐐기",
    "패턴_더블",
    "패턴_헤드숄더",
    "패턴_컵핸들",
    "패턴_캔들",
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
    # 저점 반등 지표
    "반등스코어",
    "RSI반등",
    "볼린저바운스",
    "저점거래량급증",
    "지지선테스트",
    "MACD다이버전스",
]
