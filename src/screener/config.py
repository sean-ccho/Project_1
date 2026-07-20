"""전략 전역 설정 모음.

프로젝트 전반에서 공유하는 티커 목록, 섹터 매핑, 가중치, 임계치 등을 한 곳에서 관리하여
파일이 나뉘어 있어도 수정 포인트를 명확히 하기 위한 모듈이다."""

import os
from typing import Dict

from data.sp500_tickers import SP500_TICKERS
from data.nasdaq_tickers import NASDAQ_TICKERS

# 분석 대상 종목 풀: S&P 500 + Nasdaq 100 + Market Indices
INDICES = ["SPY", "QQQ", "IWM"]

# DEV_MODE=1 환경변수 설정 시 소수 종목으로 빠르게 테스트
_DEV_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "JNJ", "XOM",
    "SPY", "QQQ", "IWM",
]
TICKERS = _DEV_TICKERS if os.environ.get("DEV_MODE") == "1" else list(set(SP500_TICKERS + NASDAQ_TICKERS + INDICES))
COMPANY_NAME_MAP = {}

# 섹터 매핑: 섹터 중립화 시 Unknown 여부를 판단하는 기준이 된다.
SECTOR_MAP: Dict[str, str] = {
    # US — Information Technology
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AMD": "Information Technology",
    "INTC": "Information Technology", "ACN": "Information Technology",
    "ADBE": "Information Technology", "ADI": "Information Technology",
    "AMAT": "Information Technology", "ANET": "Information Technology",
    "APH": "Information Technology", "CDNS": "Information Technology",
    "CSCO": "Information Technology", "DELL": "Information Technology",
    "IBM": "Information Technology", "KLAC": "Information Technology",
    "MU": "Information Technology", "ORCL": "Information Technology",
    "PANW": "Information Technology", "PLTR": "Information Technology",
    "QCOM": "Information Technology", "SNPS": "Information Technology",
    "TER": "Information Technology", "TXN": "Information Technology",
    "WDAY": "Information Technology", "XYZ": "Information Technology",
    # US — Communication Services
    "META": "Communication Services", "GOOGL": "Communication Services",
    "GOOG": "Communication Services", "NFLX": "Communication Services",
    "APP": "Communication Services", "T": "Communication Services",
    "TMUS": "Communication Services", "VZ": "Communication Services",
    # US — Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "ABNB": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "CCL": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "F": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "ULTA": "Consumer Discretionary",
    # US — Consumer Staples
    "COST": "Consumer Staples", "DG": "Consumer Staples",
    "KO": "Consumer Staples", "MDLZ": "Consumer Staples",
    "PEP": "Consumer Staples", "PGR": "Consumer Staples",
    "PM": "Consumer Staples", "TGT": "Consumer Staples",
    "WMT": "Consumer Staples",
    # US — Health Care
    "ABBV": "Health Care", "ABT": "Health Care", "AMGN": "Health Care",
    "BDX": "Health Care", "BMY": "Health Care", "BSX": "Health Care",
    "CI": "Health Care", "CVS": "Health Care", "DHR": "Health Care",
    "ISRG": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "MCK": "Health Care", "MDT": "Health Care", "MRNA": "Health Care",
    "PFE": "Health Care", "REGN": "Health Care", "TMO": "Health Care",
    "VRTX": "Health Care",
    # US — Financials
    "AXP": "Financials", "BAC": "Financials", "BRK-B": "Financials",
    "C": "Financials", "COF": "Financials", "GS": "Financials",
    "JPM": "Financials", "MS": "Financials", "PYPL": "Financials",
    "SCHW": "Financials", "V": "Financials", "WFC": "Financials",
    # US — Industrials
    "AXON": "Industrials", "BA": "Industrials", "CAT": "Industrials",
    "DE": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
    "UAL": "Industrials", "UNP": "Industrials",
    # US — Energy
    "CVX": "Energy", "DVN": "Energy", "SLB": "Energy",
    # US — Materials
    "FCX": "Materials", "NEM": "Materials",
    # US — Utilities
    "CEG": "Utilities", "NEE": "Utilities",
    # US — Real Estate
    "WELL": "Real Estate",
    # CA (캐나다)
    "SHOP.TO": "Information Technology",
    "NTR.TO": "Materials",
    "BNS.TO": "Financials", "BMO.TO": "Financials",
    "SU.TO": "Energy", "ENB.TO": "Energy", "CNQ.TO": "Energy",
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

# 다중 팩터 알파 모델 설정
ALPHA_MODEL_ENABLED = True           # False이면 기존 WEIGHTS 방식으로 폴백
ALPHA_IC_LOOKBACK = 60               # IC 계산 기간 (거래일)
ALPHA_IC_MIN_SAMPLES = 30            # IC 계산 최소 종목 수
ALPHA_FORWARD_RETURN_DAYS = 5        # IC 계산용 미래 수익률 기간

# 다중 팩터 기본 가중치 (IC 계산 불가 시 폴백)
ALPHA_DEFAULT_WEIGHTS = {
    "momentum": 0.20,
    "trend": 0.20,
    "volume": 0.25,
    "volatility": 0.15,
    "mean_reversion": 0.20,
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
GOOGLE_SHEETS_SIGNALS_WORKSHEET = "[SP500 분석]"
GOOGLE_SHEETS_SIGNALS_ENABLED = True  # Signals 워크시트 업데이트 on/off
GOOGLE_SHEETS_PORTFOLIO_WORKSHEET: str | None = "[SP500 차트분석]"
GOOGLE_SHEETS_PORTFOLIO_ENABLED = True  # 차트분석(포트폴리오) 워크시트 업데이트 on/off
GOOGLE_SHEETS_PORTFOLIO_TICKER_COLUMN = "티커"

# Signals2 워크시트 설정 (전 종목 스캔 결과)
GOOGLE_SHEETS_SIGNALS2_WORKSHEET = "[NASDAQ/NYSE 분석]"
GOOGLE_SHEETS_SIGNALS2_ENABLED = True  # Signals2 워크시트 업데이트 on/off
GOOGLE_SHEETS_SIGNALS2_CHART_WORKSHEET = "[NASDAQ/NYSE 차트분석]"
GOOGLE_SHEETS_SIGNALS2_CHART_ENABLED = True  # NASDAQ/NYSE 차트분석 워크시트 업데이트 on/off

# 전 종목 스캔: 바닥 탈출(Turnaround) 전략 임계값
TURNAROUND_MIN_DROP = -0.50          # 52주 고점 대비 최소 낙폭 (-50%)
TURNAROUND_MA200_LOOKBACK = 5       # 200일선 돌파 확인 기간 (거래일)
TURNAROUND_VOLUME_MULT = 2.0        # 돌파 시 최소 거래량 배수 (50일 평균 대비)

# 전 종목 스캔: 강세 돌파(Momentum) 전략 임계값
MOMENTUM_HIGH_THRESHOLD = 0.95       # 52주 고점 대비 위치 (0.95 = 5% 이내)
MOMENTUM_MIN_GAIN_20D = 0.15         # 최근 20일 최소 수익률 (15%)


# TradingView 차트 캡처 설정
CHARTS_ENABLED = True  # 차트 캡처 기능 on/off
CHARTS_TIMEFRAMES = ["1H", "4H", "Daily", "Weekly", "Monthly"]  # 캡처할 타임프레임 목록
CHARTS_MIN_SCORE = 0.0  # 차트 캡처 최소 매수적합도 (0.0 = 모든 종목)
CHARTS_OUTPUT_DIR = "charts/screenshots"          # SP500 차트 저장 디렉토리
CHARTS_SIGNALS2_OUTPUT_DIR = "charts/screenshots_nasdaq"  # NASDAQ/NYSE 차트 저장 디렉토리 (SP500과 분리)

# TradingView 티커 매핑 (시스템 티커 -> TradingView 티커)
# 예: {"NBM.V": "NBM", "TICKER.TO": "TICKER"}
TRADINGVIEW_TICKER_MAP = {
    "NBM.V": "NBM",
}

# 커스텀 지표(TEMA 등)를 쓰고 싶다면, TradingView에서 차트를 저장하고 그 ID를 입력하세요.
# https://www.tradingview.com/chart/FeHr4sS7/ -> "FeHr4sS7"
TRADINGVIEW_CHART_ID = "gr1tsHcA"  # None이면 기본값(TEMA 9) 사용

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
EMAIL_RECIPIENTS = [
    "chunghwan14@gmail.com",
    "roykim0311@gmail.com",
    "ssamjungtan@naver.com",
]
EMAIL_RECIPIENT = EMAIL_RECIPIENTS[0]  # 하위 호환 (기존 참조 안전)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SCORE_THRESHOLD = 4.0          # 통합 점수 (이메일 발송 기준)
EMAIL_BOTTOM_SCORE_THRESHOLD = 5.0   # 바닥반등 점수 (이메일 발송 기준)
EMAIL_MOMENTUM_SCORE_THRESHOLD = 5.0 # 모멘텀 점수 (이메일 발송 기준)


# ── 내부자 거래(openinsider) — 표시용 ──
# 이메일 후보 및 차트분석 시트(SP500/NASDAQ)에 '내부자(90일)' 요약 컬럼을 표시한다.
# 스코어링·매매 로직에는 영향을 주지 않는다.
INSIDER_ENABLED = True               # 내부자 요약 수집/표시 on/off
INSIDER_LOOKBACK_DAYS = 90           # openinsider 조회 기간(일)
INSIDER_CLUSTER_MIN_BUYERS = 2       # 매수 임원 수 ≥ 이 값이면 '클러스터매수'
INSIDER_REQUEST_DELAY_SEC = 0.4      # 티커 간 요청 지연(차단 방지)


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
PATTERN_MIN_TOUCHES = 3             # 추세선 최소 터치 횟수 (2→3: 노이즈 감소)
PATTERN_CONVERGENCE_THRESHOLD = 0.015  # 수렴 판정 기준 (가격 대비 %) - 엄격화
PEAK_PROMINENCE = 0.03              # 고점/저점 탐지 민감도 (가격 대비 %) - 노이즈 감소

# 패턴 필터링 추가 설정
PATTERN_TREND_LOOKBACK = 20         # 트렌드 판단 기간
PATTERN_MIN_R2 = 0.65               # 추세선 최소 R² 값 (0.5→0.65) — 기울어진 추세선에만 적용
PATTERN_FLAT_TOLERANCE = 0.04       # 수평 추세선(상승/하락 삼각형) 터치 허용 밴드 (평균 대비)
                                    #   수평선은 기울기≈0이라 R²가 구조적으로 0 → R² 대신 평탄도로 검증
PATTERN_BREAKOUT_CONFIRM = 0.02     # 돌파 확인 기준 (2%)
PATTERN_VOLUME_SURGE = 1.5          # 돌파 거래량 기준 (20일 평균의 배수)
PATTERN_MIN_GAP_DAYS = 10           # 더블 바텀/탑 두 극점 최소 간격 (거래일)
PATTERN_DOUBLE_TOLERANCE = 0.02     # 더블 바텀/탑 가격 허용치 (3%→2%)
PATTERN_SHOULDER_TIME_RATIO = (0.5, 2.0)  # H&S 어깨 시간 대칭 허용 범위

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

ADX_BUY_MIN = 25
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

# 바닥 반등 전략 가중치
BOTTOM_STRATEGY_WEIGHTS = {
    "저점확률": 2.5,       # AI 저점 예측 (핵심)
    "반등스코어": 2.5,     # 기술적 반등 지표 (핵심)
    "RSI_과매도": 1.5,     # RSI < 30
    "RSI_탈출": 1.0,       # RSI 30~40
    "RSI_중립하단": 0.5,   # RSI 40~45
    "상승패턴": 1.0,       # 이중바닥, 역헤드숄더 등
    "급락반등": 0.5,       # 5일 수익률 < -8%
    "강한섹터": 0.5,       # 강세 섹터 소속
    "RSI_과매수감점": -1.5, # RSI > 75
    "RSI_극과매수감점": -2.0, # RSI > 80
    "급등감점": -1.0,      # 5일 수익률 > 20%
    "볼린저과열감점": -1.0, # 볼린저 상단 돌파
    # 알파 모델 팩터
    "팩터_평균회귀_강": 1.5,   # 강한 과매도 반등 신호
    "팩터_평균회귀_약": 0.5,   # 약한 반등 신호
    "팩터_변동성": 0.5,       # 변동성 압축 (반등 에너지 축적)
}

# 모멘텀 추격 전략 가중치
MOMENTUM_STRATEGY_WEIGHTS = {
    "매수신호": 2.5,       # EMA/MACD 정배열 + 보조지표 (핵심)
    "트렌드점수": 2.0,     # 모멘텀 강도 (핵심)
    "EMA정배열": 1.5,      # EMA20 > EMA50
    "거래량돌파": 1.0,     # 거래량 급증
    "ADX강세": 1.0,        # 추세 강도 (ADX > 25)
    "강한섹터": 0.5,       # 강세 섹터 소속
    "상승패턴": 0.5,       # 상승 차트 패턴
    "RSI_과매수감점": -1.5, # RSI > 75
    "급등감점": -1.0,      # 5일 수익률 > 20%
    "볼린저과열감점": -1.0, # 볼린저 상단 돌파
    # 알파 모델 팩터
    "팩터_모멘텀_강": 1.5,   # 강한 다중기간 모멘텀
    "팩터_모멘텀_약": 0.5,   # 약한 모멘텀
    "팩터_추세": 1.0,         # 강한 추세 확인
}

# 주봉(Weekly) 패턴 가중치 — 바닥반등용 (반전형 강조, 추세형 50% 감소)
WEEKLY_PATTERN_WEIGHTS_BOTTOM = {
    # 반전형 패턴 → 유지
    "이중바닥": 1.5,                   # 주봉 이중바닥 (강력한 지지)
    "역헤드앤숄더": 1.5,              # 주봉 역헤드앤숄더 (장기 바닥)
    "하락쐐기": 1.0,                  # 주봉 하락쐐기 (반전)
    "강세잉걸핑": 0.5,                # 주봉 강세 잉걸핑
    "모닝스타": 0.5,                  # 주봉 모닝스타
    # 추세형 패턴 → 50% 감소
    "골든크로스": 1.0,                # 주봉 골든크로스 (추세형 → 감소)
    "골든크로스임박": 0.75,           # 주봉 골든크로스 임박 (추세형 → 감소)
    "컵앤핸들": 0.75,                # 주봉 컵앤핸들 (추세형 → 감소)
    "상승삼각형": 0.5,               # 주봉 상승삼각형 (추세형 → 감소)
    "정배열(완전)": 0.75,             # 주봉 완전 정배열 (추세형 → 감소)
    "정배열(눌림목)": 0.5,            # 주봉 눌림목 정배열 (추세형 → 감소)
    "정배열(형성중)": 0.25,           # 주봉 정배열 형성중 (추세형 → 감소)
}

# 주봉(Weekly) 패턴 가중치 — 모멘텀용 (추세형 강조, 반전형 50% 감소)
WEEKLY_PATTERN_WEIGHTS_MOMENTUM = {
    # 추세형 패턴 → 유지
    "골든크로스": 2.0,                # 주봉 골든크로스 (대세 상승)
    "골든크로스임박": 1.5,            # 주봉 골든크로스 임박 (선행 신호)
    "컵앤핸들": 1.5,                  # 주봉 컵앤핸들 (지속형 상승)
    "상승삼각형": 1.0,                # 주봉 상승삼각형
    "정배열(완전)": 1.5,              # 주봉 완전 정배열
    "정배열(눌림목)": 1.0,            # 주봉 눌림목 정배열
    "정배열(형성중)": 0.5,            # 주봉 정배열 형성중
    # 반전형 패턴 → 50% 감소
    "이중바닥": 0.75,                 # 주봉 이중바닥 (반전형 → 감소)
    "역헤드앤숄더": 0.75,             # 주봉 역헤드앤숄더 (반전형 → 감소)
    "하락쐐기": 0.5,                  # 주봉 하락쐐기 (반전형 → 감소)
    "강세잉걸핑": 0.25,               # 주봉 강세 잉걸핑 (반전형 → 감소)
    "모닝스타": 0.25,                 # 주봉 모닝스타 (반전형 → 감소)
}

# 월봉(Monthly) 패턴 가중치 — 바닥반등용 (반전형 강조, 추세형 50% 감소)
MONTHLY_PATTERN_WEIGHTS_BOTTOM = {
    # 반전형 패턴 → 유지
    "이중바닥": 2.0,                  # 월봉 이중바닥 (역사적 저점)
    "역헤드앤숄더": 2.0,              # 월봉 역헤드앤숄더 (역사적 바닥)
    "하락쐐기": 1.5,                  # 월봉 하락쐐기
    "강세잉걸핑": 1.0,                # 월봉 강세 잉걸핑
    "모닝스타": 1.0,                  # 월봉 모닝스타
    # 추세형 패턴 → 50% 감소
    "골든크로스": 1.5,                # 월봉 골든크로스 (추세형 → 감소)
    "골든크로스임박": 1.0,            # 월봉 골든크로스 임박 (추세형 → 감소)
    "컵앤핸들": 1.0,                  # 월봉 컵앤핸들 (추세형 → 감소)
    "상승삼각형": 0.75,               # 월봉 상승삼각형 (추세형 → 감소)
    "정배열(완전)": 1.0,              # 월봉 완전 정배열 (추세형 → 감소)
    "정배열(눌림목)": 0.75,           # 월봉 눌림목 정배열 (추세형 → 감소)
    "정배열(형성중)": 0.5,            # 월봉 정배열 형성중 (추세형 → 감소)
}

# 월봉(Monthly) 패턴 가중치 — 모멘텀용 (추세형 강조, 반전형 50% 감소)
MONTHLY_PATTERN_WEIGHTS_MOMENTUM = {
    # 추세형 패턴 → 유지
    "골든크로스": 3.0,                # 월봉 골든크로스 (초장기 대세 상승)
    "골든크로스임박": 2.0,            # 월봉 골든크로스 임박 (선행 신호)
    "컵앤핸들": 2.0,                  # 월봉 컵앤핸들 (수년간의 매집 후 상승)
    "상승삼각형": 1.5,                # 월봉 상승삼각형
    "정배열(완전)": 2.0,              # 월봉 완전 정배열
    "정배열(눌림목)": 1.5,            # 월봉 눌림목 정배열
    "정배열(형성중)": 0.75,           # 월봉 정배열 형성중
    # 반전형 패턴 → 50% 감소
    "이중바닥": 1.0,                  # 월봉 이중바닥 (반전형 → 감소)
    "역헤드앤숄더": 1.0,              # 월봉 역헤드앤숄더 (반전형 → 감소)
    "하락쐐기": 0.75,                 # 월봉 하락쐐기 (반전형 → 감소)
    "강세잉걸핑": 0.5,                # 월봉 강세 잉걸핑 (반전형 → 감소)
    "모닝스타": 0.5,                  # 월봉 모닝스타 (반전형 → 감소)
}

# 패턴 가산점 상한 (패턴만으로 5.0 초과 방지 → 전략 고유 신호 필수)
PATTERN_SCORE_CAP_BOTTOM = 3.0      # 바닥반등: 주봉+월봉 패턴 최대 3.0점
PATTERN_SCORE_CAP_MOMENTUM = 3.0    # 모멘텀: 주봉+월봉 패턴 최대 3.0점

# 알파 팩터 점수 임계치 (두 전략 공통)
ALPHA_FACTOR_STRONG_THRESHOLD = 0.3   # 강한 신호 임계치
ALPHA_FACTOR_WEAK_THRESHOLD = 0.1     # 약한 신호 임계치

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
    "fund_institutional_holders_pct",
    "fund_short_pct_float",
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
    "fund_institutional_holders_pct": "기관보유%",
    "fund_short_pct_float": "공매도%",
    "fund_short_interest": "공매도주수",
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
    "섹터",
    "전략구분",              # 전환구간 / 바닥반등 / 모멘텀 / —
    "바닥반등_적합도_표시",  # 바닥 반등 전략 적합도 (★ 표시)
    "모멘텀_적합도_표시",    # 모멘텀 추격 전략 적합도 (★ 표시)
    "내부자(90일)",          # openinsider 내부자 매수/매도 90일 요약 (표시용)
    # TradingView 차트 컬럼들
    "차트_1H",
    "차트_4H",
    "차트_Daily",
    "차트_Weekly",
    "차트_Monthly",
    "일봉패턴",   # 요약
    "주봉패턴",  # 요약
    "월봉패턴",  # 요약
    # 기존 컬럼들
    "섹터강도",
    # "판단",
    # "추천",
    # "트렌드점수_최종",
    # "buy_signal_text",
    # "sell_signal_text",
    # "buy_support_count",  # 매수 보조
    # "sell_support_count",  # 매도 보조
    # "position_size",  # 권장 수량
    # "stop_dist",  # 스탑 거리
    "RSI",
    "거래량Z(20)",
    "[시총]",
    "[헬스체크]",
    "fund_institutional_holders_pct",
    "fund_short_pct_float",
    "fund_short_interest",
    # "저점확률",
    # "고점확률",
    # "극점편차",

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
    # "dividend_yield",
    # 저점 반등 지표
    # "반등스코어",
    # "RSI반등",
    # "볼린저바운스",
    # "저점거래량급증",
    # "지지선테스트",
    # "MACD다이버전스",
]

# ═══════════════════════════════════════════════════════════════
# Paper Trading
# ═══════════════════════════════════════════════════════════════

PAPER_TRADING_ENABLED = True
PAPER_TRADING_MAX_POSITIONS = 3
PAPER_TRADING_MAX_DAILY_BUY = 1
PAPER_TRADING_PROFIT_TARGET = 0.15     # +15%
PAPER_TRADING_STOP_LOSS = 0.07         # -7%
PAPER_TRADING_TRAILING_STOP = 0.05     # 고점 대비 -5% (기존 -8%)
PAPER_TRADING_MAX_HOLDING_DAYS = 21    # 3주 (기존 2주)
PAPER_TRADING_STALE_MIN_RETURN = 0.02  # 21일 후 최소 수익률 (기존 3%)
PAPER_TRADING_DATA_DIR = "data/paper_trading"

# 후보 선정 Hard Filters
CANDIDATE_MIN_STRATEGY_SCORE = 6.0     # 전략 점수 최소 (기존 5.0)
CANDIDATE_BOTTOM_MAX_52W_POS = 0.65    # 바닥반등 전략: 52주 포지션 상한 (이미 바닥 아닌 종목 차단)
CANDIDATE_CCS_MIN_NORMAL = 0.40        # CCS 최소 임계값 - 일반 (기존 0.35)
CANDIDATE_CCS_MIN_BEAR = 0.45          # CCS 최소 임계값 - 약세장
CANDIDATE_RSI_MAX = 75
CANDIDATE_BOLLINGER_MAX = 0.95
CANDIDATE_5D_RETURN_MAX = 0.15
CANDIDATE_LIQUIDITY_MIN = 10_000_000   # $10M 일평균 거래대금
CANDIDATE_EARNINGS_BUFFER_DAYS = 3
CANDIDATE_MAX_SAME_SECTOR = 2

# CCS 가중치 (시장 레짐별)
CANDIDATE_REGIME_WEIGHTS = {
    "bull":    {"strategy": 0.25, "timing": 0.20, "alpha": 0.15, "risk": 0.20, "confluence": 0.20},
    "neutral": {"strategy": 0.25, "timing": 0.20, "alpha": 0.15, "risk": 0.25, "confluence": 0.15},
    "bear":    {"strategy": 0.20, "timing": 0.25, "alpha": 0.10, "risk": 0.30, "confluence": 0.15},
}

# 전략별 Exit 파라미터
EXIT_PARAMS: dict[str, dict[str, float]] = {
    "바닥반등": {
        "profit_target": 0.18,       # +18% (바닥 반등은 업사이드 큼)
        "stop_loss": 0.10,           # -10% (초기 변동성 감안 넓은 스탑)
        "trailing_stop": 0.06,       # 고점 대비 -6%
        "max_holding_days": 25,      # 바닥 확인 후 반등에 시간 필요
        "stale_min_return": 0.02,
        "time_profit_days": 12,      # 시간 익절 기준일
        "time_profit_min": 0.07,     # 시간 익절 최소 수익률
    },
    "모멘텀": {
        "profit_target": 0.12,       # +12% (빠른 익절)
        "stop_loss": 0.10,           # -10% (바닥반등과 동일 수준으로 완화)
        "trailing_stop": 0.05,       # 고점 대비 -5% (트레일링 후 60% 반등 → 완화)
        "max_holding_days": 18,      # 모멘텀은 빠르게 실현
        "stale_min_return": 0.02,
        "time_profit_days": 8,       # 더 빠른 시간 익절
        "time_profit_min": 0.05,     # +5%면 충분
    },
}
EXIT_PARAMS_DEFAULT: dict[str, float] = {
    "profit_target": 0.15,
    "stop_loss": 0.10,
    "trailing_stop": 0.05,
    "max_holding_days": 21,
    "stale_min_return": 0.02,
    "time_profit_days": 10,
    "time_profit_min": 0.06,
}

# Hold-Winners 재평가 파라미터 (목표가 도달 시 defer + tight trail)
HOLD_WINNERS_TIGHT_TRAIL = 0.035       # defer 후 고점 대비 -3.5% trailing
HOLD_WINNERS_MAX_DEFERS = 2            # 같은 포지션 최대 defer 횟수
# RSI_MIN 제거: RSI가 낮을수록 상승 여지가 더 많음 → 과매수 구간(>RSI_MAX)만 차단
HOLD_WINNERS_RSI_MAX = 75.0            # RSI > 75 = 과매수 → defer 거부
HOLD_WINNERS_ADX_MIN = 20.0
HOLD_WINNERS_VOLUME_MULT_MIN = 1.2
HOLD_WINNERS_VOLUME_Z_MIN = 1.0
HOLD_WINNERS_BB_PBAND_MAX = 0.95
HOLD_WINNERS_MIN_CHECKS = 5            # 5개 조건 전부 통과해야 defer (all() 동등). 04-15 HW v1 원형
HOLD_WINNERS_DEFER_FREEZE_DAYS = 0    # freeze 비활성 (defer 후에도 시간익절/목표가 즉시 재체크)

# 상승 예측 모델 (경험적 분포)
UPSIDE_MODEL_MIN_SAMPLES = 10          # 버킷 최소 샘플 (미달 시 strategy 레벨 폴백)
UPSIDE_MODEL_CCS_BUCKETS = [0.40, 0.45, 0.50, 0.55]  # 버킷 경계
UPSIDE_MODEL_CACHE_PATH = "data/paper_trading/upside_distribution.json"

# 고RSI 매수 플래그 (리포트 표시용)
HIGH_RSI_FLAG_THRESHOLD = 70.0

# 구글 시트 탭 이름
PAPER_TRADING_WORKSHEET_LOG = "페이퍼_거래로그"
PAPER_TRADING_WORKSHEET_POSITIONS = "페이퍼_포지션현황"
PAPER_TRADING_WORKSHEET_SUMMARY = "페이퍼_성과요약"

# ═══════════════════════════════════════════════════════════════
# Machine Learning
# ═══════════════════════════════════════════════════════════════
ML_ENABLED = False                      # True로 바꾸면 Phase B+ 블렌딩 활성화
ML_BLEND_WEIGHT_ALPHA = 0.4             # Phase B: alpha factor 블렌딩 비중
ML_BLEND_WEIGHT_RISK = 0.5              # Phase C: risk quality 블렌딩 비중
ML_BLEND_WEIGHT_STRATEGY = 0.5         # Phase C: strategy score 블렌딩 비중
ML_MIN_TRAINING_SAMPLES = 200           # 최소 학습 거래 수
ML_TIME_SERIES_N_SPLITS = 5             # walk-forward CV 폴드 수
ML_MODEL_DIR = "data/ml_models"         # artifact 저장 경로
ML_MODEL_MAX_AGE_DAYS = 14              # 이보다 오래되면 stale 경고
ML_FALLBACK_ON_MISSING = True           # 모델 없으면 rule-based 폴백
ML_DRIFT_ROLLING_WINDOW = 30            # 드리프트 감지 최근 N거래
ML_DRIFT_ACCURACY_THRESHOLD = 0.45     # 이하로 떨어지면 재학습 경보
