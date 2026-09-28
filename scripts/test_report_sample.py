"""샘플 PDF 리포트 생성 테스트 스크립트.

이메일에서 본 실제 데이터 (BULL 매수, ACHR/JOBY 보유)를 기반으로 PDF 생성.
실행: python scripts/test_report_sample.py
결과: output/sample_trading_report.pdf
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from pathlib import Path

# ── 샘플 데이터 구성 ─────────────────────────────────────────────────────────

# 이메일에서 본 매수 데이터 (BULL, 2026-04-03)
sample_buys = [
    {
        "ticker": "BULL",
        "entry_date": "2026-04-03",
        "entry_price": 4.82,
        "sector": "Technology",
        "strategy": "바닥반등",
        "star_rating": "★★★★★ (5.5)",
        "ccs_score": 0.4986,
        "reason": "신규매수",
        "ccs_breakdown": {
            "strategy_fit": 0.72,
            "timing": 0.65,
            "alpha": 0.58,
            "risk": 0.81,
            "confluence": 0.69,
            "penalty": 0.0,
        },
    }
]

# 이메일에서 본 매도 (없었지만 테스트용으로 추가)
sample_sells = [
    {
        "ticker": "ARRY",
        "entry_price": 6.20,
        "exit_price": 7.13,
        "sector": "Technology",
        "strategy": "바닥반등",
        "return_pct": 0.15,
        "holding_days": 12,
        "exit_reason": "목표수익+15%",
    }
]

# 현재 보유 (이메일에서: ACHR, JOBY, BULL)
sample_positions = [
    {
        "ticker": "ACHR",
        "entry_date": "2026-04-02",
        "entry_price": 5.35,
        "sector": "Industrials",
        "strategy": "신규매수",
        "ccs_score": 0.4512,
        "star_rating": "★★★★ (4.5)",
    },
    {
        "ticker": "JOBY",
        "entry_date": "2026-04-02",
        "entry_price": 8.50,
        "sector": "Industrials",
        "strategy": "바닥반등",
        "ccs_score": 0.4823,
        "star_rating": "★★★★★ (5.0)",
    },
    {
        "ticker": "BULL",
        "entry_date": "2026-04-03",
        "entry_price": 4.82,
        "sector": "Technology",
        "strategy": "바닥반등",
        "ccs_score": 0.4986,
        "star_rating": "★★★★★ (5.5)",
    },
]

# 현재가 (샘플)
sample_prices = {
    "ACHR": 5.01,
    "JOBY": 7.95,
    "BULL": 4.71,
}

# 최근 거래 이력 (샘플)
sample_trades = [
    {"ticker": "ARRY", "entry_date": "2026-03-22", "exit_date": "2026-04-03",
     "entry_price": 6.20, "exit_price": 7.13, "return_pct": 15.0,
     "holding_days": 12, "exit_reason": "목표수익+15%"},
    {"ticker": "IONQ", "entry_date": "2026-03-10", "exit_date": "2026-03-28",
     "entry_price": 22.40, "exit_price": 24.85, "return_pct": 10.9,
     "holding_days": 18, "exit_reason": "시간익절"},
    {"ticker": "SOUN", "entry_date": "2026-02-28", "exit_date": "2026-03-08",
     "entry_price": 9.80, "exit_price": 9.31, "return_pct": -5.0,
     "holding_days": 8, "exit_reason": "손절-5%"},
    {"ticker": "NVAX", "entry_date": "2026-02-15", "exit_date": "2026-02-25",
     "entry_price": 12.50, "exit_price": 13.88, "return_pct": 11.0,
     "holding_days": 10, "exit_reason": "시간익절"},
]

# merged_df 시뮬레이션 (BULL 종목 기술적 데이터 샘플)
sample_df = pd.DataFrame([
    {
        "티커": "BULL",
        "회사": "Pacer US Bull 3X ETF",
        "현재가격": 4.82,
        "섹터": "Technology",
        "전략구분": "바닥반등",
        "바닥반등_적합도": 5.5,
        "모멘텀_적합도": 3.2,
        "일봉패턴": "이중바닥, 강세잉걸핑",
        "주봉패턴": "골든크로스임박, 하락쐐기",
        "월봉패턴": "하락쐐기",
        "RSI": 38.4,
        "macd_hist": 0.0182,
        "adx": 24.7,
        "bollinger_pband": 0.31,
        "stoch_k": 42.1,
        "atr_pct": 4.82,
        "ema_gap_20_50": -0.042,
        "ema_gap_50_200": -0.128,
        "거래량Z(20)": 1.84,
        "cmf_20": 0.112,
        "ret_5d": -0.032,
        "ret_20d": -0.118,
        "pos_52w": 0.18,
        "factor_momentum": -0.38,
        "factor_trend": -0.22,
        "factor_volume": 0.45,
        "factor_volatility": 0.31,
        "factor_mean_reversion": 0.62,
        "fund_institutional_holders_pct": 28.4,
        "fund_short_float_pct": 5.2,
        "fund_roe": 0.0,
        "fund_net_margin": 0.0,
        "fund_revenue_growth": 0.0,
        "fund_earnings_growth": 0.0,
        "매수적합도_표시": "★★★★★ (5.5)",
    },
    {
        "티커": "ARRY",
        "회사": "Array Technologies Inc",
        "현재가격": 7.13,
        "섹터": "Technology",
        "전략구분": "바닥반등",
        "RSI": 68.2,
        "macd_hist": 0.0845,
        "adx": 32.1,
        "bollinger_pband": 0.87,
        "stoch_k": 72.4,
        "atr_pct": 3.21,
        "ema_gap_20_50": 0.078,
        "ema_gap_50_200": -0.045,
        "거래량Z(20)": 2.34,
        "cmf_20": 0.224,
        "ret_5d": 0.082,
        "ret_20d": 0.152,
        "pos_52w": 0.42,
        "factor_momentum": 0.72,
        "factor_trend": 0.54,
        "factor_volume": 0.68,
        "factor_volatility": 0.28,
        "factor_mean_reversion": -0.21,
        "fund_institutional_holders_pct": 45.8,
        "fund_short_float_pct": 12.4,
        "fund_roe": 8.5,
        "fund_net_margin": 4.2,
        "fund_revenue_growth": 0.18,
        "fund_earnings_growth": 0.24,
        "일봉패턴": "역헤드앤숄더",
        "주봉패턴": "정배열(형성중)",
        "월봉패턴": "",
        "매수적합도_표시": "★★★★★ (6.2)",
        "바닥반등_적합도": 6.2,
        "모멘텀_적합도": 4.8,
    },
])

# ── pt_result 구성 ─────────────────────────────────────────────────────────

pt_result = {
    "date": "2026-04-03",
    "buys": sample_buys,
    "sells": sample_sells,
    "holdings": 3,
    "selection_debug": {
        "regime": "neutral",
        "rejections": {"score_too_low": 45, "overbought": 12, "liquidity": 8},
        "top5": [
            {
                "ticker": "BULL",
                "ccs": 0.4986,
                "strategy_fit": 0.72,
                "timing": 0.65,
                "alpha": 0.58,
                "risk": 0.81,
                "confluence": 0.69,
                "penalty": 0.0,
                "strategy": "바닥반등",
                "star": "★★★★★ (5.5)",
                "sector": "Technology",
            }
        ],
    },
}

# ── PDF 생성 ──────────────────────────────────────────────────────────────────

from paper_trading.report_generator import generate_trading_report

print("PDF 리포트 생성 중...")
pdf_bytes = generate_trading_report(
    result=pt_result,
    positions=sample_positions,
    merged_df=sample_df,
    prices=sample_prices,
    trades=sample_trades,
)

if pdf_bytes:
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sample_trading_report.pdf"
    out_path.write_bytes(pdf_bytes)
    print(f"✅ PDF 생성 완료: {out_path} ({len(pdf_bytes):,} bytes)")
else:
    print("❌ PDF 생성 실패")
