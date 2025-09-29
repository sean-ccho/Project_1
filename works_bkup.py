#!/usr/bin/env python3
# pip install yfinance pandas numpy ta
from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# 1) 유니버스: 처음엔 소수 종목으로 테스트 후 확장
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","AMD","NFLX","INTC",
    # 캐나다 예시(USD/분기 조정 주의): "SHOP","NTR","BNS","BMO","SU","ENB","CNQ"
]

SIGNAL_PRIORITY = {
    "매수 후보": 0,
    "관심 관찰": 1,
    "관망 과열": 2,
    "관망 약세": 3,
}

# 판정 파라미터(전략에 맞게 수정)
BUY_SCORE_THRESHOLD = 0.15
BUY_POS_THRESHOLD = 0.7
BUY_RSI_MIN = 55
BUY_RSI_MAX = 75
WATCH_SCORE_THRESHOLD = 0.05
WATCH_POS_THRESHOLD = 0.5
OVERBOUGHT_RSI = 80


def classify_signal(row):
    rsi = row.get("rsi", np.nan)
    score = row.get("trend_score", np.nan)
    pos_52w = row.get("pos_52w", np.nan)

    if np.isnan(score) or np.isnan(pos_52w):
        return "관망 약세"

    if rsi >= OVERBOUGHT_RSI:
        return "관망 과열"
    if (
        score >= BUY_SCORE_THRESHOLD
        and pos_52w >= BUY_POS_THRESHOLD
        and BUY_RSI_MIN <= rsi <= BUY_RSI_MAX
    ):
        return "매수 후보"
    if score >= WATCH_SCORE_THRESHOLD and pos_52w >= WATCH_POS_THRESHOLD:
        return "관심 관찰"
    return "관망 약세"


def compute_trend_score(df):
    # df: 멀티인덱스(columns: ['Open','High','Low','Close','Volume'])로 종목별 붙은 형태
    scores = []
    for t in df.columns.levels[0]:
        p = df[t].dropna().copy()
        if len(p) < 120:  # 데이터 충분성
            continue

        # 수익률
        p["ret_1d"]  = p["Close"].pct_change(1)
        p["ret_5d"]  = p["Close"].pct_change(5)

        # 거래량 z-score (20일)
        p["vol_ma20"] = p["Volume"].rolling(20).mean()
        p["vol_std20"] = p["Volume"].rolling(20).std(ddof=0)
        p["vol_z20"] = (p["Volume"] - p["vol_ma20"]) / (p["vol_std20"] + 1e-9)

        # ATR% (14일)
        atr = AverageTrueRange(p["High"], p["Low"], p["Close"], window=14).average_true_range()
        p["atr_pct"] = atr / p["Close"]

        # RSI(14)
        rsi = RSIIndicator(p["Close"], window=14).rsi()
        p["rsi"] = rsi

        # 52주 돌파율
        p["roll_max_252"] = p["Close"].rolling(252, min_periods=63).max()
        p["roll_min_252"] = p["Close"].rolling(252, min_periods=63).min()
        p["range_52w"] = (p["roll_max_252"] - p["roll_min_252"]).replace(0, np.nan)
        p["pos_52w"] = (p["Close"] - p["roll_min_252"]) / p["range_52w"]

        latest = p.iloc[-1]

        # 정규화/스코어링
        s_ret5  = latest["ret_5d"]          # 그대로(짧은 모멘텀)
        s_vol   = np.tanh(latest["vol_z20"]/3)  # 극단 완화
        s_break = np.clip(latest["pos_52w"], 0, 1)  # 0~1
        s_vola  = np.clip(latest["atr_pct"], 0, 0.1) / 0.1  # 0~1
        # RSI는 50~70 구간 가점(과열 80↑는 감점도 가능)
        rsi = latest["rsi"]
        if np.isnan(rsi):
            s_rsi = 0
        elif rsi < 40:
            s_rsi = -0.2
        elif rsi <= 70:
            s_rsi = 0.2
        else:
            s_rsi = -0.1

        # 가중합(합은 임의, 프로젝트에서 튜닝)
        trend_score = (
            0.35*s_ret5 +
            0.35*s_vol +
            0.20*(0.5*s_break + 0.5*s_vola) +
            0.10*s_rsi
        )

        # 유동성 필터(최근 거래대금)
        avg_dollar_vol = (p["Close"].iloc[-20:] * p["Volume"].iloc[-20:]).mean()
        scores.append({
            "ticker": t,
            "trend_score": trend_score,
            "ret_1d": latest["ret_1d"],
            "ret_5d": latest["ret_5d"],
            "vol_z20": latest["vol_z20"],
            "pos_52w": latest["pos_52w"],
            "atr_pct": latest["atr_pct"],
            "rsi": rsi,
            "avg_$vol_20d": avg_dollar_vol
        })
    return pd.DataFrame(scores)

def rank_trending(tickers):
    data = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            # yfinance는 가격/티커 순서의 멀티인덱스를 반환하므로 티커 우선으로 재정렬
            df = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        else:
            df = data
    else:
        # 단일 종목일 때 멀티인덱스로 맞추기
        df = pd.concat({tickers[0]: data}, axis=1)
    res = compute_trend_score(df).dropna()
    # 유동성 하위 컷(예: 일평균 $3M 미만 제거)
    res = res[res["avg_$vol_20d"] >= 3_000_000]
    if res.empty:
        return res

    res = res.assign(signal=res.apply(classify_signal, axis=1))
    res = res.assign(
        signal_priority=res["signal"].map(SIGNAL_PRIORITY).fillna(99)
    ).sort_values(["signal_priority", "trend_score"], ascending=[True, False])
    res["signal_priority"] = res["signal_priority"].astype(int)
    return res


def export_results(df, folder="output"):
    path = Path(folder)
    path.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_cols = [
        "ticker","signal","signal_priority","trend_score","ret_1d","ret_5d",
        "vol_z20","pos_52w","atr_pct","rsi","avg_$vol_20d"
    ]
    ordered = df[[c for c in export_cols if c in df.columns]]
    timestamped = path / f"signals_{ts}.csv"
    ordered.to_csv(timestamped, index=False)
    latest = path / "signals_latest.csv"
    ordered.to_csv(latest, index=False)
    return timestamped, latest

if __name__ == "__main__":
    out = rank_trending(TICKERS)
    if out.empty:
        print("조건을 만족하는 종목이 없습니다.")
    else:
        cols = [
            "signal","ticker","signal_priority","trend_score","ret_5d","rsi",
            "pos_52w","vol_z20","atr_pct","avg_$vol_20d"
        ]
        display = (out[cols]
                   .rename(columns={
                       "signal": "판단",
                       "ticker": "티커",
                       "signal_priority": "우선순위",
                       "trend_score": "트렌드",
                       "ret_5d": "5일수익률",
                       "rsi": "RSI",
                       "pos_52w": "52주포지션",
                       "vol_z20": "거래량Z",
                       "atr_pct": "ATR%",
                       "avg_$vol_20d": "최근20일평균거래대금"
                   })
                   .round({
                       "트렌드": 4,
                       "5일수익률": 4,
                       "RSI": 1,
                       "52주포지션": 3,
                       "거래량Z": 2,
                       "ATR%": 4,
                   })
        )
        formatters = {
            "최근20일평균거래대금": lambda v: f"{v/1_000_000:,.1f}M$"
        }
        print(display.to_string(index=False, formatters=formatters))
        timestamped, latest = export_results(out, folder="output")
        print(f"\nCSV 저장 완료 → {latest} (백업: {timestamped})")
