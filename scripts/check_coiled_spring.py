"""돌파임박(Coiled Spring) 전략 historical 검증.

AMD/INTC 등의 과거 급등 구간을 찾아, 급등 시작 시점 기준 1~3주 전에
돌파임박 조건이 만족되었는지 확인한다.

조건:
  1. 변동성 압축: ATR%/ATR%_252d_median ≤ 0.85
  2. OBV 매집: OBV가 60일 최고치 대비 95% 이상
  3. 52주 포지션: 0.55 ~ 0.85
  4. ADX < 22 (추세 아직 약함)
  5. 거래량 안정: 20일 MA / 60일 MA ≥ 0.9
  6. 제외: EMA20 < EMA50, 5일 수익률 > 10%, RSI > 65
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """필요한 피처 계산."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    vol = out["Volume"]

    # ATR%
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    out["atr_pct"] = atr / close
    out["atr_med_252"] = out["atr_pct"].rolling(252).median()
    out["변동성압축"] = out["atr_pct"] / out["atr_med_252"]

    # OBV
    obv = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    out["obv"] = obv
    out["obv_60d_max"] = obv.rolling(60).max()
    out["obv_60d_ratio"] = obv / out["obv_60d_max"]
    # OBV 20일 기울기 — linear regression slope / |mean| 으로 정규화
    def _obv_slope(series: pd.Series) -> float:
        if series.isna().any() or len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        scale = max(abs(series.mean()), 1.0)
        return slope / scale
    out["obv_slope_20d"] = obv.rolling(20).apply(_obv_slope, raw=False)

    # 52주 포지션
    high_52w = close.rolling(252).max()
    low_52w = close.rolling(252).min()
    out["52w_pos"] = (close - low_52w) / (high_52w - low_52w + 1e-9)

    # ADX
    adx = ADXIndicator(high, low, close, window=14).adx()
    out["adx"] = adx

    # 거래량 MA 비율
    out["vol_ma20"] = vol.rolling(20).mean()
    out["vol_ma60"] = vol.rolling(60).mean()
    out["vol_ma_ratio"] = out["vol_ma20"] / (out["vol_ma60"] + 1e-9)

    # EMA
    out["ema20"] = EMAIndicator(close, window=20).ema_indicator()
    out["ema50"] = EMAIndicator(close, window=50).ema_indicator()

    # 5일 수익률
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)

    # RSI
    out["rsi"] = RSIIndicator(close, window=14).rsi()

    # 미래 수익률 (검증용)
    out["fwd_10d"] = close.shift(-10) / close - 1
    out["fwd_20d"] = close.shift(-20) / close - 1

    return out


def check_coiled_spring(row: pd.Series, variant: str = "v2") -> tuple[bool, dict]:
    """한 bar에서 돌파임박 조건 만족 여부.

    variant:
      v1 = 초기 엄격 조건 (너무 엄격해서 거의 안 잡힘)
      v2 = 완화 — 진짜 "축적 중" 포착 (OBV 상승 추세 + 가격 횡보)
    """
    if variant == "v1":
        checks = {
            "변동성압축_0.85이하": row["변동성압축"] <= 0.85,
            "OBV_60d_95%이상": row["obv_60d_ratio"] >= 0.95,
            "52w_pos_0.55~0.85": 0.55 <= row["52w_pos"] <= 0.85,
            "ADX_22미만": row["adx"] < 22,
            "vol_ratio_0.9이상": row["vol_ma_ratio"] >= 0.9,
        }
        exclusions = {
            "EMA배열_OK": row["ema20"] >= row["ema50"],
            "5d수익률_10%미만": row["ret_5d"] < 0.10,
            "RSI_65이하": row["rsi"] <= 65,
        }
    elif variant == "v2":
        checks = {
            "변동성압축_0.95이하": row["변동성압축"] <= 0.95,
            "OBV_60d_70%이상": row["obv_60d_ratio"] >= 0.70,
            "52w_pos_0.25~0.70": 0.25 <= row["52w_pos"] <= 0.70,
            "ADX_20미만": row["adx"] < 20,
            "vol_ratio_0.85이상": row["vol_ma_ratio"] >= 0.85,
            "20d수익률_횡보": -0.08 <= row.get("ret_20d", 0) <= 0.08,
        }
        exclusions = {
            "EMA배열_근접": row["ema20"] >= row["ema50"] * 0.95,
            "5d수익률_8%미만": row["ret_5d"] < 0.08,
            "RSI_중립범위": 35 <= row["rsi"] <= 65,
        }
    else:  # v3 — 강화
        checks = {
            # 변동성 압축 — 조금 더 엄격
            "변동성압축_0.90이하": row["변동성압축"] <= 0.90,
            # OBV 60일 상위 75% 이상
            "OBV_60d_75%이상": row["obv_60d_ratio"] >= 0.75,
            # OBV 20일 기울기 > 0 (매집 추세 확인 — 핵심 추가)
            "OBV_20d_상승추세": row.get("obv_slope_20d", 0) > 0,
            # 52w 포지션 0.25 ~ 0.65 (이미 오른 상태 배제)
            "52w_pos_0.25~0.65": 0.25 <= row["52w_pos"] <= 0.65,
            # ADX 약세 (횡보/수렴)
            "ADX_20미만": row["adx"] < 20,
            # 거래량 유지
            "vol_ratio_0.85이상": row["vol_ma_ratio"] >= 0.85,
            # 가격 20일간 횡보 — 더 엄격 (-5% ~ +5%)
            "20d수익률_타이트횡보": -0.05 <= row.get("ret_20d", 0) <= 0.05,
        }
        exclusions = {
            "EMA배열_근접": row["ema20"] >= row["ema50"] * 0.95,
            "5d수익률_5%미만": row["ret_5d"] < 0.05,
            "RSI_중립범위": 40 <= row["rsi"] <= 60,
        }

    # NaN 방어
    for k, v in list(checks.items()):
        if isinstance(v, (bool, np.bool_)):
            continue
        checks[k] = False
    for k, v in list(exclusions.items()):
        if isinstance(v, (bool, np.bool_)):
            continue
        exclusions[k] = False

    all_pass = all(checks.values()) and all(exclusions.values())
    return all_pass, {**checks, **exclusions}


def find_surges(df: pd.DataFrame, min_gain: float = 0.15, window: int = 20) -> pd.DataFrame:
    """급등 구간 탐지: window일간 min_gain 이상 상승한 첫 날."""
    out = df.copy()
    out["fwd_max_gain"] = (out["Close"].shift(-window).rolling(window).max() / out["Close"] - 1)
    # 더 정확히: 앞으로 window일간 최고가 대비 현재가
    fwd_highs = []
    closes = out["Close"].values
    for i in range(len(closes)):
        end = min(i + window + 1, len(closes))
        if end - i <= 1:
            fwd_highs.append(np.nan)
        else:
            fwd_highs.append(closes[i + 1:end].max() / closes[i] - 1)
    out["fwd_max_gain"] = fwd_highs
    return out


def analyze_ticker(ticker: str, start: str = "2023-01-01", end: str | None = None) -> None:
    print(f"\n{'='*70}\n  {ticker} 돌파임박 분석 ({start} ~ {end or 'latest'})\n{'='*70}")

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        print(f"  데이터 없음")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = compute_features(df)
    df = find_surges(df, min_gain=0.20, window=20)

    # 돌파임박 조건 만족 bar 찾기
    signal_dates = []
    for idx, row in df.iterrows():
        ok, checks = check_coiled_spring(row)
        if ok:
            signal_dates.append({
                "date": idx,
                "close": row["Close"],
                "fwd_10d": row["fwd_10d"],
                "fwd_20d": row["fwd_20d"],
                "fwd_max_gain": row["fwd_max_gain"],
                "변동성압축": row["변동성압축"],
                "obv_60d_ratio": row["obv_60d_ratio"],
                "52w_pos": row["52w_pos"],
                "adx": row["adx"],
                "vol_ma_ratio": row["vol_ma_ratio"],
                "rsi": row["rsi"],
            })

    if not signal_dates:
        print(f"  [결과] 해당 기간 돌파임박 신호 없음")
        return

    sig_df = pd.DataFrame(signal_dates).set_index("date")
    print(f"\n  [신호 발생 일수] {len(sig_df)}일")
    print(f"  [10일 후 평균 수익률] {sig_df['fwd_10d'].mean()*100:.2f}%")
    print(f"  [20일 후 평균 수익률] {sig_df['fwd_20d'].mean()*100:.2f}%")
    print(f"  [20일내 최고 평균] {sig_df['fwd_max_gain'].mean()*100:.2f}%")
    print(f"  [20일내 >15% 상승 비율] {(sig_df['fwd_max_gain'] > 0.15).mean()*100:.1f}%")
    print(f"  [20일내 >20% 상승 비율] {(sig_df['fwd_max_gain'] > 0.20).mean()*100:.1f}%")

    # 연속 신호 그룹핑 (앞에 있는 첫 날만 표시)
    sig_df["gap"] = sig_df.index.to_series().diff().dt.days
    sig_df["group"] = (sig_df["gap"] > 5).cumsum()
    first_of_group = sig_df.groupby("group").first()
    print(f"\n  [독립 신호 그룹 수] {len(first_of_group)}")
    print(f"\n  샘플 신호 (그룹 첫 날):")
    for d, row in first_of_group.head(15).iterrows():
        date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        fwd20 = row.get("fwd_20d", float("nan"))
        fwd_max = row.get("fwd_max_gain", float("nan"))
        fwd20_s = f"+{fwd20*100:5.1f}%" if not np.isnan(fwd20) else "   n/a"
        fwd_max_s = f"+{fwd_max*100:5.1f}%" if not np.isnan(fwd_max) else "   n/a"
        print(
            f"    {date_str} @ ${row['close']:6.2f}"
            f" | 20d후 {fwd20_s} | 20d최고 {fwd_max_s}"
            f" | 압축{row['변동성압축']:.2f} OBV{row['obv_60d_ratio']:.2f}"
            f" 52w{row['52w_pos']:.2f} ADX{row['adx']:.1f} RSI{row['rsi']:.1f}"
        )


def aggregate_analysis(tickers: list[str], start: str = "2022-01-01", variant: str = "v2") -> None:
    """다수 종목 신호를 집계하여 전략 유효성 평가."""
    all_signals = []
    for t in tickers:
        try:
            df = yf.download(t, start=start, progress=False, auto_adjust=True)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = compute_features(df)
            df = find_surges(df, min_gain=0.20, window=20)

            for idx, row in df.iterrows():
                ok, _ = check_coiled_spring(row, variant=variant)
                if ok:
                    all_signals.append({
                        "ticker": t,
                        "date": idx,
                        "close": row["Close"],
                        "fwd_10d": row["fwd_10d"],
                        "fwd_20d": row["fwd_20d"],
                        "fwd_max_gain": row["fwd_max_gain"],
                        "변동성압축": row["변동성압축"],
                        "obv_60d_ratio": row["obv_60d_ratio"],
                        "52w_pos": row["52w_pos"],
                        "adx": row["adx"],
                        "vol_ma_ratio": row["vol_ma_ratio"],
                        "rsi": row["rsi"],
                    })
        except Exception as e:
            print(f"  [{t}] 오류: {e}")

    if not all_signals:
        print("\n  전체 신호 없음")
        return

    df_sig = pd.DataFrame(all_signals)
    df_sig = df_sig.dropna(subset=["fwd_20d"])

    # 연속 신호는 같은 종목 내 5일 이내면 같은 그룹으로 묶음
    df_sig = df_sig.sort_values(["ticker", "date"]).reset_index(drop=True)
    df_sig["gap"] = df_sig.groupby("ticker")["date"].diff().dt.days
    df_sig["new_group"] = (df_sig["gap"].isna()) | (df_sig["gap"] > 5)
    df_sig["group_id"] = df_sig.groupby("ticker")["new_group"].cumsum()
    first_signals = df_sig.groupby(["ticker", "group_id"]).first().reset_index()

    print(f"\n{'='*70}")
    print(f"  전체 집계 ({len(tickers)}개 종목, {start} ~ 현재)")
    print(f"{'='*70}")
    print(f"  분석 대상 종목: {len(tickers)}개")
    print(f"  신호 발생 종목: {df_sig['ticker'].nunique()}개")
    print(f"  독립 신호 그룹: {len(first_signals)}건")
    print(f"\n  --- 독립 신호 기준 성과 ---")
    print(f"  10일 후 평균 수익률: {first_signals['fwd_10d'].mean()*100:+.2f}%")
    print(f"  20일 후 평균 수익률: {first_signals['fwd_20d'].mean()*100:+.2f}%")
    print(f"  20일 내 최고 평균:   {first_signals['fwd_max_gain'].mean()*100:+.2f}%")
    print(f"  승률 (20일 후 >0%): {(first_signals['fwd_20d'] > 0).mean()*100:.1f}%")
    print(f"  20일 내 >10% 상승:  {(first_signals['fwd_max_gain'] > 0.10).mean()*100:.1f}%")
    print(f"  20일 내 >15% 상승:  {(first_signals['fwd_max_gain'] > 0.15).mean()*100:.1f}%")
    print(f"  20일 내 >20% 상승:  {(first_signals['fwd_max_gain'] > 0.20).mean()*100:.1f}%")
    print(f"  20일 내 >10% 하락:  {(first_signals['fwd_20d'] < -0.10).mean()*100:.1f}%")

    print(f"\n  --- 수익률 분포 ---")
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:<2}  20일수익률: {np.nanpercentile(first_signals['fwd_20d'], p)*100:+6.2f}%  |  20일최고: {np.nanpercentile(first_signals['fwd_max_gain'], p)*100:+6.2f}%")

    # 종목별 성과 (신호 2+ 건)
    by_ticker = first_signals.groupby("ticker").agg(
        n=("fwd_20d", "size"),
        avg_20d=("fwd_20d", "mean"),
        max_20d=("fwd_max_gain", "mean"),
    ).sort_values("max_20d", ascending=False)

    print(f"\n  --- 종목별 성과 (Top 10 & Bottom 5) ---")
    print(f"  {'Ticker':<8} {'건수':>4} {'20d평균':>10} {'20d최고평균':>12}")
    for t, r in by_ticker.head(10).iterrows():
        print(f"  {t:<8} {int(r['n']):>4} {r['avg_20d']*100:>+9.2f}% {r['max_20d']*100:>+11.2f}%")
    print(f"  {'-'*40}")
    for t, r in by_ticker.tail(5).iterrows():
        print(f"  {t:<8} {int(r['n']):>4} {r['avg_20d']*100:>+9.2f}% {r['max_20d']*100:>+11.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--aggregate", "--aggregate-v3"):
        variant = "v3" if sys.argv[1] == "--aggregate-v3" else "v2"
        # 대표 종목: v3에서는 Energy/Healthcare 일부 제외
        if variant == "v3":
            sample = [
                # Mega tech
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                # Semi (강점 섹터)
                "AMD", "INTC", "AVGO", "QCOM", "MU", "AMAT", "KLAC",
                # Software (강점 섹터)
                "CRM", "ORCL", "ADBE", "PANW", "CRWD", "PLTR", "SNPS",
                # Finance (중립)
                "JPM", "BAC", "GS", "MS", "C", "WFC",
                # Consumer (중립)
                "HD", "NKE", "SBUX", "MCD", "WMT", "COST",
                # Healthcare — LLY만 (growth pharma)
                "LLY",
            ]
        else:
            sample = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                "AMD", "INTC", "AVGO", "QCOM", "MU", "AMAT", "KLAC",
                "CRM", "ORCL", "ADBE", "PANW", "CRWD", "PLTR", "SNPS",
                "JPM", "BAC", "GS", "MS", "C", "WFC",
                "HD", "NKE", "SBUX", "MCD", "WMT", "COST",
                "JNJ", "UNH", "PFE", "LLY", "MRNA",
                "XOM", "CVX", "SLB",
            ]
        aggregate_analysis(sample, start="2022-01-01", variant=variant)
    else:
        tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AMD", "INTC", "NVDA", "PLTR", "SMCI"]
        for t in tickers:
            try:
                analyze_ticker(t, start="2023-01-01")
            except Exception as e:
                print(f"  [{t}] 오류: {e}")
