#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""트렌드 랭킹 워크플로우 CLI 엔트리 포인트."""

from __future__ import annotations

import numpy as np

from config import PERCENT_COLUMNS, TECH_COLUMN_LABELS, TICKERS
from data.fetch import fetch_ohlcv
from exporter import export_table, export_to_google_sheet
from features import compute_all_features
from processing import apply_neutralization, liquidity_filter
from signals import attach_signals_and_sort


def main() -> None:
    """데이터 수집부터 결과 출력, 백테스트까지 전체 파이프라인을 실행한다."""

    # 1) yfinance에서 1년치 일봉 데이터를 가져온다.
    df = fetch_ohlcv(TICKERS, period="1y")

    # 2) 각 종목별로 모멘텀/거래량/변동성 특징을 계산한다.
    features = compute_all_features(df)
    if features.empty:
        print("조건을 만족하는 종목이 없습니다.")
        return

    # 3) 시장별 거래대금 분위수를 이용해 유동성이 낮은 종목을 제거한다.
    liquid = liquidity_filter(features)
    if liquid.empty:
        print("유동성 컷에 걸려 남는 종목이 없습니다. LIQUIDITY_QUANTILE을 낮춰보세요.")
        return

    # 4) 시장·섹터 중립화를 적용하고 최종 트렌드 점수를 산출한다.
    neutral = apply_neutralization(liquid)

    # 5) 시그널을 붙이고 우선순위 순으로 정렬한다.
    ranked = attach_signals_and_sort(neutral)

    display_cols = [
        "판단",
        "추천",
        "메모",
        "티커",
        "우선순위",
        "트렌드점수_최종",
        "트렌드점수",
        "RSI",
        "macd",
        "annual_dividend",
        "dividend_yield",
        "5일수익률",
        "1일수익률",
        "52주포지션",
        "거래량Z(20)",
        "ATR%",
        "macd_signal",
        "macd_hist",
        "stoch_k",
        "stoch_d",
        "roc_10",
        "adx",
        "adx_pos",
        "adx_neg",
        "ema_gap_20_50",
        "ema_gap_50_200",
        "ema_gap_20_200",
        "bollinger_pband",
        "bollinger_width",
        "keltner_pband",
        "keltner_width",
        "obv_z20",
        "cmf_20",
        "accdist_slope_5",
        "최근20일평균거래대금",
    ]

    show = ranked[display_cols].copy()

    for col in PERCENT_COLUMNS:
        if col in show.columns:
            show[col] = show[col].map(lambda value: f"{value * 100:.1f}%")

    numeric_cols = show.select_dtypes(include="number").columns
    show[numeric_cols] = show[numeric_cols].round(1)

    show = show.rename(columns=TECH_COLUMN_LABELS)

    cash_col = TECH_COLUMN_LABELS.get("최근20일평균거래대금", "최근20일평균거래대금")
    if cash_col in show.columns:
        show[cash_col] = show[cash_col].map(lambda value: f"{value/1_000_000:,.1f}M")

    print("\n=== 트렌딩 랭킹 ===")
    print(show.to_string(index=False))

    # 6) 최신 및 백업 파일을 저장한다.
    stamped, latest, export_df = export_table(ranked)
    if stamped == latest:
        print(f"\nCSV 저장 완료 → {latest}")
    else:
        print(f"\nCSV 저장 완료 → {latest} (백업: {stamped})")

    if export_to_google_sheet(export_df):
        print("Google Sheets 업데이트 완료")


if __name__ == "__main__":
    main()
