#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""트렌드 랭킹 워크플로우 CLI 엔트리 포인트."""

from __future__ import annotations

import time

from config import (
    COMPANY_NAME_MAP,
    DISPLAY_COLUMNS,
    PERCENT_COLUMNS,
    TECH_COLUMN_LABELS,
    TICKERS,
)
from data.fetch import fetch_company_names, fetch_latest_prices, fetch_ohlcv
from exporter import prepare_export_dataframe, export_to_google_sheet
from features import compute_all_features
from processing import apply_neutralization, liquidity_filter
from signals import attach_signals_and_sort


def main() -> None:
    """데이터 수집부터 결과 출력, 백테스트까지 전체 파이프라인을 실행한다."""

    start_time = time.perf_counter()
    success = False

    try:
        # 1) yfinance에서 1년치 일봉 데이터를 가져온다.
        df = fetch_ohlcv(TICKERS, period="1y")

        # 2) 각 종목별로 모멘텀/거래량/변동성 특징을 계산한다.
        features = compute_all_features(df)
        if features.empty:
            print("조건을 만족하는 종목이 없습니다.")
            success = True
            return

        # 3) 시장별 거래대금 분위수를 이용해 유동성이 낮은 종목을 제거한다.
        liquid = liquidity_filter(features)
        if liquid.empty:
            print("유동성 컷에 걸려 남는 종목이 없습니다. LIQUIDITY_QUANTILE을 낮춰보세요.")
            success = True
            return

        # 4) 시장·섹터 중립화를 적용하고 최종 트렌드 점수를 산출한다.
        neutral = apply_neutralization(liquid)

        # 5) 시그널을 붙이고 우선순위 순으로 정렬한다.
        ranked = attach_signals_and_sort(neutral)

        unique_tickers = ranked["티커"].unique().tolist()
        fetched_names = fetch_company_names(unique_tickers)
        full_name_map = {**fetched_names, **COMPANY_NAME_MAP}
        fetched_prices = fetch_latest_prices(unique_tickers)
        if "티커" in ranked.columns:
            insert_loc = ranked.columns.get_loc("티커")
            ranked.insert(
                insert_loc,
                "회사",
                ranked["티커"].map(lambda t: full_name_map.get(str(t).upper(), "")),
            )
            ranked.insert(
                insert_loc + 1,
                "현재가격",
                ranked["티커"].map(lambda t: fetched_prices.get(str(t).upper(), float("nan"))),
            )

        show = ranked[DISPLAY_COLUMNS].copy()

        for col in PERCENT_COLUMNS:
            if col in show.columns:
                show[col] = show[col].map(lambda value: f"{value * 100:.1f}%")

        numeric_cols = show.select_dtypes(include="number").columns
        show[numeric_cols] = show[numeric_cols].round(1)

        show = show.rename(columns=TECH_COLUMN_LABELS)

        cash_col = TECH_COLUMN_LABELS.get("최근20일평균거래대금", "최근20일평균거래대금")
        if cash_col in show.columns:
            show[cash_col] = show[cash_col].map(lambda value: f"{value/1_000_000:,.1f}M")

        export_df = prepare_export_dataframe(ranked)

        if export_to_google_sheet(export_df):
            print("Google Sheets 업데이트 완료")
        else:
            print("Google Sheets 업데이트를 건너뛰었습니다.")

        success = True
    finally:
        elapsed = time.perf_counter() - start_time
        minutes = elapsed / 60
        status = "완료" if success else "실패"
        print(
            f"\n[요약] 파이프라인 {status} – 총 {elapsed:.1f}초 ({minutes:.2f}분) 소요"
        )


if __name__ == "__main__":
    main()
