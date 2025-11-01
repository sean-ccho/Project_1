#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""트렌드 랭킹 워크플로우 CLI 엔트리 포인트."""

from __future__ import annotations

import time

from config import COMPANY_NAME_MAP, GOOGLE_SHEETS_PORTFOLIO_WORKSHEET, HISTORY_PERIOD
from data.fetch import (
    fetch_company_names,
    fetch_latest_news,
    fetch_latest_prices,
    fetch_ohlcv,
    fetch_intraday_ohlcv,
)
from exporter import (
    export_to_google_sheet,
    fetch_tickers_from_sheet,
    prepare_export_dataframe,
)
from features import compute_all_features
from processing import apply_neutralization, liquidity_filter

def build_export_dataframe(
    tickers: list[str],
    context_label: str,
    *,
    apply_liquidity_filter: bool = True,
) -> pd.DataFrame | None:
    """지정한 티커 집합에 대해 파이프라인을 실행하고 내보낼 DF를 만든다."""

    if not tickers:
        print(f"[{context_label}] 사용할 티커가 없어 건너뜁니다.")
        return None

    df = fetch_ohlcv(tickers, period=HISTORY_PERIOD)

    hourly_df = None
    try:
        hourly_df = fetch_intraday_ohlcv(tickers)
    except Exception as exc:
        print(f"[{context_label}] 1시간봉 데이터를 불러오지 못했습니다: {exc}")

    features = compute_all_features(df, hourly_df=hourly_df)
    if features.empty:
        print(f"[{context_label}] 조건을 만족하는 종목이 없습니다.")
        return None

    liquid = liquidity_filter(features) if apply_liquidity_filter else features
    if liquid.empty:
        print(
            f"[{context_label}] 유동성 컷에 걸려 남는 종목이 없습니다. LIQUIDITY_QUANTILE을 조정해보세요."
        )
        return None
    neutral = apply_neutralization(liquid)
    ranked = neutral.copy()

    if "티커" not in ranked.columns:
        print(f"[{context_label}] 결과에 티커 정보가 없어 건너뜁니다.")
        return None

    unique_tickers = ranked["티커"].astype(str).unique().tolist()

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "우선순위" in ranked.columns:
        sort_columns.append("우선순위")
        ascending.append(True)
    if "극점편차" in ranked.columns:
        sort_columns.append("극점편차")
        ascending.append(False)
    if "트렌드점수_최종" in ranked.columns:
        sort_columns.append("트렌드점수_최종")
        ascending.append(False)

    if sort_columns:
        ranked = ranked.sort_values(sort_columns, ascending=ascending)
    else:
        ranked = ranked.sort_values("티커")

    fetched_names = fetch_company_names(unique_tickers)
    full_name_map = {**fetched_names, **COMPANY_NAME_MAP}
    fetched_prices = fetch_latest_prices(unique_tickers)
    fetched_news = fetch_latest_news(unique_tickers)

    insert_loc = ranked.columns.get_loc("티커")
    ranked.insert(
        insert_loc,
        "회사",
        ranked["티커"].map(lambda t: full_name_map.get(str(t).upper(), "")),
    )
    ranked.insert(
        insert_loc + 1,
        "현재가격",
        ranked["티커"].map(
            lambda t: fetched_prices.get(str(t).upper(), float("nan"))
        ),
    )
    ranked.insert(
        insert_loc + 2,
        "최근뉴스",
        ranked["티커"].map(
            lambda t: fetched_news.get(str(t).upper(), "")
        ),
    )

    return prepare_export_dataframe(ranked)


def main() -> None:
    """Google Sheets 보유주식 워크시트를 최신 데이터로 업데이트한다."""

    start_time = time.perf_counter()
    success = False

    try:
        portfolio_tickers = fetch_tickers_from_sheet()
        portfolio_label = GOOGLE_SHEETS_PORTFOLIO_WORKSHEET or "주식찾기"
        if portfolio_tickers:
            portfolio_export = build_export_dataframe(
                portfolio_tickers,
                portfolio_label,
                apply_liquidity_filter=False,
            )
            if portfolio_export is not None:
                if export_to_google_sheet(
                    portfolio_export, GOOGLE_SHEETS_PORTFOLIO_WORKSHEET
                ):
                    print(f"[{portfolio_label}] Google Sheets 업데이트 완료")
                else:
                    print(
                        f"[{portfolio_label}] Google Sheets 업데이트를 건너뛰었습니다."
                    )
        elif GOOGLE_SHEETS_PORTFOLIO_WORKSHEET:
            print(
                f"[{portfolio_label}] 워크시트에 티커가 없어 업데이트를 건너뜁니다."
            )

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
