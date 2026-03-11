#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전 종목 스캔 워크플로우 (Signals2).

NYSE/NASDAQ 전 종목을 대상으로 '바닥 탈출(Turnaround)'과 '강세 돌파(Momentum)'
전략 조건에 맞는 종목을 선별하여 Signals2 워크시트에 내보낸다.

기존 main.py/Signals 로직과 완전히 독립적으로 동작한다.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    COMPANY_NAME_MAP,
    EXTREME_MODEL_ENABLED,
    EXTREME_MODEL_TRAIN_PERIOD,
    GOOGLE_SHEETS_SIGNALS2_WORKSHEET,
    GOOGLE_SHEETS_SIGNALS2_ENABLED,
    SECTOR_ROTATION_ENABLED,
    SECTOR_ETFS,
    TURNAROUND_MIN_DROP,
    TURNAROUND_MA200_LOOKBACK,
    TURNAROUND_VOLUME_MULT,
    MOMENTUM_HIGH_THRESHOLD,
    MOMENTUM_MIN_GAIN_20D,
)
from data.ticker_fetcher import fetch_all_tickers, filter_tickers_basic
from data.fetch import (
    fetch_company_names,
    fetch_latest_news,
    fetch_latest_prices,
    fetch_ohlcv,
)
from exporter import (
    export_to_google_sheet,
    prepare_export_dataframe,
    send_email_notification,
)
from features import compute_all_features
from processing import apply_neutralization, liquidity_filter
from signals import attach_signals_and_sort
from analytics.extremes import score_extremes_for_snapshot
from sector_rotation import get_strong_sectors


# ---------------------------------------------------------------------------
# 2단계: 전략별 후보 필터링 (yfinance 1년 데이터 기반)
# ---------------------------------------------------------------------------

def _stage2_filter(tickers: list[str], batch_size: int = 200) -> list[str]:
    """2단계 필터: 바닥 탈출/강세 돌파 후보만 선별한다.

    1년치 데이터를 배치로 가져와 빠르게 필터링한다.
    """
    candidates: list[str] = []
    total = len(tickers)

    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(
            f"[2단계] 배치 {batch_num}/{total_batches} "
            f"({len(batch)}개 종목 분석 중)..."
        )

        try:
            data = yf.download(
                " ".join(batch),
                period="1y",
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if data.empty:
                continue

            is_multi = isinstance(data.columns, pd.MultiIndex)

            for ticker in batch:
                try:
                    if is_multi:
                        if ticker not in data.columns.get_level_values(1):
                            continue
                        close = data["Close"][ticker].dropna()
                        volume = data["Volume"][ticker].dropna()
                    else:
                        if len(batch) > 1:
                            continue
                        close = data["Close"].dropna()
                        volume = data["Volume"].dropna()

                    if len(close) < 120:
                        continue

                    # 52주 고점/저점
                    high_52w = close.max()
                    current = close.iloc[-1]
                    drop_from_high = (current - high_52w) / high_52w

                    # 52주 포지션
                    low_52w = close.min()
                    range_52w = high_52w - low_52w
                    pos_52w = (current - low_52w) / range_52w if range_52w > 0 else 0

                    # 20일 수익률
                    ret_20d = 0.0
                    if len(close) >= 20:
                        ret_20d = (current / close.iloc[-20]) - 1.0

                    # --- 바닥 탈출 후보 ---
                    is_turnaround = drop_from_high <= TURNAROUND_MIN_DROP

                    # --- 강세 돌파 후보 ---
                    is_momentum = (
                        pos_52w >= MOMENTUM_HIGH_THRESHOLD
                        and ret_20d >= MOMENTUM_MIN_GAIN_20D
                    )

                    if is_turnaround or is_momentum:
                        candidates.append(ticker)

                except (KeyError, IndexError, ZeroDivisionError):
                    continue

        except Exception as e:
            print(f"[2단계] 배치 {batch_num} 처리 실패: {e}")
            continue

        if i + batch_size < total:
            time.sleep(1)

    print(
        f"[2단계] 필터링 완료: {total}개 → {len(candidates)}개 후보 선별"
    )
    return candidates


# ---------------------------------------------------------------------------
# 3단계: 정밀 분석 (기존 파이프라인 재사용)
# ---------------------------------------------------------------------------

def build_full_scan_dataframe(
    tickers: list[str],
    context_label: str = "Signals2",
) -> pd.DataFrame | None:
    """선별된 종목들에 대해 기존 파이프라인과 동일한 정밀 분석을 수행한다."""

    if not tickers:
        print(f"[{context_label}] 사용할 티커가 없어 건너뜁니다.")
        return None

    df = fetch_ohlcv(tickers, period="1y")

    features = compute_all_features(df)
    if features.empty:
        print(f"[{context_label}] 조건을 만족하는 종목이 없습니다.")
        return None

    liquid = liquidity_filter(features)
    if liquid.empty:
        print(
            f"[{context_label}] 유동성 컷에 걸려 남는 종목이 없습니다."
        )
        return None

    neutral = apply_neutralization(liquid)

    # --- 섹터 로테이션 ---
    if SECTOR_ROTATION_ENABLED:
        sector_etf_tickers = list(SECTOR_ETFS.values()) + ["SPY"]
        etf_raw = fetch_ohlcv(sector_etf_tickers, period="1y")

        if not etf_raw.empty:
            etf_data = {
                ticker: etf_raw[ticker].dropna(how="all")
                for ticker in etf_raw.columns.levels[0]
                if ticker in sector_etf_tickers
            }
            spy_data = etf_data.get("SPY", pd.DataFrame())

            if not spy_data.empty:
                strong_sectors = get_strong_sectors(etf_data, spy_data)
                print(
                    f"[{context_label}] 강한 섹터: "
                    f"{strong_sectors if strong_sectors else '없음'}"
                )
                neutral["in_strong_sector"] = neutral["섹터"].apply(
                    lambda s: s in strong_sectors or s == "Unknown"
                )
                neutral["섹터강도"] = neutral["섹터"].apply(
                    lambda s: "✅ 강함"
                    if (s in strong_sectors or s == "Unknown")
                    else "❌ 약함"
                )

    ranked = attach_signals_and_sort(neutral)
    unique_tickers = ranked["티커"].unique().tolist()

    # 극점 모델
    if EXTREME_MODEL_ENABLED:
        ranked, extreme_metrics = score_extremes_for_snapshot(
            unique_tickers,
            ranked,
            period=EXTREME_MODEL_TRAIN_PERIOD,
        )
        if extreme_metrics:
            for label, info in extreme_metrics.items():
                status = info.get("status")
                if status != "ok":
                    continue
                folds = info.get("folds") or []
                if not folds:
                    continue
                mean_roc = sum(f.get("roc_auc", 0.0) for f in folds) / len(folds)
                mean_ap = sum(f.get("avg_precision", 0.0) for f in folds) / len(folds)
                print(
                    f"[{context_label}] Extreme model '{label}' "
                    f"ROC {mean_roc:.2f}, AP {mean_ap:.2f}"
                )

    if {"우선순위", "극점편차", "트렌드점수_최종"}.issubset(ranked.columns):
        ranked = ranked.sort_values(
            ["우선순위", "극점편차", "트렌드점수_최종"],
            ascending=[True, False, False],
        )

    if "티커" not in ranked.columns:
        print(f"[{context_label}] 결과에 티커 정보가 없어 건너뜁니다.")
        return None

    # 회사명, 현재가, 뉴스
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
        ranked["티커"].map(lambda t: fetched_news.get(str(t).upper(), "")),
    )

    return prepare_export_dataframe(ranked)


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

def main() -> None:
    """3단계 필터링 파이프라인을 실행한다."""

    start_time = time.perf_counter()
    success = False

    try:
        # ===== 1단계: 전 종목 리스트 수집 + 기본 필터 =====
        print("=" * 60)
        print("[1단계] 전 종목 리스트 수집 및 기본 필터링")
        print("=" * 60)
        all_tickers = fetch_all_tickers()
        filtered_tickers = filter_tickers_basic(
            all_tickers, min_price=1.0, min_avg_volume=100_000
        )

        # ===== 2단계: 전략별 후보 선별 =====
        print()
        print("=" * 60)
        print("[2단계] 바닥 탈출 / 강세 돌파 후보 선별")
        print("=" * 60)
        candidates = _stage2_filter(filtered_tickers)

        if not candidates:
            print("[결과] 조건을 만족하는 후보 종목이 없습니다.")
            success = True
            return

        # ===== 3단계: 정밀 분석 =====
        print()
        print("=" * 60)
        print(f"[3단계] {len(candidates)}개 후보 종목 정밀 분석")
        print("=" * 60)
        export_df = build_full_scan_dataframe(
            candidates, GOOGLE_SHEETS_SIGNALS2_WORKSHEET
        )

        if export_df is not None and GOOGLE_SHEETS_SIGNALS2_ENABLED:
            if export_to_google_sheet(
                export_df, GOOGLE_SHEETS_SIGNALS2_WORKSHEET
            ):
                print(
                    f"[{GOOGLE_SHEETS_SIGNALS2_WORKSHEET}] "
                    f"Google Sheets 업데이트 완료 ({len(export_df)}개 종목)"
                )
                send_email_notification(export_df)
            else:
                print(
                    f"[{GOOGLE_SHEETS_SIGNALS2_WORKSHEET}] "
                    f"Google Sheets 업데이트를 건너뛰었습니다."
                )

        success = True

    finally:
        elapsed = time.perf_counter() - start_time
        minutes = elapsed / 60
        status = "완료" if success else "실패"
        print(
            f"\n[요약] 전 종목 스캔 {status} – "
            f"총 {elapsed:.1f}초 ({minutes:.2f}분) 소요"
        )


if __name__ == "__main__":
    main()
