#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전 종목 스캔 워크플로우 ([NASDAQ/NYSE 분석]).

NYSE/NASDAQ 전 종목을 대상으로 '바닥 탈출(Turnaround)'과 '강세 돌파(Momentum)'
전략 조건에 맞는 종목을 선별하여 [NASDAQ/NYSE 분석] 워크시트에 내보낸다.

기존 main.py/[SP500 분석] 로직과 완전히 독립적으로 동작한다.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from screener.config import (
    COMPANY_NAME_MAP,
    EXTREME_MODEL_ENABLED,
    EXTREME_MODEL_TRAIN_PERIOD,
    GOOGLE_SHEETS_SIGNALS2_WORKSHEET,
    GOOGLE_SHEETS_SIGNALS2_ENABLED,
    GOOGLE_SHEETS_SIGNALS2_CHART_WORKSHEET,
    GOOGLE_SHEETS_SIGNALS2_CHART_ENABLED,
    SECTOR_ROTATION_ENABLED,
    SECTOR_ETFS,
    TURNAROUND_MIN_DROP,
    TURNAROUND_MA200_LOOKBACK,
    TURNAROUND_VOLUME_MULT,
    MOMENTUM_HIGH_THRESHOLD,
    MOMENTUM_MIN_GAIN_20D,
    EMAIL_BOTTOM_SCORE_THRESHOLD,
    EMAIL_MOMENTUM_SCORE_THRESHOLD,
    PAPER_TRADING_ENABLED,
)
from data.ticker_fetcher import fetch_all_tickers, filter_tickers_basic
from data.fetch import (
    fetch_company_names,
    fetch_latest_news,
    fetch_latest_prices,
    fetch_ohlcv,
)
from screener.exporter import (
    export_to_google_sheet,
    prepare_export_dataframe,
    send_email_notification,
)
from screener.features import compute_all_features
from screener.processing import apply_neutralization, liquidity_filter
from screener.signals import attach_signals_and_sort
from analytics.extremes import score_extremes_for_snapshot
from screener.sector_rotation import get_strong_sectors


# ---------------------------------------------------------------------------
# 2단계: 전략별 후보 필터링 (yfinance 1년 데이터 기반)
# ---------------------------------------------------------------------------

_STAGE2_BATCH_SIZE = 150       # 배치 크기
_STAGE2_BATCH_DELAY = 1.5      # 배치 간 대기 시간(초)
_STAGE2_MAX_RETRIES = 3        # rate limit 재시도 횟수
_STAGE2_RETRY_BASE_DELAY = 30  # 재시도 기본 대기(초) — 30/60/120


def _stage2_download_with_retry(
    batch: list[str],
    batch_num: int,
    total_batches: int,
) -> pd.DataFrame | None:
    """2단계용 yfinance 배치 다운로드 (rate limit 재시도 포함)."""
    for attempt in range(1, _STAGE2_MAX_RETRIES + 1):
        try:
            data = yf.download(
                " ".join(batch),
                period="1y",
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            return data if not data.empty else None

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "Rate" in err_str or "Too Many" in err_str

            if is_rate_limit and attempt < _STAGE2_MAX_RETRIES:
                wait = _STAGE2_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[2단계] 배치 {batch_num}/{total_batches} "
                    f"rate limit → {wait}초 대기 후 재시도 "
                    f"({attempt}/{_STAGE2_MAX_RETRIES})..."
                )
                time.sleep(wait)
            else:
                print(f"[2단계] 배치 {batch_num}/{total_batches} 처리 실패: {e}")
                return None

    return None


def _stage2_filter(tickers: list[str]) -> list[str]:
    """2단계 필터: 바닥 탈출/강세 돌파 후보만 선별한다.

    1년치 데이터를 배치로 가져와 빠르게 필터링한다.
    """
    candidates: list[str] = []
    total = len(tickers)
    skipped_batches = 0
    skipped_tickers = 0

    for i in range(0, total, _STAGE2_BATCH_SIZE):
        batch = tickers[i : i + _STAGE2_BATCH_SIZE]
        batch_num = i // _STAGE2_BATCH_SIZE + 1
        total_batches = (total + _STAGE2_BATCH_SIZE - 1) // _STAGE2_BATCH_SIZE
        print(
            f"[2단계] 배치 {batch_num}/{total_batches} "
            f"({len(batch)}개 종목 분석 중)..."
        )

        data = _stage2_download_with_retry(batch, batch_num, total_batches)
        if data is None:
            skipped_batches += 1
            skipped_tickers += len(batch)
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

            except (KeyError, IndexError, ZeroDivisionError, TypeError):
                continue

        if i + _STAGE2_BATCH_SIZE < total:
            time.sleep(_STAGE2_BATCH_DELAY)

    skip_msg = (
        f" | 배치 실패: {skipped_batches}개 ({skipped_tickers}개 종목 미처리)"
        if skipped_batches > 0
        else ""
    )
    print(
        f"[2단계] 필터링 완료: {total}개 → {len(candidates)}개 후보 선별{skip_msg}"
    )
    return candidates


# ---------------------------------------------------------------------------
# 3단계: 정밀 분석 (기존 파이프라인 재사용)
# ---------------------------------------------------------------------------

def build_full_scan_dataframe(
    tickers: list[str],
    context_label: str = "[NASDAQ/NYSE 분석]",
) -> pd.DataFrame | None:
    """선별된 종목들에 대해 기존 파이프라인과 동일한 정밀 분석을 수행한다."""

    if not tickers:
        print(f"[{context_label}] 사용할 티커가 없어 건너뜁니다.")
        return None

    df = fetch_ohlcv(tickers, period="5y")

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

    if {"모멘텀_적합도", "바닥반등_적합도"}.issubset(ranked.columns):
        ranked = ranked.sort_values(
            ["모멘텀_적합도", "바닥반등_적합도"],
            ascending=[False, False],
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

    # ranked(전체 컬럼)를 paper trading에서 사용하기 위해 캐시
    build_full_scan_dataframe._last_ranked = ranked

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
                send_email_notification(export_df, subject_prefix="[NASDAQ/NYSE 분석]")
            else:
                print(
                    f"[{GOOGLE_SHEETS_SIGNALS2_WORKSHEET}] "
                    f"Google Sheets 업데이트를 건너뛰었습니다."
                )

        # ===== [NASDAQ/NYSE 차트분석] 시트 업데이트 =====
        if export_df is not None and GOOGLE_SHEETS_SIGNALS2_CHART_ENABLED:
            chart_label = GOOGLE_SHEETS_SIGNALS2_CHART_WORKSHEET
            print()
            print("=" * 60)
            print(f"[{chart_label}] 이메일 대상 종목 차트 분석 시작")
            print("=" * 60)

            # 이메일 기준과 동일하게 별점 4.0점 이상 종목 추출
            def _extract_score_fs2(val):
                try:
                    return float(str(val).split("(")[1].split(")")[0])
                except Exception:
                    return 0.0

            chart_tickers = []
            if "바닥반등_적합도_표시" in export_df.columns:
                scores = export_df["바닥반등_적합도_표시"].apply(_extract_score_fs2)
                chart_tickers.extend(export_df.loc[scores >= EMAIL_BOTTOM_SCORE_THRESHOLD, "티커"].unique().tolist())
            if "모멘텀_적합도_표시" in export_df.columns:
                scores = export_df["모멘텀_적합도_표시"].apply(_extract_score_fs2)
                chart_tickers.extend(export_df.loc[scores >= EMAIL_MOMENTUM_SCORE_THRESHOLD, "티커"].unique().tolist())
            chart_tickers = list(set(chart_tickers))

            if not chart_tickers:
                print(f"[{chart_label}] 차트분석 대상 종목이 없어 건너뜁니다.")
            else:
                print(f"[{chart_label}] 차트분석 대상 {len(chart_tickers)}개 종목: {chart_tickers}")
                chart_export = export_df[export_df["티커"].isin(chart_tickers)].copy()

                from screener.config import (
                    CHARTS_ENABLED, CHARTS_TIMEFRAMES, CHARTS_SIGNALS2_OUTPUT_DIR,
                    DRIVE_UPLOAD_ENABLED, DRIVE_FOLDER_NAME, DRIVE_FOLDER_ID,
                    GOOGLE_SHEETS_CREDENTIALS_PATH,
                    GITHUB_UPLOAD_ENABLED, GITHUB_REPO_NAME, GITHUB_BRANCH_NAME,
                )

                if CHARTS_ENABLED:
                    from charts.tradingview_capture import capture_multiple_timeframes
                    import shutil
                    from pathlib import Path

                    if DRIVE_UPLOAD_ENABLED:
                        from charts.gdrive_uploader import upload_to_drive

                    charts_dir = Path(CHARTS_SIGNALS2_OUTPUT_DIR)
                    if charts_dir.exists():
                        print(f"[{chart_label}] 기존 NASDAQ 차트 파일 삭제 중...")
                        shutil.rmtree(charts_dir)
                    charts_dir.mkdir(parents=True, exist_ok=True)

                    for tf in CHARTS_TIMEFRAMES:
                        col_name = f"차트_{tf}"
                        if col_name not in chart_export.columns:
                            chart_export[col_name] = ""

                    print(f"[{chart_label}] TradingView 차트 캡처 시작...")
                    for _, row in chart_export.iterrows():
                        ticker = row["티커"]
                        try:
                            chart_paths = capture_multiple_timeframes(
                                ticker,
                                output_dir=CHARTS_SIGNALS2_OUTPUT_DIR,
                                headless=True,
                            )
                            if chart_paths:
                                for tf, local_path in chart_paths.items():
                                    col_name = f"차트_{tf}"
                                    chart_export.loc[chart_export["티커"] == ticker, col_name] = local_path

                                    if DRIVE_UPLOAD_ENABLED:
                                        from screener.config import DRIVE_SHARE_EMAIL
                                        drive_url = upload_to_drive(
                                            local_path, GOOGLE_SHEETS_CREDENTIALS_PATH,
                                            DRIVE_FOLDER_NAME, folder_id=DRIVE_FOLDER_ID,
                                            share_email=DRIVE_SHARE_EMAIL,
                                        )
                                        if drive_url:
                                            chart_export.loc[chart_export["티커"] == ticker, col_name] = f'=IMAGE("{drive_url}")'
                                    elif GITHUB_UPLOAD_ENABLED and GITHUB_REPO_NAME:
                                        cache_buster = int(time.time())
                                        github_url = f"https://raw.githubusercontent.com/{GITHUB_REPO_NAME}/{GITHUB_BRANCH_NAME}/{local_path}?v={cache_buster}"
                                        chart_export.loc[chart_export["티커"] == ticker, col_name] = f'=IMAGE("{github_url}")'

                                print(f"[{chart_label}] {ticker} 차트 {len(chart_paths)}개 처리 완료")
                        except Exception as e:
                            print(f"[{chart_label}] {ticker} 차트 처리 실패: {e}")
                            continue

                    print(f"[{chart_label}] 차트 캡처 완료")

                    if GITHUB_UPLOAD_ENABLED:
                        try:
                            import subprocess
                            print(f"[{chart_label}] GitHub에 차트 이미지 푸시 중...")
                            subprocess.run(["git", "add", "charts/screenshots_nasdaq"], check=True)
                            commit_msg = f"Update NASDAQ/NYSE chart screenshots {int(time.time())}"
                            subprocess.run(["git", "commit", "-m", commit_msg], check=False)
                            subprocess.run(["git", "push"], check=True)
                            print(f"[{chart_label}] GitHub 푸시 완료")
                        except Exception as e:
                            print(f"[{chart_label}] GitHub 푸시 실패: {e}")

                # 내부자 거래 요약 컬럼 머지 (표시용, openinsider)
                from screener.insider import attach_insider_summary
                chart_export = attach_insider_summary(chart_export)

                # Google Sheets 업로드
                from screener.config import EXPORT_COLUMNS
                final_cols = [c for c in EXPORT_COLUMNS if c in chart_export.columns]
                remaining_cols = [c for c in chart_export.columns if c not in final_cols]
                chart_export = chart_export[final_cols + remaining_cols]

                if export_to_google_sheet(chart_export, GOOGLE_SHEETS_SIGNALS2_CHART_WORKSHEET):
                    print(f"[{chart_label}] Google Sheets 업데이트 완료 ({len(chart_export)}개 종목)")
                else:
                    print(f"[{chart_label}] Google Sheets 업데이트를 건너뛰었습니다.")
        # ================================================

        # ── Paper Trading: NASDAQ/NYSE ranked_df 스냅샷 저장 ──
        # 실제 PT 실행은 run_paper_trading.py에서 SP500 풀과 합산 후 통합 처리
        if PAPER_TRADING_ENABLED:
            try:
                ranked_df = getattr(build_full_scan_dataframe, "_last_ranked", None)
                if ranked_df is not None and not ranked_df.empty:
                    from paper_trading.runner import save_ranked_snapshot
                    save_ranked_snapshot(ranked_df, source="nasdaq")
                else:
                    print("[Paper Trading] NASDAQ ranked 데이터가 없어 스냅샷 저장을 건너뜁니다.")
            except Exception as exc:
                print(f"[Paper Trading] NASDAQ 스냅샷 저장 실패: {exc}")

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
