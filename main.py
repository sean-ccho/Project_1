#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""트렌드 랭킹 워크플로우 CLI 엔트리 포인트."""

from __future__ import annotations

import time

import pandas as pd

from config import (
    BACKTEST_ENABLED,
    BACKTEST_RUNS,
    BACKTEST_WORKSHEET_NAME,
    COMPANY_NAME_MAP,
    EXTREME_MODEL_ENABLED,
    EXTREME_MODEL_TRAIN_PERIOD,
    TICKERS,
    GOOGLE_SHEETS_SIGNALS_WORKSHEET,
    GOOGLE_SHEETS_PORTFOLIO_WORKSHEET,
    SECTOR_ROTATION_ENABLED,
    SECTOR_ETFS,
)
from data.fetch import (
    fetch_company_names,
    fetch_latest_news,
    fetch_latest_prices,
    fetch_ohlcv,
)
from exporter import (
    export_backtest_results,
    export_to_google_sheet,
    fetch_tickers_from_sheet,
    prepare_export_dataframe,
    send_email_notification,
)
from backtest import run_backtest
from features import compute_all_features
from processing import apply_neutralization, liquidity_filter
from signals import attach_signals_and_sort
from analytics.extremes import score_extremes_for_snapshot
from sector_rotation import get_strong_sectors


def build_export_dataframe(
    tickers: list[str],
    context_label: str,
    *,
    apply_liquidity_filter: bool = True,
    capture_charts: bool = False,
) -> pd.DataFrame | None:
    """지정한 티커 집합에 대해 파이프라인을 실행하고 내보낼 DF를 만든다."""

    if not tickers:
        print(f"[{context_label}] 사용할 티커가 없어 건너뜁니다.")
        return None

    df = fetch_ohlcv(tickers, period="1y")

    features = compute_all_features(df)
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

    # --- 섹터 로테이션: 강한 섹터 판별 ---
    if SECTOR_ROTATION_ENABLED:
        # 섹터 ETF 데이터 가져오기
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
                print(f"[{context_label}] 강한 섹터: {strong_sectors if strong_sectors else '없음'}")
                
                # in_strong_sector 컴럼 추가
                neutral["in_strong_sector"] = neutral["섹터"].apply(
                    lambda s: s in strong_sectors or s == "Unknown"
                )
                # 섹터강도 컴럼 추가 (엑셀에 표시용)
                neutral["섹터강도"] = neutral["섹터"].apply(
                    lambda s: "✅ 강함" if (s in strong_sectors or s == "Unknown") else "❌ 약함"
                )
    # -----------------------------------

    ranked = attach_signals_and_sort(neutral)
    unique_tickers = ranked["티커"].unique().tolist()

    extreme_metrics: dict | None = None
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
                    print(
                        f"[{context_label}] Extreme model '{label}' skipped: {info.get('reason', status)}"
                    )
                    continue
                folds = info.get("folds") or []
                if not folds:
                    continue
                mean_roc = sum(fold.get("roc_auc", 0.0) for fold in folds) / len(folds)
                mean_ap = sum(fold.get("avg_precision", 0.0) for fold in folds) / len(folds)
                print(
                    f"[{context_label}] Extreme model '{label}' walk-forward ROC {mean_roc:.2f}, AP {mean_ap:.2f}"
                )

    if {"우선순위", "극점편차", "트렌드점수_최종"}.issubset(ranked.columns):
        ranked = ranked.sort_values(
            ["우선순위", "극점편차", "트렌드점수_최종"],
            ascending=[True, False, False],
        )

    if "티커" not in ranked.columns:
        print(f"[{context_label}] 결과에 티커 정보가 없어 건너뜁니다.")
        return None

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
    
    # --- TradingView 차트 캡처 (CHARTS_ENABLED이고 capture_charts가 True일 때만) ---
    from config import CHARTS_ENABLED, CHARTS_MIN_SCORE, CHARTS_TIMEFRAMES
    from config import DRIVE_UPLOAD_ENABLED, DRIVE_FOLDER_NAME, DRIVE_FOLDER_ID, GOOGLE_SHEETS_CREDENTIALS_PATH
    from config import GITHUB_UPLOAD_ENABLED, GITHUB_REPO_NAME, GITHUB_BRANCH_NAME
    from config import CHARTS_OUTPUT_DIR
    
    if CHARTS_ENABLED and capture_charts:
        from charts.tradingview_capture import capture_multiple_timeframes
        import os
        import shutil
        from pathlib import Path
        
        # Drive 업로드 사용 시 모듈 import
        if DRIVE_UPLOAD_ENABLED:
            from charts.gdrive_uploader import upload_to_drive
        
        # 기존 차트 파일들 자동 삭제
        charts_dir = Path(CHARTS_OUTPUT_DIR)
        if charts_dir.exists():
            print(f"[{context_label}] 기존 차트 파일 삭제 중...")
            shutil.rmtree(charts_dir)
        charts_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{context_label}] 차트 저장 디렉토리 준비 완료: {charts_dir}")
        
        print(f"[{context_label}] TradingView 차트 캡처 시작...")
        
        # 차트 컬럼 초기화
        for tf in CHARTS_TIMEFRAMES:
            col_name = f"차트_{tf}"
            if col_name not in ranked.columns:
                ranked[col_name] = ""
        
        # 매수적합도 기준 이상 종목만 캡처 (성능 최적화)
        if "매수적합도_표시" in ranked.columns:
            # ★ 개수로 필터링 (CHARTS_MIN_SCORE = 0.0이면 모든 종목)
            if CHARTS_MIN_SCORE > 0:
                star_threshold = "★" * int(CHARTS_MIN_SCORE)
                high_score_mask = ranked["매수적합도_표시"].astype(str).str.contains(star_threshold, na=False)
                high_score_tickers = ranked.loc[high_score_mask, "티커"].unique().tolist()
            else:
                # 0.0이면 모든 종목
                high_score_tickers = ranked["티커"].unique().tolist()
            
            print(f"[{context_label}] {len(high_score_tickers)}개 종목 차트 캡처 중...")
            
            for ticker in high_score_tickers:
                try:
                    # 차트 캡처 (항상 로컬에 저장됨)
                    chart_paths = capture_multiple_timeframes(ticker, headless=True)
                    
                    if chart_paths:
                        for tf, local_path in chart_paths.items():
                            col_name = f"차트_{tf}"
                            
                            # Google Sheets에는 항상 로컬 경로 저장
                            ranked.loc[ranked["티커"] == ticker, col_name] = local_path
                            
                            # Drive 업로드가 활성화되어 있으면 추가로 업로드
                            if DRIVE_UPLOAD_ENABLED:
                                from config import DRIVE_SHARE_EMAIL
                                drive_url = upload_to_drive(
                                    local_path,
                                    GOOGLE_SHEETS_CREDENTIALS_PATH,
                                    DRIVE_FOLDER_NAME,
                                    folder_id=DRIVE_FOLDER_ID,
                                    share_email=DRIVE_SHARE_EMAIL
                                )
                                
                                if drive_url:
                                    # 업로드 성공 시 IMAGE() 함수로 교체
                                    image_formula = f'=IMAGE("{drive_url}")'
                                    ranked.loc[ranked["티커"] == ticker, col_name] = image_formula
                                # 업로드 실패 시 로컬 경로 그대로 유지
                            
                            # GitHub 호스팅 사용 (Raw URL 생성)
                            elif GITHUB_UPLOAD_ENABLED and GITHUB_REPO_NAME:
                                # GitHub Raw URL 생성: https://raw.githubusercontent.com/{USER}/{REPO}/{BRANCH}/{PATH}
                                # local_path는 상대 경로여야 함 (예: charts/screenshots/...)
                                # 구글 시트 캐싱 방지를 위해 쿼리 파라미터 추가 (?v=timestamp)
                                cache_buster = int(time.time())
                                github_url = f"https://raw.githubusercontent.com/{GITHUB_REPO_NAME}/{GITHUB_BRANCH_NAME}/{local_path}?v={cache_buster}"
                                image_formula = f'=IMAGE("{github_url}")'
                                ranked.loc[ranked["티커"] == ticker, col_name] = image_formula
                        
                        print(f"[{context_label}] {ticker} 차트 {len(chart_paths)}개 처리 완료")
                except Exception as e:
                    print(f"[{context_label}] {ticker} 차트 처리 실패: {e}")
                    continue
            
            print(f"[{context_label}] 차트 캡처 완료")
            
            # --- GitHub 자동 푸시 (이미지 업데이트 반영) ---
            if GITHUB_UPLOAD_ENABLED:
                try:
                    import subprocess
                    print(f"[{context_label}] GitHub에 차트 이미지 푸시 중...")
                    
                    # 1. 변경사항 스테이징 (삭제된 파일 포함)
                    subprocess.run(["git", "add", "charts/screenshots"], check=True)
                    
                    # 2. 커밋 (메시지에 타임스탬프)
                    commit_msg = f"Update chart screenshots {int(time.time())}"
                    subprocess.run(["git", "commit", "-m", commit_msg], check=False) # 변경사항 없으면 실패할 수 있으므로 check=False
                    
                    # 3. 푸시
                    subprocess.run(["git", "push"], check=True)
                    print(f"[{context_label}] GitHub 푸시 완료")
                    
                except Exception as e:
                    print(f"[{context_label}] GitHub 푸시 실패: {e}")
            # ---------------------------------------------
            
        else:
            print(f"[{context_label}] '매수적합도_표시' 컬럼이 없어 차트 캡처를 건너뜁니다.")
    # ----------------------------------------------

    return prepare_export_dataframe(ranked)


def main() -> None:
    """데이터 수집부터 결과 출력, 백테스트까지 전체 파이프라인을 실행한다."""


    start_time = time.perf_counter()
    success = False

    try:
        portfolio_tickers = fetch_tickers_from_sheet()



        # Signals 워크시트 업데이트 (GOOGLE_SHEETS_SIGNALS_ENABLED로 제어)
        from config import GOOGLE_SHEETS_SIGNALS_ENABLED, GOOGLE_SHEETS_PORTFOLIO_ENABLED
        
        if GOOGLE_SHEETS_SIGNALS_ENABLED:
            signals_export = build_export_dataframe(
                TICKERS, 
                GOOGLE_SHEETS_SIGNALS_WORKSHEET,
                capture_charts=False
            )
            if signals_export is not None:
                if export_to_google_sheet(
                    signals_export, GOOGLE_SHEETS_SIGNALS_WORKSHEET
                ):
                    print(f"[{GOOGLE_SHEETS_SIGNALS_WORKSHEET}] Google Sheets 업데이트 완료")
                    # 이메일 알림 발송 (매수적합도 기준)
                    send_email_notification(signals_export)
                else:
                    print(
                        f"[{GOOGLE_SHEETS_SIGNALS_WORKSHEET}] Google Sheets 업데이트를 건너뛰었습니다."
                    )
        else:
            print(f"[{GOOGLE_SHEETS_SIGNALS_WORKSHEET}] 워크시트 업데이트가 비활성화되어 있습니다.")

        # --- Dynamic Portfolio Ticker Generation ---
        # Signals 결과에서 매수적합도 상위 종목 추출 + 고정 종목(NBM.V) 추가
        if GOOGLE_SHEETS_SIGNALS_ENABLED and signals_export is not None:
             from config import EMAIL_SCORE_THRESHOLD
             
             print(f"[{GOOGLE_SHEETS_SIGNALS_WORKSHEET}] Signals 결과 기반 차트분석 종목 리스트 생성 중 (기준점수: {EMAIL_SCORE_THRESHOLD})...")
             
             # 고정 종목
             fixed_tickers = ["NBM.V"]
             
             # 고득점 종목 필터링
             if "매수적합도_표시" in signals_export.columns and EMAIL_SCORE_THRESHOLD > 0:
                 star_threshold = "★" * int(EMAIL_SCORE_THRESHOLD)
                 high_score_mask = signals_export["매수적합도_표시"].astype(str).str.contains(star_threshold, na=False)
                 high_score_tickers = signals_export.loc[high_score_mask, "티커"].unique().tolist()
             else:
                 high_score_tickers = []
             
             # 리스트 합치기 (중복 제거하면서 순서 유지)
             new_portfolio_tickers = list(dict.fromkeys(fixed_tickers + high_score_tickers))
             
             print(f"[Dynamic Portfolio] 생성된 차트분석 종목 리스트({len(new_portfolio_tickers)}개): {new_portfolio_tickers}")
             portfolio_tickers = new_portfolio_tickers
        # -------------------------------------------

        # 차트분석 워크시트 업데이트 (GOOGLE_SHEETS_PORTFOLIO_ENABLED로 제어)
        portfolio_label = GOOGLE_SHEETS_PORTFOLIO_WORKSHEET or "차트분석"
        if GOOGLE_SHEETS_PORTFOLIO_ENABLED:
            if portfolio_tickers:
                portfolio_export = build_export_dataframe(
                    portfolio_tickers,
                    portfolio_label,
                    apply_liquidity_filter=False,
                    capture_charts=True
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
        else:
            print(f"[{portfolio_label}] 워크시트 업데이트가 비활성화되어 있습니다.")

        backtest_results: dict[str, dict] = {}
        if BACKTEST_ENABLED:
            for run in BACKTEST_RUNS:
                label = str(run.get("label", "Backtest"))
                params = {k: v for k, v in run.items() if k != "label"}
                try:
                    backtest_results[label] = run_backtest(**params)
                    print(f"[Backtest] '{label}' 실행 완료")
                except Exception as exc:  # pragma: no cover - best effort
                    print(f"[Backtest] '{label}' 실행 실패: {exc}")

            if backtest_results:
                if export_backtest_results(
                    backtest_results, BACKTEST_WORKSHEET_NAME
                ):
                    print(
                        f"[{BACKTEST_WORKSHEET_NAME}] 백테스트 결과 업데이트 완료"
                    )
                else:
                    print(
                        f"[{BACKTEST_WORKSHEET_NAME}] 백테스트 결과 업데이트를 건너뜁니다."
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
