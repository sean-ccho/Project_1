"""백테스트용 날짜별 순위표 — paper_trading.backtest(PT-1 엔진)가 쓴다.

_prepare_price_map: OHLCV 를 티커별로 나눈다. _compute_ranked_snapshot: 그 날짜까지의 데이터로 피처 → 유동성 →
중립화 → 섹터 강도 → 시그널·정렬 (실거래 스크리너와 같은 순서). 피처는 디스크 캐시(screener.cache)에 저장한다.
2026-10-31: 옛 스크리너 자체 백테스트(run_backtest · 진입 점수 · 청산 · 지표, config BACKTEST_ENABLED=False 로 꺼져 있던 것)는 지웠다.
"""

from __future__ import annotations

from typing import Callable, Mapping

import pandas as pd

from screener.config import SECTOR_ETFS, SECTOR_ROTATION_ENABLED
from screener.chart_ma import ChartPrices, split_only_close
from screener.features import compute_features_snapshot
from screener.processing import apply_neutralization, liquidity_filter
from screener.signals import attach_signals_and_sort
from screener.sector_rotation import fill_unknown_sectors, get_strong_sectors, mark_strong_sectors
from screener.cache import load_feature_snapshot, load_ic_weights, save_feature_snapshot, save_ic_weights


# 실거래 스크리너(main.py)는 최근 5년치(fetch_ohlcv period="5y")로 피처를 계산한다. 백테스트도 날짜마다
# 그 시점 기준 최근 5년만 쓴다 — 10년 이상 백테스트에서 이력이 길어지며 피처가 실거래와 달라지는 것을 막는다
FEATURE_HISTORY_YEARS = 5


def _prepare_price_map(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """티커별로 OHLCV 데이터를 분리.

    CloseSplit(분할만 반영한 종가)도 붙인다 — 기준일마다 잘라 쓰면 받은 날까지의 배당이 섞여 되돌린 값이 어긋나서
    (0.1~0.2%), 전체 기간으로 한 번 되돌린다. 교차 라벨(screener.chart_ma)이 쓴다 (2026-10-26).
    """
    out = {}
    for ticker in raw.columns.levels[0]:
        frame = raw[ticker].dropna(how="all")
        if not frame.empty and "Close" in frame.columns:
            frame = frame.assign(CloseSplit=split_only_close(frame))
        out[ticker] = frame
    return out


def _compute_ranked_snapshot(
    price_map: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    *,
    include_fundamentals: bool,
    cache: dict[pd.Timestamp, pd.DataFrame],
    ic_weights_cache: dict | None = None,
    ohlcv_hash: str | None = None,
    fundamentals_df: "pd.DataFrame | None" = None,
    chart_loader: "Callable[[], Mapping[str, ChartPrices]] | None" = None,
) -> pd.DataFrame:
    """주어진 시점까지의 데이터로 특징 계산 및 시그널 부착.

    ic_weights_cache: {"key": "YYYY-MM", "weights": dict} — 월 단위 IC 가중치 (전월 말 기준)
    chart_loader: 차트 가격(분할만 반영 · 상장 이후 전체 주·월봉)을 돌려주는 함수 — 피처를 새로 계산할 때만 부른다
        (캐시에 다 있으면 안 받는다). 패턴 · 교차 라벨이 그 날짜(cutoff)까지의 봉만 본다 (screener.chart_ma.chart_view).

    ohlcv_hash: 디스크 캐시 키. 제공되면 feature 스냅샷을 디스크에 캐싱한다.
    fundamentals_df: 시뮬레이션 시작 전에 미리 가져온 펀더멘탈 DataFrame.
    """
    if cutoff in cache:
        return cache[cutoff]

    snapshot: dict[str, pd.DataFrame] = {}
    sector_etfs = set(SECTOR_ETFS.values())
    window_start = cutoff - pd.DateOffset(years=FEATURE_HISTORY_YEARS)
    for ticker, frame in price_map.items():
        if ticker in sector_etfs:  # 섹터 강도 계산용일 뿐 후보가 아니다
            continue
        history = frame.loc[window_start:cutoff]
        if history.empty:
            continue
        snapshot[ticker] = history

    if not snapshot:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    # IC 가중치: 달마다 "전월 말까지의 데이터"로 계산한다.
    # (예전엔 호출 횟수 기준 주기라 리밸런스 설정에 따라 디스크 캐시의 주기 번호가 다른 날짜를 가리켜
    #  나중 날짜로 계산한 가중치를 앞 날짜에 쓰는 미래 정보 누설이 생길 수 있었다)
    ic_override: dict | None = None
    if ic_weights_cache is not None:
        anchor = cutoff.to_period("M") - 1
        anchor_key = str(anchor)
        if ic_weights_cache.get("key") != anchor_key:
            weights = load_ic_weights(ohlcv_hash, anchor_key) if ohlcv_hash else None
            if weights is None:
                try:
                    from screener.alpha_model import compute_ic_weights
                    from screener.config import (
                        ALPHA_IC_LOOKBACK,
                        ALPHA_FORWARD_RETURN_DAYS,
                        ALPHA_IC_MIN_SAMPLES,
                        ALPHA_DEFAULT_WEIGHTS,
                        ALPHA_MODEL_ENABLED,
                    )
                    if ALPHA_MODEL_ENABLED:
                        anchor_end = anchor.end_time
                        history = {t: f.loc[:anchor_end] for t, f in snapshot.items()}
                        weights = compute_ic_weights(
                            {t: f for t, f in history.items() if not f.empty},
                            lookback=ALPHA_IC_LOOKBACK,
                            forward_days=ALPHA_FORWARD_RETURN_DAYS,
                            min_samples=ALPHA_IC_MIN_SAMPLES,
                            default_weights=ALPHA_DEFAULT_WEIGHTS,
                        )
                        if ohlcv_hash and weights:
                            save_ic_weights(ohlcv_hash, anchor_key, weights)
                except Exception:
                    weights = None
            ic_weights_cache["key"] = anchor_key
            ic_weights_cache["weights"] = weights
        ic_override = ic_weights_cache.get("weights")

    # ── 캐싱 대상: compute_features_snapshot (느린 부분) ────────────────────────
    # 디스크 캐시에서 feature 스냅샷 로드 시도
    features: pd.DataFrame | None = None
    if ohlcv_hash:
        features = load_feature_snapshot(ohlcv_hash, cutoff)

    if features is None:
        features = compute_features_snapshot(
            snapshot,
            include_fundamentals=include_fundamentals,
            ic_weights_override=ic_override,
            fundamentals_df=fundamentals_df,
            chart_map=chart_loader() if chart_loader is not None else None,
        )
        if ohlcv_hash and not features.empty:
            save_feature_snapshot(ohlcv_hash, cutoff, features)

    # ── 매번 실행: config 임계치에 의존하는 빠른 scoring 단계 ───────────────────
    if features.empty:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    liquid = liquidity_filter(features)
    if liquid.empty:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    if "섹터" in liquid.columns and "티커" in liquid.columns:
        liquid = liquid.copy()
        liquid["섹터"] = fill_unknown_sectors(liquid["섹터"], liquid["티커"])
    neutral = apply_neutralization(liquid)

    # 섹터 로테이션 필터링 (메인 워크플로우와 동일하게)
    if SECTOR_ROTATION_ENABLED and "섹터" in neutral.columns:
        sector_etf_tickers = list(SECTOR_ETFS.values())
        etf_data = {t: price_map[t].loc[:cutoff] for t in sector_etf_tickers if t in price_map}
        spy_data = price_map.get("SPY", pd.DataFrame()).loc[:cutoff]

        if not spy_data.empty:
            strong_sectors = get_strong_sectors(etf_data, spy_data)
            neutral["in_strong_sector"] = mark_strong_sectors(neutral["섹터"], strong_sectors)
        else:
            neutral["in_strong_sector"] = True
    else:
        neutral["in_strong_sector"] = True

    ranked = attach_signals_and_sort(neutral)
    cache[cutoff] = ranked
    return ranked
