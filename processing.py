"""특징 후처리(중립화, 유동성 필터)를 담당하는 모듈."""

import pandas as pd

from config import LIQUIDITY_QUANTILE, LIQUIDITY_WHITELIST, NEUTRALIZE_COLUMNS


def zscore_by_group(series: pd.Series, group: pd.Series) -> pd.Series:
    """그룹별 평균·표준편차를 사용해 Z-score를 계산한다."""

    return series.groupby(group).transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))


def apply_neutralization(df: pd.DataFrame) -> pd.DataFrame:
    """시장/섹터 단위로 값을 정규화해 특정 집단에 치우친 효과를 줄인다."""

    out = df.copy()
    for col in NEUTRALIZE_COLUMNS:
        # 시장(US/CA) 단위 중립화: 전체 지표가 시장 방향성만 반영하지 않도록 보정.
        out[col + "_mktz"] = zscore_by_group(out[col], out["시장"])

    if (out["섹터"] != "Unknown").any():
        # 섹터 정보가 있으면 추가로 섹터 단위 중립화를 적용한다.
        for col in NEUTRALIZE_COLUMNS:
            out[col + "_secz"] = zscore_by_group(out[col], out["섹터"])
        out["트렌드점수_최종"] = (
            0.5 * out["트렌드점수"]
            + 0.3 * out["트렌드점수_mktz"]
            + 0.2 * out["트렌드점수_secz"]
        )
    else:
        # 섹터 정보가 없으면 시장 중립화 지표만 혼합하여 활용한다.
        out["트렌드점수_최종"] = 0.7 * out["트렌드점수"] + 0.3 * out["트렌드점수_mktz"]

    return out


def liquidity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """시장별 거래대금 분위수를 이용해 부족한 유동성 종목을 제거한다."""

    out = df.copy()
    thresholds = out.groupby("시장")["최근20일평균거래대금"].transform(
        lambda s: s.quantile(1.0 - LIQUIDITY_QUANTILE)
    )
    mask = out["최근20일평균거래대금"] >= thresholds
    if LIQUIDITY_WHITELIST:
        mask |= out["티커"].isin(LIQUIDITY_WHITELIST)
    return out[mask]
