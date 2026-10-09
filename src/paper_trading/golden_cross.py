"""골든크로스 임박/직후 종목 추출 (PT-1 이메일 '골든크로스임박' 섹션).

입력은 ranked DataFrame 또는 그 행(dict) 리스트. pandas 없이도 동작한다.
"""

from __future__ import annotations

from typing import Any, Iterable

# 패턴 컬럼 ↔ MA갭 컬럼 매핑.
# gap_col이 없으면 conf_col로 폴백 (임박 conf만 역산 가능, 직후는 미감지).
# 폴백은 features.py 업데이트 전의 기존 parquet에서 회귀를 막기 위한 안전망.
PATTERN_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("일봉패턴", "일봉", "ema_gap_20_50", "일봉_골든크로스_신뢰도"),
    ("주봉패턴", "주봉", "주봉_MA갭", "주봉_골든크로스_신뢰도"),
    ("월봉패턴", "월봉", "월봉_MA갭", "월봉_골든크로스_신뢰도"),
)


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _conf_to_neg_gap(conf: float) -> float:
    # patterns.py 임박 공식: conf = 0.7 + (0.05 - |gap|) / 0.05 * 0.25 → |gap| 역산 (항상 음수 = 임박).
    if conf <= 0:
        return 0.0
    return -max(0.0, 0.05 - 0.2 * (conf - 0.7))


def _to_rows(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        if getattr(data, "empty", False):
            return []
        return data.to_dict("records")
    return list(data)


def extract_golden_cross(data: Any) -> list[dict]:
    """골든크로스 임박/직후 종목을 추출한다.

    각 TF(일봉/주봉/월봉) 패턴 컬럼에서:
      - "골든크로스임박" → 음수 갭 (단기MA < 장기MA, 5% 이내)
      - "골든크로스" → 양수 갭 (방금 교차, 0~5% 이내만 포함)
    부호로 임박/직후를 구분한다 (표시 레이어가 +/- 그대로 출력).

    정렬:
      1) 다중 TF 우선 (tf_count 내림차순)
      2) 일봉 포함 우선
      3) 갭 0에 가장 가까운 순 (교차 시점 근접도)
    """
    rows: Iterable[dict[str, Any]] = _to_rows(data)
    rows = list(rows)
    if not rows or "티커" not in rows[0]:
        return []

    result: list[dict] = []
    for row in rows:
        ticker = str(row.get("티커", "")).strip()
        if not ticker:
            continue
        hit_tfs: list[str] = []
        tf_gaps: list[tuple[str, float]] = []
        for col, label, gap_col, conf_col in PATTERN_COLUMNS:
            if col not in row:
                continue
            tokens = [t.strip() for t in str(row.get(col, "")).split(",")]
            is_imminent = "골든크로스임박" in tokens
            is_passed = "골든크로스" in tokens
            if not (is_imminent or is_passed):
                continue
            if gap_col in row:
                gap = _safe_float(row.get(gap_col))
            elif is_imminent:
                gap = _conf_to_neg_gap(_safe_float(row.get(conf_col)))
            else:
                continue
            if is_imminent:
                if gap > 0 or abs(gap) > 0.05:
                    continue
            else:
                # 0~5% 이내만 (HCWB +37% 같은 과이격 제외)
                if gap < 0 or gap > 0.05:
                    continue
            hit_tfs.append(label)
            tf_gaps.append((label, gap))
        if hit_tfs:
            result.append({
                "ticker": ticker,
                "sector": str(row.get("섹터", "")),
                "current_price": _safe_float(row.get("현재가격", row.get("close"))),
                "timeframes": hit_tfs,
                "tf_gaps": tf_gaps,
                "strategy": str(row.get("전략구분", "")),
                "star": str(row.get("매수적합도_표시", "")),
                "vol_ratio": _safe_float(row.get("거래량돌파배수")),
                "vol_ma20": _safe_float(row.get("volume_ma20")),
                "tf_count": len(hit_tfs),
                "has_daily": "일봉" in hit_tfs,
            })

    result.sort(
        key=lambda g: (-g["tf_count"], not g["has_daily"], min(abs(x) for _, x in g["tf_gaps"]))
    )
    return result
