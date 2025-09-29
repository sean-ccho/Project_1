"""CSV 내보내기 도우미."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from config import EXPORT_WITH_BACKUP, PERCENT_COLUMNS, TECH_COLUMN_LABELS


def export_table(df: pd.DataFrame, folder: str = "output") -> Tuple[Path, Path]:
    """시그널 표를 최신/타임스탬프 Excel 파일로 저장한다."""

    path = Path(folder)
    path.mkdir(exist_ok=True)

    latest = path / "신호_최신.xlsx"
    stamped = None
    if EXPORT_WITH_BACKUP:
        from datetime import datetime

        stamped = path / f"신호_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    keep = [
        "판단",
        "추천",
        "메모",
        "티커",
        "우선순위",
        "트렌드점수_최종",
        "트렌드점수",
        "RSI",
        "macd",
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
        "annual_dividend",
        "dividend_yield",
        "최근20일평균거래대금",
    ]

    export_df = df[keep].copy()
    for col in PERCENT_COLUMNS:
        if col in export_df.columns:
            export_df[col] = (export_df[col] * 100).round(1)

    numeric_cols = export_df.select_dtypes(include="number").columns
    export_df[numeric_cols] = export_df[numeric_cols].round(1)

    export_df = export_df.rename(columns=TECH_COLUMN_LABELS)

    cash_col = TECH_COLUMN_LABELS.get("최근20일평균거래대금", "최근20일평균거래대금")
    if cash_col in export_df.columns:
        export_df[cash_col] = (export_df[cash_col] / 1_000_000).round(1)

    with pd.ExcelWriter(latest, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Signals")
    if stamped:
        with pd.ExcelWriter(stamped, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Signals")
    return stamped or latest, latest
