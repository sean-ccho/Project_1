"""출력 및 연동 유틸리티."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import (
    EXPORT_WITH_BACKUP,
    GOOGLE_SHEETS_CREDENTIALS_PATH,
    GOOGLE_SHEETS_ENABLED,
    GOOGLE_SHEETS_SPREADSHEET_ID,
    GOOGLE_SHEETS_WORKSHEET,
    PERCENT_COLUMNS,
    TECH_COLUMN_LABELS,
)

try:  # Google Sheets는 선택적 의존성
    import gspread
    from google.oauth2.service_account import Credentials
    from gspread.exceptions import WorksheetNotFound
except ImportError:  # pragma: no cover - optional path
    gspread = None
    Credentials = None
    WorksheetNotFound = None

EXPORT_COLUMNS = [
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

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def prepare_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in EXPORT_COLUMNS if col in df.columns]
    export_df = df[cols].copy()

    for col in PERCENT_COLUMNS:
        if col in export_df.columns:
            export_df[col] = (export_df[col] * 100).round(1)

    numeric_cols = export_df.select_dtypes(include="number").columns
    export_df[numeric_cols] = export_df[numeric_cols].round(1)

    export_df = export_df.rename(columns=TECH_COLUMN_LABELS)

    cash_col = TECH_COLUMN_LABELS.get("최근20일평균거래대금", "최근20일평균거래대금")
    if cash_col in export_df.columns:
        export_df[cash_col] = (export_df[cash_col] / 1_000_000).round(1)

    return export_df


def export_table(df: pd.DataFrame, folder: str = "output") -> Tuple[Path, Path, pd.DataFrame]:
    """시그널 표를 최신/타임스탬프 Excel 파일로 저장하고 가공된 DataFrame을 반환한다."""

    path = Path(folder)
    path.mkdir(exist_ok=True)

    export_df = prepare_export_dataframe(df)

    latest = path / "신호_최신.xlsx"
    stamped: Optional[Path] = None
    if EXPORT_WITH_BACKUP:
        from datetime import datetime

        stamped = path / f"신호_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    with pd.ExcelWriter(latest, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Signals")
    if stamped:
        with pd.ExcelWriter(stamped, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Signals")

    return stamped or latest, latest, export_df


def export_to_google_sheet(df: pd.DataFrame) -> bool:
    """Google Sheets에 DataFrame을 업로드한다. 성공 시 True."""

    if not GOOGLE_SHEETS_ENABLED:
        return False

    if not GOOGLE_SHEETS_SPREADSHEET_ID or not GOOGLE_SHEETS_CREDENTIALS_PATH:
        print("[Google Sheets] spreadsheet ID 또는 credentials path가 설정되지 않았습니다.")
        return False

    if gspread is None or Credentials is None:
        print("[Google Sheets] gspread / google-auth 라이브러리가 설치되어 있지 않습니다.")
        return False

    try:
        credentials = Credentials.from_service_account_file(
            GOOGLE_SHEETS_CREDENTIALS_PATH,
            scopes=GOOGLE_SCOPES,
        )
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)

        try:
            worksheet = sheet.worksheet(GOOGLE_SHEETS_WORKSHEET)
            worksheet.clear()
        except WorksheetNotFound:
            worksheet = sheet.add_worksheet(
                title=GOOGLE_SHEETS_WORKSHEET, rows="1000", cols="50"
            )

        cleaned_df = df.where(pd.notnull(df), "")
        rows = [cleaned_df.columns.tolist()] + cleaned_df.values.tolist()
        worksheet.update(rows)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[Google Sheets] 업로드 실패: {exc}")
        return False
