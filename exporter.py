"""출력 및 연동 유틸리티."""

import pandas as pd

from config import (
    EXPORT_COLUMNS,
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
