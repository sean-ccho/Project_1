"""출력 및 연동 유틸리티."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from config import (
    EXPORT_COLUMNS,
    GOOGLE_SHEETS_CREDENTIALS_PATH,
    GOOGLE_SHEETS_ENABLED,
    GOOGLE_SHEETS_SPREADSHEET_ID,
    GOOGLE_SHEETS_PORTFOLIO_WORKSHEET,
    GOOGLE_SHEETS_PORTFOLIO_TICKER_COLUMN,
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


def _resize_worksheet_to_data(worksheet, rows: list[list[object]]) -> None:
    """Shrink worksheet grid to match payload size and avoid row bloat."""

    row_count = max(len(rows), 1)
    max_cols = max((len(row) for row in rows), default=1)
    worksheet.resize(rows=row_count, cols=max_cols)


def _open_sheet():
    """Authorize gspread client and return spreadsheet handle."""

    if not GOOGLE_SHEETS_ENABLED:
        return None

    if not GOOGLE_SHEETS_SPREADSHEET_ID or not GOOGLE_SHEETS_CREDENTIALS_PATH:
        print(
            "[Google Sheets] spreadsheet ID 또는 credentials path가 설정되지 않았습니다."
        )
        return None

    if gspread is None or Credentials is None:
        print("[Google Sheets] gspread / google-auth 라이브러리가 설치되어 있지 않습니다.")
        return None

    try:
        credentials = Credentials.from_service_account_file(
            GOOGLE_SHEETS_CREDENTIALS_PATH,
            scopes=GOOGLE_SCOPES,
        )
        client = gspread.authorize(credentials)
        return client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
    except Exception as exc:
        print(f"[Google Sheets] 스프레드시트 열기 실패: {exc}")
        return None


def prepare_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in EXPORT_COLUMNS if col in df.columns]
    export_df = df[cols].copy()

    for col in PERCENT_COLUMNS:
        if col in export_df.columns:
            export_df[col] = (export_df[col] * 100).round(1)

    numeric_cols = export_df.select_dtypes(include="number").columns
    export_df[numeric_cols] = export_df[numeric_cols].round(3)

    export_df = export_df.rename(columns=TECH_COLUMN_LABELS)

    cash_col = TECH_COLUMN_LABELS.get("최근20일평균거래대금", "최근20일평균거래대금")
    if cash_col in export_df.columns:
        export_df[cash_col] = (export_df[cash_col] / 1_000_000).round(1)

    return export_df


def export_to_google_sheet(
    df: pd.DataFrame,
    worksheet_name: str | None = None,
) -> bool:
    """Google Sheets에 DataFrame을 업로드한다. 성공 시 True."""

    sheet = _open_sheet()
    if sheet is None:
        return False

    target_worksheet = worksheet_name or GOOGLE_SHEETS_PORTFOLIO_WORKSHEET
    if not target_worksheet:
        print("[Google Sheets] 업데이트할 워크시트가 지정되지 않았습니다.")
        return False

    cleaned_df = df.where(pd.notnull(df), "")
    columns = cleaned_df.columns.tolist()
    rows = [columns] + cleaned_df.values.tolist()
    est = timezone(timedelta(hours=-5), name="EST")
    timestamp = datetime.now(est).strftime("%Y-%m-%d %H:%M:%S EST")
    for row in rows:
        row.append("")
    rows[0][-1] = timestamp

    try:
        try:
            worksheet = sheet.worksheet(target_worksheet)
            worksheet.clear()
        except WorksheetNotFound:
            worksheet = sheet.add_worksheet(
                title=target_worksheet, rows="1000", cols="50"
            )
        worksheet.update(rows, value_input_option="USER_ENTERED")
        _resize_worksheet_to_data(worksheet, rows)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        print(
            f"[Google Sheets] '{target_worksheet}' 워크시트 업데이트 실패: {exc}"
        )
        return False


def fetch_tickers_from_sheet(
    worksheet_name: str | None = GOOGLE_SHEETS_PORTFOLIO_WORKSHEET,
    ticker_column: str | None = GOOGLE_SHEETS_PORTFOLIO_TICKER_COLUMN,
) -> list[str]:
    """지정한 워크시트에서 티커 목록을 불러온다."""

    if not worksheet_name:
        return []

    sheet = _open_sheet()
    if sheet is None:
        return []

    try:
        worksheet = sheet.worksheet(worksheet_name)
    except WorksheetNotFound:
        print(
            f"[Google Sheets] '{worksheet_name}' 워크시트를 찾을 수 없습니다."
        )
        return []
    except Exception as exc:  # pragma: no cover - best effort
        print(
            f"[Google Sheets] '{worksheet_name}' 워크시트 접근 실패: {exc}"
        )
        return []

    try:
        values = worksheet.get_all_values()
    except Exception as exc:  # pragma: no cover - best effort
        print(
            f"[Google Sheets] '{worksheet_name}' 워크시트에서 값 읽기 실패: {exc}"
        )
        return []

    if not values:
        return []

    header = values[0] if values else []
    data_rows = values
    column_index = 0

    if header:
        data_rows = values[1:]
        if ticker_column:
            normalized_target = ticker_column.strip().lower()
            for idx, column_name in enumerate(header):
                if column_name.strip().lower() == normalized_target:
                    column_index = idx
                    break

    tickers: list[str] = []
    for row in data_rows:
        if column_index >= len(row):
            continue
        symbol = row[column_index].strip()
        if not symbol:
            continue
        tickers.append(symbol.upper())

    return tickers
