"""출력 및 연동 유틸리티."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from config import (
    EXPORT_COLUMNS,
    GOOGLE_SHEETS_CREDENTIALS_PATH,
    GOOGLE_SHEETS_ENABLED,
    GOOGLE_SHEETS_SPREADSHEET_ID,
    GOOGLE_SHEETS_SIGNALS_WORKSHEET,
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

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import (
    EXPORT_COLUMNS,
    GOOGLE_SHEETS_CREDENTIALS_PATH,
    GOOGLE_SHEETS_ENABLED,
    GOOGLE_SHEETS_SPREADSHEET_ID,
    GOOGLE_SHEETS_SIGNALS_WORKSHEET,
    GOOGLE_SHEETS_PORTFOLIO_WORKSHEET,
    GOOGLE_SHEETS_PORTFOLIO_TICKER_COLUMN,
    PERCENT_COLUMNS,
    TECH_COLUMN_LABELS,
    EMAIL_ENABLED,
    EMAIL_SENDER,
    EMAIL_PASSWORD,
    EMAIL_RECIPIENT,
    SMTP_SERVER,
    SMTP_PORT,
    EMAIL_SCORE_THRESHOLD,
)

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

    target_worksheet = worksheet_name or GOOGLE_SHEETS_SIGNALS_WORKSHEET
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


def export_backtest_results(
    results: dict[str, dict], worksheet_name: str = "백테스트"
) -> bool:
    """백테스트 결과를 Google Sheets에 업로드한다."""

    if not results:
        return False

    sheet = _open_sheet()
    if sheet is None or not worksheet_name:
        return False

    try:
        try:
            worksheet = sheet.worksheet(worksheet_name)
        except WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=worksheet_name, rows="4000", cols="20")

        rows: list[list[str]] = []
        for label, outcome in results.items():
            summary = outcome.get("summary", {})
            trades = outcome.get("trades")

            rows.append([str(label)])
            rows.append(["Metric", "Value"])
            for key, value in summary.items():
                if isinstance(value, float):
                    rows.append([key, f"{value:.6f}"])
                else:
                    rows.append([key, str(value)])
            rows.append([])

            if isinstance(trades, pd.DataFrame) and not trades.empty:
                rows.append(["Trades"])
                rows.append(trades.columns.tolist())
                rows.extend(trades.astype(str).values.tolist())
            else:
                rows.append(["Trades", "No trades generated."])

            rows.append([])
            rows.append([])

        worksheet.clear()
        worksheet.update(rows, value_input_option="USER_ENTERED")
        _resize_worksheet_to_data(worksheet, rows)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[Google Sheets] 백테스트 결과 업로드 실패: {exc}")
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


def _formula_to_html(formula: str) -> str:
    """Convert Excel HYPERLINK formulas to HTML <a> tags."""
    if not formula or not isinstance(formula, str):
        return ""
    if not formula.startswith("="):
        return formula.replace("\n", "<br>")

    import re

    # Extract all HYPERLINK("url", "text") parts
    # Example: =HYPERLINK("u1", "t1") & CHAR(10) & HYPERLINK("u2", "t2")
    pattern = r'HYPERLINK\("([^"]+)",\s*"([^"]+)"\)'
    matches = re.findall(pattern, formula)

    if not matches:
        return formula.replace("\n", "<br>")

    html_links = [f'<a href="{url}">{text}</a>' for url, text in matches]
    return "<br>".join(html_links)


def send_email_notification(df: pd.DataFrame) -> bool:
    """매수적합도가 임계치 이상인 종목 리스트를 이메일로 발송한다."""

    if not EMAIL_ENABLED:
        return False

    # 필터링: 매수적합도(Entry Score)가 임계치 이상인 종목만 추출
    # prepare_export_dataframe에서 컬럼명이 '매수적합도_표시'로 변경되었을 수 있으므로 처리
    score_col = TECH_COLUMN_LABELS.get("매수적합도_표시", "매수적합도_표시")
    ticker_col = TECH_COLUMN_LABELS.get("티커", "티커")
    name_col = TECH_COLUMN_LABELS.get("회사", "회사")
    price_col = TECH_COLUMN_LABELS.get("현재가격", "현재가격")
    rec_col = TECH_COLUMN_LABELS.get("추천", "추천")
    news_col = TECH_COLUMN_LABELS.get("최근뉴스", "최근뉴스")

    # '매수적합도_표시' 컬럼에서 숫자 점수 추출 (예: "★★★★☆ (4.2)" -> 4.2)
    def extract_score(text):
        try:
            if "(" in text and ")" in text:
                return float(text.split("(")[1].split(")")[0])
            return 0.0
        except Exception:
            return 0.0

    target_stocks = df.copy()
    if score_col in target_stocks.columns:
        target_stocks["temp_score"] = target_stocks[score_col].apply(extract_score)
        high_score_df = target_stocks[target_stocks["temp_score"] >= EMAIL_SCORE_THRESHOLD]
    else:
        print(f"[Email] '{score_col}' 컬럼을 찾을 수 없어 이메일을 보낼 수 없습니다.")
        return False

    if high_score_df.empty:
        print(f"[Email] 매수적합도 {EMAIL_SCORE_THRESHOLD}점 이상인 종목이 없어 이메일을 보내지 않습니다.")
        return False

    # 이메일 내용 구성
    date_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"[Stock Signals] {date_str} 매수적합도 {EMAIL_SCORE_THRESHOLD}점 이상 종목 리스트"
    
    body = f"<h2>{date_str} 분석 결과 매수적합도 {EMAIL_SCORE_THRESHOLD}점 이상 종목입니다.</h2>"
    body += "<table border='1' style='border-collapse: collapse;'>"
    body += "<tr style='background-color: #f2f2f2;'><th>티커</th><th>회사명</th><th>현재가</th><th>매수적합도</th><th>추천</th><th>최근뉴스</th></tr>"
    
    for _, row in high_score_df.iterrows():
        ticker = row.get(ticker_col, "Unknown")
        name = row.get(name_col, "Unknown")
        price = row.get(price_col, "Unknown")
        score = row.get(score_col, "Unknown")
        rec = row.get(rec_col, "Unknown")
        news_formula = row.get(news_col, "")
        news_html = _formula_to_html(str(news_formula))
        
        body += f"<tr><td>{ticker}</td><td>{name}</td><td>{price}</td><td>{score}</td><td>{rec}</td><td>{news_html}</td></tr>"
    
    body += "</table>"
    body += "<br><p>본 메일은 시스템에 의해 자동으로 발송되었습니다.</p>"

    # 이메일 발송
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECIPIENT
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"[Email] {len(high_score_df)}개의 고점수 종목 리스트를 {EMAIL_RECIPIENT}로 발송 완료")
        return True
    except Exception as exc:
        print(f"[Email] 이메일 발송 실패: {exc}")
        return False
