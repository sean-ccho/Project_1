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
    EMAIL_BOTTOM_SCORE_THRESHOLD,
    EMAIL_MOMENTUM_SCORE_THRESHOLD,
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
    import os
    import json

    if not GOOGLE_SHEETS_ENABLED:
        return None

    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        print("[Google Sheets] spreadsheet ID가 설정되지 않았습니다.")
        return None

    if gspread is None or Credentials is None:
        print("[Google Sheets] gspread / google-auth 라이브러리가 설치되어 있지 않습니다.")
        return None

    try:
        # 1. 환경 변수에서 서비스 계정 정보 확인 (GitHub Actions용)
        service_account_json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        
        if service_account_json_str:
            # 환경 변수가 있으면 JSON 파싱해서 사용
            service_account_info = json.loads(service_account_json_str)
            credentials = Credentials.from_service_account_info(
                service_account_info,
                scopes=GOOGLE_SCOPES,
            )
        elif GOOGLE_SHEETS_CREDENTIALS_PATH:
            # 환경 변수가 없으면 파일에서 로드 (로컬 개발용)
            credentials = Credentials.from_service_account_file(
                GOOGLE_SHEETS_CREDENTIALS_PATH,
                scopes=GOOGLE_SCOPES,
            )
        else:
            print("[Google Sheets] 서비스 계정 인증 정보가 없습니다 (환경 변수 또는 파일 필요)")
            return None
        
        client = gspread.authorize(credentials)
        return client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
    except Exception as exc:
        print(f"[Google Sheets] 스프레드시트 열기 실패: {exc}")
        return None


def generate_chart_paths(ticker: str) -> dict[str, str]:
    """
    티커에 대한 차트 이미지 로컬 경로를 생성합니다.
    
    Args:
        ticker: 종목 심볼
        
    Returns:
        {차트_타임프레임: 로컬경로} 딕셔너리
    """
    from pathlib import Path
    from config import CHARTS_OUTPUT_DIR, CHARTS_TIMEFRAMES
    
    paths = {}
    for tf in CHARTS_TIMEFRAMES:
        # 타임스탬프가 포함된 파일 패턴 검색 (예: AAPL_Daily_20240521_123456.png)
        pattern = f"{ticker}_{tf}_*.png"
        matches = list(Path(CHARTS_OUTPUT_DIR).glob(pattern))
        
        if matches:
            # 가장 최근 파일 선택 (파일명 정렬로 충분, 타임스탬프 순)
            latest_file = sorted(matches)[-1]
            paths[f"차트_{tf}"] = str(latest_file)
        else:
            paths[f"차트_{tf}"] = ""
    
    return paths


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

    # Extract all HYPERLINK("url", "text") parts
    # Example: =HYPERLINK("u1", "t1") & CHAR(10) & HYPERLINK("u2", "t2")
    # Using (?:[^"]|"")* to allow escaped double quotes within the text
    import re
    matches = []
    # Match HYPERLINK, then url, then text until the closing quote
    # Text can contain "" for a single " in Excel
    for match in re.finditer(r'HYPERLINK\("(?P<url>[^"]+)",\s*"(?P<text>(?:[^"]|"")*)"\)', formula):
        url = match.group("url")
        text = match.group("text").replace('""', '"') # Convert Excel escape back to normal quote
        matches.append((url, text))

    if not matches:
        return formula.replace("\n", "<br>")

    html_links = [f'<a href="{url}">{text}</a>' for url, text in matches]
    return "<br>".join(html_links)


def send_email_notification(df: pd.DataFrame) -> bool:
    """바닥 반등/모멘텀 적합도가 임계치 이상인 종목 리스트를 이메일로 발송한다."""

    if not EMAIL_ENABLED:
        return False

    ticker_col = TECH_COLUMN_LABELS.get("티커", "티커")
    name_col = TECH_COLUMN_LABELS.get("회사", "회사")
    price_col = TECH_COLUMN_LABELS.get("현재가격", "현재가격")
    rec_col = TECH_COLUMN_LABELS.get("추천", "추천")
    news_col = TECH_COLUMN_LABELS.get("최근뉴스", "최근뉴스")
    bottom_col = TECH_COLUMN_LABELS.get("바닥반등_적합도_표시", "바닥반등_적합도_표시")
    momentum_col = TECH_COLUMN_LABELS.get("모멘텀_적합도_표시", "모멘텀_적합도_표시")

    def extract_score(text):
        try:
            if "(" in str(text) and ")" in str(text):
                return float(str(text).split("(")[1].split(")")[0])
            return 0.0
        except Exception:
            return 0.0

    target = df.copy()

    # 바닥 반등 필터링
    bottom_df = pd.DataFrame()
    if bottom_col in target.columns:
        target["_bottom_score"] = target[bottom_col].apply(extract_score)
        bottom_df = target[target["_bottom_score"] >= EMAIL_BOTTOM_SCORE_THRESHOLD]

    # 모멘텀 필터링
    momentum_df = pd.DataFrame()
    if momentum_col in target.columns:
        target["_momentum_score"] = target[momentum_col].apply(extract_score)
        momentum_df = target[target["_momentum_score"] >= EMAIL_MOMENTUM_SCORE_THRESHOLD]

    if bottom_df.empty and momentum_df.empty:
        print("[Email] 두 전략 모두 임계치 이상인 종목이 없어 이메일을 보내지 않습니다.")
        return False

    def _build_table(section_df, score_col_name, score_label):
        if section_df.empty:
            return f"<p>해당 종목 없음</p>"
        html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += f"<tr style='background-color: #f2f2f2;'><th>티커</th><th>회사명</th><th>현재가</th><th>{score_label}</th><th>추천</th><th>최근뉴스</th></tr>"
        for _, row in section_df.iterrows():
            ticker = row.get(ticker_col, "")
            name = row.get(name_col, "")
            price = row.get(price_col, "")
            score_val = row.get(score_col_name, "")
            rec = row.get(rec_col, "")
            news_html = _formula_to_html(str(row.get(news_col, "")))
            html += f"<tr><td>{ticker}</td><td>{name}</td><td>{price}</td><td>{score_val}</td><td>{rec}</td><td>{news_html}</td></tr>"
        html += "</table>"
        return html

    date_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"[Stock Signals] {date_str} 전략별 매수 적합 종목 리스트"

    body = f"<h2>{date_str} 분석 결과</h2>"
    body += f"<h3>📉 바닥 반등 전략 ({EMAIL_BOTTOM_SCORE_THRESHOLD}점 이상, {len(bottom_df)}개)</h3>"
    body += _build_table(bottom_df, bottom_col, "바닥반등 적합도")
    body += "<br>"
    body += f"<h3>🚀 모멘텀 추격 전략 ({EMAIL_MOMENTUM_SCORE_THRESHOLD}점 이상, {len(momentum_df)}개)</h3>"
    body += _build_table(momentum_df, momentum_col, "모멘텀 적합도")
    body += "<br><p>본 메일은 시스템에 의해 자동으로 발송되었습니다.</p>"

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

        total = len(bottom_df) + len(momentum_df)
        print(f"[Email] 바닥반등 {len(bottom_df)}개 + 모멘텀 {len(momentum_df)}개 = 총 {total}개 종목 리스트를 {EMAIL_RECIPIENT}로 발송 완료")
        return True
    except Exception as exc:
        print(f"[Email] 이메일 발송 실패: {exc}")
        return False
