"""출력 및 연동 유틸리티."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from screener.config import (
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
from email.mime.base import MIMEBase
from email import encoders

from screener.config import (
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
    EMAIL_RECIPIENTS,
    SMTP_SERVER,
    SMTP_PORT,
    EMAIL_SCORE_THRESHOLD,
    EMAIL_BOTTOM_SCORE_THRESHOLD,
    EMAIL_MOMENTUM_SCORE_THRESHOLD,
    HOLD_WINNERS_DEFER_FREEZE_DAYS,
    HOLD_WINNERS_MIN_CHECKS,
    HOLD_WINNERS_MAX_DEFERS,
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
    from screener.config import CHARTS_OUTPUT_DIR, CHARTS_TIMEFRAMES
    
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
    if "[헬스체크]" not in df.columns or "[시총]" not in df.columns:
        try:
            from screener.fundamentals import calculate_health_score, format_market_cap
            df = df.copy()
            if "[헬스체크]" not in df.columns:
                health = df.apply(calculate_health_score, axis=1, result_type="expand")
                df["[헬스체크]"] = health[1]
            if "[시총]" not in df.columns:
                df["[시총]"] = df.apply(format_market_cap, axis=1)
        except Exception:
            pass

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



def send_email_notification(df: pd.DataFrame, subject_prefix: str = "[Stock Signals]") -> bool:
    """이메일로 매수 적합 종목 리스트를 발송한다."""
    if not EMAIL_ENABLED:
        return False

    ticker_col = TECH_COLUMN_LABELS.get("티커", "티커")
    name_col = TECH_COLUMN_LABELS.get("회사", "회사")
    price_col = TECH_COLUMN_LABELS.get("현재가격", "현재가격")
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

    # 점수 추출
    if bottom_col in target.columns:
        target["_bottom_score"] = target[bottom_col].apply(extract_score)
    else:
        target["_bottom_score"] = 0.0
    if momentum_col in target.columns:
        target["_momentum_score"] = target[momentum_col].apply(extract_score)
    else:
        target["_momentum_score"] = 0.0

    bt = EMAIL_BOTTOM_SCORE_THRESHOLD
    mt = EMAIL_MOMENTUM_SCORE_THRESHOLD

    # 2섹션 분류: 바닥반등 / 모멘텀 (전환구간 제거 — 우세 전략으로 재분류됨)
    bottom_only_df = target[target["_bottom_score"] >= bt]
    momentum_only_df = target[
        (target["_bottom_score"] < bt) & (target["_momentum_score"] >= mt)
    ]

    if bottom_only_df.empty and momentum_only_df.empty:
        print("[Email] 세 섹션 모두 해당 종목이 없어 이메일을 보내지 않습니다.")
        return False

    # 내부자 거래 요약: 이메일 후보 티커에 대해서만 조회 (표시용)
    insider_map: dict[str, str] = {}
    try:
        from screener.insider import fetch_insider_snapshots

        candidate_tickers = pd.concat(
            [bottom_only_df.get(ticker_col), momentum_only_df.get(ticker_col)]
        ).dropna().astype(str).str.upper().str.strip().unique().tolist()
        if candidate_tickers:
            insider_df = fetch_insider_snapshots(candidate_tickers)
            if not insider_df.empty:
                insider_map = {
                    str(r["티커"]).upper().strip(): r["내부자(90일)"]
                    for _, r in insider_df.iterrows()
                }
    except Exception as exc:  # 내부자 조회 실패는 이메일 발송을 막지 않음
        print(f"[Email] 내부자 정보 조회 실패 (무시): {exc}")

    def _build_table(section_df, score_cols):
        """score_cols: list of (col_name, label) tuples."""
        if section_df.empty:
            return "<p>해당 종목 없음</p>"
        score_headers = "".join(f"<th>{label}</th>" for _, label in score_cols)
        html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += f"<tr style='background-color: #f2f2f2;'><th>티커</th><th>회사명</th><th>현재가</th>{score_headers}<th>내부자(90일)</th><th>일봉패턴</th><th>주봉패턴</th><th>월봉패턴</th></tr>"
        for _, row in section_df.iterrows():
            ticker = row.get(ticker_col, "")
            name = row.get(name_col, "")
            price = row.get(price_col, "")
            score_cells = "".join(f"<td>{row.get(col, '')}</td>" for col, _ in score_cols)
            insider = insider_map.get(str(ticker).upper().strip(), "—")
            daily_pat = str(row.get("일봉패턴", "") or "-")
            weekly_pat = str(row.get("주봉패턴", "") or "-")
            monthly_pat = str(row.get("월봉패턴", "") or "-")
            html += f"<tr><td>{ticker}</td><td>{name}</td><td>{price}</td>{score_cells}<td>{insider}</td><td>{daily_pat}</td><td>{weekly_pat}</td><td>{monthly_pat}</td></tr>"
        html += "</table>"
        return html

    date_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"{subject_prefix} {date_str} 전략별 매수 적합 종목 리스트"

    body = f"<h2>{date_str} 분석 결과</h2>"
    body += f"<h3>📉 바닥반등 ({len(bottom_only_df)}개)</h3>"
    body += _build_table(bottom_only_df, [(bottom_col, "바닥반등 적합도"), (momentum_col, "모멘텀")])
    body += "<br>"
    body += f"<h3>📈 모멘텀 ({len(momentum_only_df)}개)</h3>"
    body += _build_table(momentum_only_df, [(momentum_col, "모멘텀 적합도")])
    body += "<br><p>본 메일은 시스템에 의해 자동으로 발송되었습니다.</p>"

    try:
        recipients_str = ", ".join(EMAIL_RECIPIENTS)
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_SENDER  # 발신자만 표시 (수신자 목록 숨김)
        msg["Bcc"] = recipients_str  # 실제 수신자는 BCC 처리
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        total = len(bottom_only_df) + len(momentum_only_df)
        print(
            f"[Email] 바닥반등 {len(bottom_only_df)}개 + "
            f"모멘텀 {len(momentum_only_df)}개 = "
            f"총 {total}개 종목 리스트를 {recipients_str}로 발송 완료"
        )
        return True
    except Exception as exc:
        print(f"[Email] 이메일 발송 실패: {exc}")
        return False


def _build_market_analysis_column(item: dict) -> str:
    """단일 종목의 시장 분석 컬럼 내용(헤더 박스 없음)을 생성한다.

    내용이 전혀 없으면 빈 문자열을 반환한다.
    """
    ma = item.get("market_analysis")
    if not ma:
        return ""

    ticker = item.get("ticker", "")
    analyst = ma.get("analyst") or {}
    tech = ma.get("tech") or {}
    news_html_raw = ma.get("news_html", "")
    insider_text = ma.get("insider", "") or ""

    parts: list[str] = []

    if analyst:
        rec = analyst.get("recommendation", "N/A")
        target = analyst.get("target_mean")
        upside = analyst.get("upside_pct")
        num = analyst.get("num_analysts", 0)
        rec_color = {"Buy": "#27ae60", "Sell": "#e74c3c", "Hold": "#e67e22"}.get(rec, "#888")
        analyst_bits = [f"<b style='color:{rec_color}'>{rec}</b>" + (f" ({num}명)" if num else "")]
        if target:
            analyst_bits.append(f"목표가 <b>${target:,.2f}</b>")
        if upside is not None:
            if upside >= 0:
                analyst_bits.append(f"<span style='color:#27ae60;font-weight:bold'>↑{upside:.1%}</span>")
            else:
                analyst_bits.append(f"<span style='color:#e74c3c;font-weight:bold'>↓{abs(upside):.1%}</span>")
        parts.append("<div style='color:#666;font-size:12px;margin-top:4px'>🏛️ 애널리스트</div>")
        parts.append(f"<div style='font-size:12px;padding-bottom:4px'>{' · '.join(analyst_bits)}</div>")

    if tech:
        tech_bits: list[str] = []
        if "rsi" in tech:
            tech_bits.append(f"RSI <b>{tech['rsi']:.0f}</b>")
        if "adx" in tech:
            tech_bits.append(f"ADX <b>{tech['adx']:.0f}</b>")
        if "bb_pband" in tech:
            tech_bits.append(f"BB <b>{tech['bb_pband']:.2f}</b>")
        if "vol_z" in tech:
            tech_bits.append(f"Vol Z <b>{tech['vol_z']:.1f}</b>")
        if "pos_52w" in tech:
            tech_bits.append(f"52주 <b>{tech['pos_52w']:.0%}</b>")
        if tech_bits:
            parts.append("<div style='color:#666;font-size:12px;margin-top:4px'>📈 기술적 지표</div>")
            parts.append(f"<div style='font-size:12px;padding-bottom:4px'>{' · '.join(tech_bits)}</div>")

    if insider_text and insider_text != "—":
        parts.append("<div style='color:#666;font-size:12px;margin-top:4px'>👥 내부자(90일)</div>")
        parts.append(f"<div style='font-size:12px;padding-bottom:4px'>{insider_text}</div>")

    if news_html_raw and news_html_raw not in ("뉴스 조회 실패", "최근 2일 뉴스 없음"):
        import re as _re
        if news_html_raw.startswith("="):
            hyperlinks = _re.findall(r'HYPERLINK\("([^"]*)",\s*"([^"]*)"\)', news_html_raw)
            if hyperlinks:
                bullets = "<br>".join(
                    f'• <a href="{url}" style="color:#2980b9">{txt}</a>'
                    for url, txt in hyperlinks
                )
            else:
                plain = _re.sub(r'=?HYPERLINK\("[^"]*",\s*"([^"]*)"\)', r"\1", news_html_raw)
                lines = [ln.strip() for ln in plain.split("\n") if ln.strip()]
                bullets = "<br>".join(f"• {ln}" for ln in lines)
        else:
            lines = [ln.strip() for ln in news_html_raw.split("\n") if ln.strip()]
            bullets = "<br>".join(f"• {ln}" for ln in lines) if lines else news_html_raw
        parts.append("<div style='color:#666;font-size:12px;margin-top:6px'>📰 뉴스</div>")
        parts.append(f"<div style='font-size:12px;line-height:1.5'>{bullets}</div>")

    if not parts:
        return ""

    warnings: list[str] = []
    if analyst:
        upside = analyst.get("upside_pct")
        target = analyst.get("target_mean")
        current = analyst.get("current_price")
        if upside is not None and upside < -0.05 and target and current:
            warnings.append(
                f"⚠️ 목표가 ${target:,.2f} &lt; 현재가 ${current:,.2f} ({abs(upside):.1%} 낮음)"
            )
    rsi_val = tech.get("rsi")
    if rsi_val and rsi_val > 70:
        warnings.append(f"⚠️ RSI {rsi_val:.0f} 과매수")
    bb_val = tech.get("bb_pband")
    if bb_val and bb_val > 0.80:
        warnings.append(f"⚠️ BB {bb_val:.2f} 상단 근접")
    if warnings:
        color = "#fdecea" if any("목표가" in w for w in warnings) else "#fff8e1"
        border = "#e74c3c" if any("목표가" in w for w in warnings) else "#f39c12"
        parts.append(
            f"<div style='background:{color};border-left:3px solid {border};"
            f"padding:6px;margin-top:8px;font-size:11px;border-radius:0 3px 3px 0'>"
            + "<br>".join(warnings)
            + "</div>"
        )

    header = (
        f"<div style='font-weight:bold;color:#2c3e50;margin-bottom:6px;font-size:13px'>{ticker}</div>"
    )
    return header + "".join(parts)


def _build_market_analysis_card(items: list[dict]) -> str:
    """여러 종목의 시장 분석을 단일 카드(가로 컬럼)로 렌더링한다.

    빈 분석은 자동으로 제외되며, 표시할 컬럼이 하나도 없으면 빈 문자열을 반환한다.
    """
    columns: list[str] = []
    for item in items or []:
        col_html = _build_market_analysis_column(item)
        if col_html:
            columns.append(col_html)

    if not columns:
        return ""

    n = len(columns)
    col_width = 100 / n
    cells: list[str] = []
    for i, col in enumerate(columns):
        is_last = i == n - 1
        is_first = i == 0
        border = "" if is_last else ";border-right:1px solid #c5d9e8"
        padding = "padding:0" if is_first else "padding:0 0 0 4px"
        cells.append(
            f"<td style='width:{col_width:.4f}%;vertical-align:top;{padding}{border}'>"
            f"{col}"
            f"</td>"
        )

    return (
        f"<div style='background:#eaf2f8;border-left:4px solid #3498db;"
        f"padding:14px;margin:8px 0;font-size:13px;border-radius:0 4px 4px 0'>"
        f"<b style='font-size:14px'>📊 시장 분석</b>"
        f"<table style='width:100%;border-collapse:separate;border-spacing:12px 0;"
        f"table-layout:fixed;margin-top:10px'>"
        f"<tr style='vertical-align:top'>"
        + "".join(cells)
        + "</tr></table></div>"
    )


def _build_chart_images_html(
    ticker: str,
    p75_pct: float = 0,
    p75_price: float = 0,
    color_theme: str = "holdings",
) -> str:
    """종목 차트 이미지 3개(Daily/Weekly/Monthly)를 가로로 배치한 HTML 생성.

    charts/screenshots/{ticker}/ 폴더에서 최신 파일을 찾아 GitHub Raw URL로 참조.
    파일이 없으면 빈 문자열 반환 (graceful degradation).
    """
    from pathlib import Path
    import time as _time

    try:
        from screener.config import (
            GITHUB_REPO_NAME,
            GITHUB_BRANCH_NAME,
            CHARTS_OUTPUT_DIR,
            CHARTS_SIGNALS2_OUTPUT_DIR,
        )
    except ImportError:
        return ""

    if not GITHUB_REPO_NAME:
        return ""

    timeframes = ["Daily", "Weekly", "Monthly"]
    chart_urls: dict[str, str] = {}
    cache_buster = int(_time.time())

    # SP500 디렉토리와 NASDAQ/NYSE 디렉토리 모두 탐색
    search_dirs = [CHARTS_OUTPUT_DIR, CHARTS_SIGNALS2_OUTPUT_DIR]

    for tf in timeframes:
        pattern = f"{ticker}_{tf}_*.png"
        for search_dir in search_dirs:
            matches = list(Path(search_dir).glob(f"{ticker}/{pattern}"))
            if matches:
                latest = sorted(matches)[-1]
                # GitHub Raw URL
                rel_path = str(latest)
                url = (
                    f"https://raw.githubusercontent.com/{GITHUB_REPO_NAME}"
                    f"/{GITHUB_BRANCH_NAME}/{rel_path}?v={cache_buster}"
                )
                chart_urls[tf] = url
                break  # 이 timeframe에서 파일 찾았으면 다음 tf로

    if not chart_urls:
        return ""

    # 색상 테마
    if color_theme == "candidates":
        bg = "#faf5ff"
        border = "#e0d4f0"
        icon = "🔍"
        badge_bg = "#8e44ad"
        title_color = "#8e44ad"
    elif color_theme == "buys":
        bg = "#f0faf0"
        border = "#c8e6c9"
        icon = "🟢"
        badge_bg = "#27ae60"
        title_color = "#27ae60"
    elif color_theme == "golden":
        bg = "#fffdf5"
        border = "#f0e6c0"
        icon = "⚡"
        badge_bg = "#b8860b"
        title_color = "#b8860b"
    else:  # holdings
        bg = "#f8fffe"
        border = "#e0e0e0"
        icon = "📈"
        badge_bg = "#27ae60"
        title_color = "#2c3e50"

    # P75 뱃지
    if p75_pct and p75_price:
        badge = (
            f"<span style='margin-left:8px;background:{badge_bg};color:#fff;"
            f"padding:2px 8px;border-radius:12px;font-size:11px'>"
            f"P75: {p75_pct:+.1%} → ${p75_price:,.2f} ⭐</span>"
        )
    elif p75_pct:
        badge = (
            f"<span style='margin-left:8px;background:{badge_bg};color:#fff;"
            f"padding:2px 8px;border-radius:12px;font-size:11px'>"
            f"P75: {p75_pct:+.1%} ⭐</span>"
        )
    else:
        badge = ""

    # 이미지 셀 생성
    cells = ""
    for tf in timeframes:
        url = chart_urls.get(tf, "")
        if url:
            cells += (
                f"<td style='width:33.3%;padding:2px;border:none;text-align:center;vertical-align:top'>"
                f"<div style='font-size:11px;color:#666;font-weight:bold;margin-bottom:2px'>{tf}</div>"
                f"<img src='{url}' style='width:100%;border:1px solid #ddd;border-radius:3px' alt='{tf}'>"
                f"</td>"
            )
        else:
            cells += (
                f"<td style='width:33.3%;padding:2px;border:none;text-align:center;vertical-align:top'>"
                f"<div style='font-size:11px;color:#999;margin-bottom:2px'>{tf}</div>"
                f"<div style='height:60px;background:#f5f5f5;border:1px dashed #ccc;border-radius:3px;"
                f"display:flex;align-items:center;justify-content:center;color:#ccc;font-size:11px'>N/A</div>"
                f"</td>"
            )

    return (
        f"<div style='background:{bg};border:1px solid {border};border-radius:6px;"
        f"padding:10px;margin:12px 0'>"
        f"<div style='margin-bottom:6px'>"
        f"<span style='font-size:15px;font-weight:bold;color:{title_color}'>{icon} {ticker}</span>"
        f"{badge}</div>"
        f"<table style='width:100%;border:none;border-collapse:collapse'>"
        f"<tr>{cells}</tr></table></div>"
    )


def send_paper_trading_email(
    result: dict,
    positions: list[dict],
    pdf_attachment: bytes | None = None,
    prices: dict[str, float] | None = None,
    golden_cross: list[dict] | None = None,
) -> bool:
    """페이퍼 트레이딩 매수/매도 결과를 이메일로 발송한다."""
    if not EMAIL_ENABLED:
        return False

    sells: list[dict] = result.get("sells", [])
    buys: list[dict] = result.get("buys", [])
    top5: list[dict] = result.get("selection_debug", {}).get("top5", [])
    defers: list[dict] = result.get("defers", [])
    defers_skipped: list[dict] = result.get("defers_skipped", [])
    prices = prices or {}

    # 매수/매도/재평가/후보 모두 없어도 보유 종목이 있으면 발송 (일일 현황 보고)
    if not sells and not buys and not top5 and not defers and not positions:
        print("[Email] 페이퍼 트레이딩 변동 없음, 보유 종목도 없음 — 이메일 발송 생략")
        return False

    date_str = datetime.now().strftime("%Y-%m-%d")
    if sells or buys:
        subject = f"[페이퍼 트레이딩] {date_str} 매매 알림"
    else:
        subject = f"[페이퍼 트레이딩] {date_str} 일일 리포트"

    def _price(val) -> str:
        try:
            return f"${float(val):.2f}"
        except (TypeError, ValueError):
            return str(val) if val else "-"

    def _pct(val) -> str:
        try:
            v = float(val)
            color = "#27ae60" if v >= 0 else "#e74c3c"
            return f"<span style='color:{color};font-weight:bold'>{v:+.1%}</span>"
        except (TypeError, ValueError):
            return str(val) if val else "-"

    def _vol_cell(ratio, ma20) -> str:
        """평균 거래량 대비 배수 셀 (ratio = 현재거래량 / 20일평균). ≥1배 초록↑, <1배 회색↓."""
        try:
            r = float(ratio)
        except (TypeError, ValueError):
            r = 0.0
        if not r:
            return "-"
        arrow, color = ("↑", "#27ae60") if r >= 1 else ("↓", "#999")
        try:
            m = float(ma20)
        except (TypeError, ValueError):
            m = 0.0
        avg = (f"{m/1e6:.1f}M" if m >= 1e6 else f"{m/1e3:.0f}K") if m else ""
        avg_str = f"<br><span style='color:#999;font-size:11px'>{avg} avg</span>" if avg else ""
        return f"<span style='color:{color};font-weight:bold'>{arrow} {r:.1f}×</span>{avg_str}"

    # ── 지수 출처 판별 (S&P 500 구성종목이면 SP500, 그 외는 NASDAQ/NYSE 전종목 스캔) ──
    try:
        from data.sp500_tickers import SP500_TICKERS
        _sp500_set = {str(t).strip().upper() for t in SP500_TICKERS}
    except Exception:
        _sp500_set = set()

    def _index_source(ticker) -> str:
        t = str(ticker or "").strip().upper()
        return "S&amp;P 500" if t in _sp500_set else "NASDAQ/NYSE"

    # ── 회사 건전성 / 시총 lookup (ranked.parquet에서) ──
    _health_map: dict[str, str] = {}
    _cap_map: dict[str, str] = {}
    try:
        from pathlib import Path as _Path
        import pandas as _pd
        for _p in ("data/paper_trading/sp500_ranked.parquet", "data/paper_trading/nasdaq_ranked.parquet"):
            if _Path(_p).exists():
                _df = _pd.read_parquet(_p)
                if "티커" in _df.columns:
                    if "[헬스체크]" in _df.columns:
                        for _t, _h in zip(_df["티커"], _df["[헬스체크]"]):
                            _health_map.setdefault(str(_t).upper(), str(_h))
                    if "[시총]" in _df.columns:
                        for _t, _c in zip(_df["티커"], _df["[시총]"]):
                            _cap_map.setdefault(str(_t).upper(), str(_c))
    except Exception:
        pass

    def _health(ticker) -> str:
        """헬스체크 한 줄 문자열 (lookup)."""
        return _health_map.get(str(ticker or "").upper(), "N/A")

    def _health_cell(ticker) -> str:
        """이메일용 두 줄 HTML 셀: 등급(굵게) + 성장지표(작은 회색)."""
        raw = _health(ticker)
        if " · " not in raw:
            return raw  # "N/A" 또는 fallback
        grade_part, growth_part = raw.split(" · ", 1)
        # 등급별 색상
        color = "#666"
        if "A" in grade_part:
            color = "#1e8449"
        elif "F" in grade_part or "D" in grade_part:
            color = "#c0392b"
        return (
            f"<b style='color:{color}'>{grade_part}</b>"
            f"<br><span style='font-size:11px;color:#888'>{growth_part}</span>"
        )

    def _cap(ticker) -> str:
        return _cap_map.get(str(ticker or "").upper(), "?")

    # ── 매도 테이블 ──
    sell_html = ""
    if sells:
        sell_html = """
<h3 style='color:#e74c3c'>🔴 매도 ({n}건)</h3>
<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:14px'>
<tr style='background:#fdecea;text-align:center'>
  <th>종목</th><th>섹터/지수</th><th>[시총]</th><th>[헬스체크]</th><th>매수가</th><th>매도가</th>
  <th>수익률</th><th>보유일</th><th>전략</th><th>사유</th><th>내부자(90일)</th>
</tr>
""".format(n=len(sells))
        for t in sells:
            ret = t.get("return_pct", 0)
            insider_str = t.get("insider") or t.get("market_analysis", {}).get("insider", "—")
            sell_html += (
                f"<tr style='text-align:center'>"
                f"<td><b>{t.get('ticker','')}</b></td>"
                f"<td>{t.get('sector','')}<br><span style='color:#999;font-size:11px'>{_index_source(t.get('ticker',''))}</span></td>"
                f"<td>{_cap(t.get('ticker',''))}</td>"
                f"<td>{_health_cell(t.get('ticker',''))}</td>"
                f"<td>{_price(t.get('entry_price'))}</td>"
                f"<td>{_price(t.get('exit_price'))}</td>"
                f"<td>{_pct(ret)}</td>"
                f"<td>{t.get('holding_days', '-')}일</td>"
                f"<td>{t.get('strategy','')}</td>"
                f"<td>{t.get('exit_reason', t.get('reason',''))}</td>"
                f"<td style='font-size:12px'>{insider_str}</td>"
                f"</tr>\n"
            )
        sell_html += "</table><br>"

    # ── 매수 테이블 ──
    buy_html = ""
    if buys:
        buy_html = """
<h3 style='color:#27ae60'>🟢 매수 ({n}건)</h3>
<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:14px'>
<tr style='background:#eafaf1;text-align:center'>
  <th>종목</th><th>섹터/지수</th><th>[시총]</th><th>[헬스체크]</th><th>매수가</th>
  <th>전략</th><th>별점</th><th>CCS점수</th><th>거래량</th>
</tr>
""".format(n=len(buys))
        for t in buys:
            buy_html += (
                f"<tr style='text-align:center'>"
                f"<td><b>{t.get('ticker','')}</b></td>"
                f"<td>{t.get('sector','')}<br><span style='color:#999;font-size:11px'>{_index_source(t.get('ticker',''))}</span></td>"
                f"<td>{_cap(t.get('ticker',''))}</td>"
                f"<td>{_health_cell(t.get('ticker',''))}</td>"
                f"<td>{_price(t.get('entry_price'))}</td>"
                f"<td>{t.get('strategy','')}</td>"
                f"<td>{t.get('star_rating','')}</td>"
                f"<td>{t.get('ccs_score','')}</td>"
                f"<td>{_vol_cell(t.get('vol_ratio'), t.get('vol_ma20'))}</td>"
                f"</tr>\n"
            )
        buy_html += "</table>\n"

        # 각 매수 종목별 상세 박스 (고RSI 근거 + 상승 분포)
        for t in buys:
            ticker = t.get("ticker", "")
            detail_parts: list[str] = []
            if t.get("high_rsi_flag") and t.get("high_rsi_rationale"):
                entry_rsi = t.get("entry_rsi", 0)
                detail_parts.append(
                    f"<div style='background:#fff3e0;border-left:4px solid #f39c12;"
                    f"padding:10px;margin:6px 0;font-size:13px'>"
                    f"<b>📈 {ticker} 매수 근거 (RSI {entry_rsi:.0f} — 모멘텀 허용)</b><br>"
                    f"{t['high_rsi_rationale']}"
                    f"</div>"
                )
            if t.get("upside_n", 0) >= 10:
                entry_price = t.get("entry_price", 0)
                p25 = t.get("upside_p25", 0)
                p50 = t.get("upside_p50", 0)
                p75 = t.get("upside_p75", 0)
                p90 = t.get("upside_p90", 0)
                price_p25 = t.get("upside_price_p25", 0)
                price_p50 = t.get("upside_price_p50", 0)
                price_p75 = t.get("upside_price_p75", 0)
                price_p90 = t.get("upside_price_p90", 0)
                days = t.get("upside_days_to_peak", 0)
                n = t.get("upside_n", 0)
                detail_parts.append(
                    f"<div style='background:#e8f5e9;border-left:4px solid #27ae60;"
                    f"padding:10px;margin:6px 0;font-size:13px'>"
                    f"<b>📊 {ticker} 예상 상승 분포 (유사 거래 {n}건 기반)</b><br>"
                    f"매수가 ${entry_price:.2f} 기준 예상 목표가:<br>"
                    f"&nbsp;&nbsp;보수적 (P25): <b>{p25:+.1%}</b> → ${price_p25:.2f}<br>"
                    f"&nbsp;&nbsp;중간값 (P50): <b>{p50:+.1%}</b> → ${price_p50:.2f}<br>"
                    f"&nbsp;&nbsp;낙관 (P75): <b>{p75:+.1%}</b> → ${price_p75:.2f} ⭐<br>"
                    f"&nbsp;&nbsp;홈런 (P90): <b>{p90:+.1%}</b> → ${price_p90:.2f}<br>"
                    f"평균 최고점 도달: {days:.0f}일차"
                    f"</div>"
                )
            elif 0 < t.get("upside_n", 0) < 10:
                detail_parts.append(
                    f"<div style='background:#f5f5f5;border-left:4px solid #95a5a6;"
                    f"padding:8px;margin:6px 0;font-size:12px;color:#666'>"
                    f"⚠️ {ticker} 예상 분포 샘플 부족 (N={t.get('upside_n')})"
                    f"</div>"
                )
            if detail_parts:
                buy_html += "".join(detail_parts)
            # 차트 이미지 (매수 종목)
            buy_p75 = t.get("upside_p75", 0)
            buy_entry = t.get("entry_price", 0)
            buy_p75_price = buy_entry * (1 + buy_p75) if buy_entry and buy_p75 else 0
            buy_html += _build_chart_images_html(
                ticker, p75_pct=buy_p75, p75_price=buy_p75_price, color_theme="buys"
            )
        buy_html += _build_market_analysis_card(buys)
        buy_html += "<br>"

    # ── 현재 보유 현황 (현재가/수익률/재평가 뱃지 + upside 목표가 포함) ──
    holdings_html = ""
    if positions:
        from datetime import date
        holdings_html = """
<h3 style='color:#2980b9'>📊 현재 보유 현황 ({n}종목)</h3>
<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:14px'>
<tr style='background:#ebf5fb;text-align:center'>
  <th>종목</th><th>섹터/지수</th><th>[시총]</th><th>[헬스체크]</th><th>매수일</th><th>매수가</th><th>현재가</th>
  <th>수익률</th><th>보유일</th><th>전략</th><th>Target</th><th>Next Target</th><th>Trail Stop</th><th>거래량</th><th>내부자(90일)</th>
</tr>
""".format(n=len(positions))
        from paper_trading.engine import _resolve_exit_params
        for p in positions:
            entry_date = p.get("entry_date", "")
            try:
                from datetime import datetime as _dt
                days = (date.today() - _dt.strptime(str(entry_date), "%Y-%m-%d").date()).days
            except Exception:
                days = "-"
            ticker = p.get("ticker", "")
            entry_price = p.get("entry_price", 0)
            current = prices.get(ticker)
            if current is None or not entry_price:
                current_str = "-"
                ret_str = "-"
            else:
                current_str = _price(current)
                ret_str = _pct((current - entry_price) / entry_price)
            defer_count = int(p.get("defer_count", 0))
            if defer_count > 0:
                status_parts = [f"🔄 재평가중 ({defer_count}/{HOLD_WINNERS_MAX_DEFERS})"]
                # P3: freeze 남은 일수
                last_defer = p.get("last_defer_date")
                if last_defer:
                    try:
                        from datetime import datetime as _dt
                        defer_dt = _dt.strptime(str(last_defer), "%Y-%m-%d").date()
                        freeze_remaining = max(0, HOLD_WINNERS_DEFER_FREEZE_DAYS - (date.today() - defer_dt).days)
                        if freeze_remaining > 0:
                            status_parts.append(f"🧊 freeze {freeze_remaining}일")
                    except Exception:
                        pass
                # P3: trailing stop 여유
                highest = p.get("highest_price", 0)
                trail_override = p.get("trailing_stop_override")
                if current and highest and highest > 0 and trail_override:
                    drawdown = (current - highest) / highest
                    room = trail_override + drawdown
                    status_parts.append(f"📉 고점 {drawdown:+.1%} (여유 {room:.1%})")
                # P4: defer 발동 시점 가격 + 이후 수익률
                defer_dbg = p.get("defer_debug") or {}
                defer_price = defer_dbg.get("current_price")
                if defer_price and current and defer_price > 0:
                    since_defer_ret = (current - defer_price) / defer_price
                    defer_date_short = str(last_defer)[5:] if last_defer else ""
                    status_parts.append(f"📌 발동: {defer_date_short} @ {_price(defer_price)} → {since_defer_ret:+.1%}")
                status = (
                    "<br>".join(
                        f"<span style='background:#fff3e0;color:#d35400;"
                        f"padding:2px 6px;border-radius:3px;font-weight:bold;display:inline-block;"
                        f"margin:1px 0;font-size:12px'>{part}</span>"
                        for part in status_parts
                    )
                )
            else:
                status = ""
            # 매수 당시 저장된 upside 예측값 표시
            p50 = p.get("predicted_upside_p50", 0)
            p75 = p.get("predicted_upside_p75", 0)
            if entry_price and p50:
                p50_price = entry_price * (1 + p50)
                p50_str = f"<b>{p50:+.1%}</b> → {_price(p50_price)}"
            else:
                p50_str = "-"
            if entry_price and p75:
                p75_price = entry_price * (1 + p75)
                p75_str = f"<b>{p75:+.1%}</b> → {_price(p75_price)}"
            else:
                p75_str = "-"
            # Trail Stop: 엔진 청산 로직(check_sell_conditions)과 동일하게 계산 — 표시 전용.
            # 고점이 매수가를 넘은 적 있으면 트레일링, 아니면 하드손절이 실제 보호선.
            exit_p = _resolve_exit_params(p.get("strategy", ""))
            highest = p.get("highest_price", entry_price) or entry_price
            if entry_price and highest > entry_price:
                trail = p.get("trailing_stop_override") or exit_p["trailing_stop"]
                stop_price, trail_tag = highest * (1 - trail), "트레일"
            elif entry_price:
                stop_price, trail_tag = entry_price * (1 - exit_p["stop_loss"]), "손절"
            else:
                stop_price, trail_tag = 0, ""
            if stop_price:
                trail_str = (
                    f"<span style='color:#e67e22;font-weight:bold'>{_price(stop_price)}</span>"
                    f"<br><span style='color:#999;font-size:11px'>{trail_tag}</span>"
                )
            else:
                trail_str = "-"
            vol_str = _vol_cell(p.get("vol_ratio"), p.get("vol_ma20"))
            insider_str = p.get("market_analysis", {}).get("insider") or p.get("insider", "—")
            holdings_html += (
                f"<tr style='text-align:center'>"
                f"<td><b>{ticker}</b></td>"
                f"<td>{p.get('sector','')}<br><span style='color:#999;font-size:11px'>{_index_source(ticker)}</span></td>"
                f"<td>{_cap(ticker)}</td>"
                f"<td>{_health_cell(ticker)}</td>"
                f"<td>{entry_date}</td>"
                f"<td>{_price(entry_price)}</td>"
                f"<td>{current_str}</td>"
                f"<td>{ret_str}</td>"
                f"<td>{days}일</td>"
                f"<td>{p.get('strategy','')}</td>"
                f"<td>{p50_str}</td>"
                f"<td>{p75_str}</td>"
                f"<td>{trail_str}</td>"
                f"<td>{vol_str}</td>"
                f"<td style='font-size:12px'>{insider_str}</td>"
                f"</tr>\n"
            )
        holdings_html += "</table>\n"
        for p in positions:
            ticker = p.get("ticker", "")
            entry_price = p.get("entry_price", 0)
            p75_val = p.get("predicted_upside_p75", 0)
            p75_price_val = entry_price * (1 + p75_val) if entry_price and p75_val else 0
            holdings_html += _build_chart_images_html(
                ticker, p75_pct=p75_val, p75_price=p75_price_val, color_theme="holdings"
            )
        holdings_html += _build_market_analysis_card(positions)
        holdings_html += "<br>"

    # ── 재평가 체크 라인 생성 (P1: 통과/실패 모두 표시) ──
    def _render_check_lines(dbg: dict, checks: dict) -> tuple[list[str], int]:
        """5개 체크 전부 ✅/❌로 렌더링. (lines, passed_count) 반환."""
        rsi = dbg.get("RSI", 0)
        adx = dbg.get("adx", 0)
        vol = dbg.get("vol_mult", 0)
        bb = dbg.get("bb_pband", 0)
        ret_5d = dbg.get("ret_5d", 0)
        check_defs = [
            ("rsi_not_overbought", f"RSI {rsi:.0f}", "과매수 아님 — 상승 여지 있음", f"RSI {rsi:.0f} ≥ 75 — 과매수"),
            ("volume_strong", f"거래량 {vol:.1f}x", "거래량 돌파", "거래량 부족"),
            ("ret_5d_positive", f"5일 수익률 {ret_5d:+.1%}", "양(+) 수익", "음(-) 수익"),
            ("adx_trending", f"ADX {adx:.0f}", "추세 유지", "추세 약함"),
            ("bb_not_peaked", f"볼린저 pband {bb:.2f}", "peak 아님", "상단 도달"),
        ]
        lines: list[str] = []
        passed = 0
        for key, val_str, pass_desc, fail_desc in check_defs:
            ok = checks.get(key, False) or (key == "rsi_not_overbought" and checks.get("rsi_in_range", False))
            if ok:
                passed += 1
                lines.append(
                    f"<span style='color:#27ae60'>✅ {val_str} ({pass_desc})</span>"
                )
            else:
                lines.append(
                    f"<span style='color:#e74c3c'>❌ {val_str} ({fail_desc})</span>"
                )
        return lines, passed

    # ── 재평가 상세 (오늘 defer 발동한 경우) ──
    defer_html = ""
    if defers:
        defer_blocks: list[str] = []
        for d in defers:
            ticker = d.get("ticker", "")
            ret = d.get("current_return", 0)
            dc = d.get("defer_count", 1)
            dbg = d.get("debug", {})
            checks = dbg.get("checks", {})
            check_lines, passed = _render_check_lines(dbg, checks)
            defer_blocks.append(
                f"<div style='background:#fef5e7;border-left:4px solid #e67e22;"
                f"padding:10px;margin:6px 0;font-size:13px'>"
                f"<b>📋 재평가 사유 ({ticker}, defer {dc}/{HOLD_WINNERS_MAX_DEFERS}, 현재 {ret:+.1%})</b><br>"
                + "<br>".join(check_lines)
                + f"<br><b>→ {passed}/5 통과 (기준: {HOLD_WINNERS_MIN_CHECKS}개) → defer 발동, tight trailing(-3.5%)로 전환</b>"
                + "</div>"
            )
        defer_html = (
            "<h3 style='color:#e67e22'>🔄 재평가 (목표가 도달했으나 보유 연장, "
            f"{len(defers)}건)</h3>"
            + "".join(defer_blocks) + "<br>"
        )

    # ── 재평가 실패 상세 (P2: defer 시도했으나 조건 미달 → 매도) ──
    defer_skip_html = ""
    if defers_skipped:
        skip_blocks: list[str] = []
        for d in defers_skipped:
            ticker = d.get("ticker", "")
            ret = d.get("current_return", 0)
            reason = d.get("trigger_reason", "")
            dbg = d.get("debug", {})
            checks = dbg.get("checks", {})
            check_lines, passed = _render_check_lines(dbg, checks)
            skip_blocks.append(
                f"<div style='background:#fdecea;border-left:4px solid #e74c3c;"
                f"padding:10px;margin:6px 0;font-size:13px'>"
                f"<b>📋 {ticker} — {reason} 도달, 재평가 시도</b><br>"
                + "<br>".join(check_lines)
                + f"<br><b style='color:#e74c3c'>→ {passed}/5 통과 (기준: {HOLD_WINNERS_MIN_CHECKS}개) → 재평가 실패, 매도 집행</b>"
                + "</div>"
            )
        defer_skip_html = (
            "<h3 style='color:#c0392b'>⛔ 재평가 실패 → 매도 ("
            f"{len(defers_skipped)}건)</h3>"
            + "".join(skip_blocks) + "<br>"
        )

    # ── 후보 테이블 (upside 예측 포함) ──
    candidates_html = ""
    if top5:
        # 후보별 upside 예측
        try:
            from paper_trading.upside_model import predict_upside as _predict_upside
            top5_upside = {}
            for c in top5:
                try:
                    up = _predict_upside(c.get("strategy", ""), c.get("ccs", 0.0), c.get("star", ""))
                    top5_upside[c.get("ticker", "")] = up
                except Exception:
                    top5_upside[c.get("ticker", "")] = {"n": 0}
        except Exception:
            top5_upside = {}

        _cand_header = (
            "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:14px'>\n"
            "<tr style='background:#f5eef8;text-align:center'>\n"
            "  <th>종목</th><th>섹터/지수</th><th>[시총]</th><th>[헬스체크]</th><th>전략</th><th>별점</th><th>CCS점수</th><th>거래량</th><th>Target</th><th>Next Target</th><th>내부자(90일)</th>\n"
            "</tr>\n"
        )
        candidates_html = f"<h3 style='color:#8e44ad'>🔍 오늘의 후보 ({len(top5)}종목)</h3>\n"
        candidates_html += _cand_header

        for c in top5:
            ticker_c = c.get("ticker", "")
            up = top5_upside.get(ticker_c, {})
            p50_c = up.get("p50", 0)
            p75_c = up.get("p75", 0)
            n_c = up.get("n", 0)
            cur_price = c.get("current_price") or 0
            if n_c >= 5 and p50_c:
                if cur_price:
                    p50_c_str = f"<b>{p50_c:+.1%}</b> → {_price(cur_price * (1 + p50_c))}"
                else:
                    p50_c_str = f"<b>{p50_c:+.1%}</b>"
            else:
                p50_c_str = "-"
            if n_c >= 5 and p75_c:
                if cur_price:
                    p75_c_str = f"<b>{p75_c:+.1%}</b> → {_price(cur_price * (1 + p75_c))}"
                else:
                    p75_c_str = f"<b>{p75_c:+.1%}</b>"
            else:
                p75_c_str = "-"
            insider_c = c.get("market_analysis", {}).get("insider") or c.get("insider", "—")

            row_html = (
                f"<tr style='text-align:center'>"
                f"<td><b>{ticker_c}</b></td>"
                f"<td>{c.get('sector','')}<br><span style='color:#999;font-size:11px'>{_index_source(ticker_c)}</span></td>"
                f"<td>{_cap(ticker_c)}</td>"
                f"<td>{_health_cell(ticker_c)}</td>"
                f"<td>{c.get('strategy','')}</td>"
                f"<td>{c.get('star','')}</td>"
                f"<td>{c.get('ccs', 0):.4f}</td>"
                f"<td>{_vol_cell(c.get('vol_ratio'), c.get('vol_ma20'))}</td>"
                f"<td>{p50_c_str}</td>"
                f"<td>{p75_c_str}</td>"
                f"<td style='font-size:12px'>{insider_c}</td>"
                f"</tr>\n"
            )
            candidates_html += row_html

        candidates_html += "</table>\n"
        
        for c in top5:
            ticker_c = c.get("ticker", "")
            up = top5_upside.get(ticker_c, {})
            cand_p75 = up.get("p75", 0)
            candidates_html += _build_chart_images_html(
                ticker_c, p75_pct=cand_p75, color_theme="candidates"
            )
        candidates_html += _build_market_analysis_card(top5)
        candidates_html += "<br>"

    # ── 예측 정확도 섹션 (백테스트 검증 + 실전 검증) ──
    accuracy_html = ""
    try:
        # --- 백테스트 교차검증 결과 ---
        bt_html = ""
        _bt_path = Path(__file__).resolve().parents[2] / "data" / "paper_trading" / "backtest_validation.json"
        if _bt_path.exists():
            import json as _json
            bt_data = _json.loads(_bt_path.read_text(encoding="utf-8"))
            bt_s = bt_data.get("summary", {})
            bt_n = bt_s.get("n_trades", 0)
            if bt_n >= 3:
                bt_html = f"""
<h3 style='color:#555'>📐 예측 정확도 — 백테스트 검증 ({bt_n}건, LOO 교차검증)</h3>
<table border='1' style='border-collapse:collapse;width:60%;font-size:14px'>
<tr style='background:#f0f0f0;text-align:center'>
  <th>지표</th><th>결과</th>
</tr>
<tr style='text-align:center'><td>P50 달성률</td><td><b>{bt_s.get('p50_hits',0)}/{bt_n} ({bt_s.get('p50_hit_rate',0):.0%})</b></td></tr>
<tr style='text-align:center'><td>P75 달성률</td><td><b>{bt_s.get('p75_hits',0)}/{bt_n} ({bt_s.get('p75_hit_rate',0):.0%})</b></td></tr>
<tr style='text-align:center'><td>평균 예측 (P50)</td><td>{bt_s.get('avg_predicted_p50',0):+.1%}</td></tr>
<tr style='text-align:center'><td>평균 실제 최대 상승</td><td>{bt_s.get('avg_actual_max_upside',0):+.1%}</td></tr>
</table>
<div style='color:#999;font-size:11px;margin-top:4px'>* 각 거래를 제외한 나머지로 분포 재계산 후 예측 (in-sample bias 제거)</div>
<br>
"""

        # --- 실전 페이퍼 트레이딩 검증 ---
        paper_html = ""
        from paper_trading.portfolio import load_trades as _load_trades
        closed_trades = _load_trades()
        validated = [
            t for t in closed_trades
            if t.get("predicted_upside_p50", 0) > 0 and t.get("hit_p50", 0) != 0
        ]
        n_v = len(validated)
        if n_v >= 3:
            p50_hits = sum(1 for t in validated if t.get("hit_p50") == 1)
            p75_hits = sum(1 for t in validated if t.get("hit_p75") == 1)
            avg_pred = sum(t.get("predicted_upside_p50", 0) for t in validated) / n_v
            avg_actual = sum(t.get("actual_max_upside_pct", 0) for t in validated) / n_v
            paper_html = f"""
<h3 style='color:#2980b9'>📐 예측 정확도 — 실전 페이퍼 ({n_v}건)</h3>
<table border='1' style='border-collapse:collapse;width:60%;font-size:14px'>
<tr style='background:#ebf5fb;text-align:center'>
  <th>지표</th><th>결과</th>
</tr>
<tr style='text-align:center'><td>P50 달성률</td><td><b>{p50_hits}/{n_v} ({p50_hits/n_v:.0%})</b></td></tr>
<tr style='text-align:center'><td>P75 달성률</td><td><b>{p75_hits}/{n_v} ({p75_hits/n_v:.0%})</b></td></tr>
<tr style='text-align:center'><td>평균 예측 (P50)</td><td>{avg_pred:+.1%}</td></tr>
<tr style='text-align:center'><td>평균 실제 최대 상승</td><td>{avg_actual:+.1%}</td></tr>
</table><br>
"""

        # --- ML 마일스톤 알림 (실전 기준) ---
        milestone_html = ""
        _ML_MILESTONES = {30: "통계 분석 가능", 100: "ML 학습 시작 가능", 300: "안정적 ML 모델 구축 가능"}
        for threshold, desc in _ML_MILESTONES.items():
            if n_v >= threshold and (n_v - 1) < threshold:
                milestone_html = (
                    f"<div style='background:#e8f5e9;border:2px solid #27ae60;"
                    f"padding:12px;margin:8px 0;font-size:14px;border-radius:6px'>"
                    f"🎯 <b>ML 마일스톤 달성: 실전 {n_v}건 — {desc}</b>"
                    f"</div>"
                )
                break
        if not milestone_html:
            next_target = next((t for t in sorted(_ML_MILESTONES) if n_v < t), None)
            if next_target:
                remaining = next_target - n_v
                milestone_html = (
                    f"<div style='background:#f5f5f5;padding:8px;margin:6px 0;"
                    f"font-size:12px;color:#666;border-radius:4px'>"
                    f"📊 실전 ML 데이터 수집 중: {n_v}/{next_target}건 "
                    f"({_ML_MILESTONES[next_target]}, 약 {remaining}건 남음)"
                    f"</div>"
                )

        if bt_html or paper_html:
            accuracy_html = bt_html + paper_html + milestone_html
    except Exception:
        pass

    # ── 골든크로스 임박 섹션 (테이블 + 차트만, 뉴스 없음) ──
    # 정렬은 runner._extract_golden_cross_imminent()에서 다중 TF/일봉/conf 순으로 처리됨.
    golden_cross_html = ""
    if golden_cross:
        golden_cross_html = (
            f"<h3 style='color:#b8860b'>⚡ 골든크로스 임박/직후 ({len(golden_cross)}종목)</h3>\n"
            "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:14px'>\n"
            "<tr style='background:#fdf6e3;text-align:center'>\n"
            "  <th>종목</th><th>섹터/지수</th><th>[시총]</th><th>[헬스체크]</th><th>현재가</th><th>임박 타임프레임</th>"
            "<th>전략</th><th>별점</th><th>CCS점수</th><th>거래량</th><th>내부자(90일)</th>\n"
            "</tr>\n"
        )
        for g in golden_cross:
            ticker_g = g.get("ticker", "")
            tf_gaps = g.get("tf_gaps") or []
            if tf_gaps:
                # 부호로 임박/직후 구분: 음수=임박 (단기<장기), 양수=직후 (방금 교차)
                parts = []
                for label, gap in tf_gaps:
                    sign = "+" if gap >= 0 else "-"
                    parts.append(f"{label}({sign}{abs(gap)*100:.2f}%)")
                tfs = ", ".join(parts)
            else:
                tfs = ", ".join(g.get("timeframes", []))
            # 다중 TF confirm은 ★ 표시 (2개 이상 타임프레임 임박)
            star_prefix = "★ " if g.get("tf_count", 0) >= 2 else ""
            insider_g = g.get("insider") or g.get("market_analysis", {}).get("insider", "—")
            ccs_g = g.get("ccs")
            ccs_str = f"{ccs_g:.4f}" if isinstance(ccs_g, (int, float)) else "-"
            row_bg = "background:#fff8dc;" if g.get("tf_count", 0) >= 2 else ""
            cur_price_g = g.get("current_price") or 0
            cur_price_g_str = _price(cur_price_g) if cur_price_g else "-"
            golden_cross_html += (
                f"<tr style='text-align:center;{row_bg}'>"
                f"<td><b>{star_prefix}{ticker_g}</b></td>"
                f"<td>{g.get('sector','')}<br><span style='color:#999;font-size:11px'>{_index_source(ticker_g)}</span></td>"
                f"<td>{_cap(ticker_g)}</td>"
                f"<td>{_health_cell(ticker_g)}</td>"
                f"<td>{cur_price_g_str}</td>"
                f"<td>{tfs}</td>"
                f"<td>{g.get('strategy','')}</td>"
                f"<td>{g.get('star','')}</td>"
                f"<td>{ccs_str}</td>"
                f"<td>{_vol_cell(g.get('vol_ratio'), g.get('vol_ma20'))}</td>"
                f"<td style='font-size:12px'>{insider_g}</td>"
                f"</tr>\n"
            )
        golden_cross_html += "</table>\n"
        for g in golden_cross:
            golden_cross_html += _build_chart_images_html(
                g.get("ticker", ""), color_theme="golden"
            )
        golden_cross_html += "<br>"

    body = f"""
<h2 style='font-family:sans-serif'>📈 페이퍼 트레이딩 일일 보고 — {date_str}</h2>
{sell_html}
{buy_html}
{defer_html}
{defer_skip_html}
{holdings_html}
{candidates_html}
{accuracy_html}
{golden_cross_html}
<p style='color:#999;font-size:12px'>본 메일은 시스템에 의해 자동으로 발송되었습니다.</p>
"""

    try:
        recipients_str = ", ".join(EMAIL_RECIPIENTS)
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_SENDER  # 발신자만 표시 (수신자 목록 숨김)
        msg["Bcc"] = recipients_str  # 실제 수신자는 BCC 처리
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        if pdf_attachment:
            part = MIMEBase("application", "pdf")
            part.set_payload(pdf_attachment)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="trading_report_{date_str}.pdf"',
            )
            msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        pdf_note = " (PDF 첨부)" if pdf_attachment else ""
        print(
            f"[Email] 페이퍼 트레이딩 알림 발송 완료{pdf_note} "
            f"(매도 {len(sells)}건, 매수 {len(buys)}건, 재평가 {len(defers)}건, 재평가실패 {len(defers_skipped)}건, 후보 {len(top5)}건) "
            f"→ {recipients_str}"
        )
        return True
    except Exception as exc:
        print(f"[Email] 페이퍼 트레이딩 이메일 발송 실패: {exc}")
        return False
