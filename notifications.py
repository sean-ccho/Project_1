"""SQZ/RSI 필터링 및 이메일 알림 유틸리티."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Dict, List, Sequence

import pandas as pd

from config import (
    EMAIL_NOTIFICATIONS_ENABLED,
    EMAIL_PASSWORD,
    EMAIL_RECIPIENTS,
    EMAIL_SENDER,
    EMAIL_SMTP_PORT,
    EMAIL_SMTP_SERVER,
    EMAIL_USERNAME,
    EMAIL_USE_TLS,
)

SQZ_COLUMNS: Sequence[str] = ["SQZ_Off(1H)", "SQZ_Off(1D)", "SQZ_Off(1W)", "SQZ_Off(1M)"]
RSI_COLUMNS: Sequence[str] = ["RSI(1H)", "RSI(1D)", "RSI(1W)", "RSI(1M)"]
SQZ_LABELS: Dict[str, str] = {
    "SQZ_Off(1H)": "1H",
    "SQZ_Off(1D)": "1D",
    "SQZ_Off(1W)": "1W",
    "SQZ_Off(1M)": "1M",
}
SQZ_DIRECTION_COLUMNS: Dict[str, str] = {
    "SQZ_Off(1H)": "SQZ_Off Dir(1H)",
    "SQZ_Off(1D)": "SQZ_Off Dir(1D)",
    "SQZ_Off(1W)": "SQZ_Off Dir(1W)",
    "SQZ_Off(1M)": "SQZ_Off Dir(1M)",
}
SQZ_QUAL_COLUMNS: Dict[str, str] = {
    "SQZ_Off(1H)": "SQZ_Off Qual(1H)",
    "SQZ_Off(1D)": "SQZ_Off Qual(1D)",
    "SQZ_Off(1W)": "SQZ_Off Qual(1W)",
    "SQZ_Off(1M)": "SQZ_Off Qual(1M)",
}


def notify_signal_summary(df: pd.DataFrame, context_label: str) -> None:
    """필터링 결과를 이메일로 전송한다."""

    if not EMAIL_NOTIFICATIONS_ENABLED:
        return
    if EMAIL_SMTP_SERVER is None or not EMAIL_RECIPIENTS or EMAIL_SENDER is None:
        print("[Email] 구성값이 없어 알림을 건너뜁니다.")
        return

    summary = build_summary_lines(df)
    if not summary:
        print("[Email] 전송할 SQZ/RSI 결과가 없습니다.")
        return

    subject = f"[{context_label}] SQZ/RSI Alerts"
    body = "\n".join(summary)
    if send_email(subject, body):
        print("[Email] SQZ/RSI 알림 이메일 전송 완료.")


def build_summary_lines(df: pd.DataFrame) -> List[str]:
    """요약 문자열 목록을 생성한다."""

    tickers_column = "티커"
    if tickers_column not in df.columns:
        return []

    sqz_matches = _sqz_true_timeframes(df, tickers_column)
    oversold, overbought = _collect_rsi_extremes(df, tickers_column)

    if not sqz_matches and not _has_results(oversold) and not _has_results(overbought):
        return []

    lines: List[str] = []
    lines.extend(_format_sqz_section(sqz_matches))
    lines.extend(_format_rsi_sections(oversold, overbought))

    return lines


def _format_sqz_section(sqz_matches: Dict[str, List[str]]) -> List[str]:
    lines = ["SQZ Off (Up candles only; timeframes shown):"]
    if sqz_matches:
        for ticker, periods in sqz_matches.items():
            label = ", ".join(periods) if periods else "N/A"
            lines.append(f"- {ticker} ({label})")
    else:
        lines.append("- 없음")
    lines.append("")
    return lines


def _format_rsi_sections(
    oversold: Dict[str, List[str]],
    overbought: Dict[str, List[str]],
) -> List[str]:
    lines: List[str] = []

    lines.append("RSI Oversold (<=30):")
    lines.extend(_format_rsi_block(oversold))
    lines.append("")

    lines.append("RSI Overbought (>=70):")
    lines.extend(_format_rsi_block(overbought))
    lines.append("")

    return lines


def _format_rsi_block(mapping: Dict[str, List[str]]) -> List[str]:
    if not mapping:
        return ["- 없음"]
    lines: List[str] = []
    for column in RSI_COLUMNS:
        tickers = mapping.get(column, [])
        label = column
        if tickers:
            lines.append(f"- {label}: {', '.join(tickers)}")
        else:
            lines.append(f"- {label}: 없음")
    return lines


def _collect_rsi_extremes(df: pd.DataFrame, ticker_col: str) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    oversold: Dict[str, List[str]] = {}
    overbought: Dict[str, List[str]] = {}

    for column in RSI_COLUMNS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        tickers = df[ticker_col].astype(str)

        mask_oversold = series <= 30
        if mask_oversold.any():
            oversold[column] = tickers[mask_oversold].tolist()

        mask_overbought = series >= 70
        if mask_overbought.any():
            overbought[column] = tickers[mask_overbought].tolist()

    return oversold, overbought


def _sqz_true_timeframes(df: pd.DataFrame, ticker_col: str) -> Dict[str, List[str]]:
    available = [col for col in SQZ_COLUMNS if col in df.columns]
    if not available:
        return {}

    sqz_frame = df[available].apply(lambda col: col.map(_coerce_bool)).fillna(False)
    results: Dict[str, List[str]] = {}
    for idx, row in sqz_frame.iterrows():
        periods: List[str] = []
        for col in available:
            if not row[col]:
                continue
            direction_col = SQZ_DIRECTION_COLUMNS.get(col)
            if not direction_col or direction_col not in df.columns:
                continue
            direction_value = str(df.at[idx, direction_col]).strip().lower()
            if direction_value != "up":
                continue
            qual_col = SQZ_QUAL_COLUMNS.get(col)
            if not qual_col or qual_col not in df.columns:
                continue
            if not _coerce_bool(df.at[idx, qual_col]):
                continue
            periods.append(SQZ_LABELS.get(col, col))
        if periods:
            ticker = str(df.at[idx, ticker_col])
            results[ticker] = periods
    return results


def _has_results(mapping: Dict[str, List[str]]) -> bool:
    return any(mapping.get(column) for column in RSI_COLUMNS)


def send_email(subject: str, body: str) -> bool:
    """SMTP를 이용해 이메일을 전송한다."""

    try:
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, timeout=15) as server:
            if EMAIL_USE_TLS:
                server.starttls()
            if EMAIL_USERNAME and EMAIL_PASSWORD:
                server.login(EMAIL_USERNAME, EMAIL_PASSWORD)

            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = EMAIL_SENDER
            message["To"] = ", ".join(EMAIL_RECIPIENTS)
            message.set_content(body)
            server.send_message(message)
        return True
    except Exception as exc:
        print(f"[Email] 알림 전송 실패: {exc}")
        return False


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"true", "t", "1", "y", "yes"}
    try:
        return bool(value)
    except Exception:
        return False
