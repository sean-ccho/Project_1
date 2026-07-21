#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""내계좌 포트폴리오 일일 기술 지표 리포트.

Google Sheets '내계좌' 워크시트에 있는 종목들을 읽어
1H / 4H / Daily / Weekly / Monthly 타임프레임별 RSI·MACD·거래량 지표를 계산하고:
  1) '내계좌' 시트 분석 컬럼을 업데이트한다.
  2) chunghwan14@gmail.com 으로 개인 전용 HTML 이메일을 발송한다.

시트 컬럼 구조 (총 24컬럼):
  [입력 — 사용자 직접 작성] 티커, 회사명, 거래소, 섹터, 매수가, 수량, 매수일, 메모
  [분석 — 파이프라인 자동] 1H RSI, 1H MACD, 1H 거래량, 4H RSI, 4H MACD, 4H 거래량,
                            Daily RSI, Daily MACD, Daily 거래량,
                            Weekly RSI, Weekly MACD, Weekly 거래량,
                            Monthly RSI, Monthly MACD, Monthly 거래량,
                            업데이트

사용:
    python src/screener/portfolio_report.py
"""

from __future__ import annotations

import json
import os
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD as TAmacd
except ImportError:
    RSIIndicator = None
    TAmacd = None

try:
    import gspread
    from google.oauth2.service_account import Credentials
    from gspread.exceptions import WorksheetNotFound
except ImportError:
    gspread = None
    Credentials = None
    WorksheetNotFound = None

from screener.config import (
    GOOGLE_SHEETS_CREDENTIALS_PATH,
    GOOGLE_SHEETS_ENABLED,
    GOOGLE_SHEETS_SPREADSHEET_ID,
    MY_ACCOUNT_WORKSHEET,
    MY_ACCOUNT_TICKER_COLUMN,
    MY_ACCOUNT_EMAIL,
    MY_ACCOUNT_REPORT_ENABLED,
    EMAIL_SENDER,
    EMAIL_PASSWORD,
    SMTP_SERVER,
    SMTP_PORT,
    TRADINGVIEW_TICKER_MAP,
)

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ---------------------------------------------------------------------------
# 시트 컬럼 정의
# ---------------------------------------------------------------------------

# ── 입력 컬럼 (사용자가 직접 채우는 부분) ──────────────────────────────────
INPUT_HEADERS = [
    "티커",    # ← 리포트가 이 컬럼 기준으로 읽음
    "회사명",
    "거래소",  # TSXV / NASDAQ / NYSE / TSX
    "섹터",
    "매수가",  # 평균 단가 (USD 또는 CAD)
    "수량",    # 보유 주수
    "매수일",  # YYYY-MM-DD
    "메모",
]

# ── 분석 컬럼 (파이프라인이 매일 자동 업데이트) ──────────────────────────
TIMEFRAMES = ["1H", "4H", "Daily", "Weekly", "Monthly"]

ANALYSIS_HEADERS: list[str] = ["현재가"]  # 현재가 가장 먼저
for _tf in TIMEFRAMES:
    ANALYSIS_HEADERS += [f"[{_tf}] RSI", f"[{_tf}] MACD", f"[{_tf}] 거래량"]

ANALYSIS_HEADERS.append("업데이트")  # 마지막: 마지막 갱신 시각

# 전체 헤더
MY_ACCOUNT_HEADERS = INPUT_HEADERS + ANALYSIS_HEADERS

# 분석 컬럼이 시작하는 인덱스 (0-based)
ANALYSIS_START_COL = len(INPUT_HEADERS)

# 샘플 행 (처음 시트 생성 시)
MY_ACCOUNT_SAMPLE_ROWS = [
    ["NBM.V", "Neo Battery Materials", "TSXV", "Materials", "", "", "", "샘플 — 매수가/수량/날짜 입력하세요"]
    + [""] * len(ANALYSIS_HEADERS),
]

# ---------------------------------------------------------------------------
# 지표 설정
# ---------------------------------------------------------------------------

TIMEFRAME_CONFIG = {
    "1H":      {"interval": "1h",  "period": "60d",  "resample": None},
    "4H":      {"interval": "1h",  "period": "60d",  "resample": "4h"},  # 1h → 4h 리샘플
    "Daily":   {"interval": "1d",  "period": "1y",   "resample": None},
    "Weekly":  {"interval": "1wk", "period": "5y",   "resample": None},
    "Monthly": {"interval": "1mo", "period": "5y",   "resample": None},
}

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL_W = 12, 26, 9
VOLUME_MA_PERIOD = 20


# ---------------------------------------------------------------------------
# Google Sheets 연동
# ---------------------------------------------------------------------------

def _open_sheet():
    """gspread 클라이언트 인증 후 스프레드시트 핸들 반환."""
    if not GOOGLE_SHEETS_ENABLED or not GOOGLE_SHEETS_SPREADSHEET_ID:
        return None
    if gspread is None or Credentials is None:
        print("[내계좌] gspread 라이브러리 없음")
        return None
    try:
        svc = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if svc:
            credentials = Credentials.from_service_account_info(
                json.loads(svc), scopes=GOOGLE_SCOPES
            )
        elif GOOGLE_SHEETS_CREDENTIALS_PATH:
            credentials = Credentials.from_service_account_file(
                GOOGLE_SHEETS_CREDENTIALS_PATH, scopes=GOOGLE_SCOPES
            )
        else:
            print("[내계좌] 서비스 계정 인증 정보 없음")
            return None
        client = gspread.authorize(credentials)
        return client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
    except Exception as exc:
        print(f"[내계좌] 스프레드시트 열기 실패: {exc}")
        return None


def _get_or_create_worksheet(sheet):
    """'내계좌' 워크시트를 가져오거나 없으면 생성한다."""
    try:
        return sheet.worksheet(MY_ACCOUNT_WORKSHEET)
    except Exception:
        ws = sheet.add_worksheet(
            title=MY_ACCOUNT_WORKSHEET,
            rows="200",
            cols=str(len(MY_ACCOUNT_HEADERS) + 2),
        )
        print(f"[내계좌] '{MY_ACCOUNT_WORKSHEET}' 시트 신규 생성")
        return ws


def initialize_my_account_sheet(sheet) -> bool:
    """시트가 비어 있으면 헤더 + 샘플 행 + 스타일을 초기화한다.

    이미 데이터가 있으면 건드리지 않는다.
    Returns True if sheet is ready.
    """
    try:
        ws = _get_or_create_worksheet(sheet)

        existing = ws.get_all_values()
        if existing and existing[0]:
            # 헤더가 있는데 분석 컬럼이 없으면 추가
            current_headers = existing[0]
            if len(current_headers) < len(MY_ACCOUNT_HEADERS):
                missing = MY_ACCOUNT_HEADERS[len(current_headers):]
                print(f"[내계좌] 분석 컬럼 {len(missing)}개 추가: {missing}")
                # 헤더 행 확장
                ws.update([MY_ACCOUNT_HEADERS], value_input_option="USER_ENTERED")
            else:
                print(f"[내계좌] 헤더 이미 있음 ({len(current_headers)}컬럼) — 스킵")
            return True

        # 완전히 비어있음 → 헤더 + 샘플 삽입
        rows_to_write = [MY_ACCOUNT_HEADERS] + MY_ACCOUNT_SAMPLE_ROWS
        ws.update(rows_to_write, value_input_option="USER_ENTERED")

        # ── 헤더 스타일링 ──────────────────────────────────────────────
        last_col_letter = _col_letter(len(MY_ACCOUNT_HEADERS))

        # 전체 헤더: 진한 남색 배경 + 흰 볼드
        ws.format(f"A1:{last_col_letter}1", {
            "textFormat": {
                "bold": True,
                "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                "fontSize": 10,
            },
            "backgroundColor": {"red": 0.10, "green": 0.13, "blue": 0.24},
            "horizontalAlignment": "CENTER",
            "verticalAlignment": "MIDDLE",
        })

        # 입력 컬럼 헤더: 약간 밝은 파랑
        input_last = _col_letter(len(INPUT_HEADERS))
        ws.format(f"A1:{input_last}1", {
            "backgroundColor": {"red": 0.13, "green": 0.20, "blue": 0.40},
        })

        # 분석 컬럼 헤더: 진한 남색 (ANALYSIS_START_COL+1 ~ 끝)
        analysis_first = _col_letter(ANALYSIS_START_COL + 1)
        ws.format(f"{analysis_first}1:{last_col_letter}1", {
            "backgroundColor": {"red": 0.07, "green": 0.09, "blue": 0.18},
        })

        # 열 너비 자동 조정 힌트 (gspread는 직접 지원 안 하므로 스킵)
        print(f"[내계좌] 시트 초기화 완료 — 총 {len(MY_ACCOUNT_HEADERS)}컬럼")
        print(f"  입력 컬럼: {INPUT_HEADERS}")
        print(f"  분석 컬럼: {ANALYSIS_HEADERS}")
        return True

    except Exception as exc:
        print(f"[내계좌] 시트 초기화 실패: {exc}")
        return False


def _col_letter(n: int) -> str:
    """1-based 컬럼 인덱스 → 알파벳 컬럼명 (1→A, 26→Z, 27→AA)."""
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


# ---------------------------------------------------------------------------
# 티커 읽기
# ---------------------------------------------------------------------------

def fetch_my_tickers() -> tuple[list[str], dict[str, int]]:
    """'내계좌' 시트에서 티커 목록과 각 티커의 행 번호(1-based)를 반환한다.

    Returns:
        (tickers, row_map) — row_map: {ticker: row_number}
    """
    sheet = _open_sheet()
    if sheet is None:
        print("[내계좌] Google Sheets 연결 실패")
        return [], {}

    ws = _get_or_create_worksheet(sheet)

    # 시트 초기화 확인
    existing = ws.get_all_values()
    if not existing or not existing[0]:
        initialize_my_account_sheet(sheet)
        existing = ws.get_all_values()

    # 헤더 확인 후 부족하면 확장
    if existing and len(existing[0]) < len(MY_ACCOUNT_HEADERS):
        ws.update([MY_ACCOUNT_HEADERS], value_input_option="USER_ENTERED")

    if not existing or len(existing) < 2:
        print("[내계좌] 데이터 행 없음 — '내계좌' 시트에 티커를 추가하세요.")
        return [], {}

    header = existing[0]
    try:
        ticker_col_idx = header.index(MY_ACCOUNT_TICKER_COLUMN)
    except ValueError:
        ticker_col_idx = 0  # 기본값: 첫 번째 컬럼

    tickers = []
    row_map: dict[str, int] = {}

    for row_idx, row in enumerate(existing[1:], start=2):  # 2: 헤더 다음 행 (1-based)
        if not row:
            continue
        ticker = row[ticker_col_idx].strip().upper() if len(row) > ticker_col_idx else ""
        if ticker and ticker not in ("", "샘플"):
            tickers.append(ticker)
            row_map[ticker] = row_idx

    print(f"[내계좌] {len(tickers)}개 종목: {tickers}")
    return tickers, row_map


# ---------------------------------------------------------------------------
# 데이터 수집 & 지표 계산
# ---------------------------------------------------------------------------

def _fetch_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        return df.dropna(subset=["Close"])
    except Exception as exc:
        print(f"    ⚠ {ticker} {interval} 수집 실패: {exc}")
        return pd.DataFrame()


def _resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return pd.DataFrame()
    try:
        return df_1h.resample("4h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def _calc_indicators(df: pd.DataFrame) -> dict:
    """RSI, MACD 히스토그램, 거래량 비율 계산. 마지막 캔들 기준."""
    empty = {"rsi": None, "macd_hist": None, "volume_ratio": None}
    if df.empty or len(df) < 30:
        return empty

    close = df["Close"]
    volume = df["Volume"]

    # RSI
    rsi_val = None
    try:
        if RSIIndicator:
            v = RSIIndicator(close=close, window=RSI_PERIOD).rsi().iloc[-1]
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
            loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
            v = (100 - 100 / (1 + gain / (loss + 1e-10))).iloc[-1]
        if not np.isnan(float(v)):
            rsi_val = round(float(v), 1)
    except Exception:
        pass

    # MACD Histogram
    macd_hist_val = None
    try:
        if TAmacd:
            obj = TAmacd(close=close, window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL_W)
            macd_hist_val = round(float(obj.macd_diff().iloc[-1]), 5)
        else:
            ema_f = close.ewm(span=MACD_FAST, adjust=False).mean()
            ema_s = close.ewm(span=MACD_SLOW, adjust=False).mean()
            ml = ema_f - ema_s
            sig = ml.ewm(span=MACD_SIGNAL_W, adjust=False).mean()
            macd_hist_val = round(float((ml - sig).iloc[-1]), 5)
        if macd_hist_val is not None and np.isnan(macd_hist_val):
            macd_hist_val = None
    except Exception:
        pass

    # 거래량 비율
    volume_ratio = None
    try:
        if len(volume) >= VOLUME_MA_PERIOD:
            avg = volume.iloc[-VOLUME_MA_PERIOD - 1:-1].mean()
            cur = volume.iloc[-1]
            if avg > 0:
                volume_ratio = round(float(cur / avg), 2)
    except Exception:
        pass

    return {"rsi": rsi_val, "macd_hist": macd_hist_val, "volume_ratio": volume_ratio}


def analyze_ticker(ticker: str) -> dict[str, dict]:
    """5개 타임프레임 전부 지표 계산 후 반환.

    Returns:
        {
          "1H": {rsi, macd_hist, volume_ratio},
          ...,
          "_price": float | None  # 최신 종가 (현재가)
        }
    """
    results: dict[str, dict] = {}
    current_price: Optional[float] = None

    # 1H 데이터는 4H 리샘플에도 재사용
    df_1h = _fetch_ohlcv(ticker, "1h", "60d")
    time.sleep(0.4)

    for tf in TIMEFRAMES:
        cfg = TIMEFRAME_CONFIG[tf]
        if tf == "1H":
            df = df_1h.copy() if not df_1h.empty else pd.DataFrame()
        elif tf == "4H":
            df = _resample_4h(df_1h)
        else:
            df = _fetch_ohlcv(ticker, cfg["interval"], cfg["period"])
            time.sleep(0.4)

        # Daily 데이터에서 현재가(최신 종가) 추출
        if tf == "Daily" and not df.empty:
            try:
                current_price = round(float(df["Close"].iloc[-1]), 4)
            except Exception:
                pass

        ind = _calc_indicators(df)
        results[tf] = ind

        rsi_s = f"RSI={ind['rsi']}" if ind['rsi'] is not None else "RSI=N/A"
        mh_s = f"MACD={ind['macd_hist']:+.5f}" if ind['macd_hist'] is not None else "MACD=N/A"
        vr_s = f"Vol={ind['volume_ratio']}x" if ind['volume_ratio'] is not None else "Vol=N/A"
        ok = "✅" if ind['rsi'] is not None else "❌"
        print(f"    {tf:7s}: {rsi_s:12s} {mh_s:18s} {vr_s:10s} {ok}")

    results["_price"] = current_price
    if current_price is not None:
        print(f"    현재가: ${current_price}")

    return results


# ---------------------------------------------------------------------------
# 시트에 분석 결과 업데이트
# ---------------------------------------------------------------------------

def _format_rsi(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.1f}"


def _format_macd(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:+.5f}"


def _format_vol(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.2f}x"


def update_sheet_with_indicators(
    ticker_results: dict[str, dict[str, dict]],
    row_map: dict[str, int],
) -> None:
    """분석 결과를 '내계좌' 시트의 각 행 분석 컬럼에 업데이트한다."""
    sheet = _open_sheet()
    if sheet is None:
        print("[내계좌] 시트 업데이트 스킵 (연결 실패)")
        return

    ws = _get_or_create_worksheet(sheet)
    now_str = datetime.now(tz=timezone(timedelta(hours=-4))).strftime("%Y-%m-%d %H:%M EDT")

    for ticker, tf_data in ticker_results.items():
        row_num = row_map.get(ticker)
        if row_num is None:
            continue

        # 분석 값들을 ANALYSIS_HEADERS 순서대로 나열
        analysis_values: list[str] = []
        # 현재가 (맨 앞)
        price = tf_data.get("_price")
        analysis_values.append(f"{price:.4f}" if price is not None else "")
        for tf in TIMEFRAMES:
            ind = tf_data.get(tf, {})
            analysis_values.append(_format_rsi(ind.get("rsi")))
            analysis_values.append(_format_macd(ind.get("macd_hist")))
            analysis_values.append(_format_vol(ind.get("volume_ratio")))
        analysis_values.append(now_str)  # 업데이트 시각

        # 분석 컬럼 범위: ANALYSIS_START_COL+1 ~ 끝 (1-based)
        start_col = ANALYSIS_START_COL + 1
        end_col = start_col + len(analysis_values) - 1
        cell_range = f"{_col_letter(start_col)}{row_num}:{_col_letter(end_col)}{row_num}"

        try:
            ws.update(cell_range, [analysis_values], value_input_option="USER_ENTERED")

            # RSI 셀 색상 적용 (Daily RSI 기준)
            daily_rsi = tf_data.get("Daily", {}).get("rsi")
            if daily_rsi is not None:
                daily_rsi_col_idx = ANALYSIS_START_COL + 1 + TIMEFRAMES.index("Daily") * 3
                rsi_cell = f"{_col_letter(daily_rsi_col_idx)}{row_num}"
                if daily_rsi >= 70:
                    bg = {"red": 1.0, "green": 0.85, "blue": 0.85}  # 연빨강 (과매수)
                elif daily_rsi <= 30:
                    bg = {"red": 0.85, "green": 1.0, "blue": 0.85}  # 연초록 (과매도)
                else:
                    bg = {"red": 1.0, "green": 1.0, "blue": 1.0}    # 흰색
                ws.format(rsi_cell, {"backgroundColor": bg})

            print(f"    [시트] {ticker} 행 {row_num} 업데이트 완료")
        except Exception as exc:
            print(f"    [시트] {ticker} 업데이트 실패: {exc}")


# ---------------------------------------------------------------------------
# HTML 이메일 빌드
# ---------------------------------------------------------------------------

def _rsi_badge(v: Optional[float]) -> str:
    """RSI 값을 색상 배지 HTML로 반환."""
    if v is None:
        return "<span style='color:#bbb;font-size:12px'>—</span>"
    if v >= 70:
        bg, fg = "#fde8e8", "#c0392b"
        icon = "🔴"
    elif v <= 30:
        bg, fg = "#e8f5ee", "#1e8449"
        icon = "🟢"
    else:
        bg, fg = "#f0f4ff", "#2c5282"
        icon = ""
    return (
        f"<span style='background:{bg};color:{fg};font-weight:700;font-size:13px;"
        f"padding:3px 9px;border-radius:20px;display:inline-block;white-space:nowrap'>"
        f"{icon} {v:.1f}</span>"
    )


def _rsi_cell(v: Optional[float]) -> str:
    return f"<td style='text-align:center;padding:10px 8px'>{_rsi_badge(v)}</td>"


def _macd_cell(v: Optional[float]) -> str:
    if v is None:
        return "<td style='color:#bbb;text-align:center;font-size:12px'>—</td>"
    if v > 0:
        bg, fg, arrow = "#e8f5ee", "#1a7a41", "▲"
    else:
        bg, fg, arrow = "#fde8e8", "#c0392b", "▼"
    return (
        f"<td style='text-align:center;padding:10px 8px'>"
        f"<span style='background:{bg};color:{fg};font-weight:700;font-size:12px;"
        f"padding:3px 8px;border-radius:4px;display:inline-block'>"
        f"{arrow} {v:+.4f}</span></td>"
    )


def _vol_cell(v: Optional[float]) -> str:
    if v is None:
        return "<td style='color:#bbb;text-align:center;font-size:12px'>—</td>"
    if v >= 2.0:
        txt, bg, fg = f"🔥 {v:.1f}x", "#fff3e0", "#d35400"
    elif v >= 1.5:
        txt, bg, fg = f"↑ {v:.1f}x", "#fff8e1", "#e67e22"
    elif v < 0.7:
        txt, bg, fg = f"↓ {v:.1f}x", "#f5f5f5", "#95a5a6"
    else:
        txt, bg, fg = f"{v:.1f}x", "transparent", "#555"
    return (
        f"<td style='text-align:center;padding:10px 8px'>"
        f"<span style='background:{bg};color:{fg};font-weight:600;font-size:12px;"
        f"padding:2px 8px;border-radius:4px;display:inline-block'>{txt}</span></td>"
    )


# 타임프레임별 헤더 색상
_TF_COLORS = {
    "1H":      ("#3a5a8a", "#dce8f7"),
    "4H":      ("#2e5282", "#d4e5f5"),
    "Daily":   ("#1a6b4a", "#d4f0e4"),
    "Weekly":  ("#6a4a1a", "#f5e8d0"),
    "Monthly": ("#5a1a6a", "#f0d4f5"),
}

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body{font-family:'Inter',sans-serif;background:#f4f6f9;margin:0;padding:0;color:#1a202c}
.wrap{max-width:900px;margin:0 auto;padding:24px 16px}
.hdr{
  background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f3460 100%);
  color:#fff;padding:28px 32px;border-radius:16px 16px 0 0;
  border-bottom:3px solid #3b82f6;
}
.hdr-top{display:flex;align-items:center;gap:12px;margin-bottom:6px}
.hdr h1{margin:0;font-size:22px;font-weight:700;letter-spacing:-.3px}
.hdr p{margin:0;color:rgba(255,255,255,.55);font-size:12px;letter-spacing:.2px}
.card{
  background:#fff;border-radius:0 0 16px 16px;
  box-shadow:0 4px 24px rgba(0,0,0,.07);
  padding:0;
}
.section{
  padding:24px 28px;
  border-bottom:1px solid #f0f0f0;
}
.section:last-child{border-bottom:none}
.ticker-hdr{
  display:flex;align-items:center;gap:12px;margin-bottom:8px;
}
.ticker-sym{
  font-size:20px;font-weight:700;color:#0f172a;letter-spacing:-.3px;
}
.price-tag{
  font-size:15px;font-weight:500;color:#64748b;
}
.tv-btn{
  font-size:11px;color:#2563eb;text-decoration:none;
  background:#eff6ff;padding:4px 10px;border-radius:6px;
  border:1px solid #bfdbfe;font-weight:600;white-space:nowrap;
}
.summary-row{
  display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;
}
.badge{
  font-size:12px;padding:4px 12px;border-radius:20px;
  font-weight:600;display:inline-flex;align-items:center;gap:4px;
}
.tbl-wrap{border-radius:10px;overflow:hidden;border:1px solid #e5e9ef}
table{width:100%;border-collapse:collapse;font-size:13px}
thead tr.tf-header th{
  padding:0;border:none;
}
thead tr.tf-header th .tf-label-cell{
  padding:9px 14px;
  font-weight:700;font-size:11px;letter-spacing:.5px;text-transform:uppercase;
  text-align:center;
}
thead tr.col-header th{
  background:#f8fafc;color:#64748b;font-weight:600;
  font-size:11px;text-transform:uppercase;letter-spacing:.3px;
  padding:8px 10px;text-align:center;
  border-top:1px solid #e5e9ef;border-bottom:2px solid #dde3ec;
}
tbody tr{transition:background .15s}
tbody tr:hover{background:#f9fafb}
tbody td{
  padding:11px 10px;border-bottom:1px solid #f0f4f8;
  vertical-align:middle;
}
tbody tr:last-child td{border-bottom:none}
td.tf-row-label{
  font-weight:700;font-size:12px;text-align:center;
  white-space:nowrap;padding:11px 14px;
  border-right:1px solid #e5e9ef;
}
.footer{
  text-align:center;color:#94a3b8;font-size:11px;
  margin-top:20px;line-height:2;padding:0 16px 8px;
}
</style>"""


def build_html_report(ticker_results: dict[str, dict[str, dict]]) -> str:
    now_edt = datetime.now(tz=timezone(timedelta(hours=-4)))
    date_str = now_edt.strftime("%Y-%m-%d %H:%M EDT")
    n_tickers = len(ticker_results)

    body = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>내계좌 포트폴리오 리포트 - {date_str}</title>
</head>
<body>
{CSS}
<div class="wrap">
  <div class="hdr">
    <div class="hdr-top">
      <span style='font-size:26px'>📊</span>
      <h1>내계좌 포트폴리오 일일 리포트</h1>
    </div>
    <p>{date_str} &nbsp;·&nbsp; {n_tickers}종목 &nbsp;·&nbsp; RSI · MACD · 거래량 (멀티 타임프레임)</p>
  </div>
  <div class="card">
"""

    for ticker, tf_data in ticker_results.items():
        tv_ticker = TRADINGVIEW_TICKER_MAP.get(ticker, ticker)
        tv_link = f"https://www.tradingview.com/chart/?symbol={tv_ticker}"

        current_price = tf_data.get("_price")
        price_str = f"${current_price:.4f}" if current_price is not None else ""

        # 요약 배지
        daily = tf_data.get("Daily", {})
        d_rsi = daily.get("rsi")
        d_hist = daily.get("macd_hist")
        d_vol  = daily.get("volume_ratio")
        badges = ""
        if d_rsi is not None:
            if d_rsi >= 70:
                badges += "<span class='badge' style='background:#fde8e8;color:#c0392b'>🔴 RSI 과매수</span>"
            elif d_rsi <= 30:
                badges += "<span class='badge' style='background:#e8f5ee;color:#1e8449'>🟢 RSI 과매도</span>"
        if d_hist is not None:
            if d_hist > 0:
                badges += "<span class='badge' style='background:#e8f5ee;color:#1a7a41'>▲ MACD 상승</span>"
            else:
                badges += "<span class='badge' style='background:#fde8e8;color:#c0392b'>▼ MACD 하락</span>"
        if d_vol is not None and d_vol >= 2.0:
            badges += "<span class='badge' style='background:#fff3e0;color:#d35400'>🔥 거래량 급증</span>"

        body += f"""
    <div class="section">
      <div class="ticker-hdr">
        <span class="ticker-sym">{ticker}</span>
        <span class="price-tag">{price_str}</span>
        <a href="{tv_link}" class="tv-btn" target="_blank">TradingView →</a>
      </div>
      {f'<div class="summary-row">{badges}</div>' if badges else ''}
      <div class="tbl-wrap">
      <table>
        <thead>
          <tr class="col-header">
            <th style='width:80px;text-align:left;padding-left:14px'>타임프레임</th>
            <th>RSI</th>
            <th>MACD 히스토그램</th>
            <th>거래량 (vs 20일평균)</th>
          </tr>
        </thead>
        <tbody>
"""
        for tf in TIMEFRAMES:
            ind = tf_data.get(tf, {})
            tf_color, tf_bg = _TF_COLORS.get(tf, ("#374151", "#f9fafb"))
            body += (
                f"          <tr>\n"
                f"            <td class='tf-row-label' style='background:{tf_bg};color:{tf_color}'>{tf}</td>\n"
            )
            body += _rsi_cell(ind.get("rsi"))
            body += _macd_cell(ind.get("macd_hist"))
            body += _vol_cell(ind.get("volume_ratio"))
            body += "\n          </tr>\n"

        body += "        </tbody>\n      </table>\n      </div>\n    </div>\n"

    body += """  </div>
  <div class="footer">
    🔴 RSI 70+ 과매수 &nbsp;|&nbsp; 🟢 RSI 30- 과매도 &nbsp;|&nbsp;
    ▲ MACD 양수=골든 / ▼ 음수=데드 &nbsp;|&nbsp; 🔥 거래량 2x+ 급증<br>
    거래량 = 현재 캔들 / 20일 평균 배수 &nbsp;·&nbsp; 본 메일은 시스템에 의해 자동으로 발송되었습니다.
  </div>
</div>
</body>
</html>
"""
    return body



# ---------------------------------------------------------------------------
# 이메일 발송
# ---------------------------------------------------------------------------

def send_portfolio_email(html_body: str, tickers: list[str]) -> bool:
    if not EMAIL_PASSWORD:
        print("[내계좌] EMAIL_PASSWORD 미설정 — 이메일 스킵")
        return False

    date_str = datetime.now().strftime("%Y-%m-%d")
    ticker_summary = ", ".join(tickers[:5]) + ("..." if len(tickers) > 5 else "")
    subject = f"📊 내계좌 리포트 [{ticker_summary}] - {date_str}"

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = MY_ACCOUNT_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"[내계좌] ✅ 이메일 발송 → {MY_ACCOUNT_EMAIL} ({len(tickers)}개 종목)")
        return True
    except Exception as exc:
        print(f"[내계좌] ❌ 이메일 발송 실패: {exc}")
        return False


# ---------------------------------------------------------------------------
# 메인 엔트리포인트
# ---------------------------------------------------------------------------

def run_portfolio_report() -> bool:
    print("=" * 60)
    print("[내계좌] 포트폴리오 일일 리포트 시작")
    print("=" * 60)

    if not MY_ACCOUNT_REPORT_ENABLED:
        print("[내계좌] MY_ACCOUNT_REPORT_ENABLED=False — 스킵")
        return False

    # 1. 시트에서 티커 + 행 번호 가져오기 (없으면 자동 초기화)
    tickers, row_map = fetch_my_tickers()
    if not tickers:
        print("[내계좌] 분석할 종목 없음. '내계좌' 시트에 티커를 추가하세요.")
        return False

    # 2. 각 티커별 멀티 타임프레임 지표 계산
    ticker_results: dict[str, dict[str, dict]] = {}
    for ticker in tickers:
        print(f"\n[내계좌] ▶ {ticker} 분석 중...")
        ticker_results[ticker] = analyze_ticker(ticker)

    # 3. 시트 분석 컬럼 업데이트
    print("\n[내계좌] 시트 업데이트 중...")
    update_sheet_with_indicators(ticker_results, row_map)

    # 4. HTML 이메일 빌드 & 발송
    html_body = build_html_report(ticker_results)
    success = send_portfolio_email(html_body, tickers)

    print("\n" + "=" * 60)
    print(f"[내계좌] 완료 — 이메일 {'✅ 성공' if success else '❌ 실패'}")
    print("=" * 60)
    return success


if __name__ == "__main__":
    run_portfolio_report()
