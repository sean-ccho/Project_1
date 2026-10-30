#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""내계좌 포트폴리오 일일 기술 지표 리포트.

Google Sheets '내계좌' 워크시트에 있는 종목들을 읽어
1H / 4H / Daily / Weekly / Monthly 타임프레임별 RSI·MACD·거래량 지표를 계산하고:
  1) '내계좌' 시트 분석 컬럼을 업데이트한다.
  2) chunghwan14@gmail.com 으로 개인 전용 HTML 이메일을 발송한다.

시트 컬럼 구조 (열은 머리글 **이름**으로 찾는다 — 빠진 머리글만 오른쪽 끝에 붙임):
  [입력 — 사용자 직접 작성] 티커, 회사명, 거래소, 섹터, 매수가, 수량, 매수일, 메모
  [분석 — 파이프라인 자동] 현재가, 1H RSI, 1H MACD, 1H 거래량, 4H RSI, 4H MACD, 4H 거래량,
                            Daily RSI, Daily MACD, Daily 거래량,
                            Weekly RSI, Weekly MACD, Weekly 거래량,
                            Monthly RSI, Monthly MACD, Monthly 거래량,
                            업데이트
  [리스크 — 파이프라인 자동] ATR%, 베타(1년), 평균상관, 실적일, 공시, 실적 평균, 실적 최악, 실적 시간
  [입력 — 웹 내계좌] 손절가, 매수이유, 매도일, 매도가, 상태, 목표가, 실적대응
한 줄 = 거래 하나 (2026-10-16 매매일지와 합침, Logan 탭처럼). 매도일이 비어 있으면 보유 중 → 분석·알림 대상,
매도일이 있으면 매도 완료 → 분석 안 함 (남아 있던 분석 값은 지운다). '상태' 칸은 매일 밤 매도일 기준으로 맞춘다.
'관심종목' 탭(웹에서 추가)이 있으면 현재가·목표까지·도달을 채우고, 메일 맨 위에 알림(손절가 아래·근접,
실적 14일 이내 — 이 종목 실적 때 움직임·내 수량이면 얼마·결정 마감·실적대응, 보유 종목·관심종목 목표가 도달,
3일 안에 나온 SEC 🔴·🟠 공시, 관심종목의 🔴 공시)을 붙인다.

사용:
    python src/screener/portfolio_report.py
"""

from __future__ import annotations

import json
import os
import smtplib
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
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

from screener.earnings_moves import (WINDOW_DAYS, alert_text, days_to, decision_deadline, decision_of, move_stats,
                                     reaction_moves, sessions_until, sheet_values, yf_accepts)
from screener.sec_filings import fetch_flags, format_flags
from screener.tv_symbol import tv_symbol
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

# ── 리스크 컬럼 (2026-10-16, 파이프라인 자동) ──────────────────────────────
#   ATR%      = 일봉 ATR(14) ÷ 종가 — 웹 포지션 크기 계산기의 손절폭 기본값
#   베타(1년) = 1년 일간수익률의 SPY 베타
#   평균상관  = 최근 120거래일 일간수익률로 본 다른 보유 종목과의 상관 평균 (종목이 2개 이상일 때)
#   실적일    = 다음 실적 발표일 (yfinance calendar)
#   공시      = SEC 공시 경고 🔴·🟠 (screener.sec_filings — 상장폐지 경고·재작성·보고서 지연·자금 조달 등)
#   실적 평균·실적 최악·실적 시간 (2026-10-29) = 최근 8번 실적 반응일 평균 |움직임| · 최악 · 보통 시간 (장 전/장 마감 후)
#             — screener.earnings_moves. 웹 내계좌 '다음 2주 실적'과 장중 감시 알림이 읽는다
RISK_HEADERS: list[str] = ["ATR%", "베타(1년)", "평균상관", "실적일", "공시", "실적 평균", "실적 최악", "실적 시간"]

# ── 웹 내계좌에서 쓰는 입력 칸 (2026-10-16) — 손절가는 알림 메일, 매수이유는 매수이유별 성적,
#    매도일·매도가가 있으면 판 거래 (상태 = 매도 완료). 상태는 매일 밤 매도일 기준으로 맞춘다.
#    목표가(2026-10-20)는 넘으면 알림 메일. 실적대응(2026-10-29)은 '절반 · 2026-11-04' (팔기·절반·그대로 · 그 실적일) —
#    날짜가 지금 실적일과 같을 때만 '정함'으로 본다 ──
EXTRA_INPUT_HEADERS: list[str] = ["손절가", "매수이유", "매도일", "매도가", "상태", "목표가", "실적대응"]
STATUS_OPEN, STATUS_CLOSED = "보유 중", "매도 완료"
# ── 샘플 줄 (2026-10-22) — 메모가 '샘플'로 시작하면 예시 줄. 시트·웹에 보이기만 하고 분석·알림·합계에서 뺀다
#    (처음 시트를 만들 때 넣는 줄이 이 메모로 시작한다). 상태 칸은 '샘플' ──
STATUS_SAMPLE = "샘플"
SAMPLE_MEMO_PREFIX = "샘플"

# 전체 헤더 (새 시트일 때의 순서). 이미 있는 시트는 **이름으로** 열을 찾고, 빠진 머리글만 오른쪽 끝에 붙인다.
MY_ACCOUNT_HEADERS = INPUT_HEADERS + ANALYSIS_HEADERS + RISK_HEADERS + EXTRA_INPUT_HEADERS

# 분석 컬럼이 시작하는 인덱스 (0-based) — 새 시트 머리글 색칠에만 쓴다 (값 쓰기는 머리글 이름으로)
ANALYSIS_START_COL = len(INPUT_HEADERS)

# 샘플 행 (처음 시트 생성 시)
MY_ACCOUNT_SAMPLE_ROWS = [
    ["NBM.V", "Neo Battery Materials", "TSXV", "Materials", "", "", "", "샘플 — 매수가/수량/날짜 입력하세요"]
    + [""] * (len(MY_ACCOUNT_HEADERS) - len(INPUT_HEADERS)),
]

# ── 관심종목 탭 (2026-10-16) — 웹에서 추가하고, 파이프라인이 현재가·도달 여부를 채운다 ──
WATCHLIST_WORKSHEET = "관심종목"
WATCH_INPUT_HEADERS: list[str] = ["티커", "목표가", "조건", "메모", "추가일"]  # 조건: 이하 / 이상
WATCH_ANALYSIS_HEADERS: list[str] = ["현재가", "목표까지", "도달", "업데이트"]
WATCH_HEADERS = WATCH_INPUT_HEADERS + WATCH_ANALYSIS_HEADERS

STOP_NEAR_PCT = 0.03       # 손절가 위 3% 이내면 '손절가 근접' 알림
EARNINGS_ALERT_DAYS = WINDOW_DAYS  # 실적 발표가 14일 안이면 알림 (2026-10-29 — 7일 → 14일, 연구 E1 = 10거래일)
SEC_NEW_DAYS = 3           # 이 기간 안에 나온 🔴·🟠 공시는 메일 맨 위 알림으로도 올린다
CORR_WINDOW = 120          # 평균상관 계산 거래일 수
BETA_MIN_OBS = 120         # 베타 계산 최소 거래일 수

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
            _ensure_headers(ws, existing[0], MY_ACCOUNT_HEADERS)
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


def _ensure_headers(ws, header: list[str], wanted: list[str]) -> list[str]:
    """빠진 머리글만 오른쪽 끝에 붙인다. 이미 있는 열은 옮기지 않는다 (웹·사람이 붙인 열을 덮어쓰지 않게).

    Returns:
        붙인 뒤의 머리글 (빈 꼬리 칸은 뺀 것).
    """
    header = [str(h).strip() for h in header]
    while header and not header[-1]:
        header.pop()
    missing = [h for h in wanted if h not in header]
    if not missing:
        return header
    need = len(header) + len(missing)
    if ws.col_count < need:
        ws.add_cols(need - ws.col_count)
    start = len(header) + 1
    ws.update([missing], f"{_col_letter(start)}1:{_col_letter(start + len(missing) - 1)}1",
              value_input_option="USER_ENTERED")
    print(f"[{ws.title}] 머리글 {len(missing)}개 추가: {missing}")
    return header + missing


def _to_float(v) -> Optional[float]:
    """시트 글자 → 숫자 ('$12.50', '1,234' 도). 없거나 숫자가 아니면 None."""
    s = str(v or "").replace("$", "").replace(",", "").strip()
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------------------
# 티커 읽기
# ---------------------------------------------------------------------------

def is_closed(rec: dict[str, str]) -> bool:
    """매도일이 있으면 판 거래."""
    return bool(str(rec.get("매도일", "") or "").strip())


def is_sample(rec: dict[str, str]) -> bool:
    """메모가 '샘플'로 시작하면 예시 줄 — 보유로 치지 않는다 (분석·알림·합계 제외)."""
    return str(rec.get("메모", "") or "").strip().startswith(SAMPLE_MEMO_PREFIX)


def fetch_my_tickers() -> tuple[list[str], dict[str, int], list[dict[str, str]]]:
    """'내계좌' 시트에서 보유 중 티커 목록, 각 티커의 행 번호(1-based), 행 내용(머리글 → 값)을 반환한다.

    Returns:
        (tickers, row_map, records) — tickers·row_map 은 보유 중(매도일 없음, 샘플 아님)인 줄만,
        records 는 티커가 있는 모든 줄 {머리글: 값, "_row": 행 번호, "_closed": "1"/"", "_sample": "1"/""}
    """
    sheet = _open_sheet()
    if sheet is None:
        print("[내계좌] Google Sheets 연결 실패")
        return [], {}, []

    ws = _get_or_create_worksheet(sheet)

    # 시트 초기화 확인
    existing = ws.get_all_values()
    if not existing or not existing[0]:
        initialize_my_account_sheet(sheet)
        existing = ws.get_all_values()

    # 빠진 머리글(리스크·웹 입력 칸)은 끝에 붙인다
    if existing and existing[0]:
        _ensure_headers(ws, existing[0], MY_ACCOUNT_HEADERS)

    if not existing or len(existing) < 2:
        print("[내계좌] 데이터 행 없음 — '내계좌' 시트에 티커를 추가하세요.")
        return [], {}, []

    header = [str(h).strip() for h in existing[0]]
    try:
        ticker_col_idx = header.index(MY_ACCOUNT_TICKER_COLUMN)
    except ValueError:
        ticker_col_idx = 0  # 기본값: 첫 번째 컬럼

    tickers = []
    row_map: dict[str, int] = {}
    records: list[dict[str, str]] = []

    for row_idx, row in enumerate(existing[1:], start=2):  # 2: 헤더 다음 행 (1-based)
        if not row:
            continue
        ticker = row[ticker_col_idx].strip().upper() if len(row) > ticker_col_idx else ""
        if ticker and ticker not in ("", "샘플"):
            rec = {h: (row[i] if i < len(row) else "") for i, h in enumerate(header) if h}
            rec["_row"] = str(row_idx)
            rec["_closed"] = "1" if is_closed(rec) else ""
            rec["_sample"] = "1" if is_sample(rec) else ""
            records.append(rec)
            if rec["_closed"] or rec["_sample"]:
                continue
            if ticker not in row_map:
                tickers.append(ticker)
            row_map[ticker] = row_idx

    n_closed = sum(1 for r in records if r["_closed"] and not r["_sample"])
    n_sample = sum(1 for r in records if r["_sample"])
    print(f"[내계좌] 보유 중 {len(tickers)}개 종목: {tickers} · 판 거래 {n_closed}줄"
          + (f" · 샘플 {n_sample}줄 (제외)" if n_sample else ""))
    return tickers, row_map, records


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

        # Daily 데이터에서 현재가(최신 종가)·ATR%·종가 시계열(베타·상관용) 추출
        if tf == "Daily" and not df.empty:
            try:
                current_price = round(float(df["Close"].iloc[-1]), 4)
            except Exception:
                pass
            results["_atr_pct"] = _atr_pct(df)
            results["_daily_close"] = df["Close"]

        ind = _calc_indicators(df)
        results[tf] = ind

        rsi_s = f"RSI={ind['rsi']}" if ind['rsi'] is not None else "RSI=N/A"
        mh_s = f"MACD={ind['macd_hist']:+.5f}" if ind['macd_hist'] is not None else "MACD=N/A"
        vr_s = f"Vol={ind['volume_ratio']}x" if ind['volume_ratio'] is not None else "Vol=N/A"
        ok = "✅" if ind['rsi'] is not None else "❌"
        print(f"    {tf:7s}: {rsi_s:12s} {mh_s:18s} {vr_s:10s} {ok}")

    results["_price"] = current_price
    results["_earnings"] = _next_earnings(ticker)
    # 실적 때 움직임 (최근 8번 ≈ 2년) — 일봉 1년으로는 모자라서 3년을 따로 받는다
    d3 = _fetch_ohlcv(ticker, "1d", "3y")
    results["_close_3y"] = d3["Close"] if not d3.empty else None
    time.sleep(0.4)
    if current_price is not None:
        print(f"    현재가: ${current_price}")

    return results


def _atr_pct(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    """일봉 ATR(n) ÷ 마지막 종가 (단순 평균 TR). 데이터가 모자라면 None."""
    if len(df) < n + 1:
        return None
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.iloc[-n:].mean())
    close = float(df["Close"].iloc[-1])
    return atr / close if close > 0 and np.isfinite(atr) else None


def _next_earnings(ticker: str) -> Optional[str]:
    """다음 실적 발표일 'YYYY-MM-DD' (오늘 이후 가장 가까운 날). 모르면 None."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        today = datetime.now(tz=timezone(timedelta(hours=-4))).date()
        future = sorted(d for d in (dates or []) if hasattr(d, "year") and d >= today)
        return future[0].isoformat() if future else None
    except Exception as exc:  # noqa: BLE001 — 실적일은 없어도 리포트는 계속
        print(f"    ⚠ {ticker} 실적일 조회 실패: {exc}")
        return None


def _daily_returns(close: Optional[pd.Series]) -> pd.Series:
    if close is None or len(close) < 2:
        return pd.Series(dtype=float)
    c = close.copy()
    if getattr(c.index, "tz", None) is not None:
        c.index = c.index.tz_localize(None)
    c.index = pd.DatetimeIndex(c.index).normalize()
    return c[~c.index.duplicated(keep="last")].pct_change().dropna()


def add_risk_stats(ticker_results: dict[str, dict], spy_close: Optional[pd.Series]) -> None:
    """종목마다 _beta(1년 SPY 베타)·_avg_corr(다른 보유 종목과의 평균 상관)를 채운다."""
    rets = {t: _daily_returns(r.get("_daily_close")) for t, r in ticker_results.items()}
    spy = _daily_returns(spy_close)
    for t, r in rets.items():
        both = pd.concat([r, spy], axis=1, join="inner").dropna()
        beta = None
        if len(both) >= BETA_MIN_OBS and both.iloc[:, 1].var() > 0:
            beta = float(both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var())
        ticker_results[t]["_beta"] = beta
        ticker_results[t]["_avg_corr"] = None
    usable = {t: r.iloc[-CORR_WINDOW:] for t, r in rets.items() if len(r) >= CORR_WINDOW // 2}
    if len(usable) >= 2:
        corr = pd.DataFrame(usable).corr(min_periods=CORR_WINDOW // 2)
        for t in corr.columns:
            others = corr[t].drop(t).dropna()
            ticker_results[t]["_avg_corr"] = float(others.mean()) if len(others) else None


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


def _analysis_values(tf_data: dict, now_str: str) -> dict[str, str]:
    """분석·리스크 머리글 → 시트에 쓸 값 (USER_ENTERED)."""
    price = tf_data.get("_price")
    vals: dict[str, str] = {"현재가": f"{price:.4f}" if price is not None else ""}
    for tf in TIMEFRAMES:
        ind = tf_data.get(tf, {})
        vals[f"[{tf}] RSI"] = _format_rsi(ind.get("rsi"))
        vals[f"[{tf}] MACD"] = _format_macd(ind.get("macd_hist"))
        vals[f"[{tf}] 거래량"] = _format_vol(ind.get("volume_ratio"))
    vals["업데이트"] = now_str
    atr, beta, corr, earn = (tf_data.get(k) for k in ("_atr_pct", "_beta", "_avg_corr", "_earnings"))
    vals["ATR%"] = f"{atr * 100:.2f}%" if atr is not None else ""
    vals["베타(1년)"] = f"{beta:.2f}" if beta is not None else ""
    vals["평균상관"] = f"{corr:.2f}" if corr is not None else ""
    vals["실적일"] = f"'{earn}" if earn else ""  # ' = 글자로 (시트 날짜 서식이 바꾸지 않게)
    vals["공시"] = format_flags(tf_data.get("_sec") or [])
    vals.update(sheet_values(tf_data.get("_earn_stats")))
    return vals


def update_sheet_with_indicators(
    ticker_results: dict[str, dict[str, dict]],
    row_map: dict[str, int | list[int]],
) -> None:
    """분석 결과를 '내계좌' 시트의 각 행에 **머리글 이름으로** 쓴다 (같은 티커가 여러 줄이면 모두)."""
    sheet = _open_sheet()
    if sheet is None:
        print("[내계좌] 시트 업데이트 스킵 (연결 실패)")
        return

    ws = _get_or_create_worksheet(sheet)
    now_str = datetime.now(tz=timezone(timedelta(hours=-4))).strftime("%Y-%m-%d %H:%M EDT")
    header = _ensure_headers(ws, ws.row_values(1), MY_ACCOUNT_HEADERS)
    col = {h: i + 1 for i, h in enumerate(header) if h}

    data: list[dict] = []
    colors: list[tuple[str, dict]] = []
    rsi_col = col.get("[Daily] RSI")
    for ticker, tf_data in ticker_results.items():
        rows = row_map.get(ticker)
        if rows is None:
            continue
        rows = rows if isinstance(rows, list) else [rows]
        vals = _analysis_values(tf_data, now_str)
        for row_num in rows:
            data += [{"range": f"{_col_letter(col[h])}{row_num}", "values": [[v]]} for h, v in vals.items() if h in col]
            daily_rsi = tf_data.get("Daily", {}).get("rsi")
            if daily_rsi is not None and rsi_col:
                if daily_rsi >= 70:
                    bg = {"red": 1.0, "green": 0.85, "blue": 0.85}  # 연빨강 (과매수)
                elif daily_rsi <= 30:
                    bg = {"red": 0.85, "green": 1.0, "blue": 0.85}  # 연초록 (과매도)
                else:
                    bg = {"red": 1.0, "green": 1.0, "blue": 1.0}    # 흰색
                colors.append((f"{_col_letter(rsi_col)}{row_num}", bg))
    if not data:
        return
    try:
        ws.batch_update(data, value_input_option="USER_ENTERED")
        for cell, bg in colors:
            ws.format(cell, {"backgroundColor": bg})
        print(f"    [시트] {len(ticker_results)}종목 분석 값 업데이트 완료 ({len(data)}칸)")
    except Exception as exc:
        print(f"    [시트] 업데이트 실패: {exc}")


def sync_trade_status(records: list[dict[str, str]]) -> int:
    """'상태' 칸을 매도일·메모 기준으로 맞추고 (보유 중 / 매도 완료 / 샘플),
    판 줄·샘플 줄에 남은 분석 값(현재가·RSI…)은 지운다 (더는 갱신하지 않으니 옛 값이 남지 않게).

    Returns:
        고친 칸 수.
    """
    if not records:
        return 0
    sheet = _open_sheet()
    if sheet is None:
        return 0
    ws = _get_or_create_worksheet(sheet)
    header = _ensure_headers(ws, ws.row_values(1), MY_ACCOUNT_HEADERS)
    col = {h: i + 1 for i, h in enumerate(header) if h}
    analysis = [h for h in ANALYSIS_HEADERS + RISK_HEADERS if h in col]
    data: list[dict] = []
    for rec in records:
        row = int(rec["_row"])
        want = STATUS_SAMPLE if rec.get("_sample") else STATUS_CLOSED if rec.get("_closed") else STATUS_OPEN
        if "상태" in col and str(rec.get("상태", "")).strip() != want:
            data.append({"range": f"{_col_letter(col['상태'])}{row}", "values": [[want]]})
        if rec.get("_closed") or rec.get("_sample"):
            data += [{"range": f"{_col_letter(col[h])}{row}", "values": [[""]]}
                     for h in analysis if str(rec.get(h, "")).strip()]
    if data:
        try:
            ws.batch_update(data, value_input_option="USER_ENTERED")
            print(f"[내계좌] 상태·판 줄 정리 {len(data)}칸")
        except Exception as exc:
            print(f"[내계좌] 상태 정리 실패: {exc}")
            return 0
    return len(data)


# ---------------------------------------------------------------------------
# 관심종목 · 알림
# ---------------------------------------------------------------------------

def _last_close(ticker: str) -> Optional[float]:
    df = _fetch_ohlcv(ticker, "1d", "5d")
    return round(float(df["Close"].iloc[-1]), 4) if not df.empty else None


def update_watchlist(sheet) -> list[dict[str, str]]:
    """'관심종목' 탭의 현재가·목표까지·도달을 채우고, 목표가에 닿은 종목 알림을 돌려준다. 탭이 없으면 건너뛴다."""
    if sheet is None:
        return []
    try:
        ws = sheet.worksheet(WATCHLIST_WORKSHEET)
    except Exception:
        print(f"[관심종목] '{WATCHLIST_WORKSHEET}' 탭 없음 — 건너뜀 (웹에서 첫 종목을 추가하면 생긴다)")
        return []
    values = ws.get_all_values()
    if not values or not values[0]:
        return []
    header = _ensure_headers(ws, values[0], WATCH_HEADERS)
    col = {h: i + 1 for i, h in enumerate(header) if h}
    now_str = datetime.now(tz=timezone(timedelta(hours=-4))).strftime("%Y-%m-%d %H:%M EDT")
    prices: dict[str, Optional[float]] = {}
    data: list[dict] = []
    alerts: list[dict[str, str]] = []
    warned: set[str] = set()
    watch_sec = fetch_flags([str(r[header.index("티커")]).strip().upper() for r in values[1:]
                             if "티커" in header and len(r) > header.index("티커") and str(r[header.index("티커")]).strip()])
    for row_num, row in enumerate(values[1:], start=2):
        rec = {h: (row[i] if i < len(row) else "") for i, h in enumerate(header)}
        ticker = str(rec.get("티커", "")).strip().upper()
        if not ticker:
            continue
        if ticker not in prices:
            prices[ticker] = _last_close(ticker)
            time.sleep(0.3)
        price = prices[ticker]
        target = _to_float(rec.get("목표가"))
        above = str(rec.get("조건", "")).strip() in ("이상", "above", ">=")
        hit = price is not None and target is not None and (price >= target if above else price <= target)
        out = {
            "현재가": f"{price:.4f}" if price is not None else "",
            "목표까지": f"{target / price - 1:.2%}" if price and target else "",
            "도달": ("도달" if hit else "") if target is not None else "",
            "업데이트": now_str,
        }
        data += [{"range": f"{_col_letter(col[h])}{row_num}", "values": [[v]]} for h, v in out.items() if h in col]
        for f in watch_sec.get(ticker.replace(".", "-"), []):
            if f.severity == "red" and ticker not in warned:
                warned.add(ticker)
                alerts.append({"kind": "sec", "ticker": ticker,
                               "text": f"관심종목 {ticker} 공시: {format_flags([f])} ({f.form}) — 사기 전에 확인하세요"})
        if hit:
            cond = "이상" if above else "이하"
            alerts.append({"kind": "watch", "ticker": ticker,
                           "text": f"관심종목 {ticker} 목표가 ${target:,.2f} {cond} 도달 — 현재 ${price:,.2f}"
                                   + (f" · 메모: {rec.get('메모')}" if rec.get("메모") else "")})
    if data:
        try:
            ws.batch_update(data, value_input_option="USER_ENTERED")
            print(f"[관심종목] {len(prices)}종목 갱신 · 목표가 도달 {len(alerts)}")
        except Exception as exc:
            print(f"[관심종목] 시트 업데이트 실패: {exc}")
    return alerts


def build_alerts(records: list[dict[str, str]], ticker_results: dict[str, dict], today=None) -> tuple[list[dict[str, str]], list[str]]:
    """보유 종목 알림 (손절가 아래·근접, 목표가 도달, 실적 임박) + 손절가가 비어 있는 티커 목록."""
    today = today or datetime.now(tz=timezone(timedelta(hours=-4))).date()
    alerts: list[dict[str, str]] = []
    no_stop: list[str] = []
    seen_earn: set[str] = set()
    seen_sec: set[str] = set()
    seen_target: set[tuple[str, float]] = set()
    shares: dict[str, float] = {}
    for rec in records:
        t = str(rec.get("티커", "")).strip().upper()
        shares[t] = shares.get(t, 0.0) + (_to_float(rec.get("수량")) or 0.0)
    account = sum(q * (ticker_results.get(t, {}).get("_price") or 0.0) for t, q in shares.items()) or None
    for rec in records:
        ticker = str(rec.get("티커", "")).strip().upper()
        res = ticker_results.get(ticker, {})
        price = res.get("_price")
        stop = _to_float(rec.get("손절가"))
        if stop is None:
            if ticker not in no_stop:
                no_stop.append(ticker)
        elif price is not None and stop > 0:
            if price <= stop:
                alerts.append({"kind": "stop", "ticker": ticker,
                               "text": f"{ticker} 현재 ${price:,.2f} — 손절가 ${stop:,.2f} 아래로 내려왔어요"})
            elif price <= stop * (1 + STOP_NEAR_PCT):
                alerts.append({"kind": "near", "ticker": ticker,
                               "text": f"{ticker} 손절가 ${stop:,.2f} 까지 {price / stop - 1:.1%} 남았어요 (현재 ${price:,.2f})"})
        target = _to_float(rec.get("목표가"))
        if target and price is not None and price >= target and (ticker, target) not in seen_target:
            seen_target.add((ticker, target))
            buy = _to_float(rec.get("매수가"))
            gain = f" · 매수가 대비 {price / buy - 1:+.1%}" if buy else ""
            alerts.append({"kind": "target", "ticker": ticker,
                           "text": f"{ticker} 목표가 ${target:,.2f} 도달 — 현재 ${price:,.2f}{gain}. "
                                   "일부 익절하거나 손절가를 올릴 때예요"})
        if ticker not in seen_sec:
            seen_sec.add(ticker)
            for f in res.get("_sec") or []:
                fd = asdict(f) if not isinstance(f, dict) else f
                if fd["severity"] in ("red", "orange") and 0 <= (today - date.fromisoformat(fd["date"])).days <= SEC_NEW_DAYS:
                    alerts.append({"kind": "sec", "ticker": ticker,
                                   "text": f"{ticker} 새 공시: {format_flags([fd])} ({fd['form']}) — {fd['url']}"})
        earn = res.get("_earnings")
        if earn and ticker not in seen_earn:
            days = days_to(earn, today)
            if days is not None and 0 <= days <= EARNINGS_ALERT_DAYS:
                seen_earn.add(ticker)
                st = res.get("_earn_stats")
                decision = next((d for d in (decision_of(r.get("실적대응"), earn) for r in records
                                             if str(r.get("티커", "")).strip().upper() == ticker) if d), None)
                deadline = decision_deadline(date.fromisoformat(earn[:10]), (st or {}).get("timing"))
                alerts.append({"kind": "earn", "ticker": ticker,
                               "urgent": "1" if not decision and 0 <= sessions_until(today, deadline) <= 2 else "",
                               "text": alert_text(ticker, earn, st, shares=shares.get(ticker) or None, price=price,
                                                  account=account, decision=decision, today=today)})
    return alerts, no_stop


_ALERT_ICON = {"stop": "🛑", "near": "⚠️", "target": "🎯", "earn": "📅", "watch": "🎯", "sec": "📄", "squeeze": "🔥"}


def squeeze_alerts(tickers: list[str], watch=None) -> list[dict[str, str]]:
    """보유 종목이 숏스퀴즈 감시 목록에 있으면 알림 (🔥 진행 가능 · 대기). 목록은 같은 실행의 short_squeeze 단계가 저장."""
    from screener.short_squeeze import load_watch  # noqa: PLC0415

    w = load_watch() if watch is None else watch
    if w.empty:
        return []
    by = {r.ticker: r for r in w.itertuples()}
    out = []
    for t in dict.fromkeys(tickers):
        r = by.get(t.strip().upper().replace(".", "-"))
        if r is None:
            continue
        state = "🔥 스퀴즈 진행 가능 (신고가 + 거래량 급증)" if r.status == "fire" else "숏스퀴즈 대기 (공매도 몰림)"
        out.append({"kind": "squeeze", "ticker": t,
                    "text": f"{t} {state} — 공매도 {r.short_pct:.0%} · 되사기 {r.dtc:.1f}일분. 크게 움직일 수 있어요 (위·아래 모두)"})
    return out


def _alerts_html(alerts: list[dict[str, str]], no_stop: list[str]) -> str:
    if not alerts and not no_stop:
        return ""
    # 실적 결정 마감이 2거래일 안인데 실적대응을 안 정했으면 빨갛게 (2026-10-29)
    items = "".join(f"<li style='margin:4px 0{';color:#b91c1c;font-weight:600' if a.get('urgent') else ''}'>"
                    f"{_ALERT_ICON.get(a['kind'], '•')} {a['text']}</li>" for a in alerts)
    tail = (f"<p style='margin:8px 0 0;color:#6b7280;font-size:12px'>손절가가 비어 있는 종목: {', '.join(no_stop)} "
            f"— 웹 내계좌에서 손절가를 적어 두면 근접 알림을 보내요</p>") if no_stop else ""
    head = f"⚠️ 알림 {len(alerts)}건" if alerts else "알림 없음"
    ul = f"<ul style='margin:0;padding-left:18px'>{items}</ul>" if items else ""
    return (f"<div class='card' style='padding:14px 18px;margin-bottom:14px;border-left:4px solid #d97706'>"
            f"<div style='font-weight:700;font-size:15px;margin-bottom:6px'>{head}</div>{ul}{tail}</div>")


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


def build_html_report(ticker_results: dict[str, dict[str, dict]], alerts: Optional[list[dict[str, str]]] = None,
                      no_stop: Optional[list[str]] = None) -> str:
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
  {_alerts_html(alerts or [], no_stop or [])}
  <div class="card">
"""

    for ticker, tf_data in ticker_results.items():
        tv_ticker = tv_symbol(ticker, TRADINGVIEW_TICKER_MAP)
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
        es = tf_data.get("_earn_stats")
        earn_txt = ""
        if tf_data.get("_earnings"):
            earn_txt = f"실적 {tf_data['_earnings']}" + (
                f" (지난 {es['n']}번 평균 ±{es['avg']:.1%} · 최악 {es['worst']:+.1%})" if es else "")
        risk = [f"ATR {tf_data['_atr_pct']:.1%}" if tf_data.get("_atr_pct") is not None else "",
                f"베타 {tf_data['_beta']:.2f}" if tf_data.get("_beta") is not None else "",
                f"평균상관 {tf_data['_avg_corr']:.2f}" if tf_data.get("_avg_corr") is not None else "",
                earn_txt,
                format_flags(tf_data.get("_sec") or [])]
        risk_line = " · ".join(x for x in risk if x)
        if risk_line:
            body += f"      <div style='color:#6b7280;font-size:12px;margin:0 0 8px 2px'>{risk_line}</div>\n"
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

def send_portfolio_email(html_body: str, tickers: list[str], n_alerts: int = 0) -> bool:
    if not EMAIL_PASSWORD:
        print("[내계좌] EMAIL_PASSWORD 미설정 — 이메일 스킵")
        return False

    date_str = datetime.now().strftime("%Y-%m-%d")
    ticker_summary = ", ".join(tickers[:5]) + ("..." if len(tickers) > 5 else "")
    subject = f"{f'⚠️ 알림 {n_alerts}건 · ' if n_alerts else ''}📊 내계좌 리포트 [{ticker_summary}] - {date_str}"

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

    # 1. 시트에서 티커 + 행 번호 가져오기 (없으면 자동 초기화). 상태 칸은 판 줄까지 매번 맞춘다
    tickers, row_map, records = fetch_my_tickers()
    sync_trade_status(records)
    records = [r for r in records if not r.get("_closed") and not r.get("_sample")]  # 아래 분석·알림은 보유 중인 줄만
    if not tickers:
        print("[내계좌] 분석할 종목 없음. '내계좌' 시트에 티커를 추가하세요.")
        return False

    # 2. 각 티커별 멀티 타임프레임 지표 계산 + 베타·평균상관
    ticker_results: dict[str, dict[str, dict]] = {}
    for ticker in tickers:
        print(f"\n[내계좌] ▶ {ticker} 분석 중...")
        ticker_results[ticker] = analyze_ticker(ticker)
    spy = _fetch_ohlcv("SPY", "1d", "1y")
    add_risk_stats(ticker_results, spy["Close"] if not spy.empty else None)
    accepts: dict[str, list[str]] = {}
    sec = fetch_flags(tickers, earnings=accepts)  # SEC 공시 경고 + 실적 8-K 시각 (SEC_USER_AGENT 없으면 빈 dict)
    for t in tickers:
        key = t.strip().upper().replace(".", "-")
        ticker_results[t]["_sec"] = sec.get(key, [])
        # 실적 때 움직임 — SEC 실적 8-K 시각, SEC 에 없는 종목(캐나다 등)은 yfinance 실적일
        acc = accepts.get(key) or yf_accepts(t)
        close = ticker_results[t].get("_close_3y")
        st = move_stats(reaction_moves(acc, close)) if acc and close is not None else None
        ticker_results[t]["_earn_stats"] = st
        if st:
            print(f"[내계좌] {t} 실적 때 움직임: 지난 {st['n']}번 평균 ±{st['avg']:.1%} · 최악 {st['worst']:+.1%} · "
                  f"{st['timing'] or '시간 모름'}")

    # 3. 시트 분석 컬럼 업데이트 (같은 티커 여러 줄이면 모두)
    print("\n[내계좌] 시트 업데이트 중...")
    rows_by_ticker: dict[str, list[int]] = {}
    for rec in records:
        rows_by_ticker.setdefault(str(rec.get("티커", "")).strip().upper(), []).append(int(rec["_row"]))
    update_sheet_with_indicators(ticker_results, rows_by_ticker or row_map)

    # 4. 알림 (손절가·실적 임박·관심종목 목표가) + HTML 이메일 빌드 & 발송
    alerts, no_stop = build_alerts(records, ticker_results)
    alerts += update_watchlist(_open_sheet())
    alerts += squeeze_alerts(tickers)
    for a in alerts:
        print(f"[내계좌] 알림: {a['text']}")
    html_body = build_html_report(ticker_results, alerts, no_stop)
    success = send_portfolio_email(html_body, tickers, len(alerts))

    print("\n" + "=" * 60)
    print(f"[내계좌] 완료 — 이메일 {'✅ 성공' if success else '❌ 실패'}")
    print("=" * 60)
    return success


if __name__ == "__main__":
    run_portfolio_report()
