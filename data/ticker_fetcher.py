"""NYSE/NASDAQ 전 종목 티커 리스트를 가져오는 유틸리티.

나스닥 트레이더 FTP 서버에서 상장 종목 목록을 다운로드하여
보통주(Common Stock)만 필터링한다.
"""

from __future__ import annotations

import io
from ftplib import FTP
from typing import List

import pandas as pd


FTP_HOST = "ftp.nasdaqtrader.com"
FTP_DIR = "SymbolDirectory"
NASDAQ_FILE = "nasdaqlisted.txt"
OTHER_FILE = "otherlisted.txt"


def _download_ftp_file(filename: str) -> str:
    """FTP 서버에서 파일을 다운로드하여 문자열로 반환한다."""
    buffer = io.BytesIO()
    with FTP(FTP_HOST) as ftp:
        ftp.login()  # anonymous login
        ftp.cwd(FTP_DIR)
        ftp.retrbinary(f"RETR {filename}", buffer.write)
    return buffer.getvalue().decode("utf-8")


def _parse_nasdaq_listed(raw: str) -> pd.DataFrame:
    """nasdaqlisted.txt를 파싱하여 보통주 티커만 반환한다."""
    df = pd.read_csv(io.StringIO(raw), sep="|")
    # 마지막 행은 메타데이터이므로 제거
    df = df[~df["Symbol"].str.contains("File Creation Time", na=False)]
    # ETF 제거 (ETF 컬럼이 'Y'인 경우)
    if "ETF" in df.columns:
        df = df[df["ETF"] != "Y"]
    # Test Issue 제거
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] != "Y"]
    return df[["Symbol"]].rename(columns={"Symbol": "ticker"})


def _parse_other_listed(raw: str) -> pd.DataFrame:
    """otherlisted.txt를 파싱하여 NYSE 등 기타 거래소의 보통주 티커만 반환한다."""
    df = pd.read_csv(io.StringIO(raw), sep="|")
    # 마지막 행은 메타데이터이므로 제거
    if "ACT Symbol" in df.columns:
        df = df[~df["ACT Symbol"].str.contains("File Creation Time", na=False)]
    # ETF 제거
    if "ETF" in df.columns:
        df = df[df["ETF"] != "Y"]
    # Test Issue 제거
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] != "Y"]
    # ACT Symbol을 ticker로 사용
    col = "ACT Symbol" if "ACT Symbol" in df.columns else "Symbol"
    return df[[col]].rename(columns={col: "ticker"})


def fetch_all_tickers() -> List[str]:
    """NYSE와 NASDAQ의 전체 보통주 티커 리스트를 반환한다.

    Returns:
        중복 제거된 티커 리스트 (정렬됨).
    """
    print("[ticker_fetcher] 나스닥 FTP 서버에서 종목 리스트 다운로드 중...")

    nasdaq_raw = _download_ftp_file(NASDAQ_FILE)
    other_raw = _download_ftp_file(OTHER_FILE)

    nasdaq_df = _parse_nasdaq_listed(nasdaq_raw)
    other_df = _parse_other_listed(other_raw)

    combined = pd.concat([nasdaq_df, other_df], ignore_index=True)
    # 클린업: 공백 제거, 빈 문자열 제거
    combined["ticker"] = combined["ticker"].str.strip()
    combined = combined[combined["ticker"].str.len() > 0]
    # 중복 제거 및 정렬
    tickers = sorted(combined["ticker"].unique().tolist())

    print(f"[ticker_fetcher] 총 {len(tickers)}개 종목 로드 완료")
    return tickers


def filter_tickers_basic(
    tickers: List[str],
    *,
    min_price: float = 1.0,
    min_avg_volume: int = 100_000,
) -> List[str]:
    """yfinance를 사용하여 최소 가격 및 거래량 기준으로 종목을 필터링한다.

    이 함수는 1단계 고속 필터에 해당하며, 페니스톡과 유동성 부족 종목을 제거한다.

    Args:
        tickers: 필터링할 티커 리스트.
        min_price: 최소 주가 ($).
        min_avg_volume: 최소 평균 거래량 (주).

    Returns:
        필터링된 티커 리스트.
    """
    import yfinance as yf
    import time

    BATCH_SIZE = 500
    passed: List[str] = []
    total = len(tickers)

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        batch_str = " ".join(batch)
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(
            f"[ticker_fetcher] 1단계 필터링 배치 {batch_num}/{total_batches} "
            f"({len(batch)}개 종목)..."
        )

        try:
            data = yf.download(
                batch_str,
                period="1mo",
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        if ticker not in data.columns.get_level_values(1):
                            continue
                        close = data["Close"][ticker].dropna()
                        volume = data["Volume"][ticker].dropna()
                        if close.empty or volume.empty:
                            continue
                        last_price = close.iloc[-1]
                        avg_vol = volume.mean()
                        if last_price >= min_price and avg_vol >= min_avg_volume:
                            passed.append(ticker)
                    except (KeyError, IndexError):
                        continue
            else:
                # 단일 티커인 경우
                if len(batch) == 1:
                    close = data["Close"].dropna()
                    volume = data["Volume"].dropna()
                    if not close.empty and not volume.empty:
                        last_price = close.iloc[-1]
                        avg_vol = volume.mean()
                        if last_price >= min_price and avg_vol >= min_avg_volume:
                            passed.append(batch[0])

        except Exception as e:
            print(f"[ticker_fetcher] 배치 {batch_num} 처리 실패: {e}")
            continue

        # API 부하 방지
        if i + BATCH_SIZE < total:
            time.sleep(1)

    print(
        f"[ticker_fetcher] 1단계 필터링 완료: {total}개 → {len(passed)}개 "
        f"(가격 ≥ ${min_price}, 평균 거래량 ≥ {min_avg_volume:,})"
    )
    return passed


if __name__ == "__main__":
    tickers = fetch_all_tickers()
    print(f"전체 종목 수: {len(tickers)}")
    print(f"처음 20개: {tickers[:20]}")
