"""NYSE/NASDAQ 전 종목 티커 리스트를 가져오는 유틸리티.

나스닥 트레이더 FTP 서버에서 상장 종목 목록을 다운로드하여
보통주(Common Stock)만 필터링한다.
"""

from __future__ import annotations

import io
import re
import time
from ftplib import FTP
from typing import List

import pandas as pd


FTP_HOST = "ftp.nasdaqtrader.com"
FTP_DIR = "SymbolDirectory"
NASDAQ_FILE = "nasdaqlisted.txt"
OTHER_FILE = "otherlisted.txt"

# 1단계 필터 설정
_BATCH_SIZE = 150          # yfinance 배치 크기
_BATCH_DELAY = 1.5         # 배치 간 대기 시간(초)
_MAX_RETRIES = 3           # rate limit 시 재시도 횟수
_RETRY_BASE_DELAY = 30     # 재시도 기본 대기 시간(초) — 30/60/120

# 비표준 심볼 패턴 (보통주가 아닌 것들)
#   $ 포함       → 우선주         (BAC$K, ALL$H)
#   .W / .U / .R → 워런트/유닛/권리 (ACHR.W, ALUB.U, CELG.R)
#   .A / .B      → 듀얼 클래스     (BRK.A, BF.B)  ※ Yahoo Finance에서 오류 발생 시 제거
_NON_COMMON_RE = re.compile(
    r"[$]"            # 우선주
    r"|\.W$"          # 워런트 (NYSE 형식)
    r"|\.U$"          # 유닛 (NYSE 형식)
    r"|\.R$"          # 권리 (NYSE 형식)
)

# 나스닥 관례: 5글자 이상 + W/U/R로 끝나는 티커 → 워런트/유닛/권리
# 예: CGCTW(워런트), CCIXU(유닛), AIIAR(권리)
# 단, 정상 4글자 이하 티커가 W/U/R로 끝나는 건 유지 (예: SNOW, ROKU)
_NASDAQ_WARRANT_RE = re.compile(r"^[A-Z]{4,}[WUR]$")


def _is_common_stock(ticker: str) -> bool:
    """보통주 여부를 판단한다. 우선주/워런트/유닛/권리를 제거한다."""
    if _NON_COMMON_RE.search(ticker):
        return False
    if _NASDAQ_WARRANT_RE.match(ticker):
        return False
    # Yahoo Finance 호환성: 점(.)이 포함된 심볼은 종종 에러가 발생하므로 제거
    if "." in ticker:
        return False
    return True


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

    우선주($), 워런트(.W), 유닛(.U), 권리(.R) 등
    비표준 심볼은 자동으로 제외된다.

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
    all_tickers_raw = sorted(combined["ticker"].unique().tolist())
    total_raw = len(all_tickers_raw)

    # 비표준 심볼 필터링 (우선주, 워런트, 유닛, 권리 및 점 포함 심볼 제거)
    tickers = [t for t in all_tickers_raw if _is_common_stock(t)]
    removed = total_raw - len(tickers)

    print(
        f"[ticker_fetcher] 총 {total_raw}개 종목 로드 → "
        f"비표준 심볼 {removed}개 제거 → {len(tickers)}개 보통주"
    )
    return tickers


def _download_batch_with_retry(
    batch: List[str],
    batch_num: int,
    total_batches: int,
    *,
    period: str = "1mo",
) -> pd.DataFrame | None:
    """yfinance 배치 다운로드를 rate limit 재시도와 함께 수행한다."""
    import yfinance as yf

    batch_str = " ".join(batch)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            data = yf.download(
                batch_str,
                period=period,
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            return data if not data.empty else None

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "Rate" in err_str or "Too Many" in err_str

            if is_rate_limit and attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[ticker_fetcher] 배치 {batch_num}/{total_batches} "
                    f"rate limit → {wait}초 대기 후 재시도 "
                    f"({attempt}/{_MAX_RETRIES})..."
                )
                time.sleep(wait)
            else:
                print(
                    f"[ticker_fetcher] 배치 {batch_num}/{total_batches} "
                    f"처리 실패: {e}"
                )
                return None

    return None


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
    passed: List[str] = []
    total = len(tickers)
    skipped_batches = 0
    skipped_tickers = 0

    for i in range(0, total, _BATCH_SIZE):
        batch = tickers[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        total_batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE
        print(
            f"[ticker_fetcher] 1단계 필터링 배치 {batch_num}/{total_batches} "
            f"({len(batch)}개 종목)..."
        )

        data = _download_batch_with_retry(
            batch, batch_num, total_batches, period="1mo"
        )
        if data is None:
            skipped_batches += 1
            skipped_tickers += len(batch)
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

        # API 부하 방지
        if i + _BATCH_SIZE < total:
            time.sleep(_BATCH_DELAY)

    skip_msg = (
        f" | 배치 실패: {skipped_batches}개 ({skipped_tickers}개 종목 미처리)"
        if skipped_batches > 0
        else ""
    )
    print(
        f"[ticker_fetcher] 1단계 필터링 완료: {total}개 → {len(passed)}개 "
        f"(가격 ≥ ${min_price}, 평균 거래량 ≥ {min_avg_volume:,}){skip_msg}"
    )
    return passed


if __name__ == "__main__":
    tickers = fetch_all_tickers()
    print(f"전체 종목 수: {len(tickers)}")
    print(f"처음 20개: {tickers[:20]}")
