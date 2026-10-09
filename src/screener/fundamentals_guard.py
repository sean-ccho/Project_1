"""yfinance 종목 정보(get_info) 실패 대응 — 요청 제한이면 쉬었다가 다시 받고, 끝내 못 받은 종목은 전날 스냅샷 값을 쓴다.

2026-10-09 실행에서 NASDAQ/NYSE 스캔의 종목 정보 885건이 전부 실패했다 (같은 실행의 S&P 500 은 정상, 그 전 12일은
실패 0%). yfinance 는 HTTP 429 를 YFRateLimitError 로 던지고 fetch_fundamental_snapshots 는 이를 빈 정보로 삼킨다.
그러면 섹터가 "Unknown"으로 원래 섹터까지 덮어쓰고, 예정 실적일이 비어 어닝 회피 필터를 그냥 통과하고,
CCS 펀더멘탈 가점(ROE·부채비율·기관·공매도, 최대 0.4)을 잃는다.

1) 실패가 평소(0~2%)보다 크게 많으면 쉬었다가 몇 종목으로 먼저 확인하고, 풀렸으면 실패한 종목만 다시 받는다.
2) 그래도 못 받은 종목은 전날 스냅샷(sp500_ranked·nasdaq_ranked)의 재무 값·섹터를 쓴다 (재무는 분기마다 바뀌고
   섹터는 거의 안 바뀐다). 예정 실적일이 빈 종목은 전날 값 중 오늘 이후인 것만 쓴다.

fundamentals.py·features.py 는 피처 캐시 해시 대상이라 고치지 않는다. 종목 정보를 여기서 미리 받아
compute_features_snapshot(fundamentals_df=...) 로 넘기므로, 실패가 없으면 결과는 compute_all_features(df) 와 같고
섹터·거래대금회전율·어닝 창 같은 파생 값도 채운 정보로 계산된다.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from screener.features import compute_features_snapshot
from screener.fundamentals import fetch_fundamental_snapshots

SNAPSHOT_DIR = Path("data/paper_trading")
SNAPSHOT_FILES = ("sp500_ranked.parquet", "nasdaq_ranked.parquet")
MASS_FAIL_RATIO = 0.2    # 실패 비율이 이보다 크면 요청 제한으로 본다 (평소 0~2%)
RETRY_WAITS = (60, 300)  # 다시 받기 전 쉬는 초 (최대 6분)
PROBE_SIZE = 3           # 쉰 뒤 이만큼 먼저 받아 보고, 하나라도 받히면 실패 종목 전부를 다시 받는다

Fetch = Callable[[list[str]], pd.DataFrame]


def failed_tickers(fund: pd.DataFrame) -> list[str]:
    """정보를 하나도 못 받은 종목 — 빈 정보면 섹터가 "Unknown"이고 시가총액이 없다."""
    if fund is None or fund.empty or not {"티커", "fund_sector", "fund_market_cap"} <= set(fund.columns):
        return []
    empty = fund["fund_sector"].eq("Unknown") & fund["fund_market_cap"].isna()
    return fund.loc[empty, "티커"].astype(str).tolist()


def fetch_with_retry(
    tickers: Sequence[str],
    label: str = "",
    *,
    fetch: Optional[Fetch] = None,
    waits: Sequence[float] = RETRY_WAITS,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """fetch_fundamental_snapshots 와 같은 표. 대량 실패면 쉬었다가 실패한 종목만 다시 받아 바꿔 끼운다."""
    fetch = fetch or fetch_fundamental_snapshots
    tickers = list(tickers)
    fund = fetch(tickers)
    for wait in waits:
        failed = failed_tickers(fund)
        if len(failed) <= MASS_FAIL_RATIO * len(tickers):
            return fund
        print(f"{label} ⚠️ 종목 정보 {len(failed)}/{len(tickers)}개 실패 (Yahoo 요청 제한 추정) — {wait:.0f}초 쉬고 다시 받기")
        sleep(wait)
        probe = fetch(failed[:PROBE_SIZE])
        if probe.empty or len(failed_tickers(probe)) == len(probe):
            print(f"{label} 아직 막혀 있음")
            continue
        again = fetch(failed)
        ok = again[~again["티커"].isin(failed_tickers(again))]
        if not ok.empty:
            fund = pd.concat([fund[~fund["티커"].isin(ok["티커"])], ok], ignore_index=True)
        print(f"{label} 다시 받기: {len(ok)}/{len(failed)}개 성공")
    failed = failed_tickers(fund)
    if len(failed) > MASS_FAIL_RATIO * len(tickers):
        print(f"{label} ⚠️ 종목 정보 {len(failed)}/{len(tickers)}개는 끝내 못 받음 — 전날 스냅샷 값으로 대체")
    return fund


def _previous_snapshot(snapshot_dir: Path) -> pd.DataFrame:
    """전날 스냅샷 두 개를 티커 기준으로 합친 표 (같은 티커는 뒤 파일 값)."""
    frames = []
    for name in SNAPSHOT_FILES:
        path = snapshot_dir / name
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if "티커" in frame.columns:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    prev = pd.concat(frames, ignore_index=True).drop_duplicates("티커", keep="last")
    return prev.set_index(prev["티커"].astype(str))


def fill_from_previous_snapshot(
    fund: pd.DataFrame,
    label: str = "",
    *,
    snapshot_dir: Path = SNAPSHOT_DIR,
    today: Optional[date] = None,
) -> pd.DataFrame:
    """못 받은 종목은 전날 스냅샷의 재무 값·섹터로, 빈 예정 실적일은 전날 값(오늘 이후만)으로 채운 사본."""
    if fund is None or fund.empty or "티커" not in fund.columns:
        return fund
    prev = _previous_snapshot(snapshot_dir)
    if prev.empty:
        return fund
    out = fund.copy()
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    tick = out["티커"].astype(str)

    failed = tick.isin(failed_tickers(out)) & tick.isin(prev.index)
    n_fund = int(failed.sum())
    if n_fund:
        src = prev.loc[tick[failed]]
        cols = [c for c in out.columns if c.startswith("fund_") and c != "fund_sector" and c in prev.columns]
        out.loc[failed, cols] = src[cols].to_numpy()
        if "섹터" in prev.columns:
            known = src["섹터"].notna() & (src["섹터"] != "Unknown")
            out.loc[failed, "fund_sector"] = src["섹터"].where(known, "Unknown").to_numpy()

    n_earn = 0
    if "next_earnings_date" in prev.columns and "days_to_next_earnings" in out.columns:
        dates = pd.to_datetime(prev["next_earnings_date"].replace("", pd.NA), errors="coerce").dropna().dt.date
        upcoming = dates[dates >= today]
        missing = out["days_to_next_earnings"].isna() & tick.isin(upcoming.index)
        n_earn = int(missing.sum())
        if n_earn:
            nxt = tick[missing].map(upcoming)
            out.loc[missing, "next_earnings_date"] = [d.isoformat() for d in nxt]
            out.loc[missing, "days_to_next_earnings"] = [float((d - today).days) for d in nxt]

    if n_fund or n_earn:
        print(f"{label} 전날 스냅샷으로 채움: 재무·섹터 {n_fund}개 · 실적일 {n_earn}개")
    return out


def compute_all_features_guarded(
    df: pd.DataFrame,
    label: str = "",
    *,
    fetch: Optional[Fetch] = None,
    sleep: Callable[[float], None] = time.sleep,
    snapshot_dir: Path = SNAPSHOT_DIR,
    today: Optional[date] = None,
) -> pd.DataFrame:
    """compute_all_features(df) 와 같은 피처 표 — 종목 정보만 재시도·전날 스냅샷 보충을 거쳐 미리 받아 넘긴다."""
    price_map = {ticker: df[ticker].dropna(how="all") for ticker in df.columns.levels[0]}
    tickers = [t for t, frame in price_map.items() if not frame.empty]
    fund = fetch_with_retry(tickers, label, fetch=fetch, sleep=sleep)
    fund = fill_from_previous_snapshot(fund, label, snapshot_dir=snapshot_dir, today=today)
    return compute_features_snapshot(price_map, fundamentals_df=fund)
