"""yfinance 종목 정보(get_info) 실패 대응 — 요청 제한이면 쉬었다가 다시 받고, 끝내 못 받은 종목은 전날 스냅샷 값을 쓴다.

2026-10-09 실행에서 NASDAQ/NYSE 스캔의 종목 정보 885건이 전부 실패했다 (같은 실행의 S&P 500 은 정상, 그 전 12일은
실패 0%). yfinance 는 HTTP 429 를 YFRateLimitError 로 던지고 fetch_fundamental_snapshots 는 이를 빈 정보로 삼킨다.
그러면 섹터가 "Unknown"으로 원래 섹터까지 덮어쓰고, 예정 실적일이 비어 어닝 회피 필터를 그냥 통과하고,
CCS 펀더멘탈 가점(ROE·부채비율·기관·공매도, 최대 0.4)을 잃는다.

1) 50개씩 받다가 한 묶음이 거의 다 실패하고 대형주 확인도 실패하면 막힌 것으로 보고 바로 멈춘다. 쉬었다가
   대형주로 먼저 확인하고, 풀렸으면 남은 종목을 이어서 받는다.
   2026-10-09·10-11 실행 모두 막힌 채로 880여 건을 끝까지 요청했다 (종목마다 쿠키·crumb 재발급까지 4번쯤, 3~6분).
   10-09 는 6분 넘게 계속 요청하는 동안 안 풀렸고, 10-11 은 멈추고 6분 쉬니 풀렸다.
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
from typing import Any, Callable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from screener.features import compute_features_snapshot
from screener.fundamentals import fetch_fundamental_snapshots

SNAPSHOT_DIR = Path("data/paper_trading")
SNAPSHOT_FILES = ("sp500_ranked.parquet", "nasdaq_ranked.parquet")
CHUNK_SIZE = 50                   # 이만큼씩 받으며 막혔는지 본다
BLOCK_RATIO = 0.8                 # 한 묶음에서 이 비율 넘게 실패하면 막혔는지 대형주로 확인 (평소 실패 0~2%)
RETRY_WAITS = (60, 120, 300, 600)  # 막혔을 때 다시 확인하기 전 쉬는 초 (누적 최대 약 18분)
PROBE_TICKERS = ("AAPL", "MSFT", "JPM")  # 막혔는지 확인용 — 평소 정보가 늘 받히는 대형주. 하나라도 받히면 풀린 것
TEXT_COLS = ("fund_industry", "fund_summary")  # 웹 표시 전용 글자 칸 (업종 · 회사 소개)

Fetch = Callable[[list[str]], pd.DataFrame]


def failed_tickers(fund: pd.DataFrame) -> list[str]:
    """정보를 하나도 못 받은 종목 — 빈 정보면 섹터가 "Unknown"이고 시가총액이 없다."""
    if fund is None or fund.empty or not {"티커", "fund_sector", "fund_market_cap"} <= set(fund.columns):
        return []
    empty = fund["fund_sector"].eq("Unknown") & fund["fund_market_cap"].isna()
    return fund.loc[empty, "티커"].astype(str).tolist()


def _is_blocked(fetch: Fetch) -> bool:
    """대형주 확인용 종목이 하나도 안 받히면 True."""
    probe = fetch(list(PROBE_TICKERS))
    return probe.empty or len(failed_tickers(probe)) == len(probe)


def _empty_rows(tickers: list[str], like: pd.DataFrame) -> pd.DataFrame:
    """요청하지 못한 종목의 빈 정보 행 — fetch 가 실패한 종목에 돌려주는 모양과 같다 (섹터 Unknown, 실적일 "", 나머지 NaN)."""
    rows = pd.DataFrame({col: [np.nan] * len(tickers) for col in like.columns})
    rows["티커"] = tickers
    if "fund_sector" in rows.columns:
        rows["fund_sector"] = "Unknown"
    if "next_earnings_date" in rows.columns:
        rows["next_earnings_date"] = ""
    return rows


def fetch_with_retry(
    tickers: Sequence[str],
    label: str = "",
    *,
    fetch: Optional[Fetch] = None,
    waits: Sequence[float] = RETRY_WAITS,
    sleep: Callable[[float], None] = time.sleep,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """fetch_fundamental_snapshots 와 같은 표. 묶음으로 받다가 막히면 멈추고, 쉬었다 풀리면 남은 종목을 이어서 받는다."""
    fetch = fetch or fetch_fundamental_snapshots
    tickers = list(tickers)
    todo = tickers
    frames: list[pd.DataFrame] = []
    waits_left = iter(waits)
    blocked_chunk: Optional[pd.DataFrame] = None  # 막혔을 때 받은 묶음 (빈 정보라 쓰지 않고 같은 묶음부터 다시)
    was_blocked = False
    while todo:
        if blocked_chunk is not None:
            wait = next(waits_left, None)
            if wait is None:
                break
            print(f"{label} ⚠️ 종목 정보 막힘 (Yahoo 요청 제한 추정) — {len(todo)}/{len(tickers)}개 남음, {wait:.0f}초 쉬고 확인")
            sleep(wait)
            if _is_blocked(fetch):
                print(f"{label} 아직 막혀 있음")
                continue
            print(f"{label} 풀림 — 남은 {len(todo)}개 이어서 받기")
            blocked_chunk = None
        chunk = todo[:chunk_size]
        got = fetch(chunk)
        if len(failed_tickers(got)) > BLOCK_RATIO * len(chunk) and _is_blocked(fetch):
            blocked_chunk, was_blocked = got, True
            continue
        frames.append(got)
        todo = todo[chunk_size:]

    if todo:
        frames.append(_empty_rows(todo, blocked_chunk))
    fund = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if todo:
        print(f"{label} ⚠️ 종목 정보 {len(todo)}/{len(tickers)}개는 끝내 못 받음 — 전날 스냅샷 값으로 대체")
    elif was_blocked:
        print(f"{label} 다시 받기 완료: {len(tickers) - len(failed_tickers(fund))}/{len(tickers)}개 받음")
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
        cols = [c for c in out.columns if c.startswith("fund_") and c != "fund_sector" and c not in TEXT_COLS
                and c in prev.columns]
        out.loc[failed, cols] = src[cols].to_numpy()
        # 글자 칸은 따로 — 숫자 칸과 한 배열로 넣으면 숫자 칸이 object 가 된다. 실전 스냅샷은 빈 값이 'nan' 글자로 저장돼 있다
        for c in [c for c in TEXT_COLS if c in out.columns and c in prev.columns]:
            out[c] = out[c].astype(object)
            out.loc[failed, c] = [None if pd.isna(v) or str(v) in ("nan", "None", "") else str(v) for v in src[c]]
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
    chart_map: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """compute_all_features(df) 와 같은 피처 표 — 종목 정보만 재시도·전날 스냅샷 보충을 거쳐 미리 받아 넘긴다.

    chart_map: 종목 → 차트 가격 (data.fetch.fetch_chart_prices — 패턴 · 교차 라벨용, 없으면 df 로 만든다).
    """
    price_map = {ticker: df[ticker].dropna(how="all") for ticker in df.columns.levels[0]}
    tickers = [t for t, frame in price_map.items() if not frame.empty]
    fund = fetch_with_retry(tickers, label, fetch=fetch, sleep=sleep)
    fund = fill_from_previous_snapshot(fund, label, snapshot_dir=snapshot_dir, today=today)
    return compute_features_snapshot(price_map, fundamentals_df=fund, chart_map=chart_map)
