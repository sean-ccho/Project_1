"""백테스트용 과거 실적 발표일 (SEC 8-K Item 2.02) → 그날 기준 '다음 실적까지 남은 달력일'.

실거래는 yfinance 의 예정 실적일로 days_to_next_earnings 를 채워 "실적 3일 이내 매수 금지"를 건다.
백테스트에는 그 값이 없어서 필터가 늘 통과였다. 과거 실제 발표일로 같은 값을 만들어 넣는다.

- 원본: data/research/sec_earnings_dates.parquet (scripts/fetch_sec_data.py → build_sec_data.py, ≤ 2025-09-30)
- 발표일 = 접수 시각의 미국 동부 날짜 (장 마감 후 발표도 그날). 실제 예정일은 보통 몇 주 전에 공지되므로
  "그날 이후 첫 발표일"을 그날 알 수 있었다고 본다.
- 파일이 없거나 종목이 없으면 값을 비워 둔다 (기존 동작 = 필터 통과).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/research/sec_earnings_dates.parquet")


CLUSTER_DAYS = 45


def scheduled_only(dates: pd.DataFrame, cluster_days: int = CLUSTER_DAYS) -> pd.DataFrame:
    """예정 실적 발표만 남기는 근사: 같은 회사 발표가 cluster_days 안에 몰리면 마지막 것만 남긴다.

    예정 외 발표(실적 경고·잠정치)는 보통 정규 발표 몇 주 전에 나온다. 그건 미리 알 수 없어서
    백테스트 필터에 쓰면 미래 정보가 된다. 정규 발표는 약 91일 간격이라 거의 지워지지 않는다.
    """
    d = dates.assign(day=pd.to_datetime(dates["accepted_et"]).dt.normalize()).sort_values(["티커", "day"])
    nxt = d.groupby("티커")["day"].shift(-1)
    keep = nxt.isna() | ((nxt - d["day"]).dt.days > cluster_days)
    return d.loc[keep, ["티커", "accepted_et"]].reset_index(drop=True)


class EarningsCalendar:
    """티커별 실적 발표일 목록으로 '다음 발표까지 남은 일수'를 돌려준다."""

    def __init__(self, dates: pd.DataFrame | None) -> None:
        self._by_ticker: dict[str, np.ndarray] = {}
        if dates is None or dates.empty:
            return
        d = dates.assign(day=pd.to_datetime(dates["accepted_et"]).dt.normalize())
        for ticker, g in d.groupby("티커"):
            self._by_ticker[str(ticker)] = np.sort(g["day"].to_numpy(dtype="datetime64[D]"))

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH, mode: str | None = None) -> "EarningsCalendar":
        """mode: scheduled(기본, 예정 발표만 — 실거래처럼 미리 알 수 있던 일정) | all(8-K 전부, 예정 외 경고까지 미리 아는 셈) | off(필터 끔).

        환경변수 BACKTEST_EARNINGS_CAL 로도 고른다 (진단용, 해시 대상 파일이 아니라 피처 캐시를 그대로 쓴다).
        """
        mode = (mode or os.environ.get("BACKTEST_EARNINGS_CAL", "scheduled")).lower()
        p = Path(path)
        if mode == "off":
            print("[백테스트] 실적일 달력 꺼짐 (BACKTEST_EARNINGS_CAL=off)")
            return cls(None)
        if not p.exists():
            print(f"[백테스트] 실적일 파일 없음 ({p}) → 어닝 회피 필터 미적용 (기존 동작)")
            return cls(None)
        dates = pd.read_parquet(p, columns=["티커", "accepted_et"])
        if mode == "scheduled":
            dates = scheduled_only(dates)
        cal = cls(dates)
        print(f"[백테스트] 실적일 로드: {len(cal)}개 종목, {len(dates):,}건 (mode={mode}, {p})")
        return cal

    def __len__(self) -> int:
        return len(self._by_ticker)

    def days_to_next(self, ticker: str, today: pd.Timestamp) -> float:
        """today 당일 포함 다음 발표까지 남은 달력일. 모르면 NaN."""
        arr = self._by_ticker.get(str(ticker))
        if arr is None or arr.size == 0:
            return float("nan")
        t = np.datetime64(pd.Timestamp(today).normalize().date(), "D")
        i = int(np.searchsorted(arr, t, side="left"))
        if i >= arr.size:
            return float("nan")
        return float((arr[i] - t).astype(int))

    def annotate(self, ranked_df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
        """ranked_df 에 days_to_next_earnings 열을 채운 사본 (달력이 비면 그대로)."""
        if not self._by_ticker or ranked_df.empty or "티커" not in ranked_df.columns:
            return ranked_df
        out = ranked_df.copy()
        out["days_to_next_earnings"] = [self.days_to_next(t, today) for t in out["티커"]]
        return out
