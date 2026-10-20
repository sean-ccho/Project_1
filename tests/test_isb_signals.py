"""웹 정렬용 ISB 라벨 (web_export/isb_signals.py) — 봉 완성 판정·최근 라벨 고르기·가격 정리 (네트워크 없음)."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web_export import isb_signals as S  # noqa: E402

ET = S.market_now().tzinfo


def _now(day: str, hh: int, mm: int = 0) -> datetime:
    y, m, d = map(int, day.split("-"))
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_week_complete_waits_for_friday_close() -> None:
    weeks = pd.DatetimeIndex(["2026-10-09", "2026-10-16", "2026-10-21"])  # 금 · 금 · 이번 주 수요일까지
    assert S.week_complete(weeks, _now("2026-10-21", 20)).tolist() == [True, True, False]
    # 금요일 장중엔 그 주가 안 끝났고, 16:30 뒤엔 끝남
    assert S.week_complete(weeks[:2], _now("2026-10-16", 12)).tolist() == [True, False]
    assert S.week_complete(weeks[:2], _now("2026-10-16", 16, 30)).tolist() == [True, True]


def test_drop_unfinished_day_only_during_session() -> None:
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.DatetimeIndex(["2026-10-16", "2026-10-19"]))
    assert len(S.drop_unfinished_day(df, _now("2026-10-19", 11))) == 1
    assert len(S.drop_unfinished_day(df, _now("2026-10-19", 17))) == 2
    assert len(S.drop_unfinished_day(df, _now("2026-10-20", 2, 30))) == 2


def _eng(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"ev_buy": False, "sig_type": "", "btm": False, "ev_topwarn": False, "ev_top": False,
                         "ev_exit": False}, index=idx)


def test_label_flags_match_chart_labels() -> None:
    eng = _eng(4)
    eng.loc[eng.index[0], ["ev_buy", "sig_type"]] = [True, "REV"]
    eng.loc[eng.index[1], ["ev_buy", "sig_type"]] = [True, "BO"]   # BUY·BO 는 REV 가 아니다
    eng.loc[eng.index[2], "ev_top"] = True                          # 보유 중 TRIM·TOP — ◆ TOP 아님
    eng.loc[eng.index[3], ["ev_topwarn", "btm", "ev_exit"]] = True
    f = S.label_flags(eng)
    assert f["rev"].tolist() == [True, False, False, False]
    assert f["top"].tolist() == [False, False, False, True]
    assert f["bottom"].tolist() == [False, False, False, True]
    assert f["exit"].tolist() == [False, False, False, True]


def test_recent_labels_window_and_unfinished_bar() -> None:
    eng = _eng(10)
    eng.loc[eng.index[[1, 6]], "ev_exit"] = True
    eng.loc[eng.index[2], "btm"] = True
    eng.loc[eng.index[9], "ev_topwarn"] = True  # 끝나지 않은 마지막 봉 — 차트에 아직 안 찍힘
    complete = np.array([True] * 9 + [False])
    out = S.recent_labels(eng, complete, window=5)
    # 마지막 끝난 봉 = 8 → 창은 4~8: EXIT 는 가장 최근(6) 하나, BOTTOM(2) 은 창 밖, TOP(9) 은 미완성
    assert out == {"exit": {"date": eng.index[6].strftime("%Y-%m-%d"), "ago": 2}}
    assert S.recent_labels(eng, np.zeros(10, dtype=bool), 5) == {}


def test_frame_rounds_prices_and_drops_empty_rows() -> None:
    idx = pd.DatetimeIndex(["2026-10-15", "2026-10-16", "2026-10-16"], tz="America/New_York")
    cols = pd.MultiIndex.from_product([["BRK-B"], ["Open", "High", "Low", "Close", "Volume"]])
    raw = pd.DataFrame([[25.459999084, 25.5, 25.4, 25.459999084, 100],
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                        [25.5, 25.6, 25.45, 25.55000019, 200]], index=idx, columns=cols)
    df = S._frame(raw, "BRK-B")
    assert df is not None and df.index.tz is None and len(df) == 2
    assert df["Close"].tolist() == [25.46, 25.55]
    assert S._frame(raw, "AAPL") is None
    assert S.yf_symbol("brk.b") == "BRK-B"


def test_isb_block_never_breaks_export(monkeypatch) -> None:
    import web_export.dashboard as wd

    def boom(tickers: list[str]) -> dict:
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(S, "isb_labels", boom)
    assert wd.isb_block(["AAPL"]) == {}
    assert wd.ticker_info(pd.Series({"티커": "AAPL"}))["isb"] is None  # build_dashboard 가 채우기 전 기본값
