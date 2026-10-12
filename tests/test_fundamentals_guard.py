"""종목 정보(get_info) 대량 실패 대응 테스트 — 재시도, 전날 스냅샷 보충, compute_all_features 와 같은 결과."""
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import screener.features as features_mod  # noqa: E402
from screener.fundamentals_guard import (  # noqa: E402
    compute_all_features_guarded,
    failed_tickers,
    fetch_with_retry,
    fill_from_previous_snapshot,
)

TODAY = date(2026, 10, 9)


def _record(ticker: str, ok: bool = True) -> dict:
    """fetch_fundamental_snapshots 한 행 — ok=False 면 빈 정보(요청 실패) 모양."""
    if not ok:
        return {"티커": ticker, "fund_sector": "Unknown", "fund_roe": np.nan, "fund_debt_to_equity": np.nan,
                "fund_market_cap": np.nan, "fund_institutional_holders_pct": np.nan,
                "next_earnings_date": "", "days_to_next_earnings": np.nan}
    return {"티커": ticker, "fund_sector": "Technology", "fund_roe": 0.2, "fund_debt_to_equity": 50.0,
            "fund_market_cap": 1e10, "fund_institutional_holders_pct": 0.7,
            "next_earnings_date": "2026-10-20", "days_to_next_earnings": 11.0}


class FakeYahoo:
    """fail_calls 에 든 번째 호출(1부터, 대형주 확인 포함)은 전부 실패하고 나머지는 받히는 가짜 fetch."""

    def __init__(self, fail_calls):
        self.fail_calls = set(fail_calls)
        self.calls: list[list[str]] = []

    def __call__(self, tickers):
        self.calls.append(list(tickers))
        ok = len(self.calls) not in self.fail_calls
        return pd.DataFrame([_record(t, ok) for t in tickers])


TICKERS = [f"T{i}" for i in range(120)]


def test_stops_at_first_blocked_chunk_and_resumes_after_wait(capsys):
    # 처음부터 막힘: 첫 묶음(50) 실패 → 대형주 확인 실패 → 60초 뒤 확인(아직) → 120초 뒤 확인(풀림) → 이어서 받기
    yahoo = FakeYahoo(fail_calls={1, 2, 3})
    slept: list[float] = []
    fund = fetch_with_retry(TICKERS, "[테스트]", fetch=yahoo, sleep=slept.append)
    assert slept == [60, 120]
    assert [len(c) for c in yahoo.calls] == [50, 3, 3, 3, 50, 50, 20]  # 막힌 동안엔 대형주 3개만 요청
    assert yahoo.calls[1] == ["AAPL", "MSFT", "JPM"]
    assert fund["티커"].tolist() == TICKERS and failed_tickers(fund) == []
    out = capsys.readouterr().out
    assert "120/120개 남음, 60초" in out and "아직 막혀 있음" in out and "풀림" in out
    assert "다시 받기 완료: 120/120개 받음" in out


def test_block_in_the_middle_keeps_chunks_already_received():
    # 둘째 묶음에서 막힘 → 첫 묶음은 그대로 두고 둘째 묶음부터 이어서
    yahoo = FakeYahoo(fail_calls={2, 3})
    slept: list[float] = []
    fund = fetch_with_retry(TICKERS, fetch=yahoo, sleep=slept.append)
    assert slept == [60]
    assert [len(c) for c in yahoo.calls] == [50, 50, 3, 3, 50, 20]
    assert fund["티커"].tolist() == TICKERS and failed_tickers(fund) == []


def test_no_wait_when_failures_are_normal():
    calls: list[list[str]] = []

    def fetch(tickers):
        calls.append(list(tickers))
        return pd.DataFrame([_record(t, ok=t != "T0") for t in tickers])

    slept: list[float] = []
    fund = fetch_with_retry(TICKERS, fetch=fetch, sleep=slept.append)
    assert slept == [] and [len(c) for c in calls] == [50, 50, 20]  # 대형주 확인도 안 한다
    expected = pd.DataFrame([_record(t, ok=t != "T0") for t in TICKERS])
    pd.testing.assert_frame_equal(fund, expected)  # 묶음으로 받아도 한 번에 받은 표와 같다
    assert failed_tickers(fund) == ["T0"]


def test_mostly_empty_chunk_is_kept_when_yahoo_answers():
    # 정보가 원래 없는 종목만 모인 묶음 — 대형주는 받히니 막힌 게 아니다. 쉬지 않고 그대로 쓴다
    calls: list[list[str]] = []

    def fetch(tickers):
        calls.append(list(tickers))
        return pd.DataFrame([_record(t, ok=not t.startswith("X")) for t in tickers])

    slept: list[float] = []
    fund = fetch_with_retry(["X1", "X2", "X3"], fetch=fetch, sleep=slept.append)
    assert slept == [] and [len(c) for c in calls] == [3, 3]
    assert failed_tickers(fund) == ["X1", "X2", "X3"]


def test_gives_up_after_waits_without_hammering(capsys):
    yahoo = FakeYahoo(fail_calls=range(1, 100))
    slept: list[float] = []
    fund = fetch_with_retry(TICKERS, "[테스트]", fetch=yahoo, sleep=slept.append)
    assert slept == [60, 120, 300, 600]
    assert [len(c) for c in yahoo.calls] == [50, 3, 3, 3, 3, 3]  # 남은 70개는 요청하지 않는다
    # 못 받은 종목도 fetch 가 실패에 돌려주는 모양 그대로 한 행씩 (전날 스냅샷 보충이 이 행을 채운다)
    assert fund["티커"].tolist() == TICKERS and failed_tickers(fund) == TICKERS
    assert list(fund.columns) == list(_record("A").keys())
    assert (fund["next_earnings_date"] == "").all() and fund["days_to_next_earnings"].isna().all()
    assert "120/120개는 끝내 못 받음" in capsys.readouterr().out


def _write_snapshots(tmp_path: Path) -> Path:
    pd.DataFrame({
        "티커": ["AAA", "BBB", "CCC"], "섹터": ["Technology", "Unknown", "Healthcare"],
        "fund_sector": ["Technology", "Unknown", "Healthcare"], "fund_roe": [0.3, 0.1, 0.05],
        "fund_debt_to_equity": [40.0, 80.0, 200.0], "fund_market_cap": [2e9, 3e9, 4e9],
        "fund_institutional_holders_pct": [0.8, 0.5, 0.4],
        "next_earnings_date": ["2026-10-21", "2026-10-30", "2026-10-01"],
    }).to_parquet(tmp_path / "nasdaq_ranked.parquet")
    pd.DataFrame({
        "티커": ["DDD"], "섹터": ["Energy"], "fund_sector": ["Energy"], "fund_roe": [0.12],
        "fund_debt_to_equity": [90.0], "fund_market_cap": [5e9], "fund_institutional_holders_pct": [0.65],
        "next_earnings_date": [""],
    }).to_parquet(tmp_path / "sp500_ranked.parquet")
    return tmp_path


def test_fill_failed_rows_from_previous_snapshot(tmp_path, capsys):
    fund = pd.DataFrame([_record("AAA", False), _record("BBB", False), _record("CCC", False),
                         _record("DDD", False), _record("NEW", False), _record("OKK")])
    out = fill_from_previous_snapshot(fund, "[테스트]", snapshot_dir=_write_snapshots(tmp_path), today=TODAY)
    by = out.set_index("티커")
    # 재무 값은 전날 그대로, 섹터는 알려진 값만 (BBB 는 전날도 Unknown, 새 종목은 그대로)
    assert by.loc["AAA", "fund_roe"] == 0.3 and by.loc["DDD", "fund_market_cap"] == 5e9
    assert by["fund_sector"].tolist() == ["Technology", "Unknown", "Healthcare", "Energy", "Unknown", "Technology"]
    # 실적일은 오늘 이후만: AAA 10/21 → 12일, BBB 10/30 → 21일, CCC 10/01 은 지나서 비움, DDD 는 전날도 없음
    assert by.loc["AAA", "days_to_next_earnings"] == 12.0 and by.loc["BBB", "next_earnings_date"] == "2026-10-30"
    assert math.isnan(by.loc["CCC", "days_to_next_earnings"]) and math.isnan(by.loc["DDD", "days_to_next_earnings"])
    # 받은 종목(OKK)은 건드리지 않고, 원본도 바꾸지 않는다
    assert math.isnan(by.loc["NEW", "fund_roe"]) and by.loc["OKK", "days_to_next_earnings"] == 11.0
    assert "재무·섹터 4개 · 실적일 2개" in capsys.readouterr().out
    assert fund["fund_sector"].iloc[0] == "Unknown"


def _ohlcv(tickers: list[str], days: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2025-07-01", periods=days)
    frames = {}
    for t in tickers:
        close = 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, days)))
        frames[t] = pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.005, days)), "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, days).astype(float),
        }, index=idx)
    return pd.concat(frames, axis=1)


def test_guarded_matches_compute_all_features_when_info_ok(tmp_path, monkeypatch):
    df = _ohlcv(["AAA", "BBB", "CCC", "SHORT"])
    df.loc[df.index[:-50], "SHORT"] = np.nan  # 이력이 짧아 피처 계산을 건너뛰는 종목

    def fetch(tickers):
        return pd.DataFrame([_record(t) for t in tickers])

    monkeypatch.setattr(features_mod, "fetch_fundamental_snapshots", fetch)
    expected = features_mod.compute_all_features(df)
    got = compute_all_features_guarded(df, fetch=fetch, sleep=lambda s: None, snapshot_dir=tmp_path, today=TODAY)
    assert len(expected) == 3
    pd.testing.assert_frame_equal(got, expected)
