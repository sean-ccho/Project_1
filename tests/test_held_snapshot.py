"""보유 · 고정 종목 보충 표 (main.save_held_snapshot · runner.save_ranked_snapshot 'held') — 네트워크 없음."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main  # noqa: E402
import paper_trading.runner as runner  # noqa: E402


def test_saves_only_the_separately_analyzed_tickers(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(runner, "save_ranked_snapshot", lambda df, source: calls.append((df["티커"].tolist(), source)) or True)
    assert not main.save_held_snapshot(None, ["MORN"])
    assert not main.save_held_snapshot(pd.DataFrame(), ["MORN"])
    # 따로 분석이 중간에 끝나 S&P 표가 넘어온 경우 — 보충 파일로 쓰지 않는다
    assert not main.save_held_snapshot(pd.DataFrame({"티커": ["AAPL", "MSFT", "MORN"]}), ["MORN", "NBM.V"])
    assert calls == []
    assert main.save_held_snapshot(pd.DataFrame({"티커": ["MORN", "NBM.V"]}), ["NBM.V", "MORN"])
    assert calls == [(["MORN", "NBM.V"], "held")]


def test_runner_held_path_and_unknown_source(monkeypatch, tmp_path) -> None:
    """'held' 는 따로 파일, 모르는 이름은 저장하지 않는다 (예전엔 sp500 이 아니면 나스닥 파일을 덮어썼다)."""
    monkeypatch.setattr(runner, "_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(runner, "_NASDAQ_SNAPSHOT", tmp_path / "nasdaq_ranked.parquet")
    monkeypatch.setattr(runner, "_HELD_SNAPSHOT", tmp_path / "held_ranked.parquet")
    import screener.sec_filings as sf
    monkeypatch.setattr(sf, "fetch_flags", lambda tickers, earnings=None: {})
    import screener.short_squeeze as sq
    monkeypatch.setattr(sq, "short_data", lambda: None)
    df = pd.DataFrame({"티커": ["MORN"], "현재가격": [250.0], "fund_roe": [0.3]})
    assert runner.save_ranked_snapshot(df, "held")
    saved = pd.read_parquet(tmp_path / "held_ranked.parquet")
    assert saved["티커"].tolist() == ["MORN"] and "[헬스체크]" in saved.columns
    assert not runner.save_ranked_snapshot(df, "typo")
    assert not (tmp_path / "nasdaq_ranked.parquet").exists()
