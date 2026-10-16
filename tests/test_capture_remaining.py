"""전체차트 — 차트분석에 없는 분석 시트 종목만, 점수 순, WebP 로 (브라우저 없이)."""
import sys
import types
from pathlib import Path

import pandas as pd
from PIL import Image

import charts.capture_remaining as cr


def _png(path: Path, w: int = 3000, h: int = 2000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (240, 240, 240)).save(path)
    return path


def test_selection_and_webp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    d = Path("data/paper_trading")
    d.mkdir(parents=True)
    pd.DataFrame({"티커": ["AAA", "BBB", "SPY"], "매수적합도": [3.0, 8.0, 1.0]}).to_parquet(d / "sp500_ranked.parquet")
    pd.DataFrame({"티커": ["CCC", "DDD"], "매수적합도": [9.0, 2.0]}).to_parquet(d / "nasdaq_ranked.parquet")
    _png(Path("charts/screenshots_nasdaq/CCC/CCC_Daily_20261016_060000.png"), 10, 10)   # 차트분석이 이미 찍음
    assert cr.analysis_tickers() == ["CCC", "BBB", "AAA", "DDD", "SPY"]
    assert cr.captured_today() == {"CCC"}

    calls = []

    def fake_capture(ticker, output_dir, headless=True):
        calls.append(ticker)
        if ticker == "DDD":
            raise RuntimeError("boom")             # 한 종목 실패해도 계속
        return {"Daily": str(_png(Path(output_dir) / ticker / f"{ticker}_Daily_20261016_070000.png"))}

    monkeypatch.setitem(sys.modules, "charts.tradingview_capture",
                        types.SimpleNamespace(capture_multiple_timeframes=fake_capture))
    assert cr.main(["--budget-min", "5"]) == 0
    assert calls == ["BBB", "AAA", "DDD", "SPY"]                 # 점수 순, CCC 는 건너뜀
    webp = Path("charts/screenshots_all/BBB/BBB_Daily_20261016_070000.webp")
    assert webp.exists() and not webp.with_suffix(".png").exists()
    with Image.open(webp) as im:
        assert im.width == cr.WEBP_WIDTH                          # 3000px → 1920px
