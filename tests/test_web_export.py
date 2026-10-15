"""웹사이트용 dashboard.json 내보내기 — 파싱·요약·차트 목록 (네트워크·git 없음)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import web_export.dashboard as wd


def test_parse_score_and_health() -> None:
    assert wd.parse_score("★★★★★ (11.5)") == 11.5
    assert wd.parse_score("") is None
    assert wd.parse_health("100 (A+) · 매출+48% 마진+25% 이익+86%") == {
        "score": 100, "grade": "A+", "detail": "매출+48% 마진+25% 이익+86%"}
    assert wd.parse_health("55 (D)") == {"score": 55, "grade": "D", "detail": None}
    assert wd.parse_health(float("nan")) is None


def test_label_strips_numbering_and_emoji() -> None:
    assert wd._label("1. 매수 후보") == "매수 후보"
    assert wd._label("📈 모멘텀") == "모멘텀"
    assert wd._label("—") is None
    assert wd._patterns("이중천장, 골든크로스") == ["이중천장", "골든크로스"]


def test_ticker_info_handles_missing_values() -> None:
    row = pd.Series({"회사": "Figma, Inc.", "현재가격": 21.76, "전략구분": "—", "매수적합도_표시": "★★★★☆ (4.5)",
                     "판단": "5. 관망 약세", "[헬스체크]": "55 (D) · 매출+48%", "RSI": float("nan"),
                     "next_earnings_date": pd.NaT, "일봉패턴": None, "market": "sp500"})
    info = wd.ticker_info(row)
    assert info["strategy"] is None and info["verdict"] == "관망 약세" and info["score"] == 4.5
    assert info["rsi"] is None and info["earnings_date"] is None and info["patterns"]["daily"] == []
    json.dumps(info, allow_nan=False)  # NaN 이 남아 있으면 실패


def test_trade_summary_groups_exit_reasons() -> None:
    trades = [
        {"ticker": "A", "entry_date": "2026-04-02", "exit_date": "2026-04-10", "return_pct": 0.2, "exit_reason": "목표가(+20%)"},
        {"ticker": "B", "entry_date": "2026-04-05", "exit_date": "2026-04-20", "return_pct": -0.05, "exit_reason": "트레일링(-5%)"},
        {"ticker": "C", "entry_date": "2026-04-08", "exit_date": "2026-04-15", "return_pct": -0.03, "exit_reason": "트레일링(-3%)"},
    ]
    s = wd.trade_summary(trades)
    assert s["count"] == 3 and s["wins"] == 1 and s["first_entry"] == "2026-04-02"
    assert [r["reason"] for r in s["by_reason"]] == ["목표가", "트레일링"]
    assert s["by_reason"][1]["count"] == 2
    assert [h["return"] for h in s["history"]] == [0.2, -0.03, -0.05]  # 청산일 순
    assert s["history"][1]["ticker"] == "C"
    assert s["recent"][0]["ticker"] == "B"  # 최근 청산이 먼저


def test_list_chart_files_keeps_latest_per_timeframe(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "charts" / "screenshots_sp500"
    for name in ("NBM.V/NBM.V_Daily_20261014_063540.png", "NBM.V/NBM.V_Daily_20261015_062204.png",
                 "FIG/FIG_Weekly_20261015_063508.png", "FIG/notes.txt"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    found = wd.list_chart_files("charts/screenshots_sp500", "screenshots-sp500")
    assert found == {
        "NBM.V": {"Daily": "charts/screenshots_sp500/NBM.V/NBM.V_Daily_20261015_062204.png"},
        "FIG": {"Weekly": "charts/screenshots_sp500/FIG/FIG_Weekly_20261015_063508.png"},
    }
    assert wd._captured_at(found) == "2026-10-15T06:35:08Z"
