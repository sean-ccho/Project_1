"""S&P 500 목록 월간 갱신 테스트 — 변경 계획, 보류 기준, 파일 내용, 변경 기록, 메일 내용 (네트워크 없이)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import update_sp500_tickers as up  # noqa: E402


def _table(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"티커": tickers, "회사": [f"{t} Inc." for t in tickers],
                         "섹터": ["Industrials"] * len(tickers), "편입일": ["2026-11-01"] * len(tickers)})


def test_plan_lists_added_and_removed():
    plan = up.plan_update(["AAA", "BBB", "CCC"], _table(["AAA", "CCC", "DDD"]))
    assert [a["티커"] for a in plan.added] == ["DDD"] and plan.removed == ["BBB"]
    assert plan.changed and plan.problem == ""
    assert not up.plan_update(["AAA"], _table(["AAA"])).changed


def test_plan_holds_when_too_many_changes():
    plan = up.plan_update([f"OLD{i}" for i in range(15)], _table([f"NEW{i}" for i in range(15)]))
    assert "변경 30개" in plan.problem


def test_render_is_valid_ticker_module():
    namespace: dict = {}
    exec(up.render(["MMM", "BRK-B"], "2026-11-01"), namespace)
    assert namespace["SP500_TICKERS"] == ["MMM", "BRK-B"]


def test_change_log_appends_and_message_flags_held_tickers(tmp_path):
    plan = up.plan_update(["AAA", "BBB"], _table(["AAA", "DDD"]))
    log = tmp_path / "changes.csv"
    up.append_change_log(plan, "2026-11-01", log)
    up.append_change_log(plan, "2026-12-01", log)
    df = pd.read_csv(log)
    assert df["구분"].tolist() == ["추가", "제거"] * 2 and df["티커"].tolist() == ["DDD", "BBB"] * 2

    subject, body = up.change_message(plan, "2026-11-01", {"PT-1": ["BBB", "XYZ"], "PT-1S": []})
    assert subject == "[S&P 500 목록] 2026-11-01 갱신: 추가 1 · 제거 1"
    assert "DDD  DDD Inc. (Industrials, 편입 2026-11-01)" in body
    assert "보유 중인 종목이 빠졌습니다: PT-1: BBB" in body and "XYZ" not in body
