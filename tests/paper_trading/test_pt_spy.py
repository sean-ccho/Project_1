"""PT-SPY (SPY 보유 기준 계좌) 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading.account_engine import AccountState, BarSeries, process_bar
from paper_trading.accounts import get_profile


def _series(closes: list[float]) -> dict[str, BarSeries]:
    days = [f"2026-10-{i:02d}" for i in range(5, 5 + len(closes))]
    return {"SPY": BarSeries([{"date": d, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}
                              for d, c in zip(days, closes)])}


def test_profile_has_no_email_and_own_tabs():
    p = get_profile("pt_spy")
    assert p.enabled and p.params["email"] is False and p.params["max_positions"] == 1
    assert p.worksheets["log"] == "페이퍼SPY_거래로그"


def test_buys_spy_next_open_then_holds_without_exits():
    profile = get_profile("pt_spy")
    series = _series([600.0, 606.0, 594.0, 630.0])
    rows = [{"티커": "SPY", "close": 600.0, "ema50": 590.0, "ema200": 560.0}]
    state = AccountState.new(5000.0)
    r0 = process_bar(state, profile, "2026-10-05", series, rows=rows)
    assert [o["ticker"] for o in r0["orders"]] == ["SPY"] and not state.positions
    r1 = process_bar(state, profile, "2026-10-06", series, rows=rows)  # 다음날 시가 매수
    assert [b["ticker"] for b in r1["buys"]] == ["SPY"] and len(state.positions) == 1
    assert state.cash < 1.0  # 거의 전액 투자 (비용 0.1% 제외)
    for d in ("2026-10-07", "2026-10-08"):  # -2% 하락·+6% 상승에도 팔지 않고, 중복 매수도 없다
        r = process_bar(state, profile, d, series, rows=rows)
        assert not r["sells"] and not r["buys"] and not r["orders"]
    eq = state.equity_history[-1]["equity"]
    assert abs(eq - 5000.0 / 1.001 * 630.0 / 606.0) < 1.0
