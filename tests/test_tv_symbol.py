"""야후 티커 → TradingView 심볼 (BRK-B 가 런던 종목으로 열리던 캡처 버그)."""
from screener.tv_symbol import tv_symbol


def test_share_class_uses_dot() -> None:
    assert tv_symbol("BRK-B") == "BRK.B"
    assert tv_symbol("BF-B", {}) == "BF.B"


def test_map_and_plain_tickers() -> None:
    assert tv_symbol("NBM.V", {"NBM.V": "NBM"}) == "NBM"
    assert tv_symbol("AAPL", {"NBM.V": "NBM"}) == "AAPL"
    assert tv_symbol("BRK-B", {"BRK-B": "NYSE:BRK.B"}) == "NYSE:BRK.B"
