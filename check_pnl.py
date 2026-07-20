import yfinance as yf

positions = [
    {"ticker": "MPWR", "entry_price": 1363.42, "entry_date": "2026-04-14"},
    {"ticker": "T",    "entry_price": 26.40,   "entry_date": "2026-04-16"},
    {"ticker": "NFLX", "entry_price": 97.31,   "entry_date": "2026-04-17"},
]

print(f"{'티커':<6} {'진입가':>10} {'현재가':>10} {'손익률':>8}  {'진입일'}")
print("-" * 55)
total_pnl = 0
for p in positions:
    data = yf.Ticker(p["ticker"]).history(period="1d")
    cur = float(data["Close"].iloc[-1]) if not data.empty else None
    if cur:
        pnl = (cur - p["entry_price"]) / p["entry_price"] * 100
        total_pnl += pnl
        flag = "🔴" if pnl < 0 else "🟢"
        print(f"{flag} {p['ticker']:<4} {p['entry_price']:>10.2f} {cur:>10.2f} {pnl:>+7.2f}%  {p['entry_date']}")
    else:
        print(f"{p['ticker']:<6} 데이터 없음")

print("-" * 55)
print(f"평균 손익률: {total_pnl/len(positions):>+.2f}%")
