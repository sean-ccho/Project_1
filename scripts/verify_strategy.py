
import sys
import os
sys.path.append(os.getcwd())

from backtest import run_backtest
from config import BACKTEST_RUNS

# Override config for testing
test_tickers = ["SPY", "AAPL", "NVDA", "TSLA", "MSFT"]

print("Running verification backtest on small subset...")
try:
    results = run_backtest(
        tickers=test_tickers,
        period="1y",
        max_positions=2,
        include_fundamentals=False # Speed up
    )
    print("Backtest completed successfully.")
    print("Summary:", results['summary'])
    print("Trades:", len(results['trades']))
    if not results['trades'].empty:
        print(results['trades'].head())
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
