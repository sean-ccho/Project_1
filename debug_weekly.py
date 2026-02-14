
import pandas as pd
import yfinance as yf
from patterns import get_weekly_data, detect_weekly_patterns
from config import PATTERN_LOOKBACK_DAYS


def test_weekly_patterns():
    tickers = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "INTC"]
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            df = yf.download(ticker, period="5y", progress=False)
            if df.empty:
                print("Failed to fetch data.")
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            weekly_df = get_weekly_data(df)
            
            # Simple check for MA
            if len(weekly_df) >= 200:
                ma50 = weekly_df['Close'].rolling(50).mean().iloc[-1]
                ma200 = weekly_df['Close'].rolling(200).mean().iloc[-1]
                print(f"  Weeks: {len(weekly_df)}, MA50: {ma50:.2f}, MA200: {ma200:.2f}")
            else:
                print(f"  Weeks: {len(weekly_df)} (Insufficient for Golden Cross)")
                
            patterns_str = detect_weekly_patterns(df)
            print(f"  Detected patterns: '{patterns_str}'")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_weekly_patterns()

