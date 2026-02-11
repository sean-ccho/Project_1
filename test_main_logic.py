
import pandas as pd
from main import build_export_dataframe

# Mock fetch_tickers_from_sheet to return a list
def mock_fetch_tickers():
    return ["AAPL"]

if __name__ == "__main__":
    print("Running local test with mock data...")
    try:
        df = build_export_dataframe(["AAPL"], "LocalTest")
        if df is not None:
             print("Dataframe built successfully with columns:", df.columns.tolist())
             print("Charts captured locally.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
