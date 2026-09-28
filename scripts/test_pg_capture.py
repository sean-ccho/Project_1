import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from charts.tradingview_capture import capture_tradingview_chart

def main():
    print("Capturing PG chart...")
    path_pg = capture_tradingview_chart("PG", timeframe="Daily", output_dir="charts/screenshots/test", headless=True)
    if path_pg:
        print(f"PG Success! Saved to {path_pg}")
        
    print("Capturing NBM.V chart...")
    path_nbm = capture_tradingview_chart("NBM.V", timeframe="Daily", output_dir="charts/screenshots/test", headless=True)
    if path_nbm:
        print(f"NBM.V Success! Saved to {path_nbm}")

if __name__ == "__main__":
    main()
