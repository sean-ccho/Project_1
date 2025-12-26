"""섹터 로테이션 전략 검증 스크립트."""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from data.fetch import fetch_ohlcv
from config import SECTOR_ETFS
from sector_rotation import compute_sector_strength, get_strong_sectors

print("=" * 50)
print("섹터 로테이션 전략 검증")
print("=" * 50)

# 섹터 ETF + SPY 데이터 다운로드
sector_etf_tickers = list(SECTOR_ETFS.values()) + ["SPY"]
print(f"\n[1] 섹터 ETF 다운로드 중: {sector_etf_tickers}")

etf_raw = fetch_ohlcv(sector_etf_tickers, period="1y")

if etf_raw.empty:
    print("ERROR: ETF 데이터를 가져오지 못했습니다.")
    sys.exit(1)

etf_data = {
    ticker: etf_raw[ticker].dropna(how="all")
    for ticker in etf_raw.columns.levels[0]
    if ticker in sector_etf_tickers
}
spy_data = etf_data.get("SPY", pd.DataFrame())

print(f"[2] 다운로드 완료: {len(etf_data)}개 ETF")

# 섹터 강도 계산
print("\n[3] 섹터 강도 계산 중...")
strength = compute_sector_strength(etf_data, spy_data)

print("\n--- 섹터별 상대 강도 (SPY 대비 20일 수익률) ---")
for sector, rel_str in sorted(strength.items(), key=lambda x: x[1], reverse=True):
    status = "✅ 강함" if rel_str >= 0 else "❌ 약함"
    print(f"  {sector:30s}: {rel_str:+.2%} {status}")

# 강한 섹터 목록
strong_sectors = get_strong_sectors(etf_data, spy_data)
print(f"\n[4] 현재 강한 섹터: {strong_sectors if strong_sectors else '없음'}")

print("\n" + "=" * 50)
print("검증 완료!")
print("=" * 50)
