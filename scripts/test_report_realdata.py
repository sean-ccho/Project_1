"""실제 parquet 스냅샷 + positions/trades.json 데이터로 PDF 생성.

실행: python scripts/test_report_realdata.py
결과: output/realdata_trading_report.pdf
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pandas as pd
from pathlib import Path
from datetime import date

# ── 실데이터 로드 ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "paper_trading"

# parquet 스냅샷 (SP500 + NASDAQ 합치기)
sp_path = DATA_DIR / "sp500_ranked.parquet"
nq_path = DATA_DIR / "nasdaq_ranked.parquet"

dfs = []
if sp_path.exists():
    dfs.append(pd.read_parquet(sp_path))
    print(f"SP500 로드: {len(dfs[-1])}개 종목")
if nq_path.exists():
    dfs.append(pd.read_parquet(nq_path))
    print(f"NASDAQ 로드: {len(dfs[-1])}개 종목")

if not dfs:
    print("ERROR: parquet 스냅샷 없음")
    sys.exit(1)

merged_df = pd.concat(dfs, ignore_index=True)
if "티커" in merged_df.columns:
    merged_df = merged_df.drop_duplicates(subset=["티커"], keep="first")

# positions.json
positions_path = DATA_DIR / "positions.json"
positions = json.loads(positions_path.read_text()) if positions_path.exists() else []
print(f"보유 포지션: {[p['ticker'] for p in positions]}")

# trades.json
trades_path = DATA_DIR / "trades.json"
trades = json.loads(trades_path.read_text()) if trades_path.exists() else []
print(f"거래 이력: {len(trades)}건")

# ── 현재가 조회 (yfinance) ─────────────────────────────────────────────────────
held_tickers = [p["ticker"] for p in positions]
prices: dict[str, float] = {}
if held_tickers:
    try:
        import yfinance as yf
        data = yf.download(held_tickers, period="2d", auto_adjust=True, progress=False)
        closes = data["Close"] if "Close" in data else data
        for t in held_tickers:
            try:
                prices[t] = float(closes[t].dropna().iloc[-1])
            except Exception:
                pass
        print(f"현재가 조회: {prices}")
    except Exception as e:
        print(f"현재가 조회 실패 ({e}), 진입가 사용")
        for p in positions:
            prices[p["ticker"]] = p["entry_price"]

# ── 실제 candidate_selector 실행 ─────────────────────────────────────────────
from paper_trading.candidate_selector import select_best_candidate
from paper_trading.engine import check_sell_conditions

candidate, debug = select_best_candidate(merged_df, positions)
print(f"레짐: {debug.get('regime')}")
print(f"후보: {candidate['ticker'] if candidate else '없음'}")
if debug.get("top5"):
    print("Top5:")
    for s in debug["top5"]:
        print(f"  {s['ticker']:6s} CCS={s['ccs']:.4f} 전략={s['strategy']}")

# ── 매도 체크 ─────────────────────────────────────────────────────────────────
today_str = str(date.today())
sells = []
for pos in positions:
    t = pos["ticker"]
    cp = prices.get(t, pos["entry_price"])
    should_sell, reason = check_sell_conditions(pos, cp, today_str)
    if should_sell:
        ret = (cp - pos["entry_price"]) / pos["entry_price"]
        sells.append({
            "ticker": t,
            "entry_price": pos["entry_price"],
            "exit_price": cp,
            "return_pct": ret,
            "holding_days": 0,
            "exit_reason": reason,
            "sector": pos.get("sector", ""),
            "strategy": pos.get("strategy", ""),
        })
        print(f"매도 신호: {t} ({reason})")

# ── buys 구성 ─────────────────────────────────────────────────────────────────
buys = []
if candidate:
    entry_price = prices.get(candidate["ticker"], candidate["entry_price"])
    buys.append({
        "ticker": candidate["ticker"],
        "entry_date": today_str,
        "entry_price": entry_price,
        "sector": candidate["sector"],
        "strategy": candidate["strategy"],
        "star_rating": candidate["star_rating"],
        "ccs_score": candidate["ccs_score"],
        "reason": "신규매수",
        "ccs_breakdown": candidate.get("ccs_breakdown", {}),
    })

# ── pt_result 구성 ─────────────────────────────────────────────────────────────
pt_result = {
    "date": today_str,
    "buys": buys,
    "sells": sells,
    "holdings": len(positions),
    "selection_debug": debug,
}

# ── PDF 생성 ─────────────────────────────────────────────────────────────────
from paper_trading.report_generator import generate_trading_report

print("\nPDF 리포트 생성 중...")
pdf_bytes = generate_trading_report(
    result=pt_result,
    positions=positions,
    merged_df=merged_df,
    prices=prices,
    trades=trades,
)

if pdf_bytes:
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "realdata_trading_report.pdf"
    out_path.write_bytes(pdf_bytes)
    print(f"✅ PDF 생성 완료: {out_path} ({len(pdf_bytes):,} bytes)")
    import subprocess
    subprocess.Popen(["open", str(out_path)])
else:
    print("❌ PDF 생성 실패")
