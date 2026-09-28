#!/usr/bin/env python3
"""페이퍼 트레이딩 CLI.

사용법:
    python scripts/paper_trading.py status              — 현재 포지션 출력
    python scripts/paper_trading.py history             — 전체 거래 히스토리 출력
    python scripts/paper_trading.py reset               — 포지션 초기화 (새로 시작)
    python scripts/paper_trading.py auto                — 스크리너 실행 후 자동 매매 (main.py에서 호출)
    python scripts/paper_trading.py backtest            — 백테스트 (기본 1y, S&P500 100종목)
    python scripts/paper_trading.py backtest --period 2y            — 2년 백테스트
    python scripts/paper_trading.py backtest --rebalance 3          — 3일 주기로 후보 선정
    python scripts/paper_trading.py backtest --max-tickers 50       — 50개 종목만
    python scripts/paper_trading.py backtest --no-sheets            — 구글 시트 동기화 건너뜀
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# scripts/ 디렉토리를 sys.path에서 제거 (paper_trading.py 이름 충돌 방지)
scripts_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != scripts_dir]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def cmd_status():
    """현재 보유 포지션 출력."""
    from paper_trading.portfolio import load_positions, get_position_return, get_holding_days
    from data.fetch import fetch_latest_prices

    positions = load_positions()
    if not positions:
        print("보유 종목 없음")
        return

    tickers = [p["ticker"] for p in positions]
    prices = fetch_latest_prices(tickers)

    print(f"\n{'종목':<8} {'매수일':<12} {'매수가':>8} {'현재가':>8} {'수익률':>8} {'고점':>8} {'보유일':>6} {'전략'}")
    print("-" * 80)
    for pos in positions:
        ticker = pos["ticker"]
        current = prices.get(ticker, pos["entry_price"])
        ret = get_position_return(pos, current)
        days = get_holding_days(pos)
        highest = pos.get("highest_price", pos["entry_price"])
        strategy = pos.get("strategy", "")
        # 전략구분에서 이모지 제거해서 짧게
        strat_short = strategy.replace("📉 ", "").replace("📈 ", "").replace("🔄 ", "")
        print(
            f"{ticker:<8} {pos['entry_date']:<12} "
            f"${pos['entry_price']:>7.2f} ${current:>7.2f} "
            f"{ret:>+7.1%} ${highest:>7.2f} {days:>5}일 {strat_short}"
        )
    print()


def cmd_history():
    """전체 거래 히스토리 출력."""
    from paper_trading.portfolio import load_trades

    trades = load_trades()
    if not trades:
        print("거래 기록 없음")
        return

    wins = sum(1 for t in trades if t.get("return_pct", 0) > 0)
    total = len(trades)
    avg_ret = sum(t.get("return_pct", 0) for t in trades) / total if total else 0

    print(f"\n총 {total}건 | 승률 {wins}/{total} ({wins/total:.0%}) | 평균수익 {avg_ret:+.1%}\n")
    print(f"{'종목':<8} {'매수일':<12} {'매도일':<12} {'수익률':>8} {'보유일':>6} {'사유'}")
    print("-" * 70)
    for t in trades:
        print(
            f"{t['ticker']:<8} {t['entry_date']:<12} {t['exit_date']:<12} "
            f"{t['return_pct']:>+7.1%} {t['holding_days']:>5}일 {t['exit_reason']}"
        )
    print()


def cmd_reset():
    """포지션 초기화."""
    from paper_trading.portfolio import save_positions, save_trades

    confirm = input("정말 초기화하시겠습니까? 모든 포지션과 거래 기록이 삭제됩니다. (y/N): ")
    if confirm.lower() != "y":
        print("취소됨")
        return

    save_positions([])
    save_trades([])
    print("페이퍼 트레이딩 초기화 완료 (포지션 0개, 거래 0건)")


def cmd_auto():
    """스크리너 실행 없이 기존 ranked 데이터로 자동 매매 (테스트용)."""
    print("auto 모드는 main.py 파이프라인에서 자동 호출됩니다.")
    print("사용법: python src/main.py")


def cmd_backtest():
    """과거 데이터로 페이퍼 트레이딩 로직 시뮬레이션."""
    import argparse
    from paper_trading.backtest import run_paper_trading_backtest, print_summary
    from paper_trading.sheet_sync import sync_backtest_result

    parser = argparse.ArgumentParser(prog="paper_trading backtest", add_help=False)
    parser.add_argument("--period", default="1y", help="데이터 기간 (1y, 2y 등, 기본값: 1y)")
    parser.add_argument("--rebalance", type=int, default=5, help="후보 선정 주기 (거래일, 기본값: 5)")
    parser.add_argument("--max-tickers", type=int, default=100, help="분석 종목 수 제한 (기본값: 100)")
    parser.add_argument("--no-sheets", action="store_true", help="구글 시트 동기화 건너뜀")
    parser.add_argument("--fundamentals", action="store_true", help="펀더멘탈 데이터 포함 (느림)")
    parser.add_argument("--capital", type=float, default=0.0, metavar="USD", help="초기 자본금 (예: 5000). 0이면 수익률 모드")

    # sys.argv에서 'backtest' 이후 인자만 파싱
    args = parser.parse_args(sys.argv[2:])

    print(f"\n페이퍼 트레이딩 백테스트 시작")
    print(f"  기간: {args.period}  |  리밸런스: {args.rebalance}일  |  종목수: {args.max_tickers}개")
    if args.fundamentals:
        print("  펀더멘탈 데이터 포함 (시간이 더 걸립니다)")
    if args.capital > 0:
        print(f"  초기자본: ${args.capital:,.0f}")

    result = run_paper_trading_backtest(
        period=args.period,
        rebalance_every=args.rebalance,
        max_tickers=args.max_tickers,
        include_fundamentals=args.fundamentals,
        initial_capital=args.capital,
    )

    summary = result["summary"]
    trades = result["trades"]

    print_summary(summary, trades)

    if not args.no_sheets and trades:
        print("Google Sheets 동기화 중...")
        ok = sync_backtest_result(summary, trades)
        if ok:
            print("→ [백테스트_결과] 탭 업데이트 완료\n")
        else:
            print("→ Google Sheets 동기화 실패 (로컬 결과는 위에 출력됨)\n")
    elif args.no_sheets:
        print("(--no-sheets 옵션으로 구글 시트 동기화 건너뜀)\n")
    else:
        print("(거래 없음 — 구글 시트 업데이트 건너뜀)\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {
        "status": cmd_status,
        "history": cmd_history,
        "reset": cmd_reset,
        "auto": cmd_auto,
        "backtest": cmd_backtest,
    }

    if cmd not in commands:
        print(f"알 수 없는 명령: {cmd}")
        print(f"사용 가능: {', '.join(commands.keys())}")
        sys.exit(1)

    commands[cmd]()


if __name__ == "__main__":
    main()
