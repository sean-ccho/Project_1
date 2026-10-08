#!/usr/bin/env python3
"""Tier 1.5 가설(H1~H6) A/B 백테스트.

기준선과, 가설 하나만 바꾼 변형들을 같은 조건(기간·종목·자본·비용)으로 돌려 비교한다.
설정은 config를 고치지 않고 실행 중에만 바꾼다 (paper_trading.config_override).
첫 실행만 피처 계산이 느리고, 이후 변형은 디스크 캐시로 빠르게 돈다.

사용법:
  PYTHONPATH=.:src python scripts/run_hypothesis_ab.py --period 5y --max-tickers 300
  PYTHONPATH=.:src python scripts/run_hypothesis_ab.py --only BASE H1b H4b --start 2023-01-01
  PYTHONPATH=.:src python scripts/run_hypothesis_ab.py --only T2-5 T2-10 T2-20 --pit-universe
  PYTHONPATH=.:src python scripts/run_hypothesis_ab.py --base T2-10 --pit-universe   # 분산된 기준선 위에서 H1~H6
결과: 표 출력 + output/hypothesis_ab.csv 누적 (계획서 9-8절 실험 기록표)
  - 판정: 같은 날짜끼리 묶어 재표본한 ΔSharpe 95% 구간이 0보다 위면 "채택 후보", 0을 포함하면 "운과 구분 안 됨"
  - 알파: SPY와 "1년 많이 오른 20종목 매달 교체"로 설명되고 남는 수익
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from paper_trading.backtest import _code_hash, run_paper_trading_backtest
from paper_trading.config_override import config_overrides
from paper_trading.evaluation import paired_bootstrap, verdict


def _both(key: str, value: Any) -> dict[str, Any]:
    return {"EXIT_PARAMS": {"모멘텀": {key: value}, "바닥반등": {key: value}}}


VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "BASE": ("기준선 (현재 설정)", {}),
    "H1a": ("트레일링: 고점 +3% 이후에만", _both("trail_activate_pct", 0.03)),
    "H1b": ("트레일링: 고점 +5% 이후에만", _both("trail_activate_pct", 0.05)),
    "H2": ("모멘텀만 (바닥반등 중단)", {"CANDIDATE_ALLOWED_STRATEGIES": ["모멘텀"]}),
    "H3": ("약세장(SPY<EMA200) 신규 진입 중단", {"CANDIDATE_BEAR_BLOCK_NEW": True}),
    "H4a": ("교체 마진 0.10 → 0.20", {"CCS_REPLACE_MARGIN": 0.20}),
    "H4b": ("교체 끄기", {"PT1_REPLACE_ENABLED": False}),
    "H5": ("CCS v2", {"CCS_VERSION": "v2"}),
    "R1": ("장중 손절: 저가가 손절·트레일링선을 건드리면 min(시가, 선)에 체결", {"PT1_STOP_INTRADAY": True}),
    "R2": ("교체 비교: 보유 종목도 오늘 CCS 점수로 (매수 당시 점수 대신)", {"PT1_REPLACE_TODAY_CCS": True}),
    "H6": ("바닥반등 알파: 변동성·평균회귀 비중 축소", {
        "CANDIDATE_ALPHA_WEIGHTS": {"바닥반등": {"mom": 0.25, "trend": 0.25, "vol": 0.20, "volat": 0.10, "mr": 0.20}},
    }),
    "T2-5": ("(Tier 2) 최대 5종목, 하루 2개까지 매수", {"PAPER_TRADING_MAX_POSITIONS": 5, "PAPER_TRADING_MAX_DAILY_BUY": 2}),
    "T2-10": ("(Tier 2) 최대 10종목, 하루 3개까지 매수", {"PAPER_TRADING_MAX_POSITIONS": 10, "PAPER_TRADING_MAX_DAILY_BUY": 3}),
    "T2-20": ("(Tier 2) 최대 20종목, 하루 5개까지 매수", {"PAPER_TRADING_MAX_POSITIONS": 20, "PAPER_TRADING_MAX_DAILY_BUY": 5}),
}
DEFAULT_SET = ["BASE", "H1a", "H1b", "H2", "H3", "H4a", "H4b", "H5", "H6"]
METRICS = ["총거래수", "승률", "평균수익률", "Sharpe_일간", "CAGR", "MDD", "Calmar", "총수익률_자본기준", "SPY수익률",
           "알파_연", "알파_t", "모멘텀대비_판정"]


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """b를 a 위에 깊은 병합 (기준선 설정 + 변형 설정)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _daily(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna() if equity is not None and len(equity) > 1 else pd.Series(dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description="Tier 1.5 가설 A/B 백테스트")
    p.add_argument("--only", nargs="+", choices=list(VARIANTS), help=f"실행할 변형 (기본: {' '.join(DEFAULT_SET)})")
    p.add_argument("--base", choices=[k for k in VARIANTS if k != "BASE"], default=None,
                   help="이 변형 설정을 기준선으로 삼고 나머지를 그 위에 얹는다 (예: T2-10)")
    p.add_argument("--pit-universe", action="store_true", help="그날의 S&P 500 구성종목만 후보로 (생존 편향 완화)")
    p.add_argument("--period", default="5y")
    p.add_argument("--max-tickers", type=int, default=300)
    p.add_argument("--capital", type=float, default=5000.0)
    p.add_argument("--rebalance", type=int, default=1)
    p.add_argument("--start", default=None, help="시뮬레이션 시작일 YYYY-MM-DD (홀드아웃은 빼 둘 것)")
    p.add_argument("--end", default=None, help="시뮬레이션 종료일 YYYY-MM-DD")
    p.add_argument("--fundamentals", action="store_true")
    args = p.parse_args()

    keys = [k for k in (args.only or DEFAULT_SET) if k != args.base]
    if "BASE" not in keys:
        keys = ["BASE"] + keys
    base_overrides = VARIANTS[args.base][1] if args.base else {}

    rows = []
    summaries: dict[str, dict[str, Any]] = {}
    returns: dict[str, pd.Series] = {}
    for key in keys:
        label, overrides = VARIANTS[key]
        if key == "BASE" and args.base:
            label = f"기준선 = {args.base} ({VARIANTS[args.base][0]})"
        print(f"\n===== {key}: {label} =====")
        with config_overrides(_merge(base_overrides, overrides)):
            r = run_paper_trading_backtest(
                period=args.period, max_tickers=args.max_tickers, rebalance_every=args.rebalance,
                initial_capital=args.capital, include_fundamentals=args.fundamentals,
                start_date=args.start, end_date=args.end, save_run=False,
                pit_universe=args.pit_universe,
            )
        s = r["summary"]
        summaries[key] = s
        returns[key] = _daily(r["equity_curve"])
        rows.append({"변형": key, "설명": label, **{m: s.get(m) for m in METRICS},
                     "시작": s.get("시뮬레이션_시작"), "종료": s.get("시뮬레이션_종료")})

    df = pd.DataFrame(rows).set_index("변형")
    base = df.loc["BASE"]
    for m in ("Sharpe_일간", "CAGR", "MDD"):
        df[f"Δ{m}"] = pd.to_numeric(df[m], errors="coerce") - pd.to_numeric(base[m], errors="coerce")

    # 운인지: 같은 날짜끼리 묶어 20일 블록 부트스트랩
    for key in df.index:
        if key == "BASE":
            continue
        a, b = returns[key], returns["BASE"]
        common = a.index.intersection(b.index)
        stats = paired_bootstrap(a[common].tolist(), b[common].tolist())
        if stats:
            df.loc[key, "ΔSharpe_95%"] = f"[{stats['ΔSharpe_하한']:+.2f}, {stats['ΔSharpe_상한']:+.2f}]"
            df.loc[key, "구간개선"] = f"{stats['구간_개선']}/{stats['구간_수']}"
        df.loc[key, "판정"] = verdict(stats, summaries[key].get("MDD"), summaries["BASE"].get("MDD"))

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print("\n===== 결과 (Δ = 기준선 대비) =====")
    print(df.drop(columns=["시작", "종료"]).round(4).to_string())

    bs = summaries["BASE"]
    if "기준_모멘텀_Sharpe" in bs:
        print(f"\n===== 기준선 vs 쉬운 방법{' (PIT 유니버스)' if bs.get('PIT_유니버스') else ''} =====")
        print(f"  Sharpe: 기준선 {bs.get('Sharpe_일간')} | SPY 보유 {bs['기준_SPY_Sharpe']} | "
              f"모멘텀 상위{bs['기준_모멘텀_N']} {bs['기준_모멘텀_Sharpe']}")
        if "알파_연" in bs:
            print(f"  알파 {bs['알파_연']:+.1%}/년 (t={bs['알파_t']}) | 모멘텀 대비 {bs.get('모멘텀대비_95%', '')} → {bs.get('모멘텀대비_판정', '')}")

    out = ROOT / "output" / "hypothesis_ab.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    log = df.reset_index().assign(
        실행시각=f"{datetime.now():%Y-%m-%d %H:%M}", code_hash=_code_hash(),
        기간=args.period, 종목수=args.max_tickers, 리밸런스=args.rebalance,
        기준선=args.base or "BASE", PIT=bool(args.pit_universe),
    )
    if out.exists():
        log = pd.concat([pd.read_csv(out, encoding="utf-8-sig"), log], ignore_index=True)
    log.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n누적 저장: {out}")
    print("채택 규칙: 판정이 '채택 후보'인 것만, 한 번에 하나씩 → 합치면 Baseline v2. '운과 구분 안 됨'은 채택하지 않는다")


if __name__ == "__main__":
    main()
