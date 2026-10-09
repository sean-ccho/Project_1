#!/usr/bin/env python3
"""2단계: PT-1 구조 워크포워드 최적화 — 연습 구간에서 고른 설정을 다음 해에 그대로 시험한다.

폴드 Y (2020 ~ 2026): 2016-01-04 ~ (Y-1)-12-31 로 Optuna 60회 → 최고 설정을 Y 년(1/1 ~ 12/31, 2026 은 10/7까지)에 고정 적용.
시험 연도만 이어 붙인 수익 = 정직한 표본 외 성적. 같은 해들을 현재 설정(BASE)·SPY 와 비교한다.
시험 연도는 매년 현금에서 시작한다 (BASE 도 같게).

조정 설정 8개 (나머지 약 190개 숫자는 현재 값 고정):
  보유 종목 수 3/5/7/10 (하루 매수 1/2/2/3) · 약세장 none/block(신규 매수 중단)/liquidate(보유 정리까지)
  비중 equal/inv_vol · 손절 5~15% · 트레일링 4~12% · 트레일링 시작 0~5% · 최대 보유 10~40일 · 바닥반등 허용/중단
  (청산 4개는 모멘텀·바닥반등에 같은 값)
목적함수 (계획서 옵션 A): Sharpe_일간 − max(0, |MDD| − 0.30) × 2, 거래 50건 미만은 탈락.
판정 (사전 등록, TRIAL_LOG wfo_structure_10y):
  통과 = 표본 외 ΔSharpe(vs BASE) 95% 하한 > 0 · 구간 대부분 개선 · MDD 2%p 넘게 악화 없음 (verdict == 채택 후보)
         그리고 Deflated Sharpe ≥ 0.95 (시도 전체 420회의 Sharpe 분산으로 '운으로 기대되는 최고치' 보정)
  SPY 대비는 참고로 함께 보고한다.

사용법:
  PYTHONPATH=.:src BACKTEST_EARNINGS_PATH=data/research/sec_earnings_dates_holdout.parquet \\
    python scripts/wfo_structure.py --fold 2020        (폴드마다 따로, 동시에 돌린다)
  ... python scripts/wfo_structure.py --evaluate       (폴드가 다 끝난 뒤)
산출: output/wfo/fold_YYYY.json, tier3/WFO_STRUCTURE_10y.md
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import optuna
import pandas as pd

from paper_trading.backtest import run_paper_trading_backtest
from paper_trading.config_override import config_overrides
from paper_trading.evaluation import cagr, deflated_sharpe, max_drawdown, paired_bootstrap, sharpe, verdict
from research_utils import TIER3_DIR, TRIAL_LOG, log_trials

FOLDS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)
TRAIN_START, END, PERIOD = "2016-01-04", "2026-10-07", "13y"
N_TRIALS = 60
FAMILY = "wfo_structure_10y"
OUT = ROOT / "output" / "wfo"
DAILY_BUY = {3: 1, 5: 2, 7: 2, 10: 3}


def suggest(trial: optuna.Trial) -> dict:
    return {
        "max_positions": trial.suggest_categorical("max_positions", [3, 5, 7, 10]),
        "bear_mode": trial.suggest_categorical("bear_mode", ["none", "block", "liquidate"]),
        "sizing": trial.suggest_categorical("sizing", ["equal", "inv_vol"]),
        "stop_loss": trial.suggest_float("stop_loss", 0.05, 0.15, step=0.01),
        "trailing_stop": trial.suggest_float("trailing_stop", 0.04, 0.12, step=0.01),
        "trail_activate_pct": trial.suggest_float("trail_activate_pct", 0.0, 0.05, step=0.01),
        "max_holding_days": trial.suggest_int("max_holding_days", 10, 40, step=5),
        "allow_bottom": trial.suggest_categorical("allow_bottom", [True, False]),
    }


def to_overrides(p: dict) -> dict:
    """설정 dict → config 덮어쓰기."""
    exits = {k: p[k] for k in ("stop_loss", "trailing_stop", "trail_activate_pct", "max_holding_days")}
    return {
        "PAPER_TRADING_MAX_POSITIONS": p["max_positions"],
        "PAPER_TRADING_MAX_DAILY_BUY": DAILY_BUY[p["max_positions"]],
        "CANDIDATE_BEAR_BLOCK_NEW": p["bear_mode"] != "none",
        "PT1_BEAR_MODE": "liquidate" if p["bear_mode"] == "liquidate" else "none",
        "PT1_SIZING": p["sizing"],
        "EXIT_PARAMS": {"모멘텀": dict(exits), "바닥반등": dict(exits)},
        "EXIT_PARAMS_DEFAULT": dict(exits),
        "CANDIDATE_ALLOWED_STRATEGIES": None if p["allow_bottom"] else ["모멘텀"],
    }


def run_bt(overrides: dict, start: str, end: str) -> dict:
    """조용히 백테스트 한 번 (진행 표시는 버린다)."""
    with config_overrides(overrides), contextlib.redirect_stdout(io.StringIO()):
        return run_paper_trading_backtest(period=PERIOD, max_tickers=1000, rebalance_every=1, initial_capital=5000.0,
                                          start_date=start, end_date=end, save_run=False, pit_universe=True)


def objective_score(s: dict) -> float:
    """계획서 옵션 A."""
    if (s.get("총거래수") or 0) < 50:
        raise optuna.TrialPruned()
    sh, mdd = s.get("Sharpe_일간"), s.get("MDD")
    if sh is None or mdd is None or not math.isfinite(sh):
        raise optuna.TrialPruned()
    return float(sh) - max(0.0, abs(float(mdd)) - 0.30) * 2.0


def run_fold(year: int, n_trials: int) -> None:
    train_end = f"{year - 1}-12-31"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=year, n_startup_trials=15))
    t0 = datetime.now()

    def objective(trial: optuna.Trial) -> float:
        p = suggest(trial)
        s = run_bt(to_overrides(p), TRAIN_START, train_end)["summary"]
        for k in ("Sharpe_일간", "CAGR", "MDD", "총거래수"):
            trial.set_user_attr(k, s.get(k))
        value = objective_score(s)
        print(f"[fold {year}] #{trial.number + 1}/{n_trials} 점수 {value:+.3f} (Sharpe {s.get('Sharpe_일간')}, "
              f"MDD {s.get('MDD'):.1%}, 거래 {s.get('총거래수')}) {p} · {datetime.now() - t0}", flush=True)
        return value

    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    trials = [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params, **t.user_attrs}
              for t in study.trials]
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    best = max(done, key=lambda t: t.value) if done else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fold_{year}.json").write_text(json.dumps({
        "year": year, "train": [TRAIN_START, train_end], "n_trials": n_trials,
        "best_params": best.params if best else None, "best_value": best.value if best else None,
        "best_attrs": best.user_attrs if best else None, "trials": trials,
        "finished": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[fold {year}] 끝 · 최고 {best.value if best else None} {best.params if best else None}", flush=True)


def _daily(r: dict) -> pd.Series:
    eq = r["equity_curve"].dropna()
    return eq.pct_change().dropna()


def evaluate() -> None:
    folds = [json.loads((OUT / f"fold_{y}.json").read_text(encoding="utf-8")) for y in FOLDS]
    rows, wfo_parts, base_parts = [], [], []
    for f in folds:
        y, p = f["year"], f["best_params"]
        start, end = f"{y}-01-01", min(f"{y}-12-31", END)
        rw, rb = _daily(run_bt(to_overrides(p), start, end)), _daily(run_bt({}, start, end))
        wfo_parts.append(rw)
        base_parts.append(rb)
        rows.append({"연도": y, **p, "연습_점수": f["best_value"], "연습_Sharpe": f["best_attrs"].get("Sharpe_일간"),
                     "WFO": float((1 + rw).prod() - 1), "BASE": float((1 + rb).prod() - 1)})
        print(f"[evaluate] {y}: WFO {rows[-1]['WFO']:+.1%} · BASE {rows[-1]['BASE']:+.1%} · {p}", flush=True)
    wfo, base = pd.concat(wfo_parts), pd.concat(base_parts)
    common = wfo.index.intersection(base.index)
    wfo, base = wfo[common], base[common]

    from data.fetch import fetch_ohlcv
    spy_px = fetch_ohlcv(["SPY"], period=PERIOD)["SPY"]["Close"].dropna()
    spy = spy_px.pct_change().reindex(common).fillna(0.0)
    for r in rows:
        yr = spy[spy.index.year == r["연도"]]
        r["SPY"] = float((1 + yr).prod() - 1)

    def stats(x: pd.Series) -> dict:
        v = x.tolist()
        return {"Sharpe": sharpe(v), "CAGR": cagr(v), "MDD": max_drawdown(v)}
    sw, sb, ss = stats(wfo), stats(base), stats(spy)
    vs_base = paired_bootstrap(wfo.tolist(), base.tolist())
    vs_spy = paired_bootstrap(wfo.tolist(), spy.tolist())
    v_base = verdict(vs_base, sw["MDD"], sb["MDD"])
    trial_sr = [t["Sharpe_일간"] / math.sqrt(252) for f in folds for t in f["trials"]
                if t["state"] == "COMPLETE" and t.get("Sharpe_일간") is not None]
    dsr = deflated_sharpe(sw["Sharpe"] / math.sqrt(252), trial_sr, len(wfo),
                          skew=float(wfo.skew()), kurt=float(wfo.kurt() + 3))
    passed = v_base == "채택 후보" and dsr >= 0.95
    final = "통과" if passed else "불통과"
    log_trials("stage7", FAMILY, len(trial_sr), [{"test_id": "WFO 구조 8개", "판정": final, "메모": (
        f"표본 외 2020~2026-10: Sharpe {sw['Sharpe']:.2f} vs BASE {sb['Sharpe']:.2f} vs SPY {ss['Sharpe']:.2f}, "
        f"ΔSharpe(BASE) {vs_base.get('ΔSharpe', float('nan')):+.2f} [{vs_base.get('ΔSharpe_하한', float('nan')):+.2f}, "
        f"{vs_base.get('ΔSharpe_상한', float('nan')):+.2f}] {v_base}, DSR {dsr:.2f}, MDD {sw['MDD']:.1%} vs {sb['MDD']:.1%}")}])

    names = {"max_positions": "종목 수", "bear_mode": "약세장", "sizing": "비중", "stop_loss": "손절",
             "trailing_stop": "트레일링", "trail_activate_pct": "트레일링 시작", "max_holding_days": "최대 보유",
             "allow_bottom": "바닥반등"}
    lines = ["# 2단계: PT-1 구조 워크포워드 (2016~2026, PIT S&P 500)", "",
             f"- 폴드마다 2016-01-04 ~ 전년 말로 Optuna {N_TRIALS}회 → 최고 설정을 다음 해에 고정 적용. 시험 연도는 매년 현금에서 시작",
             f"- 목적함수 = Sharpe − max(0, |MDD| − 30%) × 2, 거래 50건 미만 탈락. 완료된 시도 {len(trial_sr)}회", "",
             f"## 판정: **{final}**", "",
             f"- 표본 외 ΔSharpe (vs 현재 설정) {vs_base.get('ΔSharpe', float('nan')):+.2f}, 95% 구간 "
             f"[{vs_base.get('ΔSharpe_하한', float('nan')):+.2f}, {vs_base.get('ΔSharpe_상한', float('nan')):+.2f}], "
             f"구간 개선 {vs_base.get('구간_개선')}/{vs_base.get('구간_수')} → {v_base}",
             f"- Deflated Sharpe {dsr:.2f} (기준 0.95)",
             f"- 참고 vs SPY: ΔSharpe {vs_spy.get('ΔSharpe', float('nan')):+.2f} [{vs_spy.get('ΔSharpe_하한', float('nan')):+.2f}, "
             f"{vs_spy.get('ΔSharpe_상한', float('nan')):+.2f}]", "",
             "| 표본 외 2020-01 ~ 2026-10 | Sharpe | 연수익률 | 최대 낙폭 |", "|---|---|---|---|",
             f"| 워크포워드 설정 | {sw['Sharpe']:.2f} | {sw['CAGR']:+.1%} | {sw['MDD']:.1%} |",
             f"| 현재 설정 | {sb['Sharpe']:.2f} | {sb['CAGR']:+.1%} | {sb['MDD']:.1%} |",
             f"| SPY | {ss['Sharpe']:.2f} | {ss['CAGR']:+.1%} | {ss['MDD']:.1%} |", "",
             "## 해마다 고른 설정과 그해 수익", "",
             "| 시험 연도 | " + " | ".join(names.values()) + " | 연습 Sharpe | 워크포워드 | 현재 설정 | SPY |",
             "|---" * (len(names) + 5) + "|"]
    for r in rows:
        cells = []
        for k in names:
            v = r[k]
            cells.append(f"{v:.0%}" if isinstance(v, float) and v < 1 else str(v))
        lines.append(f"| {r['연도']} | " + " | ".join(cells) + f" | {r['연습_Sharpe']} | {r['WFO']:+.1%} | {r['BASE']:+.1%} | {r['SPY']:+.1%} |")
    (TIER3_DIR / "WFO_STRUCTURE_10y.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=FOLDS)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--register", action="store_true", help="결과 보기 전에 TRIAL_LOG 에 등록")
    args = ap.parse_args()
    if args.register and not _registered():
        log_trials("stage7", FAMILY, N_TRIALS * len(FOLDS), [{"test_id": "(등록)", "판정": "등록", "메모": (
            "PT-1 구조 8개 워크포워드 (2016~ 확장 창, 시험 2020~2026-10, 폴드당 Optuna 60회, 옵션 A 목적함수). "
            "통과 = 표본 외 vs 현재 설정 verdict 채택 후보 + DSR ≥ 0.95. SPY 대비는 참고")}])
        print("등록 완료")
    if args.fold:
        run_fold(args.fold, args.trials)
    if args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
