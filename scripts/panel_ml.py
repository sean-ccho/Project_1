#!/usr/bin/env python3
"""Tier 3-B 4단계: 패널 ML (LightGBM, 매년 재학습 워크포워드, px10y).

기존 entry_quality 모델은 거래 561건으로 학습해 CV AUC ≈ 0.5였다. 같은 아이디어를 패널 수십만 행에 적용한다.
  - 피처: px10y 가격·거래량 피처를 날짜별 백분위 순위로 변환
    · 제외: `_sn`(섹터 중립) — 섹터가 현재 구성종목에만 있어 "결측 = 나중에 지수 탈락"이라는 미래 정보가 샌다
    · 제외: `close`(가격 수준, 횡단면 의미 없음)
  - 타깃: 날짜별 fwd_ret_10d 백분위 − 0.5
  - 학습: 확장 창, 매년 1월 재학습 (OOS 2019-01 ~ 2025-09). 학습 끝 H+EMBARGO 거래일 제거 (라벨 겹침 방지)
  - 하이퍼파라미터 고정 (사전 등록, 튜닝 안 함)
판정 (시도 1건, 모두 만족해야 후보):
  OOS IC t_nw(lag 9) ≥ 3.0 · 상위 10% 비용 후 초과수익 > 0 인 해가 7년 중 5년 이상 · 시뮬(k=20, h=10) SPY 대비 ΔSharpe 하한 > 0
비교 기준: px10y 선형 워크포워드 합성(composite_research.py, 같은 OOS) — WF_top5 10일 t_nw.
확인: 날짜 안에서 타깃을 섞어 학습하면 OOS IC ≈ 0 이어야 한다 (--shuffle-check, 시도로 세지 않음).

사용법: PYTHONPATH=.:src python scripts/panel_ml.py [--shuffle-check]
산출: data/research/ml_pred_px10y.parquet, tier3/ML_REPORT_px10y.md, tier3/ml_importance_px10y.csv,
      tier3/PORTFOLIO_SIM_ml_top20_h10.md
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from factor_research import ic_by_date
from research_utils import (DEV_END, RESEARCH_DIR, ROUND_TRIP, TIER3_DIR, TRIAL_LOG, load_panel, log_trials,
                            nw_tstat, total_trials, yearly_mean)

H, EMBARGO = 10, 5
YEARS = range(2019, 2026)
TOP_Q = 0.10
SIM_K = 20
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=2000,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              random_state=42, n_jobs=-1, verbose=-1)  # 고정 (사전 등록, 튜닝 안 함)
FAMILY = "panel_ml_px10y"
IC_T_MIN, MIN_POS_YEARS = 3.0, 5


def features(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns
            if c not in {"date", "티커", "close"} and not c.startswith("fwd_ret_") and not c.endswith("_sn")
            and pd.api.types.is_numeric_dtype(panel[c])]


def walk_forward(panel: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """매년 재학습 → OOS 예측, fold별 gain 중요도."""
    dates = np.array(sorted(panel["date"].unique()))
    pred = pd.Series(np.nan, index=panel.index)
    imps = []
    for year in YEARS:
        test_start = pd.Timestamp(f"{year}-01-01")
        train_dates = dates[dates < test_start][: -(H + EMBARGO)]  # 라벨이 검증 구간과 겹치는 끝부분 제거
        tr = panel["date"].isin(train_dates) & y.notna()
        te = (panel["date"] >= test_start) & (panel["date"] <= min(pd.Timestamp(f"{year}-12-31"), DEV_END))
        model = LGBMRegressor(**PARAMS).fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        imps.append(pd.Series(model.booster_.feature_importance("gain"), index=X.columns, name=year))
        print(f"  {year}: 학습 {int(tr.sum()):,}행 (~{pd.Timestamp(train_dates[-1]).date()}) · 검증 {int(te.sum()):,}행",
              flush=True)
    return pred, pd.concat(imps, axis=1)


def evaluate(panel: pd.DataFrame, pred: pd.Series) -> dict:
    """OOS 날짜별 IC, 상위 10% 초과수익(비용 후, 연도별)."""
    y = f"fwd_ret_{H}d"
    m = pred.notna() & panel[y].notna()
    d = panel.loc[m, ["date", y]].assign(pred=pred[m])
    ic = ic_by_date(d["pred"], d[y], d["date"])
    pct = d.groupby("date")["pred"].rank(pct=True)
    ew = d.groupby("date")[y].mean()
    top = d[pct > 1 - TOP_Q].groupby("date")[y].mean()
    ex = (top - ew.reindex(top.index)).dropna()
    yr_ic = yearly_mean(ic)
    yr_net = yearly_mean(ex) - ROUND_TRIP
    return {"n_dates": int(ic.size), "mean_ic": float(ic.mean()), "t_nw": nw_tstat(ic, lag=H - 1),
            "ic_hit": float((ic > 0).mean()), "top_excess": float(ex.mean()), "top_net": float(ex.mean() - ROUND_TRIP),
            "top_t_nw": nw_tstat(ex, lag=H - 1), "pos_years": int((yr_net > 0).sum()), "n_years": int(yr_net.size),
            "yearly_ic": yr_ic, "yearly_net": yr_net}


def _registered() -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def _baseline() -> str:
    p = TIER3_DIR / "composite_px10y.csv"
    if not p.exists():
        return "없음"
    c = pd.read_csv(p)
    r = c[(c["kind"] == "walkforward") & (c["horizon"] == H)]
    return "없음" if r.empty else f"IC {r['mean_ic'].iloc[0]:+.4f}, t_nw {r['t_nw'].iloc[0]:+.2f} ({r['segment'].iloc[0]})"


def _run_sim(pred_path: Path) -> tuple[bool | None, str]:
    """0-6 시뮬 (k=20, h=10) → 통과 여부 · 보고서 경로."""
    name = f"ml_top{SIM_K}_h{H}"
    cmd = [sys.executable, str(ROOT / "scripts" / "signal_portfolio_sim.py"), "--events", str(pred_path),
           "--event-id", "ML", "--ohlcv", "data/cache/ohlcv_px10y.parquet", "--k", str(SIM_K), "--hold", str(H),
           "--name", name]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"})
    if out.returncode != 0:
        print(out.stderr[-2000:])
        return None, "시뮬 실패"
    passed = "시뮬 통과(SPY 대비 ΔSharpe 95% 하한 > 0): 예" in out.stdout
    return passed, f"tier3/PORTFOLIO_SIM_{name}.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle-check", action="store_true", help="날짜 안에서 타깃을 섞어 OOS IC ≈ 0 확인")
    args = ap.parse_args()

    panel = load_panel("px10y")
    feats = features(panel)
    X = panel[feats].replace([np.inf, -np.inf], np.nan).groupby(panel["date"]).rank(pct=True).astype("float32")
    y = panel.groupby("date")[f"fwd_ret_{H}d"].rank(pct=True) - 0.5
    print(f"행 {len(panel):,} · 피처 {len(feats)}개: {feats}")

    if args.shuffle_check:
        rng = np.random.default_rng(0)
        y_sh = y.groupby(panel["date"]).transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        pred, _ = walk_forward(panel, X, y_sh)
        r = evaluate(panel, pred)
        print(f"[셔플 확인] OOS IC {r['mean_ic']:+.4f}, t_nw {r['t_nw']:+.2f} (≈0 이어야 정상)")
        return

    if not _registered():  # 설정은 코드에 고정 → 학습 전에 등록
        log_trials("tier3b-4", FAMILY, 1, [{
            "test_id": "(등록)", "판정": "등록",
            "메모": f"LightGBM 고정 파라미터, 피처 {len(feats)}개(_sn·close 제외), 타깃 {H}일 순위, 매년 재학습 "
                    f"OOS {YEARS[0]}~2025-09. 기준: IC t_nw≥{IC_T_MIN} · 상위10% 비용후 양수 해≥{MIN_POS_YEARS}/7 · "
                    f"시뮬 k={SIM_K} h={H} ΔSharpe 하한>0"}])

    pred, imp = walk_forward(panel, X, y)
    out = panel.loc[pred.notna(), ["date", "티커"]].assign(pred=pred[pred.notna()].astype("float32"))
    pred_path = RESEARCH_DIR / "ml_pred_px10y.parquet"
    out.to_parquet(pred_path)
    out.assign(event_id="ML", score=out["pred"])[["date", "티커", "event_id", "score"]].to_parquet(
        RESEARCH_DIR / "events_ml_px10y.parquet")

    imp_mean = (imp / imp.sum()).mean(axis=1).sort_values(ascending=False)
    imp.assign(mean_share=imp_mean).sort_values("mean_share", ascending=False).to_csv(
        TIER3_DIR / "ml_importance_px10y.csv")

    r = evaluate(panel, pred)
    sim_pass, sim_path = _run_sim(RESEARCH_DIR / "events_ml_px10y.parquet")
    ok_ic = r["t_nw"] >= IC_T_MIN
    ok_years = r["pos_years"] >= MIN_POS_YEARS
    verdict = "후보" if (ok_ic and ok_years and sim_pass) else "탈락"
    log_trials("tier3b-4", FAMILY, 1, [{"test_id": f"ML|h{H}", "t": f"{r['t_nw']:.2f}", "판정": verdict,
                                        "메모": f"IC {r['mean_ic']:+.4f}, 상위10% 비용후 {r['top_net']:+.2%}, "
                                                f"양수 해 {r['pos_years']}/{r['n_years']}, 시뮬 통과 {sim_pass}"}])

    lines = ["# 4단계 패널 ML (px10y, LightGBM)", "",
             f"- 피처 {len(feats)}개 (날짜별 순위). 제외: `_sn`(섹터가 현재 구성종목만 → 탈락 여부 누설), `close`",
             f"- 타깃: {H}일 선행수익률 날짜별 순위. 매년 1월 재학습(확장 창), 학습 끝 {H + EMBARGO}거래일 제거",
             f"- OOS {YEARS[0]}-01 ~ 2025-09 ({r['n_dates']}일). 하이퍼파라미터 고정 (사전 등록, 시도 1건) · 누적 시도 {total_trials()}건",
             f"- **판정: {verdict}**", "",
             "| 기준 | 값 | 필요 | 충족 |", "|---|---|---|---|",
             f"| OOS IC t_nw (lag {H - 1}) | {r['t_nw']:+.2f} (평균 IC {r['mean_ic']:+.4f}, 양수 날 {r['ic_hit']:.0%}) | ≥ {IC_T_MIN} | {'예' if ok_ic else '아니오'} |",
             f"| 상위 10% 비용 후 초과수익 양수 해 | {r['pos_years']}/{r['n_years']} (평균 {r['top_net']:+.2%}/{H}일, 비용 전 {r['top_excess']:+.2%}, t_nw {r['top_t_nw']:+.2f}) | ≥ {MIN_POS_YEARS}/7 | {'예' if ok_years else '아니오'} |",
             f"| 시뮬 k={SIM_K}, h={H}: SPY 대비 ΔSharpe 하한 > 0 | [{sim_path}]({Path(sim_path).name}) | > 0 | {'예' if sim_pass else '아니오'} |",
             "", f"- 비교 기준 (선형 워크포워드 합성, {H}일): {_baseline()}", "",
             "## 연도별", "", "| 연도 | 평균 IC | 상위 10% 비용 후 초과 |", "|---|---|---|"]
    for yv in r["yearly_ic"].index:
        lines.append(f"| {yv} | {r['yearly_ic'][yv]:+.4f} | {r['yearly_net'].get(yv, np.nan):+.2%} |")
    lines += ["", "## 중요도 (gain 비중, fold 평균) 상위 10", "", "| 피처 | 비중 |", "|---|---|"]
    lines += [f"| {k} | {v:.1%} |" for k, v in imp_mean.head(10).items()]
    (TIER3_DIR / "ML_REPORT_px10y.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
