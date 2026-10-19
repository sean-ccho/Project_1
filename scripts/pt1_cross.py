#!/usr/bin/env python3
"""PT-1 엔진: 골든크로스·임박 라벨을 빼거나 TradingView 기준으로 바꾸면 PT-1 성과가 어떻게 되나 (pt1_cross_v1).

계기: 2026-10-16 사용자 — 사이트 크로스 라벨이 TradingView 차트와 다름 (scripts/ma_cross_research.py 와 같은 계획의 F3).
지금 라벨(일 EMA20/50 · 주 SMA10/40 · 월 SMA3/10, 배당 조정 가격, 주·월은 진행 중 봉)은 점수(주·월 패턴 가중치)와
CCS(상승 패턴)에 들어간다. src/ 는 고치지 않고, 점수 매기기 직전(screener.backtest.attach_signals_and_sort)에
'일봉패턴'·'주봉패턴'·'월봉패턴' 문자열의 골든 토큰('골든크로스', '골든크로스임박')만 바꿔 끼운다.

사전 등록 (TRIAL_LOG family pt1_cross_v1, 2건 — 결과 계산 전에 --register):
  P1 = 골든 토큰 제거 → 지금 라벨이 PT-1 에 도움이 되나
  P2 = TradingView 기준 토큰으로 교체 (ma_cross: 비조정 가격 · Pine EMA · 끝난 봉 · 일·주·월 같은 쌍)
       '골든크로스' = 그 봉 기준 최근 recent_window 봉 안 마지막 교차가 골든 (앱과 같은 남는 기간)
       '골든크로스임박' = GI+ (앱 임박 규칙 + 종가 > 단기선)
       주·월 문자열에 정배열(완전)·정배열(눌림목)이 있으면 골든 토큰을 넣지 않는다 (앱 규칙)
       TV 가격이 없는 종목은 토큰 없음 (= P1 과 같음) — 건수는 보고서에
  기간·엔진·설정 = PT-1 기준선 (validate_upside_walkforward.RUN_KW, 2016-01-04 ~ 2025-09-30, $5,000, 3슬롯)
판정 (paired_bootstrap 10,000회, 20일 블록):
  채택 후보 = verdict '채택 후보'(ΔSharpe 하한 > 0 · 3구간 중 2+ · MDD 2%p 이내) · p_개선아님 ≤ 0.05/2
  안 나빠짐 = ΔSharpe 하한 > −0.10 · MDD 2%p 이내   (P2 를 실거래 라벨로 바꿔도 되는지의 기준)
  그 밖 = 나빠짐 가능 (하한 ≤ −0.10) / 기각 (상한 < 0)
확인: BASE = 저장된 기준선, 래퍼를 모두 끼운 통과 모드 = BASE (isb_vs_pt1.check_base / check_probe). 다르면 멈춘다.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/pt1_cross.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/pt1_cross.py --run [--workers 3]
산출: tier3/pt1_cross.csv · data/research/pt1_cross_runs.parquet (git 제외) · 보고서는 MA_CROSS_REPORT.md 에 붙인다
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import isb_engine as isb  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
import paper_trading.backtest as bt  # noqa: E402
import screener.backtest as sbt  # noqa: E402
from ma_cross_research import SEAN_CONFIGS, TV_OHLCV, Config  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe, verdict  # noqa: E402
from research_utils import RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv, log_trials, total_trials  # noqa: E402
from screener.ma_cross import TF_RULES, cross_frame  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY, STEP = "pt1_cross_v1", "ma-cross"
N_TESTS, N_BOOT = 2, 10_000
P_BONF = 0.05 / N_TESTS
NONINF_MARGIN, MDD_TOL = -0.10, 0.02
GOLDEN = ("골든크로스", "골든크로스임박")
UPTREND_DONE = ("정배열(완전)", "정배열(눌림목)")
PAT_COL = {"D": "일봉패턴", "W": "주봉패턴", "M": "월봉패턴"}
RUNS_PATH = RESEARCH_DIR / "pt1_cross_runs.parquet"
OUT_CSV = TIER3_DIR / "pt1_cross.csv"
VARIANTS = {"P1": "골든 토큰 제거", "P2": "TradingView 기준 토큰"}


def p2_configs() -> dict[str, Config]:
    """P2 에 쓸 봉별 조합: 차트의 주황·보라 (EMA10/20). SeanEMA(pine/ema.pine)는 교차 규칙이 없어 SEAN_CONFIGS 가 비어 있다."""
    by_tf = {c.tf: c for c in SEAN_CONFIGS}
    return {tf: by_tf.get(tf, Config(tf, 10, 20, "ema")) for tf in ("D", "W", "M")}


# ── 토큰 표 ────────────────────────────────────────────────────────────────────────────


def _ticker_tokens(args: tuple[str, pd.Series, dict[str, Config], pd.DatetimeIndex]) -> dict[str, pd.Series]:
    """한 종목: 봉별 일간 토큰 코드 (0 없음 · 1 골든크로스 · 2 골든크로스임박), 그날까지 끝난 봉 기준."""
    ticker, close, cfgs, index = args
    out: dict[str, pd.Series] = {}
    close = close.dropna()
    if close.size < 30:
        return out
    for tf, cfg in cfgs.items():
        f = cross_frame(close, cfg.short, cfg.long, cfg.kind, tf, close.index[-1])
        f = f[f["complete"] & (f["n_bar"] >= cfg.long - 1)]
        rw = TF_RULES[tf].recent_window
        code = np.where(f["golden"] & (f["bars_since"] >= 0) & (f["bars_since"] < rw), 1,
                        np.where(f["gi_plus"], 2, 0)).astype("int8")
        s = pd.Series(code, index=f.index)
        out[tf] = s.reindex(s.index.union(index)).ffill().reindex(index).fillna(0).astype("int8")
    return out


def build_tokens(workers: int = 6) -> dict[str, pd.DataFrame]:
    """봉별 (날짜 × 종목) 토큰 코드 표. 엔진 기간의 거래일 기준."""
    cfgs = p2_configs()
    path = token_path(cfgs)
    if path.exists():
        return pd.read_pickle(path)
    px = load_ohlcv(TV_OHLCV)
    close = px.xs("Close", axis=1, level="Price").astype("float64")
    index = close.loc[RUN_KW["start_date"]:RUN_KW["end_date"]].index
    jobs = [(t, close[t], cfgs, index) for t in close.columns]
    per: dict[str, dict[str, pd.Series]] = {tf: {} for tf in cfgs}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (t, *_), res in zip(jobs, ex.map(_ticker_tokens, jobs, chunksize=8)):
            for tf, s in res.items():
                per[tf][t] = s
    tokens = {tf: pd.DataFrame(d) for tf, d in per.items()}
    pd.to_pickle(tokens, path)
    return tokens


def token_path(cfgs: dict[str, Config]) -> Path:
    """조합이 바뀌면 다른 파일 (SeanEMA 규칙을 넣은 뒤 예전 표를 잘못 쓰지 않게)."""
    tag = "_".join(c.key.replace("/", "-") for c in cfgs.values())
    return RESEARCH_DIR / f"pt1_cross_tokens_{tag}.pkl"


def rewrite(pat: Any, add: str | None, tf: str) -> str:
    """패턴 문자열에서 골든 토큰을 빼고 add 를 넣는다 (주·월은 정배열 완전·눌림목이면 넣지 않음)."""
    s = "" if pat is None or (isinstance(pat, float) and np.isnan(pat)) else str(pat)
    if s.lower() == "nan":
        s = ""
    toks = [t for t in s.split(", ") if t and t not in GOLDEN]
    if add and not (tf != "D" and any(t in UPTREND_DONE for t in toks)):
        toks.insert(0, add)
    return ", ".join(toks)


def make_attach(variant: str, ctx: iv.RunCtx, tokens: dict[str, pd.DataFrame] | None, orig: Any,
                stats: dict[str, int]) -> Any:
    """screener.backtest.attach_signals_and_sort 자리에 끼울 함수."""
    names = {1: GOLDEN[0], 2: GOLDEN[1]}

    def attach(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or variant == "PROBE":
            return orig(df)
        df = df.copy()
        day = ctx.date
        for tf, col in PAT_COL.items():
            if col not in df.columns:
                continue
            if variant == "P1":
                df[col] = [rewrite(p, None, tf) for p in df[col]]
                continue
            tab = tokens[tf]
            row = tab.loc[day] if day in tab.index else None
            new = []
            for t, p in zip(df["티커"].astype(str), df[col]):
                code = int(row.get(t, 0)) if row is not None else 0
                if row is None or t not in tab.columns:
                    stats["no_tv"] += 1
                new.append(rewrite(p, names.get(code), tf))
            df[col] = new
        stats["calls"] += 1
        return orig(df)

    return attach


def run_variant(variant: str) -> iv.RunResult:
    """엔진 한 번. 매수·매도·교체는 PT-1 그대로, 점수 매기기 직전 토큰만 바꾼다."""
    tokens = build_tokens() if variant == "P2" else None
    ov = iv.Overlay(f"X-{variant}")
    ctx = iv.RunCtx(ov, None)
    stats = {"calls": 0, "no_tv": 0}
    orig = sbt.attach_signals_and_sort
    sink = io.StringIO()
    try:
        with iv._patched(ctx), contextlib.redirect_stdout(sink):
            sbt.attach_signals_and_sort = make_attach(variant, ctx, tokens, orig, stats)
            bt.select_top_candidates = iv._ORIG["select_top_candidates"]       # ISB 래퍼 대신 원래 함수
            bt.check_sell_conditions = iv._ORIG["check_sell_conditions"]
            bt._close_position = iv._ORIG["_close_position"]
            bt.intraday_stop_exit = iv._ORIG["intraday_stop_exit"]
            result = bt.run_paper_trading_backtest(**RUN_KW)
    finally:
        sbt.attach_signals_and_sort = orig
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    if variant != "PROBE" and stats["calls"] == 0:
        raise RuntimeError(f"{variant}: 토큰 래퍼가 한 번도 안 불렸다")
    res = iv.RunResult(ov.key, eq, result["trades"], result["summary"], [], 0, 0, 0, 0, ctx.max_weight, 0, 0.0)
    res.summary = {**res.summary, "_calls": stats["calls"], "_no_tv": stats["no_tv"]}
    return res


# ── 등록·실행 ──────────────────────────────────────────────────────────────────────────


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame()


def _registered() -> bool:
    log = _log()
    return bool(len(log)) and bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def register() -> None:
    if _registered():
        raise SystemExit(f"{FAMILY} 은 이미 등록됐다")
    cfgs = p2_configs()
    log_trials(STEP, FAMILY, N_TESTS, [{
        "test_id": "(등록)", "판정": "등록",
        "메모": "P1 골든 토큰 제거 · P2 TV 토큰 (" + ", ".join(c.key for c in cfgs.values()) + ", 임박=GI+). "
                f"채택 후보 p ≤ {P_BONF:.3f} · 안 나빠짐 = 하한 > {NONINF_MARGIN} & MDD {MDD_TOL:.0%}p"}])
    print(f"등록: {FAMILY} {N_TESTS}건 · 누적 시도 {total_trials()}")


def judge(st: dict[str, float], vd: str, mdd_a: float, mdd_b: float) -> str:
    """사전 등록 판정."""
    if not st:
        return "데이터 부족"
    if vd == "채택 후보" and st["p_개선아님"] <= P_BONF:
        return "채택 후보"
    if st["ΔSharpe_하한"] > NONINF_MARGIN and mdd_a >= mdd_b - MDD_TOL:
        return "안 나빠짐"
    if st["ΔSharpe_상한"] < 0:
        return "기각"
    return "나빠짐 가능"


def run(workers: int) -> None:
    if not _registered():
        raise SystemExit("먼저 --register")
    iv.ensure_pinned()
    isb.build_events("W")  # 통과 모드(PROBE) 가 쓰는 표
    build_tokens()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {"BASE": ex.submit(iv.run_overlay, iv.overlay_specs()["BASE"]),
                "PROBE": ex.submit(iv.run_overlay, iv.overlay_specs()["PROBE"]),
                **{v: ex.submit(run_variant, v) for v in VARIANTS}}
        res = {k: f.result() for k, f in futs.items()}
    errors = [e for e in (iv.check_base(res["BASE"]), iv.check_probe(res["BASE"], res["PROBE"])) if e]
    if errors:
        raise SystemExit("확인 실패 — 결과를 쓰지 않는다: " + " / ".join(errors))
    pd.DataFrame({k: r.equity for k, r in res.items()}).to_parquet(RUNS_PATH)
    base_r = res["BASE"].equity.pct_change().dropna()
    rows = [{"test_id": "BASE", "이름": "PT-1 그대로", "Sharpe": sharpe(base_r.tolist()), "CAGR": cagr(base_r.tolist()),
             "MDD": max_drawdown(base_r.tolist()), "거래수": len(res["BASE"].trades), "판정": "기준"}]
    log_rows = []
    for v, name in VARIANTS.items():
        r = res[v].equity.pct_change().dropna()
        st, df = iv._cmp(r, base_r, N_BOOT)
        a, b = df["a"].tolist(), df["b"].tolist()
        vd = verdict(st, max_drawdown(a), max_drawdown(b))
        jd = judge(st, vd, max_drawdown(a), max_drawdown(b))
        overlap = _trade_overlap(res["BASE"].trades, res[v].trades)
        rows.append({"test_id": v, "이름": name, "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a),
                     "거래수": len(res[v].trades), "같은 거래 비율": overlap, "ΔSharpe": st.get("ΔSharpe"),
                     "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"), "p": st.get("p_개선아님"),
                     "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}", "verdict": vd, "판정": jd,
                     "래퍼 호출": res[v].summary.get("_calls"), "TV 가격 없음(행)": res[v].summary.get("_no_tv")})
        log_rows.append({"test_id": v, "t": f"{st.get('ΔSharpe', float('nan')):+.3f}", "판정": jd,
                         "메모": f"ΔSharpe {st.get('ΔSharpe'):+.2f} [{st.get('ΔSharpe_하한'):+.2f}, "
                                 f"{st.get('ΔSharpe_상한'):+.2f}] p {st.get('p_개선아님'):.4f}, 같은 거래 {overlap:.0%}"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log_trials(STEP, FAMILY, N_TESTS, log_rows)
    print(table.to_string(index=False))


def _trade_overlap(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> float:
    """BASE 거래 중 변형에서도 똑같이 (종목·진입일) 나온 비율."""
    ka = {(t["ticker"], t["entry_date"]) for t in a}
    kb = {(t["ticker"], t["entry_date"]) for t in b}
    return len(ka & kb) / len(ka) if ka else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.register:
        register()
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
