#!/usr/bin/env python3
"""ISB 주봉 BO·BOTTOM (isb_w_bo_bottom_v1): 주봉 ISB 의 BUY·BO 와 ◇ BOTTOM 에서만 사고 EXIT 에서만 팔면 PT-1·SPY 보다 나은가.

계기: 2026-10-13 사용자 — "파인스크립트 주봉 기준 BUY·BO 랑 BOTTOM 때만 사고 EXIT 에만 파는 거로 테스트".
규칙 (scripts/isb_engine.bo_bottom_bars):
  매수 = 주봉 ISB(기본값, TradingView 가격)의 BUY·BO 또는 ◇ BOTTOM. 그 주 마지막 거래일 종가에 확정 → 다음 거래일 시가.
  매도 = EXIT 에서만 전량 (TRIM 은 무시).
         BO     = ISB 가 실제로 든 거래의 EXIT (차트 라벨 그대로).
         BOTTOM = ISB 는 사지 않는 정보용 표시라 EXIT 라벨이 없다 → 그 봉에서 REV 로 샀다고 보고 같은 보유 규칙
                  (손절 = 투매 저점 − 0.25 ATR, +1R 뒤 Chandelier 4 ATR (내부 트림 뒤 3 ATR), Stage 4 전환)으로 EXIT 계산.
                  보유 관리 시뮬레이터는 엔진의 모든 거래(일봉 11,902 · 주봉 4,635건)의 EXIT 를 그대로 재현함.
  한 종목 한 포지션, 보유 중 다른 신호는 무시.
사전 등록 (TRIAL_LOG family isb_w_bo_bottom_v1, 2건 — 결과 계산 전에 --register):
  WBB-solo  PIT S&P 500 구성 종목 전체: 규칙상 보유 중인 종목에 1/N, 나머지 현금 vs 같은 함수의 동일가중 매수보유
  WBB-pt1   PT-1 메일 top5(PT-1 선택 먼저) 중 규칙상 '다음 시가 보유'인 첫 종목 매수 + 규칙의 EXIT 로만 매도
            (PT-1 손절·익절·트레일링·교체 끔) vs 같은 실행의 BASE
  기간·엔진·확인 절차는 isb_vs_pt1_v1 과 같다 (2016-01-04 ~ 2025-09-30, $5,000, 3슬롯). 홀드아웃은 쓰지 않는다.
판정:
  채택 후보 = 기준 대비 verdict '채택 후보' (ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내)
              · p_개선아님 ≤ 0.05/2  (paired_bootstrap n_boot 10,000)
  관찰     = 그 밖에 ΔSharpe 95% 하한 > 0
  참고     = 3종목 계좌(시드 20, PT-1 과 같은 엔진·후보 풀), SPY·PT-1 대비, BO·BOTTOM 거래 통계 (판정에 안 씀)

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/isb_bo_bottom.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/isb_bo_bottom.py --run [--workers 3]
산출: tier3/ISB_W_BO_BOTTOM.md · tier3/isb_w_bo_bottom.csv · data/research/isb_bo_bottom_runs.parquet (git 제외)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_engine as isb  # noqa: E402
from isb_vs_pt1 import (ETF_OHLCV, MEMBERSHIP, N_BOOT, N_BOOT_REF, START, Overlay, RunResult, _cmp, _f,  # noqa: E402
                        _yearly, blended_trades, check_base, check_probe, ensure_pinned, exit_mix, overlay_specs,
                        run_many, solo_returns)
from paper_trading.evaluation import cagr, max_drawdown, sharpe, verdict  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from research_utils import DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv, log_trials, total_trials  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "isb_w_bo_bottom_v1"
STEP = "stage13"
TESTS = ("WBB-solo", "WBB-pt1")
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
SOLO3_SEEDS = tuple(range(20))
KEY = isb.BB_KEY
PREV_CSV = TIER3_DIR / "isb_vs_pt1.csv"
RUNS_PATH = RESEARCH_DIR / "isb_bo_bottom_runs.parquet"
OUT_CSV = TIER3_DIR / "isb_w_bo_bottom.csv"
REPORT = TIER3_DIR / "ISB_W_BO_BOTTOM.md"
NAMES = {"BASE": "PT-1 그대로 (기준)", "WBB-solo": "주봉 BO·BOTTOM 단독 (전 종목)",
         "WBB-pt1": "PT-1 종목 + 주봉 BO·BOTTOM 타이밍"}


def spec() -> str:
    """사전 등록 문구."""
    return (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, isb_vs_pt1_v1 과 같은 엔진·확인 절차, ISB.pine sha256 "
            f"{isb._sha(isb.PINE_PATH)[:12]} 주봉 기본값 · TradingView 가격(배당 조정 안 함)] "
            "규칙 = 주봉 BUY·BO 또는 ◇BOTTOM 에서만 매수(다음 거래일 시가), EXIT 에서만 전량 매도(TRIM 무시). "
            "BO = ISB 실제 거래의 EXIT, BOTTOM = REV 로 샀다고 보고 같은 보유 규칙(손절 투매 저점−0.25ATR·Chandelier·Stage4). "
            "WBB-solo = PIT 구성 종목 규칙 보유분 1/N·나머지 현금 vs 동일가중 매수보유; "
            "WBB-pt1 = 메일 top5(PT-1 선택 먼저) 중 규칙상 다음 시가 보유인 첫 종목 + 규칙 EXIT 로만 매도(PT-1 매도·교체 끔) vs BASE. "
            f"채택 = verdict 채택 후보(ΔSharpe 95% 하한>0 · 3구간 중 2+ · MDD 2%p 이내) · p ≤ 0.05/{N_TESTS} "
            f"(부트스트랩 {N_BOOT:,}회). 참고 = 3종목 계좌 시드 {len(SOLO3_SEEDS)}, SPY·PT-1 대비, BO·BOTTOM 거래 통계.")


def judge(stats: dict[str, float], v: str) -> str:
    """사전 등록 판정 (p ≤ 0.05/2)."""
    if v == "채택 후보" and stats and stats["p_개선아님"] <= P_BONF:
        return "채택 후보"
    if stats and stats["ΔSharpe_하한"] > 0:
        return "관찰"
    return "탈락"


def specs() -> dict[str, Overlay]:
    base = overlay_specs()
    out = {"BASE": base["BASE"], "PROBE": base["PROBE"],
           "WBB-pt1": Overlay("WBB-pt1", tf=KEY, gate="isb", sells="isb")}
    for s in SOLO3_SEEDS:
        out[f"WBB-solo3-s{s}"] = Overlay(f"WBB-solo3-s{s}", tf=KEY, selector="solo", sells="isb", seed=s)
    return out


def trade_stats(trades: pd.DataFrame, mem: Membership) -> dict[str, dict[str, Any]]:
    """BO·BOTTOM 별 거래 통계 (신호일 S&P 500 구성 종목 · 2016-01-04 ~ 개발 끝 · 청산된 거래 · 배당 제외 가격 수익)."""
    t = trades[(trades["signal_date"] >= START) & (trades["signal_date"] <= DEV_END) & (trades["exit_bar"] >= 0)]
    t = t[[tk in mem.on(f"{d:%Y-%m-%d}") for tk, d in zip(t["ticker"], t["signal_date"])]]
    out: dict[str, dict[str, Any]] = {}
    for kind, g in [("전체", t), *[(k, g) for k, g in t.groupby("sig_type")]]:
        ret = g["ret"]
        loss = -ret[ret <= 0].sum()
        out[kind] = {"n": len(g), "win": float((ret > 0).mean()) if len(g) else np.nan,
                     "pf": float(ret[ret > 0].sum() / loss) if loss > 0 else np.nan,
                     "mean": float(ret.mean()) if len(g) else np.nan,
                     "median": float(ret.median()) if len(g) else np.nan,
                     "hold": float(g["held_bars"].mean()) if len(g) else np.nan,
                     "why": " · ".join(f"{k} {v}" for k, v in g["exit_why"].value_counts().items())}
    return out


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def register() -> None:
    log = _log()
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"등록: {FAMILY} {N_TESTS}건 · p ≤ {P_BONF:.4f} · 누적 시도 {total_trials()}건")


def run(workers: int) -> None:
    log = _log()
    if not ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    already_logged = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    ensure_pinned()
    isb.build_events("W")                      # 확인용 PROBE 가 쓰는 표
    events, user_trades = isb.build_bo_bottom_events()
    sp = specs()
    first = run_many([sp["BASE"], sp["PROBE"]], workers=min(2, workers), label="확인")
    errors = [e for e in (check_base(first["BASE"]), check_probe(first["BASE"], first["PROBE"])) if e]
    if errors:
        raise SystemExit("확인 실패 — 결과를 쓰지 않는다: " + " / ".join(errors))
    res = dict(first)
    res.update(run_many([sp[k] for k in sp if k not in ("BASE", "PROBE")], workers, "BO·BOTTOM"))
    pd.DataFrame({k: v.equity for k, v in res.items()}).to_parquet(RUNS_PATH)
    finish(res, events, user_trades, log_results=not already_logged)


def finish(res: dict[str, RunResult], events: pd.DataFrame, user_trades: pd.DataFrame, log_results: bool = True,
           csv_path: Path = OUT_CSV, report_path: Path = REPORT) -> pd.DataFrame:
    """엔진 결과 → 비교·판정 · CSV · TRIAL_LOG · 보고서."""
    print("===== 비교·판정", flush=True)
    mem = Membership(load_membership(MEMBERSHIP))
    base_r = res["BASE"].equity.pct_change().dropna()
    spy_c = load_ohlcv(ETF_OHLCV)["SPY"]["Close"].pct_change()
    raw = load_ohlcv(isb.PRICE_CACHE)
    spy_open = raw["SPY"]["Open"]
    spy_oo = spy_open.shift(-1) / spy_open - 1.0
    rows: list[dict[str, Any]] = [{
        "test_id": "BASE", "이름": NAMES["BASE"], "기준": "", "Sharpe": sharpe(base_r.tolist()),
        "CAGR": cagr(base_r.tolist()), "MDD": max_drawdown(base_r.tolist()), "거래수": len(res["BASE"].trades),
        "승률": float(np.mean([t["return_pct"] > 0 for t in res["BASE"].trades])), "판정": "기준"}]

    rr = res["WBB-pt1"]
    r = rr.equity.pct_change().dropna()
    st, df = _cmp(r, base_r, N_BOOT)
    a, b = df["a"].tolist(), df["b"].tolist()
    v = verdict(st, max_drawdown(a), max_drawdown(b))
    st_spy, df_spy = _cmp(r, spy_c, N_BOOT_REF)
    blended = blended_trades(rr)
    rows.append({
        "test_id": "WBB-pt1", "이름": NAMES["WBB-pt1"], "기준": "BASE", "일수": len(df), "Sharpe": sharpe(a),
        "CAGR": cagr(a), "MDD": max_drawdown(a), "거래수": len(rr.trades),
        "승률": float(np.mean([x > 0 for x in blended])) if blended else np.nan,
        "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
        "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "", "verdict": v,
        "SPY_Sharpe": sharpe(df_spy["b"].tolist()), "ΔSharpe_SPY": st_spy.get("ΔSharpe"),
        "하한_SPY": st_spy.get("ΔSharpe_하한"), "상한_SPY": st_spy.get("ΔSharpe_상한"),
        "게이트통과율": rr.gate_pass / rr.gate_eval if rr.gate_eval else np.nan,
        "매수없는날": rr.gate_empty / rr.gate_days if rr.gate_days else np.nan, "최대비중": rr.max_weight,
        "청산이유": exit_mix(rr.trades), "판정": judge(st, v)})

    r_isb, r_ew, wsum = solo_returns(events, raw, mem)
    st, df = _cmp(r_isb, r_ew, N_BOOT)
    a, b = df["a"].tolist(), df["b"].tolist()
    v = verdict(st, max_drawdown(a), max_drawdown(b))
    st_spy, df_spy = _cmp(r_isb, spy_oo, N_BOOT_REF)
    rows.append({
        "test_id": "WBB-solo", "이름": NAMES["WBB-solo"], "기준": "동일가중", "일수": len(df), "Sharpe": sharpe(a),
        "CAGR": cagr(a), "MDD": max_drawdown(a), "평균비중": float(wsum.mean()), "기준_Sharpe": sharpe(b),
        "기준_CAGR": cagr(b), "기준_MDD": max_drawdown(b), "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"),
        "상한": st.get("ΔSharpe_상한"), "p": st.get("p_개선아님"),
        "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "", "verdict": v,
        "SPY_Sharpe": sharpe(df_spy["b"].tolist()), "ΔSharpe_SPY": st_spy.get("ΔSharpe"),
        "하한_SPY": st_spy.get("ΔSharpe_하한"), "상한_SPY": st_spy.get("ΔSharpe_상한"), "판정": judge(st, v)})
    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False)

    if not log_results:
        print("TRIAL_LOG 에 결과를 쓰지 않는다 (이미 있거나 점검 실행)")
    else:
        log_trials(STEP, FAMILY, N_TESTS, [{
            "test_id": x["test_id"], "판정": x["판정"],
            "메모": (f"Sharpe {x['Sharpe']:.2f} vs {x['기준']} "
                     f"{(table.iloc[0]['Sharpe'] if x['기준'] == 'BASE' else x.get('기준_Sharpe', np.nan)):.2f}, "
                     f"ΔSharpe {x['ΔSharpe']:+.2f} [{x['하한']:+.2f}, {x['상한']:+.2f}] p {x['p']:.4f}, 구간 {x['구간']}, "
                     f"CAGR {x['CAGR']:+.1%}, MDD {x['MDD']:.1%}")}
            for _, x in table.iterrows() if x["test_id"] in TESTS])

    solo3 = [(k, v) for k, v in res.items() if k.startswith("WBB-solo3-")]
    series = {"BASE": base_r, "WBB-pt1": r, "WBB-solo": r_isb, "동일가중": r_ew,
              "SPY": spy_c.loc[base_r.index.min():base_r.index.max()].dropna()}
    write_report(table, solo3, trade_stats(user_trades, mem), series, base_r, report_path)
    return table


def write_report(table: pd.DataFrame, solo3: list[tuple[str, RunResult]], stats: dict[str, dict[str, Any]],
                 series: dict[str, pd.Series], base_r: pd.Series, path: Path = REPORT) -> None:
    t = table.set_index("test_id")
    b, o, s = t.loc["BASE"], t.loc["WBB-pt1"], t.loc["WBB-solo"]
    n_adopt = int((table["판정"] == "채택 후보").sum())
    n_obs = int((table["판정"] == "관찰").sum())
    n_fail = int((table["판정"] == "탈락").sum())
    if n_adopt:
        head = f"**결론: 채택 기준을 통과한 것이 있다 — {', '.join(table.loc[table['판정'] == '채택 후보', 'test_id'])}.**"
    else:
        head = ("**결론: 주봉 BUY·BO 와 ◇BOTTOM 에서만 사고 EXIT 에서만 파는 규칙도 PT-1·동일가중보다 낫다고 볼 수 없다.** "
                f"PT-1 종목에 적용하면 Sharpe {o['Sharpe']:.2f} vs PT-1 {b['Sharpe']:.2f} "
                f"(ΔSharpe {o['ΔSharpe']:+.2f} [{o['하한']:+.2f}, {o['상한']:+.2f}]), 전 종목 단독은 {s['Sharpe']:.2f} vs 동일가중 "
                f"{s['기준_Sharpe']:.2f} (ΔSharpe {s['ΔSharpe']:+.2f} [{s['하한']:+.2f}, {s['상한']:+.2f}]).")
    sh3 = np.array([[sharpe(rr.equity.pct_change().dropna().tolist()), cagr(rr.equity.pct_change().dropna().tolist()),
                     max_drawdown(rr.equity.pct_change().dropna().tolist())] for _, rr in solo3]) if solo3 else np.zeros((0, 3))
    prev = pd.read_csv(PREV_CSV).set_index("test_id") if PREV_CSV.exists() else None
    lines = [
        f"# ISB 주봉 BO·BOTTOM ({FAMILY}) — {RUN_KW['start_date']} ~ {RUN_KW['end_date']}", "", head, "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}`, {N_TESTS}건 · p ≤ 0.05/{N_TESTS} · 부트스트랩 {N_BOOT:,}회 · 누적 시도 {total_trials()}건",
        "- 계기: 사용자 — \"파인스크립트 주봉 기준 BUY·BO 랑 BOTTOM 때만 사고 EXIT 에만 파는 거로 테스트\" (2026-10-13)",
        "- 규칙: 주봉 ISB(기본값, TradingView 기본 가격)의 BUY·BO 또는 ◇BOTTOM 에서 다음 거래일 시가에 사고, EXIT 에서만 전량 판다 (TRIM 무시)",
        "- BOTTOM 은 ISB 가 사지 않는 정보용 표시라 EXIT 라벨이 없다. 그 봉에서 ISB 가 REV 로 샀다고 보고 같은 보유 규칙"
        "(손절 투매 저점 − 0.25 ATR · Chandelier 트레일 · Stage 4 전환)으로 EXIT 를 계산했다. TradingView 에서 이 EXIT 를 보려면 "
        "'REV Market Washout Filter' 를 Off 로 두면 된다 (단, 그 뒤 라벨 흐름이 기본값과 달라질 수 있음)",
        "- 보유 관리 계산은 ISB 엔진이 실제로 낸 모든 거래(일봉 11,902 · 주봉 4,635건)의 EXIT 를 그대로 재현한다",
        f"- 채택 후보 = 기준 대비 ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내 · p ≤ {P_BONF:.3f}",
        f"- **결과: 채택 후보 {n_adopt}건 · 관찰 {n_obs}건 · 탈락 {n_fail}건**", "",
        "## 판정", "",
        "| 변형 | Sharpe | CAGR | MDD | 기준 | 기준 대비 ΔSharpe [95%] | p | 구간 | SPY 대비 ΔSharpe [95%] | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| BASE {NAMES['BASE']} | {b['Sharpe']:.2f} | {b['CAGR']:+.1%} | {b['MDD']:.1%} | | | | | | 기준 |",
        f"| WBB-pt1 {o['이름']} | {o['Sharpe']:.2f} | {o['CAGR']:+.1%} | {o['MDD']:.1%} | PT-1 BASE | {o['ΔSharpe']:+.2f} "
        f"[{o['하한']:+.2f}, {o['상한']:+.2f}] | {o['p']:.4f} | {o['구간']} | {_f(o['ΔSharpe_SPY'], '+.2f')} "
        f"[{_f(o['하한_SPY'], '+.2f')}, {_f(o['상한_SPY'], '+.2f')}] | **{o['판정']}** |",
        f"| WBB-solo {s['이름']} | {s['Sharpe']:.2f} | {s['CAGR']:+.1%} | {s['MDD']:.1%} | 동일가중 {s['기준_Sharpe']:.2f} · "
        f"{s['기준_CAGR']:+.1%} · {s['기준_MDD']:.1%} | {s['ΔSharpe']:+.2f} [{s['하한']:+.2f}, {s['상한']:+.2f}] | {s['p']:.4f} | "
        f"{s['구간']} | {_f(s['ΔSharpe_SPY'], '+.2f')} [{_f(s['하한_SPY'], '+.2f')}, {_f(s['상한_SPY'], '+.2f')}] | **{s['판정']}** |",
        "",
        f"- WBB-pt1: 거래 {int(o['거래수'])}건 · 승률 {_f(o['승률'], '.0%')} · 게이트 통과율 {_f(o['게이트통과율'], '.0%')} · "
        f"매수 없는 날 {_f(o['매수없는날'], '.0%')} · 매수 시 최대 비중 {_f(o['최대비중'], '.0%')} · 청산 {o['청산이유']}",
        f"- WBB-solo: 평균 주식 비중 {s['평균비중']:.0%} (나머지 현금)",
    ]
    if prev is not None and {"W-full", "W-solo"} <= set(prev.index):
        lines.append(f"- 참고 (isb_vs_pt1_v1, 주봉 ISB 모든 신호·TRIM 포함): PT-1 종목 적용 W-full Sharpe {prev.at['W-full', 'Sharpe']:.2f}, "
                     f"전 종목 단독 W-solo {prev.at['W-solo', 'Sharpe']:.2f}")
    lines += ["", "## 참고: 이 규칙만으로 PT-1 같은 3종목 계좌 (같은 엔진·후보 풀·$5,000·3슬롯, 시드별 동시 신호 순서)", "",
              "| Sharpe 최소 / 중앙 / 최대 | CAGR 중앙 | MDD 중앙 | PT-1 Sharpe 를 넘은 시드 |", "|---|---|---|---|"]
    if len(sh3):
        lines.append(f"| {sh3[:, 0].min():.2f} / {np.median(sh3[:, 0]):.2f} / {sh3[:, 0].max():.2f} | {np.median(sh3[:, 1]):+.1%} | "
                     f"{np.median(sh3[:, 2]):.1%} | {int((sh3[:, 0] > b['Sharpe']).sum())}/{len(sh3)} |")
    lines += ["", "## 참고: BO·BOTTOM 거래 통계 (신호일 S&P 500 구성 종목, 2016-01 ~ 2025-09, 청산된 거래, 배당 제외 가격 수익, 비용 포함)", "",
              "| 신호 | 거래 | 승률 | PF | 평균 수익 | 중앙 수익 | 평균 보유(주) | 청산 이유 |", "|---|---|---|---|---|---|---|---|"]
    for kind, x in stats.items():
        lines.append(f"| {kind} | {x['n']} | {_f(x['win'], '.0%')} | {_f(x['pf'], '.2f')} | {_f(x['mean'], '+.1%')} | "
                     f"{_f(x['median'], '+.1%')} | {_f(x['hold'], '.0f')} | {x['why']} |")
    yr = pd.DataFrame({k: _yearly(v) for k, v in series.items()})
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(_f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    lines += ["", "## 한계", "",
              "- 개발 구간은 지금까지 많은 시도에 쓰였다 (isb_vs_pt1_v1 포함). 홀드아웃은 쓰지 않았다.",
              "- BOTTOM 의 EXIT 는 차트에 없는 값을 'REV 로 샀다면'으로 계산한 것이다. 실제로 차트를 보고 판단하면 다를 수 있다.",
              "- 상장폐지 종목 가격이 없는 생존 편향, PT-1 엔진의 0주 매수 특성은 isb_vs_pt1_v1 과 같다.",
              "- 결과가 좋아도 라이브 반영은 사용자 결정이다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true", help="결과 계산 전 사전 등록 (TRIAL_LOG)")
    g.add_argument("--run", action="store_true", help="BASE 확인 → 오버레이 · 3종목 계좌(병렬) → 판정 · 리포트")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    register() if args.register else run(args.workers)


if __name__ == "__main__":
    main()
