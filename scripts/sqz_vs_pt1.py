#!/usr/bin/env python3
"""스퀴즈 모멘텀 vs PT-1 엔진 (sqz_pt1_v1): PT-1 종목의 사고팔 시점을 스퀴즈 모멘텀으로 잡으면, 또는 3종목 계좌를
스퀴즈 신호로만 굴리면 PT-1 보다 나은가.

계기: 2026-10-19 사용자 — "패턴들로 백테스팅 해보고싶어 PT1 을 이기는지 … 스퀴즈 모멘텀으로 한번 시작".
scripts/sqz_research.py (F1 이벤트 · F2 전 종목 포트폴리오) 와 같은 계획의 F3. 엔진 래퍼는 isb_vs_pt1 그대로 쓴다
(src/ 는 고치지 않고 paper_trading.backtest 의 이름만 실행 중에 바꿔 끼움, BASE·통과 모드 확인 후 진행).

신호 = src/screener/squeeze_momentum.py (LazyBear SQZMOM_LB 기본값). 가격 = PT-1 엔진 가격 캐시를 TradingView 기본
(배당 조정 안 함)으로 되돌린 것 (isb_engine.load_daily_prices("tv") — ISB 연구와 같음). 끝난 봉 → 다음 거래일 시가.
주봉 신호는 그 주 마지막 거래일 종가에 확정된다.

사전 등록 (TRIAL_LOG family sqz_pt1_v1, 4건 — 결과 계산 전에 --register):
  D-entry / W-entry  매수: [PT-1 실제 선택] + [나머지 top5 를 CCS 순] 중 그 봉(일·주)의 마지막 끝난 봉 막대가 연두
                     (모멘텀 > 0 이고 오름)인 첫 종목 (없으면 그날 안 삼). 매도는 PT-1 규칙 그대로.
  D-exit  / W-exit   PT-1 매수·매도 그대로 + 보유 중 그 봉 모멘텀이 꺾이면(직전 봉 오름 → 이번 봉 떨어짐: 연두→초록,
                     갈색→빨강) 다음 시가 전량 매도. 같은 날 우선순위: 스퀴즈 매도 > PT-1 매도.
  월봉은 PT-1 보유 기간(수 주) 안에 봉이 거의 안 바뀌어 넣지 않는다 (F1·F2 에는 있음).
  기간·엔진·설정 = PT-1 기준선 (validate_upside_walkforward.RUN_KW, 2016-01-04 ~ 2025-09-30, $5,000, 3슬롯).
판정 (paired_bootstrap 10,000회, 20일 블록):
  채택 후보 = BASE 대비 verdict '채택 후보' (ΔSharpe 95% 하한 > 0 · 3구간 중 2+ · MDD 2%p 이내) · p_개선아님 ≤ 0.05/4
  관찰     = 그 밖에 ΔSharpe 95% 하한 > 0
참고 (판정·시도 수 없음):
  3종목 계좌 = PT-1 과 같은 엔진·후보 풀에서 그날 F2 규칙(일·주·월 × FIRE·MOM0·TURN, FADE 청산)의 새 진입 종목을
              무작위 순서(시드 10)로 사고, 규칙이 청산하면 판다 (PT-1 손절·익절·트레일링·교체 끔).
              → PT-1 과 같은 조건(3종목·같은 비용·같은 국면 슬롯 제한)에서의 맞대결.
  무작위 게이트 위약 = *-entry 와 같은 통과율로 아무 종목이나 통과 (시드 10).  SPY 대비.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/sqz_vs_pt1.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/sqz_vs_pt1.py --run [--workers 4]
  PYTHONPATH=.:src .venv/bin/python scripts/sqz_vs_pt1.py --report      # F1·F2·F3 → tier3/SQZ_REPORT.md
산출: tier3/sqz_pt1.csv · tier3/sqz_solo3.csv · data/research/sqz_book_*.parquet · sqz_pt1_runs.parquet (git 제외)
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_engine as isb  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe, verdict  # noqa: E402
from research_utils import DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv, log_trials, total_trials  # noqa: E402
from screener.squeeze_momentum import ENTRIES, squeeze_frame, system_position  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY, STEP = "sqz_pt1_v1", "sqz"
TESTS = ("D-entry", "W-entry", "D-exit", "W-exit")
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
N_BOOT, N_BOOT_REF = 10_000, 1_000
SOLO3_SEEDS = tuple(range(10))
PLACEBO_SEEDS = tuple(range(10))
SOLO_RULES = [f"{tf}-{e.upper()}" for tf in ("D", "W", "M") for e in ENTRIES]
OUT_CSV, SOLO3_CSV = TIER3_DIR / "sqz_pt1.csv", TIER3_DIR / "sqz_solo3.csv"
RUNS_PATH = RESEARCH_DIR / "sqz_pt1_runs.parquet"
REPORT = TIER3_DIR / "SQZ_REPORT.md"
NAMES = {"BASE": "PT-1 그대로 (기준)",
         "D-entry": "일봉 연두일 때만 매수", "W-entry": "주봉 연두일 때만 매수",
         "D-exit": "일봉 모멘텀 꺾이면 매도 추가", "W-exit": "주봉 모멘텀 꺾이면 매도 추가"}


def spec() -> str:
    return (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 엔진·13y PIT S&P 500·$5,000·3슬롯, SQZMOM_LB 기본값 "
            "(20·2.0·20·1.5·TR, 원본 dev=multKC), TradingView 가격(배당 조정 안 함), 끝난 봉 → 다음 시가] "
            "*-entry = 메일 top5(PT-1 선택 먼저) 중 마지막 끝난 봉 연두(val>0·오름)인 첫 종목 매수(없으면 안 삼) + PT-1 매도; "
            "*-exit = PT-1 그대로 + 보유 중 모멘텀 꺾임(직전 오름→이번 떨어짐) 다음 시가 전량(스퀴즈>PT-1). "
            f"채택 = verdict 채택 후보 · p ≤ 0.05/{N_TESTS} (부트스트랩 {N_BOOT:,}). 참고 = 3종목 계좌(F2 9규칙 × 시드 "
            f"{len(SOLO3_SEEDS)}), 무작위 게이트 위약 시드 {len(PLACEBO_SEEDS)}, SPY.")


# ============================================================
# 조회 표 (IsbBook 과 같은 모양: date · ticker · hold_next · exit_full · trim_ev · t1_fill · buy_ev · grade)
# ============================================================
def _book_path(key: str) -> Path:
    return RESEARCH_DIR / f"sqz_book_{key}.parquet"


def ticker_books(ticker: str, daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """한 종목: 일·주 타이밍 표 (SQZ-D, SQZ-W) + 규칙별 3종목 계좌 표 (SQZ-D-FIRE …), 엔진 일봉 날짜로."""
    out: dict[str, pd.DataFrame] = {}
    days = daily.index
    as_of = days[-1]
    for tf in ("D", "W", "M"):
        f = squeeze_frame(daily, tf, as_of)
        f = f[f["complete"]]
        if f.empty:
            continue

        def to_days(bar_values: np.ndarray, ffill: bool) -> np.ndarray:
            s = pd.Series(bar_values, index=f.index)
            s = s.reindex(days).ffill() if ffill else s.reindex(days)
            return s.fillna(False).to_numpy(dtype=bool)

        if tf in ("D", "W"):
            lime = (f["color"] == "lime").to_numpy() & f["valid"].to_numpy()
            turn_down = f["falling"].to_numpy() & np.r_[False, f["rising"].to_numpy()[:-1]]
            out[f"SQZ-{tf}"] = pd.DataFrame({"date": days, "ticker": ticker,
                                             "hold_next": to_days(lime, True), "exit_full": to_days(turn_down, False)})
        for e in ENTRIES:
            pos = system_position(f, e)
            new = pos & ~np.r_[False, pos[:-1]]
            out[f"SQZ-{tf}-{e.upper()}"] = pd.DataFrame({"date": days, "ticker": ticker,
                                                        "hold_next": to_days(pos, True), "exit_full": False,
                                                        "buy_ev": to_days(new, False)})
    for k, df in out.items():
        df["trim_ev"] = False
        df["t1_fill"] = np.float32(np.nan)
        if "buy_ev" not in df:
            df["buy_ev"] = False
        df["grade"] = np.int8(1)
    return out


def _books_worker(args: tuple[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return ticker_books(*args)


def build_books(workers: int) -> list[str]:
    """전 종목 표를 만들어 저장. Returns 표 이름 목록."""
    prices = isb.load_daily_prices("tv")
    acc: dict[str, list[pd.DataFrame]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_books_worker, list(prices.items()), chunksize=8):
            for k, df in res.items():
                acc.setdefault(k, []).append(df[df["date"] >= iv.START - pd.Timedelta(days=10)])
    for k, parts in acc.items():
        pd.concat(parts, ignore_index=True).to_parquet(_book_path(k))
    print(f"[sqz] 표 {len(acc)}개 저장 (종목 {len(prices)})", flush=True)
    return sorted(acc)


_BOOKS: dict[str, iv.IsbBook] = {}


def _book(key: str) -> iv.IsbBook:
    """표 하나만 메모리에 둔다 (워커마다 실행을 표 순서로 받으므로 바뀔 때만 다시 읽는다)."""
    if key not in _BOOKS:
        _BOOKS.clear()
        _BOOKS[key] = iv.IsbBook(pd.read_parquet(_book_path(key)))
    return _BOOKS[key]


# ============================================================
# 실행
# ============================================================
def overlay_specs() -> dict[str, iv.Overlay]:
    base = iv.overlay_specs()
    out = {"BASE": base["BASE"], "PROBE": base["PROBE"]}
    for tf in ("D", "W"):
        out[f"{tf}-entry"] = iv.Overlay(f"{tf}-entry", tf=f"SQZ-{tf}", gate="isb", sells="pt1")
        out[f"{tf}-exit"] = iv.Overlay(f"{tf}-exit", tf=f"SQZ-{tf}", gate="none", sells="both")
    return out


def solo3_specs() -> list[iv.Overlay]:
    return [iv.Overlay(f"{r}-solo3-s{s}", tf=f"SQZ-{r}", selector="solo", sells="isb", seed=s)
            for r in SOLO_RULES for s in SOLO3_SEEDS]


def placebo_specs(rates: dict[str, float]) -> list[iv.Overlay]:
    return [iv.Overlay(f"{tf}-entry-rand-s{s}", tf=f"SQZ-{tf}", gate="random", sells="pt1", seed=s, pass_rate=rates[tf])
            for tf in ("D", "W") for s in PLACEBO_SEEDS]


def _chain(specs: list[iv.Overlay]) -> list[iv.RunResult]:
    """한 워커에서 차례로 (후보표 재사용으로 두 번째부터 빠르다). 'SQZ-' 표는 이 스크립트의 표를 쓴다."""
    iv.enable_snapshot_cache()
    out = []
    for ov in specs:
        book = _book(ov.tf) if ov.tf and ov.tf.startswith("SQZ-") else None
        res = iv.run_overlay(ov, book=book)
        iv._progress("sqz", res, len(out) + 1, len(specs))
        out.append(res)
    return out


def run_chains(specs: list[iv.Overlay], workers: int, label: str) -> dict[str, iv.RunResult]:
    print(f"===== 엔진 {label}: {len(specs)}회 (워커 {workers})", flush=True)
    specs = sorted(specs, key=lambda ov: ov.tf or "")
    cuts = np.linspace(0, len(specs), workers + 1).round().astype(int)
    groups = [specs[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    out: dict[str, iv.RunResult] = {}
    with ProcessPoolExecutor(max_workers=len(groups)) as ex:
        futs = [ex.submit(_chain, g) for g in groups]
        for fut in as_completed(futs):
            for res in fut.result():
                out[res.key] = res
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
    already = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    iv.ensure_pinned()
    isb.build_events("W")  # 통과 모드(PROBE)가 쓰는 ISB 표
    build_books(workers)
    sp = overlay_specs()
    first = iv.run_many([sp["BASE"], sp["PROBE"]], workers=2, label="확인")
    errors = [e for e in (iv.check_base(first["BASE"]), iv.check_probe(first["BASE"], first["PROBE"])) if e]
    if errors:
        raise SystemExit("확인 실패 — 결과를 쓰지 않는다: " + " / ".join(errors))
    res = dict(first)
    res.update(run_chains([sp[k] for k in TESTS] + solo3_specs(), workers, "오버레이·3종목 계좌"))
    rates = {tf: res[f"{tf}-entry"].gate_pass / max(1, res[f"{tf}-entry"].gate_eval) for tf in ("D", "W")}
    res.update(run_chains(placebo_specs(rates), workers, "위약"))
    pd.DataFrame({k: v.equity for k, v in res.items()}).to_parquet(RUNS_PATH)
    finish(res, log_results=not already)


# ============================================================
# 판정
# ============================================================
def judge(st: dict[str, float], v: str) -> str:
    if v == "채택 후보" and st and st["p_개선아님"] <= P_BONF:
        return "채택 후보"
    if st and st["ΔSharpe_하한"] > 0:
        return "관찰"
    return "탈락"


def _ret(res: iv.RunResult) -> pd.Series:
    return res.equity.pct_change().dropna()


def finish(res: dict[str, iv.RunResult], log_results: bool = True) -> None:
    print("===== 비교·판정", flush=True)
    base_r = _ret(res["BASE"])
    spy_c = load_ohlcv(iv.ETF_OHLCV)["SPY"]["Close"].pct_change()
    rows: list[dict[str, Any]] = [{"test_id": "BASE", "이름": NAMES["BASE"], "Sharpe": sharpe(base_r.tolist()),
                                   "CAGR": cagr(base_r.tolist()), "MDD": max_drawdown(base_r.tolist()),
                                   "거래수": len(res["BASE"].trades), "판정": "기준"}]
    for k in TESTS:
        rr = res[k]
        r = _ret(rr)
        st, df = iv._cmp(r, base_r, N_BOOT)
        a, b = df["a"].tolist(), df["b"].tolist()
        v = verdict(st, max_drawdown(a), max_drawdown(b))
        st_spy, df_spy = iv._cmp(r, spy_c, N_BOOT_REF)
        same = {iv._trade_key(t)[:2] for t in rr.trades} & {iv._trade_key(t)[:2] for t in res["BASE"].trades}
        rand = [sharpe(_ret(res[f"{k[0]}-entry-rand-s{s}"]).tolist()) for s in PLACEBO_SEEDS
                if f"{k[0]}-entry-rand-s{s}" in res] if k.endswith("entry") else []
        rows.append({"test_id": k, "이름": NAMES[k], "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a),
                     "거래수": len(rr.trades), "같은 거래": len(same) / max(1, len(res["BASE"].trades)),
                     "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
                     "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "",
                     "verdict": v, "SPY_Sharpe": sharpe(df_spy["b"].tolist()),
                     "게이트통과율": rr.gate_pass / rr.gate_eval if rr.gate_eval else np.nan,
                     "매수없는날": rr.gate_empty / rr.gate_days if rr.gate_days else np.nan,
                     "위약_중앙": float(np.median(rand)) if rand else np.nan,
                     "위약_범위": f"{min(rand):.2f}~{max(rand):.2f}" if rand else "",
                     "청산이유": iv.exit_mix(rr.trades), "판정": judge(st, v)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    s3 = []
    base_sh = sharpe(base_r.tolist())
    for rule in SOLO_RULES:
        runs = [res[f"{rule}-solo3-s{s}"] for s in SOLO3_SEEDS if f"{rule}-solo3-s{s}" in res]
        if not runs:
            continue
        sh = [sharpe(_ret(x).tolist()) for x in runs]
        cg = [cagr(_ret(x).tolist()) for x in runs]
        md = [max_drawdown(_ret(x).tolist()) for x in runs]
        s3.append({"rule": rule, "Sharpe_중앙": float(np.median(sh)), "Sharpe_최소": min(sh), "Sharpe_최대": max(sh),
                   "PT1보다_높은_시드": sum(x > base_sh for x in sh), "시드": len(sh),
                   "CAGR_중앙": float(np.median(cg)), "MDD_중앙": float(np.median(md)),
                   "거래_중앙": float(np.median([len(x.trades) for x in runs]))})
    pd.DataFrame(s3).to_csv(SOLO3_CSV, index=False, encoding="utf-8-sig")
    if log_results:
        log_trials(STEP, FAMILY, N_TESTS, [{"test_id": r["test_id"], "t": f"{r['ΔSharpe']:+.3f}", "판정": r["판정"],
                                            "메모": f"ΔSharpe {r['ΔSharpe']:+.2f} [{r['하한']:+.2f}, {r['상한']:+.2f}] "
                                                    f"p {r['p']:.4f}"} for r in rows[1:]])
    pd.set_option("display.width", 250)
    print(table[["test_id", "Sharpe", "CAGR", "MDD", "거래수", "ΔSharpe", "하한", "상한", "p", "구간", "위약_중앙",
                 "판정"]].to_string(index=False))
    print(pd.DataFrame(s3).to_string(index=False))


# ============================================================
# 보고서 (F1·F2·F3 → SQZ_REPORT.md)
# ============================================================
SUMMARY: list[str] = [  # 결과를 본 뒤 쓴 쉬운 말 요약 (2026-10-19 실행)
    "1. **계산은 TradingView 와 같다.** LazyBear 기본값을 Pine 그대로 옮겼고, 사용자가 AAPL·MSFT·KO 일·주·월 값을 "
    "차트 범례와 맞춰 봤다 (2026-10-19).",
    "2. **신호 뒤 수익(①) 9건 모두 탈락.** FIRE·0선 돌파·빨강→갈색 어느 것도 같은 날 S&P 500 평균보다 낫지 않다 "
    "(가장 나은 월봉 FIRE 도 126일 +0.35%, t +0.55).",
    "3. **규칙대로 사고팔기(②) 9건 모두 탈락.** 일봉은 1년에 108~135번 갈아타 비용만 연 11~13% (Sharpe −0.16~0.18). "
    "주봉은 들고 있기보다 낮다 (0.35~0.57 vs 0.78). 월봉은 늘 60~145종목을 들고 있게 돼서 들고 있기와 같아진다 "
    "(0.77~0.82) — PT-1(0.62)보다 높아 보이는 건 분산 효과이고, 2006~2015 에서는 월봉도 들고 있기보다 낮았다.",
    "4. **거래 하나하나는 + 다** (주봉 TURN 거래당 +2.6%, 월봉 TURN +13.6%). TradingView 전략 테스터로 보면 돈 버는 것처럼 "
    "보이지만, 그동안 그냥 들고 있었으면 더 벌었다.",
    "5. **사용자 규칙 '검은 십자로 눌려 있다가 연두'도 탈락.** 주봉은 63일 −0.45% (t −2.47, 10년 중 9년 마이너스, "
    "2006~2015 도 −0.56%)로 오히려 약간 나쁜 쪽이고, 월봉은 효과 없음. KO 월봉 FIRE 3번 모두 좋았던 건 10년에 3번이라 "
    "운과 구분이 안 된다 (전 종목 월봉 FIRE 955번 평균 +0.35%).",
    "6. **PT-1 에 붙여도(③) 4건 모두 탈락.** 연두일 때만 사면 0.52·0.43 으로 PT-1(0.62)보다 낮고 같은 통과율의 무작위 "
    "게이트(0.44·0.51)와 다르지 않다. 일봉 모멘텀이 꺾이면 팔기는 확실히 나쁘다 (0.12, ΔSharpe 상한 −0.12). 주봉 꺾이면 "
    "팔기는 PT-1 과 거의 같다 (0.60, MDD −33% vs −44% 는 좋지만 운과 구분 안 됨). (청산 이유 'ISB매도' = 이 연구의 스퀴즈 매도 — "
    "ISB 연구 래퍼를 그대로 써서 붙은 이름)",
    "7. **3종목 계좌 맞대결도 PT-1 을 못 이긴다.** 9개 규칙 모두 시드 10개 중앙 Sharpe 가 PT-1 보다 낮다 (가장 나은 일봉 TURN "
    "0.57, 월봉 FIRE 0.54).",
    "8. **그래서:** 스퀴즈 모멘텀은 매매 규칙으로 쓰지 않는다. 사이트에는 차트와 같은 값을 **표시만** 한다 "
    "(web_export/sqz_signals.py, 점수·CCS·메일에 안 넣음). 입력값을 이 구간에서 다시 고르지 않는다.",
]
TF_NAME = {"D": "일봉", "W": "주봉", "M": "월봉"}
EV_NAME = {"fire": "FIRE 스퀴즈 풀림 + 연두", "mom0": "MOM0 0선 상향 돌파", "turn": "TURN 빨강 → 갈색"}


def _read(name: str) -> pd.DataFrame | None:
    path = TIER3_DIR / name
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def _p(v: Any, signed: bool = True) -> str:
    return "" if v is None or pd.isna(v) else (f"{v:+.2%}" if signed else f"{v:.0%}")


def _ci(d: Any, lo: Any, hi: Any) -> str:
    return "" if pd.isna(d) else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def write_report() -> None:
    f1, f2, desc = _read("sqz_events.csv"), _read("sqz_solo.csv"), _read("sqz_describe.csv")
    f1c, f2c = _read("sqz_events_confirm.csv"), _read("sqz_solo_confirm.csv")
    f3, s3 = _read("sqz_pt1.csv"), _read("sqz_solo3.csv")
    if f1 is None or f2 is None:
        raise SystemExit("sqz_research.py --run 먼저")
    prim = f1[f1["primary"]]
    ew = f2.iloc[0]
    base = f3[f3["test_id"] == "BASE"].iloc[0] if f3 is not None else None
    L = ["# 스퀴즈 모멘텀 백테스트 — TradingView SQZMOM_LB 와 같은 계산, 일·주·월봉 (sqz_v1)", "",
         "- 기간 2016-01-04 ~ 2025-09-30 (개발 구간, 홀드아웃 2025-10-01~ 안 씀) · 그날 S&P 500 구성 종목",
         "- 지표 = `src/screener/squeeze_momentum.py`: LazyBear 'Squeeze Momentum Indicator' 기본값 (BB 20 · KC 20 × 1.5 · "
         "TrueRange, 원본 그대로 BB 폭도 1.5σ). Pine 을 한 줄씩 따라 계산한 참조 구현과 값·십자·막대 색이 같다 (tests/test_squeeze_momentum.py)",
         "- 가격 = TradingView 기본 차트 (분할만 반영, 배당 조정 안 함) · 주봉 월~금 · 월봉 달력 월 · **끝난 봉** 종가 신호 → "
         "다음 거래일 시가. 수익은 배당 포함, 비용 편도 0.1%",
         "- 진입: **FIRE** = 검은 십자 → 회색(스퀴즈 풀림) + 연두 막대 (LazyBear 설명의 Carter 규칙) · **MOM0** = 막대 0선 "
         "상향 돌파 · **TURN** = 빨강 → 갈색 (바닥에서 꺾임). 청산(②·3종목 계좌): **FADE** = 막대가 떨어진 첫 봉 (색이 바뀌면 나옴)",
         f"- 사전 등록: ① `sqz_events_v1` 9건 (z 2.77) · ② `sqz_solo_v1` 9건 (p ≤ 0.05/9) · ③ `sqz_pt1_v1` 4건 "
         f"(p ≤ 0.05/4) · 누적 시도 {total_trials()}건", ""]
    if SUMMARY:
        L += ["## 쉬운 말 요약", "", *SUMMARY, ""]
    L += ["## ① 신호 뒤 수익 (주 보유기간: 일 21 · 주 63 · 월 126 거래일, 같은 날 S&P 500 평균 대비)", "",
          "| 판정 | 봉 | 신호 | n | 날짜 수 | 평균 초과(비용 전) | 비용 후 | 적중률 | t | 연도+ | 앞/뒤 반 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in prim.itertuples():
        L.append(f"| {r.판정} | {TF_NAME[r.tf]} | {EV_NAME[r.event]} | {r.n_events} | {r.n_days} | {_p(r.mean_ar)} | "
                 f"{_p(r.mean_net)} | {_p(r.hit, False)} | {r.t_nw:+.2f} | {r.years_pos}/{r.n_years} | "
                 f"{_p(r.first_half)} / {_p(r.second_half)} |")
    L += ["", "참고 — 보유기간별 비용 후 초과수익 (t):", "", "| 규칙 | 21일 | 63일 | 126일 |", "|---|---|---|---|"]
    for rule, g in f1.groupby("rule", sort=False):
        cells = [f"{_p(x['mean_net'].iloc[0])} ({x['t_nw'].iloc[0]:+.1f})" for h in (21, 63, 126)
                 for x in [g[g["h"] == h]] if len(x)]
        L.append(f"| {rule} | " + " | ".join(cells) + " |")
    L += ["", "## ② 규칙대로 사고팔기 — 신호 난 종목을 전부 동일가중 (FADE 청산)", "",
          f"- 비교: 구성 종목 동일가중 들고 있기 Sharpe {ew.Sharpe_EW:.2f} · CAGR {_p(ew.CAGR_EW)} · MDD {_p(ew.MDD_EW)} / "
          f"PT-1 {ew.Sharpe_PT1:.2f} / SPY {ew.Sharpe_SPY:.2f}",
          "- '거래당 평균' = 진입 봉 종가 → 청산 봉 종가 (TradingView 전략 테스터가 보여 주는 값과 비슷한 것). + 여도 들고 "
          "있기보다 적으면 소용없다", "",
          "| 판정 | 규칙 | Sharpe | CAGR | MDD | vs 들고 있기 ΔSharpe [95%] | vs PT-1 ΔSharpe [95%] | 평균 종목 | 회전율/년 | "
          "거래 | 보유 봉(중앙) | 거래당 평균 | 승률 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in f2.itertuples():
        L.append(f"| {r.판정} | {r.test_id} | {r.Sharpe:.2f} | {_p(r.CAGR)} | {_p(r.MDD)} | "
                 f"{_ci(r.ΔEW, r.ΔEW_하한, r.ΔEW_상한)} | {_ci(r.ΔPT1, r.ΔPT1_하한, r.ΔPT1_상한)} | {r.평균종목:.0f} | "
                 f"{r.회전율_연:.0f}× | {r.거래} | {r.보유봉_중앙:.0f} | {_p(r.봉수익_평균)} | {_p(r.승률, False)} |")
    le, ls = _read("sqz_lime_events.csv"), _read("sqz_lime_solo.csv")
    lec, lsc = _read("sqz_lime_events_confirm.csv"), _read("sqz_lime_solo_confirm.csv")
    if le is not None and ls is not None:
        L += ["", "### 추가: 사용자 규칙 '검은 십자로 눌려 있다가 연두로 바뀔 때' (LIME_SQZ, ①·② 결과를 본 뒤 따로 사전 등록)", "",
              "- `sqz_lime_events_v1` 2건 (z 2.24) · `sqz_lime_solo_v1` 2건 (p ≤ 0.05/2). 직전 봉 검은 십자 + 이번 봉 막대가 "
              "연두로 바뀜 (직전 봉 연두 아님), 주봉·월봉. 나머지 정의는 ①·② 와 같다.", "",
              "| 판정 ① | 규칙 | n | 비용 후 초과 | t | 연도+ | 판정 ② | Sharpe (들고 있기) | vs 들고 있기 ΔSharpe [95%] | "
              "평균 종목 | 2006~2015 참고: ① 비용 후 (t) · ② Sharpe (들고 있기) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        lp = le[le["primary"]].set_index("rule")
        lcp = lec[lec["primary"]].set_index("rule") if lec is not None else None
        lsc_i = lsc.set_index("test_id") if lsc is not None else None
        for r in ls.itertuples():
            e = lp.loc[r.test_id]
            ref = ""
            if lcp is not None and lsc_i is not None and r.test_id in lcp.index:
                c, s = lcp.loc[r.test_id], lsc_i.loc[r.test_id]
                ref = f"{_p(c['mean_net'])} ({c['t_nw']:+.2f}) · {s['Sharpe']:.2f} ({s['Sharpe_EW']:.2f})"
            L.append(f"| {e['판정']} | {r.test_id} | {int(e['n_events'])} | {_p(e['mean_net'])} | {e['t_nw']:+.2f} | "
                     f"{int(e['years_pos'])}/{int(e['n_years'])} | {r.판정} | {r.Sharpe:.2f} ({r.Sharpe_EW:.2f}) | "
                     f"{_ci(r.ΔEW, r.ΔEW_하한, r.ΔEW_상한)} | {r.평균종목:.0f} | {ref} |")
    if f3 is not None and base is not None:
        L += ["", "## ③ PT-1 엔진 (2016-01-04 ~ 2025-09-30, $5,000 · 3슬롯)", "",
              "| 판정 | 변형 | Sharpe | CAGR | MDD | 거래 | PT-1 과 같은 거래 | ΔSharpe [95%] | p | 구간 | 무작위 게이트 위약 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        L.append(f"| 기준 | PT-1 그대로 | {base.Sharpe:.2f} | {_p(base.CAGR)} | {_p(base.MDD)} | {int(base.거래수)} | | | | | |")
        for _, r in f3[f3["test_id"] != "BASE"].iterrows():
            plc = f"{r['위약_중앙']:.2f} ({r['위약_범위']})" if not pd.isna(r["위약_중앙"]) else ""
            L.append(f"| **{r['판정']}** | {r['test_id']} {r['이름']} | {r['Sharpe']:.2f} | {_p(r['CAGR'])} | "
                     f"{_p(r['MDD'])} | {int(r['거래수'])} | {_p(r['같은 거래'], False)} | "
                     f"{_ci(r['ΔSharpe'], r['하한'], r['상한'])} | {r['p']:.3f} | {r['구간']} | {plc} |")
    if s3 is not None and base is not None:
        L += ["", f"### 참고: 3종목 계좌 맞대결 — PT-1 과 같은 엔진·후보 풀·비용, 신호 종목을 무작위 순서로 (시드 {len(SOLO3_SEEDS)})", "",
              f"| 규칙 | Sharpe 중앙 (최소~최대) | PT-1({base.Sharpe:.2f})보다 높은 시드 | CAGR 중앙 | MDD 중앙 | 거래 중앙 |",
              "|---|---|---|---|---|---|"]
        for r in s3.itertuples():
            L.append(f"| {r.rule} | {r.Sharpe_중앙:.2f} ({r.Sharpe_최소:.2f}~{r.Sharpe_최대:.2f}) | {r.PT1보다_높은_시드}/{r.시드} | "
                     f"{_p(r.CAGR_중앙)} | {_p(r.MDD_중앙)} | {r.거래_중앙:.0f} |")
    if f1c is not None and f2c is not None:
        L += ["", "## 참고: 2006-01 ~ 2015-12 (판정·시도 기록에 안 씀 — 사전 등록상 확인할 후보가 없었다)", "",
              f"- 들고 있기 Sharpe {f2c['Sharpe_EW'].iloc[0]:.2f}. 상장폐지 종목이 더 많이 빠져 생존 편향이 크다.", "",
              "| 규칙 | ① 비용 후 (t) | ② Sharpe | ② vs 들고 있기 ΔSharpe [95%] |", "|---|---|---|---|"]
        pc = f1c[f1c["primary"]].set_index("rule")
        for r in f2c.itertuples():
            e = pc.loc[r.test_id]
            L.append(f"| {r.test_id} | {_p(e['mean_net'])} ({e['t_nw']:+.2f}) | {r.Sharpe:.2f} | "
                     f"{_ci(r.ΔEW, r.ΔEW_하한, r.ΔEW_상한)} |")
    if desc is not None:
        L += ["", "### 신호 빈도 (설명용)", "", "| 규칙 | 신호 / 100종목·년 | FIRE 직전 스퀴즈 길이(봉, 중앙) |", "|---|---|---|"]
        for r in desc.itertuples():
            L.append(f"| {r.rule} | {r.per_100_ticker_years:.0f} | {'' if pd.isna(r.sqz_bars_median) else f'{r.sqz_bars_median:.0f}'} |")
    L += ["", "## 한계", "",
          "- LazyBear 판(가장 많이 쓰는 것) 기본값만 봤다. 다른 스퀴즈 지표(TTM Squeeze Pro 의 KC 3단계 등)나 다른 입력값은 "
          "안 봤다 — 입력값을 이 구간에서 고르면 과적합이 된다.",
          "- 야후에 가격이 없는 상장폐지 종목은 빠진다 (2006~2025 S&P 500 943종목 중 638종목) → 결과가 좋은 쪽으로 기운다.",
          "- 신호는 끝난 봉 기준. 진행 중인 주·월봉으로 보면 같은 날 색이 다를 수 있다.",
          "- 개발 구간은 그동안 많은 시도에 쓰였다. 결과가 좋아도 라이브 반영은 사용자 결정이다."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:60]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--books", action="store_true", help="표만 만들기 (성과 계산 없음)")
    g.add_argument("--report", action="store_true", help="F1·F2·F3 CSV → SQZ_REPORT.md")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.register:
        register()
    elif args.books:
        build_books(args.workers)
    elif args.report:
        write_report()
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
