#!/usr/bin/env python3
"""내 매수 방법 (my_picks_v1): 사용자가 내계좌에서 직접 산 방식(골든크로스 라벨 + 최근 상승, 점수 높은 순)을
규칙으로 옮겨 사면 PT-1 보다 나은가.

계기: 2026-10-22 사용자 — "내계좌 5종목(샘플 제외)은 매수이유를 보고 직접 샀다. 1~2일 승률이 꽤 좋다.
그 방법으로 사면 PT1 보다 좋은 결과를 가져올 수 있을까?" 매수 메모(사이트 매수 팝업이 그때 신호를 자동으로 적은 것)
5건의 공통점: 모두 일·주·월봉 패턴 중 하나 이상에 '골든크로스' 또는 '골든크로스임박' 라벨, 모두 20일 수익률 > 0,
3건은 주·월봉에 골든 라벨. 점수(★ = 매수적합도)는 1.0 ~ 13.0 으로 제각각이고, RSI 과열·실적 임박도 가리지 않았다.
'차트가 좋아 보임'은 규칙으로 옮길 수 없다 → 같은 규칙 안에서 고르는 순서를 무작위로 바꾼 실행으로 운의 폭을 본다.
(공개 저장소라 실제 종목은 적지 않는다.)

규칙 (그날까지 알 수 있던 값만 — PT-1 백테스트 엔진의 날짜별 후보표, 그날 S&P 500 구성 종목):
  골든 봉   = '일봉패턴'·'주봉패턴'·'월봉패턴' 중 '골든크로스'·'골든크로스임박' 토큰이 있는 봉 (사이트·메모와 같은 라벨)
  MY1       = 골든 봉 1개 이상 · 20일수익률 > 0
  MY2       = 주봉 또는 월봉이 골든 봉 · 20일수익률 > 0
  순서      = 매수적합도(★) 높은 순 (동점은 티커순) — 사이트 차트추천 목록 순서
  제외      = 지수 ETF · 보유 중 · 보유 종목과 같은 회사. PT-1 의 다른 필터(점수 문턱·RSI 과열·실적 임박·약세장 신규 중단 등)는
              쓰지 않는다 (사용자 매수가 그것들을 가리지 않았다)
사전 등록 (TRIAL_LOG family my_picks_v1, 4건 — 결과 계산 전에 --register):
  MY1-pt1 · MY2-pt1  PT-1 엔진·계좌($5,000 · 3슬롯 · 약세장 2슬롯 · 하루 1건): 규칙에 맞는 종목 중 순서 첫 종목을 산다
                     (없으면 안 삼). 매도는 PT-1 규칙 그대로(전략구분 기준 익절·손절·트레일링·보유일), 교체 끔. vs PT-1 BASE
  MY1-ev · MY2-ev    신호 품질: 규칙에 새로 든 날 다음 시가부터 63거래일 보유하는 달력시간 포트폴리오
                     (종목당 ≤ 5%, 남는 비중 SPY, 편도 0.1%) vs SPY + 이벤트 통계(63일 초과수익 vs 그날 구성 종목 동일가중, NW t)
  판정: 채택 후보 = verdict '채택 후보'(ΔSharpe 95% 하한 > 0 · 3구간 중 2+ · MDD 2%p 이내) · p ≤ 0.05/4,
        -ev 는 이벤트 판정도 '후보'. 관찰 = 그 밖에 ΔSharpe 하한 > 0 (-ev 는 이벤트 '후보'·'관찰'도). 나머지 탈락
  참고 (판정 없음): MY1 고르는 순서 무작위 20개 시드 (운의 폭), 21거래일 보유, SPY 대비,
        '무작위 5종목이 1~2일 뒤 4개 이상 오를 확률' (짧은 승률이 얼마나 흔한지)
기간 2016-01-04 ~ 2025-09-30 (홀드아웃 안 씀). 확인: BASE = 저장된 기준선, 래퍼 통과 모드 = BASE (아니면 멈춘다).
한계: 사용자 5종목 중 여럿은 S&P 500 밖(나스닥·NYSE) 종목이지만 여기선 그날 S&P 500 구성 종목만 (생존 편향 없는 자료가 그것뿐).

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/my_picks_research.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/my_picks_research.py --run [--workers 3]
  PYTHONPATH=.:src .venv/bin/python scripts/my_picks_research.py --finish     # 저장된 엔진 결과로 보고서만 다시
  PYTHONPATH=.:src .venv/bin/python scripts/my_picks_research.py --followup   # 사후 확인: MY2 무작위 순서 (판정 없음)
  PYTHONPATH=.:src .venv/bin/python scripts/my_picks_research.py --shift      # 사후 확인 2: 시작일 밀기 (판정 없음)
  PYTHONPATH=.:src .venv/bin/python scripts/my_picks_research.py --score      # 사후 확인 3: 점수 1등 vs 나머지 (판정 없음)
  PYTHONPATH=.:src .venv/bin/python scripts/my_picks_research.py --sleeves    # 사후 확인 4: 매일 1등 묶음 vs SPY (판정 없음)
산출: tier3/MY_PICKS_REPORT.md · tier3/my_picks.csv · data/research/my_picks_* (git 제외)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_engine as isb  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
import paper_trading.backtest as bt  # noqa: E402
from idea_batch_research import active_from_entries, calendar_time_returns, event_stats, member_mask  # noqa: E402
from paper_trading import candidate_selector as cs  # noqa: E402
from paper_trading.config_override import config_overrides  # noqa: E402
from paper_trading.evaluation import cagr, max_drawdown, sharpe, verdict  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from pead_research import grade  # noqa: E402
from research_utils import (DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, bonferroni_z, load_ohlcv,  # noqa: E402
                            log_trials, nw_tstat, total_trials)
from screener import config as scfg  # noqa: E402
from signal_portfolio_sim import open_to_open  # noqa: E402
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "my_picks_v1"
STEP = "my_picks"
TESTS = ("MY1-pt1", "MY2-pt1", "MY1-ev", "MY2-ev")
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
N_BOOT, N_BOOT_REF = iv.N_BOOT, iv.N_BOOT_REF
HOLD, HOLD_REF = 63, 21
START = iv.START
RULES: dict[str, str] = {"MY1": "골든 봉 1개 이상 + 20일 상승", "MY2": "주·월봉 골든 + 20일 상승"}
SEEDS = tuple(range(1, 21))          # MY1 고르는 순서 무작위 (참고)
GOLD = frozenset({"골든크로스", "골든크로스임박"})
PAT_COL = {"D": "일봉패턴", "W": "주봉패턴", "M": "월봉패턴"}
LISTS_PATH = RESEARCH_DIR / "my_picks_lists.parquet"
RUNS_PATH = RESEARCH_DIR / "my_picks_runs.parquet"
RES_PATH = RESEARCH_DIR / "my_picks_res.pkl"
OUT_CSV = TIER3_DIR / "my_picks.csv"
REPORT = TIER3_DIR / "MY_PICKS_REPORT.md"
NAN = float("nan")


def spec() -> str:
    """사전 등록 문구."""
    return (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, PT-1 백테스트 엔진·13y PIT S&P 500·$5,000·3슬롯] "
            "골든 봉 = 일봉·주봉·월봉패턴의 골든크로스/골든크로스임박 토큰; MY1 = 골든 봉 ≥1 · 20일수익률>0; "
            "MY2 = 주봉 또는 월봉 골든 · 20일수익률>0; 순서 = 매수적합도 높은 순(동점 티커순); "
            "제외 = 지수 ETF·보유 중·같은 회사 (PT-1 의 다른 필터 안 씀). "
            "MYn-pt1 = 순서 첫 종목 매수(없으면 안 삼) + PT-1 매도 규칙, 교체 끔 vs PT-1 BASE; "
            f"MYn-ev = 새로 든 날 다음 시가부터 {HOLD}거래일 달력시간(종목당 ≤5%, 나머지 SPY, 편도 0.1%) vs SPY + "
            "이벤트 통계(초과수익 vs 그날 구성종목 동일가중, NW t, 공통 판정). "
            f"채택 = verdict 채택 후보(ΔSharpe 95% 하한>0 · 3구간 중 2+ · MDD 2%p 이내) · p ≤ 0.05/{N_TESTS} "
            f"(부트스트랩 {N_BOOT:,}회), -ev 는 이벤트 '후보'도. 참고 = MY1 무작위 순서 {len(SEEDS)}시드, "
            f"{HOLD_REF}일 보유, SPY 대비, 무작위 5종목 1~2일 승률.")


# ============================================================
# 규칙 (순수 함수 — tests/test_my_picks_research.py)
# ============================================================
def tokens(pat: Any) -> set[str]:
    """'이중바닥, 골든크로스' → {'이중바닥', '골든크로스'} (빈 값·NaN → 빈 집합)."""
    if pat is None or (isinstance(pat, float) and math.isnan(pat)):
        return set()
    return {t.strip() for t in str(pat).split(",") if t.strip() and t.strip().lower() != "nan"}


def gold_tfs(row: pd.Series | dict[str, Any]) -> str:
    """골든 라벨이 있는 봉을 'DWM' 순서 문자열로 (없으면 '')."""
    return "".join(tf for tf, col in PAT_COL.items() if tokens(row.get(col)) & GOLD)


def eligible(rule: str, tfs: str, ret20: float) -> bool:
    """MY1 = 골든 봉 1개 이상, MY2 = 주봉 또는 월봉 골든 — 둘 다 20일수익률 > 0."""
    if not (ret20 == ret20 and ret20 > 0):
        return False
    if rule == "MY1":
        return bool(tfs)
    if rule == "MY2":
        return "W" in tfs or "M" in tfs
    raise ValueError(f"모르는 규칙 {rule}")


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return NAN
    return x if math.isfinite(x) else NAN


def pool_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """그날 후보표 → 골든 봉이 있는 종목 (지수 ETF 제외). 순서는 아직 안 매김."""
    indices = set(scfg.INDICES)
    out = []
    for idx, row in df.iterrows():
        tk = str(row.get("티커", ""))
        if not tk or tk in indices:
            continue
        tfs = gold_tfs(row)
        if not tfs:
            continue
        out.append({"idx": idx, "ticker": tk, "tfs": tfs, "ret20": _num(row.get("20일수익률")),
                    "score": _num(row.get("매수적합도")), "strategy": str(row.get("전략구분", "") or ""),
                    "star": str(row.get("매수적합도_표시", "") or ""), "sector": str(row.get("섹터", "") or ""),
                    "close": _num(row.get("close", row.get("현재가격")))})
    return out


def order_key(r: dict[str, Any], seed: int | None, day: Any) -> tuple:
    """점수 높은 순(동점 티커순). seed 를 주면 그 시드·날짜·종목으로 정해지는 무작위 순서."""
    if seed is not None:
        return (iv.stable_uniform("my_picks", seed, day, r["ticker"]), r["ticker"])
    score = r["score"] if r["score"] == r["score"] else -np.inf
    return (-score, r["ticker"])


def pick(df: pd.DataFrame, holdings: list[dict[str, Any]], rule: str, k: int, seed: int | None = None,
         day: Any = None, rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """규칙에 맞는 종목 중 순서대로 k개 (보유 중·보유 종목과 같은 회사·같은 날 같은 회사 두 번 제외)."""
    rows = pool_rows(df) if rows is None else rows
    held = {str(h["ticker"]) for h in holdings}
    cands = [r for r in rows if r["ticker"] not in held and eligible(rule, r["tfs"], r["ret20"])]
    if not cands:
        return []
    sub = df.loc[[r["idx"] for r in cands]]
    same = cs._same_issuer_mask(sub, df, held)
    cands = [r for r in cands if not bool(same.loc[r["idx"]])]
    cands.sort(key=lambda r: order_key(r, seed, day))
    picks: list[dict[str, Any]] = []
    issuers: set[str] = set()
    for r in cands:
        if len(picks) >= max(1, k):
            break
        row = df.loc[r["idx"]]
        issuer = cs._SHARE_CLASS_GROUPS.get(r["ticker"]) or cs._issuer_key(row.get("회사")) or r["ticker"]
        if issuer in issuers:
            continue
        issuers.add(issuer)
        picks.append({"ticker": r["ticker"], "entry_price": r["close"], "strategy": r["strategy"] or "골든크로스",
                      "star_rating": r["star"], "ccs_score": 0.0, "sector": r["sector"], "ccs_breakdown": {}})
    return picks


# ============================================================
# 엔진
# ============================================================
def make_selector(rule: str | None, ctx: iv.RunCtx, seed: int | None = None,
                  record: list[dict[str, Any]] | None = None) -> Callable:
    """bt.select_top_candidates 자리에 끼울 함수. rule None = 목록만 기록하고 사지 않음."""

    def select(df: pd.DataFrame, holdings: list[dict[str, Any]], k: int = 1) -> tuple[list[dict[str, Any]], dict]:
        debug: dict[str, Any] = {"regime": cs.detect_market_regime(df)}
        rows = pool_rows(df)
        if record is not None:
            record.extend({"date": ctx.date, "ticker": r["ticker"], "tfs": r["tfs"], "ret20": r["ret20"],
                           "score": r["score"]} for r in rows)
        if rule is None:
            return [], debug
        return pick(df, holdings, rule, k, seed, ctx.date, rows), debug

    return select


def run_variant(rule: str | None, seed: int | None = None,
                record: list[dict[str, Any]] | None = None) -> iv.RunResult:
    """엔진 한 번: 래퍼(가격 고정·후보표 재사용·날짜) + 내 방법 선택기 + 교체 끔. 매도는 PT-1 규칙 그대로."""
    key = f"{rule or 'lists'}" + (f"-s{seed}" if seed is not None else "")
    ctx = iv.RunCtx(iv.Overlay(f"MY-{key}"), None)
    sink = io.StringIO()
    with iv._patched(ctx), config_overrides({"PT1_REPLACE_ENABLED": False}), contextlib.redirect_stdout(sink):
        bt.select_top_candidates = make_selector(rule, ctx, seed, record)
        result = bt.run_paper_trading_backtest(**RUN_KW)
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    return iv.RunResult(key, eq, result["trades"], result["summary"], [], 0, 0, 0, 0, ctx.max_weight, 0, 0.0)


def _chain() -> dict[str, Any]:
    """한 워커에서 차례로: 래퍼 통과 모드(PROBE) → 목록 기록 → MY1 · MY2 (후보표 재사용으로 두 번째부터 빠르다)."""
    iv.enable_snapshot_cache()
    out: dict[str, Any] = {"PROBE": iv.run_overlay(iv.overlay_specs()["PROBE"])}
    print("  [체인] 통과 모드 끝", flush=True)
    rec: list[dict[str, Any]] = []
    run_variant(None, record=rec)
    out["lists"] = pd.DataFrame(rec)
    print(f"  [체인] 목록 기록 끝 ({len(rec):,}행)", flush=True)
    for rule in RULES:
        out[rule] = run_variant(rule)
        print(f"  [체인] {rule} 끝 · 거래 {len(out[rule].trades)}", flush=True)
    return out


def _seed_chain(seeds: tuple[int, ...]) -> dict[str, iv.RunResult]:
    """MY1 고르는 순서 무작위 (참고)."""
    iv.enable_snapshot_cache()
    out = {}
    for k, s in enumerate(seeds, 1):
        out[f"MY1-s{s}"] = run_variant("MY1", seed=s)
        print(f"  [시드] {k}/{len(seeds)} 끝", flush=True)
    return out


# ============================================================
# 목록 → 이벤트 (달력시간·이벤트 통계용)
# ============================================================
def qualify_matrix(lists: pd.DataFrame, rule: str, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """날짜 × 종목: 그날 규칙에 맞았는지."""
    ok = np.array([eligible(rule, t, r) for t, r in zip(lists["tfs"], lists["ret20"])], dtype=bool)
    x = lists[ok]
    m = np.zeros((len(index), len(columns)), dtype=bool)
    rows = index.get_indexer(pd.to_datetime(x["date"]))
    cols = columns.get_indexer(x["ticker"])
    keep = (rows >= 0) & (cols >= 0)
    m[rows[keep], cols[keep]] = True
    return pd.DataFrame(m, index=index, columns=columns)


def new_entries(q: pd.DataFrame) -> pd.DataFrame:
    """새로 든 날(전 거래일엔 없음) → 진입 목록 (signal_date, 티커, entry_pos = 다음 거래일 위치)."""
    fresh = q & ~q.shift(1, fill_value=False)
    st = fresh.stack(future_stack=True)
    st = st[st]
    ev = st.index.to_frame(index=False)
    ev.columns = ["signal_date", "티커"]
    ev["entry_pos"] = q.index.searchsorted(ev["signal_date"], side="right")
    return ev[ev["entry_pos"] < len(q.index)].sort_values(["signal_date", "티커"]).reset_index(drop=True)


def list_counts(lists: pd.DataFrame) -> pd.DataFrame:
    """규칙별: 맞은 종목-일, 하루 평균 종목 수, 새로 든 사건 수 (연도별) — 성과 아님."""
    x = lists[(pd.to_datetime(lists["date"]) >= START) & (pd.to_datetime(lists["date"]) <= DEV_END)].copy()
    x["date"] = pd.to_datetime(x["date"])
    days = pd.DatetimeIndex(sorted(x["date"].unique()))
    cols = pd.Index(sorted(x["ticker"].unique()))
    rows = []
    for rule, name in RULES.items():
        q = qualify_matrix(x, rule, days, cols)
        ev = new_entries(q)
        per_year = ev.groupby(ev["signal_date"].dt.year).size()
        rows.append({"규칙": rule, "이름": name, "종목일": int(q.to_numpy().sum()), "하루 평균": float(q.sum(axis=1).mean()),
                     "새로 든 사건": len(ev), **{str(y): int(per_year.get(y, 0)) for y in range(2016, 2026)}})
    return pd.DataFrame(rows)


# ============================================================
# 참고: 짧은 승률이 얼마나 흔한가
# ============================================================
def luck_table(closes: pd.DataFrame, member: pd.DataFrame, spy: pd.Series, horizons: tuple[int, ...] = (1, 2),
               n_pick: int = 5, need: int = 4) -> pd.DataFrame:
    """그날 구성 종목에서 무작위 n_pick 개를 사서 h 거래일 뒤(종가) need 개 이상 올라 있을 확률.

    그날 오른 종목 비율 p 로 이항 확률을 구해 날짜 평균 (500종목 중 5개라 복원·비복원 차이는 무시할 만하다).
    SPY 가 그 구간에 오른 날 / 내린 날로도 나눈다.
    """
    rows = []
    for h in horizons:
        fwd = closes.shift(-h) / closes - 1.0
        up = (fwd > 0).astype(float).where(member & fwd.notna())
        p = up.mean(axis=1).loc[START:DEV_END].dropna()
        prob = sum(math.comb(n_pick, j) * p ** j * (1 - p) ** (n_pick - j) for j in range(need, n_pick + 1))
        spy_up = (spy.shift(-h) / spy - 1.0).reindex(prob.index) > 0
        rows.append({"보유": h, "전체": float(prob.mean()), "SPY 오른 구간": float(prob[spy_up].mean()),
                     "SPY 내린 구간": float(prob[~spy_up].mean()), "SPY 오른 구간 비율": float(spy_up.mean()),
                     "날짜 수": int(prob.size)})
    return pd.DataFrame(rows)


# ============================================================
# 등록 · 실행 · 판정
# ============================================================
def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])


def _registered() -> bool:
    log = _log()
    return bool(((log["family"] == FAMILY) & (log["판정"] == "등록")).any())


def register() -> None:
    if _registered():
        print(f"{FAMILY} 는 이미 등록됨 — 다시 등록하지 않는다")
        return
    log_trials(STEP, FAMILY, N_TESTS, [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"등록: {FAMILY} {N_TESTS}건 · p ≤ {P_BONF:.4f} · 누적 시도 {total_trials()}건")


def run(workers: int) -> None:
    if not _registered():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    log = _log()
    already_logged = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    iv.ensure_pinned()
    isb.build_events("W")                                  # 래퍼 통과 모드(PROBE)가 쓰는 표
    print(f"===== 엔진: BASE · (통과 모드 → 목록 → MY1 · MY2) · 무작위 순서 {len(SEEDS)}시드 — 동시 {workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        f_base, f_chain = ex.submit(iv.run_overlay, iv.overlay_specs()["BASE"]), ex.submit(_chain)
        n_split = max(1, workers - 2)
        f_seeds = [ex.submit(_seed_chain, SEEDS[i::n_split]) for i in range(n_split)]
        base = f_base.result()
        print("  [BASE] 끝", flush=True)
        chain = f_chain.result()
        seeds: dict[str, iv.RunResult] = {}
        for f in f_seeds:
            seeds.update(f.result())
    errors = [e for e in (iv.check_base(base), iv.check_probe(base, chain["PROBE"])) if e]
    if errors:
        raise SystemExit("확인 실패 — 결과를 쓰지 않는다: " + " / ".join(errors))
    lists = chain["lists"]
    lists.to_parquet(LISTS_PATH, index=False)
    res = {"BASE": base, **{r: chain[r] for r in RULES}, **seeds}
    pd.DataFrame({k: r.equity for k, r in res.items()}).to_parquet(RUNS_PATH)
    pd.to_pickle(res, RES_PATH)                            # 비교·보고서만 다시 돌릴 때 (--finish)
    finish(res, lists, log_results=not already_logged)


def finish_saved() -> None:
    """저장된 엔진 결과로 비교·판정·보고서만 다시 (엔진은 안 돌린다)."""
    log = _log()
    already_logged = ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    finish(pd.read_pickle(RES_PATH), pd.read_parquet(LISTS_PATH), log_results=not already_logged)


# ============================================================
# 사후 확인 (사전 등록 아님): MY2 도 고르는 순서를 무작위로
# ============================================================
FOLLOWUP_FAMILY = "my_picks_followup_v1"
FOLLOWUP_PATH = RESEARCH_DIR / "my_picks_followup.pkl"
FOLLOWUP_MARK = "## 사후 확인: "
SHIFT_PATH = RESEARCH_DIR / "my_picks_shift.pkl"
SHIFT_MARK = "## 사후 확인 (시작일)"
SHIFTS = (1, 2, 3, 5, 8, 13, 21)     # 시작일을 이만큼 거래일 뒤로
SCORE_PATH = RESEARCH_DIR / "my_picks_score.pkl"
SCORE_MARK = "## 사후 확인 (점수 1등)"
SCORE_HOLDS = (10, 21)               # PT-1 매도(시간익절 8~13일 · 장기보유)와 비슷한 길이
SLEEVE_PATH = RESEARCH_DIR / "my_picks_sleeves.pkl"
SLEEVE_MARK = "## 사후 확인 (매일 1등 묶음)"


SUMMARY_MARK = "## 사후 확인 요약"


def _put_section(mark: str, lines: list[str], before: str = "## 한계") -> None:
    """보고서에서 mark 로 시작하는 절을 (있으면) 지우고 before 절 앞에 다시 넣는다."""
    text = REPORT.read_text(encoding="utf-8")
    if mark in text:
        head, _, rest = text.partition(mark)
        text = head + (rest[rest.find("\n## ") + 1:] if "\n## " in rest else "")
    anchor = f"\n{before}"
    i = text.find(anchor)
    REPORT.write_text(text[:i] + "\n" + "\n".join(lines) + "\n" + text[i:] if i >= 0 else text, encoding="utf-8")


def _put_summary() -> None:
    """사후 확인이 다 있으면 '쉬운 말 요약' 바로 뒤에 요약 몇 줄 (판정은 사전 등록 그대로)."""
    if not all(p.exists() for p in (FOLLOWUP_PATH, SHIFT_PATH, SCORE_PATH, SLEEVE_PATH)):
        return
    res, seeds = pd.read_pickle(RES_PATH), pd.read_pickle(FOLLOWUP_PATH)
    sc = pd.read_pickle(SCORE_PATH).set_index(["규칙", "보유"])
    sl = pd.read_pickle(SLEEVE_PATH).set_index(["규칙", "보유", "고르기"])
    sh = lambda eq: sharpe(eq.pct_change().dropna().tolist())  # noqa: E731
    s2 = pd.Series([sh(r.equity) for r in seeds.values()])
    s1 = pd.Series([sh(r.equity) for k, r in res.items() if k.startswith("MY1-s")])
    spy_c = load_ohlcv(iv.ETF_OHLCV)["SPY"]["Close"].pct_change()
    base_r = res["BASE"].equity.pct_change().dropna()
    spy_sh = sharpe(spy_c.loc[base_r.index.min():base_r.index.max()].dropna().tolist())
    m2 = sc.loc[("MY2", 10)]
    hs = SCORE_HOLDS
    top = " · ".join(f"{sl.loc[('MY2', h, 'top1'), 'Sharpe']:.2f}" for h in hs)
    pool2 = " · ".join(f"{sl.loc[('MY2', h, 'pool'), 'Sharpe']:.2f}" for h in hs)
    pool1 = " · ".join(f"{sl.loc[('MY1', h, 'pool'), 'Sharpe']:.2f}" for h in hs)
    rand2 = " · ".join(f"{sl.loc[('MY2', h, 'rand'), 'Sharpe']:.2f}" for h in hs)
    spy_s = " · ".join(f"{sl.loc[('MY2', h, 'top1'), 'SPY']:.2f}" for h in hs)
    ci = " · ".join(f"{sl.loc[('MY2', h, 'top1'), 'ΔSPY']:+.2f} [{sl.loc[('MY2', h, 'top1'), 'ΔSPY_하한']:+.2f}, "
                    f"{sl.loc[('MY2', h, 'top1'), 'ΔSPY_상한']:+.2f}]" for h in hs)
    lines = [SUMMARY_MARK + " (사전 등록 아님 — 판정은 위 그대로)", "",
             f"- **주·월봉 골든(MY2)이 일봉까지 넓힌 것(MY1)보다 낫다 — 이 표본에서는 일관됨.** 3슬롯 계좌에서 고르는 순서를 "
             f"무작위로 바꾼 20번 가운데 {s2.median():.2f} vs {s1.median():.2f}, 계좌 없이 그날 규칙 종목 전부를 "
             f"{hs[0]}·{hs[1]}일 들면 {pool2} vs {pool1}.",
             f"- **계좌 1.26 은 운이 더해진 숫자다.** 같은 MY2 종목군에서 아무거나 고르면 계좌 가운데 {s2.median():.2f} "
             f"(PT-1 {sh(res['BASE'].equity):.2f} 수준, SPY {spy_sh:.2f} 를 넘은 것 {int((s2 > spy_sh).sum())}/{len(s2)}). "
             f"시작일 확인은 경로가 다시 합쳐져서 운을 가려 주지 못했다.",
             f"- **운을 빼고 보면 '주·월봉 골든 + 점수 1등'은 SPY 와 비슷하다.** 계좌 경로 없이 매일 1등을 {hs[0]}·{hs[1]}일 든 "
             f"칸을 겹치면 Sharpe {top} vs SPY {spy_s} (차이 {ci}) · 풀 전부 {pool2} · 무작위 1개 가운데 {rand2}. "
             f"점수 1등은 같은 날 나머지보다 {hs[0]}거래일에 {m2['1등−풀']:+.2%} (t {m2['1등 t']:.2f}) 더 올랐지만 상위 3개로 "
             f"넓히면 {m2['상위3−풀']:+.2%} 로 사라진다.",
             "- **그래서:** 일봉 골든만 보고 사는 것보다 주·월봉 골든 종목이 나았고, 그중 점수 1등이면 PT-1 보다 나을 가능성은 있다. "
             "하지만 SPY 보다 낫다는 근거는 없고, 이 모든 것이 결과를 본 뒤 한 확인이다 → 새 기간(앞으로의 페이퍼·실거래)에서 "
             "다시 확인해야 한다."]
    _put_section(SUMMARY_MARK, lines, before="## 얼마나 자주 나오나")

def score_check() -> None:
    """사후 확인 3 (사전 등록 아님): 계좌 경로와 상관없이, 매일 규칙에 든 종목 중 점수 1등(·상위 3)이 같은 날 나머지 평균보다
    더 올랐나. 다음날 시가 → h 거래일 뒤 시가, 종목이 3개 이상인 날만, NW t. 판정 없음 (참고)."""
    lists = pd.read_parquet(LISTS_PATH)
    lists["date"] = pd.to_datetime(lists["date"])
    opens = load_ohlcv(isb.PRICE_CACHE).xs("Open", axis=1, level="Price").sort_index()
    rows = []
    for rule in ("MY2", "MY1"):
        ok = np.array([eligible(rule, t, r) for t, r in zip(lists["tfs"], lists["ret20"])], dtype=bool)
        x = lists[ok]
        for h in SCORE_HOLDS:
            fwd = opens.shift(-(h + 1)) / opens.shift(-1) - 1.0
            st = fwd.stack()
            st.index.names = ["date", "ticker"]
            y = x.join(st.rename("fwd"), on=["date", "ticker"]).dropna(subset=["fwd"])
            y = y.sort_values(["date", "score", "ticker"], ascending=[True, False, True])
            y["rank"] = y.groupby("date").cumcount()
            n = y.groupby("date").size()
            y = y[y["date"].isin(n[n >= 3].index)]
            pool = y.groupby("date")["fwd"].mean()
            d1 = (y[y["rank"] == 0].set_index("date")["fwd"] - pool).dropna()
            d3 = (y[y["rank"] < 3].groupby("date")["fwd"].mean() - pool).dropna()
            ic = y.groupby("date")[["score", "fwd"]].apply(lambda g: g["score"].corr(g["fwd"], method="spearman")).dropna()
            half = len(d1) // 2
            rows.append({"규칙": rule, "보유": h, "날": len(d1), "1등−풀": float(d1.mean()), "1등 t": nw_tstat(d1, h),
                         "상위3−풀": float(d3.mean()), "상위3 t": nw_tstat(d3, h), "순위상관": float(ic.mean()),
                         "순위상관 t": nw_tstat(ic, h), "앞 반": float(d1.iloc[:half].mean()), "뒤 반": float(d1.iloc[half:].mean())})
    tb = pd.DataFrame(rows)
    pd.to_pickle(tb, SCORE_PATH)
    lines = [SCORE_MARK + " vs 같은 날 나머지 — 계좌 경로와 상관없이 (사전 등록 아님, 판정 없음)", "",
             "- 왜: 계좌 한 번의 결과는 경로 운이 크다. 매일 규칙에 든 종목(3개 이상인 날)을 점수순으로 세우고, 다음날 시가에 사서 "
             "h 거래일 뒤 시가까지 수익을 같은 날 나머지 평균과 비교했다 (NW t).", "",
             "| 규칙 | 보유 | 날 | 점수 1등 − 풀 평균 (t) | 상위 3 − 풀 (t) | 점수·수익 순위상관 (t) | 앞 반 / 뒤 반 |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['규칙']} | {int(r['보유'])}일 | {int(r['날']):,} | {r['1등−풀']:+.2%} ({r['1등 t']:.2f}) | "
              f"{r['상위3−풀']:+.2%} ({r['상위3 t']:.2f}) | {r['순위상관']:+.3f} ({r['순위상관 t']:.2f}) | "
              f"{r['앞 반']:+.2%} / {r['뒤 반']:+.2%} |" for _, r in tb.iterrows()]
    res = pd.read_pickle(RES_PATH)
    acct = {k: sharpe(res[k].equity.pct_change().dropna().tolist()) for k in RULES}
    lines += ["", f"- 점수 1등은 조금 더 올랐지만 (t {tb['1등 t'].min():.1f} ~ {tb['1등 t'].max():.1f}) 상위 3개로 넓히면 "
              f"{tb['상위3−풀'].min():+.2%} ~ {tb['상위3−풀'].max():+.2%} 로 거의 사라진다. 여러 번 본 것 중 하나라 우연과 구분되지 않는다. "
              f"두 규칙의 점수 효과가 비슷한데 계좌 결과(MY1 {acct['MY1']:.2f} vs MY2 {acct['MY2']:.2f})가 크게 다른 것은 "
              "경로 운 때문으로 보인다."]
    _put_section(SCORE_MARK, lines)
    _put_summary()
    if not _logged("score-rank"):
        r2 = tb.set_index(["규칙", "보유"]).loc[("MY2", 10)]
        log_trials(STEP, FOLLOWUP_FAMILY, 1, [{
            "test_id": "score-rank", "t": f"{r2['1등 t']:.2f}", "판정": "참고",
            "메모": (f"사후 확인(사전 등록 아님): 매일 규칙 종목 중 점수 1등 − 같은 날 나머지 평균 (다음날 시가→h일 시가, NW t): "
                     + "; ".join(f"{r['규칙']} h{int(r['보유'])} {r['1등−풀']:+.2%} (t {r['1등 t']:.2f}), 상위3 "
                                 f"{r['상위3−풀']:+.2%} (t {r['상위3 t']:.2f})" for _, r in tb.iterrows()))}])
    print("\n".join(lines), flush=True)


def _logged(test_id: str) -> bool:
    log = _log()
    return bool(((log["family"] == FOLLOWUP_FAMILY) & (log["test_id"] == test_id)).any())


def _seed_chain_rule(rule: str, seeds: tuple[int, ...]) -> dict[str, iv.RunResult]:
    iv.enable_snapshot_cache()
    out = {}
    for k, s in enumerate(seeds, 1):
        out[f"{rule}-s{s}"] = run_variant(rule, seed=s)
        print(f"  [{rule} 시드] {k}/{len(seeds)} 끝", flush=True)
    return out


def followup(workers: int) -> None:
    """MY2-pt1 이 PT-1 보다 크게 좋았지만(사전 기준은 못 넘음) MY1 은 점수순이 무작위 순서보다 나빴다 →
    MY2 의 점수순 결과가 '고르는 순서의 운'인지 MY2 무작위 순서 20번으로 본다. 판정 없음 (참고)."""
    if not RES_PATH.exists():
        raise SystemExit("먼저 --run")
    if FOLLOWUP_PATH.exists():
        seeds = pd.read_pickle(FOLLOWUP_PATH)
    else:
        print(f"===== 사후 확인: MY2 무작위 순서 {len(SEEDS)}시드 — 동시 {workers}", flush=True)
        n = max(1, workers)
        with ProcessPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(_seed_chain_rule, "MY2", SEEDS[i::n]) for i in range(n)]
            seeds = {}
            for f in futs:
                seeds.update(f.result())
        pd.to_pickle(seeds, FOLLOWUP_PATH)
    res = pd.read_pickle(RES_PATH)
    spy_c = load_ohlcv(iv.ETF_OHLCV)["SPY"]["Close"].pct_change()
    base_r = res["BASE"].equity.pct_change().dropna()
    spy_sh = sharpe(spy_c.loc[base_r.index.min():base_r.index.max()].dropna().tolist())
    base_sh = sharpe(base_r.tolist())

    def table(prefix: str) -> pd.DataFrame:
        src = seeds if prefix == "MY2" else res
        rows = []
        for key in sorted((k for k in src if k.startswith(f"{prefix}-s")), key=lambda k: int(k.split("-s")[1])):
            r = src[key].equity.pct_change().dropna()
            tr = [t for t in src[key].trades if float(t.get("shares") or 0.0) > 0]
            rows.append({"시드": key, "Sharpe": sharpe(r.tolist()), "CAGR": cagr(r.tolist()), "MDD": max_drawdown(r.tolist()),
                         "거래당 평균": float(np.mean([t["return_pct"] for t in tr])) if tr else NAN})
        return pd.DataFrame(rows)

    t1, t2 = table("MY1"), table("MY2")
    score = {k: sharpe(res[k].equity.pct_change().dropna().tolist()) for k in RULES}
    lines = [FOLLOWUP_MARK + "고르는 순서 무작위 20번 — MY1 · MY2 (사전 등록 아님, 판정 없음)", "",
             "- 왜: MY2-pt1 이 PT-1 보다 크게 좋았지만(사전 기준은 못 넘음) MY1 은 점수순이 무작위 순서 가운데값보다 나빴다. "
             "점수순 한 번의 결과가 '고르는 순서의 운'인지 보려고 MY2 도 같은 20개 시드로 돌렸다. 결과를 보고 정한 확인이라 판정에 쓰지 않는다.",
             "", "| 규칙 | 점수순 Sharpe | 무작위 20번 최소 · 가운데 · 최대 | 80% 구간 | PT-1 넘음 | SPY 넘음 | 거래당 평균 수익 (가운데) |",
             "|---|---|---|---|---|---|---|"]
    for rule, tb in (("MY1", t1), ("MY2", t2)):
        q = tb["Sharpe"].quantile([0.1, 0.5, 0.9])
        lines.append(f"| {rule} {RULES[rule]} | {score[rule]:.2f} | {tb['Sharpe'].min():.2f} · {q.loc[0.5]:.2f} · "
                     f"{tb['Sharpe'].max():.2f} | {q.loc[0.1]:.2f} ~ {q.loc[0.9]:.2f} | "
                     f"{int((tb['Sharpe'] > base_sh).sum())}/{len(tb)} | {int((tb['Sharpe'] > spy_sh).sum())}/{len(tb)} | "
                     f"{tb['거래당 평균'].median():+.2%} |")
    lines += ["", f"- 기준: PT-1 Sharpe {base_sh:.2f} · SPY {spy_sh:.2f} (같은 기간)", "",
              "| MY2 시드 | Sharpe | CAGR | MDD | 거래당 평균 |", "|---|---|---|---|---|"]
    lines += [f"| {r['시드']} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | {r['거래당 평균']:+.2%} |"
              for _, r in t2.sort_values("Sharpe").iterrows()]
    _put_section(FOLLOWUP_MARK, lines)
    _put_summary()
    if not _logged("MY2-seeds"):
        q2 = t2["Sharpe"].quantile([0.1, 0.5, 0.9])
        log_trials(STEP, FOLLOWUP_FAMILY, 1, [{
            "test_id": "MY2-seeds", "t": "", "판정": "참고",
            "메모": (f"사후 확인(사전 등록 아님): MY2 무작위 순서 {len(t2)}시드 Sharpe 가운데 {q2.loc[0.5]:.2f} "
                     f"(80% {q2.loc[0.1]:.2f}~{q2.loc[0.9]:.2f}, 최소 {t2['Sharpe'].min():.2f}, 최대 {t2['Sharpe'].max():.2f}) "
                     f"vs 점수순 {score['MY2']:.2f} · PT-1 {base_sh:.2f} · SPY {spy_sh:.2f}; PT-1 넘음 "
                     f"{int((t2['Sharpe'] > base_sh).sum())}/{len(t2)}, SPY 넘음 {int((t2['Sharpe'] > spy_sh).sum())}/{len(t2)}")}])
    print("\n".join(lines), flush=True)


def shift_start(offset: int) -> str:
    """기준 시작일(RUN_KW)에서 offset 거래일 뒤 (엔진이 읽는 가격 캐시의 날짜)."""
    nxt = iv._engine_prices()["next"]
    d = min(k for k in nxt if k >= RUN_KW["start_date"])
    for _ in range(offset):
        d = nxt[d]
    return d


def run_shift(kind: str, offset: int) -> pd.Series:
    """시작일만 offset 거래일 미룬 엔진 한 번. kind = PT1 (PT-1 그대로) · MY1 · MY2 (점수순, 교체 끔)."""
    start = shift_start(offset)
    px = iv._engine_prices()
    px["start"] = start                      # 래퍼의 장중 날짜 확인도 같은 시작일로
    ctx = iv.RunCtx(iv.Overlay(f"SH-{kind}-{offset}"), None)
    overrides = {} if kind == "PT1" else {"PT1_REPLACE_ENABLED": False}
    sink = io.StringIO()
    try:
        with iv._patched(ctx), config_overrides(overrides), contextlib.redirect_stdout(sink):
            if kind != "PT1":
                bt.select_top_candidates = make_selector(kind, ctx)
            result = bt.run_paper_trading_backtest(**{**RUN_KW, "start_date": start})
    finally:
        px["start"] = shift_start(0)
    eq = result["equity_curve"].dropna()
    eq.index = pd.to_datetime(eq.index)
    return eq


def _shift_chain(jobs: list[tuple[str, int]]) -> dict[tuple[str, int], pd.Series]:
    iv.enable_snapshot_cache()
    out = {}
    for k, (kind, off) in enumerate(jobs, 1):
        out[(kind, off)] = run_shift(kind, off)
        print(f"  [시작일] {kind} +{off} 끝 ({k}/{len(jobs)})", flush=True)
    return out


def shift(workers: int) -> None:
    """사후 확인 2 (사전 등록 아님): 시작일을 1~21거래일 미뤄도 점수순 결과가 유지되나 — 경로 운이면 크게 흔들린다.
    PT-1 · MY1 · MY2 를 같은 시작일로 돌려 같은 기간끼리 비교한다. 판정 없음 (참고)."""
    if not RES_PATH.exists():
        raise SystemExit("먼저 --run")
    if SHIFT_PATH.exists():
        runs = pd.read_pickle(SHIFT_PATH)
    else:
        jobs = [(k, o) for o in SHIFTS for k in ("PT1", "MY1", "MY2")]
        n = max(1, workers)
        print(f"===== 사후 확인 2: 시작일 밀기 {SHIFTS} × PT-1·MY1·MY2 = {len(jobs)}회 — 동시 {n}", flush=True)
        with ProcessPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(_shift_chain, jobs[i::n]) for i in range(n)]
            runs = {}
            for f in futs:
                runs.update(f.result())
        pd.to_pickle(runs, SHIFT_PATH)
    res = pd.read_pickle(RES_PATH)
    spy_c = load_ohlcv(iv.ETF_OHLCV)["SPY"]["Close"].pct_change()
    orig = {"PT1": res["BASE"].equity, "MY1": res["MY1"].equity, "MY2": res["MY2"].equity}
    rows = []
    for off in (0, *SHIFTS):
        eqs = orig if off == 0 else {k: runs[(k, off)] for k in ("PT1", "MY1", "MY2")}
        r = {k: v.pct_change().dropna() for k, v in eqs.items()}
        lo, hi = r["PT1"].index.min(), r["PT1"].index.max()
        rows.append({"밀기": off, "시작": f"{eqs['PT1'].index[0]:%Y-%m-%d}",
                     **{k: sharpe(v.tolist()) for k, v in r.items()},
                     "SPY": sharpe(spy_c.loc[lo:hi].dropna().tolist())})
    tb = pd.DataFrame(rows)
    sh = tb[tb["밀기"] > 0]
    lines = [SHIFT_MARK + " — 시작일을 1~21거래일 미루기 (사전 등록 아님, 판정 없음)", "",
             "- 왜: 3슬롯·하루 1건 계좌는 첫 매수가 달라지면 그 뒤 매수가 줄줄이 바뀐다 (MY1·MY2 점수순은 같은 거래가 21%뿐). "
             "점수순 결과가 규칙 덕인지 그 한 번의 경로 운인지 보려고 시작일만 미뤄 같은 기간끼리 비교했다.", "",
             "| 밀기(거래일) | 시작 | PT-1 | MY1 | MY2 | SPY |", "|---|---|---|---|---|---|"]
    lines += [f"| {int(x['밀기'])}{' (본 실행)' if x['밀기'] == 0 else ''} | {x['시작']} | {x['PT1']:.2f} | {x['MY1']:.2f} | "
              f"{x['MY2']:.2f} | {x['SPY']:.2f} |" for _, x in tb.iterrows()]
    lines += ["", f"- 미룬 {len(sh)}번 가운데값: PT-1 {sh['PT1'].median():.2f} · MY1 {sh['MY1'].median():.2f} · "
              f"MY2 {sh['MY2'].median():.2f} (범위 {sh['MY2'].min():.2f} ~ {sh['MY2'].max():.2f}) · SPY {sh['SPY'].median():.2f}",
              f"- MY2 가 같은 시작일 PT-1 보다 높았던 것 {int((sh['MY2'] > sh['PT1']).sum())}/{len(sh)}, "
              f"SPY 보다 높았던 것 {int((sh['MY2'] > sh['SPY']).sum())}/{len(sh)}"]
    # 경로가 합쳐지나: 미룬 실행의 일간 수익이 본 실행과 똑같은 날 비율 · 마지막으로 달랐던 날
    conv = []
    for kind in ("PT1", "MY1", "MY2"):
        o = orig[kind].pct_change()
        c = pd.concat([o, runs[(kind, max(SHIFTS))].pct_change()], axis=1, keys=["o", "s"]).dropna()
        same = (c["o"] - c["s"]).abs() < 1e-10
        last = f"{c.index[~same][-1]:%Y-%m-%d}" if (~same).any() else "없음"
        conv.append(f"{kind} 같은 날 {same.mean():.0%} · 마지막으로 달랐던 날 {last}")
    lines += [f"- **주의 — 경로가 다시 합쳐진다:** {max(SHIFTS)}거래일 미룬 실행이 본 실행과 일간 수익까지 똑같은 날: "
              + " / ".join(conv) + ". 점수순은 매일 같은 1등을 고르니 자리가 비는 때가 맞아지면 그 뒤로는 같은 경로가 된다 → "
              "이 확인은 '그 경로의 운'을 가려 주지 못한다 (점수 1등 확인을 따로 했다)."]
    _put_section(SHIFT_MARK, lines)
    _put_summary()
    if not _logged("start-shift"):
        log_trials(STEP, FOLLOWUP_FAMILY, 1, [{
            "test_id": "start-shift", "t": "", "판정": "참고",
            "메모": (f"사후 확인(사전 등록 아님): 시작일 +{list(SHIFTS)} 거래일 — 가운데 Sharpe PT-1 {sh['PT1'].median():.2f} · "
                     f"MY1 {sh['MY1'].median():.2f} · MY2 {sh['MY2'].median():.2f} (범위 {sh['MY2'].min():.2f}~"
                     f"{sh['MY2'].max():.2f}); MY2>PT-1 {int((sh['MY2'] > sh['PT1']).sum())}/{len(sh)}, "
                     f"MY2>SPY {int((sh['MY2'] > sh['SPY']).sum())}/{len(sh)}")}])
    print("\n".join(lines), flush=True)


def judge(st: dict[str, float], v: str, ev_grade: str | None = None) -> str:
    """사전 등록 판정."""
    bonf = bool(st) and st["p_개선아님"] <= P_BONF
    if v == "채택 후보" and bonf and ev_grade in (None, "후보"):
        return "채택 후보"
    if (st and st["ΔSharpe_하한"] > 0) or ev_grade in ("후보", "관찰"):
        return "관찰"
    return "탈락"


def _win_rate(trades: list[dict[str, Any]]) -> float:
    real = [t for t in trades if float(t.get("shares") or 0.0) > 0]   # 0주 매수(약세장 2슬롯) 제외
    return float(np.mean([t["return_pct"] > 0 for t in real])) if real else NAN


def _per_trade(trades: list[dict[str, Any]]) -> tuple[float, float]:
    """거래당 수익률 평균·중앙 (0주 매수 제외) — 몇 개 큰 거래 덕인지 고르게 나은지 보려고."""
    r = [float(t["return_pct"]) for t in trades if float(t.get("shares") or 0.0) > 0]
    return (float(np.mean(r)), float(np.median(r))) if r else (NAN, NAN)


def finish(res: dict[str, iv.RunResult], lists: pd.DataFrame, log_results: bool = True) -> pd.DataFrame:
    print("===== 비교·판정", flush=True)
    mem = Membership(load_membership(iv.MEMBERSHIP))
    raw = load_ohlcv(isb.PRICE_CACHE)
    opens = raw.xs("Open", axis=1, level="Price").sort_index()
    closes = raw.xs("Close", axis=1, level="Price").sort_index()
    r_oo = open_to_open(raw)
    idx, cols = opens.index, opens.columns
    member = member_mask(idx, cols, mem)
    tradable = opens.notna()
    spy_oo = (opens["SPY"].shift(-1) / opens["SPY"] - 1.0)
    spy_close = load_ohlcv(iv.ETF_OHLCV)["SPY"]["Close"]
    spy_c = spy_close.pct_change()
    base_r = res["BASE"].equity.pct_change().dropna()
    z = bonferroni_z(N_TESTS)
    rows: list[dict[str, Any]] = [{"test_id": "BASE", "이름": "PT-1 그대로 (기준)", "종류": "엔진",
                                   "Sharpe": sharpe(base_r.tolist()), "CAGR": cagr(base_r.tolist()),
                                   "MDD": max_drawdown(base_r.tolist()), "거래수": len(res["BASE"].trades),
                                   "승률": _win_rate(res["BASE"].trades),
                                   **dict(zip(("거래당 평균", "거래당 중앙"), _per_trade(res["BASE"].trades))),
                                   "판정": "기준"}]
    spy_win = spy_c.loc[base_r.index.min():base_r.index.max()].dropna()
    rows.append({"test_id": "SPY", "이름": "SPY 들고 있기", "종류": "기준", "Sharpe": sharpe(spy_win.tolist()),
                 "CAGR": cagr(spy_win.tolist()), "MDD": max_drawdown(spy_win.tolist()), "판정": "기준"})
    series: dict[str, pd.Series] = {"PT-1": base_r, "SPY(종가)": spy_win}
    for rule, name in RULES.items():
        rr = res[rule]
        r = rr.equity.pct_change().dropna()
        tid = f"{rule}-pt1"
        st, df = iv._cmp(r, base_r, N_BOOT)
        a, b = df["a"].tolist(), df["b"].tolist()
        vd = verdict(st, max_drawdown(a), max_drawdown(b))
        st_spy, df_spy = iv._cmp(r, spy_c, N_BOOT_REF)
        days_held = (rr.equity.diff().fillna(0) != 0)
        rows.append({
            "test_id": tid, "이름": f"PT-1 계좌 · {name}", "종류": "엔진", "기준": "PT-1 BASE",
            "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a), "거래수": len(rr.trades),
            "승률": _win_rate(rr.trades), **dict(zip(("거래당 평균", "거래당 중앙"), _per_trade(rr.trades))),
            "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
            "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "",
            "verdict": vd, "SPY_Sharpe": sharpe(df_spy["b"].tolist()), "ΔSharpe_SPY": st_spy.get("ΔSharpe"),
            "하한_SPY": st_spy.get("ΔSharpe_하한"), "상한_SPY": st_spy.get("ΔSharpe_상한"),
            "주식 보유일 비율": float(days_held.loc[START:].mean()), "최대비중": rr.max_weight,
            "청산이유": iv.exit_mix(rr.trades), "판정": judge(st, vd)})
        series[tid] = r

    spy_s = spy_oo.loc[START:DEV_END].iloc[:-1]
    for rule, name in RULES.items():
        q = qualify_matrix(lists, rule, idx, cols)
        ent = new_entries(q.loc[:DEV_END])
        for hold in (HOLD, HOLD_REF):
            tid = f"{rule}-ev" + ("" if hold == HOLD else f"{hold}")
            act = active_from_entries(ent, idx, cols, hold)
            ret, n_names, w_sum = calendar_time_returns(act, r_oo, tradable, spy_oo)
            s = ret.loc[START:DEV_END].iloc[:-1]
            st, df = iv._cmp(s, spy_s, N_BOOT if tid in TESTS else N_BOOT_REF)
            a, b = df["a"].tolist(), df["b"].tolist()
            vd = verdict(st, max_drawdown(a), max_drawdown(b))
            ev = event_stats(ent, opens, member, hold)
            g = grade(ev, z)
            rows.append({
                "test_id": tid, "이름": f"신호 품질 · {name} · {hold}거래일", "종류": "달력시간", "기준": "SPY",
                "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a), "거래수": len(ent),
                "ΔSharpe": st.get("ΔSharpe"), "하한": st.get("ΔSharpe_하한"), "상한": st.get("ΔSharpe_상한"),
                "p": st.get("p_개선아님"), "구간": f"{st.get('구간_개선')}/{st.get('구간_수')}" if st else "",
                "verdict": vd, "SPY_Sharpe": sharpe(b), "평균 보유 종목": float(n_names.loc[START:DEV_END].mean()),
                "종목 비중 평균": float(w_sum.loc[START:DEV_END].mean()), "이벤트_판정": g,
                **{f"ev_{k}": val for k, val in ev.items() if k != "yearly"},
                "판정": judge(st, vd, g) if tid in TESTS else "참고"})
            if hold == HOLD:
                series[tid] = s
    series["SPY(시가)"] = spy_s

    seed_rows = []
    for key in sorted((k for k in res if k.startswith("MY1-s")), key=lambda k: int(k.split("-s")[1])):
        r = res[key].equity.pct_change().dropna()
        seed_rows.append({"시드": key, "Sharpe": sharpe(r.tolist()), "CAGR": cagr(r.tolist()),
                          "MDD": max_drawdown(r.tolist()), "거래수": len(res[key].trades)})
    seeds = pd.DataFrame(seed_rows)
    luck = luck_table(closes, member, spy_close.reindex(closes.index))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    if log_results:
        t0 = table.set_index("test_id")
        log_trials(STEP, FAMILY, N_TESTS, [{
            "test_id": x["test_id"], "t": f"{x['ev_t_nw']:.2f}" if pd.notna(x.get("ev_t_nw", np.nan)) else "",
            "판정": x["판정"],
            "메모": (f"Sharpe {x['Sharpe']:.2f} vs {x['기준']} "
                     f"{(t0.at['BASE', 'Sharpe'] if x['기준'] == 'PT-1 BASE' else x['SPY_Sharpe']):.2f}, "
                     f"ΔSharpe {x['ΔSharpe']:+.2f} [{x['하한']:+.2f}, {x['상한']:+.2f}] p {x['p']:.4f}, 구간 {x['구간']}, "
                     f"CAGR {x['CAGR']:+.1%}, MDD {x['MDD']:.1%}, 거래 {int(x['거래수'])}"
                     + (f", 이벤트 {x['이벤트_판정']}" if isinstance(x.get('이벤트_판정'), str) else ""))}
            for _, x in table.iterrows() if x["test_id"] in TESTS])
    else:
        print("TRIAL_LOG 에 결과를 쓰지 않는다 (이미 있음)", flush=True)
    write_report(table, list_counts(lists), series, seeds, luck)
    return table


def _yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def write_report(table: pd.DataFrame, counts: pd.DataFrame, series: dict[str, pd.Series], seeds: pd.DataFrame,
                 luck: pd.DataFrame, path: Path = REPORT) -> None:
    f = iv._f
    t = table.set_index("test_id")
    b, spy = t.loc["BASE"], t.loc["SPY"]
    passed = [x for x in TESTS if t.at[x, "판정"] == "채택 후보"]
    watch = [x for x in TESTS if t.at[x, "판정"] == "관찰"]
    m1, m2, e1, e2 = (t.loc[x] for x in TESTS)
    nums = (f"PT-1 계좌로 사면 Sharpe {m1['Sharpe']:.2f}(MY1) · {m2['Sharpe']:.2f}(MY2) vs PT-1 {b['Sharpe']:.2f} · "
            f"SPY {spy['Sharpe']:.2f}. 신호 자체(63거래일 보유)는 {e1['Sharpe']:.2f} · {e2['Sharpe']:.2f} vs "
            f"SPY {e1['SPY_Sharpe']:.2f}.")
    if passed:
        head = f"**결론: 사전 등록 기준을 통과한 것이 있다 — {', '.join(passed)}.** {nums}"
    else:
        head = (f"**결론: 사전 등록 기준을 통과한 규칙은 없다 — 이 방법(골든크로스 라벨 + 최근 상승, 점수 높은 순)으로 사면 "
                f"PT-1 보다 낫다고 아직 말할 수 없다.** {nums}"
                + (f" ('관찰': {', '.join(watch)} — 운과 구분되지 않음)" if watch else ""))

    def vs_base(x: pd.Series) -> str:
        if x["판정"] == "채택 후보":
            return "기준 통과"
        if x["Sharpe"] > b["Sharpe"]:
            return f"PT-1 보다 높았지만 운과 구분할 만큼은 아니다 (p {x['p']:.3f}, 기준 {P_BONF:.4f})"
        return "PT-1 보다 낮았다"

    ev_word = ("SPY 보다 낫다는 증거가 없다" if all(t.at[x, "판정"] == "탈락" for x in ("MY1-ev", "MY2-ev"))
               else "일부가 SPY 대비 기준을 넘었다 (아래 판정)")
    c = counts.set_index("규칙")
    lines = [
        f"# 내 매수 방법 백테스트 ({FAMILY}) — {RUN_KW['start_date']} ~ {RUN_KW['end_date']}", "", head, "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · {N_TESTS}건 · p ≤ 0.05/{N_TESTS} · 부트스트랩 {N_BOOT:,}회 · "
        f"누적 시도 {total_trials():,}건",
        "- 계기: 사용자 — 내계좌에서 매수이유를 보고 직접 산 5종목이 1~2일 승률이 좋았다. \"그 방법으로 사면 PT1 보다 좋을까?\" "
        "(2026-10-22). 공개 저장소라 실제 종목은 적지 않는다.",
        "- 매수 메모 5건의 공통점 → 규칙: 5건 모두 일·주·월봉 패턴 중 하나 이상에 '골든크로스(임박 포함)' 라벨이 있고 "
        "20일 수익률이 플러스였다. 3건은 주·월봉에 골든 라벨. 점수(★)는 1.0~13.0 으로 제각각, RSI 과열·실적 임박도 가리지 않았다.",
        f"  - **MY1** = {RULES['MY1']} · **MY2** = {RULES['MY2']} · 둘 다 점수(매수적합도) 높은 순 = 사이트 차트추천 순서",
        "  - 라벨은 사이트·메모와 같은 것 (일봉 EMA20/50 · 주봉 SMA10/40 · 월봉 SMA3/10, 배당 조정 가격)",
        "  - '차트가 좋아 보임'은 규칙으로 옮길 수 없다 → 같은 규칙에서 고르는 순서를 무작위로 바꾼 20번으로 운의 폭을 봤다",
        "- PT-1 계좌: $5,000 · 3슬롯(약세장 2) · 하루 1건, 규칙에 맞는 첫 종목을 사고 PT-1 매도 규칙으로 판다 (없으면 안 삼, 교체 끔). "
        "PT-1 의 다른 필터(점수 문턱·RSI 과열·실적 임박·약세장 신규 중단)는 쓰지 않았다",
        f"- 신호 품질: 규칙에 새로 든 날 다음 시가부터 {HOLD}거래일 보유 (종목당 ≤ 5%, 나머지 SPY) → SPY 와 비교",
        "- 확인: BASE 재현 · 래퍼 투명성 통과",
        f"- 채택 후보 = 기준 대비 ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내 · p ≤ {P_BONF:.4f}"
        " (신호 품질은 이벤트 판정 '후보'도)",
        "", "## 쉬운 말 요약", "",
        f"1. **PT-1 과 같은 계좌·같은 매도 규칙에서 사는 종목만 바꿨다.** MY1 Sharpe {m1['Sharpe']:.2f} — {vs_base(m1)}. "
        f"MY2 {m2['Sharpe']:.2f} — {vs_base(m2)}. (PT-1 {b['Sharpe']:.2f} · SPY {spy['Sharpe']:.2f}, 높을수록 같은 위험에 "
        f"수익이 많다.) CAGR {m1['CAGR']:+.1%} · {m2['CAGR']:+.1%} vs PT-1 {b['CAGR']:+.1%} · SPY {spy['CAGR']:+.1%}.",
        f"2. **신호 자체 (규칙에 든 종목을 63거래일 보유) → {ev_word}.** Sharpe {e1['Sharpe']:.2f} · {e2['Sharpe']:.2f} "
        f"vs SPY {e1['SPY_Sharpe']:.2f}. 같은 날 S&P 500 평균 대비 63일 초과수익(비용 뺀 값) "
        f"{f(e1.get('ev_mean_net', NAN), '+.2%')} · {f(e2.get('ev_mean_net', NAN), '+.2%')}.",
    ]
    if len(seeds):
        q = seeds["Sharpe"].quantile([0.1, 0.5, 0.9])
        lines.append(
            f"3. **같은 규칙이라도 무엇을 고르느냐(운)에 따라 결과가 크게 갈린다.** 고르는 순서만 무작위로 바꾼 20번: Sharpe "
            f"{seeds['Sharpe'].min():.2f} ~ {seeds['Sharpe'].max():.2f} (가운데 {q.loc[0.5]:.2f}, 80% 구간 {q.loc[0.1]:.2f} ~ "
            f"{q.loc[0.9]:.2f}). PT-1 을 넘은 것 {int((seeds['Sharpe'] > b['Sharpe']).sum())}/20, SPY 를 넘은 것 "
            f"{int((seeds['Sharpe'] > spy['Sharpe']).sum())}/20.")
    if len(luck):
        l1 = luck.set_index("보유")
        lines.append(
            f"4. **1~2일 승률은 거의 정보가 없다.** 무작위로 5종목을 사도 1일 뒤 4개 이상 올라 있을 확률이 "
            f"{l1.at[1, '전체']:.0%}, SPY 가 오른 날이면 {l1.at[1, 'SPY 오른 구간']:.0%}다 (2일 뒤: {l1.at[2, '전체']:.0%} · "
            f"SPY 오른 구간 {l1.at[2, 'SPY 오른 구간']:.0%}). 같은 날 시장이 오르면 대부분 같이 오른다.")
    lines += ["", "## 얼마나 자주 나오나 (2016-01 ~ 2025-09, 그날 S&P 500)", "",
              "| 규칙 | 맞은 종목-일 | 하루 평균 종목 수 | 새로 든 사건 | " + " | ".join(str(y) for y in range(2016, 2026)) + " |",
              "|---|---|---|---|" + "---|" * 10]
    for rule in RULES:
        r = c.loc[rule]
        lines.append(f"| {rule} {r['이름']} | {int(r['종목일']):,} | {r['하루 평균']:.0f} | {int(r['새로 든 사건']):,} | "
                     + " | ".join(str(int(r[str(y)])) for y in range(2016, 2026)) + " |")
    lines += ["", "## 판정", "",
              "| 검정 | Sharpe | CAGR | MDD | 거래·사건 | 승률 | 기준 | 기준 대비 ΔSharpe [95%] | p | 구간 | 이벤트 | 판정 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|",
              f"| BASE PT-1 그대로 | {b['Sharpe']:.2f} | {b['CAGR']:+.1%} | {b['MDD']:.1%} | {int(b['거래수'])} | "
              f"{f(b['승률'], '.0%')} | | | | | | 기준 |",
              f"| SPY 들고 있기 | {spy['Sharpe']:.2f} | {spy['CAGR']:+.1%} | {spy['MDD']:.1%} | | | | | | | | 기준 |"]
    for tid in TESTS:
        x = t.loc[tid]
        bench = f"PT-1 {b['Sharpe']:.2f}" if x["기준"] == "PT-1 BASE" else f"SPY {x['SPY_Sharpe']:.2f}"
        ev_g = x.get("이벤트_판정") if isinstance(x.get("이벤트_판정"), str) else ""
        lines.append(f"| {tid} {x['이름']} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | {int(x['거래수'])} | "
                     f"{f(x.get('승률', NAN), '.0%')} | {bench} | {f(x['ΔSharpe'], '+.2f')} [{f(x['하한'], '+.2f')}, "
                     f"{f(x['상한'], '+.2f')}] | {f(x['p'], '.4f')} | {x['구간']} | {ev_g} | **{x['판정']}** |")
    for tid in ("MY1-ev", "MY2-ev"):
        e = t.loc[tid]
        if pd.notna(e.get("ev_mean_ar", np.nan)):
            lines.append(f"- {tid} 이벤트: {int(e['ev_n_events'])}건 · 63거래일 평균 초과수익 {e['ev_mean_ar']:+.2%} "
                         f"(비용 뺀 {e['ev_mean_net']:+.2%}) · 중앙 {e['ev_median_ar']:+.2%} · 이긴 비율 {e['ev_hit']:.0%} · "
                         f"NW t {e['ev_t_nw']:.2f} · 플러스였던 해 {int(e['ev_years_pos'])}/{int(e['ev_n_years'])} · "
                         f"평균 보유 {e['평균 보유 종목']:.1f}종목 · 종목 비중 평균 {e['종목 비중 평균']:.0%} (나머지 SPY)")
    lines.append(f"- 거래당 수익률 (평균 · 중앙): PT-1 {f(b['거래당 평균'], '+.2%')} · {f(b['거래당 중앙'], '+.2%')}, "
                 + ", ".join(f"{x} {f(t.at[x, '거래당 평균'], '+.2%')} · {f(t.at[x, '거래당 중앙'], '+.2%')}"
                             for x in ("MY1-pt1", "MY2-pt1")))
    for tid in ("MY1-pt1", "MY2-pt1"):
        p1 = t.loc[tid]
        lines.append(f"- {tid}: 거래 {int(p1['거래수'])}건 · 승률 {f(p1['승률'], '.0%')} · 주식을 들고 있던 날 "
                     f"{f(p1['주식 보유일 비율'], '.0%')} · 청산 {p1['청산이유']} · SPY 대비 ΔSharpe "
                     f"{f(p1['ΔSharpe_SPY'], '+.2f')} [{f(p1['하한_SPY'], '+.2f')}, {f(p1['상한_SPY'], '+.2f')}]")
    lines += ["", "## 참고: 21거래일 보유 (판정 없음)", "",
              "| 규칙 | Sharpe | SPY | ΔSharpe [95%] | 이벤트 평균 초과수익(비용 뺀) · t |", "|---|---|---|---|---|"]
    for rule in RULES:
        tid = f"{rule}-ev{HOLD_REF}"
        if tid not in t.index:
            continue
        x = t.loc[tid]
        evs = f"{f(x.get('ev_mean_net', NAN), '+.2%')} · {f(x.get('ev_t_nw', NAN), '.2f')}"
        lines.append(f"| {rule} | {x['Sharpe']:.2f} | {x['SPY_Sharpe']:.2f} | {f(x['ΔSharpe'], '+.2f')} "
                     f"[{f(x['하한'], '+.2f')}, {f(x['상한'], '+.2f')}] | {evs} |")
    if len(seeds):
        lines += ["", "## 참고: MY1 고르는 순서를 무작위로 (운의 폭, 판정 없음)", "",
                  f"- 규칙에 맞는 종목은 하루 평균 {c.at['MY1', '하루 평균']:.0f}개인데 계좌는 하루 1개만 산다. "
                  "점수순 대신 시드마다 정해진 무작위 순서로 고른 20번.",
                  "", "| 시드 | Sharpe | CAGR | MDD | 거래 |", "|---|---|---|---|---|"]
        lines += [f"| {r['시드']} | {r['Sharpe']:.2f} | {r['CAGR']:+.1%} | {r['MDD']:.1%} | {int(r['거래수'])} |"
                  for _, r in seeds.sort_values("Sharpe").iterrows()]
    if len(luck):
        lines += ["", "## 참고: 무작위 5종목을 사면 1~2일 뒤 4개 이상 오를 확률", "",
                  "| 보유 | 전체 | SPY 오른 구간 | SPY 내린 구간 | SPY 오른 구간 비율 | 날짜 수 |", "|---|---|---|---|---|---|"]
        lines += [f"| {int(r['보유'])}일 | {r['전체']:.0%} | {r['SPY 오른 구간']:.0%} | {r['SPY 내린 구간']:.0%} | "
                  f"{r['SPY 오른 구간 비율']:.0%} | {int(r['날짜 수']):,} |" for _, r in luck.iterrows()]
    yr = pd.DataFrame({k: _yearly(v) for k, v in series.items()})
    lines += ["", "## 연도별 수익", "", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    lines += ["", "## 한계", "",
              "- 사용자가 산 종목 여럿은 S&P 500 밖(나스닥·NYSE)이지만, 여기선 그날 S&P 500 구성 종목만 봤다 "
              "(생존 편향 없는 자료가 그것뿐). 상장폐지 종목은 가격이 없어 빠진다.",
              "- 사용자의 매도 방법은 아직 없다 (손절가만 일부) → PT-1 매도 규칙과 63·21거래일 보유로 대신했다.",
              "- '차트가 좋아 보임'(사람 눈)은 규칙으로 옮길 수 없다. 그 눈에 실력이 있는지는 실제 거래가 쌓여야 안다 "
              "(웹 내계좌의 매수이유별 성적: 10건 이상 · |t| ≥ 2).",
              "- 개발 구간은 지금까지 많은 시도에 쓰였다. 홀드아웃은 쓰지 않았다. 결과가 좋아도 라이브 반영은 사용자 결정이다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {path}", flush=True)


def sleeve_returns(picks: pd.DataFrame, r_oo: pd.DataFrame, spy_oo: pd.Series, hold: int,
                   cost: float = 0.001) -> pd.Series:
    """매일 고른 종목 묶음 → hold 개의 칸을 겹쳐 쌓은 일간 수익 (계좌 경로가 없는 계산).

    picks: date · ticker · w (그날 칸 안 비중, 합 1). 다음날 시가에 사서 hold 거래일 뒤 시가에 판다 (active_from_entries 와 같은
    [entry, entry + hold)). 칸 하나 = 1/hold. 고른 게 없는 날의 칸과 남는 비중은 SPY. 비용 = 비중 변화 × 편도 비용.
    """
    idx, cols = r_oo.index, r_oo.columns
    col_pos = {t: j for j, t in enumerate(cols)}
    w = np.zeros((len(idx), len(cols)))
    entry = idx.searchsorted(pd.to_datetime(picks["date"]), side="right")
    for p, tk, wt in zip(entry, picks["ticker"], picks["w"]):
        j = col_pos.get(tk)
        if j is not None and p < len(idx):
            w[p:p + hold, j] += wt / hold
    W = pd.DataFrame(w, index=idx, columns=cols)
    w_spy = 1.0 - W.sum(axis=1)
    gross = (W * r_oo.fillna(0.0)).sum(axis=1) + w_spy * spy_oo.reindex(idx).fillna(0.0)
    turnover = W.diff().abs().sum(axis=1) + w_spy.diff().abs()
    return gross - turnover * cost


def daily_picks(lists: pd.DataFrame, rule: str, how: str, seed: int | None = None) -> pd.DataFrame:
    """규칙에 든 종목에서 매일: top1 = 점수 1등 · pool = 전부 같은 비중 · rand = 시드 무작위 1개."""
    ok = np.array([eligible(rule, t, r) for t, r in zip(lists["tfs"], lists["ret20"])], dtype=bool)
    x = lists[ok].copy()
    if how == "pool":
        x["w"] = 1.0 / x.groupby("date")["ticker"].transform("size")
        return x[["date", "ticker", "w"]]
    if how == "rand":
        x["k"] = [iv.stable_uniform("my_picks_sleeve", seed, d, tk) for d, tk in zip(x["date"], x["ticker"])]
        x = x.sort_values(["date", "k"])
    else:
        x = x.assign(s=x["score"].fillna(-np.inf)).sort_values(["date", "s", "ticker"], ascending=[True, False, True])
    out = x.groupby("date").head(1)[["date", "ticker"]].copy()
    out["w"] = 1.0
    return out


def sleeves() -> None:
    """사후 확인 4 (사전 등록 아님): 매일 '점수 1등'을 사서 h 거래일 든 칸을 겹쳐 쌓는다 — 3슬롯 계좌의 경로 운 없이
    '주·월봉 골든 + 점수 1등'이 SPY·풀 평균·무작위 1개보다 나은지. 판정 없음 (참고)."""
    lists = pd.read_parquet(LISTS_PATH)
    lists["date"] = pd.to_datetime(lists["date"])
    raw = load_ohlcv(isb.PRICE_CACHE)
    r_oo = open_to_open(raw)
    opens = raw.xs("Open", axis=1, level="Price").sort_index()
    spy_oo = (opens["SPY"].shift(-1) / opens["SPY"] - 1.0)
    spy_s = spy_oo.loc[START:DEV_END].iloc[:-1]
    rows, series = [], {}
    for h in SCORE_HOLDS:
        for rule in ("MY2", "MY1"):
            variants = {"top1": daily_picks(lists, rule, "top1"), "pool": daily_picks(lists, rule, "pool")}
            if rule == "MY2":
                variants.update({f"rand{s}": daily_picks(lists, rule, "rand", s) for s in SEEDS})
            rets = {k: sleeve_returns(v, r_oo, spy_oo, h).loc[START:DEV_END].iloc[:-1] for k, v in variants.items()}
            for k in ("top1", "pool"):
                st, df = iv._cmp(rets[k], spy_s, N_BOOT_REF)
                st_pool, _ = iv._cmp(rets["top1"], rets["pool"], N_BOOT_REF) if k == "top1" else ({}, None)
                a = df["a"].tolist()
                rows.append({"규칙": rule, "보유": h, "고르기": k, "Sharpe": sharpe(a), "CAGR": cagr(a), "MDD": max_drawdown(a),
                             "SPY": sharpe(df["b"].tolist()), "ΔSPY": st.get("ΔSharpe"), "ΔSPY_하한": st.get("ΔSharpe_하한"),
                             "ΔSPY_상한": st.get("ΔSharpe_상한"), "Δ풀": st_pool.get("ΔSharpe"),
                             "Δ풀_하한": st_pool.get("ΔSharpe_하한"), "Δ풀_상한": st_pool.get("ΔSharpe_상한")})
            if rule == "MY2":
                rs = pd.Series([sharpe(rets[f"rand{s}"].tolist()) for s in SEEDS])
                rows.append({"규칙": rule, "보유": h, "고르기": "rand", "Sharpe": float(rs.median()),
                             "무작위 최소": float(rs.min()), "무작위 최대": float(rs.max()),
                             "1등보다 높은 무작위": int((rs >= sharpe(rets["top1"].tolist())).sum())})
                series[f"MY2 1등 · {h}일"] = rets["top1"]
    tb = pd.DataFrame(rows)
    pd.to_pickle(tb, SLEEVE_PATH)
    t = tb.set_index(["규칙", "보유", "고르기"])
    f = iv._f
    lines = [SLEEVE_MARK + " — 계좌 경로 없이 (사전 등록 아님, 판정 없음)", "",
             "- 왜: 3슬롯 계좌는 경로 운이 크다. 매일 규칙 종목 중 점수 1등을 다음날 시가에 사서 h 거래일 들고, 그런 칸 h 개를 "
             "겹쳐 쌓았다 (칸 = 1/h, 고른 게 없는 칸은 SPY, 편도 0.1%). 같은 방식으로 '그날 규칙 종목 전부'(풀)와 "
             f"'무작위 1개'({len(SEEDS)}시드)도 만들어 비교했다.", "",
             "| 규칙 | 보유 | 고르기 | Sharpe | CAGR | MDD | SPY | SPY 대비 ΔSharpe [95%] | 풀 대비 ΔSharpe [95%] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (rule, h, k), x in t.iterrows():
        if k == "rand":
            lines.append(f"| {rule} | {h}일 | 무작위 1개 (가운데) | {x['Sharpe']:.2f} | | | | 범위 {x['무작위 최소']:.2f} ~ "
                         f"{x['무작위 최대']:.2f} · 1등 이상 {int(x['1등보다 높은 무작위'])}/{len(SEEDS)} | |")
            continue
        name = "점수 1등" if k == "top1" else "풀 전부"
        lines.append(f"| {rule} | {h}일 | {name} | {x['Sharpe']:.2f} | {x['CAGR']:+.1%} | {x['MDD']:.1%} | {x['SPY']:.2f} | "
                     f"{f(x['ΔSPY'], '+.2f')} [{f(x['ΔSPY_하한'], '+.2f')}, {f(x['ΔSPY_상한'], '+.2f')}] | "
                     + (f"{f(x['Δ풀'], '+.2f')} [{f(x['Δ풀_하한'], '+.2f')}, {f(x['Δ풀_상한'], '+.2f')}]" if k == "top1" else "")
                     + " |")
    yr = pd.DataFrame({**{k: _yearly(v) for k, v in series.items()}, "SPY(시가)": _yearly(spy_s)})
    lines += ["", "| 연도 | " + " | ".join(yr.columns) + " |", "|---" * (len(yr.columns) + 1) + "|"]
    lines += [f"| {y} | " + " | ".join(f(v, "+.1%") for v in row) + " |" for y, row in yr.iterrows()]
    _put_section(SLEEVE_MARK, lines)
    _put_summary()
    if not _logged("daily-top1-sleeves"):
        m = {h: t.loc[("MY2", h, "top1")] for h in SCORE_HOLDS}
        log_trials(STEP, FOLLOWUP_FAMILY, 1, [{
            "test_id": "daily-top1-sleeves", "t": "", "판정": "참고",
            "메모": ("사후 확인(사전 등록 아님): 매일 MY2 점수 1등 h일 칸 겹치기 — "
                     + "; ".join(f"h{h} Sharpe {m[h]['Sharpe']:.2f} vs SPY {m[h]['SPY']:.2f} "
                                 f"(ΔSharpe {m[h]['ΔSPY']:+.2f} [{m[h]['ΔSPY_하한']:+.2f}, {m[h]['ΔSPY_상한']:+.2f}]), "
                                 f"풀 대비 {m[h]['Δ풀']:+.2f} [{m[h]['Δ풀_하한']:+.2f}, {m[h]['Δ풀_상한']:+.2f}]"
                                 for h in SCORE_HOLDS))}])
    print("\n".join(lines), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--finish", action="store_true", help="저장된 엔진 결과로 비교·보고서만 다시")
    g.add_argument("--followup", action="store_true", help="사후 확인: MY2 무작위 순서 (판정 없음)")
    g.add_argument("--shift", action="store_true", help="사후 확인 2: 시작일 밀기 (판정 없음)")
    g.add_argument("--score", action="store_true", help="사후 확인 3: 점수 1등 vs 같은 날 나머지 (판정 없음)")
    g.add_argument("--sleeves", action="store_true", help="사후 확인 4: 매일 1등 묶음 vs SPY (판정 없음)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.register:
        register()
    elif args.finish:
        finish_saved()
    elif args.followup:
        followup(args.workers)
    elif args.shift:
        shift(args.workers)
    elif args.score:
        score_check()
    elif args.sleeves:
        sleeves()
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
