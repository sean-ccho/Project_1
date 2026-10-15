#!/usr/bin/env python3
"""ISB 라벨 조합 탐색 (isb_label_search_v1): 어느 매수 라벨에서 사고, 매도 라벨(EXIT·TRIM·TOP)에 어떻게 반응해야 하나.

계기: 2026-10-13 사용자 — "optuna 이용해서 여러가지 테스트 ... 한번은 BuyBO 때 사고 한번은 Bottom 때 사고
      팔 때도 trim, exit, top 바꿔가면서".
조합 (isb_engine.all_rules, 686개). 조합이 적어 Optuna TPE 로 일부만 고르지 않고 전부 돌린다
(결과는 Optuna 저장소로도 내보내 optuna-dashboard 로 볼 수 있다):
  주기  일봉 · 주봉 (ISB 기본값, TradingView 가격)
  매수  라벨 부분집합 — 일봉 PB·BO·EARLY·REV·S2·◇BOTTOM (63) · 주봉 PB·BO·EARLY·REV·◇BOTTOM (31)
  매도  EXIT 전량/무시 · 주봉 TRIM⅓(T1 +2R) 무시/⅓/전량 · TRIM·TOP 무시/⅓/전량 (전량 매도 수단이 없는 조합 제외)
  라벨 위치는 ISB 기본값 그대로이고 어느 라벨에 반응하는지만 바꾼다. BOTTOM 매수는 REV 로 샀다고 보고 ISB 보유 관리로
  그 뒤 라벨을 계산한다 (isb_w_bo_bottom_v1 과 같음).
적용 (조합마다 둘 다):
  pt1   메일 top5(PT-1 선택 먼저) 중 규칙상 다음 시가 보유인 첫 종목 매수 + 규칙 매도만 (PT-1 매도·교체 끔) — 실제 사용 방식
  solo  PIT S&P 500 구성 종목 전체: 규칙상 보유 종목에 (1/N)×남은 비율, 나머지 현금 (주봉 T1 은 장중 체결가 반영)
선택 = 워크포워드 (wfo_structure 와 같은 틀):
  시험 연도 Y = 2020 ~ 2025(9/30까지). 2016-01-04 ~ (Y−1)-12-31 점수 = Sharpe − max(0, |MDD| − 30%) × 2
  (그 구간 거래 50건 미만 제외)가 가장 높은 조합을 Y 에 그대로 쓴다. 시험 연도만 이어 붙인 수익이 정직한 표본 외 성적이다.
  조합마다 2016-01-04 부터 한 번 돌린 결과를 연도별로 잘라 쓴다 (엔진은 과거만 보므로 짧게 돌린 것과 같다).
사전 등록 (TRIAL_LOG family isb_label_search_v1, 시도 1,372 = 686 × 2):
  WFO-pt1   표본 외(2020-01 ~ 2025-09) vs PT-1 BASE
  WFO-solo  표본 외 vs 동일가중 매수보유
  통과 = verdict '채택 후보' (ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내) · p ≤ 0.05/2 (부트스트랩 10,000회)
  참고 (판정 없음) = Deflated Sharpe, 전체 구간 상위 조합(표본 내라 낙관적), 라벨·반응별 평균, SPY 대비, 지난 규칙의 순위
확인 (하나라도 다르면 결과를 쓰지 않고 멈춘다):
  BASE = 저장된 기준선 · 래퍼 통과 모드 = BASE (ranked 후보표 재사용 전·후 둘 다)
  라벨 규칙 계산이 ISB 기본 규칙(엔진)·주봉 BO·BOTTOM 규칙(지난 연구)의 봉별 표를 전 종목 그대로 재현
  같은 규칙의 pt1 실행 = 지난 연구의 D-full · W-full · WBB-pt1 자본곡선
속도: 날짜별 ranked 후보표(엔진 시간의 약 80%)는 규칙과 무관해 워커 안에서 실행 간 재사용한다 (1회 약 3분 → 40초).

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/isb_label_search.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/isb_label_search.py --run [--workers 4]   (멈춰도 다시 실행하면 이어서 돈다)
  PYTHONPATH=.:src .venv/bin/python scripts/isb_label_search.py --finish             (보고서만 다시)
  .venv/bin/optuna-dashboard sqlite:///data/research/isb_label_search/optuna.db      (조합별 결과 둘러보기)
산출: tier3/ISB_LABEL_SEARCH.md · tier3/isb_label_search.csv · data/research/isb_label_search/ (git 제외)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isb_engine as isb  # noqa: E402
import isb_vs_pt1 as iv  # noqa: E402
from paper_trading.evaluation import cagr, deflated_sharpe, max_drawdown, sharpe, verdict  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from research_utils import (COST_PER_SIDE, DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, load_ohlcv,  # noqa: E402
                            log_trials, total_trials)
from validate_upside_walkforward import RUN_KW  # noqa: E402

FAMILY = "isb_label_search_v1"
STEP = "stage14"
TESTS = ("WFO-pt1", "WFO-solo")
N_TESTS = len(TESTS)
P_BONF = 0.05 / N_TESTS
N_BOOT = iv.N_BOOT
N_BOOT_REF = iv.N_BOOT_REF
FOLDS = (2020, 2021, 2022, 2023, 2024, 2025)
MIN_TRADES = 50
MDD_FREE, MDD_PENALTY = 0.30, 2.0
START = iv.START
RESULT_VERSION = 1
OUT_DIR = RESEARCH_DIR / "isb_label_search"
RUNS_DIR = OUT_DIR / "runs"
BASE_PATH = OUT_DIR / "base.pkl"
OPTUNA_DB = OUT_DIR / "optuna.db"
OUT_CSV = TIER3_DIR / "isb_label_search.csv"
REPORT = TIER3_DIR / "ISB_LABEL_SEARCH.md"
APPS = ("pt1", "solo")
APP_NAMES = {"pt1": "PT-1 종목에 적용", "solo": "전 종목 단독"}
BENCH_NAMES = {"pt1": "PT-1 BASE", "solo": "동일가중"}
PREV = {  # 지난 연구에서 같은 규칙으로 돌린 pt1 실행 — 앵커 확인
    isb.native_rule("D").key: (iv.RUNS_PATH, "D-full"),
    isb.native_rule("W").key: (iv.RUNS_PATH, "W-full"),
    isb.WBB_RULE.key: (RESEARCH_DIR / "isb_bo_bottom_runs.parquet", "WBB-pt1"),
}
PREV_NAMES = {isb.native_rule("D").key: "ISB 일봉 기본 그대로 (isb_vs_pt1_v1 D-full)",
              isb.native_rule("W").key: "ISB 주봉 기본 그대로 (isb_vs_pt1_v1 W-full)",
              isb.WBB_RULE.key: "주봉 BO·BOTTOM 매수 · EXIT 만 (isb_w_bo_bottom_v1)"}
OPTUNA_LABELS = ("PB", "BO", "EARLY", "REV", "S2", "BOTTOM")
NAN = float("nan")


def spec() -> str:
    """사전 등록 문구 (TRIAL_LOG 등록 행)."""
    n = len(isb.all_rules())
    return (f"[{RUN_KW['start_date']}~{RUN_KW['end_date']}, isb_vs_pt1_v1 과 같은 엔진·확인 절차, ISB.pine sha256 "
            f"{isb._sha(isb.PINE_PATH)[:12]} 기본값 · TradingView 가격] 라벨 규칙 전수 {n}개 = 일봉(매수 PB·BO·EARLY·REV·S2·"
            "BOTTOM 부분집합 63 × 매도 4) · 주봉(PB·BO·EARLY·REV·BOTTOM 31 × 14); 매도 = EXIT 전량/무시 · 주봉 TRIM(T1) "
            "무시/⅓/전량 · TOP 무시/⅓/전량, 전량 매도 수단이 없는 조합 제외. 라벨 위치는 ISB 기본값, BOTTOM 은 REV 로 샀다고 보고 "
            "보유 관리. 적용 pt1 = 메일 top5(PT-1 선택 먼저) 중 규칙상 다음 시가 보유 첫 종목 + 규칙 매도만(PT-1 매도·교체 끔); "
            "solo = PIT 구성 종목 규칙 보유분 (1/N)×남은 비율·나머지 현금(T1 장중 체결가 반영). "
            f"선택 = 워크포워드: 시험 연도 {FOLDS[0]}~{FOLDS[-1]}(9/30까지)마다 2016-01-04~전년 말 점수 "
            f"(Sharpe − max(0,|MDD|−{MDD_FREE:.0%})×{MDD_PENALTY:g}, 거래 {MIN_TRADES}건 미만 제외) 최고 조합을 그해에 적용, "
            "시험 연도만 이어 붙인 수익을 기준(pt1: PT-1 BASE, solo: 동일가중 매수보유)과 비교. "
            f"통과 = verdict 채택 후보(ΔSharpe 95% 하한>0 · 3구간 중 2+ · MDD 2%p 이내) · p ≤ 0.05/{N_TESTS} "
            f"(부트스트랩 {N_BOOT:,}회). 참고 = DSR, 전체 구간 상위 조합(표본 내), 라벨·반응별 평균, SPY 대비, 지난 규칙 순위.")


# ============================================================
# 순수 함수 (단위 테스트 대상)
# ============================================================
def score(r: pd.Series, n_trades: int) -> float:
    """워크포워드 점수 (wfo_structure 옵션 A). 거래 MIN_TRADES 미만이거나 계산이 안 되면 nan (후보에서 제외)."""
    if n_trades < MIN_TRADES or len(r) < 40:
        return NAN
    v = r.tolist()
    sh, mdd = sharpe(v), max_drawdown(v)
    if sh is None or mdd is None or not (math.isfinite(sh) and math.isfinite(mdd)):
        return NAN
    return float(sh) - max(0.0, abs(float(mdd)) - MDD_FREE) * MDD_PENALTY


def train_end(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year - 1}-12-31")


def test_span(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), min(pd.Timestamp(f"{year}-12-31"), DEV_END)


def fold_scores(returns: dict[str, pd.Series],
                trade_dates: dict[str, pd.DatetimeIndex]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """조합 × 시험 연도: 그 연도 직전까지(2016-01-04 ~ 전년 말)의 (점수, Sharpe)."""
    sc: dict[str, dict[int, float]] = {}
    sh: dict[str, dict[int, float]] = {}
    for key, r in returns.items():
        d = trade_dates[key]
        sc[key], sh[key] = {}, {}
        for y in FOLDS:
            part = r.loc[START:train_end(y)]
            sc[key][y] = score(part, int(((d >= START) & (d <= train_end(y))).sum()))
            sh[key][y] = sharpe(part.tolist()) if len(part) > 1 else NAN
    cols = list(FOLDS)
    return pd.DataFrame.from_dict(sc, orient="index")[cols], pd.DataFrame.from_dict(sh, orient="index")[cols]


def wfo_picks(scores: pd.DataFrame) -> dict[int, str | None]:
    """시험 연도마다 점수가 가장 높은 조합 (동점은 키 순서가 앞선 것). 후보가 없으면 None."""
    out: dict[int, str | None] = {}
    for y in FOLDS:
        col = scores[y].dropna()
        out[y] = sorted(col.index[col == col.max()])[0] if len(col) else None
    return out


def stitch(returns: dict[str, pd.Series], picks: dict[int, str | None], ref_index: pd.DatetimeIndex) -> pd.Series:
    """시험 연도마다 고른 조합의 그해 수익을 이어 붙인다 (고를 조합이 없던 해는 기준 날짜에 현금 0)."""
    parts = []
    for y in FOLDS:
        lo, hi = test_span(y)
        key = picks[y]
        if key is None:
            days = ref_index[(ref_index >= lo) & (ref_index <= hi)]
            parts.append(pd.Series(0.0, index=days))
        else:
            parts.append(returns[key].loc[lo:hi])
    return pd.concat(parts)


def weighted_np(w: np.ndarray, r: np.ndarray, cost: float = COST_PER_SIDE) -> np.ndarray:
    """iv.weighted_returns 의 numpy 판: t행 비중 × (t 시가 → t+1 시가) − 비중 변화 × 편도 비용."""
    gross = (w * np.nan_to_num(r)).sum(axis=1)
    turnover = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, w.shape[1])))).sum(axis=1)
    return gross - turnover * cost


def judge(stats: dict[str, float], v: str) -> str:
    """사전 등록 판정: 채택 후보 / 관찰 / 탈락 (p ≤ 0.05/2)."""
    if v == "채택 후보" and stats and stats["p_개선아님"] <= P_BONF:
        return "채택 후보"
    if stats and stats["ΔSharpe_하한"] > 0:
        return "관찰"
    return "탈락"


def rule_cols(key: str) -> dict[str, Any]:
    """CSV·효과 표용 규칙 열."""
    r = isb.LabelRule.parse(key)
    return {"key": key, "설명": r.describe(), "tf": r.tf, "entries": "+".join(r.entries), "exit": r.exit,
            "t1": r.t1, "top": r.top}


def label_effects(table: pd.DataFrame, col: str) -> list[tuple[str, str, str]]:
    """주기별 · 매수 라벨 포함/제외 · 매도 반응별 col 평균. Returns [(주기, 항목, '선택 평균 (조합 수) · ...')]."""
    out = []
    for tf in ("D", "W"):
        t = table[(table["tf"] == tf) & table[col].notna()]
        if t.empty:
            continue
        name = "일봉" if tf == "D" else "주봉"
        for lab in isb.ENTRY_LABELS[tf]:
            has = t["entries"].str.split("+").apply(lambda xs, lab=lab: lab in xs)
            out.append((name, f"매수 {lab}", f"넣음 {t.loc[has, col].mean():.2f} ({int(has.sum())}) · "
                                            f"뺌 {t.loc[~has, col].mean():.2f} ({int((~has).sum())})"))
        out.append((name, "EXIT", " · ".join(f"{'전량' if v else '무시'} {g[col].mean():.2f} ({len(g)})"
                                             for v, g in t.groupby("exit", sort=False))))
        for k, lab in (("t1", "TRIM(T1)"), ("top", "TOP")):
            if tf == "D" and k == "t1":
                continue
            out.append((name, lab, " · ".join(f"{isb._ACT_KO[a]} {t.loc[t[k] == a, col].mean():.2f} ({int((t[k] == a).sum())})"
                                              for a in isb.ACTIONS if (t[k] == a).any())))
    return out


# ============================================================
# 규칙 하나 → 엔진 조회 표 · 단독 수익
# ============================================================
class MatBook:
    """규칙 하나의 일봉 이벤트를 (종목, 날짜)로 찾는 표 — iv.IsbBook 과 같은 get() (없으면 기본값, 빈 칸은 False·nan)."""

    FIELDS = {"hold_next": bool, "exit_full": bool, "trim_ev": bool, "t1_fill": float}

    def __init__(self, dates: pd.DatetimeIndex, tickers: Iterable[str]) -> None:
        self.dates = pd.DatetimeIndex(dates)
        tickers = list(tickers)
        self.row = {f"{d:%Y-%m-%d}": i for i, d in enumerate(self.dates)}
        self.col = {t: j for j, t in enumerate(tickers)}
        shape = (len(self.dates), len(tickers))
        self.mat = {k: np.zeros(shape, dtype=bool) if kind is bool else np.full(shape, np.nan)
                    for k, kind in self.FIELDS.items()}

    def put(self, ticker: str, ev: pd.DataFrame) -> None:
        j = self.col[ticker]
        rows = self.dates.get_indexer(ev.index)
        ok = rows >= 0
        for k in self.FIELDS:
            self.mat[k][rows[ok], j] = ev[k].to_numpy()[ok]

    def get(self, name: str, ticker: str, day: str, default: Any) -> Any:
        i, j = self.row.get(day), self.col.get(ticker)
        if i is None or j is None:
            return default
        return self.mat[name][i, j]


class SoloCtx:
    """단독(달력시간) 공통 표 — iv.solo_returns 와 같은 정의: 시가→시가 수익, 그날 PIT 구성 종목 ∩ 실거래 봉, 1/N."""

    def __init__(self, raw: pd.DataFrame, mem: Membership, tickers: Iterable[str]) -> None:
        from idea_batch_research import member_mask
        opens = raw.xs("Open", axis=1, level="Price").sort_index()
        r = opens.shift(-1) / opens - 1.0
        idx = r.loc[START:DEV_END].index[:-1]
        cols = pd.Index(sorted(set(tickers) & set(r.columns)))
        vol = raw.xs("Volume", axis=1, level="Price").reindex(index=idx, columns=cols)
        hi = raw.xs("High", axis=1, level="Price").reindex(index=idx, columns=cols)
        lo = raw.xs("Low", axis=1, level="Price").reindex(index=idx, columns=cols)
        r = r.reindex(index=idx, columns=cols)
        uni = member_mask(idx, cols, mem) & r.notna() & (vol > 0) & (hi > lo)
        n = uni.sum(axis=1).replace(0, np.nan)
        self.idx, self.cols = idx, cols
        self.col = {t: j for j, t in enumerate(cols)}
        self.r = r.to_numpy(dtype=float)
        self.uni = uni.to_numpy(dtype=bool)
        self.inv_n = (1.0 / n).to_numpy(dtype=float)
        self.open = opens.reindex(index=idx, columns=cols).to_numpy(dtype=float)
        self.next_open = opens.shift(-1).reindex(index=idx, columns=cols).to_numpy(dtype=float)
        self.ew = pd.Series(weighted_np(np.nan_to_num(self.uni * self.inv_n[:, None]), self.r), index=idx)


class SoloAcc:
    """규칙 하나의 단독 비중을 종목별로 쌓는다. 주봉 T1 장중 체결분은 그날 '시가 → 체결가' 수익으로 보정한다."""

    def __init__(self, ctx: SoloCtx) -> None:
        self.ctx = ctx
        self.held = np.zeros(ctx.r.shape)
        self.corr = np.zeros(len(ctx.idx))

    def add(self, ticker: str, ev: pd.DataFrame) -> None:
        c = self.ctx
        j = c.col.get(ticker)
        if j is None:
            return
        rows = c.idx.get_indexer(ev.index)
        ok = rows >= 0
        self.held[rows[ok], j] = (ev["pos"].to_numpy(dtype=float) * ev["rem"].to_numpy(dtype=float))[ok]
        fill = ev["t1_fill"].to_numpy(dtype=float)
        m = ok & ~np.isnan(fill)
        if m.any():
            rr = rows[m]
            sold = ev["t1_sold"].to_numpy(dtype=float)[m]
            with np.errstate(invalid="ignore", divide="ignore"):
                adj = sold * c.inv_n[rr] * (fill[m] - c.next_open[rr, j]) / c.open[rr, j]
            np.add.at(self.corr, rr, np.where(c.uni[rr, j] & np.isfinite(adj), adj, 0.0))

    def result(self) -> tuple[pd.Series, float]:
        c = self.ctx
        w = np.nan_to_num(np.where(c.uni, self.held, 0.0) * c.inv_n[:, None])
        return pd.Series(weighted_np(w, c.r) + self.corr, index=c.idx), float(w.sum(axis=1).mean())


# ============================================================
# 결과 저장 (규칙마다 파일 하나 → 멈춰도 이어서)
# ============================================================
def fingerprint() -> str:
    """결과에 영향을 주는 것: Pine · 엔진 코드 · 가격·VIX/DXY 캐시(isb.cache_key) · 오버레이 코드 · 결과 형식 버전."""
    parts = [json.dumps(isb.cache_key("W"), sort_keys=True), isb._sha(Path(iv.__file__)), str(RESULT_VERSION)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _path(key: str) -> Path:
    return RUNS_DIR / f"{re.sub(r'[^A-Za-z0-9]+', '_', key)}.pkl"


def save_result(out: dict[str, Any]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(out["key"]).with_suffix(".tmp")
    pd.to_pickle(out, tmp)
    tmp.replace(_path(out["key"]))


def load_result(key: str, fp: str) -> dict[str, Any] | None:
    path = _path(key)
    if not path.exists():
        return None
    out = pd.read_pickle(path)
    return out if out.get("fp") == fp and out.get("key") == key else None


# ============================================================
# 워커
# ============================================================
_G: dict[str, Any] = {}


def _init_search() -> None:
    """워커 준비: 라벨 캐시(일봉·주봉) · 단독 공통 표 · 날짜 목록 · 매수 라벨의 그날 S&P 500 여부, ranked 후보표 재사용."""
    mem = Membership(load_membership(iv.MEMBERSHIP))
    caches = {tf: pd.read_pickle(RESEARCH_DIR / f"isb_labels_{tf}.pkl") for tf in ("D", "W")}
    for cache in caches.values():
        for t, item in cache.items():
            idx = item["index"]
            for sg in item["signals"]:
                sg["member"] = t in mem.on(f"{idx[sg['s']]:%Y-%m-%d}")
    tickers = sorted(caches["D"])
    days = pd.DatetimeIndex(sorted(set().union(*(item["daily"].index for item in caches["D"].values()))))
    _G.update(cache=caches, solo=SoloCtx(load_ohlcv(isb.PRICE_CACHE), mem, tickers),
              dates=days[days >= START - pd.Timedelta(days=10)], fp=fingerprint())
    iv.enable_snapshot_cache()


def _task(key: str) -> dict[str, Any]:
    """규칙 하나: 라벨 → 일봉 이벤트 → 단독 수익 + PT-1 엔진 실행. 결과 파일을 쓰고 진행 표시용 요약을 돌려준다."""
    rule = isb.LabelRule.parse(key)
    p = isb.IsbParams(rule.tf)
    cache = _G["cache"][rule.tf]
    book = MatBook(_G["dates"], cache)
    acc = SoloAcc(_G["solo"])
    sig_dates: list[pd.Timestamp] = []
    for t, item in cache.items():
        ev, trades = isb.rule_events(item, rule, p)
        book.put(t, ev)
        acc.add(t, ev)
        idx, n = item["index"], len(item["index"])
        sig_dates += [idx[tr["s"]] for tr in trades if tr["member"] and tr["e"] < n]
    solo_r, expo = acc.result()
    ov = iv.Overlay(key, tf=key, gate="isb", sells="isb", t1_mode=rule.t1 if rule.tf == "W" else "off")
    res = iv.run_overlay(ov, book=book)
    blended = iv.blended_trades(res)
    out = {
        "fp": _G["fp"], "key": key,
        "solo_r": solo_r, "solo_expo": expo, "solo_sig": pd.DatetimeIndex(sorted(sig_dates)),
        "equity": res.equity, "entries": pd.DatetimeIndex(sorted(pd.to_datetime([t["entry_date"] for t in res.trades]))),
        "n_trades": len(res.trades), "n_trims": len(res.trims),
        "win": float(np.mean([x > 0 for x in blended])) if blended else NAN,
        "gate_rate": res.gate_pass / res.gate_eval if res.gate_eval else NAN,
        "empty_rate": res.gate_empty / res.gate_days if res.gate_days else NAN,
        "max_weight": res.max_weight, "exit_mix": iv.exit_mix(res.trades), "rss": res.peak_rss_gb,
    }
    save_result(out)
    return {"key": key, "n_trades": out["n_trades"], "rss": out["rss"]}


def _probe_twice() -> tuple[iv.RunResult, iv.RunResult]:
    """래퍼 통과 모드를 같은 프로세스에서 두 번 (ranked 후보표 재사용 전 → 후)."""
    iv.enable_snapshot_cache()
    ov = iv.overlay_specs()["PROBE"]
    return iv.run_overlay(ov), iv.run_overlay(ov)


def _hm(sec: float) -> str:
    m = int(sec // 60)
    return f"{m // 60}:{m % 60:02d}"


def _drain(ex: ProcessPoolExecutor, keys: list[str], label: str) -> None:
    """규칙들을 풀에 넣고 끝나는 대로 진행 표시 (성과 숫자는 찍지 않는다). 하나라도 실패하면 남은 것을 취소하고 멈춘다."""
    if not keys:
        return
    print(f"===== {label}: {len(keys)}개", flush=True)
    t0 = time.time()
    futs = [ex.submit(_task, k) for k in keys]
    try:
        for k, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            el = time.time() - t0
            print(f"  [{label}] {k}/{len(keys)} {s['key']} · PT-1 거래 {s['n_trades']} · 메모리 {s['rss']:.1f}GB · "
                  f"경과 {_hm(el)} · 남은 약 {_hm(el / k * (len(keys) - k))} [{k / len(keys) * 100:5.1f}%]", flush=True)
    except BaseException:
        for f in futs:
            f.cancel()
        raise


# ============================================================
# 확인
# ============================================================
def check_engine() -> iv.RunResult:
    """BASE = 저장된 기준선, 래퍼 통과 모드 = BASE (ranked 후보표 재사용 전·후)."""
    print("===== 확인: BASE 재현 · 래퍼 투명성 (후보표 재사용 전·후)", flush=True)
    specs = iv.overlay_specs()
    with ProcessPoolExecutor(max_workers=2) as ex:
        f_base, f_probe = ex.submit(iv.run_overlay, specs["BASE"]), ex.submit(_probe_twice)
        base, (cold, warm) = f_base.result(), f_probe.result()
    errors = [e for e in (iv.check_base(base), iv.check_probe(base, cold), iv.check_probe(base, warm)) if e]
    if errors:
        raise SystemExit("확인 실패 — 결과를 쓰지 않는다: " + " / ".join(errors))
    return base


def check_anchor(key: str, fp: str) -> str | None:
    """같은 규칙의 지난 연구 pt1 자본곡선과 같은지 (지난 실행 파일이 없으면 건너뜀)."""
    path, col = PREV[key]
    if not path.exists():
        print(f"[label] 앵커 {col}: 지난 실행 파일 없음 — 건너뜀", flush=True)
        return None
    prev = pd.read_parquet(path)[col].dropna()
    prev.index = pd.to_datetime(prev.index)
    res = load_result(key, fp)
    if res is None:
        return f"{col}: 결과 없음"
    eq = res["equity"]
    common = prev.index.intersection(eq.index)
    max_rel = float((eq.loc[common] / prev.loc[common] - 1).abs().max()) if len(common) else float("inf")
    if len(common) != len(prev) or max_rel > 1e-9:
        return f"{col} 재현 실패 (겹친 날 {len(common)}/{len(prev)}, 최대 차이 {max_rel:.1e})"
    print(f"[label] 앵커 {key} = 지난 연구 {col} 자본곡선 ({len(common)}일, 최대 차이 {max_rel:.1e})", flush=True)
    return None


# ============================================================
# 등록 · 실행
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
    n = len(isb.all_rules()) * len(APPS)
    log_trials(STEP, FAMILY, n, [{"test_id": "(등록)", "판정": "등록", "메모": spec()}])
    print(f"등록: {FAMILY} 시도 {n}건 · 검정 {N_TESTS}건 · p ≤ {P_BONF:.3f} · 누적 시도 {total_trials()}건")


def run(workers: int) -> None:
    if not _registered():
        raise SystemExit("먼저 --register 로 등록한다 (결과 계산 전 사전 등록)")
    iv.ensure_pinned()
    isb.build_events("W")                      # 래퍼 통과 모드(PROBE)가 쓰는 표
    for tf in ("D", "W"):
        isb.build_label_cache(tf)              # 재현 확인이 틀리면 여기서 멈춘다
    base = check_engine()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"equity": base.equity, "n_trades": len(base.trades), "exit_mix": iv.exit_mix(base.trades),
                  "win": float(np.mean([t["return_pct"] > 0 for t in base.trades]))}, BASE_PATH)
    fp = fingerprint()
    keys = [r.key for r in isb.all_rules()]
    todo = [k for k in keys if load_result(k, fp) is None]
    print(f"[label] 규칙 {len(keys)}개 — 이미 끝남 {len(keys) - len(todo)} · 남음 {len(todo)} · 동시 {workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_search) as ex:
        _drain(ex, [k for k in PREV if k in todo], "앵커 규칙 (지난 연구와 대조)")
        errors = [e for e in (check_anchor(k, fp) for k in PREV) if e]
        if errors:
            raise SystemExit("앵커 확인 실패 — 결과를 쓰지 않는다: " + " / ".join(errors))
        _drain(ex, [k for k in todo if k not in PREV], "라벨 규칙")
    finish(fp)


# ============================================================
# 판정 · 보고서
# ============================================================
def _stats(r: pd.Series) -> dict[str, float]:
    v = r.dropna().tolist()
    return {"Sharpe": sharpe(v), "CAGR": cagr(v), "MDD": max_drawdown(v)}


def _full_table(res: dict[str, dict[str, Any]], rets: dict[str, dict[str, pd.Series]],
                trades: dict[str, dict[str, pd.DatetimeIndex]], scores: dict[str, pd.DataFrame],
                picks: dict[str, dict[int, str | None]]) -> pd.DataFrame:
    """규칙 × (pt1, solo) 전체 구간 숫자 · 폴드 점수 · 워크포워드 선택 여부."""
    rows = []
    for key in res:
        row = rule_cols(key)
        for app in APPS:
            st = _stats(rets[app][key])
            row.update({f"{app}_Sharpe": st["Sharpe"], f"{app}_CAGR": st["CAGR"], f"{app}_MDD": st["MDD"],
                        f"{app}_거래": len(trades[app][key]),
                        f"{app}_점수": score(rets[app][key], len(trades[app][key]))})
            for y in FOLDS:
                row[f"{app}_점수_{y}"] = scores[app].at[key, y]
            row[f"{app}_선택연도"] = " ".join(str(y) for y, k in picks[app].items() if k == key)
        r = res[key]
        row.update({"pt1_트림": r["n_trims"], "pt1_승률": r["win"], "pt1_게이트통과율": r["gate_rate"],
                    "pt1_매수없는날": r["empty_rate"], "pt1_최대비중": r["max_weight"], "pt1_청산이유": r["exit_mix"],
                    "solo_평균비중": r["solo_expo"]})
        rows.append(row)
    return pd.DataFrame(rows)


def finish(fp: str | None = None, log_results: bool | None = None) -> pd.DataFrame:
    """규칙 결과 → 워크포워드 선택 · 판정 · CSV · TRIAL_LOG · Optuna 저장소 · 보고서."""
    print("===== 워크포워드 선택 · 판정", flush=True)
    fp = fp or fingerprint()
    keys = [r.key for r in isb.all_rules()]
    res = {k: load_result(k, fp) for k in keys}
    missing = [k for k, v in res.items() if v is None]
    if missing:
        raise SystemExit(f"결과가 없는 규칙 {len(missing)}개 (예: {missing[:3]}) — --run 으로 마저 돌린다")
    if not BASE_PATH.exists():
        raise SystemExit("BASE 결과 없음 — --run 을 먼저")
    base = pd.read_pickle(BASE_PATH)
    if log_results is None:
        log = _log()
        log_results = not ((log["family"] == FAMILY) & (log["판정"] != "등록")).any()
    mem = Membership(load_membership(iv.MEMBERSHIP))
    raw = load_ohlcv(isb.PRICE_CACHE)
    label_tickers = sorted(pd.read_pickle(RESEARCH_DIR / "isb_labels_D.pkl"))
    ew_r = SoloCtx(raw, mem, label_tickers).ew
    base_r = base["equity"].pct_change().dropna()
    spy_open = raw["SPY"]["Open"]
    spy = {"pt1": load_ohlcv(iv.ETF_OHLCV)["SPY"]["Close"].pct_change(), "solo": spy_open.shift(-1) / spy_open - 1.0}
    bench = {"pt1": base_r, "solo": ew_r}
    rets = {"pt1": {k: v["equity"].pct_change().dropna() for k, v in res.items()},
            "solo": {k: v["solo_r"] for k, v in res.items()}}
    trades = {"pt1": {k: v["entries"] for k, v in res.items()}, "solo": {k: v["solo_sig"] for k, v in res.items()}}

    tests: dict[str, dict[str, Any]] = {}
    scores: dict[str, pd.DataFrame] = {}
    picks: dict[str, dict[int, str | None]] = {}
    for app in APPS:
        sc, sh = fold_scores(rets[app], trades[app])
        scores[app], picks[app] = sc, wfo_picks(sc)
        oos = stitch(rets[app], picks[app], bench[app].index)
        st, df = iv._cmp(oos, bench[app], N_BOOT)
        a, b = df["a"].tolist(), df["b"].tolist()
        v = verdict(st, max_drawdown(a), max_drawdown(b))
        st_spy, df_spy = iv._cmp(oos, spy[app], N_BOOT_REF)
        trial_sr = [x / math.sqrt(252) for x in sh.where(sc.notna()).to_numpy().ravel() if x == x]
        oos_s = pd.Series(a, index=df.index)
        dsr = deflated_sharpe(sharpe(a) / math.sqrt(252), trial_sr, len(a), skew=float(oos_s.skew()),
                              kurt=float(oos_s.kurt() + 3))
        years = []
        for y in FOLDS:
            lo, hi = test_span(y)
            key = picks[app][y]
            years.append({"연도": y, "key": key, "연습점수": sc.at[key, y] if key else NAN,
                          "선택": float((1 + oos.loc[lo:hi]).prod() - 1),
                          "기준": float((1 + df["b"].loc[lo:hi]).prod() - 1),
                          "SPY": float((1 + spy[app].reindex(df.index).fillna(0.0).loc[lo:hi]).prod() - 1)})
        tests[f"WFO-{app}"] = {
            "app": app, "stats": st, "verdict": v, "판정": judge(st, v), "dsr": dsr, "n_trials": len(trial_sr),
            "oos": _stats(df["a"]), "bench": _stats(df["b"]), "spy": _stats(df_spy["b"]), "st_spy": st_spy,
            "start": df.index.min().date(), "end": df.index.max().date(), "days": len(df), "years": years}
        print(f"[label] WFO-{app}: 표본 외 Sharpe {tests[f'WFO-{app}']['oos']['Sharpe']:.2f} vs {BENCH_NAMES[app]} "
              f"{tests[f'WFO-{app}']['bench']['Sharpe']:.2f} → {tests[f'WFO-{app}']['판정']}", flush=True)

    table = _full_table(res, rets, trades, scores, picks)
    table.to_csv(OUT_CSV, index=False)
    if log_results:
        log_trials(STEP, FAMILY, len(keys) * len(APPS), [{
            "test_id": tid, "판정": t["판정"],
            "메모": (f"표본 외 {t['start']}~{t['end']}: Sharpe {t['oos']['Sharpe']:.2f} vs {BENCH_NAMES[t['app']]} "
                     f"{t['bench']['Sharpe']:.2f}, ΔSharpe {t['stats'].get('ΔSharpe', NAN):+.2f} "
                     f"[{t['stats'].get('ΔSharpe_하한', NAN):+.2f}, {t['stats'].get('ΔSharpe_상한', NAN):+.2f}] "
                     f"p {t['stats'].get('p_개선아님', NAN):.4f}, 구간 {t['stats'].get('구간_개선')}/{t['stats'].get('구간_수')}, "
                     f"CAGR {t['oos']['CAGR']:+.1%}, MDD {t['oos']['MDD']:.1%}, DSR {t['dsr']:.2f}")}
            for tid, t in tests.items()])
    else:
        print("TRIAL_LOG 에 결과를 쓰지 않는다 (이미 있음)", flush=True)
    importances = export_optuna(table)
    write_report(tests, table, base, base_r, ew_r, importances, rets)
    return table


def export_optuna(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """조합별 전체 구간 결과를 Optuna 저장소로 (optuna-dashboard 로 보기용). Returns 적용별 파라미터 중요도."""
    import optuna
    from optuna.distributions import CategoricalDistribution as Cat
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    OPTUNA_DB.unlink(missing_ok=True)
    storage = f"sqlite:///{OPTUNA_DB}"
    dist = {"tf": Cat(["D", "W"]), **{f"buy_{x}": Cat([True, False]) for x in OPTUNA_LABELS},
            "exit": Cat([True, False]), "t1": Cat(list(isb.ACTIONS)), "top": Cat(list(isb.ACTIONS))}
    out: dict[str, dict[str, float]] = {}
    for app in APPS:
        study = optuna.create_study(study_name=f"isb_labels_{app}", storage=storage, direction="maximize")
        study.set_user_attr("설명", f"{APP_NAMES[app]} · 값 = 전체 구간(2016-01~2025-09) Sharpe — 표본 내라 낙관적. "
                                   "판정은 보고서의 워크포워드")
        trials = []
        for _, row in table.iterrows():
            rule = isb.LabelRule.parse(row["key"])
            params = {"tf": rule.tf, **{f"buy_{x}": x in rule.entries for x in OPTUNA_LABELS}, "exit": rule.exit,
                      "t1": rule.t1, "top": rule.top}
            val = row[f"{app}_Sharpe"]
            attrs = {"규칙": row["설명"], "CAGR": row[f"{app}_CAGR"], "MDD": row[f"{app}_MDD"],
                     "거래": int(row[f"{app}_거래"]), "워크포워드_선택연도": row[f"{app}_선택연도"]}
            ok = val is not None and math.isfinite(val)
            trials.append(optuna.trial.create_trial(
                params=params, distributions=dist, value=float(val) if ok else None,
                state=optuna.trial.TrialState.COMPLETE if ok else optuna.trial.TrialState.FAIL,
                user_attrs={k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in attrs.items()}))
        study.add_trials(trials)
        try:
            out[app] = optuna.importance.get_param_importances(study)
        except Exception as exc:  # noqa: BLE001 — 중요도는 참고용
            print(f"[label] Optuna 중요도 계산 실패 ({app}): {exc}", flush=True)
            out[app] = {}
    print(f"[label] Optuna 저장소: {OPTUNA_DB.relative_to(ROOT)} (optuna-dashboard {storage})", flush=True)
    return out


def _f(v: Any, fmt: str) -> str:
    return iv._f(v, fmt)


def _rank(table: pd.DataFrame, app: str, key: str) -> str:
    col = table[f"{app}_Sharpe"]
    v = float(table.loc[table["key"] == key, f"{app}_Sharpe"].iloc[0])
    return f"{v:.2f} ({int((col > v).sum()) + 1}위/{int(col.notna().sum())})"


_PARAM_KO = {"tf": "주기", "exit": "EXIT", "t1": "TRIM(T1)", "top": "TOP", **{f"buy_{x}": f"매수 {x}" for x in OPTUNA_LABELS}}


def write_report(tests: dict[str, dict[str, Any]], table: pd.DataFrame, base: dict[str, Any], base_r: pd.Series,
                 ew_r: pd.Series, importances: dict[str, dict[str, float]], rets: dict[str, dict[str, pd.Series]],
                 path: Path = REPORT) -> None:
    tp, ts = tests["WFO-pt1"], tests["WFO-solo"]
    base_st, ew_st = _stats(base_r), _stats(ew_r)
    passed = [tid for tid, t in tests.items() if t["판정"] == "채택 후보"]
    n_rules = len(table)
    if passed:
        head = f"**결론: 사전 등록 기준을 통과한 것이 있다 — {', '.join(passed)}.** 라이브 반영 전 표본 외 기간을 더 지켜봐야 한다."
    else:
        head = (f"**결론: 라벨 조합 {n_rules}개 중에서 해마다 그때까지 가장 좋았던 조합을 골라 써도 PT-1·동일가중보다 낫다고 "
                f"볼 수 없다.** 2020-01 ~ 2025-09 표본 외 Sharpe: PT-1 종목 적용 {tp['oos']['Sharpe']:.2f} vs PT-1 "
                f"{tp['bench']['Sharpe']:.2f}, 전 종목 단독 {ts['oos']['Sharpe']:.2f} vs 동일가중 {ts['bench']['Sharpe']:.2f}.")
    lines = [
        f"# ISB 라벨 조합 탐색 ({FAMILY}) — {RUN_KW['start_date']} ~ {RUN_KW['end_date']}", "", head, "",
        f"- 사전 등록: TRIAL_LOG family `{FAMILY}` · 시도 {n_rules * len(APPS):,}건({n_rules}개 × 적용 2) · 검정 {N_TESTS}건 · "
        f"p ≤ 0.05/{N_TESTS} · 부트스트랩 {N_BOOT:,}회 · 누적 시도 {total_trials():,}건",
        "- 계기: 사용자 — \"optuna 이용해서 … 한번은 BuyBO 때 사고 한번은 Bottom 때 사고 팔 때도 trim, exit, top 바꿔가면서\" (2026-10-13)",
        "- 조합: 주기(일봉·주봉) × 매수 라벨 부분집합(일봉 PB·BO·EARLY·REV·S2·◇BOTTOM 63, 주봉 PB·BO·EARLY·REV·◇BOTTOM 31) × "
        "매도 반응(EXIT 전량/무시 · 주봉 TRIM(T1) 무시/⅓/전량 · TOP 무시/⅓/전량, 전량 매도 수단이 없는 조합 제외) = "
        f"일봉 {int((table['tf'] == 'D').sum())} + 주봉 {int((table['tf'] == 'W').sum())}. 조합이 적어 Optuna 로 일부만 고르지 않고 전부 돌렸다",
        "- 라벨 위치는 ISB 기본값 그대로이고, 어느 라벨에 어떻게 반응하는지만 바꿨다. BOTTOM 매수는 그 봉에서 REV 로 샀다고 보고 "
        "ISB 보유 관리로 그 뒤 라벨(TRIM·TOP·EXIT)을 계산했다 (지난 BO·BOTTOM 연구와 같음)",
        "- 적용: PT-1 종목 = 메일 상위 5종목 중 규칙상 보유 중인 첫 종목을 사고 규칙대로만 판다 (PT-1 손절·익절·교체 끔, $5,000·3슬롯) · "
        "전 종목 단독 = 그날 S&P 500 구성 종목 중 규칙상 보유 종목에 1/N, 나머지 현금",
        f"- 선택(워크포워드): 시험 연도 2020 ~ 2025(9/30)마다, 2016-01-04 ~ 전년 말 점수(Sharpe − max(0, |MDD| − 30%) × 2, "
        f"거래 {MIN_TRADES}건 미만 제외)가 가장 높은 조합을 그해에 그대로 썼다. 시험 연도만 이어 붙인 수익이 표본 외 성적이다",
        "- 확인: BASE 재현 · 래퍼 투명성(후보표 재사용 전·후) · 라벨 규칙이 ISB 기본 규칙·지난 BO·BOTTOM 규칙의 봉별 표를 전 종목 재현 · "
        "같은 규칙의 PT-1 실행이 지난 연구(D-full·W-full·WBB-pt1) 자본곡선과 같음 — 모두 통과",
        f"- 채택 후보 = 기준 대비 ΔSharpe 95% 하한 > 0 · 3구간 중 2개 이상 개선 · MDD 2%p 이내 · p ≤ {P_BONF:.3f}",
        "", "## 판정 (표본 외 2020-01 ~ 2025-09)", "",
        "| 검정 | 표본 외 Sharpe | CAGR | MDD | 기준 | 기준 대비 ΔSharpe [95%] | p | 구간 | SPY 대비 ΔSharpe [95%] | DSR(참고) | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tid, t in tests.items():
        st, sp = t["stats"], t["st_spy"]
        lines.append(
            f"| {tid} {APP_NAMES[t['app']]} | {t['oos']['Sharpe']:.2f} | {t['oos']['CAGR']:+.1%} | {t['oos']['MDD']:.1%} | "
            f"{BENCH_NAMES[t['app']]} {t['bench']['Sharpe']:.2f} · {t['bench']['CAGR']:+.1%} · {t['bench']['MDD']:.1%} | "
            f"{_f(st.get('ΔSharpe'), '+.2f')} [{_f(st.get('ΔSharpe_하한'), '+.2f')}, {_f(st.get('ΔSharpe_상한'), '+.2f')}] | "
            f"{_f(st.get('p_개선아님'), '.4f')} | {st.get('구간_개선')}/{st.get('구간_수')} | {_f(sp.get('ΔSharpe'), '+.2f')} "
            f"[{_f(sp.get('ΔSharpe_하한'), '+.2f')}, {_f(sp.get('ΔSharpe_상한'), '+.2f')}] | {_f(t['dsr'], '.2f')} | **{t['판정']}** |")
    lines += ["", f"- 참고: 같은 기간 SPY Sharpe — PT-1 종목 비교 {tp['spy']['Sharpe']:.2f}, 단독 비교(시가 기준) {ts['spy']['Sharpe']:.2f}. "
              f"DSR 은 연습 구간 시도 {tp['n_trials']:,}·{ts['n_trials']:,}개의 Sharpe 분산으로 '운으로 기대되는 최고치'를 뺀 값(0.95 이상이면 운 아님)"]
    for tid, t in tests.items():
        app = t["app"]
        lines += ["", f"## 해마다 고른 조합 — {APP_NAMES[app]}", "",
                  "| 시험 연도 | 고른 조합 (그 전까지 점수 최고) | 연습 점수 | 그해 수익 | " + BENCH_NAMES[app] + " | SPY |",
                  "|---|---|---|---|---|---|"]
        for y in t["years"]:
            desc = isb.LabelRule.parse(y["key"]).describe() if y["key"] else "(후보 없음 → 현금)"
            lines.append(f"| {y['연도']} | {desc} | {_f(y['연습점수'], '.2f')} | {y['선택']:+.1%} | {y['기준']:+.1%} | {y['SPY']:+.1%} |")

    for app in APPS:
        bench_sh = base_st["Sharpe"] if app == "pt1" else ew_st["Sharpe"]
        ok = table[table[f"{app}_거래"] >= MIN_TRADES].sort_values(f"{app}_Sharpe", ascending=False)
        srs = [x / math.sqrt(252) for x in ok[f"{app}_Sharpe"] if x == x]
        top = ok.iloc[0] if len(ok) else None
        dsr_top = NAN
        if top is not None:
            r_top = rets[app][top["key"]].dropna()
            dsr_top = deflated_sharpe(float(top[f"{app}_Sharpe"]) / math.sqrt(252), srs, len(r_top),
                                      skew=float(r_top.skew()), kurt=float(r_top.kurt() + 3))
        lines += ["", f"## 참고: 전체 구간 상위 10개 — {APP_NAMES[app]} (표본 내 · 낙관적)", "",
                  f"- 거래 {MIN_TRADES}건 이상 {len(ok)}개 중. {BENCH_NAMES[app]} Sharpe {bench_sh:.2f}"
                  + (f" · 1위의 DSR {dsr_top:.2f} ({len(srs)}개 시도로 보정)" if top is not None else ""),
                  f"- {BENCH_NAMES[app]}보다 Sharpe 가 높은 조합 {int((ok[f'{app}_Sharpe'] > bench_sh).sum())}개 / {len(ok)}개",
                  "", "| 순위 | 조합 | Sharpe | CAGR | MDD | 거래 |" + (" 게이트 통과율 |" if app == "pt1" else " 평균 비중 |"),
                  "|---|---|---|---|---|---|---|"]
        for i, (_, row) in enumerate(ok.head(10).iterrows(), 1):
            extra = _f(row["pt1_게이트통과율"], ".0%") if app == "pt1" else _f(row["solo_평균비중"], ".0%")
            lines.append(f"| {i} | {row['설명']} | {row[f'{app}_Sharpe']:.2f} | {row[f'{app}_CAGR']:+.1%} | "
                         f"{row[f'{app}_MDD']:.1%} | {int(row[f'{app}_거래'])} | {extra} |")

    lines += ["", "## 참고: 라벨·반응별 평균 Sharpe (전체 구간, 조합 수)", "",
              "그 선택지를 넣은 조합들의 평균이다. 다른 선택과 섞여 있어 대략적인 방향만 보여 준다.", "",
              "| 주기 | 항목 | PT-1 종목에 적용 | 전 종목 단독 |", "|---|---|---|---|"]
    eff_p, eff_s = label_effects(table, "pt1_Sharpe"), label_effects(table, "solo_Sharpe")
    for (tf, item, vp), (_, _, vs) in zip(eff_p, eff_s):
        lines.append(f"| {tf} | {item} | {vp} | {vs} |")
    if any(importances.values()):
        lines += ["", "## 참고: Optuna 파라미터 중요도 (전체 구간 Sharpe 기준, fANOVA)", "",
                  "| 적용 | 중요도 (큰 순) |", "|---|---|"]
        for app in APPS:
            imp = importances.get(app) or {}
            lines.append(f"| {APP_NAMES[app]} | " + " · ".join(f"{_PARAM_KO.get(k, k)} {v:.2f}" for k, v in imp.items()) + " |")

    lines += ["", "## 참고: 지난 연구 규칙의 위치 (전체 구간 Sharpe, 순위)", "",
              "| 규칙 | PT-1 종목에 적용 | 전 종목 단독 |", "|---|---|---|"]
    for key, name in PREV_NAMES.items():
        lines.append(f"| {name} | {_rank(table, 'pt1', key)} | {_rank(table, 'solo', key)} |")
    lines += [f"| PT-1 그대로 (기준) | {base_st['Sharpe']:.2f} · 거래 {base['n_trades']} | 동일가중 {ew_st['Sharpe']:.2f} |", "",
              "## 둘러보기", "",
              f"- 조합 {n_rules}개 전체 숫자: `{OUT_CSV.relative_to(ROOT)}` (폴드별 점수·고른 연도 포함)",
              f"- Optuna 대시보드: `.venv/bin/optuna-dashboard sqlite:///{OPTUNA_DB.relative_to(ROOT)}` → 브라우저에서 "
              "isb_labels_pt1 / isb_labels_solo (값 = 전체 구간 Sharpe, 표본 내)",
              "", "## 한계", "",
              "- 개발 구간(2016-01 ~ 2025-09)은 이번 전까지 952건의 시도에 쓰였고, ISB 연구 두 건의 결과를 본 뒤 조합을 정했다. "
              "워크포워드 표본 외 성적도 이 구간 안의 값이라 진짜 표본 외(라이브)보다 낙관적일 수 있다. 홀드아웃은 쓰지 않았다.",
              "- BOTTOM 뒤 라벨은 차트에 없는 값을 'REV 로 샀다면'으로 계산한 것이다 (TradingView 에서 보려면 REV Market Washout Filter Off).",
              "- 일봉과 주봉을 섞는 규칙(주봉으로 사고 일봉으로 팔기), 등급(⚡⭐🏆) 필터, ISB 숫자 설정은 바꾸지 않았다.",
              "- 상장폐지 종목 가격이 없는 생존 편향, PT-1 엔진의 0주 매수 특성은 isb_vs_pt1_v1 과 같다.",
              "- 결과가 좋아도 라이브 반영은 사용자 결정이다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트: {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true", help="결과 계산 전 사전 등록 (TRIAL_LOG)")
    g.add_argument("--run", action="store_true", help="확인 → 라벨 규칙 전부(병렬, 이어서 가능) → 판정 · 보고서")
    g.add_argument("--finish", action="store_true", help="저장된 결과로 판정 · 보고서만 다시")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.register:
        register()
    elif args.run:
        run(args.workers)
    else:
        finish()


if __name__ == "__main__":
    main()
