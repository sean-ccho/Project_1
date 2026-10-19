#!/usr/bin/env python3
"""골든크로스·골든크로스 임박 10년 백테스트 — TradingView 와 같은 정의 + SMA vs EMA (ma_cross_v1).

계기: 2026-10-16 사용자 — 사이트 크로스 라벨이 TradingView 차트와 안 맞음 → "골든임박, 골든크로스 백테스팅 … 완벽해야돼
트레이딩뷰랑 … EMA / SMA 비교해서 뭐가 더 좋은지". 계획: ~/.claude/plans/soft-wibbling-moonbeam.md

신호 = src/screener/ma_cross.py (TradingView 와 1센트까지 같은 계산: 분할만 반영한 가격, Pine ta.ema/ta.sma,
TV 주봉·월봉, Pine ta.crossover). 가격 = data/research/tv_ohlcv_max.parquet (1985 ~ 2025-09-30, SHA256 고정).
**끝난 봉**의 종가에서 신호 → 다음 거래일 시가 진입 (주봉 = 그 주 마지막 거래일, 월봉 = 그 달 마지막 거래일).
수익률 = 총수익(배당 포함) 시가 = Open × Adj Close / Close. 유니버스 = 그날 S&P 500 구성 종목.
이동평균이 long 봉 넘게 쌓이기 전(상장 직후) 사건은 뺀다 (앱도 len < long 이면 라벨 없음).

사전 등록 (--count 로 건수만 보고 → --register, 수익률 계산 전):
  F1 ma_cross_events_v1 — 이벤트 스터디
    조합 = 종류 {EMA, SMA} × 쌍 {일 10/20 · 20/50 · 20/200 · 50/200, 주 10/20 · 10/40, 월 10/20 · 3/10} (SeanEMA 는 교차 규칙 없음 — pine/ema.pine)
    사건 = GC 골든크로스 · GI 임박(앱 규칙, ma_cross._imminent) · GIP 임박 + 종가 > 단기선. 시작 봉만, 10봉 쿨다운.
    주 보유기간 = 일 21 · 주 63 · 월 126 거래일 (나머지 21/63/126 은 참고). 초과수익 = 사건 종목 − 같은 구간 구성 종목
    동일가중. 비용 왕복 0.2%. NW t 의 lag = ceil(h / 봉 길이) 관측 (일 1 · 주 5 · 월 21 거래일).
    판정 = pattern_confirm_research.grade (후보 = t ≥ z · 비용 후 > 0 · 앞/뒤 반 + · 연도 2/3 이상 +). family =
    사건 ≥ 300 & 사건 날짜 ≥ 100 인 검정만 (EVENT_STUDY 와 같은 규칙).
  F2 ma_type_v1 — SMA vs EMA (쌍마다 1건)
    '단기선 > 장기선(끝난 봉)인 구성 종목 동일가중' 포트폴리오, 다음 시가 리밸런스, 비중 변화 × 편도 0.1%.
    EMA판 − SMA판: paired_bootstrap 10,000회 (20일 블록). EMA 우세 = ΔSharpe 하한 > 0 · p_개선아님 ≤ 0.05/N ·
    3구간 중 2+, SMA 우세 = 반대 (상한 < 0 · 1 − p_개선아님 ≤ 0.05/N), 그 밖 = 차이 없음.
  확인 구간 2006-01-03 ~ 2015-12-31: F1 '후보' · F2 '우세'만 다시 본다. 확인 = 같은 부호 · F1 t ≥ 1.96 / F2 95% 구간이
    0을 안 넘음. 홀드아웃(2025-10-01~)은 쓰지 않는다.
  설명용 숫자 (판정·시도 수 없음): 연간 사건 수 · 휩쏘율(반대 교차가 일 10 / 주 4 / 월 2봉 안) · 데드 구간 최저가 봉 →
    골든크로스까지 봉 수와 이미 오른 폭 · 임박 중 갭 축소의 70% 이상이 장기선 하락에서 온 비율(바닥 다지기에서도 흔하다)
    · 임박인데 종가 < 단기선 비율 (HL 처럼 가격은 떨어지는데 임박 — GIP 가 막는 경우).
  참고 (판정 없음): 지금 앱 그대로 (배당 조정 가격 · 일 EMA20/50 · 주 SMA10/40 · 월 SMA3/10) GC·GI.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/ma_cross_research.py --count
  PYTHONPATH=.:src .venv/bin/python scripts/ma_cross_research.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/ma_cross_research.py --run
  PYTHONPATH=.:src .venv/bin/python scripts/ma_cross_research.py --confirm
산출: tier3/MA_CROSS_REPORT.md · ma_cross_counts.csv · ma_cross_events.csv · ma_type.csv · ma_cross_describe.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from idea_batch_research import member_mask  # noqa: E402
from paper_trading.evaluation import paired_bootstrap, sharpe  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from pattern_confirm_research import grade, stats  # noqa: E402
from research_utils import (COST_PER_SIDE, DEV_END, ROUND_TRIP, TIER3_DIR, TRIAL_LOG, bonferroni_z,  # noqa: E402
                            load_ohlcv, log_trials, total_trials)
from screener.ma_cross import cross_frame  # noqa: E402

TV_OHLCV = ROOT / "data" / "research" / "tv_ohlcv_max.parquet"
FAMILY_EV, FAMILY_TYPE = "ma_cross_events_v1", "ma_type_v1"
PERIODS = {"dev": ("2016-01-04", str(DEV_END.date())), "confirm": ("2006-01-03", "2015-12-31")}
PAIRS = {"D": [(10, 20), (20, 50), (20, 200), (50, 200)], "W": [(10, 20), (10, 40)], "M": [(10, 20), (3, 10)]}
KINDS = ("ema", "sma")
EVENTS = {"GC": "골든크로스", "GI": "임박 (앱 규칙)", "GIP": "임박 + 종가 > 단기선"}
PRIMARY_H = {"D": 21, "W": 63, "M": 126}
HORIZONS = (21, 63, 126)
BAR_DAYS = {"D": 1, "W": 5, "M": 21}
COOLDOWN_BARS = 10
WHIPSAW_BARS = {"D": 10, "W": 4, "M": 2}
MIN_EVENTS, MIN_DAYS = 300, 100
N_BOOT = 10_000
TF_NAME = {"D": "일봉", "W": "주봉", "M": "월봉"}


@dataclass(frozen=True)
class Config:
    """이동평균 조합 하나."""

    tf: str
    short: int
    long: int
    kind: str
    adjusted: bool = False  # True = 지금 앱처럼 배당 조정 가격 (참고용)

    @property
    def key(self) -> str:
        tag = "adj-" if self.adjusted else ""
        return f"{tag}{self.tf}-{self.kind.upper()}{self.short}/{self.long}"


CONFIGS = [Config(tf, s, l, k) for tf, pairs in PAIRS.items() for (s, l) in pairs for k in KINDS]
# SeanEMA 원본 크로스 규칙: pine/ema.pine ("6 EMAs Integrated", 2026-10-16 사용자 제공)은 ta.ema 선 6개만 그리고
# 교차 표시·라벨 규칙이 없다. 차트 입력값 20·10·20·200·100·200 → 보이는 선은 EMA 10/20/100/200 뿐.
# 그래서 따로 넣을 조합이 없다 (차트 쌍 10/20 은 CONFIGS 에 이미 있음).
SEAN_CONFIGS: list[Config] = []
SEAN_PINE_CHECKED = True
APP_CONFIGS = [Config("D", 20, 50, "ema", True), Config("W", 10, 40, "sma", True), Config("M", 3, 10, "sma", True)]


# ── 데이터 ─────────────────────────────────────────────────────────────────────────────


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(비조정 종가, 배당 조정 종가, 총수익 시가) — 개발 구간 끝(DEV_END)까지."""
    px = load_ohlcv(TV_OHLCV)
    close = px.xs("Close", axis=1, level="Price").astype("float64")
    adj = px.xs("Adj Close", axis=1, level="Price").astype("float64")
    open_ = px.xs("Open", axis=1, level="Price").astype("float64")
    open_tr = open_ * (adj / close)
    return close, adj, open_tr


def membership() -> Membership:
    return Membership(load_membership(ROOT / "data" / "universe" / "sp500_membership.csv"))


# ── 사건 ───────────────────────────────────────────────────────────────────────────────


def _cooldown(pos: np.ndarray, n: int = COOLDOWN_BARS) -> np.ndarray:
    """봉 번호 배열(오름차순) → n봉 안 재사건 제외 마스크."""
    keep = np.zeros(len(pos), dtype=bool)
    last = -10**9
    for i, p in enumerate(pos):
        if p - last >= n:
            keep[i] = True
            last = p
    return keep


def ticker_events(args: tuple[str, pd.Series, list[Config], str]) -> tuple[list[dict], dict[str, pd.Series]]:
    """한 종목: 조합마다 사건 행 + 끝난 봉의 골든 상태 시리즈 (F2 용)."""
    ticker, close, configs, as_of = args
    rows: list[dict] = []
    states: dict[str, pd.Series] = {}
    close = close.dropna()
    if close.size < 30:
        return rows, states
    for cfg in configs:
        f = cross_frame(close, cfg.short, cfg.long, cfg.kind, cfg.tf, pd.Timestamp(as_of))
        f = f[f["complete"]]
        ok = f["n_bar"] >= cfg.long - 1
        states[cfg.key] = f["golden"].where(ok)
        f = f[ok]
        if f.empty:
            continue
        pos = f["n_bar"].to_numpy()
        gi_on = f["gi"] & ~f["gi"].shift(1, fill_value=False)
        gip_on = f["gi_plus"] & ~f["gi_plus"].shift(1, fill_value=False)
        dc_pos = pos[f["dc"].to_numpy()]
        closes = f["close"].to_numpy()
        lag = _gap_lag(cfg.tf)
        for ev, mask in (("GC", f["gc"].to_numpy()), ("GI", gi_on.to_numpy()), ("GIP", gip_on.to_numpy())):
            idx = np.flatnonzero(mask)
            idx = idx[_cooldown(pos[idx])]
            for i in idx:
                row = {"config": cfg.key, "event": ev, "티커": ticker, "date": f.index[i]}
                if ev == "GC":
                    row.update(_gc_describe(f, i, pos, dc_pos, closes, cfg.tf))
                else:
                    row.update(_gi_describe(f, i, lag))
                rows.append(row)
    return rows, states


def _gap_lag(tf: str) -> int:
    """임박 규칙이 갭을 비교하는 봉 수 (ma_cross.TF_RULES slope_lookback − 1)."""
    from screener.ma_cross import TF_RULES
    return TF_RULES[tf].slope_lookback - 1


def _gc_describe(f: pd.DataFrame, i: int, pos: np.ndarray, dc_pos: np.ndarray, closes: np.ndarray, tf: str) -> dict:
    """골든크로스 설명용: 휩쏘 · 직전 데드 구간 최저가 봉에서 몇 봉 · 이미 오른 폭."""
    p = pos[i]
    later = dc_pos[dc_pos > p]
    whipsaw = bool(later.size and later[0] - p <= WHIPSAW_BARS[tf])
    earlier = dc_pos[dc_pos < p]
    start = int(np.searchsorted(pos, earlier[-1])) if earlier.size else max(0, i - 260)
    seg = closes[start:i + 1]
    j = int(np.argmin(seg))
    return {"whipsaw": whipsaw, "lag_bars": int(len(seg) - 1 - j), "missed": float(closes[i] / seg[j] - 1)}


def _gi_describe(f: pd.DataFrame, i: int, lag: int) -> dict:
    """임박 설명용: 갭이 좁혀진 몫 중 장기선이 내려와서 생긴 비율 · 종가 < 단기선."""
    if i < lag:
        return {"drop_share": np.nan, "below_short": bool(f["close"].iloc[i] < f["ma_s"].iloc[i])}
    ds = f["ma_s"].iloc[i] - f["ma_s"].iloc[i - lag]
    dl = f["ma_l"].iloc[i] - f["ma_l"].iloc[i - lag]
    narrowing = ds - dl  # 단기 − 장기 차이가 커진 양 (임박이면 > 0)
    share = float(-dl / narrowing) if narrowing > 0 else np.nan
    return {"drop_share": share, "below_short": bool(f["close"].iloc[i] < f["ma_s"].iloc[i])}


def build_events(close: pd.DataFrame, configs: list[Config], workers: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """모든 종목·조합의 사건 + 상태 행렬(끝난 봉 날짜 기준, F2 에서 일별로 채움)."""
    as_of = str(close.index.max().date())
    jobs = [(t, close[t], configs, as_of) for t in close.columns]
    rows: list[dict] = []
    per_cfg: dict[str, dict[str, pd.Series]] = {c.key: {} for c in configs}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (r, st) in enumerate(ex.map(ticker_events, jobs, chunksize=8)):
            rows.extend(r)
            for key, s in st.items():
                per_cfg[key][jobs[k][0]] = s
    events = pd.DataFrame(rows)
    states = {key: pd.DataFrame(d) for key, d in per_cfg.items()}
    return events, states


def in_period(events: pd.DataFrame, mem_ok: pd.DataFrame, period: str) -> pd.DataFrame:
    """기간 안 + 그날 구성 종목인 사건만."""
    a, b = PERIODS[period]
    e = events[(events["date"] >= a) & (events["date"] <= b)].copy()
    stacked = mem_ok.stack()
    stacked = stacked[stacked]
    key = pd.MultiIndex.from_arrays([e["date"], e["티커"]])
    return e[key.isin(stacked.index)]


# ── F1 이벤트 스터디 ──────────────────────────────────────────────────────────────────


def tests(configs: list[Config]) -> list[tuple[Config, str]]:
    return [(c, ev) for c in configs for ev in EVENTS]


def test_id(cfg: Config, ev: str, h: int) -> str:
    return f"{cfg.key}|{ev}|h{h}"


def count_table(ev: pd.DataFrame, configs: list[Config]) -> pd.DataFrame:
    """검정마다 사건 수·사건 날짜 수 (수익률 안 봄)."""
    rows = []
    for cfg, e in tests(configs):
        sub = ev[(ev["config"] == cfg.key) & (ev["event"] == e)]
        rows.append({"test_id": test_id(cfg, e, PRIMARY_H[cfg.tf]), "config": cfg.key, "event": e,
                     "n_events": len(sub), "n_days": sub["date"].nunique(),
                     "family": len(sub) >= MIN_EVENTS and sub["date"].nunique() >= MIN_DAYS})
    return pd.DataFrame(rows)


def excess_matrix(open_tr: pd.DataFrame, mem: pd.DataFrame, h: int) -> pd.DataFrame:
    """사건일 d: d+1 시가 → d+1+h 시가 − 같은 구간 구성 종목 동일가중 (pattern_confirm_research.excess_returns 와 같음)."""
    fwd = open_tr.shift(-(1 + h)) / open_tr.shift(-1) - 1
    return fwd.sub(fwd.where(mem).mean(axis=1), axis=0)


def event_table(ev: pd.DataFrame, open_tr: pd.DataFrame, mem: pd.DataFrame, configs: list[Config],
                family: set[str], z: float) -> pd.DataFrame:
    """검정마다 통계 + 판정 (family 안 + 주 보유기간만 판정, 나머지는 '참고')."""
    ars = {h: excess_matrix(open_tr, mem, h).stack() for h in HORIZONS}
    rows = []
    for cfg, e in tests(configs):
        sub = ev[(ev["config"] == cfg.key) & (ev["event"] == e)]
        key = pd.MultiIndex.from_arrays([sub["date"], sub["티커"]])
        for h in HORIZONS:
            s = ars[h].reindex(key)
            s.index = key
            lag = math.ceil(h / BAR_DAYS[cfg.tf])
            r = stats(s, lag)  # stats 의 두 번째 인자는 NW lag 로만 쓰인다
            tid = test_id(cfg, e, h)
            primary = h == PRIMARY_H[cfg.tf]
            judged = primary and tid in family
            rows.append({"test_id": tid, "config": cfg.key, "tf": cfg.tf, "kind": cfg.kind,
                         "pair": f"{cfg.short}/{cfg.long}", "event": e, "h": h, "primary": primary,
                         "nw_lag": lag, **r, "판정": grade(r, z) if judged else "참고"})
    return pd.DataFrame(rows)


# ── F2 SMA vs EMA ─────────────────────────────────────────────────────────────────────


def state_weights(state: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index, mem: pd.DataFrame,
                  tradable: pd.DataFrame) -> pd.DataFrame:
    """끝난 봉 상태를 일별로 채워 다음 날 시가부터 보유 → 동일가중 비중 (t행 = t 시가 → t+1 시가)."""
    st = state.reindex(columns=columns)
    st = st.reindex(st.index.union(index)).ffill().reindex(index)
    hold = st.shift(1).fillna(False).astype(bool) & mem & tradable
    n = hold.sum(axis=1)
    return hold.astype(float).div(n.where(n > 0), axis=0).fillna(0.0)


def weighted_returns(w: pd.DataFrame, r: pd.DataFrame, cost: float = COST_PER_SIDE) -> pd.Series:
    """isb_vs_pt1.weighted_returns 와 같다: t 비중 × (t 시가 → t+1 시가) − 비중 변화 × 편도 비용."""
    gross = (w * r.fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    return gross - turnover * cost


def type_table(states: dict[str, pd.DataFrame], open_tr: pd.DataFrame, mem: pd.DataFrame,
               period: str, n_tests: int) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """쌍마다 EMA판 vs SMA판."""
    a, b = PERIODS[period]
    idx = open_tr.loc[a:b].index
    r = (open_tr.shift(-1) / open_tr - 1).loc[idx]
    tradable = (open_tr.notna() & open_tr.shift(-1).notna()).loc[idx]
    m = mem.loc[idx]
    ew = weighted_returns(m.astype(float).div(m.sum(axis=1).where(lambda x: x > 0), axis=0).fillna(0.0), r)
    series: dict[str, pd.Series] = {"구성 종목 동일가중": ew}
    rows = []
    for tf, pairs in PAIRS.items():
        for s_, l_ in pairs:
            ret = {}
            for k in KINDS:
                key = Config(tf, s_, l_, k).key
                ret[k] = weighted_returns(state_weights(states[key], idx, open_tr.columns, m, tradable), r)
                series[key] = ret[k]
            st = paired_bootstrap(ret["ema"].tolist(), ret["sma"].tolist(), n_boot=N_BOOT)
            p_bonf = 0.05 / n_tests
            if st["ΔSharpe_하한"] > 0 and st["p_개선아님"] <= p_bonf and st["구간_개선"] >= st["구간_수"] - 1:
                vd = "EMA 우세"
            elif st["ΔSharpe_상한"] < 0 and 1 - st["p_개선아님"] <= p_bonf and st["구간_개선"] <= 1:
                vd = "SMA 우세"
            else:
                vd = "차이 없음"
            rows.append({"test_id": f"{tf}-{s_}/{l_}|EMA-SMA", "tf": tf, "pair": f"{s_}/{l_}",
                         "Sharpe_EMA": sharpe(ret["ema"].tolist()), "Sharpe_SMA": sharpe(ret["sma"].tolist()),
                         "Sharpe_EW": sharpe(ew.tolist()),
                         "CAGR_EMA": _cagr(ret["ema"]), "CAGR_SMA": _cagr(ret["sma"]), "CAGR_EW": _cagr(ew),
                         **st, "판정": vd})
    return pd.DataFrame(rows), series


def _cagr(r: pd.Series) -> float:
    eq = float((1 + r).prod())
    years = len(r) / 252
    return eq ** (1 / years) - 1 if years > 0 and eq > 0 else float("nan")


# ── 설명용 ─────────────────────────────────────────────────────────────────────────────


def describe(ev: pd.DataFrame, mem: pd.DataFrame, configs: list[Config], period: str) -> pd.DataFrame:
    """조합마다 잡음·지연 숫자 (판정 없음)."""
    a, b = PERIODS[period]
    ticker_years = float(mem.loc[a:b].sum().sum()) / 252
    rows = []
    for cfg in configs:
        gc = ev[(ev["config"] == cfg.key) & (ev["event"] == "GC")]
        gi = ev[(ev["config"] == cfg.key) & (ev["event"] == "GI")]
        rows.append({"config": cfg.key, "tf": cfg.tf, "kind": cfg.kind, "pair": f"{cfg.short}/{cfg.long}",
                     "GC_per_100_ticker_years": len(gc) / ticker_years * 100,
                     "whipsaw": gc["whipsaw"].mean() if len(gc) else np.nan,
                     "lag_bars_median": gc["lag_bars"].median() if len(gc) else np.nan,
                     "missed_median": gc["missed"].median() if len(gc) else np.nan,
                     "GI_n": len(gi),
                     "GI_drop_share_ge70": (gi["drop_share"] >= 0.7).mean() if len(gi) else np.nan,
                     "GI_below_short": gi["below_short"].mean() if len(gi) else np.nan})
    return pd.DataFrame(rows)


# ── 등록 ───────────────────────────────────────────────────────────────────────────────


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame()


def _registered(family: str) -> pd.Series | None:
    log = _log()
    if log.empty:
        return None
    hit = log[(log["family"] == family) & (log["판정"] == "등록")]
    return hit.iloc[-1] if len(hit) else None


def all_configs() -> list[Config]:
    return CONFIGS + SEAN_CONFIGS


def counts_path() -> Path:
    return TIER3_DIR / "ma_cross_counts.csv"


def do_count(workers: int) -> None:
    close, _, _ = load_prices()
    mem = member_mask(close.index, close.columns, membership())
    ev, _ = build_events(close, all_configs(), workers)
    ev = in_period(ev, mem, "dev")
    ct = count_table(ev, all_configs())
    ct.to_csv(counts_path(), index=False, encoding="utf-8-sig")
    print(ct.to_string(index=False))
    print(f"\nfamily (사건 ≥ {MIN_EVENTS} & 날짜 ≥ {MIN_DAYS}): {int(ct['family'].sum())} / {len(ct)} · "
          f"Bonferroni z = {bonferroni_z(int(ct['family'].sum())):.2f} · F2 {sum(len(p) for p in PAIRS.values())}건")


def do_register() -> None:
    if _registered(FAMILY_EV) is not None:
        raise SystemExit(f"{FAMILY_EV} 은 이미 등록됐다")
    if not SEAN_PINE_CHECKED:
        raise SystemExit("SeanEMA Pine 을 아직 안 읽었다 — pine/ema.pine 을 확인한 뒤 등록")
    ct = pd.read_csv(counts_path(), encoding="utf-8-sig")
    fam = ct[ct["family"]]
    n_ev, n_ty = len(fam), sum(len(p) for p in PAIRS.values())
    log_trials("ma-cross", FAMILY_EV, n_ev, [{
        "test_id": "(등록)", "판정": "등록",
        "메모": f"F1 {n_ev}건 z={bonferroni_z(n_ev):.2f}: " + ",".join(fam["test_id"])}])
    log_trials("ma-cross", FAMILY_TYPE, n_ty, [{
        "test_id": "(등록)", "판정": "등록",
        "메모": f"F2 {n_ty}건 EMA vs SMA 상태 포트폴리오, p ≤ 0.05/{n_ty}, paired_bootstrap {N_BOOT}"}])
    print(f"등록: {FAMILY_EV} {n_ev}건 · {FAMILY_TYPE} {n_ty}건 · 누적 시도 {total_trials()}")


def do_run(workers: int, period: str = "dev") -> None:
    reg = _registered(FAMILY_EV)
    if reg is None or _registered(FAMILY_TYPE) is None:
        raise SystemExit("등록 먼저 (--count → --register)")
    family = set(str(reg["메모"]).split(": ", 1)[1].split(","))
    n_ev, n_ty = len(family), sum(len(p) for p in PAIRS.values())
    z = bonferroni_z(n_ev)
    close, adj, open_tr = load_prices()
    mem = member_mask(close.index, close.columns, membership())
    ev, states = build_events(close, all_configs(), workers)
    ev = in_period(ev, mem, period)
    a, b = PERIODS[period]
    # 개발 구간은 DEV_END 뒤(홀드아웃)를 못 본다. 확인 구간은 끝난 뒤 수익까지 본다 (2016 년 초까지 걸쳐도 됨)
    ot, mm = (open_tr.loc[:b], mem.loc[:b]) if period == "dev" else (open_tr, mem)
    f1 = event_table(ev, ot, mm, all_configs(), family, z)
    f2, series = type_table(states, open_tr, mem, period, n_ty)
    desc = describe(ev, mem, all_configs(), period)
    app_ev, _ = build_events(adj, APP_CONFIGS, workers)
    app = event_table(in_period(app_ev, mem, period), ot, mm, APP_CONFIGS, set(), z)
    tag = "" if period == "dev" else f"_{period}"
    f1.to_csv(TIER3_DIR / f"ma_cross_events{tag}.csv", index=False, encoding="utf-8-sig")
    f2.to_csv(TIER3_DIR / f"ma_type{tag}.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(TIER3_DIR / f"ma_cross_describe{tag}.csv", index=False, encoding="utf-8-sig")
    app.to_csv(TIER3_DIR / f"ma_cross_app{tag}.csv", index=False, encoding="utf-8-sig")
    if period == "dev":
        judged = f1[f1["판정"] != "참고"]
        log_trials("ma-cross", FAMILY_EV, n_ev,
                   [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}", "판정": r["판정"]} for _, r in judged.iterrows()])
        log_trials("ma-cross", FAMILY_TYPE, n_ty,
                   [{"test_id": r.test_id, "t": f"{r['ΔSharpe']:+.3f}", "판정": r["판정"]} for _, r in f2.iterrows()])
    print(f1[f1["primary"]][["test_id", "n_events", "mean_ar", "mean_net", "t_nw", "판정"]].to_string(index=False))
    print(f2[["test_id", "Sharpe_EMA", "Sharpe_SMA", "ΔSharpe", "ΔSharpe_하한", "ΔSharpe_상한", "판정"]].to_string(index=False))
    print(desc.to_string(index=False))
    print(app[app["primary"]][["test_id", "n_events", "mean_ar", "mean_net", "t_nw"]].to_string(index=False))


# ── 보고서 ─────────────────────────────────────────────────────────────────────────────

SUMMARY: list[str] = [  # 결과를 본 뒤 쓴 쉬운 말 요약 (2026-10-16 실행)
    "1. **계산은 TradingView 와 같다.** S&P 500 종목의 일 10·20·200 / 주 10·20 / 월 10·20 선이 차트 범례와 586/586 "
    "1센트까지 일치. TradingView EMA 는 처음 n봉 평균에서 시작한다(Pine ta.ema) — 이걸 맞추자 상장 몇 년 된 종목도 맞았다.",
    "2. **골든크로스·골든크로스 임박은 돈이 안 된다.** EMA·SMA × 8쌍 × 3사건 = 48가지 정의가 모두 탈락. 비용을 빼면 "
    "같은 날 S&P 500 평균보다 낫지 않다. 지금 앱의 일봉 임박(EMA20/50)은 21일 −0.34% (t −2.17)로 오히려 약간 나쁘다 "
    "(기준 z 3.28 에는 못 미침). 이전 연구(EVENT_STUDY_pit, GC_INSIDER_REV)와 같은 결론이다.",
    "3. **SMA vs EMA — 수익으로는 차이가 없다.** 8쌍 모두 '차이 없음'. 주봉·월봉 10/20 은 EMA 가 조금 나았고"
    "(ΔSharpe +0.10 · +0.05) 2006~2015 참고 구간에서도 방향은 같았지만, 두 번 다 운과 구분되지 않는다.",
    "4. **성격 차이는 뚜렷하다.** EMA 는 0~12봉 먼저 교차하지만(대부분 1~4봉, 일 50/200 은 12봉) 바로 되돌리는 "
    "휩쏘 비율이 약 2배다(일 20/50: 19% vs 9%, 주 10/20: 17% vs 9%, 월 10/20: 10% vs 5%). 반대로 HL 같은 "
    "'가격은 떨어지는데 임박'은 대부분의 쌍에서 SMA 가 더 많다(일 50/200: 18% vs 5%, 주 10/40: 13% vs 6%, "
    "월 3/10: 24% vs 7%) — SMA 기간에서 옛 고점이 빠질 때 생긴다. '임박 + 종가 > 단기선'(GIP) 이 이걸 막는다.",
    "5. **추세 보유도 그냥 들고 있기를 못 이긴다.** '단기선 > 장기선'인 종목만 들고 있으면 EMA·SMA 모두 S&P 500 "
    "동일가중 보유(연 13.1%)보다 수익이 낮다. Sharpe 는 일·주봉이 더 낮고(0.48~0.76 vs 0.78) 월봉만 비슷하다(0.74~0.81).",
    "6. **PT-1 에서는 라벨을 빼거나(P1) TradingView 기준으로 바꿔도(P2) 차이가 운으로 설명된다** (ΔSharpe −0.11 · −0.05, "
    "95% 구간 ±0.4). 거래가 1/3 쯤 바뀌는데 좋아진다는 증거도, '안 나빠진다'는 증거도 없다 → 점수 규칙은 그대로 둔다.",
    "7. **그래서:** 사이트·메일 **표시**는 차트와 같은 계산(비조정 가격 · EMA · 끝난 봉/진행 중 봉 구분)으로 바꾸는 게 맞고, "
    "점수에 쓰는 라벨은 바꿀 이유가 없다. 둘 다 사용자 결정.",
]


def _pct(v: float, signed: bool = True) -> str:
    return "" if pd.isna(v) else (f"{v:+.2%}" if signed else f"{v:.0%}")


def _read(name: str) -> pd.DataFrame | None:
    path = TIER3_DIR / name
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def write_report() -> None:
    """CSV → MA_CROSS_REPORT.md."""
    f1, f2, desc, app = (_read(n) for n in ("ma_cross_events.csv", "ma_type.csv", "ma_cross_describe.csv",
                                             "ma_cross_app.csv"))
    pt1 = _read("pt1_cross.csv")
    f1c, f2c = _read("ma_cross_events_confirm.csv"), _read("ma_type_confirm.csv")
    if f1 is None or f2 is None:
        raise SystemExit("--run 먼저")
    reg = _registered(FAMILY_EV)
    n_ev = len(str(reg["메모"]).split(": ", 1)[1].split(",")) if reg is not None else 0
    n_ty = len(f2)
    prim = f1[f1["primary"]].copy()
    verdicts = prim["판정"].value_counts()
    L = ["# 골든크로스 · 골든크로스 임박 백테스트 — TradingView 와 같은 계산 (ma_cross_v1)", "",
         f"- 기간 {PERIODS['dev'][0]} ~ {PERIODS['dev'][1]} (개발 구간, 홀드아웃 2025-10-01~ 안 씀) · 그날 S&P 500 구성 종목",
         "- 신호 = `src/screener/ma_cross.py`: 분할만 반영한 가격 (TradingView 기본 차트) · Pine ta.ema/ta.sma · TV 주봉·월봉 · "
         "Pine ta.crossover. TradingView 범례와 대조 → [MA_CROSS_TV_CHECK.md](MA_CROSS_TV_CHECK.md) "
         "(S&P 500 · 연구에 쓰는 선 100% 일치)",
         "- **끝난 봉** 종가에서 신호 → 다음 거래일 시가 진입 → h 거래일 뒤 시가. 수익은 배당 포함. 비용 왕복 0.2%",
         "- 초과수익 = 사건 종목 − 같은 구간 구성 종목 동일가중. 같은 종목·사건 10봉 쿨다운",
         f"- 사전 등록: F1 `{FAMILY_EV}` {n_ev}건 (Bonferroni z = {bonferroni_z(n_ev):.2f}) · F2 `{FAMILY_TYPE}` "
         f"{n_ty}건 (p ≤ 0.05/{n_ty}) · F3 `pt1_cross_v1` 2건 · 누적 시도 {total_trials()}건",
         "- SeanEMA(`pine/ema.pine`)는 ta.ema 선만 그리고 교차 규칙이 없다 → 차트 쌍은 EMA10/20 (주황 × 보라)", ""]
    if SUMMARY:
        L += ["## 쉬운 말 요약", "", *SUMMARY, ""]
    L += ["## F1 — 사건 뒤 수익 (주 보유기간: 일 21 · 주 63 · 월 126 거래일)", "",
          "- 판정: " + " · ".join(f"{k} {v}" for k, v in verdicts.items()), "",
          "| 판정 | 봉 | 종류 | 쌍 | 사건 | n | 평균 초과(비용 전) | 비용 후 | 적중률 | t | 연도+ | 앞/뒤 반 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"D": 0, "W": 1, "M": 2}
    ev_order = {e: i for i, e in enumerate(EVENTS)}
    prim["_k"] = [(order[t], *map(int, pr.split("/")), ev_order[e], k) for t, pr, e, k in
                  zip(prim["tf"], prim["pair"], prim["event"], prim["kind"])]
    prim = prim.sort_values("_k").drop(columns="_k")
    for r in prim.itertuples():
        L.append(f"| {r.판정} | {TF_NAME[r.tf]} | {r.kind.upper()} | {r.pair} | {EVENTS[r.event]} | {r.n_events} | "
                 f"{_pct(r.mean_ar)} | {_pct(r.mean_net)} | {_pct(r.hit, False)} | {r.t_nw:+.2f} | "
                 f"{r.years_pos}/{r.n_years} | {_pct(r.first_half)} / {_pct(r.second_half)} |")
    L += ["", "### 참고: 보유기간별 비용 후 초과수익 (t)", "", "| 조합 | 사건 | 21일 | 63일 | 126일 |", "|---|---|---|---|---|"]
    for (cfg, ev), g in f1.groupby(["config", "event"], sort=False):
        cells = []
        for h in HORIZONS:
            x = g[g["h"] == h]
            cells.append(f"{_pct(x['mean_net'].iloc[0])} ({x['t_nw'].iloc[0]:+.1f})" if len(x) else "")
        L.append(f"| {cfg} | {EVENTS[ev]} | " + " | ".join(cells) + " |")
    if app is not None:
        L += ["", "### 참고: 지금 앱 그대로 (배당 조정 가격) vs TradingView 가격 — 주 보유기간", "",
              "| 조합 | 사건 | 앱(배당 조정) n · 비용 후 · t | TV 가격 n · 비용 후 · t |", "|---|---|---|---|"]
        for r in app[app["primary"] & app["event"].isin(["GC", "GI"])].itertuples():
            tv_key = r.config.replace("adj-", "")
            m = prim[(prim["config"] == tv_key) & (prim["event"] == r.event)]
            tv = f"{int(m['n_events'].iloc[0])} · {_pct(m['mean_net'].iloc[0])} · {m['t_nw'].iloc[0]:+.2f}" if len(m) else ""
            L.append(f"| {tv_key} | {EVENTS[r.event]} | {r.n_events} · {_pct(r.mean_net)} · {r.t_nw:+.2f} | {tv} |")
    L += ["", "## F2 — SMA vs EMA (같은 쌍, '단기선 > 장기선' 종목 동일가중 보유)", "",
          "| 판정 | 봉 | 쌍 | Sharpe EMA | Sharpe SMA | 구성 종목 동일가중 | ΔSharpe(EMA−SMA) [95%] | p(EMA 개선 아님) | 구간 | CAGR EMA / SMA |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in f2.itertuples():
        L.append(f"| {r.판정} | {TF_NAME[r.tf]} | {r.pair} | {r.Sharpe_EMA:.2f} | {r.Sharpe_SMA:.2f} | {r.Sharpe_EW:.2f} | "
                 f"{r.ΔSharpe:+.3f} [{r.ΔSharpe_하한:+.3f}, {r.ΔSharpe_상한:+.3f}] | {r.p_개선아님:.3f} | "
                 f"{r.구간_개선}/{r.구간_수} | {_pct(r.CAGR_EMA)} / {_pct(r.CAGR_SMA)} |")
    if desc is not None:
        L += ["", "### 잡음·지연 (설명용, 판정 없음)", "",
              "- 휩쏘 = 골든크로스 뒤 일 10 / 주 4 / 월 2봉 안에 데드크로스. 지연 = 직전 데드 구간 최저가 봉 → 골든크로스 봉.",
              "- 장기선 하락 몫 = 임박 중 갭이 좁혀진 몫의 70% 이상이 **장기선이 내려와서** 생긴 비율. 가격이 멈추고 바닥을 다질 때도 "
              "흔해서 그 자체로 가짜는 아니다.",
              "- **임박인데 종가 < 단기선** = HL 처럼 가격은 떨어지는데 임박이 뜬 비율 (GIP 가 막는 경우).", "",
              "| 봉 | 쌍 | 종류 | 골든크로스 / 100종목·년 | 휩쏘 | 지연(봉, 중앙) | 이미 오른 폭(중앙) | 임박 n | 장기선 하락 몫 ≥70% | 임박인데 종가 < 단기선 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in desc.itertuples():
            L.append(f"| {TF_NAME[r.tf]} | {r.pair} | {r.kind.upper()} | {r.GC_per_100_ticker_years:.1f} | "
                     f"{_pct(r.whipsaw, False)} | {r.lag_bars_median:.0f} | {_pct(r.missed_median)} | {r.GI_n} | "
                     f"{_pct(r.GI_drop_share_ge70, False)} | {_pct(r.GI_below_short, False)} |")
    if pt1 is not None:
        L += ["", "## F3 — PT-1 엔진 (2016-01-04 ~ 2025-09-30, $5,000 · 3슬롯)", "",
              "| 변형 | Sharpe | CAGR | MDD | 거래 | BASE 와 같은 거래 | ΔSharpe [95%] | p | 구간 | 판정 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in pt1.iterrows():
            if r["test_id"] == "BASE":
                L.append(f"| PT-1 그대로 | {r['Sharpe']:.2f} | {_pct(r['CAGR'])} | {_pct(r['MDD'])} | {int(r['거래수'])} "
                         "| | | | | 기준 |")
                continue
            L.append(f"| {r['test_id']} {r['이름']} | {r['Sharpe']:.2f} | {_pct(r['CAGR'])} | {_pct(r['MDD'])} | "
                     f"{int(r['거래수'])} | {_pct(r['같은 거래 비율'], False)} | {r['ΔSharpe']:+.2f} "
                     f"[{r['하한']:+.2f}, {r['상한']:+.2f}] | {r['p']:.3f} | {r['구간']} | **{r['판정']}** |")
    cand = prim[prim["판정"] == "후보"]
    winners = f2[f2["판정"] != "차이 없음"]
    if cand.empty and winners.empty:
        L += ["", "## 확인 구간 2006-01 ~ 2015-12", "",
              "- 사전 등록상 확인할 것(F1 후보 · F2 우세)이 없다."]
        if f1c is not None and f2c is not None:
            L += ["- **참고로만** 같은 계산을 2006~2015 에 돌렸다 (판정·시도 기록에 안 씀). 상장폐지 종목이 더 많이 빠져 "
                  "생존 편향이 크다.", "",
                  "| 봉 | 쌍 | Sharpe EMA | Sharpe SMA | 구성 종목 동일가중 | ΔSharpe(EMA−SMA) [95%] |", "|---|---|---|---|---|---|"]
            for r in f2c.itertuples():
                L.append(f"| {TF_NAME[r.tf]} | {r.pair} | {r.Sharpe_EMA:.2f} | {r.Sharpe_SMA:.2f} | {r.Sharpe_EW:.2f} | "
                         f"{r.ΔSharpe:+.3f} [{r.ΔSharpe_하한:+.3f}, {r.ΔSharpe_상한:+.3f}] |")
            pc = f1c[f1c["primary"]]
            L += ["", f"- F1 같은 48개 (참고): 비용 후 평균이 + 인 것 {int((pc['mean_net'] > 0).sum())}개, "
                  f"t ≥ {bonferroni_z(n_ev):.2f} 인 것 {int((pc['t_nw'] >= bonferroni_z(n_ev)).sum())}개, "
                  f"t ≥ 1.96 인 것 {int((pc['t_nw'] >= 1.96).sum())}개."]
    if f1c is not None and not (cand.empty and winners.empty):
        L += ["", "## 확인 구간 2006-01 ~ 2015-12 (F1 후보 · F2 우세만)", ""]
        for r in cand.itertuples():
            m = f1c[(f1c["test_id"] == r.test_id)]
            if len(m):
                ok = np.sign(m["mean_net"].iloc[0]) == np.sign(r.mean_net) and m["t_nw"].iloc[0] >= 1.96
                L.append(f"- {r.test_id}: 비용 후 {_pct(m['mean_net'].iloc[0])} · t {m['t_nw'].iloc[0]:+.2f} → "
                         f"{'확인' if ok else '확인 안 됨'}")
        if f2c is not None:
            for r in f2[f2["판정"] != "차이 없음"].itertuples():
                m = f2c[f2c["test_id"] == r.test_id]
                if len(m):
                    same = np.sign(m["ΔSharpe"].iloc[0]) == np.sign(r.ΔSharpe)
                    excl = m["ΔSharpe_하한"].iloc[0] > 0 or m["ΔSharpe_상한"].iloc[0] < 0
                    L.append(f"- {r.test_id}: ΔSharpe {m['ΔSharpe'].iloc[0]:+.3f} [{m['ΔSharpe_하한'].iloc[0]:+.3f}, "
                             f"{m['ΔSharpe_상한'].iloc[0]:+.3f}] → {'확인' if same and excl else '확인 안 됨'}")
        if len(L) and L[-1] == "":
            L.append("- 확인할 후보 없음")
    L += ["", "## 한계", "",
          "- 야후에 가격이 없는 상장폐지 종목은 빠진다 (2006~2025 S&P 500 943종목 중 638종목만 가격 있음 → "
          "[tv_prices_coverage.csv](tv_prices_coverage.csv)). 망한 종목의 골든크로스가 빠져 결과가 좋은 쪽으로 기운다.",
          "- 신호는 **끝난 봉** 기준이다. 라이브 사이트·메일은 주·월봉을 **진행 중 봉**으로 보여 주므로 같은 날 라벨이 다를 수 있다.",
          "- 재상장·합병 종목과 아주 오래된 기록은 TradingView 와 야후가 다르다 (대조 보고서 참고).",
          "- 개발 구간은 그동안 많은 시도에 쓰였다. 결과가 좋아도 라이브 반영은 사용자 결정이다."]
    (TIER3_DIR / "MA_CROSS_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))


def do_app(workers: int, period: str = "dev") -> None:
    """참고: 지금 앱 그대로(배당 조정 가격 · 일 EMA20/50 · 주 SMA10/40 · 월 SMA3/10). 판정·시도 기록 없음."""
    close, adj, open_tr = load_prices()
    mem = member_mask(close.index, close.columns, membership())
    a, b = PERIODS[period]
    ot, mm = (open_tr.loc[:b], mem.loc[:b]) if period == "dev" else (open_tr, mem)
    app_ev, _ = build_events(adj, APP_CONFIGS, workers)
    app = event_table(in_period(app_ev, mem, period), ot, mm, APP_CONFIGS, set(), bonferroni_z(1))
    tag = "" if period == "dev" else f"_{period}"
    app.to_csv(TIER3_DIR / f"ma_cross_app{tag}.csv", index=False, encoding="utf-8-sig")
    print(app[app["primary"]][["test_id", "n_events", "mean_ar", "mean_net", "t_nw"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--confirm", action="store_true")
    g.add_argument("--app", action="store_true", help="참고: 지금 앱 그대로 표만")
    g.add_argument("--report", action="store_true", help="CSV 들로 MA_CROSS_REPORT.md 쓰기")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.count:
        do_count(args.workers)
    elif args.register:
        do_register()
    elif args.run:
        do_run(args.workers, "dev")
    elif args.app:
        do_app(args.workers, "dev")
    elif args.report:
        write_report()
    else:
        do_run(args.workers, "confirm")


if __name__ == "__main__":
    main()
