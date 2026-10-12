#!/usr/bin/env python3
"""예상상승분포 검증: 메일의 '예상 상승 분포' 박스와 표의 Target / Next Target 이 맞는가.

메일 숫자는 upside_model 의 버킷(전략 × CCS 구간 × 별점 구간)별 과거 거래 '보유 중 최고 상승(MFE)' 분위수였다.
Target = P50, Next Target = P75. 표시 전용이라 매매와는 무관했다.
2026-10-11 결과 불통과 → 메일 표시와 upside_model.py 를 지웠다. 버킷 규칙은 이 파일에 그대로 옮겨 두어 다시 돌릴 수 있다.

방법 (개발 구간만, 홀드아웃 안 씀):
  1. 10년 PIT PT-1 백테스트(2016-01-04 ~ 2025-09-30)를 그대로 재현해 거래를 얻는다.
  2. 거래마다 정답 MFE = 보유 중 장중 최고가 / 진입가 − 1.
     진입은 entry_date 시가 체결이라 그날 고가부터 넣고, 청산은 exit_date 시가 체결이라 그날 고가는 빼고 청산가를 넣는다.
  3. 워크포워드: 진입일 전에 끝난 거래만으로 M1(지금 버킷 모델)과 M0(모든 종목 같은 숫자 = 전체 분위수)을 만든다.
  4. 실전 PT-1 거래(2026-04~)의 저장된 P50/P75 를 장중 고가로 다시 잰 MFE 와 비교한다
     (배포된 표는 2026-04-15 에 만들어졌으니 그 뒤 거래는 표본 외).

사전 등록 (TRIAL_LOG family upside_display, 1건):
  U1 버킷 모델이 '모든 종목 같은 숫자'보다 P50 을 잘 맞히는가.
     통과 = M0−M1 P50 pinball 손실 차이의 진입월 블록 부트스트랩 95% 구간 하한 > 0
            그리고 M1 P50 도달률 45~55%.
  통과 → 메일에 숫자 하나(P50)만 남긴다 / 불통과 → 예상상승 표시를 전부 지운다.

사용법: PYTHONPATH=.:src python scripts/validate_upside_walkforward.py
산출: tier3/UPSIDE_VALIDATION.md, tier3/upside_validation.csv
"""

from __future__ import annotations

import json
import re
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import paper_trading.backtest as bt
from data.fetch import _ohlcv_cache_path
from research_utils import DEV_END, TIER3_DIR, TRIAL_LOG, log_trials, total_trials

FAMILY = "upside_display"
STEP = "email"
RUN_KW = dict(period="13y", start_date="2016-01-04", end_date=str(DEV_END.date()), max_tickers=1000,
              rebalance_every=1, initial_capital=5000.0, save_run=False, pit_universe=True)
WARMUP = 100        # 학습 거래가 이만큼 쌓인 뒤부터 평가
N_BOOT = 5000
COVER_BAND = (0.45, 0.55)
LIVE_TRADES = ROOT / "data" / "paper_trading" / "trades.json"
_ORIGINAL_FETCH = bt.fetch_ohlcv
_LOADED: dict[str, pd.DataFrame] = {}

# ── 메일에 쓰이던 버킷 규칙 (2026-10-11 삭제한 paper_trading/upside_model.py 와 같게) ──
CCS_EDGES = [0.40, 0.45, 0.50, 0.55]   # 옛 config.UPSIDE_MODEL_CCS_BUCKETS
MIN_SAMPLES = 10                       # 옛 config.UPSIDE_MODEL_MIN_SAMPLES
_STAR_PATTERN = re.compile(r"\(([\d.]+)\)")


def _ccs_bucket(ccs: float) -> str:
    for i, edge in enumerate(CCS_EDGES):
        if ccs < edge:
            return f"<{edge:.2f}" if i == 0 else f"{CCS_EDGES[i - 1]:.2f}-{edge:.2f}"
    return f"{CCS_EDGES[-1]:.2f}+"


def _star_bucket(star_rating: str) -> str:
    stars = star_rating.count("★")
    if stars < 5:
        return f"{stars}★이하"
    m = _STAR_PATTERN.search(star_rating)
    if not m:
        return "5★(기본)"
    score = float(m.group(1))
    return "5★(9+)" if score >= 9.0 else "5★(7-9)" if score >= 7.0 else "5★(<7)"


def _strategy_key(strategy: str) -> str:
    return re.sub(r"[^\w가-힣]", "", strategy).strip()


def _bucket_key(strategy: str, ccs: float, star_rating: str) -> str:
    return f"{_strategy_key(strategy)}|{_ccs_bucket(ccs)}|{_star_bucket(star_rating)}"


def _build_bucket_stats(records: list[tuple[str, float, float]]) -> dict[str, dict]:
    """버킷별 P25~P90 + 폴백 (표본 MIN_SAMPLES 미만 버킷 → 전략 → 전체). records = (bucket_key, mfe, days)."""
    by_bucket: dict[str, list[float]] = {}
    by_strategy: dict[str, list[float]] = {}
    everything: list[float] = []
    for key, mfe, _ in records:
        by_bucket.setdefault(key, []).append(mfe)
        by_strategy.setdefault(key.split("|", 1)[0], []).append(mfe)
        everything.append(mfe)

    def stats(values: list[float], level: str) -> dict:
        out: dict = {f"p{q}": float(np.percentile(values, q)) for q in (25, 50, 75, 90)}
        out.update(n=len(values), level=level)
        return out

    result: dict[str, dict] = {}
    for key, values in by_bucket.items():
        strat = key.split("|", 1)[0]
        if len(values) >= MIN_SAMPLES:
            result[key] = stats(values, "bucket")
        elif len(by_strategy[strat]) >= MIN_SAMPLES:
            result[key] = stats(by_strategy[strat], f"strategy:{strat}")
        else:
            result[key] = stats(everything, "global")
    for strat, values in by_strategy.items():
        if len(values) >= MIN_SAMPLES:
            result[f"__strategy__|{strat}"] = stats(values, f"strategy:{strat}")
    if everything:
        result["__global__"] = stats(everything, "global")
    return result


def frozen_fetch(tickers: list[str], period: str = "1y", force_download: bool = False) -> pd.DataFrame:
    """디스크 OHLCV 캐시를 나이와 상관없이 읽는다 (24시간 지난 캐시를 다시 받지 않게 — 재현성)."""
    path = _ohlcv_cache_path(tickers, period)
    if path.exists():
        print(f"[upside] 고정 OHLCV 캐시: {path.name}", flush=True)
        df = pd.read_parquet(path)
    else:
        print(f"[upside] ⚠️ 캐시 없음 → 다운로드: {path.name}", flush=True)
        df = _ORIGINAL_FETCH(tickers, period=period, force_download=force_download)
    _LOADED[path.name] = df
    return df


def _registered(family: str) -> bool:
    if not TRIAL_LOG.exists():
        return False
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig")
    return bool(((log["family"] == family) & (log["판정"] == "등록")).any())


def holding_mfe(trade: dict, high: pd.DataFrame) -> float | None:
    """보유 중 최고 상승률 (진입가 대비). 교체는 전날 종가, 기간종료는 그날 종가에 판다."""
    tkr = trade["ticker"]
    if tkr not in high.columns:
        return None
    entry, exit_ = pd.Timestamp(trade["entry_date"]), pd.Timestamp(trade["exit_date"])
    last = exit_ if trade.get("exit_reason") == "기간종료" else exit_ - timedelta(days=1)
    vals = high.loc[entry:last, tkr].dropna().tolist()
    exit_price = trade.get("exit_price")
    if exit_price is not None and not np.isnan(float(exit_price)):
        vals.append(float(exit_price))
    if not vals:
        return None
    return max(vals) / float(trade["entry_price"]) - 1.0


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def block_boot_ci(diff: np.ndarray, blocks: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """진입월 블록 부트스트랩으로 평균 차이의 95% 구간 (겹치는 보유 기간의 상관을 덜어 낸다)."""
    g = pd.DataFrame({"d": diff, "b": blocks}).groupby("b")["d"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(N_BOOT, len(g)))
    stat = sums[idx].sum(1) / counts[idx].sum(1)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def walk_forward(trades: pd.DataFrame) -> pd.DataFrame:
    """거래마다 진입일 전에 끝난 거래만으로 M1(버킷)·M0(전체) 분위수를 만든다."""
    trades = trades.sort_values(["entry_date", "ticker"]).reset_index(drop=True)
    out = []
    for r in trades.itertuples():
        past = trades[trades["exit_date"] < r.entry_date]
        if len(past) < WARMUP:
            continue
        stats = _build_bucket_stats([(k, m, 0.0) for k, m in zip(past["bucket_key"], past["mfe"])])
        m1 = stats.get(r.bucket_key) or stats.get(f"__strategy__|{r.strategy_key}") or stats["__global__"]
        m0 = stats["__global__"]
        out.append({"ticker": r.ticker, "entry_date": r.entry_date, "exit_date": r.exit_date, "strategy": r.strategy_key,
                    "ccs": r.ccs_score, "bucket_key": r.bucket_key, "level": m1["level"], "n_train": len(past),
                    "mfe": r.mfe, "m1_p50": m1["p50"], "m1_p75": m1["p75"], "m0_p50": m0["p50"], "m0_p75": m0["p75"]})
    return pd.DataFrame(out)


def score(wf: pd.DataFrame) -> dict:
    """도달률·pinball·순위상관. diff>0 이면 버킷 모델(M1)이 더 잘 맞힘."""
    y = wf["mfe"].to_numpy()
    n = len(wf)
    res: dict = {"n": n}
    for tau, col in ((0.5, "p50"), (0.75, "p75")):
        for m in ("m1", "m0"):
            k = int((y >= wf[f"{m}_{col}"].to_numpy()).sum())
            res[f"{m}_{col}_hit"] = k / n
            res[f"{m}_{col}_hit_ci"] = wilson(k, n)
        l1 = pinball(y, wf[f"m1_{col}"].to_numpy(), tau)
        l0 = pinball(y, wf[f"m0_{col}"].to_numpy(), tau)
        diff = l0 - l1
        res[f"{col}_diff"] = float(diff.mean())
        res[f"{col}_diff_ci"] = block_boot_ci(diff, pd.to_datetime(wf["entry_date"]).dt.strftime("%Y-%m").to_numpy())
        res[f"{col}_skill"] = 1.0 - float(l1.mean() / l0.mean())
    rho = spearmanr(wf["m1_p50"], wf["mfe"])
    res["spearman"], res["spearman_p"] = float(rho.statistic), float(rho.pvalue)
    res["distinct_p50"] = int(wf["m1_p50"].round(4).nunique())
    res["level_share"] = wf["level"].str.split(":").str[0].value_counts(normalize=True).to_dict()
    return res


def _match_bar(sub: pd.DataFrame, day: pd.Timestamp, price: float) -> pd.Timestamp:
    """기록된 날짜 앞뒤 일봉 중 종가가 기록 가격과 가장 가까운 날 (10-01 전 기록은 날짜가 실행일(UTC)이라 밀려 있다)."""
    near = sub.loc[day - timedelta(days=6):day, "Close"].dropna()
    if near.empty:
        return day
    gap = (near / price - 1.0).abs()
    return gap.idxmin() if gap.min() < 0.01 else day


def live_check() -> pd.DataFrame:
    """실전 PT-1: 저장된 P50/P75 vs 장중 고가로 다시 잰 MFE (진입 다음 일봉 ~ 청산 일봉)."""
    import yfinance as yf

    # 표본 0 (모델 생성 전·조회 실패) 은 예측이 없던 거래라 뺀다
    trades = [t for t in json.loads(LIVE_TRADES.read_text(encoding="utf-8"))
              if t.get("predicted_upside_p50") is not None and (t.get("upside_sample_size") or 0) > 0]
    tickers = sorted({t["ticker"] for t in trades})
    start = min(pd.Timestamp(t["entry_date"]) for t in trades) - timedelta(days=10)
    end = max(pd.Timestamp(t["exit_date"]) for t in trades) + timedelta(days=3)
    px = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker", threads=True)
    rows = []
    for t in trades:
        tkr, entry_px = t["ticker"], float(t["entry_price"])
        sub = px[tkr].dropna(how="all") if tkr in px.columns.get_level_values(0) else pd.DataFrame()
        mfe = None
        if not sub.empty:
            e_bar = _match_bar(sub, pd.Timestamp(t["entry_date"]), entry_px)
            x_bar = _match_bar(sub, pd.Timestamp(t["exit_date"]), float(t["exit_price"]))
            highs = sub.loc[e_bar + timedelta(days=1):x_bar, "High"].dropna()
            vals = highs.tolist() + [float(t["exit_price"])]
            mfe = max(vals) / entry_px - 1.0
        rows.append({"ticker": tkr, "entry_date": t["entry_date"], "exit_date": t["exit_date"],
                     "p50": t["predicted_upside_p50"], "p75": t["predicted_upside_p75"], "level": t.get("upside_level"),
                     "mfe_close": t.get("actual_max_upside_pct"), "mfe_high": mfe, "exit_reason": t.get("exit_reason")})
    df = pd.DataFrame(rows)
    df["mfe_close"] = pd.to_numeric(df["mfe_close"], errors="coerce")
    return df


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _ci(ci: tuple[float, float], pct: bool = True) -> str:
    lo, hi = ci
    return f"[{lo:.1%}, {hi:.1%}]" if pct else f"[{lo:+.5f}, {hi:+.5f}]"


def main() -> None:
    span = f"{RUN_KW['start_date']} ~ {RUN_KW['end_date']}"
    if not _registered(FAMILY):
        log_trials(STEP, FAMILY, 1, [{"test_id": "(등록)", "판정": "등록",
                   "메모": f"[{span}, 13y PIT PT-1, 워크포워드] U1 메일 예상상승(버킷 P50) vs 모든 종목 같은 숫자. "
                           "통과 = P50 pinball 차이 진입월 블록 부트스트랩 95% 하한 > 0 그리고 P50 도달률 45~55%"}])

    print("===== 1/3 10년 PIT PT-1 백테스트", flush=True)
    bt.fetch_ohlcv = frozen_fetch
    try:
        result = bt.run_paper_trading_backtest(**RUN_KW)
    finally:
        bt.fetch_ohlcv = _ORIGINAL_FETCH
    raw = max(_LOADED.values(), key=lambda d: d.shape[1])
    high = raw.xs("High", level=1, axis=1)

    print("===== 2/3 MFE · 워크포워드", flush=True)
    tr = pd.DataFrame(result["trades"])
    tr["mfe"] = [holding_mfe(t, high) for t in result["trades"]]
    n_all = len(tr)
    tr = tr.dropna(subset=["mfe"])
    tr["bucket_key"] = [_bucket_key(s, float(c), r) for s, c, r in zip(tr["strategy"], tr["ccs_score"], tr["star_rating"])]
    tr["strategy_key"] = tr["strategy"].map(_strategy_key)
    wf = walk_forward(tr)
    res = score(wf)
    wf.to_csv(TIER3_DIR / "upside_validation.csv", index=False)

    lo, _ = res["p50_diff_ci"]
    cover_ok = COVER_BAND[0] <= res["m1_p50_hit"] <= COVER_BAND[1]
    verdict = "통과" if (lo > 0 and cover_ok) else "불통과"

    print("===== 3/3 실전 PT-1 확인", flush=True)
    try:
        live = live_check()
    except Exception as exc:  # noqa: BLE001 — 실전 확인은 참고용이라 보고서만 빠진다
        print(f"[upside] 실전 확인 실패: {exc}")
        live = pd.DataFrame()

    memo = (f"버킷 P50 도달률 {_pct(res['m1_p50_hit'])}, pinball 차이 {res['p50_diff']:+.5f} "
            f"{_ci(res['p50_diff_ci'], pct=False)}, 순위상관 {res['spearman']:+.3f} (n={res['n']})")
    log_trials(STEP, FAMILY, 1, [{"test_id": "U1 메일 예상상승 P50", "판정": verdict, "메모": memo}])

    lines = [f"# 메일 예상상승분포 · Target / Next Target 검증 (PT-1 10년 PIT 워크포워드, {span})", "",
             f"**판정: {verdict}** — " + ("메일에 P50 숫자 하나만 남긴다." if verdict == "통과"
                                          else "버킷 모델이 '모든 종목 같은 숫자'보다 낫지 않다 → 예상상승 표시를 전부 지운다."), "",
             "- 메일의 Target = 버킷(전략 × CCS 구간 × 별점 구간) 과거 거래 MFE 의 P50, Next Target = P75",
             "- 정답 MFE = 보유 중 장중 최고가 / 진입가 − 1 (청산일은 시가 체결이라 그날 고가 대신 청산가)",
             f"- 거래 {n_all}건 중 MFE 계산 {len(tr)}건, 학습 {WARMUP}건이 쌓인 뒤부터 평가 {res['n']}건 "
             f"(진입 {wf['entry_date'].min()} ~ {wf['entry_date'].max()})",
             f"- 사전 등록: family `{FAMILY}` 1건 · 누적 시도 {total_trials()}건", "",
             "## 10년 워크포워드", "",
             "| 항목 | 버킷 모델 (메일) | 모든 종목 같은 숫자 | 기대 |", "|---|---|---|---|",
             f"| Target(P50) 도달률 | {_pct(res['m1_p50_hit'])} {_ci(res['m1_p50_hit_ci'])} | "
             f"{_pct(res['m0_p50_hit'])} {_ci(res['m0_p50_hit_ci'])} | 50% |",
             f"| Next Target(P75) 도달률 | {_pct(res['m1_p75_hit'])} {_ci(res['m1_p75_hit_ci'])} | "
             f"{_pct(res['m0_p75_hit'])} {_ci(res['m0_p75_hit_ci'])} | 25% |",
             f"| P50 pinball 손실 개선 (같은 숫자 대비) | {res['p50_skill']:+.1%} · 차이 {res['p50_diff']:+.5f} "
             f"{_ci(res['p50_diff_ci'], pct=False)} | 기준 | 0 초과 |",
             f"| P75 pinball 손실 개선 | {res['p75_skill']:+.1%} · 차이 {res['p75_diff']:+.5f} "
             f"{_ci(res['p75_diff_ci'], pct=False)} | 기준 | 0 초과 |",
             f"| 예측 P50 과 실제 MFE 순위상관 | {res['spearman']:+.3f} (p={res['spearman_p']:.3f}) | — | 양수 |",
             f"| 서로 다른 P50 값 개수 | {res['distinct_p50']} | 1 | |", "",
             "버킷 단계 비율: " + ", ".join(f"{k} {v:.0%}" for k, v in res["level_share"].items()), "",
             "### 연도별 Target(P50) 도달률", "", "| 연도 | 건수 | 버킷 모델 | 같은 숫자 | MFE 중앙값 |", "|---|---|---|---|---|"]
    wf["year"] = pd.to_datetime(wf["entry_date"]).dt.year
    for year, g in wf.groupby("year"):
        lines.append(f"| {year} | {len(g)} | {_pct((g['mfe'] >= g['m1_p50']).mean())} | "
                     f"{_pct((g['mfe'] >= g['m0_p50']).mean())} | {_pct(g['mfe'].median())} |")

    lines += ["", "## 실전 PT-1 (배포된 2026-04-15 표, 표본 외)", ""]
    if live.empty:
        lines.append("- 실전 확인을 하지 못했다 (가격 다운로드 실패).")
    else:
        ok = live.dropna(subset=["mfe_high"])
        k50, k75 = int((ok["mfe_high"] >= ok["p50"]).sum()), int((ok["mfe_high"] >= ok["p75"]).sum())
        c50, c75 = int((live["mfe_close"] >= live["p50"]).sum()), int((live["mfe_close"] >= live["p75"]).sum())
        rho_live = spearmanr(ok["p50"], ok["mfe_high"]).statistic if len(ok) > 2 else float("nan")
        lines += ["| 항목 | 장중 고가 기준 (다시 잰 값) | 종가 기준 (기록된 값) | 기대 |", "|---|---|---|---|",
                  f"| 건수 | {len(ok)} | {len(live)} | |",
                  f"| Target(P50) 도달 | {k50}/{len(ok)} = {_pct(k50 / max(len(ok), 1))} {_ci(wilson(k50, len(ok)))} | "
                  f"{c50}/{len(live)} = {_pct(c50 / max(len(live), 1))} | 50% |",
                  f"| Next Target(P75) 도달 | {k75}/{len(ok)} = {_pct(k75 / max(len(ok), 1))} {_ci(wilson(k75, len(ok)))} | "
                  f"{c75}/{len(live)} = {_pct(c75 / max(len(live), 1))} | 25% |",
                  f"| 예측 P50 과 실제 순위상관 | {rho_live:+.3f} | | 양수 |",
                  f"| MFE 중앙값 | {_pct(ok['mfe_high'].median())} | {_pct(live['mfe_close'].median())} | |",
                  f"| 예측 P50 값 종류 | {live['p50'].round(4).nunique()} | | |"]
        live.to_csv(TIER3_DIR / "upside_validation_live.csv", index=False)
    lines += ["", "## 참고", "",
              "- 백테스트 체결: 신호 다음 날 시가 진입·청산. 실전: 신호일 종가 진입, 다음 실행 때 종가 기준 청산.",
              "- 지금 CCS 식에서는 후보 대부분이 `0.55+` 구간이라 버킷이 사실상 전략 × 별점으로만 갈린다 (실전에서 쓰일 모습 그대로).",
              "- 생존 편향: yfinance 에 상장폐지 종목 가격이 없어 초기 연도 유니버스가 조금 작다 (HANDOFF 4절)."]
    report = TIER3_DIR / "UPSIDE_VALIDATION.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[upside] 보고서: {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
