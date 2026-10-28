#!/usr/bin/env python3
"""골든크로스 통일 2단계: 밤 실행과 같은 길로 계산한 차트 선(EMA10 주황 · EMA20 보라)이 TradingView 캡처 범례 숫자와 같은지.

tv_legend_check.py (2026-10-16, 연구 코드: 비조정 가격 · 상장 첫날부터) 와 같은 캡처 · OCR(macOS Vision) 을 쓰고,
가격만 밤 실행(main.py) 길로 바꾼다 (docs/goldencross.md 2단계):
- 받기: data.fetch 와 같은 yfinance 설정 (auto_adjust=True · actions=True → 배당 조정 가격 + Dividends 칸)
- screener.chart_ma.split_only_close 로 분할만 반영 가격으로 되돌린 뒤 캡처 전날까지 최근 5년만 (main.py period="5y")
- screener.ma_cross.cross_frame(EMA10/20) 의 마지막 봉 값 = 범례 값 (범례는 진행 중인 봉 값을 보여 준다)

통과 기준 (결과 보기 전에 정함, 2026-10-26):
- S&P 500 구성 종목 (심볼 버그 제외): 일·주봉 EMA10 · EMA20 값의 99% 이상이 ±1센트 또는 ±0.05% 안
- 일·주·월봉 상태 (주황이 보라 위인지 아래인지) 99% 이상 같음
- 월봉 EMA10 · EMA20 값은 보고만 한다 (5년치로 시작해서 EMA20 은 0.1% 안팎 어긋날 수 있음 — 상태가 같으면 라벨은 같다)
- 나스닥 · NYSE 종목은 따로 보고만 한다 (재상장 · 기록 시작 차이로 10-16 에도 판정에서 뺐다)

사용법: PYTHONPATH=.:src .venv/bin/python scripts/chart_ma_tv_check.py [--limit N]
산출: docs/quant_improvement/tier3/CHART_MA_TV_CHECK.md · chart_ma_tv_check.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import tv_legend_check as tlc  # noqa: E402
from data.fetch import _download_ohlcv_batch  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from screener.chart_ma import LONG, SHORT, cross_label, split_only_close  # noqa: E402
from screener.ma_cross import cross_frame  # noqa: E402

TIER3 = ROOT / "docs" / "quant_improvement" / "tier3"
BRANCHES = ("origin/screenshots-sp500", "origin/screenshots-nasdaq", "origin/screenshots-all")
HISTORY_YEARS = 5           # main.py fetch_ohlcv(period="5y")
DOWNLOAD_PERIOD = "6y"      # 캡처 전날에서 5년 + 그 뒤 며칠을 덮게
BATCH, BATCH_DELAY = 50, 3.0  # data.fetch 와 같은 배치 크기 · 쉬는 시간
PASS_RATE = 0.99
LABEL_KO = {"golden_cross": "골든", "death_cross": "데드", "golden_cross_imminent": "골든임박",
            "death_cross_imminent": "데드임박", "none": "—", "": "—"}


def pin_branches() -> dict[str, str]:
    """브랜치 → 지금 커밋 (실행 중에 git fetch 가 끝나 브랜치가 옮겨 가도 같은 그림을 읽게)."""
    out = {}
    for br in BRANCHES:
        sha = subprocess.run(["git", "rev-parse", br], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if sha:
            out[sha] = br.split("/")[-1]
    return out


def fetch_frames(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """밤 실행과 같은 설정으로 종목별 일봉 (배당 조정 OHLCV + Dividends)."""
    out: dict[str, pd.DataFrame] = {}
    n = (len(tickers) + BATCH - 1) // BATCH
    for k, i in enumerate(range(0, len(tickers), BATCH), 1):
        data = _download_ohlcv_batch(tickers[i:i + BATCH], DOWNLOAD_PERIOD)
        if data is not None and not data.empty:
            for t in data.columns.get_level_values(0).unique():
                frame = data[t].dropna(how="all")
                if not frame.empty and frame["Close"].notna().any():
                    out[str(t)] = frame
        print(f"  가격 받기 {k}/{n} 배치 · 받은 종목 {len(out)} [{k / n * 100:5.1f}%]", flush=True)
        if k < n:
            time.sleep(BATCH_DELAY)
    return out


def night_close(frame: pd.DataFrame, cap_day: pd.Timestamp) -> pd.Series:
    """캡처한 날 밤 실행이 봤을 종가: 전체 기간으로 분할만 반영 가격으로 되돌린 뒤 캡처 전날까지 · 최근 5년."""
    close = split_only_close(frame).loc[: cap_day - pd.Timedelta(days=1)].dropna()
    if close.empty:
        return close
    return close.loc[close.index[-1] - pd.DateOffset(years=HISTORY_YEARS):]


def _ok(ours: float, tv: float) -> bool:
    if np.isnan(ours) or np.isnan(tv):
        return bool(np.isnan(ours) and np.isnan(tv))   # 둘 다 '∅' (봉 수 < 기간) 이어야 일치
    return abs(ours - tv) <= tlc.ABS_TOL or abs(ours - tv) / abs(tv) <= tlc.REL_TOL


def _state(e10: float, e20: float) -> str:
    if np.isnan(e10) or np.isnan(e20):
        return "없음"
    return "위" if e10 > e20 else ("아래" if e10 < e20 else "같음")


def compare(images: pd.DataFrame, ocr: dict[str, list[dict]], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """이미지마다 범례 EMA10 · EMA20 vs 밤 실행 길 계산 (진행 중인 마지막 봉)."""
    rows = []
    for r in images.itertuples():
        base = {"ticker": r.ticker, "tf": r.tf, "captured": r.captured, "branch": r.branch_name}
        vals = tlc.parse_legend(ocr.get(f"{r.ticker}_{r.tf}.png", []))
        if vals is None:
            rows.append({**base, "status": "OCR 실패"})
            continue
        if not (np.isclose(vals[0], vals[2], equal_nan=True) and np.isclose(vals[3], vals[5], equal_nan=True)):
            rows.append({**base, "status": "OCR 자기검사 실패"})
            continue
        frame = frames.get(r.ticker)
        if frame is None:
            rows.append({**base, "status": "가격 없음"})
            continue
        cap_day = r.captured.tz_convert("America/New_York").normalize().tz_localize(None)
        close = night_close(frame, cap_day)
        if len(close) < 2:
            rows.append({**base, "status": "가격 없음"})
            continue
        fr = cross_frame(close, SHORT, LONG, "ema", r.tf, None)
        o10, o20 = float(fr["ma_s"].iloc[-1]), float(fr["ma_l"].iloc[-1])
        t10, t20 = float(vals[1]), float(vals[0])
        cl = cross_label(close, r.tf)
        rows.append({**base, "status": "비교", "bar": close.index[-1].date(), "first_bar": close.index[0].date(),
                     "tv10": t10, "ours10": round(o10, 4), "ok10": _ok(o10, t10),
                     "rel10": abs(o10 - t10) / abs(t10) if np.isfinite(o10 - t10) else np.nan,
                     "tv20": t20, "ours20": round(o20, 4), "ok20": _ok(o20, t20),
                     "rel20": abs(o20 - t20) / abs(t20) if np.isfinite(o20 - t20) else np.nan,
                     "state_tv": _state(t10, t20), "state_ours": _state(o10, o20),
                     "state_ok": _state(t10, t20) == _state(o10, o20),
                     "label": cl.label, "preview": cl.preview})
    return pd.DataFrame(rows)


def _rate(s: pd.Series) -> str:
    return f"{int(s.sum())}/{len(s)} ({s.mean():.2%})" if len(s) else "—"


def write_report(df: pd.DataFrame, n_images: int, pinned: dict[str, str]) -> tuple[str, list[str]]:
    """요약 md + 판정."""
    cmp_ = df[df["status"] == "비교"].copy()
    for c in ("ok10", "ok20", "state_ok"):
        cmp_[c] = cmp_[c].astype(bool)
    mem = Membership(load_membership(ROOT / "data" / "universe" / "sp500_membership.csv"))
    sp = mem.on(str(pd.Timestamp(cmp_["bar"].max()).date())) if len(cmp_) else frozenset()
    cmp_["sp500"] = cmp_["ticker"].isin(sp) & ~cmp_["ticker"].isin(tlc.SYMBOL_BUGS)
    spx = cmp_[cmp_["sp500"]]
    dw = spx[spx["tf"].isin(["D", "W"])]
    val_ok = pd.concat([dw["ok10"], dw["ok20"]])
    st_ok = spx["state_ok"]
    passed = len(val_ok) > 0 and val_ok.mean() >= PASS_RATE and st_ok.mean() >= PASS_RATE
    verdict = "통과" if passed else "실패 — 3단계로 가지 않고 원인부터 본다"
    status = df["status"].value_counts()

    lines = [
        "# 골든크로스 통일 2단계 — 밤 실행 계산 vs TradingView 범례 (EMA10 주황 · EMA20 보라)", "",
        f"- 캡처: {df['captured'].min()} ~ {df['captured'].max()} (UTC) · 이미지 {n_images}장 "
        f"(브랜치 {', '.join(f'{b} {s[:8]}' for s, b in pinned.items())}, 종목·봉마다 가장 최근 한 장)",
        "- 우리 계산: 밤 실행과 같은 길 — 배당 조정 가격 + 배당 기록으로 분할만 반영 가격 되돌림, 캡처 전날까지 최근 5년, "
        "EMA 는 TradingView 방식(첫 n봉 평균에서 시작), 범례처럼 진행 중인 마지막 봉 값",
        "- 이미지 상태: " + " · ".join(f"{k} {v}" for k, v in status.items()),
        f"- 통과 기준 (결과 전에 정함): S&P 500 · 일·주봉 EMA10/20 값 {PASS_RATE:.0%} 이상 ±{tlc.ABS_TOL:.3f} 또는 "
        f"±{tlc.REL_TOL:.2%} 안 + 일·주·월봉 상태(주황 위/아래) {PASS_RATE:.0%} 이상 같음",
        f"- **S&P 500 ({spx['ticker'].nunique()}종목): 일·주봉 값 {_rate(val_ok)} · 상태 {_rate(st_ok)} → {verdict}**", "",
        "## 봉별 (S&P 500)", "",
        "| 봉 | 비교 | EMA10 값 일치 | EMA20 값 일치 | 상태 같음 | EMA10 최대 차이 | EMA20 최대 차이 |",
        "|---|---|---|---|---|---|---|",
    ]
    for tf in ("D", "W", "M"):
        g = spx[spx["tf"] == tf]
        if len(g):
            lines.append(f"| {tf} | {len(g)} | {_rate(g['ok10'])} | {_rate(g['ok20'])} | {_rate(g['state_ok'])} | "
                         f"{g['rel10'].max():.3%} | {g['rel20'].max():.3%} |")
    oth = cmp_[~cmp_["sp500"]]
    lines += ["", f"## 나스닥 · NYSE 등 S&P 500 밖 ({oth['ticker'].nunique()}종목, 판정에 안 씀)", "",
              "| 봉 | 비교 | EMA10 값 일치 | EMA20 값 일치 | 상태 같음 |", "|---|---|---|---|---|"]
    for tf in ("D", "W", "M"):
        g = oth[oth["tf"] == tf]
        if len(g):
            lines.append(f"| {tf} | {len(g)} | {_rate(g['ok10'])} | {_rate(g['ok20'])} | {_rate(g['state_ok'])} |")
    m = spx[spx["tf"] == "M"]
    if len(m):
        lines += ["", "## 월봉 값 차이 (S&P 500, 보고만)", "",
                  f"- EMA20 차이 가운데 {m['rel20'].median():.3%} · 90% {m['rel20'].quantile(0.9):.3%} · "
                  f"최대 {m['rel20'].max():.3%} — 5년치로 시작한 EMA20 이 상장 첫날부터 계산한 TradingView 와 조금 다르다",
                  f"- 상태(주황 위/아래)는 {_rate(m['state_ok'])} 같다"]
    bad_state = cmp_[~cmp_["state_ok"]]
    lines += ["", f"## 상태가 다른 것 ({len(bad_state)}개)", ""]
    if len(bad_state):
        lines += ["| 종목 | 봉 | S&P | TV 10 · 20 | 우리 10 · 20 | 첫 봉(5년 창) |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.ticker} | {r.tf} | {'○' if r.sp500 else ''} | {r.tv10} · {r.tv20} | {r.ours10:.2f} · {r.ours20:.2f} | "
                  f"{r.first_bar} |" for r in bad_state.head(60).itertuples()]
    bad_val = spx[spx["tf"].isin(["D", "W"]) & ~(spx["ok10"] & spx["ok20"])]
    lines += ["", f"## S&P 500 일·주봉 값이 안 맞은 것 ({len(bad_val)}개)", ""]
    if len(bad_val):
        lines += ["| 종목 | 봉 | TV 10 | 우리 10 | TV 20 | 우리 20 |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.ticker} | {r.tf} | {r.tv10} | {r.ours10:.4f} | {r.tv20} | {r.ours20:.4f} |"
                  for r in bad_val.head(60).itertuples()]
    lab = spx[spx["label"].isin(["golden_cross", "death_cross"])].sort_values(["tf", "ticker"])
    lines += ["", "## 눈으로 확인할 라벨 (끝난 봉 기준 골든 · 데드, S&P 500)", "",
              "TradingView 기본 차트(ADJ 꺼짐)에서 최근 몇 봉 안에 주황선이 보라선을 넘었는지 봐 주세요.", "",
              "| 종목 | 봉 | 라벨 | 예고(진행 중인 봉) |", "|---|---|---|---|"]
    lines += [f"| {r.ticker} | {r.tf} | {LABEL_KO.get(r.label, r.label)} | {LABEL_KO.get(r.preview, r.preview)} |"
              for r in lab.head(30).itertuples()]
    (TIER3 / "CHART_MA_TV_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verdict, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="종목 수 제한 (시험용)")
    args = ap.parse_args()

    print("===== 2단계 1/3 캡처 목록 · OCR", flush=True)
    pinned = pin_branches()
    tlc.BRANCHES = tuple(pinned)
    images = tlc.list_images()                                  # branch 칸 = 커밋 (그림 꺼낼 때 그대로 씀)
    images["branch_name"] = images["branch"].map(pinned)
    if args.limit:
        keep = sorted(images["ticker"].unique())[: args.limit]
        images = images[images["ticker"].isin(keep)]
    print(f"  이미지 {len(images)}장 · 종목 {images['ticker'].nunique()}개 · 캡처 "
          f"{images['captured'].min()} ~ {images['captured'].max()}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        paths = tlc.extract(images, work)
        print(f"  그림 꺼냄 {len(paths)}장 → 글자 읽기 (macOS Vision)", flush=True)
        ocr = tlc.run_ocr(paths, work)
    print(f"  OCR 끝 {len(ocr)}장", flush=True)

    print("===== 2단계 2/3 가격 받기 (밤 실행과 같은 설정)", flush=True)
    frames = fetch_frames(sorted(images["ticker"].unique()))

    print("===== 2단계 3/3 비교 · 판정", flush=True)
    df = compare(images, ocr, frames)
    df.to_csv(TIER3 / "chart_ma_tv_check.csv", index=False, encoding="utf-8-sig")
    verdict, lines = write_report(df, len(images), pinned)
    print("\n".join(lines[: next((i for i, x in enumerate(lines) if x.startswith("## 상태가 다른 것")), len(lines)) + 1]),
          flush=True)
    print(f"\n2단계 판정: {verdict}", flush=True)
    sys.exit(0 if verdict == "통과" else 2)


if __name__ == "__main__":
    main()
