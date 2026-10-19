#!/usr/bin/env python3
"""TradingView 대조: 매일 밤 캡처한 차트의 SeanEMA 범례 숫자 vs ma_cross 로 계산한 EMA 10/20/100/200.

TradingView 로그인 없이 한다. 스크린샷(screenshots-sp500 · screenshots-nasdaq 브랜치의 일/주/월 PNG)을 꺼내
macOS Vision OCR(scripts/ocr_vision.swift)로 범례 줄 'SeanEMA 20 10 20 200 100 200 v1 … v6' 을 읽는다.
범례 순서 = EMA20 · EMA10 · EMA20 · EMA200 · EMA100 · EMA200 (2026-10-16 HL·T·UNH 로 확인).
OCR 자기검사: v1 = v3 (EMA20), v4 = v6 (EMA200) 가 아니면 그 이미지는 버린다.

가격 = yfinance auto_adjust=False(분할만 반영) period="max", 캡처한 날(미 동부) 전날까지. 주·월봉은 캡처 때처럼 진행 중 봉 그대로.
통과 기준: 일·주 EMA10/20/100/200 과 월 EMA10/20/100 이 99% 이상 ±1센트 또는 ±0.05% 안. 월 EMA200 은 따로 보고.

사용법: PYTHONPATH=.:src .venv/bin/python scripts/tv_legend_check.py [--limit N]
산출: docs/quant_improvement/tier3/MA_CROSS_TV_CHECK.md · ma_cross_tv_check.csv
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading.universe import Membership, load_membership  # noqa: E402
from screener.ma_cross import cross_frame, moving_average, to_bars  # noqa: E402

TIER3 = ROOT / "docs" / "quant_improvement" / "tier3"
BRANCHES = ("origin/screenshots-sp500", "origin/screenshots-nasdaq")
TF_FILE = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
LEGEND_ORDER = (20, 10, 20, 200, 100, 200)
ABS_TOL, REL_TOL = 0.011, 0.0005
PASS_RATE = 0.99
# 연구(ma_cross_research)가 쓰는 선 중 범례에 있는 것: 일 10·20·200, 주 10·20, 월 10·20
RESEARCH_LINES = {("D", 10), ("D", 20), ("D", 200), ("W", 10), ("W", 20), ("M", 10), ("M", 20)}
SYMBOL_BUGS = {"BRK-B": "TradingView 가 'BRK-B' 를 런던(LSE) 종목으로 연다 — 캡처 심볼 버그 (NYSE:BRK.B 여야 함)"}
NAME_RE = re.compile(r"/([^/]+)/\1_(Daily|Weekly|Monthly)_(\d{8})_(\d{6})\.png$")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
NA_TOKENS = {"∅", "Ø", "ø", "⌀", "⊘"}


def list_images() -> pd.DataFrame:
    """브랜치별 PNG 목록 → 종목·봉마다 가장 최근 캡처 한 장."""
    rows = []
    for br in BRANCHES:
        names = subprocess.run(["git", "ls-tree", "-r", "--name-only", br], cwd=ROOT, check=True,
                               capture_output=True, text=True).stdout.split()
        for n in names:
            m = NAME_RE.search("/" + n)
            if m:
                ts = pd.Timestamp(f"{m.group(3)} {m.group(4)}", tz="UTC")
                rows.append({"branch": br, "path": n, "ticker": m.group(1), "tf": TF_FILE[m.group(2)], "captured": ts})
    df = pd.DataFrame(rows)
    return df.sort_values("captured").groupby(["ticker", "tf"], as_index=False).last()


def extract(images: pd.DataFrame, out_dir: Path) -> list[Path]:
    """git show 로 PNG 를 꺼낸다."""
    paths = []
    for r in images.itertuples():
        dest = out_dir / f"{r.ticker}_{r.tf}.png"
        with dest.open("wb") as fh:
            subprocess.run(["git", "show", f"{r.branch}:{r.path}"], cwd=ROOT, check=True, stdout=fh)
        paths.append(dest)
    return paths


def run_ocr(paths: list[Path], work: Path) -> dict[str, list[dict]]:
    """swift 스크립트를 한 번 컴파일하고 모든 이미지에 돌린다."""
    exe = work / "ocr_vision"
    subprocess.run(["swiftc", "-O", str(ROOT / "scripts" / "ocr_vision.swift"), "-o", str(exe)], check=True)
    res = subprocess.run([str(exe)], input="\n".join(str(p) for p in paths) + "\n", capture_output=True,
                         text=True, check=True)
    out = {}
    for line in res.stdout.splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            out[Path(d["path"]).name] = d["obs"]
    return out


def parse_legend(obs: list[dict]) -> list[float] | None:
    """'SeanEMA' 줄의 값 6개. 줄이 여러 조각이면 같은 높이의 조각을 왼쪽부터 잇는다."""
    head = next((o for o in obs if "SeanEMA" in o["t"]), None)
    if head is None:
        return None
    tol = max(head["h"], 0.004) * 0.6
    parts = sorted((o for o in obs if abs(o["y"] - head["y"]) <= tol), key=lambda o: o["x"])
    text = " ".join(o["t"] for o in parts)
    tokens = text[text.index("SeanEMA") + len("SeanEMA"):].split()
    vals: list[float] = []
    for tok in tokens:
        if tok in NA_TOKENS:
            vals.append(float("nan"))  # TV '∅' = 아직 값 없음 (봉 수 < 기간)
        elif NUM_RE.fullmatch(tok):
            vals.append(float(tok.replace(",", "")))
    if len(vals) < 12 or tuple(vals[:6]) != LEGEND_ORDER:
        return None
    return vals[6:12]


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """비조정 일봉 종가, 상장 첫날부터."""
    parts = []
    for i in range(0, len(tickers), 100):
        d = yf.download(tickers[i:i + 100], period="max", interval="1d", auto_adjust=False,
                        group_by="column", progress=False, threads=True)
        parts.append(d["Close"])
    return pd.concat(parts, axis=1)


def recent_crosses(close: pd.Series, tf: str, as_of: pd.Timestamp) -> list[dict]:
    """EMA10/20 의 최근 교차 (사용자가 TV 에서 눈으로 볼 목록)."""
    f = cross_frame(close, 10, 20, "ema", tf, as_of)
    ev = f[(f["gc"] | f["dc"]) & f["complete"]].tail(2)
    return [{"date": d.date(), "kind": "골든" if r.gc else "데드"} for d, r in ev.iterrows()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="종목 수 제한 (시험용)")
    args = ap.parse_args()

    images = list_images()
    if args.limit:
        keep = sorted(images["ticker"].unique())[: args.limit]
        images = images[images["ticker"].isin(keep)]
    print(f"[tv-check] 이미지 {len(images)}장 · 종목 {images['ticker'].nunique()}개", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        paths = extract(images, work)
        ocr = run_ocr(paths, work)
    tickers = sorted(images["ticker"].unique())
    closes = fetch_prices(tickers)

    rows, eyeball = [], []
    for r in images.itertuples():
        vals = parse_legend(ocr.get(f"{r.ticker}_{r.tf}.png", []))
        cap_day = r.captured.tz_convert("America/New_York").normalize().tz_localize(None)
        base = {"ticker": r.ticker, "tf": r.tf, "captured": r.captured}
        if vals is None:
            rows.append({**base, "status": "OCR 실패"})
            continue
        if not (np.isclose(vals[0], vals[2], equal_nan=True) and np.isclose(vals[3], vals[5], equal_nan=True)):
            rows.append({**base, "status": "OCR 자기검사 실패"})
            continue
        if r.ticker not in closes.columns or closes[r.ticker].dropna().empty:
            rows.append({**base, "status": "가격 없음"})
            continue
        close = closes[r.ticker].loc[: cap_day - pd.Timedelta(days=1)].dropna()
        bars = to_bars(close, r.tf)
        tv = {20: vals[0], 10: vals[1], 200: vals[3], 100: vals[4]}
        for n in (10, 20, 100, 200):
            ours = float(moving_average(bars, n, "ema").iloc[-1])
            diff = ours - tv[n]
            if np.isnan(ours) or np.isnan(tv[n]):
                ok, diff = bool(np.isnan(ours) and np.isnan(tv[n])), float("nan")  # 둘 다 '∅' 여야 일치
            else:
                ok = abs(diff) <= ABS_TOL or abs(diff) / abs(tv[n]) <= REL_TOL
            rows.append({**base, "status": "비교", "bar": close.index[-1].date(), "ema": n, "tv": tv[n],
                         "ours": round(ours, 4), "diff": round(diff, 4),
                         "rel": abs(diff) / abs(tv[n]) if np.isfinite(diff) else (0.0 if ok else 1.0), "ok": ok,
                         "first_bar": close.index[0].date()})
        if len(eyeball) < 24 and r.tf in ("D", "W"):
            for c in recent_crosses(close, r.tf, close.index[-1]):
                eyeball.append({"ticker": r.ticker, "tf": r.tf, **c})

    df = pd.DataFrame(rows)
    df.to_csv(TIER3 / "ma_cross_tv_check.csv", index=False, encoding="utf-8-sig")
    write_report(df, eyeball)


def write_report(df: pd.DataFrame, eyeball: list[dict]) -> None:
    """요약 md."""
    cmp_ = df[df["status"] == "비교"].copy()
    cmp_["ok"] = cmp_["ok"].astype(bool)
    gate = cmp_[~((cmp_["tf"] == "M") & (cmp_["ema"] == 200))]
    rate = float(gate["ok"].mean()) if len(gate) else float("nan")
    mem = Membership(load_membership(ROOT / "data" / "universe" / "sp500_membership.csv"))
    sp = mem.on(str(pd.Timestamp(cmp_["bar"].max()).date())) if len(cmp_) else frozenset()
    spx = cmp_[cmp_["ticker"].isin(sp) & ~cmp_["ticker"].isin(SYMBOL_BUGS)]
    lines_used = spx[[(tf, int(n)) in RESEARCH_LINES for tf, n in zip(spx["tf"], spx["ema"])]]
    used_rate = float(lines_used["ok"].mean()) if len(lines_used) else float("nan")
    # 판정 기준 (2026-10-16 사용자 승인): 전체 99% 는 데이터 출처 차이(재상장·옛 기록·심볼 버그)로 못 넘는다 →
    # 연구에 쓰는 선 · S&P 500 구성 종목 · 심볼 버그 제외에서 99% 이상이면 통과
    verdict = "통과" if used_rate >= PASS_RATE else "**실패 — 연구를 멈춘다**"
    status = df.drop_duplicates(["ticker", "tf", "status"])["status"].value_counts()
    lines = [
        "# TradingView 범례 대조 (SeanEMA EMA 10/20/100/200)", "",
        f"- 캡처: {df['captured'].min()} ~ {df['captured'].max()} (UTC) · 이미지 {df[['ticker', 'tf']].drop_duplicates().shape[0]}장",
        "- 우리 계산: 비조정(분할만 반영) 일봉, 상장 첫날부터, 캡처 전날까지 · 주·월봉은 진행 중 봉 그대로 (캡처 때와 같음)",
        f"- 처음 계획한 기준: 일·주 EMA10/20/100/200 · 월 EMA10/20/100 이 {PASS_RATE:.0%} 이상 ±{ABS_TOL:.3f} 또는 ±{REL_TOL:.2%} 안",
        f"- 이미지 상태: " + " · ".join(f"{k} {v}" for k, v in status.items()),
        f"- 전체: {len(gate)}개 값 중 {int(gate['ok'].sum())}개 일치 ({rate:.2%}) — 처음 계획한 99% 에 못 미침 (이유는 아래)",
        f"- **판정 기준 (2026-10-16 사용자 승인): S&P 500 구성 종목 · 연구에 쓰는 선 · 심볼 버그 제외 → "
        f"{int(lines_used['ok'].sum())}/{len(lines_used)} 일치 ({used_rate:.2%}) → {verdict}**", "",
        "## 봉·선별 일치율", "", "| 봉 | EMA | 비교 | 일치 | 일치율 | 최대 차이(%) |", "|---|---|---|---|---|---|",
    ]
    for (tf, n), g in cmp_.groupby(["tf", "ema"]):
        lines.append(f"| {tf} | {n} | {len(g)} | {int(g['ok'].sum())} | {g['ok'].mean():.1%} | {g['rel'].max():.3%} |")
    lines += ["", "## S&P 500 구성 종목만 (캡처 심볼 버그 종목 제외)", "",
              f"- 종목 {spx['ticker'].nunique()}개 · **연구에 쓰는 선(일 10·20·200, 주 10·20, 월 10·20) "
              f"{int(lines_used['ok'].sum())}/{len(lines_used)} 일치 ({lines_used['ok'].mean():.2%})**", "",
              "| 봉 | EMA | 비교 | 일치율 | 안 맞은 종목 |", "|---|---|---|---|---|"]
    for (tf, n), g in spx.groupby(["tf", "ema"]):
        lines.append(f"| {tf} | {int(n)} | {len(g)} | {g['ok'].mean():.1%} | {', '.join(g[~g['ok']]['ticker']) or '—'} |")
    lines += ["", "## 안 맞는 이유", "",
              *[f"- **{t}**: {why}" for t, why in SYMBOL_BUGS.items()],
              "- **기록 시작이 다른 종목** (재상장·합병·다른 거래소에서 옮겨 옴: WOLF · BMNR · GLXY · XRPN · GLND · SAIQ · "
              "UNIT · CAPR · STLA · NBM.V 등): TradingView 는 예전 기록을 이어 붙이고 야후는 새로 시작한다 (또는 반대). "
              "우리 값이 '없음'인데 TV 는 있는 경우가 여기다.",
              "- **아주 오래된 기록 차이** (C · RTX · GOOG · MS · AA · GE · VZ …): 월봉 EMA100/200 처럼 8~17년 전까지 거슬러 "
              "올라가는 선만 다르다. 짧은 선은 모두 맞는다.",
              "- 계산식 차이로 보이는 경우는 없다. EMA 시작값을 Pine 과 같게(처음 n봉 평균) 고친 뒤 상장 몇 년 된 종목도 1센트까지 맞았다."]
    bad = cmp_[~cmp_["ok"]].sort_values("rel", ascending=False)
    lines += ["", f"## 안 맞은 값 ({len(bad)}개)", ""]
    if len(bad):
        lines += ["| 종목 | 봉 | EMA | TV | 우리 | 차이(%) | 첫 봉(야후) |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r.ticker} | {r.tf} | {r.ema} | {r.tv} | {r.ours:.4f} | {r.rel:.3%} | {r.first_bar} |"
                  for r in bad.head(60).itertuples()]
    lines += ["", "## 눈으로 확인할 교차 (EMA10 주황 × EMA20 보라, 끝난 봉)", "",
              "TradingView 기본 차트(ADJ 꺼짐)에서 아래 날짜 봉에 주황선이 보라선을 넘는지 봐 주세요. 주봉 날짜 = 그 주 마지막 거래일.", "",
              "| 종목 | 봉 | 날짜 | 교차 |", "|---|---|---|---|"]
    lines += [f"| {e['ticker']} | {e['tf']} | {e['date']} | {e['kind']} |" for e in eyeball]
    (TIER3 / "MA_CROSS_TV_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16 + cmp_.groupby(['tf', 'ema']).ngroups]))


if __name__ == "__main__":
    main()
