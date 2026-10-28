#!/usr/bin/env python3
"""골든크로스 통일 'C' 확인: 밤 실행과 같은 차트 가격으로 계산한 4선 · 캔들 · 정배열 상태가 TradingView 캡처와 같은지.

C (2026-10-27 사용자 결정, docs/goldencross.md): 모든 패턴을 차트와 같은 바탕으로 — 분할만 반영한 가격 (ADJ 꺼짐) ·
주·월봉 끝난 봉 · 선은 Pine 계산 · 정배열은 차트 4선 EMA10/20/100/200 · 주·월봉은 상장 이후 전체 기록.
2단계(chart_ma_tv_check.py)는 EMA10 · 20 만 봤다. 여기서는 범례의 네 선 값과 맨 위 시가 · 고가 · 저가 · 종가까지 본다.

- 캡처 · OCR: tv_legend_check (macOS Vision) — 스크린샷 브랜치를 지금 커밋으로 고정, 종목 · 봉마다 가장 최근 한 장
- 우리 계산: data.fetch.fetch_chart_prices (밤 실행과 같은 길 — period="max", auto_adjust=False) 를 캡처 전날까지 자르고,
  screener.chart_ma.chart_view 의 봉으로 Pine EMA. 범례 · 맨 위 값은 진행 중인 마지막 봉이라 그 봉끼리 비교한다
- 정배열 상태: 같은 종가로 TV 네 선 vs 우리 네 선에 uptrend_stage 규칙 (완전 · 눌림목 · 형성중 · 없음)
- 차트반전: 차트에 그려지는 선이 아니라 입력(EMA20 · EMA200 · 캔들)이 맞는지로 본다 — 위 값 대조에 들어 있다

통과 기준 (결과 보기 전에 정함, 2026-10-27 — 10-16 검사에서 월봉 EMA100/200 은 기록 시작 차이로 5~10% 어긋났다):
- S&P 500 구성 종목 (심볼 버그 제외), 값은 ±1센트 또는 ±0.05% 안이면 일치
- 일봉 EMA10/20/100/200 · 주봉 EMA10/20/100 · 월봉 EMA10/20 값 99% 이상 일치, 주봉 EMA200 98% 이상
- 일 · 주 · 월봉 마지막 봉 시가 · 고가 · 저가 · 종가 99% 이상 일치 (OCR 로 읽힌 것)
- 교차 상태 (EMA10 이 EMA20 위/아래) 일 · 주 · 월 99% 이상 같음
- 정배열 상태 주봉 99% · 월봉 97% 이상 같음
- 월봉 EMA100 · EMA200 값, S&P 500 밖 종목은 보고만 한다
- 하나라도 못 미치면 종료 코드 2 — 백테스트로 가지 않고 원인부터 본다
- 상태(교차 · 정배열)는 TV 범례가 보여 주는 자릿수(대부분 소수 둘째 자리)로 우리 값 · 종가도 반올림해 같은 규칙으로 본다.
  12종목 시험(2026-10-27)에서 ADI 주봉 종가 391.07 · 우리 EMA10 391.0707 (TV 391.07) 처럼 0.001 차이로 단계가 갈린 것
  하나를 보고 고쳤다 (기준 숫자는 그대로)

사용법: PYTHONPATH=.:src .venv/bin/python scripts/chart_tv_check_c.py [--limit N]
산출: docs/quant_improvement/tier3/CHART_TV_CHECK_C.md · chart_tv_check_c.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import chart_ma_tv_check as cm  # noqa: E402
import tv_legend_check as tlc  # noqa: E402
from data.fetch import fetch_chart_prices  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from screener.chart_ma import LINES, chart_view  # noqa: E402
from screener.ma_cross import pine_ema  # noqa: E402

TIER3 = ROOT / "docs" / "quant_improvement" / "tier3"
OHLC_RE = re.compile(r"[O0o]\s?(\d[\d,]*\.?\d*)\s*H\s?(\d[\d,]*\.?\d*)\s*L\s?(\d[\d,]*\.?\d*)\s*C\s?(\d[\d,]*\.?\d*)")
GATES = {("D", 10): 0.99, ("D", 20): 0.99, ("D", 100): 0.99, ("D", 200): 0.99,
         ("W", 10): 0.99, ("W", 20): 0.99, ("W", 100): 0.99, ("W", 200): 0.98,
         ("M", 10): 0.99, ("M", 20): 0.99}
OHLC_GATE, CROSS_GATE, STAGE_GATE = 0.99, 0.99, {"W": 0.99, "M": 0.97}
STAGE_KO = {"full": "완전", "pullback": "눌림목", "forming": "형성중", "": "—"}


def parse_ohlc(obs: list[dict]) -> tuple[float, float, float, float] | None:
    """맨 위 줄의 'O… H… L… C…' (OCR 이 O 를 0 으로 읽기도 한다)."""
    for o in sorted(obs, key=lambda o: -o["y"])[:6]:
        m = OHLC_RE.search(o["t"].replace(" ", ""))
        if m:
            return tuple(float(x.replace(",", "")) for x in m.groups())  # type: ignore[return-value]
    return None


def stage(close: float, e: dict[int, float]) -> str:
    """chart_ma.uptrend_stage 와 같은 규칙을 값으로 (없는 선은 빼고 있는 선으로만)."""
    e10, e20 = e[10], e[20]
    if not (np.isfinite(e10) and np.isfinite(e20)):
        return ""
    lower = [e[n] for n in (100, 200) if np.isfinite(e[n])]
    chain = [e10, e20, *lower]
    stacked = bool(lower) and all(a > b for a, b in zip(chain, chain[1:]))
    if stacked and close > e10:
        return "full"
    if stacked and e10 >= close > e20:
        return "pullback"
    if not stacked and e10 > e20 and close > e10:
        return "forming"
    return ""


def compare(images: pd.DataFrame, ocr: dict[str, list[dict]], charts: dict) -> pd.DataFrame:
    rows = []
    for r in images.itertuples():
        base = {"ticker": r.ticker, "tf": r.tf, "captured": r.captured, "branch": r.branch_name}
        obs = ocr.get(f"{r.ticker}_{r.tf}.png", [])
        vals = tlc.parse_legend(obs)
        if vals is None:
            rows.append({**base, "status": "OCR 실패"})
            continue
        if not (np.isclose(vals[0], vals[2], equal_nan=True) and np.isclose(vals[3], vals[5], equal_nan=True)):
            rows.append({**base, "status": "OCR 자기검사 실패"})
            continue
        cp = charts.get(r.ticker)
        if cp is None:
            rows.append({**base, "status": "가격 없음"})
            continue
        as_of = r.captured.tz_convert("America/New_York").normalize().tz_localize(None) - pd.Timedelta(days=1)
        as_of = cp.daily.index[cp.daily.index <= as_of].max() if (cp.daily.index <= as_of).any() else None
        if as_of is None or pd.isna(as_of):
            rows.append({**base, "status": "가격 없음"})
            continue
        daily, weekly, monthly = chart_view(cp, as_of)
        bars = {"D": daily, "W": weekly, "M": monthly}[r.tf]
        close = bars["Close"]
        ours = {n: float(pine_ema(close, n).iloc[-1]) for n in LINES}
        tv = {10: float(vals[1]), 20: float(vals[0]), 100: float(vals[4]), 200: float(vals[3])}
        row = {**base, "status": "비교", "bar": bars.index[-1].date(), "running": not bool(bars["complete"].iloc[-1])
               if "complete" in bars.columns else False}
        for n in LINES:
            row[f"tv{n}"], row[f"ours{n}"] = tv[n], round(ours[n], 4)
            row[f"ok{n}"] = cm._ok(ours[n], tv[n])
            row[f"rel{n}"] = abs(ours[n] - tv[n]) / abs(tv[n]) if np.isfinite(ours[n] - tv[n]) and tv[n] else np.nan
        last = bars.iloc[-1]
        ohlc = parse_ohlc(obs)
        if ohlc is not None:
            for k, (col, v) in enumerate(zip(("Open", "High", "Low", "Close"), ohlc)):
                row[f"tv_{col}"], row[f"ours_{col}"] = v, round(float(last[col]), 4)
            row["ohlc_ok"] = all(cm._ok(float(last[c]), v) for c, v in zip(("Open", "High", "Low", "Close"), ohlc))
        # 상태는 TV 범례 자릿수로 맞춰 본다 (범례 값이 모두 소수 둘째 자리면 2, 아니면 4)
        dec = 2 if all(round(v, 2) == v for v in tv.values() if np.isfinite(v)) else 4
        c0 = round(float(close.iloc[-1]), dec)
        ours_r = {n: round(v, dec) if np.isfinite(v) else v for n, v in ours.items()}
        row["cross_tv"], row["cross_ours"] = cm._state(tv[10], tv[20]), cm._state(ours_r[10], ours_r[20])
        row["cross_ok"] = row["cross_tv"] == row["cross_ours"]
        row["stage_tv"], row["stage_ours"] = stage(c0, tv), stage(c0, ours_r)
        row["stage_ok"] = row["stage_tv"] == row["stage_ours"]
        rows.append(row)
    return pd.DataFrame(rows)


def _rate(s: pd.Series) -> str:
    s = s.dropna().astype(bool)
    return f"{int(s.sum())}/{len(s)} ({s.mean():.2%})" if len(s) else "—"


def _mean(s: pd.Series) -> float:
    s = s.dropna().astype(bool)
    return float(s.mean()) if len(s) else float("nan")


def write_report(df: pd.DataFrame, n_images: int, pinned: dict[str, str]) -> tuple[bool, list[str]]:
    cmp_ = df[df["status"] == "비교"].copy()
    mem = Membership(load_membership(ROOT / "data" / "universe" / "sp500_membership.csv"))
    sp = mem.on(str(pd.Timestamp(cmp_["bar"].max()).date())) if len(cmp_) else frozenset()
    cmp_["sp500"] = cmp_["ticker"].isin(sp) & ~cmp_["ticker"].isin(tlc.SYMBOL_BUGS)
    spx = cmp_[cmp_["sp500"]]
    checks: list[tuple[str, float, float]] = []
    for (tf, n), gate in GATES.items():
        checks.append((f"{tf} EMA{n} 값", _mean(spx.loc[spx["tf"] == tf, f"ok{n}"]), gate))
    for tf in ("D", "W", "M"):
        g = spx[spx["tf"] == tf]
        checks.append((f"{tf} 시가·고가·저가·종가", _mean(g["ohlc_ok"]) if "ohlc_ok" in g else float("nan"), OHLC_GATE))
        checks.append((f"{tf} 교차 상태", _mean(g["cross_ok"]), CROSS_GATE))
    for tf, gate in STAGE_GATE.items():
        checks.append((f"{tf} 정배열 상태", _mean(spx.loc[spx["tf"] == tf, "stage_ok"]), gate))
    passed = all(np.isfinite(v) and v >= gate for _, v, gate in checks)
    status = df["status"].value_counts()
    lines = [
        "# 골든크로스 통일 C — 차트 가격(상장 이후 전체 · 분할만 반영) vs TradingView 캡처 (4선 · 캔들 · 정배열)", "",
        f"- 캡처: {df['captured'].min()} ~ {df['captured'].max()} (UTC) · 이미지 {n_images}장 "
        f"(브랜치 {', '.join(f'{b} {s[:8]}' for s, b in pinned.items())}, 종목·봉마다 가장 최근 한 장)",
        "- 우리 계산: 밤 실행과 같은 길 — fetch_chart_prices (period=\"max\", auto_adjust=False) 를 캡처 전날까지 자르고 "
        "chart_view 의 봉에 Pine EMA. 범례 · 맨 위 값은 진행 중인 마지막 봉이라 그 봉끼리 비교",
        "- 이미지 상태: " + " · ".join(f"{k} {v}" for k, v in status.items()),
        f"- **S&P 500 ({spx['ticker'].nunique()}종목) 판정: {'통과' if passed else '실패 — 백테스트로 가지 않는다'}**", "",
        "## 통과 기준별 (S&P 500, 결과 전에 정함)", "", "| 항목 | 결과 | 기준 | |", "|---|---|---|---|",
    ]
    for name, v, gate in checks:
        lines.append(f"| {name} | {v:.2%} | {gate:.0%} | {'✅' if np.isfinite(v) and v >= gate else '❌'} |")
    lines += ["", "## 봉 · 선별 (S&P 500)", "", "| 봉 | 비교 | EMA10 | EMA20 | EMA100 | EMA200 | 캔들 | 교차 상태 | 정배열 상태 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for tf in ("D", "W", "M"):
        g = spx[spx["tf"] == tf]
        if len(g):
            lines.append(f"| {tf} | {len(g)} | " + " | ".join(_rate(g[f"ok{n}"]) for n in LINES) +
                         f" | {_rate(g['ohlc_ok']) if 'ohlc_ok' in g else '—'} | {_rate(g['cross_ok'])} | {_rate(g['stage_ok'])} |")
    oth = cmp_[~cmp_["sp500"]]
    lines += ["", f"## S&P 500 밖 ({oth['ticker'].nunique()}종목, 보고만)", "",
              "| 봉 | 비교 | EMA10 | EMA20 | EMA100 | EMA200 | 캔들 | 교차 상태 | 정배열 상태 |", "|---|---|---|---|---|---|---|---|---|"]
    for tf in ("D", "W", "M"):
        g = oth[oth["tf"] == tf]
        if len(g):
            lines.append(f"| {tf} | {len(g)} | " + " | ".join(_rate(g[f"ok{n}"]) for n in LINES) +
                         f" | {_rate(g['ohlc_ok']) if 'ohlc_ok' in g else '—'} | {_rate(g['cross_ok'])} | {_rate(g['stage_ok'])} |")
    bad = cmp_[~cmp_["stage_ok"].astype(bool)]
    lines += ["", f"## 정배열 상태가 다른 것 ({len(bad)}개)", ""]
    if len(bad):
        lines += ["| 종목 | 봉 | S&P | TV | 우리 | TV 10 · 20 · 100 · 200 | 우리 10 · 20 · 100 · 200 |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r.ticker} | {r.tf} | {'○' if r.sp500 else ''} | {STAGE_KO[r.stage_tv]} | {STAGE_KO[r.stage_ours]} | "
                  f"{r.tv10} · {r.tv20} · {r.tv100} · {r.tv200} | {r.ours10:.2f} · {r.ours20:.2f} · {r.ours100:.2f} · {r.ours200:.2f} |"
                  for r in bad.head(60).itertuples()]
    vbad = spx[~spx[[f"ok{n}" for n in LINES]].astype(bool).all(axis=1)]
    lines += ["", f"## S&P 500 값이 하나라도 안 맞은 것 ({len(vbad)}개 — 월봉 EMA100/200 포함)", ""]
    if len(vbad):
        lines += ["| 종목 | 봉 | 안 맞은 선 (TV → 우리) |", "|---|---|---|"]
        for r in vbad.head(80).itertuples():
            miss = [f"EMA{n} {getattr(r, f'tv{n}')} → {getattr(r, f'ours{n}'):.2f}" for n in LINES if not getattr(r, f"ok{n}")]
            lines.append(f"| {r.ticker} | {r.tf} | {' · '.join(miss)} |")
    if "ohlc_ok" in spx:
        obad = spx[spx["ohlc_ok"].notna() & ~spx["ohlc_ok"].fillna(True).astype(bool)]
        lines += ["", f"## S&P 500 캔들이 안 맞은 것 ({len(obad)}개)", ""]
        if len(obad):
            lines += ["| 종목 | 봉 | TV O · H · L · C | 우리 O · H · L · C |", "|---|---|---|---|"]
            lines += [f"| {r.ticker} | {r.tf} | {r.tv_Open} · {r.tv_High} · {r.tv_Low} · {r.tv_Close} | "
                      f"{r.ours_Open:.2f} · {r.ours_High:.2f} · {r.ours_Low:.2f} · {r.ours_Close:.2f} |" for r in obad.head(40).itertuples()]
    (TIER3 / "CHART_TV_CHECK_C.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="종목 수 제한 (시험용)")
    args = ap.parse_args()

    print("===== C 확인 1/3 캡처 목록 · OCR", flush=True)
    pinned = cm.pin_branches()
    tlc.BRANCHES = tuple(pinned)
    images = tlc.list_images()
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

    print("===== C 확인 2/3 차트 가격 받기 (밤 실행과 같은 길: 상장 이후 전체 · 분할만 반영)", flush=True)
    charts = fetch_chart_prices(sorted(images["ticker"].unique()), daily_since=pd.Timestamp("2015-01-01"))

    print("===== C 확인 3/3 비교 · 판정", flush=True)
    df = compare(images, ocr, charts)
    df.to_csv(TIER3 / "chart_tv_check_c.csv", index=False, encoding="utf-8-sig")
    passed, lines = write_report(df, len(images), pinned)
    print("\n".join(lines[: next((i for i, x in enumerate(lines) if x.startswith("## S&P 500 밖")), len(lines))]), flush=True)
    print(f"\nC 확인 판정: {'통과' if passed else '실패'}", flush=True)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
