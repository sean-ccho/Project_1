"""표준 라이브러리만으로 백테스트 거래 로그를 분석한다 (pandas 불가 환경).

사용법:
    python3 scripts/analyze_trades_stdlib.py                       # 기본: 캐시 미사용 run 3개
    python3 scripts/analyze_trades_stdlib.py 2026-04-26_17-03 ...  # run 지정
"""

import csv
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

RUNS_DIR = next(p / "output" / "runs" for p in Path(__file__).resolve().parents if (p / "output" / "runs").is_dir())
RUNS = sys.argv[1:] or ["2026-04-22_00-07", "2026-04-24_02-08", "2026-04-26_17-03"]
COST_ROUNDTRIP = 0.002


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = st.fmean(a), st.fmean(b)
    num_ = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num_ / den if den else 0.0


def spearman(a, b):
    return pearson(rank(a), rank(b))


def summarize(trades):
    r = [t["ret"] for t in trades]
    if not r:
        return "n=0"
    win = sum(1 for x in r if x > 0) / len(r)
    return f"n={len(r):4d}  승률={win:5.1%}  평균={st.fmean(r):+.2%}  비용후={st.fmean(r) - COST_ROUNDTRIP:+.2%}  중앙값={st.median(r):+.2%}"


rows, seen = [], set()
for run in RUNS:
    for row in csv.DictReader(open(RUNS_DIR / run / "backtest_trades_enhanced.csv", encoding="utf-8")):
        key = (row["ticker"], row["entry_date"])
        if key in seen:
            continue
        seen.add(key)
        m = re.search(r"(바닥반등|모멘텀|숏스퀴즈)", row["strategy"])
        row["strat"] = m.group(1) if m else "기타"
        row["ret"] = num(row["return_pct"])
        row["reason"] = re.sub(r"[\(→].*", "", row["exit_reason"]).strip() or "?"
        rows.append(row)
rows = [r for r in rows if r["ret"] is not None]

print(f"# 고유 거래 {len(rows)}건 (run {len(RUNS)}개 합침, ticker+entry_date 중복 제거)")
print(f"# |IC| > {2 / math.sqrt(len(rows)):.3f} 이면 전체 기준 대략 유의 (2σ)\n")

print("## 1. 전략별 성과")
by_strat = defaultdict(list)
for r in rows:
    by_strat[r["strat"]].append(r)
print(f"  {'전체':6s} {summarize(rows)}")
for s, g in sorted(by_strat.items()):
    print(f"  {s:6s} {summarize(g)}")

print("\n## 2. 문제 D 검증: feat_low_prob (저점확률)")
for s, g in [("전체", rows)] + sorted(by_strat.items()):
    vals = [num(r["feat_low_prob"]) for r in g]
    filled = [v for v in vals if v is not None]
    nonzero = [v for v in filled if v != 0]
    print(f"  {s:6s} 값있음={len(filled):4d}/{len(g):4d}  0이아님={len(nonzero):4d}  "
          f"최대={max(filled) if filled else None}")

print("\n## 3. 청산 사유별 성과 + 청산 후 20일 수익 (너무 일찍 팔았나?)")
by_reason = defaultdict(list)
for r in rows:
    by_reason[r["reason"]].append(r)
for reason, g in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
    post = [num(r["post_exit_return_20d"]) for r in g]
    post = [p for p in post if p is not None]
    print(f"  {reason:8s} {summarize(g)}  청산후20일={st.fmean(post) if post else float('nan'):+.2%}")

print("\n## 4. CCS 점수가 수익 순위를 가르는가 (5분위)")
for s, g in [("전체", rows)] + sorted(by_strat.items()):
    pts = [(num(r["ccs_score"]), r["ret"]) for r in g if num(r["ccs_score"]) is not None]
    if len(pts) < 25:
        continue
    pts.sort()
    n = len(pts)
    buckets = [pts[i * n // 5:(i + 1) * n // 5] for i in range(5)]
    desc = "  ".join(f"Q{i + 1}({b[0][0]:.2f}~)={st.fmean(x[1] for x in b):+.2%}" for i, b in enumerate(buckets))
    ic = spearman([p[0] for p in pts], [p[1] for p in pts])
    print(f"  {s:6s} IC={ic:+.3f}  {desc}")

print("\n## 5. 레짐별 성과")
by_regime = defaultdict(list)
for r in rows:
    by_regime[r.get("feat_regime") or "?"].append(r)
for reg, g in sorted(by_regime.items()):
    print(f"  {reg:8s} {summarize(g)}")

print("\n## 6. 연도별 성과 (진입 연도)")
by_year = defaultdict(list)
for r in rows:
    by_year[r["entry_date"][:4]].append(r)
for y, g in sorted(by_year.items()):
    print(f"  {y}  {summarize(g)}")

print("\n## 7. 피처별 예측력 (Spearman IC, 5분위 Q5-Q1)")
feats = [c for c in rows[0] if c.startswith("feat_") and c not in ("feat_sector", "feat_regime", "feat_in_strong_sector")]
for s, g in [("전체", rows)] + sorted(by_strat.items()):
    out = []
    for f in feats:
        pts = [(num(r[f]), r["ret"]) for r in g if num(r[f]) is not None]
        if len(pts) < 30 or len({p[0] for p in pts}) < 5:
            continue
        ic = spearman([p[0] for p in pts], [p[1] for p in pts])
        t = ic * math.sqrt((len(pts) - 2) / max(1e-9, 1 - ic * ic))
        pts.sort()
        n = len(pts)
        q1 = st.fmean(x[1] for x in pts[: n // 5])
        q5 = st.fmean(x[1] for x in pts[4 * n // 5:])
        out.append((abs(ic), f[5:], len(pts), ic, t, q5 - q1))
    out.sort(reverse=True)
    print(f"\n  === {s} ===  (|t| ≥ 2 이면 유의)")
    print(f"  {'피처':28s} {'n':>4s} {'IC':>7s} {'t':>6s} {'Q5-Q1':>8s}")
    for _, name, n, ic, t, spread in out[:12]:
        flag = " *" if abs(t) >= 2 else ""
        print(f"  {name:28s} {n:4d} {ic:+7.3f} {t:+6.2f} {spread:+8.2%}{flag}")
