"""백테스트 run 비교 스크립트.

두 run 폴더의 meta.json + trades JSON을 읽어 성과와 파라미터를 나란히 비교한다.

Usage:
    # 가장 최근 2개 run 자동 선택
    python scripts/compare_runs.py

    # 특정 run 지정
    python scripts/compare_runs.py output/runs/2026-04-07_baseline output/runs/2026-04-07_14-30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────

def _sep(char: str = "═", width: int = 70) -> str:
    return char * width


def _load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """meta.json + trades JSON 로드."""
    meta_path = run_dir / "meta.json"
    trades_path = run_dir / "backtest_trades_enhanced.json"

    meta: dict = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    trades: list[dict] = []
    if trades_path.exists():
        with open(trades_path, encoding="utf-8") as f:
            trades = json.load(f)

    return meta, trades


def _find_latest_runs(base: Path, n: int = 2) -> list[Path]:
    """output/runs/ 하위 폴더를 이름 기준 정렬 후 최근 n개 반환."""
    runs = sorted([d for d in base.iterdir() if d.is_dir()])
    return runs[-n:] if len(runs) >= n else runs


def _delta(a: float, b: float) -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:.4f})"


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt(v: float) -> str:
    return f"{v:+.2%}"


# ─────────────────────────────────────────────────────────────────
# 섹션별 출력
# ─────────────────────────────────────────────────────────────────

def _print_params_diff(m1: dict, m2: dict, label1: str, label2: str) -> None:
    """파라미터 diff — 변경된 값 강조."""
    print(f"\n{_sep()}")
    print(f"  PARAMETER DIFF")
    print(_sep())

    p1 = m1.get("params", {})
    p2 = m2.get("params", {})

    def _flat(d: dict, prefix: str = "") -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flat(v, full_key))
            else:
                out[full_key] = str(v)
        return out

    f1, f2 = _flat(p1), _flat(p2)
    all_keys = sorted(set(f1) | set(f2))

    changed = [(k, f1.get(k, "—"), f2.get(k, "—")) for k in all_keys if f1.get(k) != f2.get(k)]
    unchanged = [(k, f1.get(k, "—")) for k in all_keys if f1.get(k) == f2.get(k)]

    if changed:
        print(f"\n  ★ 변경된 파라미터 ({len(changed)}개):")
        print(f"  {'파라미터':<45} {label1:>20} {label2:>20}")
        print(f"  {'-'*45} {'-'*20} {'-'*20}")
        for k, v1, v2 in changed:
            print(f"  {k:<45} {v1:>20} {v2:>20}")
    else:
        print("\n  (파라미터 변경 없음)")

    print(f"\n  변경 없는 파라미터: {len(unchanged)}개 (동일)")


def _print_overall(m1: dict, m2: dict, label1: str, label2: str) -> None:
    """전체 성과 비교."""
    print(f"\n{_sep()}")
    print(f"  OVERALL PERFORMANCE")
    print(_sep())

    p1 = m1.get("performance", {})
    p2 = m2.get("performance", {})

    rows = [
        ("총 거래수",      p1.get("total_trades", 0),  p2.get("total_trades", 0),  False),
        ("승률",           p1.get("win_rate", 0),       p2.get("win_rate", 0),       True),
        ("평균수익",       p1.get("avg_return", 0),     p2.get("avg_return", 0),     True),
        ("평균승",         p1.get("avg_win", 0),        p2.get("avg_win", 0),        True),
        ("평균패",         p1.get("avg_loss", 0),       p2.get("avg_loss", 0),       True),
    ]

    print(f"\n  {'지표':<12} {label1:>18} {label2:>18} {'변화':>12}")
    print(f"  {'-'*12} {'-'*18} {'-'*18} {'-'*12}")
    for name, v1, v2, is_pct in rows:
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if is_pct and name != "총 거래수":
                s1 = _fmt(float(v1))
                s2 = _fmt(float(v2))
                d = float(v2) - float(v1)
                ds = f"{'+'if d>=0 else ''}{d:.2%}"
            else:
                s1 = str(int(v1))
                s2 = str(int(v2))
                d = int(v2) - int(v1)
                ds = f"{'+'if d>=0 else ''}{d}"
            print(f"  {name:<12} {s1:>18} {s2:>18} {ds:>12}")


def _print_strategy(m1: dict, m2: dict, label1: str, label2: str) -> None:
    """전략별 비교."""
    print(f"\n{_sep()}")
    print(f"  STRATEGY BREAKDOWN")
    print(_sep())

    s1 = m1.get("performance", {}).get("by_strategy", {})
    s2 = m2.get("performance", {}).get("by_strategy", {})
    all_strats = sorted(set(s1) | set(s2))

    print(f"\n  {'전략':<12} {'지표':<10} {label1:>14} {label2:>14} {'변화':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*14} {'-'*10}")
    for strat in all_strats:
        g1 = s1.get(strat, {})
        g2 = s2.get(strat, {})
        for metric, key in [("건수", "n"), ("승률", "win_rate"), ("평균수익", "avg_return")]:
            v1 = g1.get(key, 0)
            v2 = g2.get(key, 0)
            if key == "n":
                s_v1, s_v2 = str(v1), str(v2)
                ds = f"{'+' if v2-v1>=0 else ''}{v2-v1}"
            else:
                s_v1 = f"{v1:+.2%}" if isinstance(v1, float) else str(v1)
                s_v2 = f"{v2:+.2%}" if isinstance(v2, float) else str(v2)
                d = float(v2) - float(v1)
                ds = f"{'+'if d>=0 else ''}{d:.2%}"
            strat_label = strat if metric == "건수" else ""
            print(f"  {strat_label:<12} {metric:<10} {s_v1:>14} {s_v2:>14} {ds:>10}")
        print()


def _print_sector(trades1: list[dict], trades2: list[dict], label1: str, label2: str) -> None:
    """섹터별 승률 비교 + Unknown 비율."""
    print(f"\n{_sep()}")
    print(f"  SECTOR ANALYSIS")
    print(_sep())

    def _sector_stats(trades: list[dict]) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for t in trades:
            sector = (t.get("entry_features") or {}).get("sector", "Unknown") or "Unknown"
            if sector not in stats:
                stats[sector] = {"n": 0, "wins": 0}
            stats[sector]["n"] += 1
            stats[sector]["wins"] += int(t.get("return_pct", 0) > 0)
        return stats

    st1, st2 = _sector_stats(trades1), _sector_stats(trades2)
    n1_total = sum(v["n"] for v in st1.values()) or 1
    n2_total = sum(v["n"] for v in st2.values()) or 1

    unk1 = st1.get("Unknown", {}).get("n", 0)
    unk2 = st2.get("Unknown", {}).get("n", 0)
    print(f"\n  Unknown 비율: {label1} {unk1/n1_total:.1%} ({unk1}건)  →  {label2} {unk2/n2_total:.1%} ({unk2}건)")

    all_sectors = sorted(set(st1) | set(st2))
    print(f"\n  {'섹터':<28} {label1+' 승률':>16} {label2+' 승률':>16}")
    print(f"  {'-'*28} {'-'*16} {'-'*16}")
    for sec in all_sectors:
        g1 = st1.get(sec, {"n": 0, "wins": 0})
        g2 = st2.get(sec, {"n": 0, "wins": 0})
        wr1 = f"{g1['wins']/g1['n']:.1%}({g1['n']})" if g1["n"] >= 3 else f"—({g1['n']})"
        wr2 = f"{g2['wins']/g2['n']:.1%}({g2['n']})" if g2["n"] >= 3 else f"—({g2['n']})"
        print(f"  {sec:<28} {wr1:>16} {wr2:>16}")


def _print_exit_summary(trades1: list[dict], trades2: list[dict], label1: str, label2: str) -> None:
    """exit reason 카테고리별 비율 비교."""
    print(f"\n{_sep()}")
    print(f"  EXIT REASON SUMMARY")
    print(_sep())

    def _categorize(reason: str) -> str:
        r = str(reason)
        if "손절" in r:
            return "손절"
        if "트레일" in r or "trail" in r.lower():
            return "트레일링"
        if "목표가" in r:
            return "목표가 익절"
        if "시간익절" in r:
            return "시간 익절"
        if "장기보유" in r:
            return "장기보유"
        return "기타"

    def _exit_stats(trades: list[dict]) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for t in trades:
            cat = _categorize(t.get("exit_reason", ""))
            if cat not in stats:
                stats[cat] = {"n": 0, "wins": 0}
            stats[cat]["n"] += 1
            stats[cat]["wins"] += int(t.get("return_pct", 0) > 0)
        return stats

    ex1, ex2 = _exit_stats(trades1), _exit_stats(trades2)
    n1 = sum(v["n"] for v in ex1.values()) or 1
    n2 = sum(v["n"] for v in ex2.values()) or 1

    cats = ["손절", "트레일링", "목표가 익절", "시간 익절", "장기보유", "기타"]
    print(f"\n  {'청산유형':<14} {label1+' 비율':>16} {label2+' 비율':>16}")
    print(f"  {'-'*14} {'-'*16} {'-'*16}")
    for cat in cats:
        g1 = ex1.get(cat, {"n": 0, "wins": 0})
        g2 = ex2.get(cat, {"n": 0, "wins": 0})
        if g1["n"] == 0 and g2["n"] == 0:
            continue
        r1 = f"{g1['n']/n1:.1%}({g1['n']})"
        r2 = f"{g2['n']/n2:.1%}({g2['n']})"
        print(f"  {cat:<14} {r1:>16} {r2:>16}")


def _print_monthly(trades1: list[dict], trades2: list[dict], label1: str, label2: str) -> None:
    """월별 승률 비교."""
    print(f"\n{_sep()}")
    print(f"  MONTHLY WIN RATE")
    print(_sep())

    def _monthly(trades: list[dict]) -> dict[int, dict]:
        stats: dict[int, dict] = {}
        for t in trades:
            try:
                m = int(str(t.get("entry_date", ""))[:7].split("-")[1])
            except Exception:
                continue
            if m not in stats:
                stats[m] = {"n": 0, "wins": 0}
            stats[m]["n"] += 1
            stats[m]["wins"] += int(t.get("return_pct", 0) > 0)
        return stats

    mo1, mo2 = _monthly(trades1), _monthly(trades2)
    mn = {1:"1월",2:"2월",3:"3월",4:"4월",5:"5월",6:"6월",7:"7월",8:"8월",9:"9월",10:"10월",11:"11월",12:"12월"}

    print(f"\n  {'월':<6} {label1+' 승률':>16} {label2+' 승률':>16}")
    print(f"  {'-'*6} {'-'*16} {'-'*16}")
    for m in range(1, 13):
        g1 = mo1.get(m, {"n": 0, "wins": 0})
        g2 = mo2.get(m, {"n": 0, "wins": 0})
        if g1["n"] == 0 and g2["n"] == 0:
            continue
        wr1 = f"{g1['wins']/g1['n']:.1%}({g1['n']})" if g1["n"] >= 2 else f"—({g1['n']})"
        wr2 = f"{g2['wins']/g2['n']:.1%}({g2['n']})" if g2["n"] >= 2 else f"—({g2['n']})"
        print(f"  {mn[m]:<6} {wr1:>16} {wr2:>16}")


# ─────────────────────────────────────────────────────────────────
# 전체 run 요약 테이블
# ─────────────────────────────────────────────────────────────────

def _print_all_runs_summary(base: Path) -> None:
    """output/runs/ 하위 모든 run의 핵심 지표를 한 줄씩 출력."""
    runs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not runs:
        print("  (run 없음)")
        return

    print(f"\n{'═'*90}")
    print(f"  ALL RUNS SUMMARY  ({len(runs)}개)")
    print(f"{'═'*90}")
    print(f"  {'#':<3} {'run':<28} {'거래수':>6} {'승률':>8} {'평균수익':>10} {'평균승':>9} {'평균패':>9} {'모멘텀WR':>9} {'바닥WR':>8}")
    print(f"  {'-'*3} {'-'*28} {'-'*6} {'-'*8} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")

    for i, run_dir in enumerate(runs, 1):
        meta, _ = _load_run(run_dir)
        perf = meta.get("performance", {})
        label = meta.get("run_label", run_dir.name)
        n = perf.get("total_trades", 0)
        wr = perf.get("win_rate", 0)
        ar = perf.get("avg_return", 0)
        aw = perf.get("avg_win", 0)
        al = perf.get("avg_loss", 0)

        strat = perf.get("by_strategy", {})
        # 전략명에 "모멘텀" 포함된 키 합산
        mom_wr = next(
            (v["win_rate"] for k, v in strat.items() if "모멘텀" in k or "momentum" in k.lower()),
            None,
        )
        bot_wr = next(
            (v["win_rate"] for k, v in strat.items() if "바닥" in k or "반등" in k or "bottom" in k.lower()),
            None,
        )

        mom_s = f"{mom_wr:.1%}" if mom_wr is not None else "—"
        bot_s = f"{bot_wr:.1%}" if bot_wr is not None else "—"

        print(
            f"  {i:<3} {label:<28} {n:>6} {wr:>8.1%} {ar:>+10.2%} "
            f"{aw:>+9.2%} {al:>+9.2%} {mom_s:>9} {bot_s:>8}"
        )

    print(f"\n  ※ 상세 비교: python scripts/compare_runs.py <run1_폴더> <run2_폴더>")
    print(f"     예시:  python scripts/compare_runs.py output/runs/2026-04-07_baseline output/runs/2026-04-07_15-00")
    print(f"{'═'*90}\n")


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="백테스트 run 비교")
    parser.add_argument("run1", nargs="?", default=None, help="첫 번째 run 폴더 경로")
    parser.add_argument("run2", nargs="?", default=None, help="두 번째 run 폴더 경로")
    parser.add_argument("--all", action="store_true", help="전체 run 요약 테이블 출력")
    args = parser.parse_args()

    base = Path("output/runs")

    if not base.exists():
        print("[오류] output/runs/ 폴더가 없습니다. 백테스트를 먼저 실행하세요.")
        return

    # --all 또는 인자 없이 실행 → 전체 요약 + 최근 2개 상세 비교
    if args.all or (not args.run1 and not args.run2):
        _print_all_runs_summary(base)

    if args.all:
        return  # --all 단독 실행 시 여기서 종료

    # 상세 비교 (2개)
    if args.run1 and args.run2:
        dir1, dir2 = Path(args.run1), Path(args.run2)
    else:
        latest = _find_latest_runs(base, n=2)
        if len(latest) < 2:
            print("[오류] output/runs/ 에 run이 2개 이상 필요합니다.")
            print("  먼저 실행: python scripts/run_sp500_backtest.py --period 5y --capital 5000")
            return
        dir1, dir2 = latest[0], latest[1]

    if not dir1.exists() or not dir2.exists():
        print(f"[오류] 폴더를 찾을 수 없습니다: {dir1} / {dir2}")
        return

    m1, t1 = _load_run(dir1)
    m2, t2 = _load_run(dir2)

    label1 = m1.get("run_label", dir1.name)
    label2 = m2.get("run_label", dir2.name)

    print(f"\n{'═'*70}")
    print(f"  BACKTEST COMPARISON REPORT")
    print(f"  BEFORE: {label1}  ({len(t1)} trades)")
    print(f"  AFTER:  {label2}  ({len(t2)} trades)")
    print(f"{'═'*70}")

    _print_params_diff(m1, m2, label1, label2)
    _print_overall(m1, m2, label1, label2)
    _print_strategy(m1, m2, label1, label2)
    _print_sector(t1, t2, label1, label2)
    _print_exit_summary(t1, t2, label1, label2)
    _print_monthly(t1, t2, label1, label2)

    print(f"\n{'═'*70}\n")


if __name__ == "__main__":
    main()
