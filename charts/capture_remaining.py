"""전체차트 — [SP500 분석]·[NASDAQ/NYSE 분석] 시트의 모든 종목 중, 오늘 '차트분석'에서 캡처하지 않은 종목만 캡처한다.

'차트분석'(웹 '차트추천')은 매수적합도 5.0 이상 + 고정 종목만 캡처한다 (main.py · run_full_scan.py). 나머지는 여기서.
- 대상: data/paper_trading/sp500_ranked.parquet + nasdaq_ranked.parquet 의 종목 (= 두 분석 시트의 종목), 매수적합도 높은 순.
  charts/screenshots_sp500 · charts/screenshots_nasdaq 에 오늘 파일이 있는 종목은 건너뛴다.
- 저장: charts/screenshots_all/<티커>/<티커>_<봉>_<시각>.webp — PNG 를 가로 1920px · WebP 품질 80 으로 줄인다
  (2026-10-16 표본: 799KB → 120KB). 기존 차트분석 PNG 는 시트가 쓰므로 그대로 둔다.
- CI 에서 실전 단계(PT·메일·상태 커밋)가 끝난 뒤에 돌고, 시간 한도(기본 90분)를 넘으면 남은 종목은 건너뛴다.

사용: PYTHONPATH=.:src python -m charts.capture_remaining [--budget-min 90] [--limit N]
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

ALL_DIR = Path("charts/screenshots_all")
SNAPSHOTS = ("data/paper_trading/sp500_ranked.parquet", "data/paper_trading/nasdaq_ranked.parquet")
CAPTURED_DIRS = ("charts/screenshots_sp500", "charts/screenshots_nasdaq")
WEBP_WIDTH, WEBP_QUALITY = 1920, 80


def analysis_tickers(snapshots: tuple[str, ...] = SNAPSHOTS) -> list[str]:
    """두 분석 시트의 종목 — 매수적합도 높은 순 (시간 한도에 걸려도 볼 만한 종목부터)."""
    frames = []
    for p in snapshots:
        try:
            frames.append(pd.read_parquet(p, columns=None))
        except Exception as exc:  # noqa: BLE001
            print(f"[전체차트] {p} 못 읽음: {exc}")
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    score = pd.to_numeric(df.get("매수적합도", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    df = df.assign(_s=score).sort_values("_s", ascending=False)
    return list(dict.fromkeys(str(t).strip() for t in df["티커"] if str(t).strip()))


def captured_today(dirs: tuple[str, ...] = CAPTURED_DIRS) -> set[str]:
    """차트분석 단계가 오늘 캡처한 종목 (폴더 안에 이미지가 있는 것)."""
    out: set[str] = set()
    for d in dirs:
        root = Path(d)
        if root.exists():
            out |= {p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.png"))}
    return out


def to_webp(png: Path) -> Path:
    """PNG → 가로 1920px WebP (품질 80), PNG 는 지운다."""
    from PIL import Image  # noqa: PLC0415

    out = png.with_suffix(".webp")
    with Image.open(png) as im:
        im = im.convert("RGB")
        if im.width > WEBP_WIDTH:
            im = im.resize((WEBP_WIDTH, round(im.height * WEBP_WIDTH / im.width)), Image.LANCZOS)
        im.save(out, "WEBP", quality=WEBP_QUALITY, method=6)
    png.unlink()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="전체차트 — 차트분석에 없는 분석 시트 종목 캡처")
    ap.add_argument("--budget-min", type=float, default=90.0, help="이 시간(분)을 넘으면 남은 종목은 건너뜀")
    ap.add_argument("--limit", type=int, default=0, help="최대 종목 수 (0 = 전부, 시험용)")
    args = ap.parse_args(argv)

    from charts.tradingview_capture import capture_multiple_timeframes  # noqa: PLC0415

    todo = [t for t in analysis_tickers() if t not in captured_today()]
    if args.limit:
        todo = todo[: args.limit]
    if ALL_DIR.exists():
        shutil.rmtree(ALL_DIR)
    ALL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[전체차트] 캡처할 종목 {len(todo)}개 (시간 한도 {args.budget_min:.0f}분)")
    start, done, failed = time.monotonic(), 0, []
    for i, ticker in enumerate(todo, 1):
        if time.monotonic() - start > args.budget_min * 60:
            print(f"[전체차트] 시간 한도 — {len(todo) - i + 1}종목 건너뜀")
            break
        try:
            paths = capture_multiple_timeframes(ticker, output_dir=str(ALL_DIR), headless=True)
            for p in paths.values():
                to_webp(Path(p))
            done += bool(paths)
            if not paths:
                failed.append(ticker)
        except Exception as exc:  # noqa: BLE001 — 한 종목 실패로 멈추지 않는다
            failed.append(ticker)
            print(f"[전체차트] {ticker} 실패: {exc}")
        if i % 10 == 0:
            print(f"[전체차트] {i}/{len(todo)} ({(time.monotonic() - start) / 60:.0f}분)", flush=True)
    size = sum(f.stat().st_size for f in ALL_DIR.rglob("*.webp")) / 1e6
    print(f"[전체차트] 완료 {done}종목 · 실패 {len(failed)} {failed[:10]} · {size:.0f}MB · {(time.monotonic() - start) / 60:.0f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
