"""차트 캡처 → 사이트용 WebP (2026-11-03). 밤 실행에서 캡처가 다 끝난 뒤, 웹 데이터(dashboard.json)를 만들기 전에 돈다.

- 차트분석 (charts/screenshots_sp500 · charts/screenshots_nasdaq): PNG 옆에 같은 이름의 .webp 사본을 만든다.
  PNG 는 그대로 둔다 — 구글 시트 [차트분석] 의 =IMAGE() 와 후보 메일이 PNG 주소를 쓴다.
- 전체차트 · 1차 선별 (charts/screenshots_all): 사이트만 쓰므로 PNG 를 WebP 로 바꾸고 PNG 는 지운다.
- 크기는 그대로(가로 약 3,000px) · 품질 80 — LIN 일봉 표본 800KB → 204KB, 2배로 키워 봐도 PNG 와 거의 같다
  (가로 1,920px 로 줄이면 123KB 지만 레티나 화면에서 작은 글씨가 흐려진다).
- 사이트는 dashboard.json 의 주소를 따라간다. 같은 이름이면 이름순으로 .webp 가 .png 뒤라
  web_export.dashboard.list_chart_files 가 WebP 를 고른다.
- Pillow 가 없거나 변환이 실패해도 멈추지 않고 실행 요약에 경고만 띄운다 (그럼 사이트는 PNG 를 그대로 쓴다).
  10-11 matplotlib · fpdf2 를 빼면서 Pillow 도 같이 빠져, 전체차트 WebP 변환이 아무 표시 없이 실패하고 있었다.

사용: PYTHONPATH=.:src python -m charts.webp_convert [--workers N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

COPY_DIRS = ("charts/screenshots_sp500", "charts/screenshots_nasdaq")  # PNG 는 남기고 사본
REPLACE_DIRS = ("charts/screenshots_all",)  # PNG 를 지우고 WebP 만
QUALITY, METHOD = 80, 4  # method 4 는 6 보다 약 25% 빠르고 크기는 2% 크다 (표본 0.3초 · 204KB)


def to_webp(png: Path, *, keep_png: bool) -> int:
    """PNG 하나 → 같은 이름 .webp (크기 그대로). 줄어든 바이트 수를 돌려준다."""
    from PIL import Image  # noqa: PLC0415 — 없으면 main 이 경고만 띄우고 끝낸다

    out = png.with_suffix(".webp")
    with Image.open(png) as im:
        im.convert("RGB").save(out, "WEBP", quality=QUALITY, method=METHOD)
    saved = png.stat().st_size - out.stat().st_size
    if not keep_png:
        png.unlink()
    return saved


def _job(task: tuple[str, bool]) -> tuple[str, int | None, str]:
    """작업 하나 (프로세스 풀용) — (경로, 줄어든 바이트 또는 실패면 None, 오류 글)."""
    path, keep = task
    try:
        return path, to_webp(Path(path), keep_png=keep), ""
    except Exception as exc:  # noqa: BLE001 — 한 장 실패로 멈추지 않는다
        return path, None, str(exc)


def stale_copies(dirs: tuple[str, ...] = COPY_DIRS) -> list[Path]:
    """PNG 가 없는 옛 .webp 사본 — 폴더가 남아 있는 로컬에서 지난 캡처의 사본이 남지 않게 (CI 는 매번 새 폴더)."""
    return [w for d in dirs if Path(d).exists() for w in sorted(Path(d).glob("*/*.webp"))
            if not w.with_suffix(".png").exists()]


def plan(copy_dirs: tuple[str, ...] = COPY_DIRS, replace_dirs: tuple[str, ...] = REPLACE_DIRS) -> list[tuple[str, bool]]:
    """바꿀 PNG 목록 [(경로, PNG 남김 여부)] — 사본이 이미 있고 PNG 보다 새로우면 건너뛴다."""
    tasks: list[tuple[str, bool]] = []
    for dirs, keep in ((copy_dirs, True), (replace_dirs, False)):
        for d in dirs:
            root = Path(d)
            for png in sorted(root.glob("*/*.png")) if root.exists() else []:
                webp = png.with_suffix(".webp")
                if keep and webp.exists() and webp.stat().st_mtime >= png.stat().st_mtime:
                    continue
                tasks.append((png.as_posix(), keep))
    return tasks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="차트 캡처 → 사이트용 WebP (차트분석은 사본, 전체차트는 바꿈)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="동시에 바꿀 프로세스 수 (1 = 한 프로세스)")
    args = ap.parse_args(argv)

    try:
        import PIL  # noqa: F401, PLC0415
    except ImportError:
        print("::warning title=차트 WebP 변환 못 함::Pillow 가 없어 사이트가 PNG 를 그대로 씁니다 (requirements.txt 확인).")
        return 0

    for old in stale_copies():
        old.unlink()
    tasks = plan()
    if args.workers <= 1:
        results = [_job(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_job, tasks, chunksize=8))

    failed = [(p, err) for p, saved, err in results if saved is None]
    saved_mb = sum(saved for _p, saved, _e in results if saved) / 1e6
    print(f"[WebP] {len(results) - len(failed)}장 변환 · {saved_mb:.0f}MB 줄임 · 실패 {len(failed)} {failed[:3]}")
    if failed:
        print(f"::warning title=차트 WebP 일부 실패::{len(failed)}장은 사이트가 PNG 를 그대로 씁니다. 예: {failed[0][1][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
