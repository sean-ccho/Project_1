"""자료 점검 — 밤 universe-scan 끝에 사이트가 읽는 자료 파일마다 '몇 개 중 몇 개를 받았나'를 세어 data_status.json 에 (2026-11-13).

왜: 밤 작업이 '성공'이어도 안이 빌 수 있다. 11-12 밤 예측 모델은 야후가 종목 정보를 막아 계산 0/513 이었는데, 작업 결과와
파일 시각은 멀쩡해서 사이트 관리 탭으로는 알 수 없었다. 사이트가 이 숫자로 화면 머리 날짜 옆에 정상 · 일부만 · 실패 표시를 붙인다.
색 판정은 사이트 lib/dataStatus.ts 가 한다 — 여기는 숫자를 세고, 같은 규칙(row_level)으로 실패가 있으면 ntfy 로 한 번 알린다.

보는 파일: 전체분석 미국(sp500 · nasdaq · new · extra), 같은 밤 Canada Scan 의 ca, 예측 모델(value_model.json),
국회의원 거래(congress.json). dashboard.json · etf.json · macro.json 은 Run Screener 가 이 작업보다 늦게(밤 10시쯤)
만들어서 여기서는 못 본다 — 사이트가 그 파일을 읽을 때 같은 칸을 직접 센다.

칸 = {key, label, got, want?, floor?, zero_red?, prev?}
- want 가 있으면: got 0 이면 실패, got/want 가 floor 밑이면 일부만
- want 가 없으면: zero_red 이고 0 이면 실패, 지난 밤 got(prev)의 80% 밑이면 일부만
파일 = {label, date, generated, rows, missing?, stale?} — missing = 파일 없음 · stale = 12시간 넘게 지난 파일
(그 단계가 실패해 전날 파일이 남은 것). 둘 다 실패.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

SCHEMA = 1
OUT_NAME = "data_status.json"
ET = ZoneInfo("America/Toronto")
SITE_URL = "https://seancho.vercel.app/quant/admin"  # ntfy 를 누르면 여는 곳 (관리 · 파이프라인)
NTFY_URL = "https://ntfy.sh"
DROP = 0.8  # 지난 밤의 80% 밑이면 일부만 (사이트 lib/dataStatus.ts 의 DROP 과 같은 값)
STALE_HOURS = 12  # 이보다 오래 전에 만든 파일은 이번 밤 것이 아니다 (전체분석 · 캐나다 · 예측 모델은 같은 저녁에 만든다)
# 종목 칸 하한 — 평소 거의 다 차 있다 (2026-11-12 밤: 가격 · ISB · SQZ 100%, 시가총액 98~100%, 미국 내부자 100%)
FLOOR_FULL = 0.95
FLOOR_PART = 0.9
VM_FLOOR = 0.5  # 예측 모델 계산된 종목 ÷ 대상 (11-11 로컬 433/513 = 84%, 넓힌 대상 첫 밤은 재무를 다 못 받는다)
US_FILES = (("sp500", "전체분석 · S&P 500"), ("nasdaq", "전체분석 · 나스닥·NYSE"), ("new", "전체분석 · 신규상장"),
            ("extra", "전체분석 · 번외"))
LEVELS = ("green", "yellow", "red")


def load_json(path: Path | None) -> Any | None:
    """path 또는 같은 이름 .gz (sp500.json → sp500.json.gz) — 없거나 못 읽으면 None."""
    if path is None:
        return None
    cands = [path] if path.name.endswith(".gz") else [path, path.with_name(path.name + ".gz")]
    for p in cands:
        if not p.exists():
            continue
        try:
            if p.name.endswith(".gz"):
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    return json.load(fh)
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError, EOFError):
            continue
    return None


def row(key: str, label: str, got: int, want: int | None = None, *, floor: float | None = None,
        zero_red: bool = False) -> dict[str, Any]:
    """칸 하나 — 사이트 목록에 보이는 말 그대로의 이름."""
    out: dict[str, Any] = {"key": key, "label": label, "got": int(got)}
    if want is not None:
        out["want"] = int(want)
        out["floor"] = FLOOR_FULL if floor is None else floor
    if zero_red:
        out["zero_red"] = True
    return out


def stock_rows(items: Iterable[Mapping[str, Any]], *, insider: bool) -> list[dict[str, Any]]:
    """종목 목록 → 종목 수 · 가격 · ISB · SQZ · 시가총액 (· 내부자) 칸. None 이면 그 칸을 못 받은 것 ({} 는 계산했지만 없음)."""
    xs = [x for x in items if isinstance(x, Mapping)]
    n = len(xs)

    def cnt(f: Callable[[Mapping[str, Any]], bool]) -> int:
        return sum(1 for x in xs if f(x))

    rows = [
        row("items", "종목", n, zero_red=True),
        row("price", "가격", cnt(lambda x: x.get("price") is not None), n),
        row("isb", "ISB 라벨", cnt(lambda x: x.get("isb") is not None), n),
        row("sqz", "SQZ 모멘텀", cnt(lambda x: x.get("sqzmom") is not None), n),
        row("mcap", "시가총액", cnt(lambda x: (x.get("fundamentals") or {}).get("market_cap") is not None), n,
            floor=FLOOR_PART),
    ]
    if insider:
        rows.append(row("insider", "내부자 거래", cnt(lambda x: x.get("insider") is not None), n, floor=FLOOR_PART))
    return rows


def value_model_rows(vm: Mapping[str, Any]) -> list[dict[str, Any]]:
    """예측 모델 — 계산된 종목 ÷ 대상, 기록 준비 중이면 재무 받은 종목 ÷ 기록 시작 기준."""
    f = vm.get("funnel") or {}
    rows = [row("computed", "계산된 종목", int(f.get("computed") or 0), int(f.get("universe") or 0), floor=VM_FLOOR)]
    warm = (vm.get("tracking") or {}).get("warmup")
    if isinstance(warm, Mapping):
        rows.append(row("warmup", "재무 받은 종목 (기록 준비 중)", int(warm.get("have") or 0), int(warm.get("want") or 0),
                        floor=1.0))
    return rows


def congress_rows(c: Mapping[str, Any]) -> list[dict[str, Any]]:
    """국회의원 거래 — 지난 30일 신고 수 (0 이어도 실패는 아님 · 지난 밤보다 크게 줄면 일부만)."""
    return [row("recent", "최근 30일 신고", len(c.get("recent") or []))]


def entry(label: str, data: Any, rows_fn: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
          date_key: str) -> dict[str, Any]:
    """파일 하나 — 못 읽었으면 missing."""
    if not isinstance(data, Mapping):
        return {"label": label, "missing": True, "rows": []}
    return {"label": label, "date": data.get(date_key), "generated": data.get("generated_at") or data.get("generated"),
            "rows": rows_fn(data)}


def market_entry(label: str, data: Any, *, insider: bool) -> dict[str, Any]:
    """전체분석 목록 파일 (items = {티커: 정보})."""
    return entry(label, data, lambda d: stock_rows((d.get("items") or {}).values(), insider=insider), "bar_date")


def _when(s: Any) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=ET)


def mark_stale(files: Mapping[str, dict[str, Any]], now: datetime) -> None:
    """STALE_HOURS 넘게 지난 파일 — 그 단계 · 작업이 실패해 전날 파일이 남은 것 (날짜 대신 만든 시각으로 봐서 휴장일은 괜찮다)."""
    for f in files.values():
        t = _when(f.get("generated"))
        if t is not None and now - t > timedelta(hours=STALE_HOURS):
            f["stale"] = True


def add_prev(files: Mapping[str, dict[str, Any]], prev: Any) -> None:
    """want 없는 칸(종목 수 등)에 지난 밤 got 을 prev 로 — 20% 넘게 줄었는지 보려고."""
    pfiles = (prev or {}).get("files") if isinstance(prev, Mapping) else None
    for key, f in files.items():
        pf = (pfiles or {}).get(key) if isinstance(pfiles, Mapping) else None
        prow = {r.get("key"): r for r in (pf or {}).get("rows") or [] if isinstance(r, Mapping)}
        for r in f.get("rows", []):
            p = prow.get(r["key"])
            if "want" not in r and p is not None and isinstance(p.get("got"), int):
                r["prev"] = p["got"]


def row_level(r: Mapping[str, Any]) -> str:
    """칸 색 — 사이트 lib/dataStatus.ts rowLevel 과 같은 규칙."""
    got, want = int(r.get("got") or 0), r.get("want")
    if want is not None:
        if want > 0 and got == 0:
            return "red"
        if want > 0 and got / want < float(r.get("floor", FLOOR_FULL)):
            return "yellow"
        return "green"
    if got == 0 and r.get("zero_red"):
        return "red"
    prev = r.get("prev")
    if prev and got < DROP * prev:
        return "yellow"
    return "green"


def file_level(f: Mapping[str, Any]) -> str:
    """파일 색 — 없음 · 오래됨은 실패, 아니면 칸 중 가장 나쁜 것."""
    if f.get("missing") or f.get("stale"):
        return "red"
    return max((row_level(r) for r in f.get("rows", [])), key=LEVELS.index, default="green")


def build_status(universe_dir: Path, ca_path: Path | None, prev: Any = None, now: datetime | None = None) -> dict[str, Any]:
    """universe_dir(이번 밤 전체분석 결과 폴더) + 캐나다 파일 → data_status.json 내용."""
    now = now or datetime.now(ET)
    files: dict[str, dict[str, Any]] = {}
    for key, label in US_FILES:
        data = load_json(universe_dir / f"{key}.json")
        if data is None and key == "extra":
            continue  # 번외 종목이 없는 설정이면 파일도 없다 (사이트도 탭을 숨긴다)
        files[key] = market_entry(label, data, insider=key != "extra")
    files["ca"] = market_entry("전체분석 · 캐나다", load_json(ca_path), insider=False)
    files["value_model"] = entry("예측 모델", load_json(universe_dir / "value_model.json"), value_model_rows, "date")
    files["congress"] = entry("국회의원 거래", load_json(universe_dir / "congress.json"), congress_rows, "as_of")
    mark_stale(files, now)
    add_prev(files, prev)
    date = max((str(files[k].get("date") or "") for k in ("sp500", "nasdaq") if k in files), default="") or None
    return {"schema": SCHEMA, "generated": now.isoformat(timespec="minutes"), "date": date, "files": files}


def _fmt_row(r: Mapping[str, Any]) -> str:
    want = r.get("want")
    return f"{r['label']} {r['got']:,}" + (f"/{want:,}" if want is not None else "")


def alert(status: Mapping[str, Any]) -> tuple[str, str] | None:
    """실패한 파일이 있으면 (제목, 본문) — 없으면 None."""
    lines = []
    for f in (status.get("files") or {}).values():
        if file_level(f) != "red":
            continue
        if f.get("missing"):
            why = "파일 없음"
        elif f.get("stale"):
            why = f"이번 밤 파일 아님 (만든 시각 {str(f.get('generated'))[:16]})"
        else:
            why = ", ".join(_fmt_row(r) for r in f.get("rows", []) if row_level(r) == "red")
        lines.append(f"{f['label']}: {why}")
    if not lines:
        return None
    return f"자료 점검 · 실패 {len(lines)}곳", "\n".join(lines)


def send_ntfy(topic: str, title: str, body: str) -> bool:
    """ntfy.sh 로 폰 푸시 (JSON — 한글 제목). 누르면 관리 · 파이프라인 탭."""
    import requests  # noqa: PLC0415

    try:
        resp = requests.post(NTFY_URL, json={"topic": topic, "title": title, "message": body, "priority": 4,
                                             "tags": ["warning"], "click": SITE_URL}, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 — 알림 실패로 작업을 멈추지 않는다
        print(f"[자료 점검] ntfy 실패: {type(exc).__name__}")
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="사이트 자료 점검 → data_status.json (전체분석 · 캐나다 · 예측 모델 · 국회의원 거래)")
    ap.add_argument("--universe-dir", required=True, type=Path, help="이번 밤 전체분석 결과 폴더 (universe-data 에 올릴 것)")
    ap.add_argument("--ca", type=Path, help="같은 밤 canada-data 의 ca.json.gz")
    ap.add_argument("--prev", type=Path, help="지난 밤 data_status.json (없으면 지난 밤 비교 없이)")
    ap.add_argument("--out", type=Path, help=f"쓸 곳 (기본: universe-dir/{OUT_NAME})")
    ap.add_argument("--no-notify", action="store_true", help="실패가 있어도 ntfy 를 보내지 않는다")
    args = ap.parse_args(argv)
    status = build_status(args.universe_dir, args.ca, load_json(args.prev))
    out = args.out or args.universe_dir / OUT_NAME
    out.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in status["files"].values():
        rows = " · ".join(_fmt_row(r) for r in f.get("rows", []))
        flags = " (파일 없음)" if f.get("missing") else " (이번 밤 파일 아님)" if f.get("stale") else ""
        print(f"[자료 점검] {file_level(f):6} {f['label']}{flags} — {rows}")
    note = alert(status)
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if note and topic and not args.no_notify:
        print(f"[자료 점검] ntfy: {note[0]} → {'보냄' if send_ntfy(topic, *note) else '실패'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
