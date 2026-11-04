"""종목 상세 '분기 실적' · '애널리스트 의견 수' (2026-11-03) — yfinance 종목마다 요청 3~4번이라 캐시하며 조금씩 받는다.

전체분석(src/run_universe_scan.py)이 S&P 500 + 시총 $2B 이상 종목만 대상으로, 매일 밤 시간 한도 안에서
① 아직 없는 종목 ② 최근 실적 발표 뒤 아직 덜 받은 종목 ③ 오래된 종목 순으로 새로 받는다. 못 받으면 전날 값을 그대로 쓴다.
캐시는 universe-data 브랜치의 snapshot/detail.json.gz — 다음 밤 실행이 전날 스냅샷처럼 되살린다 (universe-scan.yml).
표시 전용 — 점수 · 필터 · PT-1 무관.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

CACHE_NAME = "detail.json.gz"
MAX_AGE_DAYS = 14        # 이보다 오래되면 다시 (의견 수 · 다음 분기 예상이 바뀐다)
AFTER_REPORT_DAYS = 4    # 실적 발표 뒤 이 날까지는 매일 다시 (yfinance 가 매출을 며칠 늦게 넣는다)
EPS_QUARTERS = 8
REV_QUARTERS = 8
WORKERS = 4
MAX_FAIL_STREAK = 30     # 연속으로 이만큼 실패하면 (막힘) 그날은 그만 받는다
PARTIAL_RETRIES = 3      # 4개 요청 중 일부만 실패한 종목은 다음 밤에 다시 — 연속 이 횟수까지 (그 뒤는 14일 주기로)
DETAIL_KEYS = ("rec", "eps", "revenue", "next")  # fetch_one 이 채우는 칸 (실패한 칸은 결과에 없다)

Fetch = Callable[[str], dict[str, Any]]


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _r(v: Any, digits: int) -> float | None:
    x = _f(v)
    return None if x is None else round(x, digits)


# ── yfinance 결과 → 웹 모양 (순수 함수, 테스트 가능)

def parse_rec(df: pd.DataFrame | None) -> dict[str, int] | None:
    """recommendations_summary → 이번 달(period 0m) 의견 수."""
    if df is None or df.empty or "period" not in df.columns:
        return None
    cur = df[df["period"] == "0m"]
    if cur.empty:
        return None
    r = cur.iloc[0]
    keys = {"strong_buy": "strongBuy", "buy": "buy", "hold": "hold", "sell": "sell", "strong_sell": "strongSell"}
    out = {k: int(_f(r.get(src)) or 0) for k, src in keys.items()}
    return out if sum(out.values()) > 0 else None


def parse_eps(df: pd.DataFrame | None, n: int = EPS_QUARTERS) -> list[dict[str, Any]]:
    """get_earnings_dates → 발표된 최근 n 번 EPS 예상 · 실제 · 서프라이즈(%) (오래된 순)."""
    if df is None or df.empty or "Reported EPS" not in df.columns:
        return []
    done = df[df["Reported EPS"].notna()].sort_index().iloc[-n:]
    out = []
    for ts, r in done.iterrows():
        out.append({"date": pd.Timestamp(ts).date().isoformat(), "est": _r(r.get("EPS Estimate"), 4),
                    "act": _r(r.get("Reported EPS"), 4), "surprise": _r(r.get("Surprise(%)"), 2)})
    return out


def parse_revenue(df: pd.DataFrame | None, n: int = REV_QUARTERS) -> list[dict[str, Any]]:
    """quarterly_income_stmt → 분기 매출 (분기 끝 날짜, 오래된 순). 매출이 없는 분기는 뺀다."""
    if df is None or df.empty or "Total Revenue" not in df.index:
        return []
    out = []
    for col in sorted(df.columns)[-n:]:
        rev = _f(df.loc["Total Revenue", col])
        if rev and rev > 0:
            eps = _f(df.loc["Diluted EPS", col]) if "Diluted EPS" in df.index else None
            out.append({"end": pd.Timestamp(col).date().isoformat(), "revenue": rev,
                        "eps_gaap": round(eps, 4) if eps is not None else None})
    return out


def parse_next(eps_est: pd.DataFrame | None, rev_est: pd.DataFrame | None) -> dict[str, Any]:
    """earnings_estimate · revenue_estimate 의 이번 분기(0q) → 다음 실적 예상. 날짜는 사이트가 종목 정보로 채운다."""
    out: dict[str, Any] = {}
    if eps_est is not None and not eps_est.empty and "0q" in eps_est.index:
        r = eps_est.loc["0q"]
        out.update({"eps_est": _r(r.get("avg"), 4), "eps_year_ago": _r(r.get("yearAgoEps"), 4),
                    "n": int(_f(r.get("numberOfAnalysts")) or 0) or None})
    if rev_est is not None and not rev_est.empty and "0q" in rev_est.index:
        r = rev_est.loc["0q"]
        out.update({"rev_est": _f(r.get("avg")), "rev_growth": _r(r.get("growth"), 4)})
    return out


def fetch_one(ticker: str) -> dict[str, Any]:
    """yfinance 로 한 종목 — 요청마다 따로 잡아서 일부만 실패해도 나머지는 남긴다. 다 실패하면 RuntimeError."""
    import yfinance as yf  # noqa: PLC0415

    tk = yf.Ticker(ticker.replace(".", "-"))
    out: dict[str, Any] = {}
    errors = 0
    for key, get in (("rec", lambda: parse_rec(tk.recommendations_summary)),  # 키 = DETAIL_KEYS
                     ("eps", lambda: parse_eps(tk.get_earnings_dates(limit=16))),
                     ("revenue", lambda: parse_revenue(tk.quarterly_income_stmt)),
                     ("next", lambda: parse_next(tk.earnings_estimate, tk.revenue_estimate))):
        try:
            out[key] = get()
        except Exception:  # noqa: BLE001 — 표시용, 한 칸 실패는 넘어간다
            errors += 1
    if errors == 4:
        raise RuntimeError(f"{ticker}: 4개 요청 모두 실패")
    return out


# ── 무엇을 다시 받을지

def refresh_order(tickers: list[str], cache: Mapping[str, Mapping[str, Any]], today: date,
                  last_report: Mapping[str, str | None]) -> list[str]:
    """다시 받을 종목 — ① 캐시에 없음 ② 최근 실적 발표 뒤 AFTER_REPORT_DAYS 일이 지나기 전에 받음
    · 지난번에 일부 칸만 받음(missing, 연속 PARTIAL_RETRIES 번까지 · 오늘 받은 건 빼고) ③ MAX_AGE_DAYS 일 넘음.
    같은 묶음 안에서는 오래 받은 순. 이미 충분히 새것이면 빠진다."""
    tiers: list[tuple[int, str, str]] = []
    for t in tickers:
        ent = cache.get(t)
        fetched = str(ent.get("fetched", "")) if ent else ""
        if not fetched:
            tiers.append((0, "", t))
            continue
        rep = last_report.get(t)
        if rep and rep <= today.isoformat():
            settle = (datetime.strptime(rep[:10], "%Y-%m-%d").date() + timedelta(days=AFTER_REPORT_DAYS)).isoformat()
            if fetched < settle:
                tiers.append((1, fetched, t))
                continue
        if ent.get("missing") and int(ent.get("tries", 0)) < PARTIAL_RETRIES and fetched < today.isoformat():
            tiers.append((1, fetched, t))
            continue
        if fetched <= (today - timedelta(days=MAX_AGE_DAYS)).isoformat():
            tiers.append((2, fetched, t))
    return [t for _, _, t in sorted(tiers)]


def refresh(tickers: list[str], cache: dict[str, dict[str, Any]], today: date, last_report: Mapping[str, str | None],
            *, budget_sec: float, fetch: Fetch = fetch_one, workers: int = WORKERS,
            log: Callable[[str], None] = print) -> dict[str, int]:
    """시간 한도 안에서 cache 를 고쳐 쓴다 → {"want", "ok", "fail", "left"}. 받은 종목은 fetched = today.
    일부 칸만 받으면 못 받은 칸은 전에 받은 값을 그대로 두고 missing · tries 를 적는다 (→ 다음 밤에 다시, refresh_order)."""
    order = refresh_order(tickers, cache, today, last_report)
    start = time.monotonic()
    ok = fail = streak = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = iter(order)
        running: dict[Any, str] = {}

        def submit_next() -> bool:
            if time.monotonic() - start > budget_sec or streak >= MAX_FAIL_STREAK:
                return False
            t = next(it, None)
            if t is None:
                return False
            running[pool.submit(fetch, t)] = t
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while running:
            fut = next(as_completed(list(running)))
            t = running.pop(fut)
            done += 1
            try:
                cache[t] = merge_entry(cache.get(t), fut.result(), today)
                ok += 1
                streak = 0
            except Exception:  # noqa: BLE001
                fail += 1
                streak += 1
            submit_next()
    if streak >= MAX_FAIL_STREAK:
        log(f"[종목 상세] 연속 {streak}번 실패 — 막힌 것 같아 그만 받음 (전날 값 사용)")
    stats = {"want": len(order), "ok": ok, "fail": fail, "left": len(order) - done}
    log(f"[종목 상세] 받을 {stats['want']} · 받음 {ok} · 실패 {fail} · 다음 밤으로 {stats['left']} "
        f"({(time.monotonic() - start) / 60:.1f}분)")
    return stats


def merge_entry(old: Mapping[str, Any] | None, new: Mapping[str, Any], today: date) -> dict[str, Any]:
    """새로 받은 칸 + 이번에 못 받은 칸은 전 값 (없으면 비움). 빠진 칸이 있으면 missing · tries(연속 횟수)."""
    old = old or {}
    ent: dict[str, Any] = {k: old[k] for k in DETAIL_KEYS if k in old}
    ent.update({k: v for k, v in new.items() if k in DETAIL_KEYS})
    ent["fetched"] = today.isoformat()
    missing = [k for k in DETAIL_KEYS if k not in new]
    if missing:
        ent["missing"] = missing
        ent["tries"] = int(old.get("tries", 0)) + 1 if old.get("missing") else 1
    return ent


def quarters_block(ent: Mapping[str, Any] | None, earnings_date: str | None) -> dict[str, Any] | None:
    """캐시 한 종목 → 웹 quarters (eps · revenue · next). 숫자가 하나도 없으면 None."""
    if not ent or not (ent.get("eps") or ent.get("revenue")):
        return None
    return {"eps": list(ent.get("eps") or []), "revenue": list(ent.get("revenue") or []),
            "next": {"date": earnings_date, **(ent.get("next") or {})}}


# ── 캐시 파일

def load_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    """전날 캐시 (없거나 못 읽으면 빈 dict)."""
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        return {str(k): v for k, v in data.get("items", {}).items() if isinstance(v, dict)}
    except (OSError, ValueError) as exc:
        print(f"[종목 상세] 캐시 읽기 실패 (처음부터): {exc}")
        return {}


def save_cache(path: Path, cache: Mapping[str, Mapping[str, Any]], keep: set[str] | None = None) -> int:
    """캐시 저장 — keep 이 있으면 그 종목만 (목록에서 빠진 종목은 버린다). 바이트 수."""
    items = {t: v for t, v in cache.items() if keep is None or t in keep}
    body = json.dumps({"schema": 1, "items": items}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    packed = gzip.compress(body.encode("utf-8"), compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed)
    return len(packed)
