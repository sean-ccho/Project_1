#!/usr/bin/env python3
"""그림자 추적 (shadow_v1): 돈을 쓰지 않고 신호가 고른 종목 목록만 매일 기록 → 나중에 SPY·같은 풀과 비교한다.

계기: 2026-10-15 사용자 — "그림자 추적부터 시작" (Sharadar 대신 앞으로 쌓이는 실전 기록으로 확인).
기록 시점에 정해 둔 목록이라 미래 정보·생존 편향이 없다 (나중에 상장폐지돼도 기록에 남는다).
실행은 이 Mac mini 의 LaunchAgent — 실전 파이프라인(CI)은 건드리지 않는다. macOS 가 예약 작업의 ~/Documents 접근을 막아서
~/shadow-track 에 이 리포의 얕은 복제(매번 origin/main 으로 맞춤)와 venv 를 두고 거기서 돈다 (~/shadow-track/run.sh).
PT 후보표는 CI 가 커밋한 origin/main 의 data/paper_trading/*_ranked.parquet 를 git 으로 읽는다.

사전 등록 (TRIAL_LOG family shadow_v1, 5건 → Bonferroni z = 2.58). **판정은 2027-10-15 에 한 번** — 그 전 보고서는 참고용,
규칙을 바꾸는 근거로 쓰지 않는다.
  PT1S_TOP5 (buy, h21)  PT-1 규칙 그대로 (candidate_selector.select_top_candidates, k=5, 보유 없음) S&P 500 후보 중
                        고른 종목(최대 5) → 기록일 다음 거래일 시가부터 21거래일 시가, 동일가중 − SPY 같은 구간. 기록일별 평균 → NW t
  PT1N_TOP5 (buy, h21)  같은 규칙, S&P 500 밖 (PT-1N 과 같은 후보표) − SPY
  DTC_HIGH (avoid, h63) 현재 S&P 500 목록의 FINRA days to cover (공개된 가장 최근 보고) 상위 20% − 값 있는 종목 평균. 월말 기록만
  EPS_REV  (buy, h21)   올해(0y) EPS 추정치 30일 변화율 = (current − 30daysAgo) / |30daysAgo| (yfinance eps_trend, 주 1회,
                        |30daysAgo| ≥ 0.05). S&P 500 + S&P 밖 후보 풀. 월말 직전 스냅샷의 상위 20% − 값 있는 종목 평균
  RUNNER_N (buy, h63)   S&P 밖 후보 풀에서 SEC 분기 매출 YoY ≥ 15% 이고 2분기 연속 가속 (idea_batch_v2 RUNNER 와 같은 정의)
                        − 매출 YoY 있는 풀 평균. 월말. SEC 는 User-Agent 에 연락처가 필요 → SEC_USER_AGENT 가 있을 때만
  참고 (판정 없음): PT1_TOP5 (PT-1 전체 후보표), 각 TOP5 − IWM · − 그날 후보 풀 동일가중
  판정 = idea_batch_v2.judge (buy: 후보 = t ≥ z · 비용 후 > 0 · 앞/뒤 반 + · 연도 2/3 +; avoid 는 부호를 뒤집어 같은 기준).
  약세장(SPY < EMA200)이라 PT 규칙이 신규 진입을 막은 날은 TOP5 가 비어 그날은 관측에서 빠진다 (날짜 수를 보고서에 적는다).

기록: $SHADOW_DIR/days/<일봉날짜>.parquet (기본 data/research/shadow, LaunchAgent 는 ~/shadow-track/data)
      — 그날 파일이 있으면 빠진 신호만 채운다 (이미 쓴 값은 고치지 않는다). 보고서는 $SHADOW_DIR/SHADOW_REPORT.md.
      열: date, signal, ticker, rank, value, flag, price
사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/shadow_track.py --register      # 한 번
  PYTHONPATH=.:src .venv/bin/python scripts/shadow_track.py --record        # 매일 (LaunchAgent)
  PYTHONPATH=.:src .venv/bin/python scripts/shadow_track.py --backfill-pt   # 9-28 부터 git 에 남은 후보표로 PT TOP5 만 채움
  PYTHONPATH=.:src .venv/bin/python scripts/shadow_track.py --report        # $SHADOW_DIR/SHADOW_REPORT.md
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from research_utils import RESEARCH_DIR, TRIAL_LOG, bonferroni_z, log_trials  # noqa: E402

FAMILY = "shadow_v1"
STEP = "shadow"
JUDGE_DATE = "2027-10-15"
SHADOW_DIR = Path(os.environ["SHADOW_DIR"]).expanduser() if os.environ.get("SHADOW_DIR") else RESEARCH_DIR / "shadow"
DAYS_DIR = SHADOW_DIR / "days"
CACHE_DIR = SHADOW_DIR / "cache"
SNAPSHOTS = {"sp500": "data/paper_trading/sp500_ranked.parquet", "nasdaq": "data/paper_trading/nasdaq_ranked.parquet"}
# (test_id, 신호, h, 종류, 방향)  종류: top = 기록일 TOP5 − SPY · xs = 월말 상위 20% − 평균 · set = 월말 집합 − 풀 평균
TESTS: list[tuple[str, str, int, str, str]] = [
    ("PT1S_TOP5|h21", "PT1S_TOP5", 21, "top", "buy"),
    ("PT1N_TOP5|h21", "PT1N_TOP5", 21, "top", "buy"),
    ("DTC_HIGH|h63", "DTC", 63, "xs", "avoid"),
    ("EPS_REV|h21", "EPS_REV", 21, "xs", "buy"),
    ("RUNNER_N|h63", "RUNNER", 63, "set", "buy"),
]
REFERENCE = [("PT1_TOP5|h21", "PT1_TOP5", 21, "top", "buy")]
PT_UNIVERSES = {"PT1S_TOP5": "sp500", "PT1N_TOP5": "ex_sp500", "PT1_TOP5": "all"}
POOL_SIGNAL = {"PT1S_TOP5": "POOL_S", "PT1N_TOP5": "POOL_N", "PT1_TOP5": "POOL_ALL"}
EPS_EVERY_DAYS = 6       # eps_trend · SEC 매출은 6일 넘게 지났을 때만 다시 받는다 (주 1회)
EPS_MIN_BASE = 0.05
RUNNER_MIN_YOY = 0.15
REV_STALE_DAYS = 200
TOP_Q = 0.2
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
NAN = float("nan")
COLUMNS = ["date", "signal", "ticker", "rank", "value", "flag", "price"]


def _norm(t: Any) -> str:
    return str(t).strip().upper().replace(".", "-").replace("/", "-")


def _quiet(fn, *a, **kw):
    """PT 모듈이 찍는 진행 메시지는 숨긴다."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ============================================================
# PT 후보표 (CI 가 커밋한 origin/main)
# ============================================================
def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def snapshot_files(rev: str, dest: Path) -> dict[str, Path]:
    """rev 의 두 후보표를 dest 에 꺼낸다 (없는 파일은 빠진다)."""
    out = {}
    for key, rel in SNAPSHOTS.items():
        try:
            data = _git("show", f"{rev}:{rel}")
        except subprocess.CalledProcessError:
            continue
        p = dest / Path(rel).name
        p.write_bytes(data)
        out[key] = p
    return out


def bar_date_of(files: dict[str, Path]) -> str | None:
    from paper_trading.runner import snapshot_bar_date  # noqa: PLC0415

    dates = [snapshot_bar_date(pd.read_parquet(p)) for p in files.values()]
    dates = [d for d in dates if d]
    return max(dates) if dates else None


def pt_rows(files: dict[str, Path], day: str) -> list[dict[str, Any]]:
    """PT-1 · PT-1S · PT-1N 규칙으로 고른 최대 5종목 + 그날 후보 풀 (풀 동일가중 비교용)."""
    from paper_trading import runner  # noqa: PLC0415
    from paper_trading.candidate_selector import select_top_candidates  # noqa: PLC0415

    old = runner._SP500_SNAPSHOT, runner._NASDAQ_SNAPSHOT
    runner._SP500_SNAPSHOT = files.get("sp500", Path("/nonexistent/sp500.parquet"))
    runner._NASDAQ_SNAPSHOT = files.get("nasdaq", Path("/nonexistent/nasdaq.parquet"))
    rows: list[dict[str, Any]] = []
    try:
        for signal, universe in PT_UNIVERSES.items():
            df = _quiet(runner.load_and_merge_snapshots, universe)
            if df is None or df.empty:
                continue
            picks, _ = _quiet(select_top_candidates, df, [], 5)
            for i, p in enumerate(picks, 1):
                rows.append({"date": day, "signal": signal, "ticker": _norm(p["ticker"]), "rank": i,
                             "value": float(p.get("ccs_score") or NAN), "flag": True,
                             "price": float(p.get("entry_price") or NAN)})
            pool = df[~df["티커"].astype(str).isin({"SPY", "QQQ", "IWM", "DIA"})]
            rows += [{"date": day, "signal": POOL_SIGNAL[signal], "ticker": _norm(t), "rank": None, "value": NAN,
                      "flag": False, "price": float(pd.to_numeric(px, errors="coerce"))}
                     for t, px in zip(pool["티커"], pool.get("현재가격", pd.Series(NAN, index=pool.index)))]
    finally:
        runner._SP500_SNAPSHOT, runner._NASDAQ_SNAPSHOT = old
    return rows


# ============================================================
# FINRA days to cover
# ============================================================
def _finra(body: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    for attempt in range(4):
        try:
            r = requests.post(FINRA_URL, json=body, headers={"Accept": "application/json"}, timeout=90)
            if r.status_code == 200:
                return (r.json() if r.text.strip() else []), int(r.headers.get("record-total", "0") or 0)
            print(f"  [finra] HTTP {r.status_code} 재시도 {attempt + 1}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [finra] {exc} 재시도 {attempt + 1}", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("FINRA 요청 실패")


def finra_latest(asof: str) -> pd.DataFrame:
    """asof 까지 공개된 가장 최근 결제일의 전 종목 공매도 잔고 (결제일별 캐시)."""
    since = (pd.Timestamp(asof) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    rows, _ = _finra({"limit": 100, "fields": ["settlementDate"],
                      "compareFilters": [{"compareType": "equal", "fieldName": "symbolCode", "fieldValue": "AAPL"},
                                         {"compareType": "gte", "fieldName": "settlementDate", "fieldValue": since}]})
    dates = sorted(d for d in {r["settlementDate"] for r in rows} if d <= asof)
    if not dates:
        return pd.DataFrame()
    settle = dates[-1]
    cache = CACHE_DIR / f"finra_{settle}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    got, offset, total = [], 0, None
    while total is None or offset < total:
        page, total = _finra({"limit": 5000, "offset": offset,
                              "fields": ["settlementDate", "symbolCode", "currentShortPositionQuantity", "daysToCoverQuantity"],
                              "compareFilters": [{"compareType": "equal", "fieldName": "settlementDate", "fieldValue": settle}]})
        got += page
        offset += 5000
        if not page:
            break
    df = pd.DataFrame(got).rename(columns={"symbolCode": "ticker", "daysToCoverQuantity": "dtc",
                                           "currentShortPositionQuantity": "si", "settlementDate": "settle"})
    df["ticker"] = df["ticker"].map(_norm)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def dtc_rows(day: str) -> list[dict[str, Any]]:
    from data.sp500_tickers import SP500_TICKERS  # noqa: PLC0415

    fin = finra_latest(day)
    if fin.empty:
        return []
    members = {_norm(t) for t in SP500_TICKERS}
    d = fin[fin["ticker"].isin(members)].dropna(subset=["dtc"]).drop_duplicates("ticker")
    cut = d["dtc"].astype(float).quantile(1 - TOP_Q)
    print(f"[shadow] FINRA {d['settle'].iloc[0]} · S&P {len(d)}종목 · DTC 상위 20% 기준 {cut:.2f}일", flush=True)
    return [{"date": day, "signal": "DTC", "ticker": t, "rank": None, "value": float(v), "flag": bool(float(v) >= cut),
             "price": NAN} for t, v in zip(d["ticker"], d["dtc"])]


# ============================================================
# 주 1회: EPS 추정치 · SEC 분기 매출
# ============================================================
def _last_weekly(signal: str) -> str | None:
    """그 신호를 마지막으로 기록한 일봉 날짜."""
    last = None
    for p in sorted(DAYS_DIR.glob("*.parquet")):
        s = pd.read_parquet(p, columns=["signal"])["signal"]
        if (s == signal).any():
            last = p.stem
    return last


def _due(signal: str, day: str) -> bool:
    last = _last_weekly(signal)
    return last is None or (pd.Timestamp(day) - pd.Timestamp(last)).days >= EPS_EVERY_DAYS


def eps_rows(day: str, tickers: list[str]) -> list[dict[str, Any]]:
    import yfinance as yf  # noqa: PLC0415

    rows, fails = [], 0
    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        try:
            tr = yf.Ticker(tk).eps_trend
            if tr is None or tr.empty or "0y" not in tr.index:
                fails += 1
            else:
                cur, base = float(tr.loc["0y", "current"]), float(tr.loc["0y", "30daysAgo"])
                rev = (cur - base) / abs(base) if np.isfinite(base) and abs(base) >= EPS_MIN_BASE else NAN
                rows.append({"date": day, "signal": "EPS_REV", "ticker": tk, "rank": None, "value": rev,
                             "flag": False, "price": NAN})
        except Exception:  # noqa: BLE001 — 한 종목 실패로 멈추지 않는다
            fails += 1
        if i % 100 == 0:
            print(f"[shadow] eps_trend {i}/{len(tickers)} ({time.time() - t0:.0f}s, 실패 {fails})", flush=True)
        time.sleep(0.35)
    vals = pd.Series([r["value"] for r in rows], dtype=float)
    cut = vals.quantile(1 - TOP_Q) if vals.notna().any() else NAN
    for r in rows:
        r["flag"] = bool(np.isfinite(r["value"]) and r["value"] >= cut)
    print(f"[shadow] EPS_REV {len(rows)}종목 (값 있음 {int(vals.notna().sum())}) · 실패 {fails}", flush=True)
    return rows


def _sec_get(url: str, ua: str) -> bytes | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 * (attempt + 1))
    return None


def runner_rows(day: str, pool: list[str]) -> list[dict[str, Any]]:
    """S&P 밖 후보 풀의 SEC 분기 매출 YoY → 러너형 (YoY ≥ 15% · 2분기 연속 가속). SEC_USER_AGENT 가 없으면 건너뛴다."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        print("[shadow] SEC_USER_AGENT 없음 — RUNNER 건너뜀 (SEC 규칙: User-Agent 에 연락처)", flush=True)
        return []
    from gc_insider_rev import quarterly_revenue, yoy_table  # noqa: PLC0415

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tick_path = CACHE_DIR / "sec_company_tickers.json"
    if not tick_path.exists() or time.time() - tick_path.stat().st_mtime > 30 * 86400:
        raw = _sec_get(SEC_TICKERS_URL, ua)
        if raw:
            tick_path.write_bytes(raw)
    if not tick_path.exists():
        return []
    cik = {_norm(v["ticker"]): int(v["cik_str"]) for v in json.loads(tick_path.read_text()).values()}
    rows = []
    dd = pd.Timestamp(day)
    for tk in pool:
        c = cik.get(tk)
        if c is None:
            continue
        fp = CACHE_DIR / "companyfacts" / f"CIK{c:010d}.json.gz"
        if not fp.exists() or time.time() - fp.stat().st_mtime > EPS_EVERY_DAYS * 86400:
            raw = _sec_get(SEC_FACTS_URL.format(cik=c), ua)
            time.sleep(0.15)  # 초당 10회 이하
            if raw is None:
                continue
            fp.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(fp, "wb") as f:
                f.write(raw)
        with gzip.open(fp) as f:
            gaap = json.load(f).get("facts", {}).get("us-gaap", {})
        y = yoy_table(quarterly_revenue(gaap))
        y = y[pd.to_datetime(y["known"]) < dd].sort_values("end")
        if y.empty or (dd - pd.Timestamp(y["end"].iloc[-1])).days > REV_STALE_DAYS:
            continue
        ends, yoy = pd.to_datetime(y["end"]).to_numpy(), y["yoy"].to_numpy(float)

        def prev(i: int) -> int | None:
            gap = np.abs((ends - (ends[i] - np.timedelta64(91, "D"))) / np.timedelta64(1, "D"))
            j = int(np.argmin(gap))
            return j if gap[j] <= 20 and j != i else None

        i0 = len(ends) - 1
        i1 = prev(i0)
        i2 = prev(i1) if i1 is not None else None
        y0, y1 = yoy[i0], (yoy[i1] if i1 is not None else NAN)
        y2 = yoy[i2] if i2 is not None else NAN
        runner = bool(np.isfinite(y2) and y0 >= RUNNER_MIN_YOY and y0 > y1 > y2)
        rows.append({"date": day, "signal": "RUNNER", "ticker": tk, "rank": None, "value": float(y0), "flag": runner,
                     "price": NAN})
    print(f"[shadow] RUNNER 풀 {len(pool)} · 매출 있음 {len(rows)} · 러너형 {sum(r['flag'] for r in rows)}", flush=True)
    return rows


# ============================================================
# 기록
# ============================================================
def save_day(day: str, rows: list[dict[str, Any]]) -> Path:
    DAYS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int16")
    path = DAYS_DIR / f"{day}.parquet"
    df.to_parquet(path, index=False)
    return path


def record() -> int:
    """오늘(최신 후보표의 일봉 날짜) 기록. 그날 파일이 있으면 **빠진 신호만** 채운다 (저녁 재시도·백필한 날)."""
    try:
        _git("fetch", "-q", "origin", "main")
    except subprocess.CalledProcessError as exc:
        print(f"[shadow] git fetch 실패 (마지막으로 받은 origin/main 사용): {exc.stderr.decode()[:200]}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        files = snapshot_files("origin/main", Path(tmp))
        day = bar_date_of(files) if files else None
        if not day:
            print("[shadow] 후보표 일봉 날짜 없음 (장중 실행 등) — 건너뜀", flush=True)
            return 0
        path = DAYS_DIR / f"{day}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
        have = set(old["signal"])
        rows: list[dict[str, Any]] = []
        if "POOL_S" not in have:  # 후보 풀이 있으면 PT 는 기록된 것 (TOP5 는 비어 있을 수 있다)
            rows = pt_rows(files, day)
            for sig in PT_UNIVERSES:
                picks = [r["ticker"] for r in rows if r["signal"] == sig]
                print(f"[shadow] {sig}: {picks or '없음 (규칙 통과 종목 없음)'}", flush=True)
    print(f"===== 그림자 기록 {day} (이미 있는 신호: {sorted(have) or '없음'})", flush=True)
    if "DTC" not in have:
        try:
            rows += dtc_rows(day)
        except Exception as exc:  # noqa: BLE001 — 한 신호 실패로 나머지를 버리지 않는다 (저녁에 다시)
            print(f"[shadow] DTC 실패: {exc}", flush=True)
    pool_n = sorted({r["ticker"] for r in rows + old.to_dict("records") if r["signal"] == "POOL_N"})
    if "EPS_REV" not in have and _due("EPS_REV", day):
        from data.sp500_tickers import SP500_TICKERS  # noqa: PLC0415

        rows += eps_rows(day, sorted({_norm(t) for t in SP500_TICKERS} | set(pool_n)))
    if "RUNNER" not in have and _due("RUNNER", day) and pool_n:
        try:
            rows += runner_rows(day, pool_n)
        except Exception as exc:  # noqa: BLE001
            print(f"[shadow] RUNNER 실패: {exc}", flush=True)
    if not rows:
        print(f"[shadow] {day} 더 채울 신호 없음", flush=True)
        return 0
    merged = rows if old.empty else old.to_dict("records") + rows
    path = save_day(day, merged)
    print(f"[shadow] 저장 {path.name} — {pd.DataFrame(merged)['signal'].value_counts().to_dict()}", flush=True)
    return 0


def backfill_pt() -> None:
    """git 에 남은 후보표 버전마다 PT TOP5 만 기록 (DTC·EPS 는 그때 값을 모르므로 채우지 않는다)."""
    revs = _git("log", "--format=%H", "origin/main", "--", SNAPSHOTS["sp500"], SNAPSHOTS["nasdaq"]).decode().split()
    done = 0
    for rev in reversed(revs):
        with tempfile.TemporaryDirectory() as tmp:
            files = snapshot_files(rev, Path(tmp))
            day = bar_date_of(files) if files else None
            if not day or (DAYS_DIR / f"{day}.parquet").exists():
                continue
            rows = pt_rows(files, day)
        save_day(day, rows)
        done += 1
        print(f"[backfill] {day} ({rev[:7]}): " + " · ".join(
            f"{s} {[r['ticker'] for r in rows if r['signal'] == s]}" for s in PT_UNIVERSES), flush=True)
    print(f"[backfill] {done}일 기록", flush=True)


# ============================================================
# 보고서
# ============================================================
def load_days() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(DAYS_DIR.glob("*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_opens(tickers: list[str], start: pd.Timestamp) -> pd.DataFrame:
    """기록 종목 + SPY·IWM 일봉 시가 (배당 조정)."""
    import yfinance as yf  # noqa: PLC0415

    frames = []
    tks = sorted(set(tickers) | {"SPY", "IWM"})
    for i in range(0, len(tks), 100):
        data = yf.download(tks[i:i + 100], start=(start - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        if data is None or data.empty:
            continue
        o = data["Open"] if isinstance(data.columns, pd.MultiIndex) else data[["Open"]].rename(columns={"Open": tks[i]})
        frames.append(o)
    opens = pd.concat(frames, axis=1)
    opens.index = pd.to_datetime(opens.index).tz_localize(None) if opens.index.tz is not None else pd.to_datetime(opens.index)
    return opens.loc[:, ~opens.columns.duplicated()].sort_index()


def fwd(opens: pd.DataFrame, day: pd.Timestamp, h: int) -> pd.Series | None:
    """기록일 다음 거래일 시가 → h 거래일 뒤 시가 수익률 (아직 안 끝났으면 None)."""
    pos = opens.index.searchsorted(day, side="right")
    if pos + h >= len(opens.index):
        return None
    return opens.iloc[pos + h] / opens.iloc[pos] - 1.0


def month_ends(dates: pd.Series) -> list[pd.Timestamp]:
    d = pd.Series(sorted(dates.unique()))
    return list(d.groupby(d.dt.to_period("M")).max())


def evaluate(days: pd.DataFrame, opens: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, pd.Series]]:
    from idea_batch_v2 import _summ, judge  # noqa: PLC0415 — 같은 요약·판정

    z = bonferroni_z(len(TESTS))
    out, series = [], {}
    for tid, sig, h, kind, direction in TESTS + REFERENCE:
        obs, ref_iwm, ref_pool, empty_days = [], [], [], 0
        if kind == "top":
            all_days = sorted(days.loc[days["signal"] == POOL_SIGNAL[sig], "date"].unique())
            for d in all_days:
                picks = days[(days["date"] == d) & (days["signal"] == sig)]["ticker"].tolist()
                if not picks:
                    empty_days += 1
                    continue
                r = fwd(opens, d, h)
                if r is None:
                    continue
                pr = r.reindex(picks).dropna()
                pool = r.reindex(days[(days["date"] == d) & (days["signal"] == POOL_SIGNAL[sig])]["ticker"]).dropna()
                if pr.empty or "SPY" not in r:
                    continue
                obs.append({"date": d, "ex": pr.mean() - r["SPY"], "spread": NAN, "ic": NAN, "n": len(pr)})
                ref_iwm.append(pr.mean() - r.get("IWM", NAN))
                ref_pool.append(pr.mean() - pool.mean() if len(pool) else NAN)
            lag = h
        else:
            sub = days[days["signal"] == sig]
            for d in month_ends(days["date"]) if not sub.empty else []:
                snap_dates = sub.loc[sub["date"] <= d, "date"]
                if snap_dates.empty:
                    continue
                s_day = snap_dates.max()
                if (d - s_day).days > 10:
                    continue
                g = sub[(sub["date"] == s_day) & sub["value"].notna()]
                r = fwd(opens, d, h)
                if r is None or g.empty:
                    continue
                vals = r.reindex(g["ticker"]).to_numpy()
                g = g.assign(fwd=vals).dropna(subset=["fwd"])
                top = g[g["flag"]]
                if len(g) < 20 or top.empty:
                    continue
                obs.append({"date": d, "ex": top["fwd"].mean() - g["fwd"].mean(), "spread": NAN, "ic": NAN,
                            "n": len(top)})
            lag = max(1, h // 21)
        res, s = _summ(pd.DataFrame(obs), lag)
        verdict = judge(res, z, direction) if res else "표본 없음"
        out.append({"test_id": tid, "kind": kind, "direction": direction, "registered": (tid, sig, h, kind, direction) in TESTS,
                    "empty_days": empty_days, "vs_iwm": float(np.nanmean(ref_iwm)) if ref_iwm else NAN,
                    "vs_pool": float(np.nanmean(ref_pool)) if ref_pool else NAN, **res,
                    "판정(참고)": verdict})
        series[tid] = s
    return out, series


def report() -> None:
    days = load_days()
    if days.empty:
        print("[shadow] 기록 없음")
        return
    tickers = sorted(set(days["ticker"]))
    opens = load_opens(tickers, days["date"].min())
    rows, _ = evaluate(days, opens)
    first, last = days["date"].min().date(), days["date"].max().date()
    n_days = days["date"].nunique()
    lines = [
        "# 그림자 추적 (shadow_v1) — 진행 보고서", "",
        f"- 작성 {datetime.now():%Y-%m-%d %H:%M} · 기록 {n_days}일 ({first} ~ {last}) · 정의는 `scripts/shadow_track.py` 머리말",
        f"- **판정은 {JUDGE_DATE} 에 한 번** (사전 등록 {len(TESTS)}건, z = {bonferroni_z(len(TESTS)):.2f}). "
        "그 전 숫자는 참고용 — 이걸 보고 규칙을 바꾸지 않는다.",
        "- TOP5 = 기록일 다음 거래일 시가부터 h 거래일 시가, 동일가중 − SPY. 월말 신호 = 상위 20%(또는 집합) − 값 있는 종목 평균.", "",
        "| test | 등록 | 끝난 관측 | 평균 초과 | t_nw | 연도+ | 앞/뒤 반 | − IWM | − 후보 풀 | 빈 날 | 지금 판정(참고) |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda v: f"{v:+.2%}" if v is not None and pd.notna(v) else "–"  # noqa: E731
    for r in rows:
        lines.append(f"| {r['test_id']} | {'✓' if r['registered'] else '참고'} | {r.get('n_obs', 0)} | {pct(r.get('mean_ex'))} | "
                     f"{r.get('t_nw', NAN):+.2f} | {r.get('years_pos', '–')}/{r.get('n_years', '–')} | "
                     f"{pct(r.get('first_half'))} / {pct(r.get('second_half'))} | {pct(r['vs_iwm'])} | {pct(r['vs_pool'])} | "
                     f"{r['empty_days'] or '–'} | {r['판정(참고)']} |")
    counts = days.groupby("signal")["date"].nunique().to_dict()
    lines += ["", "기록한 날 수: " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())),
              "", "빈 날 = PT 규칙을 통과한 종목이 없던 날 (약세장 신규 진입 중단 포함). 그날은 관측에서 빠진다."]
    (SHADOW_DIR / "SHADOW_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def register() -> None:
    log = pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame(columns=["family", "판정"])
    if ((log["family"] == FAMILY) & (log["판정"] == "등록")).any():
        print(f"[등록] {FAMILY} 이미 등록됨")
        return
    log_trials(STEP, FAMILY, len(TESTS), [{"test_id": "(등록)", "판정": "등록", "메모": (
        f"앞으로 쌓는 그림자 기록 {len(TESTS)}건 (PT1S_TOP5·PT1N_TOP5 h21 vs SPY, DTC_HIGH h63 피하기, EPS_REV h21, "
        f"RUNNER_N h63), z={bonferroni_z(len(TESTS)):.2f}, 판정 {JUDGE_DATE} 한 번. 정의는 scripts/shadow_track.py 머리말")}])
    print(f"[등록] {FAMILY} {len(TESTS)}건, 판정 {JUDGE_DATE}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--backfill-pt", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.register:
        register()
    if args.backfill_pt:
        backfill_pt()
    if args.record:
        record()
    if args.report:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
