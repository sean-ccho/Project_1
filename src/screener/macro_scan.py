"""경기 상태 · 금리 결정 — 사이트 오늘 탭(/quant)의 macro.json · 시트 [경제] (2026-10-30, 표시 전용).

밤 실행(run-screener.yml 'Macro scan')이 부른다. 점수 · PT-1 에 안 쓴다.
    PYTHONPATH=.:src python -m screener.macro_scan --out "$RUNNER_TEMP/web/macro.json" --sheet
    CI=1 PYTHONPATH=.:src python -m screener.macro_scan --out /tmp/macro.json   # 로컬 (시트 안 씀)

- 경제 지표 (FRED, 키 없이 fredgraph.csv): 실업률 Sahm 규칙(UNRATE) · 실업수당 청구 4주 평균(ICSA) ·
  장단기 금리차 10년−3개월(T10Y3M) · 하이일드 스프레드(BAMLH0A0HYM2, FRED 는 최근 약 3년만 준다)
- 시장 지표 (yfinance 배당 포함 가격): SPY 와 200일선 · 순환주(XLY · XLI · XLF · XLB) ÷ 방어주(XLP · XLU · XLV) 같은 비중
- 판정(좋음 · 주의 · 나쁨)은 널리 쓰는 기준을 그대로 옮긴 참고값 — 우리가 검증한 매매 신호가 아니다.
  rotation_v1: Sahm 규칙으로 순환 ↔ 방어를 갈아타도 SPY 와 같았다 (ΔSharpe −0.00, 1950–2025).
- 금리 결정: 일정은 data.rate_calendar, 결정(인상 · 동결 · 인하 · 폭)은 금리 숫자로 —
  FOMC = FRED DFEDTARU/DFEDTARL (발표 다음 날부터 새 범위), BoC = Valet V39079 (발표 다음 영업일부터)
"""

from __future__ import annotations

import argparse
import io
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.rate_calendar import (BOC_PAGE, FOMC_PAGE, boc_url, fomc_url, merge_boc, merge_fomc, parse_boc_page,
                                parse_fomc_page)

SHEET_TAB = "[경제]"
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (personal research; macro_scan)"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
BOC_VALET_URL = "https://www.bankofcanada.ca/valet/observations/V39079/json?start_date={start}"
RATE_START = "2020-12-01"           # 금리 결정 표는 2021년부터
CYCLICAL_ETFS = ["XLY", "XLI", "XLF", "XLB"]
DEFENSIVE_ETFS = ["XLP", "XLU", "XLV"]

# 판정 기준 (참고 — 널리 쓰는 값)
SAHM_WATCH, SAHM_BAD = 0.30, 0.50
CLAIMS_WATCH, CLAIMS_BAD = 0.10, 0.20          # 4주 평균이 1년 최저보다 이만큼 늘면
CURVE_WATCH = 0.25                             # 이보다 낮으면 주의, 0 아래(역전)면 나쁨
HY_WATCH, HY_BAD = 4.5, 6.0                    # 하이일드 스프레드 수준 (%)
HY_JUMP_WATCH, HY_JUMP_BAD = 0.5, 1.5          # 3달 사이 넓어진 폭 (%p)
SPY_BAD = -0.10                                # 200일선보다 10% 넘게 아래면 나쁨

STATUS_KO = {"good": "좋음", "watch": "주의", "bad": "나쁨"}


# ── 받기 ─────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 30, headers: dict[str, str] | None = UA) -> bytes:
    import requests

    last: Exception | None = None
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001 — 네트워크 실패는 몇 번 더 해 보고 넘긴다
            last = e
    raise RuntimeError(f"{url}: {last}")


def parse_fred_csv(raw: bytes | str) -> pd.Series:
    """fredgraph.csv → 날짜 인덱스 float (빈 값 '.' 은 뺀다)."""
    text = raw.decode() if isinstance(raw, bytes) else raw
    df = pd.read_csv(io.StringIO(text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna().sort_index()


def fetch_fred(sid: str) -> pd.Series:
    # FRED 는 브라우저 흉내 · 직접 만든 User-Agent 에 응답을 안 준다 (2026-10-30 확인) — requests 기본 헤더로
    return parse_fred_csv(_get(FRED_URL.format(sid=sid), timeout=20, headers=None))


def parse_boc_valet(raw: bytes | str) -> pd.Series:
    d = json.loads(raw)
    obs = [(o["d"], o.get("V39079", {}).get("v")) for o in d.get("observations", [])]
    s = pd.Series({pd.Timestamp(k): pd.to_numeric(v, errors="coerce") for k, v in obs}, dtype=float)
    return s.dropna().sort_index()


def fetch_prices(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    px = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)["Close"]
    return px.dropna(how="all")


# ── 지표 ─────────────────────────────────────────────────────────────────────

def _hist(s: pd.Series, n: int, rule: str | None = "W-FRI", nd: int = 3) -> list[list[Any]]:
    """작은 차트용 [[날짜, 값], ...] — rule 이 있으면 주마다 마지막 값."""
    x = s.dropna()
    if rule:
        x = x.resample(rule).last().dropna()
    return [[d.strftime("%Y-%m-%d"), round(float(v), nd)] for d, v in x.tail(n).items()]


def sahm_series(unrate: pd.Series) -> pd.Series:
    """Sahm 규칙 — 3달 평균 실업률 − 지난 12달 동안 3달 평균의 최저."""
    a3 = unrate.rolling(3).mean()
    return (a3 - a3.rolling(12).min().shift(1)).dropna()


def ind_sahm(unrate: pd.Series) -> dict[str, Any]:
    s = sahm_series(unrate)
    v = float(s.iloc[-1])
    status = "bad" if v >= SAHM_BAD else "watch" if v >= SAHM_WATCH else "good"
    return {"key": "sahm", "group": "economy", "date": s.index[-1].strftime("%Y-%m-%d"), "value": round(v, 2),
            "extra": {"unrate": round(float(unrate.iloc[-1]), 1), "max12": round(float(s.tail(12).max()), 2)},
            "status": status, "history": _hist(s, 48, rule=None, nd=2)}


def ind_claims(icsa: pd.Series) -> dict[str, Any]:
    a4 = icsa.rolling(4).mean().dropna()
    v = float(a4.iloc[-1])
    low = float(a4.tail(52).min())
    rise = v / low - 1 if low > 0 else 0.0
    status = "bad" if rise >= CLAIMS_BAD else "watch" if rise >= CLAIMS_WATCH else "good"
    return {"key": "claims", "group": "economy", "date": a4.index[-1].strftime("%Y-%m-%d"), "value": round(v),
            "extra": {"low52": round(low), "rise": round(rise, 4)}, "status": status,
            "history": _hist(a4, 104, rule=None, nd=0)}


def ind_curve(t10y3m: pd.Series) -> dict[str, Any]:
    v = float(t10y3m.iloc[-1])
    status = "bad" if v < 0 else "watch" if v < CURVE_WATCH else "good"
    last_year = t10y3m[t10y3m.index > t10y3m.index[-1] - pd.Timedelta(days=365)]
    inv = t10y3m[t10y3m < 0]
    return {"key": "curve", "group": "economy", "date": t10y3m.index[-1].strftime("%Y-%m-%d"), "value": round(v, 2),
            "extra": {"inverted_days_1y": int((last_year < 0).sum()),
                      "last_inverted": inv.index[-1].strftime("%Y-%m-%d") if len(inv) else None},
            "status": status, "history": _hist(t10y3m, 156, nd=2)}


def ind_hy(hy: pd.Series) -> dict[str, Any]:
    v = float(hy.iloc[-1])
    ago = hy[hy.index <= hy.index[-1] - pd.Timedelta(days=91)]
    prev = float(ago.iloc[-1]) if len(ago) else float(hy.iloc[0])
    jump = v - prev
    yr = hy[hy.index > hy.index[-1] - pd.Timedelta(days=365)]
    if v >= HY_BAD or jump >= HY_JUMP_BAD:
        status = "bad"
    elif v >= HY_WATCH or jump >= HY_JUMP_WATCH:
        status = "watch"
    else:
        status = "good"
    return {"key": "hy", "group": "economy", "date": hy.index[-1].strftime("%Y-%m-%d"), "value": round(v, 2),
            "extra": {"ago3m": round(prev, 2), "chg3m": round(jump, 2), "lo1y": round(float(yr.min()), 2),
                      "hi1y": round(float(yr.max()), 2)},
            "status": status, "history": _hist(hy, 156, nd=2)}


def ind_spy200(spy: pd.Series) -> dict[str, Any]:
    spy = spy.dropna()
    ma = spy.rolling(200).mean()
    gap = (spy / ma - 1).dropna()
    v = float(gap.iloc[-1])
    status = "good" if v >= 0 else "bad" if v <= SPY_BAD else "watch"
    hi = float(spy.tail(252).max())
    return {"key": "spy200", "group": "market", "date": spy.index[-1].strftime("%Y-%m-%d"), "value": round(v, 4),
            "extra": {"from_high": round(float(spy.iloc[-1]) / hi - 1, 4), "price": round(float(spy.iloc[-1]), 2)},
            "status": status, "history": _hist(gap, 104, nd=4)}


def cyc_def_ratio(px: pd.DataFrame) -> pd.Series:
    """순환주 ETF 같은 비중 ÷ 방어주 ETF 같은 비중 (매일 다시 맞춤) — 처음 = 1."""
    r = px.pct_change()
    cyc = r[[t for t in CYCLICAL_ETFS if t in r]].mean(axis=1)
    dfn = r[[t for t in DEFENSIVE_ETFS if t in r]].mean(axis=1)
    return ((1 + cyc.fillna(0)).cumprod() / (1 + dfn.fillna(0)).cumprod()).iloc[1:]


def ind_cycdef(px: pd.DataFrame) -> dict[str, Any]:
    ratio = cyc_def_ratio(px)
    chg = {k: round(float(ratio.iloc[-1] / ratio.iloc[-1 - n] - 1), 4) if len(ratio) > n else None
           for k, n in (("m1", 21), ("m3", 63), ("m6", 126))}
    vs200 = float(ratio.iloc[-1] / ratio.rolling(200).mean().iloc[-1] - 1)
    weekly = ratio.resample("W-FRI").last().dropna().tail(104)   # 차트는 2년 전 = 100
    return {"key": "cycdef", "group": "market", "date": ratio.index[-1].strftime("%Y-%m-%d"), "value": chg["m3"],
            "extra": {**chg, "vs200": round(vs200, 4), "lead": "cyclical" if vs200 >= 0 else "defensive"},
            "status": "good" if vs200 >= 0 else "watch", "history": _hist(weekly / weekly.iloc[0] * 100, 104, None, 2)}


# ── 금리 결정 ────────────────────────────────────────────────────────────────

def _decision(before: float, after: float) -> tuple[str, int]:
    bp = int(round((after - before) * 100))
    return ("hike" if bp > 0 else "cut" if bp < 0 else "hold"), bp


def fomc_decisions(meetings: list[tuple[str, str, bool]], upper: pd.Series, lower: pd.Series,
                   today: date) -> dict[str, Any]:
    """회의마다 결정 — 발표일 값 → 이틀 뒤 값. 숫자가 아직 없으면 예정(upcoming) · 결과 대기(pending)."""
    last = upper.index[-1].date() if len(upper) else date.min
    rows = []
    for start, day, sep in meetings:
        d = date.fromisoformat(day)
        row: dict[str, Any] = {"date": day, "start": start, "sep": sep, "url": fomc_url(day),
                               "decision": None, "change_bp": None, "upper": None, "lower": None}
        if last > d:
            before = float(upper.asof(pd.Timestamp(d)))
            after_u, after_l = float(upper.asof(pd.Timestamp(d + timedelta(days=2)))), float(
                lower.asof(pd.Timestamp(d + timedelta(days=2))))
            row["decision"], row["change_bp"] = _decision(before, after_u)
            row["upper"], row["lower"] = after_u, after_l
        else:
            row["decision"] = "upcoming" if d >= today else "pending"
        rows.append(row)
    nxt = next((r for r in rows if r["decision"] in ("upcoming", "pending") and r["date"] >= today.isoformat()), None)
    cur = {"upper": float(upper.iloc[-1]), "lower": float(lower.iloc[-1]), "date": last.isoformat()} if len(upper) else None
    return {"current": cur, "next": nxt, "meetings": rows}


def boc_decisions(ann: list[tuple[str, bool]], rate: pd.Series, today: date) -> dict[str, Any]:
    """발표마다 결정 — 발표일 값 → 4일 뒤까지의 마지막 값 (다음 영업일부터 바뀐다)."""
    last = rate.index[-1].date() if len(rate) else date.min
    rows = []
    for day, mpr in ann:
        d = date.fromisoformat(day)
        row: dict[str, Any] = {"date": day, "mpr": mpr, "url": boc_url(day), "decision": None, "change_bp": None,
                               "rate": None}
        if last > d:
            before = float(rate.asof(pd.Timestamp(d)))
            after = float(rate.asof(pd.Timestamp(d + timedelta(days=4))))
            row["decision"], row["change_bp"] = _decision(before, after)
            row["rate"] = after
        else:
            row["decision"] = "upcoming" if d >= today else "pending"
        rows.append(row)
    nxt = next((r for r in rows if r["decision"] in ("upcoming", "pending") and r["date"] >= today.isoformat()), None)
    cur = {"rate": float(rate.iloc[-1]), "date": last.isoformat()} if len(rate) else None
    return {"current": cur, "next": nxt, "meetings": rows}


# ── 모으기 ───────────────────────────────────────────────────────────────────

def _try(label: str, fn: Callable[[], Any], errors: list[str]) -> Any:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — 하나가 실패해도 나머지는 싣는다
        errors.append(f"{label}: {str(e)[:160]}")
        print(f"[경제] ⚠️ {label} 실패 — {e}")
        return None


def build(today: date | None = None) -> dict[str, Any]:
    """지표 6개 + FOMC · BoC 결정 → macro.json 내용."""
    today = today or datetime.now(ET).date()
    errors: list[str] = []
    fred = {sid: _try(f"FRED {sid}", lambda sid=sid: fetch_fred(sid), errors)
            for sid in ("UNRATE", "ICSA", "T10Y3M", "BAMLH0A0HYM2", "DFEDTARU", "DFEDTARL")}
    start = (pd.Timestamp(today) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    px = _try("yfinance ETF", lambda: fetch_prices(["SPY", *CYCLICAL_ETFS, *DEFENSIVE_ETFS], start), errors)

    indicators = []
    for label, fn, src in (("sahm", ind_sahm, fred["UNRATE"]), ("claims", ind_claims, fred["ICSA"]),
                           ("curve", ind_curve, fred["T10Y3M"]), ("hy", ind_hy, fred["BAMLH0A0HYM2"])):
        if src is not None and len(src):
            ind = _try(label, lambda fn=fn, src=src: fn(src), errors)
            if ind:
                indicators.append(ind)
    if px is not None and "SPY" in px:
        for label, fn, arg in (("spy200", ind_spy200, px["SPY"]), ("cycdef", ind_cycdef, px)):
            ind = _try(label, lambda fn=fn, arg=arg: fn(arg), errors)
            if ind:
                indicators.append(ind)

    fomc_page = _try("FOMC 일정 페이지", lambda: parse_fomc_page(_get(FOMC_PAGE).decode("utf-8", "replace")), errors)
    boc_page = _try("BoC 일정 페이지", lambda: parse_boc_page(_get(BOC_PAGE).decode("utf-8", "replace")), errors)
    rates: dict[str, Any] = {}
    if fred["DFEDTARU"] is not None and fred["DFEDTARL"] is not None:
        up, lo = fred["DFEDTARU"], fred["DFEDTARL"]
        rates["fomc"] = fomc_decisions(merge_fomc(fomc_page or []), up[up.index >= RATE_START],
                                       lo[lo.index >= RATE_START], today)
    boc_rate = _try("BoC Valet", lambda: parse_boc_valet(_get(BOC_VALET_URL.format(start=RATE_START))), errors)
    if boc_rate is not None and len(boc_rate):
        rates["boc"] = boc_decisions(merge_boc(boc_page or []), boc_rate, today)
    for k in ("fomc", "boc"):
        if k in rates and rates[k]["next"] is None:
            errors.append(f"{k}: 다음 일정이 목록에 없음 — data.rate_calendar 를 새 해 일정으로 고쳐야 함")
    return {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": today.isoformat(),
        "indicators": indicators,
        "rates": rates,
        "calendar_scraped": {"fomc": bool(fomc_page), "boc": bool(boc_page)},
        "errors": errors,
    }


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ── 시트 [경제] ──────────────────────────────────────────────────────────────

def _sgn(v: float, nd: int = 2, unit: str = "%p") -> str:
    return f"{v:+.{nd}f}{unit}".replace("-", "−")


def _pct(v: float | None, nd: int = 1) -> str:
    return "" if v is None else f"{v * 100:+.{nd}f}%".replace("-", "−")


def indicator_row(ind: dict[str, Any]) -> dict[str, str]:
    k, v, x = ind["key"], ind["value"], ind["extra"]
    status = STATUS_KO[ind["status"]]
    if k == "sahm":
        return {"구분": "경제 지표", "항목": "실업률 · Sahm 규칙", "값": f"실업률 {x['unrate']:.1f}% · Sahm {v:.2f}",
                "비교": f"12달 최고 {x['max12']:.2f}", "판정": status,
                "설명": "Sahm 0.5 이상 = 침체 신호 (보통 침체가 시작되고 3~5달 뒤 켜진다)"}
    if k == "claims":
        return {"구분": "경제 지표", "항목": "실업수당 청구 (4주 평균)", "값": f"{v / 10000:.1f}만 건",
                "비교": f"1년 최저 {x['low52'] / 10000:.1f}만 대비 {_pct(x['rise'])}", "판정": status,
                "설명": "1년 최저보다 10% 넘게 늘면 주의, 20% 넘으면 나쁨 (참고 기준)"}
    if k == "curve":
        return {"구분": "경제 지표", "항목": "장단기 금리차 (10년 − 3개월)", "값": _sgn(v),
                "비교": f"지난 1년 역전 {x['inverted_days_1y']}일", "판정": status,
                "설명": "0 아래(역전) = 침체 경고 — 2022년 말~2025년 역전이 이어졌지만 침체는 없었다"}
    if k == "hy":
        return {"구분": "경제 지표", "항목": "하이일드 스프레드", "값": f"{v:.2f}%",
                "비교": f"3달 전 {x['ago3m']:.2f}% ({_sgn(x['chg3m'])}) · 1년 {x['lo1y']:.2f}~{x['hi1y']:.2f}%",
                "판정": status, "설명": "넓어지면 돈줄이 마른다 — 4.5% 이상이거나 3달 새 +0.5%p 이상이면 주의"}
    if k == "spy200":
        return {"구분": "시장 지표", "항목": "S&P 500 (SPY) · 200일선", "값": f"200일선 {_pct(v)}",
                "비교": f"52주 고점 {_pct(x['from_high'])}", "판정": status,
                "설명": "200일선 아래 = 하락 추세 (lab_v1: 이걸로 피하면 낙폭은 절반, 수익은 SPY 보다 낮았다)"}
    return {"구분": "시장 지표", "항목": "순환주 ÷ 방어주", "값": f"3달 {_pct(x['m3'])}",
            "비교": f"1달 {_pct(x['m1'])} · 6달 {_pct(x['m6'])} · 200일선 {_pct(x['vs200'])}",
            "판정": "순환 우세" if x["lead"] == "cyclical" else "방어 우세",
            "설명": "순환 XLY · XLI · XLF · XLB ÷ 방어 XLP · XLU · XLV (rotation_v1: 이걸로 갈아타도 SPY 와 같았다)"}


DECISION_KO = {"hike": "인상", "hold": "동결", "cut": "인하", "upcoming": "예정", "pending": "결과 대기"}


def rate_rows(name: str, block: dict[str, Any], today: date) -> list[dict[str, str]]:
    """앞으로의 일정(가까운 순) → 지난 결정(최근 순)."""
    rows = []
    future = [m for m in block["meetings"] if m["decision"] in ("upcoming", "pending")]
    past = [m for m in block["meetings"] if m["decision"] not in ("upcoming", "pending")][::-1]
    cur = block.get("current") or {}
    for m in future + past:
        d = date.fromisoformat(m["date"])
        if name == "FOMC":
            level = f"{m['lower']:.2f}–{m['upper']:.2f}%" if m["upper"] is not None else ""
            note = "경제 전망(SEP) 같이" if m["sep"] else ""
            now = f"지금 {cur['lower']:.2f}–{cur['upper']:.2f}%" if cur else ""
        else:
            level = f"{m['rate']:.2f}%" if m["rate"] is not None else ""
            note = "통화정책보고서(MPR) 같이" if m["mpr"] else ""
            now = f"지금 {cur['rate']:.2f}%" if cur else ""
        if m["decision"] in ("upcoming", "pending"):
            change = f"D-{(d - today).days}" if d >= today else ""
            level = now
        else:
            change = "0" if m["change_bp"] == 0 else _sgn(m["change_bp"] / 100)
        rows.append({"구분": name, "항목": DECISION_KO[m["decision"]], "날짜": m["date"], "값": level, "비교": change,
                     "판정": note, "설명": m["url"] if m["decision"] not in ("upcoming", "pending") else ""})
    return rows


def sheet_frame(data: dict[str, Any]) -> pd.DataFrame:
    """시트 [경제] — 지표 6줄 → 빈 줄 → FOMC → 빈 줄 → BoC."""
    cols = ["구분", "항목", "날짜", "값", "비교", "판정", "설명"]
    today = date.fromisoformat(data["as_of"])
    rows: list[dict[str, str]] = [{**indicator_row(i), "날짜": i["date"]} for i in data.get("indicators", [])]
    for key, name in (("fomc", "FOMC"), ("boc", "BoC")):
        if key in data.get("rates", {}):
            rows.append({c: "" for c in cols})
            rows += rate_rows(name, data["rates"][key], today)
    return pd.DataFrame(rows, columns=cols).fillna("")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="경기 상태 · 금리 결정 (macro.json · 시트 [경제])")
    ap.add_argument("--out", type=Path, help="macro.json 경로")
    ap.add_argument("--sheet", action="store_true", help="시트 [경제] 탭에 쓰기")
    args = ap.parse_args(argv)

    data = _clean(build())
    for ind in data["indicators"]:
        print(f"[경제] {ind['key']}: {ind['value']} ({ind['status']}, {ind['date']})")
    for k, b in data["rates"].items():
        print(f"[경제] {k}: 지금 {b['current']} · 다음 {(b['next'] or {}).get('date')} · 회의 {len(b['meetings'])}개")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"[경제] {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    if args.sheet:
        from screener.exporter import export_to_google_sheet

        ok = export_to_google_sheet(sheet_frame(data), SHEET_TAB)
        print(f"[경제] 시트 {SHEET_TAB}: {'완료' if ok else '실패'}")
    if not data["indicators"] and not data["rates"]:
        print("::warning title=경제 데이터 없음::지표 · 금리 결정을 하나도 받지 못했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
