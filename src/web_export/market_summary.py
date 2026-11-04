"""웹 '투자자 한눈에' 숫자 (2026-11-03) — 전체분석(src/run_universe_scan.py)이 같이 만든다. 표시 전용 (점수 · 필터 · PT-1 무관).

- price_stats      종목 일봉 → 하루 수익률 · 52주 최저/최고 · 50/200일선 위 · 52주 신고가/신저가
- industry_medians 세부 업종별 밸류에이션 중앙값 (시총 $1B+, 적자 · 500배 넘는 값 제외)
- valuation_block / analyst_block  종목 상세 카드 (stocks/{티커}.json 의 detail)
- market_summary   오늘 탭 '시장 한눈에' · '실적 발표 일정' (market.json, 약 10KB)
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

import pandas as pd

SCHEMA_VERSION = 1
VAL_COLS = {  # 웹 키 → 종목 정보 칸 (screener.fundamentals, yfinance info)
    "pe": "fund_pe",
    "fpe": "fund_forward_pe",
    "peg": "fund_peg",
    "ps": "fund_ps",
    "pb": "fund_pb",
    "ev_ebitda": "fund_ev_ebitda",
}
PEER_MIN_CAP = 1e9      # 업종 중앙값에 넣는 시가총액 하한
PEER_MAX_VALUE = 500.0  # 이보다 큰 배수는 중앙값에서 뺀다 (이익이 거의 0 인 회사)
PEER_MIN_N = 3          # 값이 이보다 적으면 업종 중앙값 없음
MOVER_MIN_CAP = 2e9     # 많이 오른 · 내린 · 거래량 급증 목록의 시가총액 하한
MOVERS_N = 6
CAL_DAYS = 5            # 실적 발표 일정 — 거래일 수
CAL_PER_DAY = 8
CAL_BIG_CAP = 10e9


def _f(v: Any) -> float | None:
    """숫자로 (NaN · inf · 글자 → None)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _r(v: Any, digits: int) -> float | None:
    x = _f(v)
    return None if x is None else round(x, digits)


def price_stats(adj: pd.DataFrame, chart_daily: pd.DataFrame | None = None) -> dict[str, Any]:
    """한 종목 일봉 → 웹 숫자. adj = 배당 조정 OHLCV (수익률 · 이동평균선 · 신고가), chart_daily = 분할만 반영 일봉
    (52주 최저/최고 — 차트 · Yahoo 화면과 같은 값, 없으면 adj). 일봉이 모자라면 그 칸은 None."""
    out: dict[str, Any] = {"ret_1d": None, "above50": None, "above200": None, "new_high": None, "new_low": None,
                           "range_52w": None}
    if adj is None or adj.empty or "Close" not in adj.columns:
        return out
    close = adj["Close"].dropna()
    if len(close) >= 2 and close.iloc[-2] > 0:
        out["ret_1d"] = float(close.iloc[-1] / close.iloc[-2] - 1)
    if len(close) >= 50:
        out["above50"] = bool(close.iloc[-1] > close.iloc[-50:].mean())
    if len(close) >= 200:
        out["above200"] = bool(close.iloc[-1] > close.iloc[-200:].mean())
    hi = adj["High"].dropna().iloc[-252:] if "High" in adj.columns else pd.Series(dtype=float)
    lo = adj["Low"].dropna().iloc[-252:] if "Low" in adj.columns else pd.Series(dtype=float)
    if len(hi) > 20:
        out["new_high"] = bool(hi.iloc[-1] >= hi.iloc[:-1].max())
    if len(lo) > 20:
        out["new_low"] = bool(lo.iloc[-1] <= lo.iloc[:-1].min())
    src = chart_daily if chart_daily is not None and not chart_daily.empty and {"High", "Low"} <= set(chart_daily.columns) \
        else adj
    if {"High", "Low"} <= set(src.columns):
        h = src["High"].dropna().iloc[-252:]
        l_ = src["Low"].dropna().iloc[-252:]
        if len(h) and len(l_) and h.max() > 0:
            out["range_52w"] = {"low": round(float(l_.min()), 2), "high": round(float(h.max()), 2)}
    return out


def industry_medians(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """[{industry, cap, pe, fpe, …}] → {업종: {"n": 시총 $1B+ 종목 수, "pe": 중앙값 | None, …}}."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        ind = r.get("industry")
        cap = _f(r.get("cap"))
        if ind and cap is not None and cap >= PEER_MIN_CAP:
            groups[str(ind)].append(r)
    out: dict[str, dict[str, Any]] = {}
    for ind, members in groups.items():
        med: dict[str, Any] = {"n": len(members)}
        for k in VAL_COLS:
            xs = [x for x in (_f(m.get(k)) for m in members) if x is not None and 0 < x < PEER_MAX_VALUE]
            med[k] = round(statistics.median(xs), 2) if len(xs) >= PEER_MIN_N else None
        out[ind] = med
    return out


def valuation_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    """종목 정보 한 줄 → {pe, fpe, peg, ps, pb, ev_ebitda} (소수 2자리)."""
    return {k: _r(row.get(col), 2) for k, col in VAL_COLS.items()}


def valuation_block(row: Mapping[str, Any], industry: str | None,
                    medians: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """밸류에이션 카드 — 값이 하나도 없으면 None."""
    vals = valuation_values(row)
    if all(v is None for v in vals.values()):
        return None
    med = medians.get(industry or "", {})
    return {**vals, "ind": {k: med.get(k) for k in VAL_COLS}, "n_peers": int(med.get("n", 0)), "industry": industry}


def analyst_block(row: Mapping[str, Any], rec: Mapping[str, int] | None = None) -> dict[str, Any] | None:
    """애널리스트 카드 — 목표주가 · 의견(yfinance info) + 이번 달 의견 수(rec, stock_detail 캐시). 목표주가가 없으면 None."""
    mean = _r(row.get("fund_target_mean"), 2)
    if mean is None:
        return None
    n = _f(row.get("fund_analyst_n"))
    key = row.get("fund_rec_key")
    key = str(key) if isinstance(key, str) and key and key.lower() not in ("none", "nan") else None
    return {
        "n": int(n) if n is not None else None,
        "mean": mean,
        "median": _r(row.get("fund_target_median"), 2),
        "high": _r(row.get("fund_target_high"), 2),
        "low": _r(row.get("fund_target_low"), 2),
        "rec_mean": _r(row.get("fund_rec_mean"), 2),
        "rec_key": key,
        "rec": dict(rec) if rec else None,
    }


def trading_days_after(bar_date: str, n: int) -> list[str]:
    """bar_date 다음 평일 n 개 (휴장일은 모른다 — 그날은 발표가 거의 없어 빈 칸이 된다)."""
    d = datetime.strptime(bar_date, "%Y-%m-%d").date() + timedelta(days=1)
    out: list[str] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _cap(info: Mapping[str, Any]) -> float:
    return _f((info.get("fundamentals") or {}).get("market_cap")) or 0.0


def _mover(t: str, info: Mapping[str, Any]) -> dict[str, Any]:
    return {"t": t, "name": info.get("company"), "px": _r(info.get("price"), 2), "chg": _r(info.get("ret_1d"), 4),
            "vr": _r(info.get("vol_ratio"), 2), "cap": _cap(info), "sector": info.get("sector")}


def market_summary(items: Mapping[str, Mapping[str, Any]], sp500: Iterable[str],
                   breadth_flags: Mapping[str, Mapping[str, Any]], bar_date: str | None,
                   spy_ret_1d: float | None, generated_at: str) -> dict[str, Any] | None:
    """오늘 탭 market.json. items = 전체분석 종목 정보(TickerInfo, 두 시장), sp500 = 시장 폭에 쓸 S&P 500 티커,
    breadth_flags = {티커: price_stats 결과} (50/200일선 · 신고가/신저가 — 목록에는 안 싣는 칸). bar_date 가 없으면 None."""
    if not bar_date:
        return None
    sp = [t for t in sp500 if t in items and _f(items[t].get("ret_1d")) is not None]
    chg = [_f(items[t]["ret_1d"]) for t in sp]
    flags = [breadth_flags.get(t, {}) for t in sp]

    def share(key: str) -> float | None:
        xs = [f.get(key) for f in flags if f.get(key) is not None]
        return round(sum(1 for x in xs if x) / len(xs), 4) if xs else None

    breadth = {
        "n": len(sp),
        "adv": sum(1 for c in chg if c is not None and c > 0),
        "dec": sum(1 for c in chg if c is not None and c < 0),
        "above50": share("above50"),
        "above200": share("above200"),
        "new_high": sum(1 for f in flags if f.get("new_high")),
        "new_low": sum(1 for f in flags if f.get("new_low")),
        "spy": _r(spy_ret_1d, 4),
        "equal": round(statistics.mean(c for c in chg if c is not None), 4) if chg else None,
    }
    sp_set = set(sp500)
    pool = [t for t, v in items.items() if _f(v.get("ret_1d")) is not None and (t in sp_set or _cap(v) >= MOVER_MIN_CAP)]
    by_chg = sorted(pool, key=lambda t: _f(items[t]["ret_1d"]) or 0.0)
    surge = sorted((t for t in pool if _f(items[t].get("vol_ratio")) is not None),
                   key=lambda t: -(_f(items[t]["vol_ratio"]) or 0.0))

    by_sector: dict[str, list[float]] = defaultdict(list)
    for t, c in zip(sp, chg):
        sec = items[t].get("sector")
        if sec and sec != "Unknown" and c is not None:
            by_sector[str(sec)].append(c)
    sectors = sorted(({"sector": s, "chg": round(statistics.median(v), 4), "n": len(v)} for s, v in by_sector.items()),
                     key=lambda x: -x["chg"])

    calendar = []
    for day in trading_days_after(bar_date, CAL_DAYS):
        rows = sorted(((t, v) for t, v in items.items() if v.get("earnings_date") == day), key=lambda x: -_cap(x[1]))
        calendar.append({
            "date": day,
            "n": len(rows),
            "n_big": sum(1 for _, v in rows if _cap(v) >= CAL_BIG_CAP),
            "items": [{"t": t, "name": v.get("company"), "cap": _cap(v),
                       "timing": (v.get("earnings_move") or {}).get("timing"),
                       "move": _r((v.get("earnings_move") or {}).get("avg"), 4)} for t, v in rows[:CAL_PER_DAY]],
        })
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": generated_at,
        "bar_date": bar_date,
        "breadth": breadth,
        "min_cap": MOVER_MIN_CAP,
        "pool": len(pool),
        "gainers": [_mover(t, items[t]) for t in by_chg[::-1][:MOVERS_N]],
        "losers": [_mover(t, items[t]) for t in by_chg[:MOVERS_N]],
        "volume": [_mover(t, items[t]) for t in surge[:MOVERS_N]],
        "sectors": sectors,
        "calendar": calendar,
    }
