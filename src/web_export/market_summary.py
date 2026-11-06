"""웹 '투자자 한눈에' 숫자 (2026-11-03) — 전체분석(src/run_universe_scan.py)이 같이 만든다. 표시 전용 (점수 · 필터 · PT-1 무관).

- price_stats      종목 일봉 → 하루 수익률 · 52주 최저/최고 · 50/200일선 위 · 52주 신고가/신저가
- industry_medians 세부 업종별 밸류에이션 중앙값 (시총 $1B+, 적자 · 500배 넘는 값 제외)
- valuation_block / analyst_block  종목 상세 카드 (stocks/{티커}.json 의 detail)
- market_summary   오늘 탭 '시장 한눈에' · '실적 발표 일정' (market.json, 약 10KB) — 미국(universe-data)과
                   캐나다(canada-data, 2026-11-06 — TSX 시총 C$1B 이상으로 시장 폭, 지수는 S&P/TSX) 각각 한 파일
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

import pandas as pd

from screener.chart_ma import ChartPrices, extend_chart, split_only_ohlcv

SCHEMA_VERSION = 1
# 목록 행(universe.item_info)에도 실리는 price_stats 표시 — market.json 에 적어 두면 사이트가 숫자를 목록 링크로 켠다 (2026-11-04)
ROW_FLAGS = ("new_high", "new_low")
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
# 캐나다 (2026-11-06) — TSXV 동전주를 넣으면 신저가 · 하락 수가 숫자를 흐려서(11-05 신저가 331개 중 273개가 TSXV)
# 시장 폭 · 섹터는 TSX 시총 C$1B 이상(약 310종목 — S&P/TSX 종합지수 규모)만, 목록은 두 거래소 C$1B 이상
CA_MIN_CAP = 1e9
CA_INDEX = ("^GSPTSE", "XIC.TO")                       # 앞 기호부터 — 지수를 못 받으면 그 지수를 따르는 ETF
INDEX_LABEL = {"^GSPTSE": "S&P/TSX", "XIC.TO": "S&P/TSX (XIC)", "SPY": "SPY"}
# 오늘 탭 '다가오는 큰 일정'의 대형주 실적 (2026-11-06) — 그 시장 시총 상위(같은 회사 종류주는 하나) 중 이 기간 안에 발표
MEGA_N = 15
MEGA_DAYS = 60


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


def price_round(v: float) -> float:
    """주가 반올림 — 1달러 이상은 소수 2자리, 미만은 4자리 (0.115 같은 동전주가 0.12 로 뭉개지지 않게 · 2026-11-05)."""
    return round(float(v), 2 if abs(v) >= 1 else 4)


def chart_daily_as_of(adj: pd.DataFrame, chart: ChartPrices | None) -> pd.DataFrame:
    """adj 마지막 날까지의 차트 일봉 (분할만 반영 = TradingView · Yahoo 화면 가격) — features 차트 패턴과 같은 방법.

    - 장중에 돌면 차트 일봉(period="max")에만 그날 장중 봉이 들어올 수 있어서 adj 마지막 날까지 자른다.
    - 전체 기록은 가장 최근 봉이 하루 늦게 들어오기도 해서, 모자란 날은 받은 일봉을 분할만 반영으로 되돌려 채운다 (extend_chart).
    - 차트 가격이 없으면 받은 일봉을 되돌린 것 (split_only_ohlcv).
    """
    as_of = adj.index[-1]
    if chart is None or chart.daily.empty:
        return split_only_ohlcv(adj)
    if as_of > chart.daily.index[-1]:
        chart = extend_chart(chart, split_only_ohlcv(adj))
    daily = chart.daily.loc[:as_of]
    return daily if not daily.empty else split_only_ohlcv(adj)


def price_stats(adj: pd.DataFrame, chart: ChartPrices | None = None) -> dict[str, Any]:
    """한 종목 일봉 → 웹 숫자. adj = 배당 조정 OHLCV + Dividends (수익률 · 이동평균선), chart = 분할만 반영 차트 가격.

    52주 최저/최고 · 신고가/신저가는 차트 가격(chart_daily_as_of — 차트 · Yahoo 화면과 같은 값)으로 본다.
    신고가 = 마지막 봉 고가 ≥ 직전 251봉 최고 (신저가는 저가 ≤ 최저, 같으면 포함) — 2026-11-04 부터 차트 가격
    (전에는 배당 조정 가격이라 배당 많은 종목이 차트로는 신고가가 아닌데 신고가로 잡혔다). 일봉이 모자라면 그 칸은 None.
    """
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
    daily = chart_daily_as_of(adj, chart)
    if {"High", "Low"} <= set(daily.columns):
        h = daily["High"].dropna().iloc[-252:]
        l_ = daily["Low"].dropna().iloc[-252:]
        if len(h) > 20:
            out["new_high"] = bool(h.iloc[-1] >= h.iloc[:-1].max())
        if len(l_) > 20:
            out["new_low"] = bool(l_.iloc[-1] <= l_.iloc[:-1].min())
        if len(h) and len(l_) and h.max() > 0:
            out["range_52w"] = {"low": price_round(float(l_.min())), "high": price_round(float(h.max()))}
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


def item_cap(info: Mapping[str, Any]) -> float:
    """웹 종목 정보(TickerInfo)의 시가총액 — 없으면 0 (그 시장 통화)."""
    return _cap(info)


def day_return(close: pd.Series, bar_date: str | None) -> float | None:
    """bar_date 종가 ÷ 전 거래일 종가 − 1. bar_date 봉이 없으면 None (하루 늦은 지수를 다른 날 값으로 쓰지 않게)."""
    if not bar_date or close is None or close.empty:
        return None
    s = close.dropna()
    s = s[~s.index.duplicated(keep="last")]
    days = [str(pd.Timestamp(d).date()) for d in s.index]
    if bar_date not in days:
        return None
    i = days.index(bar_date)
    if i == 0 or not s.iloc[i - 1] > 0:
        return None
    return _f(s.iloc[i] / s.iloc[i - 1] - 1)


def _mover(t: str, info: Mapping[str, Any]) -> dict[str, Any]:
    return {"t": t, "name": info.get("company"), "px": _r(info.get("price"), 2), "chg": _r(info.get("ret_1d"), 4),
            "vr": _r(info.get("vol_ratio"), 2), "cap": _cap(info), "sector": info.get("sector")}


def mega_earnings(items: Mapping[str, Mapping[str, Any]], bar_date: str) -> list[dict[str, Any]]:
    """시총 상위 MEGA_N 회사(같은 회사 이름은 한 번 — GOOGL · GOOG) 중 bar_date 다음 날부터 MEGA_DAYS 일 안에 실적 발표 →
    날짜 순 [{t, name, cap, date, timing}]. 실적일은 Yahoo 예정일이라 회사가 바꾸면 하루 늦게 반영된다."""
    seen: set[str] = set()
    tops: list[tuple[str, Mapping[str, Any]]] = []
    for t, v in sorted(items.items(), key=lambda x: -_cap(x[1])):
        # 같은 회사 종류주 — 이름(GOOGL · GOOG 'Alphabet Inc.')이나 티커 뿌리(BRK-A 'Berkshire Hathaway Inc.' ·
        # BRK-B '… Inc. New', CCL-A.TO · CCL-B.TO)가 같으면 하나만
        name = re.sub(r"[^a-z0-9 ]", "", str(v.get("company") or t).lower()).strip()
        root = t.split("-")[0] if "-" in t else t
        if name in seen or root in seen or _cap(v) <= 0:
            continue
        seen.update({name, root})
        tops.append((t, v))
        if len(tops) >= MEGA_N:
            break
    limit = (datetime.strptime(bar_date, "%Y-%m-%d").date() + timedelta(days=MEGA_DAYS)).isoformat()
    rows = [{"t": t, "name": v.get("company"), "cap": _cap(v), "date": str(v["earnings_date"])[:10],
             "timing": (v.get("earnings_move") or {}).get("timing")}
            for t, v in tops if v.get("earnings_date") and bar_date < str(v["earnings_date"])[:10] <= limit]
    return sorted(rows, key=lambda r: (r["date"], -r["cap"]))


def market_summary(items: Mapping[str, Mapping[str, Any]], sp500: Iterable[str],
                   breadth_flags: Mapping[str, Mapping[str, Any]], bar_date: str | None,
                   spy_ret_1d: float | None, generated_at: str, *, market: str = "us", index_label: str = "SPY",
                   min_cap: float = MOVER_MIN_CAP) -> dict[str, Any] | None:
    """오늘 탭 market.json. items = 전체분석 종목 정보(TickerInfo), sp500 = 시장 폭 · 섹터에 쓸 티커(미국 = S&P 500,
    캐나다 = TSX 시총 C$1B 이상), breadth_flags = {티커: price_stats 결과} (50/200일선 · 신고가/신저가).
    신고가/신저가는 목록 행에도 같은 표시가 실려서 (ROW_FLAGS → row_flags) 사이트가 숫자를 누르면 그 종목들을 거른다.
    spy_ret_1d = 그 시장 지수 하루 수익률 (breadth.spy — 이름은 예전 그대로, 무슨 지수인지는 index_label),
    min_cap = 많이 오른 · 내린 · 거래량 급증 목록 하한 (그 시장 통화). bar_date 가 없으면 None."""
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
    pool = [t for t, v in items.items() if _f(v.get("ret_1d")) is not None and (t in sp_set or _cap(v) >= min_cap)]
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
        "mega": mega_earnings(items, bar_date),
        "market": market,
        "index_label": index_label,
        "bar_date": bar_date,
        "breadth": breadth,
        "row_flags": list(ROW_FLAGS),
        "min_cap": min_cap,
        "pool": len(pool),
        "gainers": [_mover(t, items[t]) for t in by_chg[::-1][:MOVERS_N]],
        "losers": [_mover(t, items[t]) for t in by_chg[:MOVERS_N]],
        "volume": [_mover(t, items[t]) for t in surge[:MOVERS_N]],
        "sectors": sectors,
        "calendar": calendar,
    }
