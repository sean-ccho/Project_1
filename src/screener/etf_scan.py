"""ETF 탭 데이터 — 묶음별 ETF 의 정보 · 수익률 · 추세 · 위험 (사이트 /quant/etf 의 etf.json · 시트 [ETF], 2026-10-28).

밤 실행(run-screener.yml 'ETF scan')이 부른다. 표시 전용 — 점수 · PT-1 에 안 쓴다.
    PYTHONPATH=.:src python -m screener.etf_scan --out "$RUNNER_TEMP/web/etf.json" \
        --detail-out "$RUNNER_TEMP/web/etf_detail.json" --sheet
    CI=1 PYTHONPATH=.:src python -m screener.etf_scan --out /tmp/etf.json --detail-out /tmp/etf_detail.json  # 로컬 (캐시 안 씀)

- 수익률 · 변동성 · 낙폭 · 상관 · 12개월 추세: 배당까지 넣은 가격(yfinance auto_adjust) — 총수익
- 가격 · 200일선 · 52주 고점 · 교차 라벨: TradingView 차트와 같은 분할만 반영 가격 (교차는 끝난 봉 — screener.chart_ma,
  주식 화면과 같은 규칙). 10-28 종가로 보니 배당 포함 가격이면 52주 고점이 19개 ETF 에서 1%p 넘게 어긋났다 (BITO · 채권)
- 12개월 추세: 1년 수익이 단기 국채 ETF(BIL) 1년 수익보다 높으면 살아 있음 (lab_B 추세추종 규칙의 신호, BIL 자신은 없음)
- 정보(이름 · 운용사 · 보수 · 운용자산 · 배당률 · 설정일): yfinance Ticker.info — 못 받으면 빈칸
- 상세(설명 · 상위 10개 보유 종목 · 섹터 비중 · 자산 구성 · 채권 등급, 2026-10-29): yfinance Ticker.funds_data → etf_detail.json
  (사이트가 줄을 펼칠 때만 받는다). Yahoo 채권 듀레이션 · 만기는 틀린 값이 있어(TLT 듀레이션 3.6) 안 싣는다
- 추천(2026-10-29): 역할별 기본 1개 = 같은 것을 담는 후보 중 운용자산 10억 이상에서 보수가 가장 싼 것 (근거 있는 기준),
  요즘 강한 ETF = 6달 수익 순 참고 목록 (검증 안 됨 — rotation_v1 에서 섹터 ETF 6달 모멘텀은 SPY 와 같았다)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.etf_universe import (BENCHMARK_ETF, ETF_GROUPS, ETF_ROLES, ETF_TWINS, MOMENTUM_GROUPS, MOMENTUM_TOP,
                               RISK_FREE_ETF, ROLE_FEE_TIE, ROLE_MIN_AUM, etf_group_of, etf_tickers, etf_twin_of)
from screener.chart_ma import ChartPrices, chart_prices, chart_view, cross_label, extend_chart, split_only_ohlcv
from screener.features import CROSS_KO

WINDOWS = {"w1": 5, "m1": 21, "m3": 63, "m6": 126, "y1": 252}   # 거래일
VOL_DAYS = 63
YEAR_DAYS = 252
SHEET_TAB = "ETF"
ET = ZoneInfo("America/Toronto")
# funds_data 섹터 키 → 주식 데이터와 같은 Yahoo 섹터 이름 (사이트가 한국어로 바꾼다)
SECTOR_KEYS = {
    "technology": "Technology", "financial_services": "Financial Services", "healthcare": "Healthcare",
    "consumer_cyclical": "Consumer Cyclical", "consumer_defensive": "Consumer Defensive",
    "communication_services": "Communication Services", "industrials": "Industrials", "energy": "Energy",
    "basic_materials": "Basic Materials", "utilities": "Utilities", "realestate": "Real Estate",
}
RATING_KEYS = ("aaa", "aa", "a", "bbb", "bb", "b", "below_b", "other")
STYLE_DOMINANT = 0.60        # 한 칸이 이 이상이면 그 칸, 아니면 '섞임'
STYLE_COVERED = 0.50         # 성격을 아는 종목 비중이 (전체 또는 상위 10개) 합의 이 이상일 때만
# N-PORT 가 없는 ETF(SPY 는 UIT, 캐나다 상장)는 같은 지수를 담는 쌍둥이의 전체 보유 종목을 빌린다 — 묶음 키끼리
LOOKTHROUGH_TWIN = {"ca_sp500": "sp500", "ca_nasdaq100": "nasdaq100"}


def _num(v: Any) -> float | None:
    """JSON 에 넣을 숫자 (NaN · inf · None → None)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _ret(close: pd.Series, n: int) -> float | None:
    c = close.dropna()
    return _num(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None


def _ytd(close: pd.Series) -> float | None:
    """올해 수익 — 작년 마지막 종가 대비 (올해 상장이면 None)."""
    c = close.dropna()
    if c.empty:
        return None
    prev = c[c.index.year < c.index[-1].year]
    return _num(c.iloc[-1] / prev.iloc[-1] - 1) if len(prev) else None


def _max_drawdown(close: pd.Series, days: int = YEAR_DAYS) -> float | None:
    c = close.dropna().tail(days)
    return _num((c / c.cummax() - 1).min()) if len(c) > 1 else None


def _ann_vol(close: pd.Series, days: int = VOL_DAYS) -> float | None:
    r = close.dropna().pct_change().dropna().tail(days)
    return _num(r.std() * math.sqrt(252)) if len(r) > 10 else None


def _corr(close: pd.Series, bench: pd.Series | None, days: int = YEAR_DAYS) -> float | None:
    """일간 수익 상관 (두 시장 모두 열린 날만)."""
    if bench is None:
        return None
    both = pd.concat({"a": close, "b": bench}, axis=1).dropna().pct_change().dropna().tail(days)
    return _num(both["a"].corr(both["b"])) if len(both) > 20 else None


def cross_labels(p: pd.DataFrame, chart: ChartPrices | None) -> dict[str, dict[str, Any]]:
    """일 · 주 · 월봉 교차 라벨 (주식 화면과 같은 규칙 — screener.features 와 같은 차트 바탕). 라벨이 없으면 label None."""
    as_of = p.index[-1]
    c_daily = pd.DataFrame()
    if chart is not None and not chart.daily.empty:
        if as_of > chart.daily.index[-1]:          # 전체 기록에 아직 없는 최근 날은 받은 일봉으로 채움
            chart = extend_chart(chart, split_only_ohlcv(p))
        c_daily, c_weekly, c_monthly = chart_view(chart, as_of)
    if c_daily.empty or c_daily.index[-1] < as_of:
        c_daily, c_weekly, c_monthly = chart_view(chart_prices(split_only_ohlcv(p)), as_of)
    out: dict[str, dict[str, Any]] = {}
    for tf, frame, code in (("daily", c_daily, "D"), ("weekly", c_weekly, "W"), ("monthly", c_monthly, "M")):
        if frame.empty:
            out[tf] = {"label": None, "gap": None}
            continue
        cl = cross_label(frame["Close"], code) if code == "D" else cross_label(frame["Close"], code, as_of, bars=True)
        out[tf] = {"label": CROSS_KO.get(cl.label), "gap": _num(cl.gap) if cl.label != "none" else None}
    return out


def fetch_info(tickers: list[str], pause: float = 0.2) -> dict[str, dict[str, Any]]:
    """ETF 정보 — 이름 · 운용사 · 분류 · 통화 · 보수(소수) · 운용자산 · 배당률(소수) · 설정일. 못 받은 칸은 None."""
    import yfinance as yf

    out: dict[str, dict[str, Any]] = {}
    for t in tickers:
        info: dict[str, Any] = {}
        for attempt in range(2):
            try:
                info = yf.Ticker(t).info or {}
                break
            except Exception:  # noqa: BLE001 — 한 종목 실패로 전체를 멈추지 않는다
                time.sleep(1.0 + attempt)
        exp = _num(info.get("netExpenseRatio"))          # yfinance 는 % 단위 (0.0945 = 0.0945%)
        inception = info.get("fundInceptionDate")
        out[t] = {
            "name": info.get("longName") or info.get("shortName"),
            "family": info.get("fundFamily"),
            "category": info.get("category"),
            "currency": info.get("currency"),
            "expense": exp / 100 if exp is not None else None,
            "aum": _num(info.get("totalAssets")),
            "yield": _num(info.get("yield")),
            "inception": (datetime.fromtimestamp(int(inception), tz=timezone.utc).date().isoformat()
                          if isinstance(inception, (int, float)) and inception > 0 else None),
        }
        time.sleep(pause)
    return out


def _text(v: Any) -> str | None:
    s = str(v).strip() if v is not None else ""
    return s if s and s.lower() not in ("nan", "none") else None


def parse_funds(fd: Any) -> dict[str, Any]:
    """yfinance FundsData → ETF 상세 한 개. 못 읽은 칸은 None (금 · 가상자산 ETF 는 보유 종목이 없다).

    holdings: 상위 10개 [{s 티커, n 이름, w 비중}], sectors: {Yahoo 섹터 이름: 비중}, assets: {stock, bond, cash, other},
    ratings: 채권 ETF(채권 50% 이상)만 신용등급 비중 + us_government(미국 국채 비중, 등급과 겹친다).
    """
    def get(name: str) -> Any:
        try:
            return getattr(fd, name)
        except Exception:  # noqa: BLE001 — 칸 하나 실패로 나머지를 버리지 않는다
            return None

    holdings: list[dict[str, Any]] = []
    top = get("top_holdings")
    if isinstance(top, pd.DataFrame) and not top.empty:
        for sym, r in top.iterrows():
            w = _num(r.get("Holding Percent"))
            if _text(sym) and w is not None and w > 0:
                holdings.append({"s": _text(sym), "n": _text(r.get("Name")), "w": w})
    sectors = {SECTOR_KEYS[k]: f for k, v in (get("sector_weightings") or {}).items()
               if k in SECTOR_KEYS and (f := _num(v)) is not None and f > 0}
    ac = get("asset_classes") or {}
    parts = {"stock": _num(ac.get("stockPosition")), "bond": _num(ac.get("bondPosition")),
             "cash": _num(ac.get("cashPosition")),
             "other": sum(_num(ac.get(k)) or 0.0 for k in ("preferredPosition", "convertiblePosition", "otherPosition"))}
    assets = {k: max(v or 0.0, 0.0) for k, v in parts.items()} if any(parts.values()) else None
    ratings = None
    if assets and assets["bond"] >= 0.5:
        br = get("bond_ratings") or {}
        got = {k: f for k in (*RATING_KEYS, "us_government") if (f := _num(br.get(k))) is not None}
        ratings = got if any(got.values()) else None
    return {"desc": _text(get("description")), "holdings": holdings, "sectors": sectors or None,
            "assets": assets, "ratings": ratings}


def fetch_holdings(tickers: list[str], pause: float = 0.2) -> dict[str, dict[str, Any]]:
    """ETF 상세 (설명 · 상위 10개 · 섹터 · 자산 구성 · 채권 등급). 두 번 다 못 받은 ETF 는 빠진다."""
    import yfinance as yf

    out: dict[str, dict[str, Any]] = {}
    for t in tickers:
        for attempt in range(2):
            try:
                fd = yf.Ticker(t).funds_data
                fd.description                              # 여기서 받아 온다 — 실패하면 다시
                out[t] = parse_funds(fd)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.0 + attempt)
        time.sleep(pause)
    return out


def holding_styles(holdings: dict[str, dict[str, Any]], extra: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """보유 종목(상위 10개 + extra = 전체 보유 종목 티커) → 4칸 성격 (전체분석 universe-data 의 업종 + GEO_REVENUE_FILE 의
    SEC 미국 매출). 못 받으면 {}."""
    import gzip
    import urllib.request

    from data.geo_revenue import UNIVERSE_BASE
    from data.industry_style import style_for

    syms = {h["s"] for d in holdings.values() for h in d.get("holdings") or []} | (extra or set())
    industry: dict[str, str | None] = {}
    try:
        for key in ("sp500", "nasdaq"):
            with urllib.request.urlopen(f"{UNIVERSE_BASE}/{key}.json.gz", timeout=60) as r:
                for t, v in json.loads(gzip.decompress(r.read())).get("items", {}).items():
                    industry.setdefault(t, v.get("industry"))
    except Exception as exc:  # noqa: BLE001 — 표시용, 없으면 성격 칸만 빈다
        print(f"[ETF] 전체분석 업종 못 받음 — 성격 칸 건너뜀: {exc}")
        return {}
    out = {}
    for t in syms:
        key = t.upper().replace(".", "-")
        if key in industry and (st := style_for(key, industry[key])):
            out[t] = st
    return out


def _mix_of(pairs: list[tuple[str, float]], styles: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """(티커, 비중) → {mix: {'cyclical_domestic': 0.4, …}, top: 칸 · 'mixed', covered: 아는 비중}.
    성격을 아는 종목이 합의 절반이 안 되면 None (해외 · 채권 · 금 ETF)."""
    total = sum(w for _t, w in pairs)
    mix: dict[str, float] = {}
    for t, w in pairs:
        st = styles.get(t)
        if st and st.get("cycle") and st.get("region"):
            k = f"{st['cycle']}_{st['region']}"
            mix[k] = mix.get(k, 0.0) + w
    known = sum(mix.values())
    if not total or known < STYLE_COVERED * total:
        return None
    shares = {k: v / known for k, v in sorted(mix.items(), key=lambda kv: -kv[1])}
    top = next(iter(shares))
    return {"mix": shares, "top": top if shares[top] >= STYLE_DOMINANT else "mixed", "covered": known / total}


def style_mix(detail: dict[str, Any] | None, styles: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """상위 10개 보유 종목(Yahoo)의 4칸 비중 — 전체 보유 종목(N-PORT)이 없는 ETF 용. basis = 'top10'."""
    out = _mix_of([(h["s"], h["w"]) for h in (detail or {}).get("holdings") or []], styles)
    return {**out, "basis": "top10"} if out else None


def load_full_holdings(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """ETF_HOLDINGS_FILE (data.etf_holdings 의 etf_holdings.csv) → {ETF: {rows: [(티커 · 못 찾으면 '', 비중)], as_of}}.
    없거나 못 읽으면 {} — 그러면 성격은 상위 10개로."""
    import csv

    if not path or not Path(path).exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with Path(path).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                e = out.setdefault(r["etf"], {"rows": [], "as_of": r.get("rep_date") or None})
                e["rows"].append((r.get("ticker") or "", float(r["weight"])))
    except Exception as exc:  # noqa: BLE001 — 표시용
        print(f"[ETF] 전체 보유 종목 파일 못 읽음 — 상위 10개로: {exc}")
        return {}
    return out


def fill_twins(full: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """같은 지수 쌍둥이(ETF_TWINS 의 same + LOOKTHROUGH_TWIN)에 전체 보유 종목이 없으면 있는 쪽 것을 빌린다 (from = 빌려준 ETF)."""
    same = {key: ts for key, _ko, _en, kind, ts in ETF_TWINS if kind == "same"}
    out = dict(full)
    for key, ts in same.items():
        donors = [t for t in ts + same.get(LOOKTHROUGH_TWIN.get(key, ""), []) if t in full]
        if not donors:
            continue
        for t in ts:
            if t not in out:
                out[t] = {**full[donors[0]], "from": donors[0]}
    return out


def style_mix_full(entry: dict[str, Any] | None, styles: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """전체 보유 종목(N-PORT)의 4칸 비중. basis = 'full', as_of = 공시 기준일, from = 빌려 온 쌍둥이 ETF."""
    if not entry:
        return None
    out = _mix_of([(t, w) for t, w in entry["rows"]], styles)
    if not out:
        return None
    return {**out, "basis": "full", "as_of": entry.get("as_of"), **({"from": entry["from"]} if entry.get("from") else {})}


def build_rows(prices: pd.DataFrame, chart_map: dict[str, ChartPrices], info: dict[str, dict[str, Any]],
               tickers: list[str] | None = None,
               holdings: dict[str, dict[str, Any]] | None = None,
               styles: dict[str, dict[str, Any]] | None = None,
               full: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """ETF 한 줄씩. prices = fetch_ohlcv 결과 ((티커, 필드) 멀티인덱스, 배당 조정 + Dividends 칸).
    full = 전체 보유 종목 (load_full_holdings + fill_twins) — 있는 ETF 는 성격을 그걸로, 없으면 상위 10개로."""
    group_of = etf_group_of()
    have = set(prices.columns.get_level_values(0))
    closes = {t: prices[t]["Close"].dropna() for t in have}
    rf_y1 = _ret(closes[RISK_FREE_ETF], YEAR_DAYS) if RISK_FREE_ETF in closes else None
    bench = closes.get(BENCHMARK_ETF)
    rows: list[dict[str, Any]] = []
    for t in tickers or etf_tickers():
        if t not in have:
            continue
        p = prices[t].dropna(subset=["Close"])
        if p.empty:
            continue
        c = p["Close"]
        y1 = _ret(c, YEAR_DAYS)
        split_close = split_only_ohlcv(p)["Close"].dropna()
        chart = split_close if len(split_close) else c.dropna()   # 차트 가격 (분할만 반영)
        last = float(chart.iloc[-1])
        sma200 = float(chart.tail(200).mean()) if len(chart) >= 200 else math.nan
        held = ((holdings or {}).get(t) or {}).get("holdings") or []
        rows.append({
            "ticker": t,
            "group": group_of.get(t, ""),
            **{k: None for k in ("name", "family", "category", "currency", "expense", "aum", "yield", "inception")},
            **info.get(t, {}),
            "date": c.index[-1].date().isoformat(),
            "price": _num(last),
            "ret": {**{k: _ret(c, n) for k, n in WINDOWS.items()}, "ytd": _ytd(c)},
            "above_200": bool(last > sma200) if math.isfinite(sma200) else None,
            # BIL 은 자기와 비교라 늘 꺼짐 → 없음
            "trend_12m": None if y1 is None or t == RISK_FREE_ETF else bool(y1 - (rf_y1 or 0.0) > 0),
            "high_52w_gap": _num(last / chart.tail(YEAR_DAYS).max() - 1),
            "vol": _ann_vol(c),
            "mdd_1y": _max_drawdown(c),
            "corr_spy": 1.0 if t == BENCHMARK_ETF else _corr(c, bench),
            "cross": cross_labels(p, chart_map.get(t)),
            "top10_weight": _num(sum(h["w"] for h in held)) if held else None,   # 상위 10개 합 (쏠림 정도)
            # 경기순환/방어 × 내수/글로벌 — 전체 보유 종목이 있으면 그걸로 (아는 비중이 절반 안 되면 None, 상위 10개로 안 돌아감)
            "style_mix": (style_mix_full(full[t], styles or {}) if full and t in full
                          else style_mix((holdings or {}).get(t), styles or {})),
        })
    return rows


def pick_roles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """역할별 기본 1개 — 후보 중 운용자산 ROLE_MIN_AUM 이상 · 보수 아는 것에서 보수 최저 (차이가 아주 작으면 운용자산 큰 쪽).

    고를 것이 없는 역할은 빠진다. alts = 나머지 후보 (보수 싼 순, 모르면 끝).
    """
    by = {r["ticker"]: r for r in rows}
    out: list[dict[str, Any]] = []
    for key, ko, en, cands in ETF_ROLES:
        have = [by[t] for t in cands if t in by]
        ok = [r for r in have if r.get("expense") is not None and (r.get("aum") or 0) >= ROLE_MIN_AUM]
        if not ok:
            continue
        low = min(r["expense"] for r in ok)
        pick = max((r for r in ok if r["expense"] - low < ROLE_FEE_TIE), key=lambda r: r["aum"])["ticker"]
        alts = sorted((r for r in have if r["ticker"] != pick),
                      key=lambda r: (r.get("expense") is None, r.get("expense") or 0.0))
        out.append({"key": key, "ko": ko, "en": en, "pick": pick, "alts": [r["ticker"] for r in alts]})
    return out


def momentum_list(rows: list[dict[str, Any]]) -> list[str]:
    """요즘 강한 ETF (참고) — 섹터 · 테마 · 나라 · 원자재 · 부동산 중 200일선 위 + 12개월 추세 살아 있는 것을 6달 수익 순으로.

    같은 것 묶음(SMH · SOXX 등)에서는 하나만. 앞으로 오른다는 근거는 없다 (표시 전용).
    """
    twin_of = etf_twin_of()
    cands = sorted((r for r in rows if r["group"] in MOMENTUM_GROUPS and r.get("above_200") is True
                    and r.get("trend_12m") is True and r["ret"].get("m6") is not None),
                   key=lambda r: r["ret"]["m6"], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for r in cands:
        family = twin_of.get(r["ticker"], r["ticker"])
        if family in seen:
            continue
        seen.add(family)
        out.append(r["ticker"])
        if len(out) == MOMENTUM_TOP:
            break
    return out


def _rounded(obj: Any) -> Any:
    """JSON 을 줄이려고 소수는 6자리까지 (0.0034999999999999996 → 0.0035)."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v) for v in obj]
    return obj


def payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """etf.json 내용."""
    present = {r["ticker"] for r in rows}
    picks = {"roles": pick_roles(rows), "momentum": momentum_list(rows)}
    rows = _rounded(rows)
    twins = [{"key": k, "ko": ko, "en": en, "kind": kind, "tickers": have}
             for k, ko, en, kind, ts in ETF_TWINS if len(have := [t for t in ts if t in present]) >= 2]
    return {
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "as_of": max((r["date"] for r in rows), default=None),
        "risk_free": RISK_FREE_ETF,
        "benchmark": BENCHMARK_ETF,
        "groups": [{"key": k, "ko": ko, "en": en, "tickers": [t for t in ts if t in present]}
                   for k, ko, en, ts in ETF_GROUPS],
        "twins": twins,
        "picks": picks,
        "items": rows,
    }


def detail_payload(holdings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """etf_detail.json 내용 — 티커 → 상세 (사이트 /api/etf/[티커] 가 줄을 펼칠 때 읽는다)."""
    return {
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "items": {t: _rounded(d) for t, d in holdings.items()},
    }


def _pct(v: Any, sign: bool = True) -> str:
    f = _num(v)
    return "" if f is None else (f"{f:+.1%}" if sign else f"{f:.1%}")


def _cross_text(c: dict[str, Any]) -> str:
    if not c.get("label"):
        return ""
    return f"{c['label']} {_pct(c.get('gap'))}".strip()


def _money(v: Any) -> str:
    f = _num(v)
    if f is None:
        return ""
    for unit, div in (("조", 1e12), ("억", 1e8)):
        if f >= div:
            return f"{f / div:,.1f}{unit}"
    return f"{f:,.0f}"


def _top_text(detail: dict[str, Any] | None, n: int = 3) -> str:
    """상위 보유 'NVDA 8.4% · AAPL 7.4% · MSFT 5.8%'."""
    return " · ".join(f"{h['s']} {h['w']:.1%}" for h in ((detail or {}).get("holdings") or [])[:n])


def _mix_text(mix: dict[str, Any] | None) -> str:
    """'경기방어·글로벌 64%' · '섞임 (경기순환·글로벌 45%)'."""
    from data.industry_style import CYCLE_KO, REGION_KO

    if not mix:
        return ""
    name = lambda k: f"{CYCLE_KO[k.split('_')[0]]}·{REGION_KO[k.split('_')[1]]}"  # noqa: E731
    first, share = next(iter(mix["mix"].items()))
    text = f"{name(first)} {share:.0%}" if mix["top"] != "mixed" else f"섞임 ({name(first)} {share:.0%})"
    if mix.get("basis") == "full":
        return f"{text} · 전체 보유 {mix.get('as_of') or ''}{' (' + mix['from'] + ' 기준)' if mix.get('from') else ''}"
    return f"{text} · 상위 10개"


def sheet_frame(rows: list[dict[str, Any]], holdings: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    """시트 [ETF] — 묶음 순서대로, 퍼센트는 글자로. 추천 · 같은 것 · 상위 보유는 맨 끝 칸 (2026-10-29)."""
    names = {k: ko for k, ko, _en, _t in ETF_GROUPS}
    role_of = {p["pick"]: f"⭐ {p['ko']}" for p in pick_roles(rows)}
    role_of.update({t: f"요즘 강한 {i}위 (참고)" for i, t in enumerate(momentum_list(rows), 1) if t not in role_of})
    present = {r["ticker"] for r in rows}
    twins = {t: [x for x in ts if x != t and x in present] for _k, _ko, _en, _kind, ts in ETF_TWINS for t in ts}
    out = []
    for r in rows:
        ret = r["ret"]
        out.append({
            "묶음": names.get(r["group"], r["group"]), "티커": r["ticker"], "이름": r.get("name") or "",
            "운용사": r.get("family") or "", "가격": r["price"], "통화": r.get("currency") or "",
            "1주": _pct(ret["w1"]), "1달": _pct(ret["m1"]), "3달": _pct(ret["m3"]), "6달": _pct(ret["m6"]),
            "1년": _pct(ret["y1"]), "올해": _pct(ret["ytd"]),
            "200일선": {True: "위", False: "아래"}.get(r["above_200"], ""),
            "12개월 추세": {True: "살아 있음", False: "꺼짐"}.get(r["trend_12m"], ""),
            "교차(일)": _cross_text(r["cross"]["daily"]), "교차(주)": _cross_text(r["cross"]["weekly"]),
            "교차(월)": _cross_text(r["cross"]["monthly"]),
            "52주 고점 대비": _pct(r["high_52w_gap"]), "변동성(63일)": _pct(r["vol"], sign=False),
            "1년 최대낙폭": _pct(r["mdd_1y"]), "SPY 상관(1년)": "" if r["corr_spy"] is None else f"{r['corr_spy']:.2f}",
            "보수": _pct(r.get("expense"), sign=False).replace("%", "") + "%" if r.get("expense") is not None else "",
            "운용자산": _money(r.get("aum")), "배당률": _pct(r.get("yield"), sign=False),
            "설정일": r.get("inception") or "", "기준일": r["date"],
            "추천": role_of.get(r["ticker"], ""), "같은 것": " · ".join(twins.get(r["ticker"], [])),
            "상위 보유": _top_text((holdings or {}).get(r["ticker"])),
            "성격": _mix_text(r.get("style_mix")),
        })
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    from data.fetch import fetch_chart_prices, fetch_ohlcv

    ap = argparse.ArgumentParser(description="ETF 탭 데이터 (etf.json · 시트 [ETF])")
    ap.add_argument("--out", type=Path, help="etf.json 경로")
    ap.add_argument("--detail-out", type=Path, help="etf_detail.json 경로 (설명 · 보유 종목 · 섹터 비중)")
    ap.add_argument("--sheet", action="store_true", help="구글 시트 [ETF] 탭에도 쓴다")
    args = ap.parse_args(argv)

    t0 = time.time()
    tickers = etf_tickers()
    prices = fetch_ohlcv(tickers, period="5y")
    chart_map = fetch_chart_prices(list(prices.columns.levels[0]))
    info = fetch_info(tickers)
    holdings = fetch_holdings(tickers)
    full = fill_twins(load_full_holdings(os.environ.get("ETF_HOLDINGS_FILE")))
    styles = holding_styles(holdings, {t for e in full.values() for t, _w in e["rows"] if t})
    rows = build_rows(prices, chart_map, info, tickers, holdings, styles, full)
    by_basis = [(r.get("style_mix") or {}).get("basis") for r in rows]
    print(f"[ETF] 성격: 전체 보유 {by_basis.count('full')}개 · 상위 10개 {by_basis.count('top10')}개 · "
          f"없음 {by_basis.count(None)}개 (전체 보유 파일 ETF {len(full)}개)")
    data = payload(rows)
    missing = [t for t in tickers if t not in {r["ticker"] for r in rows}]
    print(f"[ETF] {len(rows)}/{len(tickers)}개 · 기준일 {data['as_of']} · 정보 없음 "
          f"{sum(1 for r in rows if not r.get('name'))}개 · 가격 없음 {missing} · 상세 {len(holdings)}개 "
          f"(보유 종목 {sum(1 for d in holdings.values() if d['holdings'])}) · {time.time() - t0:.0f}초")
    print(f"[ETF] 역할별 기본: {' · '.join(p['key'] + '=' + p['pick'] for p in data['picks']['roles'])} · "
          f"요즘 강한: {' · '.join(data['picks']['momentum'])}")
    for path, body in ((args.out, data), (args.detail_out, detail_payload(holdings) if holdings else None)):
        if path and body:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
            print(f"[ETF] {path} ({path.stat().st_size // 1024} KB)")
    if args.sheet:
        from screener.exporter import export_to_google_sheet

        ok = export_to_google_sheet(sheet_frame(rows, holdings), SHEET_TAB)
        print(f"[ETF] 시트 [{SHEET_TAB}] {'업데이트 완료' if ok else '건너뜀'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
