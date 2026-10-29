"""ETF 탭 데이터 — 묶음별 ETF 의 정보 · 수익률 · 추세 · 위험 (사이트 /quant/etf 의 etf.json · 시트 [ETF], 2026-10-28).

밤 실행(run-screener.yml 'ETF scan')이 부른다. 표시 전용 — 점수 · PT-1 에 안 쓴다.
    PYTHONPATH=.:src python -m screener.etf_scan --out "$RUNNER_TEMP/web/etf.json" --sheet
    CI=1 PYTHONPATH=.:src python -m screener.etf_scan --out /tmp/etf.json        # 로컬 미리보기 (캐시 안 씀)

- 수익률 · 변동성 · 낙폭 · 상관: 배당까지 넣은 가격(yfinance auto_adjust) — 총수익
- 가격 · 교차 라벨: TradingView 차트와 같은 분할만 반영 가격 · 끝난 봉 (screener.chart_ma, 주식 화면과 같은 규칙)
- 12개월 추세: 1년 수익이 단기 국채 ETF(BIL) 1년 수익보다 높으면 살아 있음 (lab_B 추세추종 규칙의 신호)
- 정보(이름 · 운용사 · 보수 · 운용자산 · 배당률 · 설정일): yfinance Ticker.info — 못 받으면 빈칸
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.etf_universe import BENCHMARK_ETF, ETF_GROUPS, RISK_FREE_ETF, etf_group_of, etf_tickers
from screener.chart_ma import ChartPrices, chart_prices, chart_view, cross_label, extend_chart, split_only_ohlcv
from screener.features import CROSS_KO

WINDOWS = {"w1": 5, "m1": 21, "m3": 63, "m6": 126, "y1": 252}   # 거래일
VOL_DAYS = 63
YEAR_DAYS = 252
SHEET_TAB = "ETF"
ET = ZoneInfo("America/Toronto")


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


def build_rows(prices: pd.DataFrame, chart_map: dict[str, ChartPrices], info: dict[str, dict[str, Any]],
               tickers: list[str] | None = None) -> list[dict[str, Any]]:
    """ETF 한 줄씩. prices = fetch_ohlcv 결과 ((티커, 필드) 멀티인덱스, 배당 조정 + Dividends 칸)."""
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
        last = float(c.iloc[-1])
        sma200 = float(c.tail(200).mean()) if len(c) >= 200 else math.nan
        y1 = _ret(c, YEAR_DAYS)
        split_close = split_only_ohlcv(p)["Close"].dropna()
        rows.append({
            "ticker": t,
            "group": group_of.get(t, ""),
            **{k: None for k in ("name", "family", "category", "currency", "expense", "aum", "yield", "inception")},
            **info.get(t, {}),
            "date": c.index[-1].date().isoformat(),
            "price": _num(split_close.iloc[-1]) if len(split_close) else _num(last),
            "ret": {**{k: _ret(c, n) for k, n in WINDOWS.items()}, "ytd": _ytd(c)},
            "above_200": bool(last > sma200) if math.isfinite(sma200) else None,
            "trend_12m": None if y1 is None else bool(y1 - (rf_y1 or 0.0) > 0),
            "high_52w_gap": _num(last / c.tail(YEAR_DAYS).max() - 1),
            "vol": _ann_vol(c),
            "mdd_1y": _max_drawdown(c),
            "corr_spy": 1.0 if t == BENCHMARK_ETF else _corr(c, bench),
            "cross": cross_labels(p, chart_map.get(t)),
        })
    return rows


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
    rows = _rounded(rows)
    return {
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "as_of": max((r["date"] for r in rows), default=None),
        "risk_free": RISK_FREE_ETF,
        "benchmark": BENCHMARK_ETF,
        "groups": [{"key": k, "ko": ko, "en": en, "tickers": [t for t in ts if t in present]}
                   for k, ko, en, ts in ETF_GROUPS],
        "items": rows,
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


def sheet_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """시트 [ETF] — 묶음 순서대로, 퍼센트는 글자로."""
    names = {k: ko for k, ko, _en, _t in ETF_GROUPS}
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
        })
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    from data.fetch import fetch_chart_prices, fetch_ohlcv

    ap = argparse.ArgumentParser(description="ETF 탭 데이터 (etf.json · 시트 [ETF])")
    ap.add_argument("--out", type=Path, help="etf.json 경로")
    ap.add_argument("--sheet", action="store_true", help="구글 시트 [ETF] 탭에도 쓴다")
    args = ap.parse_args(argv)

    t0 = time.time()
    tickers = etf_tickers()
    prices = fetch_ohlcv(tickers, period="5y")
    chart_map = fetch_chart_prices(list(prices.columns.levels[0]))
    info = fetch_info(tickers)
    rows = build_rows(prices, chart_map, info, tickers)
    data = payload(rows)
    missing = [t for t in tickers if t not in {r["ticker"] for r in rows}]
    print(f"[ETF] {len(rows)}/{len(tickers)}개 · 기준일 {data['as_of']} · 정보 없음 "
          f"{sum(1 for r in rows if not r.get('name'))}개 · 가격 없음 {missing} · {time.time() - t0:.0f}초")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        print(f"[ETF] {args.out} ({args.out.stat().st_size // 1024} KB)")
    if args.sheet:
        from screener.exporter import export_to_google_sheet

        ok = export_to_google_sheet(sheet_frame(rows), SHEET_TAB)
        print(f"[ETF] 시트 [{SHEET_TAB}] {'업데이트 완료' if ok else '건너뜀'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
