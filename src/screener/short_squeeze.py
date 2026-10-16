"""공매도 정보 · 숏스퀴즈 감시 (표시 전용) — FINRA 공매도 잔고 + 나스닥 시가총액 + yfinance 유통주식·일봉.

숏스퀴즈는 예측하기 어렵다. 공매도가 몰린 종목 대부분은 스퀴즈 없이 오히려 더 떨어진다 (2026-10-15 idea_batch_v2:
S&P days to cover 상위 20% 는 63일 −0.55%). 그래서 '살 목록'이 아니라 '크게 움직일 수 있는 종목 감시 목록'이다.
매매 판단(CCS)에는 쓰지 않는다. 실제로 큰 움직임이 더 잦은지는 그림자 추적 shadow_v2 로 확인한다.

단계:
  연료 (fuel, '스퀴즈 대기')      공매도 비중(유통주식 대비, 모르면 발행주식 대비) ≥ 20% · 되사기(days to cover) ≥ 5일분
  불씨 (fire, '🔥 스퀴즈 진행 가능') 연료 + 마지막 종가가 직전 20일 최고 종가를 넘음 + 거래량 ≥ 20일 평균의 2배
  대상: 나스닥·NYSE 보통주, 주가 ≥ $2, 시가총액 ≥ $1억, FINRA 평균 거래량 ≥ 20만 주
  순서: 🔥 먼저, 대기는 5일 수익률 높은 순 (위로 움직이기 시작한 종목이 위로)
  나스닥 스크리너 API(시가총액)가 막히면: 나스닥 FTP 상장 목록 + 되사기 일수 상위만 yfinance 로 (시가총액 조건 없음)
되사기(days to cover) = 공매도 잔고 ÷ 하루 평균 거래량 (FINRA 값) — 공매도한 사람들이 평소 거래량으로 다 되사려면 며칠 걸리나.
FINRA 는 한 달에 두 번(15일 무렵·월말) 결제일 기준으로 집계하고 약 1~2주 뒤 공개한다 → 'M/D 기준' = 그 결제일.
대여 수수료·대여 가능 수량(스퀴즈를 가장 잘 보여 주는 값)은 유료 데이터라 없다.
"""

from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155 Safari/537.36"

FUEL_SHORT_PCT = 0.20
FUEL_DTC = 5.0
FIRE_LOOKBACK = 20
FIRE_VOL_RATIO = 2.0
MIN_PRICE, MIN_MCAP, MIN_ADV = 2.0, 1e8, 200_000
PRESELECT = 150           # 발행주식 대비 공매도 비중 상위 몇 종목만 yfinance 로 유통주식·일봉을 받는다
POOL_SIZE = 200           # 그림자 추적 비교용 무작위 표본 (같은 조건의 일반 종목)
WATCH_COLUMNS = ["ticker", "name", "status", "price", "short_pct", "short_basis", "dtc", "si", "si_change", "settle",
                 "ret_5d", "vol_ratio", "breakout", "mcap"]


def _norm(t: Any) -> str:
    return str(t).strip().upper().replace(".", "-").replace("/", "-").replace("^", "-")


# ============================================================
# 데이터
# ============================================================
def _finra(body: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    for attempt in range(3):
        try:
            r = requests.post(FINRA_URL, json=body, headers={"Accept": "application/json"}, timeout=90)
            if r.status_code == 200:
                return (r.json() if r.text.strip() else []), int(r.headers.get("record-total", "0") or 0)
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    raise RuntimeError("FINRA 요청 실패")


def finra_settlements(asof: date, n: int = 2) -> list[str]:
    """asof 까지 공개된 최근 결제일 n 개 (최근 순)."""
    since = (asof - timedelta(days=60)).isoformat()
    rows, _ = _finra({"limit": 100, "fields": ["settlementDate"],
                      "compareFilters": [{"compareType": "equal", "fieldName": "symbolCode", "fieldValue": "AAPL"},
                                         {"compareType": "gte", "fieldName": "settlementDate", "fieldValue": since}]})
    return sorted({r["settlementDate"] for r in rows if r["settlementDate"] <= asof.isoformat()}, reverse=True)[:n]


def finra_table(settle: str) -> pd.DataFrame:
    """그 결제일의 전 종목 공매도 잔고 (ticker, si, dtc, adv)."""
    got, offset, total = [], 0, None
    while total is None or offset < total:
        page, total = _finra({"limit": 5000, "offset": offset,
                              "fields": ["symbolCode", "currentShortPositionQuantity", "daysToCoverQuantity",
                                         "averageDailyVolumeQuantity"],
                              "compareFilters": [{"compareType": "equal", "fieldName": "settlementDate", "fieldValue": settle}]})
        got += page
        offset += 5000
        if not page:
            break
    df = pd.DataFrame(got).rename(columns={"symbolCode": "ticker", "currentShortPositionQuantity": "si",
                                           "daysToCoverQuantity": "dtc", "averageDailyVolumeQuantity": "adv"})
    if df.empty:
        return pd.DataFrame(columns=["ticker", "si", "dtc", "adv"])
    df["ticker"] = df["ticker"].map(_norm)
    for c in ("si", "dtc", "adv"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("ticker").reset_index(drop=True)


def nasdaq_table() -> pd.DataFrame:
    """나스닥 스크리너 — 나스닥·NYSE·AMEX 전 종목 가격·시가총액 (한 번에)."""
    r = requests.get(NASDAQ_SCREENER_URL, params={"tableonly": "true", "download": "true"},
                     headers={"User-Agent": BROWSER_UA, "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    df = pd.DataFrame(rows)
    df["ticker"] = df["symbol"].map(_norm)
    df["price"] = pd.to_numeric(df["lastsale"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    df["mcap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    return df[["ticker", "name", "price", "mcap"]].drop_duplicates("ticker")


class ShortData:
    """최근 공개된 FINRA 결제일 두 번치 (변화율용). 한 프로세스에서 한 번만 받는다."""

    def __init__(self, asof: date | None = None) -> None:
        self.asof = asof or datetime.now(timezone(timedelta(hours=-4))).date()
        settles = finra_settlements(self.asof)
        if not settles:
            raise RuntimeError("FINRA 결제일 없음")
        self.settle = settles[0]
        cur = finra_table(settles[0])
        if len(settles) > 1:
            prev = finra_table(settles[1])[["ticker", "si"]].rename(columns={"si": "si_prev"})
            cur = cur.merge(prev, on="ticker", how="left")
        else:
            cur["si_prev"] = np.nan
        cur["si_change"] = np.where(cur["si_prev"] > 0, cur["si"] / cur["si_prev"] - 1, np.nan)
        self.table = cur.set_index("ticker")

    def get(self, ticker: str) -> dict[str, Any] | None:
        t = _norm(ticker)
        if t not in self.table.index:
            return None
        r = self.table.loc[t]
        return {"si": float(r["si"]), "dtc": float(r["dtc"]), "si_change": float(r["si_change"]), "settle": self.settle}


_CACHE: dict[str, Any] = {}


def short_data() -> ShortData | None:
    """프로세스 안에서 한 번만 받는다. 실패하면 None (표시만 빠진다)."""
    if "data" not in _CACHE:
        try:
            _CACHE["data"] = ShortData()
            print(f"[공매도] FINRA {_CACHE['data'].settle} 기준 {len(_CACHE['data'].table):,}종목")
        except Exception as exc:  # noqa: BLE001
            print(f"[공매도] FINRA 실패 — 공매도 표시 건너뜀: {exc}")
            _CACHE["data"] = None
    return _CACHE["data"]


# ============================================================
# 헬스체크 옆 공매도 한 줄
# ============================================================
def short_line(short_pct: float | None, dtc: float | None, settle: str | None) -> str:
    """'공매도 18% · 되사기 6.1일분 (9/30 기준)' — 값이 없으면 빈 글자."""
    parts = []
    if short_pct is not None and np.isfinite(short_pct):
        parts.append(f"공매도 {short_pct * 100:.0f}%")
    if dtc is not None and np.isfinite(dtc):
        parts.append(f"되사기 {dtc:.1f}일분")
    if not parts:
        return ""
    when = f" ({int(settle[5:7])}/{int(settle[8:10])} 기준)" if settle else ""
    return " · ".join(parts) + when


def is_high(short_pct: float | None, dtc: float | None) -> bool:
    """헬스체크 공매도 줄을 주황으로 — 스퀴즈 연료 조건 둘 다 (비중 ≥ 20% · 되사기 ≥ 5일분).
    하나만 보면 S&P 밖 후보 풀은 절반 넘게 걸린다 (2026-10-16: 후보 풀 공매도 비중 중앙값 15.8%)."""
    return bool(short_pct is not None and np.isfinite(short_pct) and short_pct >= FUEL_SHORT_PCT
                and dtc is not None and np.isfinite(dtc) and dtc >= FUEL_DTC)


# ============================================================
# 숏스퀴즈 감시
# ============================================================
def _yf_floats(tickers: list[str]) -> dict[str, float]:
    """yfinance 유통주식 수 (실패한 종목은 빠진다)."""
    import yfinance as yf  # noqa: PLC0415

    out = {}
    for t in tickers:
        try:
            v = yf.Ticker(t).info.get("floatShares")
            if v and float(v) > 0:
                out[t] = float(v)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.25)
    return out


def _yf_daily(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """최근 3개월 일봉 (종가, 거래량)."""
    import yfinance as yf  # noqa: PLC0415

    closes, vols = [], []
    for i in range(0, len(tickers), 100):
        d = yf.download(tickers[i:i + 100], period="3mo", interval="1d", auto_adjust=True, progress=False, threads=True)
        if d is None or d.empty:
            continue
        if isinstance(d.columns, pd.MultiIndex):
            closes.append(d["Close"])
            vols.append(d["Volume"])
        else:
            closes.append(d[["Close"]].rename(columns={"Close": tickers[i]}))
            vols.append(d[["Volume"]].rename(columns={"Volume": tickers[i]}))
    if not closes:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(closes, axis=1), pd.concat(vols, axis=1)


def price_trigger(close: pd.Series, volume: pd.Series) -> dict[str, Any]:
    """마지막 종가가 직전 20일 최고 종가를 넘었나 · 거래량 배수 · 5일 수익률."""
    c, v = close.dropna(), volume.reindex(close.index).fillna(0)
    if len(c) < FIRE_LOOKBACK + 2:
        return {"breakout": False, "vol_ratio": np.nan, "ret_5d": np.nan}
    prev_high = float(c.iloc[-(FIRE_LOOKBACK + 1):-1].max())
    avg_vol = float(v.loc[c.index].iloc[-(FIRE_LOOKBACK + 1):-1].mean())
    vol_ratio = float(v.loc[c.index].iloc[-1]) / avg_vol if avg_vol > 0 else np.nan
    return {"breakout": bool(c.iloc[-1] > prev_high), "vol_ratio": vol_ratio,
            "ret_5d": float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else np.nan}


def classify(short_pct: float, dtc: float, breakout: bool, vol_ratio: float) -> str | None:
    fuel = np.isfinite(short_pct) and short_pct >= FUEL_SHORT_PCT and np.isfinite(dtc) and dtc >= FUEL_DTC
    if not fuel:
        return None
    return "fire" if breakout and np.isfinite(vol_ratio) and vol_ratio >= FIRE_VOL_RATIO else "fuel"


def build_watch(sd: ShortData | None = None, nasdaq: pd.DataFrame | None = None, *, floats: dict[str, float] | None = None,
                daily: tuple[pd.DataFrame, pd.DataFrame] | None = None, seed: int | None = None) -> pd.DataFrame:
    """감시 목록 + 비교용 무작위 표본(status='pool'). floats·daily 를 주면 yfinance 를 부르지 않는다 (시험용)."""
    sd = sd or short_data()
    if sd is None:
        return pd.DataFrame(columns=WATCH_COLUMNS)
    if nasdaq is None:
        try:
            nasdaq = nasdaq_table()
        except Exception as exc:  # noqa: BLE001
            print(f"[숏스퀴즈] 나스닥 스크리너 실패 → FTP 상장 목록으로 대신: {exc}")
    if nasdaq is not None:
        u = nasdaq.merge(sd.table.reset_index(), on="ticker", how="inner")
        u = u[(u["price"] >= MIN_PRICE) & (u["mcap"] >= MIN_MCAP) & (u["adv"] >= MIN_ADV) & (u["si"] > 0)].copy()
        u["shares_out"] = u["mcap"] / u["price"]
        u["si_pct_out"] = u["si"] / u["shares_out"]
        pre = u[(u["si_pct_out"] >= FUEL_SHORT_PCT / 2) | (u["dtc"] >= FUEL_DTC)].sort_values("si_pct_out", ascending=False)
    else:
        from data.ticker_fetcher import fetch_all_tickers  # noqa: PLC0415

        listed = {_norm(t) for t in fetch_all_tickers()}
        u = sd.table.reset_index()
        u = u[u["ticker"].isin(listed) & (u["adv"] >= MIN_ADV) & (u["si"] > 0)].copy()
        u["name"], u["price"], u["mcap"], u["shares_out"], u["si_pct_out"] = "", np.nan, np.nan, np.nan, np.nan
        pre = u[u["dtc"] >= FUEL_DTC].sort_values("dtc", ascending=False)
    pre = pre.head(PRESELECT)
    rng = random.Random(seed if seed is not None else int(sd.asof.strftime("%Y%m%d")))
    pool_tickers = rng.sample(sorted(u["ticker"]), min(POOL_SIZE, len(u)))
    names = list(pre["ticker"])
    floats = floats if floats is not None else _yf_floats(names)
    closes, vols = daily if daily is not None else _yf_daily(sorted(set(names) | set(pool_tickers)))
    rows = []
    for r in pre.itertuples():
        price = r.price
        if not (price and np.isfinite(price)) and r.ticker in closes.columns and closes[r.ticker].notna().any():
            price = float(closes[r.ticker].dropna().iloc[-1])
            if price < MIN_PRICE:
                continue
        fl = floats.get(r.ticker)
        pct_float = r.si / fl if fl else np.nan
        short_pct, basis = (pct_float, "float") if np.isfinite(pct_float) and pct_float < 3 else (r.si_pct_out, "shares_out")
        trig = (price_trigger(closes[r.ticker], vols[r.ticker]) if r.ticker in closes.columns
                else {"breakout": False, "vol_ratio": np.nan, "ret_5d": np.nan})
        rows.append({"ticker": r.ticker, "name": r.name, "status": classify(short_pct, r.dtc, trig["breakout"], trig["vol_ratio"]),
                     "price": price, "short_pct": short_pct, "short_basis": basis, "dtc": r.dtc, "si": r.si,
                     "si_change": r.si_change, "settle": sd.settle, "mcap": r.mcap, **trig})
    watch = pd.DataFrame(rows, columns=WATCH_COLUMNS)
    watch = watch[watch["status"].notna()]
    order = {"fire": 0, "fuel": 1}
    watch = (watch.assign(_o=watch["status"].map(order))
             .sort_values(["_o", "ret_5d", "short_pct"], ascending=[True, False, False], na_position="last").drop(columns="_o"))
    pool = u[u["ticker"].isin(pool_tickers)][["ticker", "name", "price", "dtc", "si", "si_change", "mcap", "si_pct_out"]]
    pool = pool.rename(columns={"si_pct_out": "short_pct"}).assign(status="pool", short_basis="shares_out", settle=sd.settle,
                                                                    ret_5d=np.nan, vol_ratio=np.nan, breakout=False)
    out = pd.concat([watch, pool[WATCH_COLUMNS]], ignore_index=True)
    n_fire, n_fuel = int((out["status"] == "fire").sum()), int((out["status"] == "fuel").sum())
    print(f"[숏스퀴즈] 대상 {len(u):,}종목 · 후보 {len(pre)} · 🔥 진행 가능 {n_fire} · 대기 {n_fuel} (FINRA {sd.settle} 기준)")
    return out.reset_index(drop=True)


def main() -> int:
    """CI 단계: data/paper_trading/squeeze_watch.parquet 저장. 실패해도 0 (표시용)."""
    from pathlib import Path  # noqa: PLC0415

    out = Path("data/paper_trading/squeeze_watch.parquet")
    try:
        watch = build_watch()
        out.parent.mkdir(parents=True, exist_ok=True)
        watch.to_parquet(out, index=False)
        print(f"[숏스퀴즈] 저장 {out} ({len(watch)}행)")
    except Exception as exc:  # noqa: BLE001 — 표시용이라 파이프라인을 멈추지 않는다
        print(f"[숏스퀴즈] 실패 (건너뜀): {exc}")
    return 0


def load_watch(path: str = "data/paper_trading/squeeze_watch.parquet") -> pd.DataFrame:
    """저장된 감시 목록 (없으면 빈 표). status 가 fire·fuel 인 줄만."""
    try:
        w = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=WATCH_COLUMNS)
    return w[w["status"].isin(["fire", "fuel"])].reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
