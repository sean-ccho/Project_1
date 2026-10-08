#!/usr/bin/env python3
"""Tier 3 후속: 10년 가격 기반 연구 패널 (PIT 유니버스, 벡터화 피처).

기존 패널(3.1년, 기술지표 44개)과 달리 일봉 OHLCV만으로 피처를 직접 계산한다 → 피처 캐시 불필요, 수 분 소요.
- 유니버스: 2016-01 ~ 2025-09 사이 한 번이라도 S&P 500이었던 종목 (data/universe/sp500_membership.csv)
  각 날짜의 행은 **그날 구성종목만** 남긴다 (PIT). 상장폐지 등으로 야후에 가격이 없는 종목은 빠진다 → 커버리지 표 참고
- 기간: 패널 2016-01-04 ~ 2025-09-30 (홀드아웃 2025-10-01 이후는 다운로드 자체를 하지 않는다)
- 선행수익률: t+1 시가 진입 → t+1+h 시가 청산 (fwd_ret_{5,10,20}d), 기존 패널과 동일 정의
- 섹터: 위키피디아 현재 S&P 500 GICS + config.SECTOR_MAP. **현재 구성종목만** 섹터가 있다 → `_sn`(섹터중립) 피처는
  상장폐지 종목이 빠진 생존 편향 표본이다. 해석 시 주의.

피처 (날짜 t 종가까지의 데이터만 사용):
  단기/중기 수익률  ret_1d ret_5d ret_10d ret_21d ret_63d ret_126d ret_252d
  모멘텀           mom_12_1 mom_6_1 resid_mom_12_1 (시장 베타 제거 잔차 모멘텀)
  반전             resid_ret_5d resid_ret_21d gap_high_10
  52주 위치        gap_high_252 gap_low_252
  변동성           vol_20 vol_60 atr_pct_14 idio_vol_60 beta_252 max_ret_21
  유동성/거래량    log_dollar_vol_20 vol_trend amihud_20 vol_ratio_5_20
  추세/과열        close_vs_sma50 close_vs_sma200 rsi_14
  섹터중립(_sn)    ret_5d ret_21d mom_12_1 resid_mom_12_1 vol_60 log_dollar_vol_20 gap_high_10 를 섹터 평균에서 뺀 값

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/build_price_panel.py
산출: data/research/panel_px10y.parquet, data/cache/ohlcv_px10y.parquet,
      data/universe/sector_map.csv, docs/quant_improvement/tier3/px10y_coverage.csv
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from paper_trading.universe import Membership, load_membership
from screener import config as cfg

START = "2015-01-01"        # 252일 피처 워밍업 포함
END = "2025-10-01"          # yfinance end는 배타적 → 마지막 데이터 2025-09-30 (홀드아웃 제외)
PANEL_START = "2016-01-04"
PANEL_END = "2025-09-30"
UNIVERSE_START = "2016-01-01"
HORIZONS = (5, 10, 20)
BATCH = 50

OHLCV_CACHE = ROOT / "data/cache/ohlcv_px10y.parquet"
SECTOR_CSV = ROOT / "data/universe/sector_map.csv"
COVERAGE_CSV = ROOT / "docs/quant_improvement/tier3/px10y_coverage.csv"
PANEL_OUT = ROOT / "data/research/panel_px10y.parquet"
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
SECTOR_NEUTRAL = ["ret_5d", "ret_21d", "mom_12_1", "resid_mom_12_1", "vol_60",
                  "log_dollar_vol_20", "gap_high_10"]


def _norm(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def download_ohlcv(tickers: list[str], refresh: bool) -> pd.DataFrame:
    """(필드, 티커) 멀티인덱스 컬럼의 일봉. 디스크 캐시가 있으면 재사용."""
    if OHLCV_CACHE.exists() and not refresh:
        print(f"OHLCV 캐시 로드: {OHLCV_CACHE.name}")
        return pd.read_parquet(OHLCV_CACHE)
    import yfinance as yf

    frames = []
    nb = (len(tickers) + BATCH - 1) // BATCH
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        print(f"[다운로드] {i // BATCH + 1}/{nb} ({len(batch)}개)", flush=True)
        data = None
        for attempt in range(3):
            try:
                data = yf.download(batch, start=START, end=END, interval="1d", auto_adjust=True,
                                   threads=True, progress=False)
                if data is not None and not data.empty:
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"  재시도 {attempt + 1}: {exc}", flush=True)
            time.sleep(10 * (attempt + 1))
        if data is None or data.empty:
            print("  → 배치 실패, 건너뜀", flush=True)
            continue
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" not in data.columns.get_level_values(0):  # (티커, 필드) 순서면 뒤집기
                data = data.swaplevel(0, 1, axis=1)
        else:
            data = pd.concat({batch[0]: data}, axis=1).swaplevel(0, 1, axis=1)
        frames.append(data[[f for f in FIELDS if f in data.columns.get_level_values(0)]])
        time.sleep(3)
    raw = pd.concat(frames, axis=1).sort_index(axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw.columns.names = ["Price", "Ticker"]
    OHLCV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(OHLCV_CACHE)
    print(f"저장: {OHLCV_CACHE} {raw.shape}")
    return raw


def load_sector_map(refresh: bool) -> pd.Series:
    if SECTOR_CSV.exists() and not refresh:
        return pd.read_csv(SECTOR_CSV).set_index("ticker")["sector"]
    rows = {_norm(t): (s, "config") for t, s in cfg.SECTOR_MAP.items()}
    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (research script)"})
        html = urllib.request.urlopen(req, timeout=30).read().decode()
        wiki = pd.read_html(io.StringIO(html))[0]
        for t, s in zip(wiki["Symbol"], wiki["GICS Sector"]):
            rows[_norm(t)] = (s, "wikipedia")  # 위키가 우선
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 위키 섹터 다운로드 실패 ({exc}) → config.SECTOR_MAP만 사용")
    df = pd.DataFrame([(t, s, src) for t, (s, src) in sorted(rows.items())],
                      columns=["ticker", "sector", "source"])
    SECTOR_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SECTOR_CSV, index=False)
    print(f"섹터 맵 저장: {SECTOR_CSV} ({len(df)}종목)")
    return df.set_index("ticker")["sector"]


def build_features(raw: pd.DataFrame, member: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    o, h, l, c, v = (raw[f] for f in FIELDS)
    M = member.reindex(index=c.index, columns=c.columns).fillna(False).astype(bool)
    ret = c.pct_change(fill_method=None)
    retc = ret.clip(-0.5, 0.5)
    mkt = retc.where(M).mean(axis=1)  # PIT 구성종목 동일가중 시장수익률
    beta = retc.rolling(252, min_periods=126).cov(mkt).div(
        mkt.rolling(252, min_periods=126).var(), axis=0)
    resid = retc.sub(beta.shift(1).mul(mkt, axis=0))
    dvol = c * v
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))

    f: dict[str, pd.DataFrame] = {
        "ret_1d": ret,
        "ret_5d": c / c.shift(5) - 1, "ret_10d": c / c.shift(10) - 1, "ret_21d": c / c.shift(21) - 1,
        "ret_63d": c / c.shift(63) - 1, "ret_126d": c / c.shift(126) - 1, "ret_252d": c / c.shift(252) - 1,
        "mom_12_1": c.shift(21) / c.shift(252) - 1, "mom_6_1": c.shift(21) / c.shift(126) - 1,
        "resid_mom_12_1": resid.rolling(231, min_periods=150).sum().shift(21),
        "resid_ret_5d": resid.rolling(5).sum(), "resid_ret_21d": resid.rolling(21).sum(),
        "gap_high_10": c / h.rolling(10).max() - 1,
        "gap_high_252": c / h.rolling(252, min_periods=200).max() - 1,
        "gap_low_252": c / l.rolling(252, min_periods=200).min() - 1,
        "vol_20": retc.rolling(20).std(), "vol_60": retc.rolling(60).std(),
        "atr_pct_14": tr.rolling(14).mean() / c,
        "idio_vol_60": resid.rolling(60, min_periods=40).std(),
        "beta_252": beta, "max_ret_21": retc.rolling(21).max(),
        "log_dollar_vol_20": np.log(dvol.rolling(20).mean().where(lambda x: x > 0)),
        "vol_trend": np.log(dvol.rolling(20).mean() / dvol.rolling(120, min_periods=80).mean()),
        "amihud_20": (retc.abs() / dvol.where(dvol > 0)).rolling(20).mean() * 1e9,
        "vol_ratio_5_20": v.rolling(5).mean() / v.rolling(20).mean(),
        "close_vs_sma50": c / c.rolling(50).mean() - 1,
        "close_vs_sma200": c / c.rolling(200, min_periods=150).mean() - 1,
        "rsi_14": 100 - 100 / (1 + gain / loss),
    }
    fwd = {}
    for hz in HORIZONS:
        fwd[f"fwd_ret_{hz}d"] = o.shift(-(1 + hz)) / o.shift(-1) - 1
    f["close"] = c
    return {**f, **fwd}, M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="OHLCV·섹터 다시 다운로드")
    args = ap.parse_args()

    mem = Membership(load_membership(ROOT / "data/universe/sp500_membership.csv"))
    tickers = sorted(mem.union_between(UNIVERSE_START, PANEL_END))
    print(f"유니버스(기간 중 한 번이라도 구성종목): {len(tickers)}종목")
    raw = download_ohlcv(tickers, args.refresh)
    raw = raw.loc[:PANEL_END]
    have = sorted(raw["Close"].columns[raw["Close"].notna().any()])
    print(f"가격이 있는 종목: {len(have)} / {len(tickers)}")
    raw = raw.loc[:, raw.columns.get_level_values("Ticker").isin(have)]
    sector = load_sector_map(args.refresh)

    dates = raw.index
    member = pd.DataFrame(False, index=dates, columns=have)
    sets = {d: mem.on(d.strftime("%Y-%m-%d")) for d in dates}
    for d, s in sets.items():
        cols = [t for t in s if t in member.columns]
        member.loc[d, cols] = True

    feats, M = build_features(raw, member)
    in_range = (dates >= PANEL_START) & (dates <= PANEL_END)
    mask = M.loc[in_range]
    cols = {}
    for name, w in feats.items():
        s = w.loc[in_range].where(mask).stack()
        cols[name] = s.astype("float32")
    panel = pd.concat(cols, axis=1)
    panel = panel[panel["close"].notna()]  # 구성종목이 아니거나 가격이 없는 날짜의 빈 행 제거
    panel.index.names = ["date", "티커"]
    panel = panel.reset_index()
    panel["섹터"] = panel["티커"].map(sector)

    # 섹터 중립 피처
    g = panel.groupby(["date", "섹터"])
    for name in SECTOR_NEUTRAL:
        panel[f"{name}_sn"] = (panel[name] - g[name].transform("mean")).astype("float32")

    # 커버리지: 각 날짜의 PIT 구성종목 중 가격이 있는 비율
    cov = []
    for d in dates[in_range][::21]:
        want = len(mem.on(d.strftime("%Y-%m-%d")))
        got = int(mask.loc[d].sum())
        cov.append({"date": d.date(), "pit_members": want, "with_price": got, "coverage": got / want})
    covdf = pd.DataFrame(cov)
    covdf["year"] = pd.to_datetime(covdf["date"]).dt.year
    COVERAGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    covdf.to_csv(COVERAGE_CSV, index=False)

    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_OUT)
    print(f"\n저장: {PANEL_OUT}")
    print(f"행 {len(panel):,} | 날짜 {panel['date'].nunique()} | 종목 {panel['티커'].nunique()} | "
          f"피처 {panel.shape[1] - 2 - 1 - len(HORIZONS) - 1}개")
    print(f"기간 {panel['date'].min().date()} ~ {panel['date'].max().date()}")
    print(f"섹터 있는 행 비율: {panel['섹터'].notna().mean():.1%}")
    print("\n연도별 PIT 구성종목 대비 가격 보유 비율 (생존 편향 지표):")
    print(covdf.groupby("year")["coverage"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
