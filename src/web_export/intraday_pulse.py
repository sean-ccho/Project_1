"""장중 펄스 (2026-11-09) — 사이트(seancho.vercel.app) 장중 숫자. 표시 전용 — 매매 판단 · PT 상태 파일과 무관하다.

Mac mini LaunchAgent(com.seancho.market-pulse) → .github/workflows/market-pulse.yml 이 월~금 09:45~15:45 15분마다와
16:20(마감 펄스)에 한 번씩 돌려 live-data 고아 브랜치에 세 파일을 올린다. 사이트는 읽기만 한다.

- 기준값(base.json.gz): 그날 첫 실행에 전 거래일까지의 일봉 2년으로 종목별 전일 종가 · 직전 49/199 종가 합(배당 조정) ·
  직전 251봉 고가 최고/저가 최저(분할만 반영) · 직전 19일 거래량 합과 종목 정보(회사 · 섹터 · 시총 · 미니 차트)를 만든다.
  같은 날 다음 실행은 브랜치에 둔 것을 다시 쓴다.
- 시세: 대상 종목의 일봉 5일 — 오늘 장중 봉의 종가(= 지금 가격) · 고가 · 저가 · 거래량(지금까지), 전날 종가.
  거래량은 지금까지 ÷ 시간 곡선(VOLUME_CURVE) = 그날 예상 거래량. 오늘 5분봉은 쓰지 않는다 — 11-09 장중에
  일부 봉에 그때까지의 하루 누적 거래량이 찍혀 있었다 (NVDA 10:40 봉 3.86억 주 → 5분봉 합 39.7억 주, 실제는 약 3.9억).
  그런 봉을 빼면 5분봉 합 = 장중 일봉 거래량 (AAPL 2,574만 · 2,650만, MSFT 1,446만 · 1,500만).
- 계산은 밤과 같은 함수: 시장 한눈에 market_summary · 경기 상태 ind_spy200/ind_cycdef · ETF price_fields ·
  PT-1 매도선 exit_levels · 후보 지표 compute_features_for_ticker (+ PT-1 하드필터 상수) · 숏스퀴즈 price_trigger/classify.
  밤 값은 '그날 종가'로, 장중 값은 '지금 가격으로 마감하면'으로 같은 규칙을 적용한 잠정값이다.

출력 (--out 폴더):
- live.json   시각 · 장 상태 · 지수 · 시장 한눈에(미국 · 캐나다) · 경기 상태 · ETF · PT-1 · 후보 · 숏스퀴즈 (수십 KB)
- quotes.json 종목별 [가격, 오늘 등락, 거래량 배수, 52주 최저, 52주 최고, 신고가1/신저가2] — 목록 · 종목 페이지용
- base.json.gz 기준값 (위)

공개 브랜치에 올라가므로 시장 전체 자료만 담는다 — 내계좌 종목 목록은 넣지 않는다 (사이트 서버가 quotes 에서 골라 쓴다).
로컬에서 돌릴 때는 CI=1 (data/cache 에 OHLCV 캐시를 쓰지 않게): CI=1 PYTHONPATH=.:src python -m web_export.intraday_pulse --out DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time as _time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
ET = ZoneInfo("America/Toronto")
RAW = "https://raw.githubusercontent.com/sean-ccho/Project_1"
UNIVERSE_RAW = f"{RAW}/universe-data"
CANADA_RAW = f"{RAW}/canada-data"
DASHBOARD_URL = f"{RAW}/dashboard-data/dashboard.json"
POSITIONS_PATH = Path("data/paper_trading/positions.json")
SQUEEZE_PATH = Path("data/paper_trading/squeeze_watch.parquet")

OPEN = time(9, 30)
CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
CLOSE_WINDOW = timedelta(minutes=45)  # 마감 뒤 이 안의 실행 = 마감 펄스 (final — 사이트 '16:00 마감 · 잠정')
SESSION_MIN = 390
BATCH = 200                           # yf.download 한 번에 넣는 종목 수 (11-09: 일봉 5일 2,574종목 50초)
THREADS = 16                          # yf.download 동시 받기 수 — 기본(threads=True)은 CPU×2 라 GitHub 러너(4코어)는 8개뿐:
                                      # 11-10 7,540종목 시세 117~376초 · 기준값 236초. Mac(8코어 = 16개)은 7,547종목 153초 · 오류 0
BASE_PERIOD = "2y"                    # 기준값 일봉 — 251봉 고가/저가 · 199일 합에 넉넉하게
INDEX_SYMBOLS = ("SPY", "QQQ", "^VIX", "^GSPTSE", "XIC.TO", "CAD=X")
CA_SUFFIXES = (".TO", ".V", ".CN", ".NE")

# 거래량 시간 곡선 — 개장 뒤 N분까지의 거래량 ÷ 그날 확정 거래량 (중앙값). 지난날 5분봉(끝난 날 — 이상한 봉 없음)으로
# 2026-11-09 측정: 시총 $50B+ 25 · $2~10B 25 종목 × 최근 60거래일 (2,786 종목·일). 장중 일봉 거래량 = 그때까지 5분봉 합.
# 16:00 에도 0.81 인 것은 마감 동시호가 · 늦은 체결이 그 뒤에 들어와서. 단순 시간비율(지난 분 ÷ 390)은 오후로 갈수록
# 크게 틀린다 (14:00 실제 0.46 vs 단순 0.69 → 거래량 배수를 1/3 작게 본다).
VOLUME_CURVE: tuple[tuple[int, float], ...] = (
    (0, 0.0), (5, 0.0285), (15, 0.051), (30, 0.083), (45, 0.117), (60, 0.147), (75, 0.177), (90, 0.203),
    (105, 0.23), (120, 0.257), (135, 0.281), (150, 0.303), (165, 0.325), (180, 0.346), (195, 0.366),
    (210, 0.385), (225, 0.404), (240, 0.423), (255, 0.444), (270, 0.462), (285, 0.486), (300, 0.51),
    (315, 0.532), (330, 0.556), (345, 0.584), (360, 0.616), (375, 0.661), (385, 0.716), (390, 0.811),
)
MIN_FRACTION_MINUTES = 15  # 개장 15분 안에는 거래량 배수를 내지 않는다 (몇 분 치로 하루를 짐작하면 너무 흔들린다)
# 마감 펄스(16:20) — 일봉 거래량에 마감 동시호가 · 늦은 체결이 들어가 있다. 11-09 측정: 나스닥 · TSX 는 16:06 에 이미,
# NYSE 는 16:14 까지 안 들어오다가 16:18 에 들어왔다 (JPM 일봉 ÷ 5분봉 합 1.02 → 1.56). 그래서 마감 펄스는 16:20 보다 앞당기지 않는다
AFTER_CLOSE_FRACTION = 1.0


# ── 숫자 ──────────────────────────────────────────────────────────────────────

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


def _px(v: Any) -> float | None:
    """주가 반올림 — 1달러 이상 소수 2자리, 미만 4자리 (market_summary.price_round 와 같다)."""
    x = _f(v)
    return None if x is None else round(x, 2 if abs(x) >= 1 else 4)


def _clean(obj: Any) -> Any:
    """JSON 에 넣을 수 있게 — numpy 숫자 · NaN 정리."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _f(obj)
    return obj


def is_ca(ticker: str) -> bool:
    """캐나다 상장 (Yahoo 접미사)."""
    return ticker.upper().endswith(CA_SUFFIXES)


# ── 장 시간 ───────────────────────────────────────────────────────────────────

def _calendar(market: str) -> tuple[set[date], set[date]]:
    """(휴장일, 조기 폐장일) — data.event_calendar 의 NYSE · TMX 공식 달력."""
    from data.event_calendar import NYSE_CLOSED, NYSE_EARLY, TSX_CLOSED, TSX_EARLY

    closed, early = (NYSE_CLOSED, NYSE_EARLY) if market == "us" else (TSX_CLOSED, TSX_EARLY)
    return ({date.fromisoformat(d) for d, _k, _e in closed}, {date.fromisoformat(d) for d, _k, _e in early})


@dataclass(frozen=True)
class Session:
    """한 시장의 오늘 장 상태 (토론토 = 뉴욕 시각)."""

    market: str   # "us" | "ca"
    day: date
    state: str    # "pre" 개장 전 · "open" 장중 · "closed" 마감 뒤 · "holiday" 주말 · 휴장
    close: time   # 16:00 · 조기 폐장 13:00

    @property
    def minutes(self) -> int:
        """정규장 길이 (분)."""
        return (self.close.hour * 60 + self.close.minute) - (OPEN.hour * 60 + OPEN.minute)

    def close_at(self) -> datetime:
        return datetime.combine(self.day, self.close, tzinfo=ET)

    def active(self, now: datetime) -> bool:
        """펄스를 낼 때 — 장중이거나 마감 직후(마감 펄스)."""
        return self.state == "open" or (self.state == "closed" and now - self.close_at() <= CLOSE_WINDOW)


def market_session(market: str, now: datetime) -> Session:
    """now(토론토 시각)의 그 시장 상태."""
    closed, early = _calendar(market)
    day = now.date()
    close = EARLY_CLOSE if day in early else CLOSE
    if now.weekday() >= 5 or day in closed:
        state = "holiday"
    elif now.time() < OPEN:
        state = "pre"
    elif now.time() < close:
        state = "open"
    else:
        state = "closed"
    return Session(market, day, state, close)


def volume_fraction(elapsed_min: float, session_min: int = SESSION_MIN) -> float | None:
    """개장 뒤 elapsed_min 분까지 5분봉 거래량 합이 그날 확정 거래량의 몇 % 인지 (VOLUME_CURVE 사이는 직선).

    조기 폐장(13:00)은 시간을 정규장 길이로 늘려 같은 곡선을 쓴다. 개장 MIN_FRACTION_MINUTES 분 안이면 None.
    """
    if session_min <= 0 or elapsed_min < MIN_FRACTION_MINUTES * session_min / SESSION_MIN:
        return None
    x = min(float(elapsed_min), float(session_min)) * SESSION_MIN / session_min
    for (x0, y0), (x1, y1) in zip(VOLUME_CURVE, VOLUME_CURVE[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return VOLUME_CURVE[-1][1]


def vol_ratio(volume_so_far: float | None, fraction: float | None, v19: float | None) -> float | None:
    """밤 '거래량돌파배수'(오늘 거래량 ÷ 오늘 포함 20일 평균)와 같은 식에 오늘 예상 거래량을 넣는다.

    예상 거래량 Vp = 5분봉 합 ÷ fraction. 배수 = Vp ÷ ((직전 19일 합 + Vp) ÷ 20). 마감 때(곡선 끝)는 밤 값과 거의 같다.
    """
    vs, fr, base = _f(volume_so_far), _f(fraction), _f(v19)
    if vs is None or fr is None or fr <= 0 or base is None or vs <= 0:
        return None
    vp = vs / fr
    return 20 * vp / (base + vp)


# ── 받기 ──────────────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: int = 60) -> Any:
    """URL → JSON (.gz 는 풀어서). 실패하면 예외."""
    req = urllib.request.Request(url, headers={"User-Agent": "project1-market-pulse"})
    with urllib.request.urlopen(req, timeout=timeout) as res:  # noqa: S310 — 고정된 GitHub raw 주소
        raw = res.read()
    if url.endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _frames(data: pd.DataFrame, batch: Sequence[str]) -> dict[str, pd.DataFrame]:
    """yf.download(group_by="ticker") 결과 → {티커: 표} (한 종목뿐이어도)."""
    if data is None or data.empty:
        return {}
    if not isinstance(data.columns, pd.MultiIndex):
        return {batch[0]: data}
    if "Close" in data.columns.get_level_values(0) and batch[0] not in data.columns.get_level_values(0):
        data = data.swaplevel(0, 1, axis=1)
    have = set(data.columns.get_level_values(0))
    return {t: data[t] for t in batch if t in have}


def _download(tickers: Sequence[str], **kw: Any) -> dict[str, pd.DataFrame]:
    """yf.download 를 BATCH 개씩 (한 번 다시 시도). 못 받은 종목은 빠진다."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), BATCH):
        batch = list(tickers[i:i + BATCH])
        for attempt in (1, 2):
            try:
                data = yf.download(batch, group_by="ticker", progress=False, threads=THREADS, auto_adjust=False, **kw)
                got = _frames(data, batch)
                if got:
                    out.update(got)
                    break
            except Exception as exc:  # noqa: BLE001 — 표시용이라 나머지는 계속
                print(f"[펄스] 받기 실패 ({len(batch)}종목, 시도 {attempt}): {exc}")
            if attempt == 1:
                _time.sleep(5)
    return out


@dataclass(frozen=True)
class Bar:
    """한 종목 오늘 장중 일봉 (Yahoo — 장이 끝나기 전엔 종가 = 지금 가격)."""

    price: float
    high: float
    low: float
    volume: float              # 오늘 지금까지 거래량
    prev_close: float | None   # 전 거래일 종가 (분할만 반영)


def today_bars(tickers: Sequence[str], today: date) -> dict[str, Bar]:
    """일봉 5일 → {티커: 오늘 Bar}. 오늘 봉이 없는 종목(휴장 · 아직 거래 없음)은 빠진다."""
    out: dict[str, Bar] = {}
    for t, df in _download(tickers, period="5d", interval="1d").items():
        sub = df.dropna(subset=["Close"])
        if sub.empty or pd.Timestamp(sub.index[-1]).date() != today:
            continue
        last = sub.iloc[-1]
        vol = _f(last.get("Volume"))
        out[t] = Bar(price=float(last["Close"]), high=_f(last.get("High")) or float(last["Close"]),
                     low=_f(last.get("Low")) or float(last["Close"]), volume=vol or 0.0,
                     prev_close=_f(sub["Close"].iloc[-2]) if len(sub) >= 2 else None)
    return out


# ── 종목 정보 · 기준값 ────────────────────────────────────────────────────────

def _cap(info: Mapping[str, Any]) -> float | None:
    return _f((info.get("fundamentals") or {}).get("market_cap"))


def load_universe(universe_raw: str = UNIVERSE_RAW, canada_raw: str = CANADA_RAW,
                  get: Callable[[str], Any] = _get_json) -> dict[str, Any]:
    """전체분석 목록 파일(밤 universe-scan · canada-scan) → 종목 정보.

    {"meta": {티커: [회사, 섹터, 시장, 시총, 밤 종가, 미니 차트]}, "sp500": [...], "ca_breadth": [...]}
    sp500 = S&P 500 시장 폭 · 섹터, ca_breadth = TSX 시총 C$1B 이상 (밤 canada_summary 와 같다).
    """
    from web_export.market_summary import CA_MIN_CAP

    meta: dict[str, list[Any]] = {}
    lists: dict[str, list[str]] = {}
    for key, url in (("sp500", f"{universe_raw}/sp500.json.gz"), ("nasdaq", f"{universe_raw}/nasdaq.json.gz"),
                     ("new", f"{universe_raw}/new.json.gz"), ("extra", f"{universe_raw}/extra.json.gz"),
                     ("ca", f"{canada_raw}/ca.json.gz")):
        try:
            payload = get(url)
        except Exception as exc:  # noqa: BLE001 — 번외(extra)는 없을 수 있다
            print(f"[펄스] {key} 목록 못 받음: {exc}")
            continue
        items = payload.get("items") or {}
        lists[key] = list(items)
        for t, v in items.items():
            if t in meta:
                continue
            meta[t] = [v.get("company"), v.get("sector"), v.get("market") or key, _cap(v), _f(v.get("price")),
                       v.get("spark_w")]
    ca_breadth = [t for t in lists.get("ca", []) if meta[t][2] == "tsx" and (meta[t][3] or 0) >= CA_MIN_CAP]
    return {"meta": meta, "sp500": lists.get("sp500", []), "ca_breadth": ca_breadth}


def coverage(universe: Mapping[str, Any], extra: Iterable[str] = (), scope: str = "all") -> list[str]:
    """시세를 받을 종목. all = 전체분석 전부 (미국 · 신규 상장 · 번외 · 캐나다, 약 7,400) + extra (기본, 11-09 —
    2,574종목 50초라 전부 15분 안에 든다), cover = S&P 500 + 미국 시총 $2B+ + 캐나다 C$1B+ (시장 한눈에 목록과 같은 범위)
    + extra (지수 · ETF · 사이트 후보 · 차트 목록 · 숏스퀴즈 · 번외) — Yahoo 가 막히면 줄여서 돌리는 범위."""
    from web_export.market_summary import CA_MIN_CAP, MOVER_MIN_CAP

    meta = universe["meta"]
    if scope == "all":
        base = list(meta)
    else:
        sp = set(universe["sp500"])
        base = [t for t, m in meta.items()
                if t in sp or m[2] == "extra"
                or (m[3] or 0) >= (CA_MIN_CAP if m[2] in ("tsx", "tsxv") else MOVER_MIN_CAP)]
    return list(dict.fromkeys([*INDEX_SYMBOLS, *base, *extra]))


def base_row(df: pd.DataFrame) -> list[Any] | None:
    """전 거래일까지의 일봉(auto_adjust=False: 분할만 반영 OHLC + Adj Close) → 기준값.

    [전일 종가, 직전 49 조정 종가 합, 직전 199 합, 직전 251봉 고가 최고, 저가 최저, 직전 19일 거래량 합, 마지막 날짜]
    — market_summary.price_stats(50/200일선 · 신고가/신저가)와 features(거래량돌파배수)에 오늘 값 하나를 넣으면 되는 모양.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None
    c = df["Close"].dropna()
    if c.empty:
        return None
    adj = (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).dropna()
    hi, lo = df["High"].dropna().iloc[-251:], df["Low"].dropna().iloc[-251:]
    vol = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
    return [
        _r(c.iloc[-1], 6),
        _r(adj.iloc[-49:].sum(), 4) if len(adj) >= 49 else None,
        _r(adj.iloc[-199:].sum(), 4) if len(adj) >= 199 else None,
        _r(hi.max(), 6) if len(hi) >= 20 else None,
        _r(lo.min(), 6) if len(lo) >= 20 else None,
        _r(vol.iloc[-19:].sum(), 0) if len(vol) >= 19 and vol.iloc[-19:].sum() > 0 else None,
        str(c.index[-1].date()),
    ]


def build_base(tickers: Sequence[str], today: date, universe: Mapping[str, Any], now: datetime,
               scope: str = "all") -> dict[str, Any]:
    """그날 첫 실행 — 일봉 2년을 받아 오늘 장중 봉을 빼고 기준값을 만든다 (scope 를 적어 둔다 — 범위가 바뀌면 다시 만든다)."""
    rows: dict[str, list[Any]] = {}
    for t, df in _download(tickers, period=BASE_PERIOD, interval="1d").items():
        df = df[pd.to_datetime(df.index).date < today]
        row = base_row(df)
        if row:
            rows[t] = row
    days = [r[6] for r in rows.values()]
    day = max(days, key=days.count) if days else None   # 가장 많은 종목의 마지막 날 = 전 거래일
    return {"schema": SCHEMA_VERSION, "built_at": now.isoformat(timespec="seconds"), "for_day": today.isoformat(),
            "day": day, "scope": scope, "tickers": list(tickers), "t": rows, **universe}


def load_base(path: Path | None, today: date, scope: str = "all") -> dict[str, Any] | None:
    """오늘 같은 범위로 만든 기준값이 있으면 그것 (없거나 다른 날 · 다른 범위 것이면 None → 새로 만든다)."""
    if path is None or not path.exists():
        return None
    try:
        base = json.loads(gzip.decompress(path.read_bytes()))
    except Exception as exc:  # noqa: BLE001
        print(f"[펄스] 기준값 파일을 못 읽음 — 새로 만든다: {exc}")
        return None
    ok = (base.get("for_day") == today.isoformat() and base.get("schema") == SCHEMA_VERSION
          and base.get("scope", "cover") == scope)
    return base if ok else None


# ── 종목 하나 ────────────────────────────────────────────────────────────────

def live_quote(row: Sequence[Any] | None, bar: Bar, fraction: float | None) -> dict[str, Any]:
    """기준값 + 오늘 시세 → 종목 하나의 장중 숫자 (기준값이 없으면 가격만)."""
    p = bar.price
    q: dict[str, Any] = {"p": p, "c": None, "v": None, "lo": None, "hi": None, "nh": None, "nl": None,
                         "a50": None, "a200": None}
    pc = bar.prev_close or (_f(row[0]) if row else None)   # Yahoo 전날 종가 (밤새 분할이 있어도 맞다) → 없으면 기준값
    q["c"] = p / pc - 1 if pc else None
    if not row:
        return q
    s49, s199, h251, l251, v19 = (_f(x) for x in row[1:6])
    q["v"] = vol_ratio(bar.volume, fraction, v19)
    if s49:
        q["a50"] = p > s49 / 49      # 오늘 포함 50일 평균보다 위 ⟺ p > 직전 49일 평균
    if s199:
        q["a200"] = p > s199 / 199
    if h251 is not None and l251 is not None:
        q["nh"], q["nl"] = bar.high >= h251, bar.low <= l251
        q["lo"], q["hi"] = min(l251, bar.low), max(h251, bar.high)
    return q


def quote_list(quotes: Mapping[str, Mapping[str, Any]]) -> dict[str, list[Any]]:
    """quotes.json 의 q — {티커: [가격, 오늘 등락, 거래량 배수, 52주 최저, 52주 최고, 표시]} (표시 1 = 신고가, 2 = 신저가)."""
    out = {}
    for t, q in quotes.items():
        flag = (1 if q.get("nh") else 0) | (2 if q.get("nl") else 0)
        out[t] = [_px(q["p"]), _r(q.get("c"), 4), _r(q.get("v"), 2), _px(q.get("lo")), _px(q.get("hi")), flag]
    return out


# ── 시장 한눈에 ──────────────────────────────────────────────────────────────

def market_live(quotes: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Sequence[Any]], breadth: Sequence[str],
                tickers: Iterable[str], day: str, index_chg: float | None, generated_at: str, *, market: str = "us",
                index_label: str = "SPY", min_cap: float | None = None) -> dict[str, Any] | None:
    """market_summary(밤 market.json 과 같은 함수)를 장중 값으로 — 시장 폭 · 많이 오른/내린/거래량 급증 · 섹터.
    실적 일정(calendar · mega)은 장중에 안 바뀌어서 뺀다 (사이트가 밤 것을 그대로 쓴다)."""
    from web_export.market_summary import MOVER_MIN_CAP, market_summary

    items: dict[str, dict[str, Any]] = {}
    flags: dict[str, dict[str, Any]] = {}
    for t in tickers:
        q, m = quotes.get(t), meta.get(t)
        if not q or not m or q.get("c") is None:
            continue
        cap, px_night = _f(m[3]), _f(m[4])
        if cap and px_night:
            cap = cap * q["p"] / px_night        # 시총도 지금 가격으로
        items[t] = {"company": m[0], "sector": m[1], "price": q["p"], "ret_1d": q["c"], "vol_ratio": q.get("v"),
                    "fundamentals": {"market_cap": cap}, "spark_w": m[5]}
        flags[t] = {"above50": q.get("a50"), "above200": q.get("a200"), "new_high": q.get("nh"),
                    "new_low": q.get("nl")}
    out = market_summary(items, breadth, flags, day, index_chg, generated_at, market=market, index_label=index_label,
                         min_cap=MOVER_MIN_CAP if min_cap is None else min_cap)
    if out:
        out.pop("calendar", None)
        out.pop("mega", None)
    return out


# ── 경기 상태 · ETF ───────────────────────────────────────────────────────────

def macro_live(today: date, fx: Bar | None) -> dict[str, Any] | None:
    """경기 상태 중 시장 가격 지표 둘(SPY 200일선 · 순환/방어) + 환율 — macro_scan 과 같은 함수에 오늘 장중 봉까지."""
    from screener.macro_scan import CYCLICAL_ETFS, DEFENSIVE_ETFS, fetch_prices, ind_cycdef, ind_spy200

    start = (pd.Timestamp(today) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    px = fetch_prices(["SPY", *CYCLICAL_ETFS, *DEFENSIVE_ETFS], start)
    if px is None or px.empty or pd.Timestamp(px.index[-1]).date() != today:
        return None
    return {"indicators": [ind_spy200(px["SPY"]), ind_cycdef(px)],
            "fx": {"usdcad": round(fx.price, 4), "date": today.isoformat()} if fx else None}


def etf_live(today: date, quotes: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """ETF 탭 — 가격 · 기간 수익률 · 200일선 위 · 52주 고점 대비 (etf_scan.price_fields — 밤과 같은 함수) + 오늘 등락."""
    from data.etf_universe import etf_tickers
    from data.fetch import fetch_ohlcv
    from screener.etf_scan import price_fields

    tickers = etf_tickers()
    prices = fetch_ohlcv(tickers, period="2y")
    have = set(prices.columns.get_level_values(0))
    out: dict[str, dict[str, Any]] = {}
    for t in tickers:
        if t not in have:
            continue
        p = prices[t].dropna(subset=["Close"])
        if p.empty or pd.Timestamp(p.index[-1]).date() != today:
            continue
        pf = price_fields(p)
        out[t] = {"date": pf["date"], "price": _px(pf["price"]), "chg": _r((quotes.get(t) or {}).get("c"), 4),
                  "ret": {k: _r(v, 4) for k, v in pf["ret"].items()}, "above_200": pf["above_200"],
                  "high_52w_gap": _r(pf["high_52w_gap"], 4)}
    return out


# ── PT-1 ─────────────────────────────────────────────────────────────────────

def _spy_ref(dash: Mapping[str, Any], price_date: str | None) -> float | None:
    """밤 $5,000 계좌의 SPY 평가가 기준으로 쓴 SPY 종가 (benchmark 의 price_date 종가)."""
    closes = (dash.get("benchmark") or {}).get("closes") or []
    for d, v in reversed(closes):
        if price_date is None or d <= price_date:
            return _f(v)
    return None


def pt1_live(dash: Mapping[str, Any], positions: Sequence[Mapping[str, Any]], quotes: Mapping[str, Mapping[str, Any]],
             today: date) -> dict[str, Any] | None:
    """PT-1 — 보유 종목 장중 가격 · 수익률 · 잠정 매도선(engine.exit_levels, 고점 = max(기록, 지금 가격) — 저장 안 함) ·
    $5,000 계좌 평가액(현금 + 주식 수 × 지금 가격) · 같은 날 SPY. 엔진은 종가로만 판정한다 (PT1_STOP_INTRADAY=False)."""
    from paper_trading.engine import exit_levels
    from paper_trading.portfolio import get_holding_days

    acct = next((a for a in dash.get("accounts") or [] if a.get("key") == "pt1"), None)
    if acct is None:
        return None
    spy = quotes.get("SPY")
    price_date = dash.get("price_date")
    ref = _spy_ref(dash, price_date)
    spy_move = spy["p"] / ref if spy and ref else None
    by_key = {(str(p["ticker"]), str(p.get("entry_date"))): p for p in positions}
    rows, held_value, any_live = [], 0.0, False
    for r in acct.get("positions") or []:
        t = str(r["ticker"])
        q = quotes.get(t)
        price = q["p"] if q else _f(r.get("price"))
        any_live = any_live or q is not None
        pos = dict(by_key.get((t, str(r.get("entry_date")))) or {"ticker": t, "entry_date": r.get("entry_date"),
                                                                 "entry_price": r.get("entry_price"),
                                                                 "highest_price": r.get("highest_price"),
                                                                 "strategy": r.get("strategy") or ""})
        entry = float(pos["entry_price"])
        highest = max(float(pos.get("highest_price") or entry), price or 0.0)
        lv = exit_levels({**pos, "highest_price": highest}, as_of=today)
        nxt = lv.get("next")
        shares = _f(r.get("shares"))
        if shares is not None and price is not None:
            held_value += shares * price
        spy_ret = _f(r.get("spy_return"))
        rows.append({
            "ticker": t, "live": q is not None, "price": _r(price, 4), "chg": _r((q or {}).get("c"), 4),
            "return": _r(price / entry - 1, 4) if price else None,
            "highest_price": _r(highest, 4), "days_held": get_holding_days(pos, today),
            "value": _r(shares * price, 2) if shares is not None and price is not None else None,
            "spy_return": _r((1 + spy_ret) * spy_move - 1, 4) if spy_ret is not None and spy_move else spy_ret,
            "exit": {"up": _r(lv.get("up"), 4), "up_tag": lv.get("up_tag") or None,
                     "down": _r(lv.get("down"), 4), "down_tag": lv.get("down_tag") or None,
                     "next": {"day": int(nxt["day"]), "side": nxt["side"], "price": _r(nxt["price"], 4),
                              "tag": nxt["tag"]} if nxt else None},
        })
    d = acct.get("dollar") or None
    dollar = None
    if d:
        initial, cash = _f(d.get("initial")) or 0.0, _f(d.get("cash")) or 0.0
        equity = cash + held_value
        spy_eq = _f(d.get("spy_equity"))
        spy_eq = spy_eq * spy_move if spy_eq is not None and spy_move else spy_eq
        dollar = {"equity": _r(equity, 2), "return": _r(equity / initial - 1, 4) if initial else None,
                  "spy_equity": _r(spy_eq, 2), "spy_return": _r(spy_eq / initial - 1, 4) if initial and spy_eq else None,
                  "cash": _r(cash, 2), "initial": _r(initial, 2), "date": today.isoformat()}
    return {"key": "pt1", "base_date": price_date, "live": any_live, "positions": rows, "dollar": dollar}


# ── 후보 지표 ────────────────────────────────────────────────────────────────

def heat_flags(row: Mapping[str, Any], strategy: str | None, regime: str | None) -> list[dict[str, Any]]:
    """PT-1 하드필터 중 가격으로 바뀌는 것 — 지금 가격으로 마감하면 걸리는 칸 (candidate_selector._apply_hard_filters 와 같은 상수).

    [{"k": 종류, "v": 지금 값, "lim": 기준}] — rsi (RSI ≥ 75, 약세장 70) · bband (볼린저 위치 ≥ 0.95) · ret5 (5일 수익 ≥ 15%) ·
    bottom_pos (바닥반등인데 52주 위치 > 0.65) · bottom_rsi (바닥반등 RSI 30~35 데드존 — lim 없음).
    """
    from screener.config import (CANDIDATE_5D_RETURN_MAX, CANDIDATE_BOLLINGER_MAX, CANDIDATE_BOTTOM_MAX_52W_POS,
                                 CANDIDATE_RSI_MAX)

    rsi_max = 70.0 if regime == "bear" else float(CANDIDATE_RSI_MAX)
    rsi, bb, r5, pos = (_f(row.get(k)) for k in ("RSI", "bollinger_pband", "5일수익률", "52주포지션"))
    bottom = "바닥" in (strategy or "")
    out: list[dict[str, Any]] = []
    if rsi is not None and rsi >= rsi_max:
        out.append({"k": "rsi", "v": _r(rsi, 1), "lim": rsi_max})
    if bb is not None and bb >= CANDIDATE_BOLLINGER_MAX:
        out.append({"k": "bband", "v": _r(bb, 2), "lim": float(CANDIDATE_BOLLINGER_MAX)})
    if r5 is not None and r5 >= CANDIDATE_5D_RETURN_MAX:
        out.append({"k": "ret5", "v": _r(r5, 4), "lim": float(CANDIDATE_5D_RETURN_MAX)})
    if bottom and pos is not None and pos > CANDIDATE_BOTTOM_MAX_52W_POS:
        out.append({"k": "bottom_pos", "v": _r(pos, 2), "lim": float(CANDIDATE_BOTTOM_MAX_52W_POS)})
    if bottom and rsi is not None and 30 <= rsi < 35:
        out.append({"k": "bottom_rsi", "v": _r(rsi, 1), "lim": None})
    return out


def detail_live(tickers: Sequence[str], strategies: Mapping[str, str | None], bars: Mapping[str, Bar],
                fractions: Mapping[str, float | None], today: date) -> tuple[dict[str, dict[str, Any]], str | None]:
    """후보 · 보유 · SPY — 밤 피처 함수(compute_features_for_ticker)에 오늘 장중 봉을 넣어 RSI · 5일 · 52주 위치 · 볼린저 ·
    거래량 배수 · 이동평균 갭 · 이번 봉 교차 예고를 다시 계산한다. 오늘 봉 거래량은 5분봉 합 ÷ 시간 곡선 (장중 일봉 거래량은
    한참 작게 나와서). 돌려주는 것: ({티커: 숫자}, 레짐 — SPY 로 candidate_selector.detect_market_regime)."""
    from data.fetch import fetch_chart_prices, fetch_ohlcv
    from paper_trading.candidate_selector import detect_market_regime
    from screener.features import compute_features_for_ticker, feature_row_from_set

    want = list(dict.fromkeys([*tickers, "SPY"]))
    prices = fetch_ohlcv(want, period="5y")
    charts = fetch_chart_prices(want)
    have = set(prices.columns.get_level_values(0))
    rows: dict[str, dict[str, Any]] = {}
    for t in want:
        if t not in have:
            continue
        f = prices[t].dropna(how="all").copy()
        if f.empty or pd.Timestamp(f.index[-1]).date() != today:
            continue       # 오늘 봉이 아직 없다 → 밤 값 그대로
        bar, frac = bars.get(t), fractions.get("ca" if is_ca(t) else "us")
        f["Volume"] = f["Volume"].astype("float64")
        f.loc[f.index[-1], "Volume"] = bar.volume / frac if bar and frac else np.nan
        try:
            fs = compute_features_for_ticker(f, charts.get(t))
        except Exception as exc:  # noqa: BLE001
            print(f"[펄스] {t} 지표 계산 실패: {exc}")
            continue
        if fs is not None:
            rows[t] = feature_row_from_set(t, fs)
    regime = None
    if "SPY" in rows:
        r = rows["SPY"]
        regime = detect_market_regime(pd.DataFrame([{"티커": "SPY", "현재가격": r.get("close"), "ema50": r.get("ema50"),
                                                     "ema200": r.get("ema200")}]))
    out: dict[str, dict[str, Any]] = {}
    for t in tickers:
        r = rows.get(t)
        if not r:
            continue
        heat = heat_flags(r, strategies.get(t), regime)
        out[t] = {"rsi": _r(r.get("RSI"), 1), "ret_5d": _r(r.get("5일수익률"), 4),
                  "position_52w": _r(r.get("52주포지션"), 4), "bband": _r(r.get("bollinger_pband"), 3),
                  "vol_ratio": _r(r.get("거래량돌파배수"), 2),
                  "ma_gap": {"daily": _r(r.get("일봉_MA갭"), 4), "weekly": _r(r.get("주봉_MA갭"), 4),
                             "monthly": _r(r.get("월봉_MA갭"), 4)},
                  "preview": {"weekly": r.get("주봉_예고") or None, "monthly": r.get("월봉_예고") or None},
                  "heat": heat}
    return out, regime


# ── 숏스퀴즈 ──────────────────────────────────────────────────────────────────

def squeeze_live(watch: pd.DataFrame, bars: Mapping[str, Bar], fraction: float | None,
                 today: date) -> dict[str, Any] | None:
    """숏스퀴즈 감시 목록(밤 short_squeeze, fire · fuel)의 장중 상태 — price_trigger · classify (밤과 같은 함수).
    오늘 거래량은 5분봉 합 ÷ 시간 곡선. 오늘 봉이 없는 종목은 밤 상태 그대로."""
    from screener.short_squeeze import _yf_daily, classify, price_trigger

    if watch is None or watch.empty:
        return None
    tickers = [str(t) for t in watch["ticker"]]
    closes, vols = _yf_daily(tickers)
    items = []
    for r in watch.itertuples():
        t = str(r.ticker)
        status, price, ret5, vr = str(r.status), _f(r.price), _f(r.ret_5d), _f(r.vol_ratio)
        live = False
        if t in closes.columns:
            c = closes[t].dropna()
            if len(c) and pd.Timestamp(c.index[-1]).date() == today:
                v = vols[t].reindex(c.index).astype("float64").copy()
                bar = bars.get(t)
                v.iloc[-1] = bar.volume / fraction if bar and fraction else np.nan
                trig = price_trigger(c, v)
                new = classify(_f(r.short_pct) or float("nan"), _f(r.dtc) or float("nan"), trig["breakout"],
                               trig["vol_ratio"] if np.isfinite(trig["vol_ratio"]) else float("nan"))
                if new:
                    status, live = new, True
                price, ret5, vr = float(c.iloc[-1]), _f(trig["ret_5d"]), _f(trig["vol_ratio"])
        items.append({"ticker": t, "status": status, "live": live, "price": _px(price), "ret_5d": _r(ret5, 4),
                      "vol_ratio": _r(vr, 2)})
    return {"items": items, "fire": sum(1 for x in items if x["status"] == "fire"),
            "fuel": sum(1 for x in items if x["status"] == "fuel")}


# ── 한 번 실행 ────────────────────────────────────────────────────────────────

def _try(label: str, fn: Callable[[], Any], errors: list[str]) -> Any:
    """표시용 — 한 칸이 실패해도 나머지는 낸다."""
    t0 = _time.perf_counter()
    try:
        out = fn()
        print(f"[펄스] {label} {_time.perf_counter() - t0:.1f}초")
        return out
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{label}: {exc}")
        print(f"[펄스] {label} 실패: {exc}")
        return None


def _load_squeeze_watch(path: Path) -> pd.DataFrame:
    from screener.short_squeeze import load_watch

    return load_watch(str(path))


def run(out_dir: Path, *, base_path: Path | None = None, now: datetime | None = None, force: bool = False,
        scope: str = "all", dashboard: Mapping[str, Any] | None = None,
        universe: Mapping[str, Any] | None = None) -> int:
    """펄스 한 번. 0 = 파일을 씀 · 3 = 열린 시장이 없어 아무것도 안 씀 · 1 = 시세를 하나도 못 받음."""
    now = now or datetime.now(ET)
    today = now.date()
    sessions = {m: market_session(m, now) for m in ("us", "ca")}
    active = [m for m, s in sessions.items() if s.active(now)]
    print(f"[펄스] {now:%Y-%m-%d %H:%M} ET · 미국 {sessions['us'].state} · 캐나다 {sessions['ca'].state}")
    if not active and not force:
        print("[펄스] 열린 시장이 없음 — 아무것도 안 씀")
        return 3
    errors: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = now.isoformat(timespec="seconds")

    dash = dashboard if dashboard is not None else _try("dashboard.json", lambda: _get_json(DASHBOARD_URL), errors)
    dash = dash or {}
    watch = _try("숏스퀴즈 목록", lambda: _load_squeeze_watch(SQUEEZE_PATH), errors)
    if watch is None:
        watch = pd.DataFrame()
    held = [str(p["ticker"]) for a in dash.get("accounts") or [] for p in a.get("positions") or []]
    cand_rows = list(dash.get("candidates") or [])
    from data.etf_universe import etf_tickers

    extra = [*etf_tickers(), *held, *(c["ticker"] for c in cand_rows), *(dash.get("stocks") or {}),
             *(str(t) for t in (watch["ticker"] if not watch.empty else []))]

    base = load_base(base_path, today, scope)
    if base is None:
        uni = universe if universe is not None else _try("전체분석 목록", load_universe, errors)
        if not uni or not uni.get("meta"):
            print("[펄스] 종목 목록이 없어 기준값을 못 만듦")
            return 1
        tickers = coverage(uni, extra, scope)
        base = _try(f"기준값 {len(tickers)}종목 ({scope})", lambda: build_base(tickers, today, uni, now, scope), errors)
        if not base:
            return 1
    else:
        tickers = list(dict.fromkeys([*base["tickers"], *extra]))   # 오늘 새로 생긴 후보도
    print(f"[펄스] 기준값 {base['day']} · {len(base['t'])}종목 (만든 시각 {base['built_at']})")

    bars = _try(f"시세 {len(tickers)}종목", lambda: today_bars(tickers, today), errors) or {}
    if not bars:
        print("[펄스] 오늘 시세를 하나도 못 받음")
        return 1

    # 시장마다 지금이 정규장의 몇 분째인지 → 거래량 곡선. 마감 뒤(마감 펄스)는 동시호가까지 들어간 그날 거래량
    fractions: dict[str, float | None] = {}
    as_of: dict[str, str | None] = {}
    for m, s in sessions.items():
        if s.state in ("holiday", "pre"):
            fractions[m], as_of[m] = None, None
            continue
        end = min(now, s.close_at())
        elapsed = (end - datetime.combine(today, OPEN, tzinfo=ET)).total_seconds() / 60
        fractions[m] = AFTER_CLOSE_FRACTION if s.state == "closed" else volume_fraction(elapsed, s.minutes)
        as_of[m] = end.isoformat(timespec="minutes")

    rows = base["t"]
    quotes = {t: live_quote(rows.get(t), b, fractions["ca" if is_ca(t) else "us"]) for t, b in bars.items()
              if sessions["ca" if is_ca(t) else "us"].state not in ("holiday", "pre")}
    meta = base.get("meta") or {}
    day = today.isoformat()

    def idx(sym: str) -> dict[str, Any] | None:
        q = quotes.get(sym)
        return {"p": _px(q["p"]), "c": _r(q.get("c"), 4)} if q else None

    markets: dict[str, Any] = {}
    if quotes.get("SPY"):
        markets["us"] = _try("시장 한눈에 미국", lambda: market_live(
            quotes, meta, base.get("sp500") or [], [t for t in meta if not is_ca(t)], day, quotes["SPY"].get("c"),
            generated_at), errors)
    ca_sym = next((s for s in ("^GSPTSE", "XIC.TO") if quotes.get(s)), None)
    if ca_sym:
        from web_export.market_summary import CA_MIN_CAP, INDEX_LABEL

        markets["ca"] = _try("시장 한눈에 캐나다", lambda: market_live(
            quotes, meta, base.get("ca_breadth") or [], [t for t in meta if is_ca(t)], day, quotes[ca_sym].get("c"),
            generated_at, market="ca", index_label=INDEX_LABEL.get(ca_sym, "S&P/TSX"), min_cap=CA_MIN_CAP), errors)

    us_open = sessions["us"].state not in ("holiday", "pre") and bool(quotes.get("SPY"))
    macro = _try("경기 상태", lambda: macro_live(today, bars.get("CAD=X")), errors) if us_open else None
    etf = (_try("ETF", lambda: etf_live(today, quotes), errors) or {}) if us_open else {}
    positions = json.loads(POSITIONS_PATH.read_text()) if POSITIONS_PATH.exists() else []
    pt1 = _try("PT-1", lambda: pt1_live(dash, positions, quotes, today), errors) if us_open else None
    strategies = {c["ticker"]: c.get("strategy") for c in cand_rows}
    detail, regime = (_try("후보 지표", lambda: detail_live(list(dict.fromkeys([*(c["ticker"] for c in cand_rows),
                                                                                 *held])),
                                                            strategies, bars, fractions, today), errors)
                      or ({}, None)) if us_open else ({}, None)
    squeeze = _try("숏스퀴즈", lambda: squeeze_live(watch, bars, fractions.get("us"), today), errors) if us_open else None

    final = all(sessions[m].state != "open" for m in active) if active else True
    live = {
        "schema": SCHEMA_VERSION,
        "generated_at": generated_at,
        "day": day,
        "base_day": base.get("day"),
        "final": final,
        "session": {m: {"state": s.state, "close": s.close.strftime("%H:%M"), "as_of": as_of.get(m),
                        "pace": _r(fractions.get(m), 3)} for m, s in sessions.items()},
        "indices": {s: idx(s) for s in INDEX_SYMBOLS if idx(s)},
        "market": markets,
        "macro": macro,
        "etf": etf,
        "pt1": pt1,
        "regime": regime,
        "cands": detail,
        "squeeze": squeeze,
        "counts": {"tickers": len(tickers), "quotes": len(quotes), "base": len(rows)},
        "errors": errors,
    }
    qfile = {"schema": SCHEMA_VERSION, "generated_at": generated_at, "day": day, "final": final,
             "as_of": as_of.get("us") or as_of.get("ca"), "fields": ["p", "c", "v", "lo", "hi", "f"],
             "q": quote_list(quotes)}
    (out_dir / "live.json").write_text(json.dumps(_clean(live), ensure_ascii=False, allow_nan=False,
                                                  separators=(",", ":")), encoding="utf-8")
    (out_dir / "quotes.json").write_text(json.dumps(_clean(qfile), ensure_ascii=False, allow_nan=False,
                                                    separators=(",", ":")), encoding="utf-8")
    # 고아 브랜치는 매번 통째로 바뀌어서 기준값도 매번 같이 올린다 (같은 날은 같은 내용)
    (out_dir / "base.json.gz").write_bytes(gzip.compress(json.dumps(
        _clean(base), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")))
    sizes = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}
    b = (markets.get("us") or {}).get("breadth") or {}
    print(f"[펄스] 시세 {len(quotes)}종목 · 미국 오른 {b.get('adv')} 내린 {b.get('dec')} 신고가 {b.get('new_high')} · "
          f"PT-1 {((pt1 or {}).get('dollar') or {}).get('equity')} · 후보 {len(detail)} · 레짐 {regime} · "
          f"{'마감' if final else '장중'} · 파일 {sizes} · 오류 {len(errors)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="사이트 장중 숫자 (live.json · quotes.json · base.json.gz)")
    ap.add_argument("--out", type=Path, required=True, help="파일을 쓸 폴더")
    ap.add_argument("--base", type=Path, default=None, help="오늘 만든 기준값 (live-data 브랜치의 base.json.gz). "
                                                           "없으면 --out 폴더의 것, 그것도 없거나 다른 날이면 새로 만든다")
    ap.add_argument("--force", action="store_true", help="열린 시장이 없어도 실행 (시험용)")
    ap.add_argument("--scope", choices=("all", "cover"), default="all",
                    help="all = 전체분석 전부 (기본) · cover = S&P 500 + 시총 $2B+ + 캐나다 C$1B+ (Yahoo 가 막힐 때)")
    args = ap.parse_args(argv)
    base_path = args.base or (args.out / "base.json.gz")
    return run(args.out, base_path=base_path, force=args.force, scope=args.scope)


if __name__ == "__main__":
    sys.exit(main())
