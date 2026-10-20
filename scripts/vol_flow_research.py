#!/usr/bin/env python3
"""거래량·수급 신호 10년 연구 (vol_flow_v1) — 지금까지 한 번도 안 본 '선행' 쪽 신호 6개.

계기: 2026-10-20 사용자 — "후행성 지표 말고 거래량 등 선행성 지표로 추론하는 영역을 넓힐수있어?"
→ 연구 먼저 (라이브 점수 안 바꿈). A 가격·거래량 4종 + B FINRA 일일 공매도 거래량 2종.
이미 실패한 것(52주 신고가+거래량 2배, 갭+거래량 3배, vol_trend·OBV·CMF IC, 내부자, 애널리스트, 공매도 잔고,
스퀴즈)은 다시 보지 않는다.

가격 = data/research/tv_ohlcv_max.parquet (분할만 반영 OHLC + 분할 반영 거래량, 배당 조정 종가로 총수익 시가).
유니버스 = 그날 S&P 500 구성 종목 (PIT). 신호는 t 종가까지의 데이터만 → t+1 시가 진입 → t+1+h 시가 청산,
같은 구간 구성 종목 동일가중 대비 초과수익, 비용 왕복 0.2% (피하기 검정은 비용 없음).

사전 등록 (family vol_flow_v1, 6건, Bonferroni z 는 --count 뒤 family 크기로):
  A1-UDV50|h63  상승/하락일 거래량 비율: 50일 Σ(전일 대비 오른 날 거래량) / Σ(내린 날 거래량), 보합일 제외,
                유효일 ≥ 40. 월말 단면 상위 20% − 신호 있는 구성 종목 평균 (idea_batch_v2.xs_test).
  A2-HVP|h21    고거래량 프리미엄 (Gervais·Kaniel·Mingelgrin 2001): 주봉(월~금, 거래일 ≥ 3) 거래량이 직전 10주 최대
                이상 & 그 주 종가 수익률 |r| ≤ 3% (가격은 거의 그대로인데 거래량만 터진 주). 사건일 = 그 주 마지막
                거래일. 같은 종목 4주 쿨다운.
  A3-DRYBO|h63  거래량 고갈 뒤 돌파: 직전 10일(t−10~t−1) 중 하루라도 MA5(V)/MA50(V) ≤ 0.6 · 종가 > 직전 20일 최고 종가
                · V ≥ 1.5 × MA50(V)(어제까지) · 종가 > SMA50. 10일 쿨다운.
  A4-BDIV|h63   바닥권 매집 다이버전스: 종가 ≤ 60일 최저가 × 1.05 · 종가 < SMA200 · CMF20 ≥ +0.05. 10일 쿨다운.
                (참고 대조군: 같은 바닥권인데 CMF20 ≤ −0.05)
  B1-SVR20|h21  (피하기) FINRA 공매도 거래량 비율 20일 Σ ShortVolume / Σ TotalVolume (유효일 ≥ 15), 월말 상위 20%.
  B2-DSVR|h21   (피하기) 공매도 비율 급등: SVR5(유효 ≥ 4) − SVR60(유효 ≥ 50), 월말 상위 20%.
  이벤트 검정(A2~A4)은 사건 ≥ 300 & 사건 날짜 ≥ 100 일 때만 family 에 남는다 (sqz 와 같음, 수익 보기 전 --count 로 확정).

판정:
  1) 개발 2016-01-04 ~ 2025-09-30: 후보 = t ≥ z · 비용 후 > 0 · 앞/뒤 반 모두 + · 연도 2/3 이상 +
     (pattern_confirm_research.grade · 피하기는 idea_batch_v2.judge 로 부호 뒤집어 같은 기준).
  2) 확인 2006-01-03 ~ 2015-12-31 (A 만): 1) 후보만 판정 — 같은 부호 · t ≥ 1.96 · 비용 후 > 0.
     family vol_flow_confirm_v1 (n = 후보 수). 후보가 아닌 A 신호의 확인 구간 값은 '참고'로만 적는다.
  3) B 는 FINRA 일일 파일이 2018-08 부터라 확인 구간이 없다 → '후보'여도 '관찰'까지.
  4) 확인까지 통과한 것만 PT-1 결합을 따로 사전 등록·승인 후 진행 (이 스크립트 범위 밖).
  홀드아웃(2025-10-01~)은 쓰지 않는다.

FINRA: https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt (Date|Symbol|ShortVolume|ShortExemptVolume|
  TotalVolume|Market, 마지막 줄은 건수). FINRA 에 보고된 장외 거래만이라 전체 거래량이 아니다. 심볼 BRK/B → BRK-B.

사용법:
  PYTHONPATH=.:src .venv/bin/python scripts/vol_flow_research.py --fetch-finra   # 한 번 (중간에 끊겨도 이어 받음)
  PYTHONPATH=.:src .venv/bin/python scripts/vol_flow_research.py --count         # 사건 수·커버리지만 (수익 안 봄)
  PYTHONPATH=.:src .venv/bin/python scripts/vol_flow_research.py --register
  PYTHONPATH=.:src .venv/bin/python scripts/vol_flow_research.py --run           # 개발 구간 + 확인 구간 + 보고서
  PYTHONPATH=.:src .venv/bin/python scripts/vol_flow_research.py --report        # 저장된 표로 보고서만 다시
산출: tier3/VOL_FLOW_REPORT.md · tier3/vol_flow_{counts,tests,confirm,placebo}.csv · data/research/vf_* (git 제외)
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from idea_batch_research import member_mask  # noqa: E402
from idea_batch_v2 import judge, xs_test  # noqa: E402
from ma_cross_research import PERIODS, TV_OHLCV, _cooldown, excess_matrix, membership  # noqa: E402
from pattern_confirm_research import grade, stats  # noqa: E402
from research_utils import (DEV_END, RESEARCH_DIR, TIER3_DIR, TRIAL_LOG, bonferroni_z, load_ohlcv,  # noqa: E402
                            log_trials, total_trials)

STEP = "vol_flow"
FAMILY = "vol_flow_v1"
FAMILY_CONFIRM = "vol_flow_confirm_v1"
LOAD_START = "2004-01-01"           # 확인 구간 SMA200 워밍업
FINRA_START = "2018-08-01"          # cdn 에 일일 파일이 있는 첫 달 (2018-07 이전은 403)
FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
FINRA_PATH = RESEARCH_DIR / "vf_finra_shvol.parquet"
MIN_EVENTS, MIN_DAYS = 300, 100
CONFIRM_Z = 1.96
HORIZONS = (21, 63)
N_PLACEBO = 20

# 지표 상수 (사전 등록 — 결과 보고 바꾸지 않는다)
UDV_N, UDV_MIN = 50, 40
HVP_LOOKBACK_W, HVP_MAX_ABS_RET, HVP_MIN_DAYS, HVP_COOLDOWN_W = 10, 0.03, 3, 4
DRY_RATIO, DRY_WINDOW, BO_LOOKBACK, BO_VOL_MULT = 0.6, 10, 20, 1.5
BDIV_LOW_N, BDIV_BAND, BDIV_CMF = 60, 1.05, 0.05
CMF_N = 20
EVENT_COOLDOWN = 10
SVR_N, SVR_MIN = 20, 15
DSVR_SHORT, DSVR_SHORT_MIN, DSVR_LONG, DSVR_LONG_MIN = 5, 4, 60, 50


# (test_id, 종류, 방향, 주 보유기간, 참고 보유기간, 묶음, 시작) 종류: xs = 월말 상위 20% · ev = 이벤트
TESTS: list[tuple[str, str, str, int, int, str, str]] = [
    ("A1-UDV50", "xs", "buy", 63, 21, "A", "2016-01-01"),
    ("A2-HVP", "ev", "buy", 21, 63, "A", "2016-01-01"),
    ("A3-DRYBO", "ev", "buy", 63, 21, "A", "2016-01-01"),
    ("A4-BDIV", "ev", "buy", 63, 21, "A", "2016-01-01"),
    ("B1-SVR20", "xs", "avoid", 21, 63, "B", "2018-09-01"),
    ("B2-DSVR", "xs", "avoid", 21, 63, "B", "2018-11-01"),
]
NAMES = {
    "A1-UDV50": "상승/하락일 거래량 비율 50일 상위 20% (매집)",
    "A2-HVP": "가격은 그대로인데 주간 거래량 10주 최대 (고거래량 프리미엄)",
    "A3-DRYBO": "거래량 고갈 뒤 20일 고점 돌파 + 거래량 1.5배",
    "A4-BDIV": "60일 저점권·200일선 아래인데 CMF20 ≥ 0.05 (바닥 매집)",
    "B1-SVR20": "공매도 거래량 비율 20일 상위 20% (피하기)",
    "B2-DSVR": "공매도 비율 급등 (5일 − 60일) 상위 20% (피하기)",
    "A4-BDIV-대조": "같은 바닥권인데 CMF20 ≤ −0.05 (분산, 대조군)",
}

# 2026-10-20 실행 결과를 보고 쓴 쉬운 말 요약 (--report 가 그대로 넣는다)
SUMMARY = [
    "1. **6개 모두 탈락.** 같은 날 S&P 500 평균보다 나은 신호가 하나도 없다. 가장 큰 |t| 가 1.58 (기준 2.64).",
    "2. **A1 매집 (오른 날 거래량 > 내린 날 거래량)**: 상위 20% 가 63일 −0.16% (t −0.51). 2006~2015 도 −0.32%. "
    "거래량이 오른 날에 몰린 종목이 그 뒤에 더 오르지 않는다.",
    "3. **A2 고거래량 프리미엄 (가격은 그대로인데 거래량만 10주 최대)**: 21일 −0.16% (t −1.11), 2006~2015 도 0 근처 (t +0.23). "
    "1963~1996년 NYSE 데이터로 나온 논문 효과가 요즘 대형주에서는 보이지 않는다.",
    "4. **A3 거래량 고갈 뒤 돌파 (VCP·포켓 피벗 류)**: 63일 t −0.97, 21일은 오히려 −0.33% (t −2.06, 참고) · 10년 중 4년만 +. "
    "앞서 탈락한 '52주 신고가 + 거래량 2배' 와 같은 결론 — S&P 500 에서 거래량 터진 돌파는 단기 되돌림이 더 흔하다.",
    "5. **A4 바닥 매집 (저점권인데 CMF 가 +)**: 사건 평균은 +0.28% 지만, 사건의 11% 가 급락일 19일 "
    "(2016-01 · 2020-03 · 2022-06 · 2025-04-03 등)에 몰려 그날들 평균이 +2.7% 인 탓이다. 나머지 날은 평균 −0.35%, "
    "날짜 평균으로 보면 t −0.86. 같은 바닥권에서 CMF 가 마이너스(분산)인 종목과도 차이가 없다 (③) — "
    "'바닥에서 돈이 들어온다'는 CMF 판단이 수익을 가르지 못한다.",
    "6. **B FINRA 일일 공매도 거래량**: 비율이 높은 종목(B1)은 오히려 약간 + (t +0.46), 급등(B2)은 약간 − (t −1.58) — "
    "둘 다 기준에 한참 못 미친다. 이 데이터는 장외 거래만이고 시장조성자 매매가 많이 섞여 있어 '공매도 세력' 신호로 읽기 어렵다.",
    "7. **계산 점검**: 같은 날 같은 수를 무작위로 뽑은 가짜 사건은 t 평균 −0.25 ~ +0.49 (④) → 계산은 정상. "
    "다만 NW t 의 흩어짐이 정규분포보다 약간 넓어 (무작위인데 최대 |t| 3.07), 기준 근처 결과는 더 조심해야 한다.",
    "8. **그래서**: 거래량·수급 지표를 점수·CCS 에 더 넣지 않는다. PT-1 결합 단계로 올릴 후보가 없다. "
    "지금 CCS 안의 거래량 팩터(OBV·CMF·거래량 z)도 IC ≈ 0 이지만, 빼는 것도 규칙 변경이라 그대로 둔다.",
    "9. **남은 길**: 대형주(S&P 500)는 거래량 정보가 이미 가격에 빨리 반영된다. 거래량·수급 신호가 살아 있을 가능성이 큰 곳은 "
    "중소형주인데, 상장폐지 종목까지 든 데이터(Sharadar 등)가 있어야 제대로 볼 수 있다.",
]


def test_id(name: str, h: int) -> str:
    return f"{name}|h{h}"


# ── 지표 (날짜 × 종목 DataFrame, t 까지의 데이터만) ─────────────────────────────────────


def udv_ratio(close: pd.DataFrame, vol: pd.DataFrame, n: int = UDV_N, min_days: int = UDV_MIN) -> pd.DataFrame:
    """n일 Σ상승일 거래량 / Σ하락일 거래량. 보합일·결측 제외, 유효일 < min_days 면 NaN, 하락 거래량 0 이면 NaN."""
    d = close.diff()
    up = vol.where(d > 0, 0.0).where(d.notna() & vol.notna())
    dn = vol.where(d < 0, 0.0).where(d.notna() & vol.notna())
    valid = ((d != 0) & d.notna() & vol.notna()).astype(float).rolling(n, min_periods=1).sum()
    us = up.rolling(n, min_periods=1).sum()
    ds = dn.rolling(n, min_periods=1).sum()
    out = us / ds.where(ds > 0)
    return out.where(valid >= min_days)


def cmf(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame, n: int = CMF_N) -> pd.DataFrame:
    """Chaikin Money Flow: Σ((C−L)−(H−C))/(H−L)·V / ΣV, H == L 인 날은 0."""
    rng = high - low
    mfm = ((close - low) - (high - close)) / rng.where(rng > 0)
    mfm = mfm.where(rng > 0, 0.0).where(close.notna())
    mfv = mfm * vol
    m = n - 2  # 거래량 결측 하루 이틀은 허용
    return mfv.rolling(n, min_periods=m).sum() / vol.rolling(n, min_periods=m).sum().where(lambda v: v > 0)


def weekly_bars(close: pd.DataFrame, vol: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(주 거래량 합, 주 마지막 종가, 주 거래일 수) — 인덱스 = 그 주 마지막 거래일."""
    wk = close.index.to_period("W-FRI")
    last_day = pd.Series(close.index, index=close.index).groupby(wk).max()
    vw = vol.groupby(wk).sum(min_count=1)
    cw = close.groupby(wk).last()
    nd = vol.notna().groupby(wk).sum()
    for f in (vw, cw, nd):
        f.index = pd.DatetimeIndex(last_day.loc[f.index].to_numpy())
    return vw, cw, nd


def hvp_flags(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """주봉 사건: 거래량 ≥ 직전 10주 최대 & |주 수익률| ≤ 3% & 거래일 ≥ 3 (인덱스 = 주 마지막 거래일)."""
    vw, cw, nd = weekly_bars(close, vol)
    prev_max = vw.shift(1).rolling(HVP_LOOKBACK_W, min_periods=HVP_LOOKBACK_W).max()
    ret = cw / cw.shift(1) - 1
    return (vw >= prev_max) & (vw > 0) & (ret.abs() <= HVP_MAX_ABS_RET) & (nd >= HVP_MIN_DAYS)


def drybo_flags(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """거래량 고갈(직전 10일 중 MA5/MA50 ≤ 0.6) 뒤 20일 고점 돌파 + V ≥ 1.5×MA50(어제까지) + 종가 > SMA50."""
    ma5 = vol.rolling(5, min_periods=4).mean()      # 거래량 0/결측 하루는 빼고 평균 (0.4% 정도)
    ma50 = vol.rolling(50, min_periods=45).mean()
    dry = (ma5 / ma50.where(ma50 > 0)).shift(1).rolling(DRY_WINDOW, min_periods=DRY_WINDOW).min() <= DRY_RATIO
    breakout = close > close.shift(1).rolling(BO_LOOKBACK, min_periods=BO_LOOKBACK).max()
    surge = vol >= BO_VOL_MULT * ma50.shift(1)
    trend = close > close.rolling(50, min_periods=50).mean()
    return dry & breakout & surge & trend


def bdiv_flags(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame,
               sign: int = 1) -> pd.DataFrame:
    """60일 저점 × 1.05 이내 · 종가 < SMA200 · CMF20 ≥ +0.05 (sign=−1 이면 대조군 CMF20 ≤ −0.05)."""
    near_low = close <= BDIV_BAND * low.rolling(BDIV_LOW_N, min_periods=BDIV_LOW_N).min()
    below = close < close.rolling(200, min_periods=200).mean()
    c = cmf(high, low, close, vol)
    flow = c >= BDIV_CMF if sign > 0 else c <= -BDIV_CMF
    return near_low & below & flow


def svr_signals(sv: pd.DataFrame, tv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(SVR20, DSVR = SVR5 − SVR60). 공매도 거래량 / FINRA 보고 거래량, 유효일 부족이면 NaN."""
    def ratio(n: int, m: int) -> pd.DataFrame:
        s = sv.rolling(n, min_periods=m).sum()
        t = tv.rolling(n, min_periods=m).sum()
        return s / t.where(t > 0)
    return ratio(SVR_N, SVR_MIN), ratio(DSVR_SHORT, DSVR_SHORT_MIN) - ratio(DSVR_LONG, DSVR_LONG_MIN)


def flag_events(flags: pd.DataFrame, cooldown: int) -> pd.DataFrame:
    """bool 행렬 → (date, 티커) 사건 표. 같은 종목은 cooldown 봉 안 재사건 제외 (봉 = flags 의 행)."""
    rows = []
    arr = flags.fillna(False).to_numpy(dtype=bool)
    for j, t in enumerate(flags.columns):
        pos = np.flatnonzero(arr[:, j])
        if len(pos):
            pos = pos[_cooldown(pos, cooldown)]
            rows.append(pd.DataFrame({"date": flags.index[pos], "티커": t}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "티커"])


def norm_symbol(s: pd.Series) -> pd.Series:
    """FINRA 심볼 → yfinance 표기 (BRK/B → BRK-B, BF.B → BF-B)."""
    return s.astype(str).str.strip().str.upper().str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)


def parse_finra(text: str, keep: set[str] | None = None) -> pd.DataFrame:
    """CNMS 일일 파일 → (date, 티커, sv, tv). 마지막 건수 줄·빈 심볼은 버린다."""
    df = pd.read_csv(io.StringIO(text), sep="|", dtype={"Symbol": str, "Date": str})
    df = df.dropna(subset=["Symbol", "TotalVolume"])
    df = df[df["Date"].str.fullmatch(r"\d{8}")]
    out = pd.DataFrame({"date": pd.to_datetime(df["Date"], format="%Y%m%d"), "티커": norm_symbol(df["Symbol"]),
                        "sv": pd.to_numeric(df["ShortVolume"], errors="coerce"),
                        "tv": pd.to_numeric(df["TotalVolume"], errors="coerce")})
    if keep is not None:
        out = out[out["티커"].isin(keep)]
    return out.groupby(["date", "티커"], as_index=False)[["sv", "tv"]].sum()


# ── 데이터 ─────────────────────────────────────────────────────────────────────────────


def load_data() -> dict[str, pd.DataFrame]:
    """2004 ~ DEV_END, 2005 이후 한 번이라도 S&P 500 이었던 종목만. 필드별 날짜 × 종목 + 총수익 시가 + 구성 마스크."""
    px = load_ohlcv(TV_OHLCV).loc[LOAD_START:]
    ever = membership().union_between("2005-01-01", str(DEV_END.date()))
    cols = [t for t in dict.fromkeys(px.columns.get_level_values("Ticker")) if t in ever]
    px = px.loc[:, px.columns.get_level_values("Ticker").isin(cols)]
    f = {k: px.xs(k, axis=1, level="Price").astype("float64").reindex(columns=cols)
         for k in ("Open", "High", "Low", "Close", "Adj Close", "Volume")}
    del px
    f["Volume"] = f["Volume"].where(f["Volume"] > 0)
    f["open_tr"] = f["Open"] * (f["Adj Close"] / f["Close"])
    f["mem"] = member_mask(f["Close"].index, f["Close"].columns, membership())
    return f


def load_finra(index: pd.DatetimeIndex, columns: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(FINRA_PATH)
    sv = raw.pivot_table(index="date", columns="티커", values="sv", aggfunc="sum")
    tv = raw.pivot_table(index="date", columns="티커", values="tv", aggfunc="sum")
    days = index[index >= FINRA_START]
    return sv.reindex(index=days, columns=columns), tv.reindex(index=days, columns=columns)


def fetch_finra() -> None:
    """2018-08 ~ DEV_END 거래일마다 CNMS 파일 → 구성 종목만 남겨 parquet. 이미 받은 날은 건너뛴다."""
    f = load_data()
    keep = set(f["Close"].columns)
    days = [d for d in f["Close"].index if d >= pd.Timestamp(FINRA_START)]
    have = pd.read_parquet(FINRA_PATH) if FINRA_PATH.exists() else pd.DataFrame(columns=["date", "티커", "sv", "tv"])
    done = set(pd.to_datetime(have["date"]).unique())
    todo = [d for d in days if d not in done]
    print(f"[finra] 거래일 {len(days)} · 이미 {len(done)} · 받을 {len(todo)} · 종목 {len(keep)}", flush=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (research; vol_flow_research.py)"
    frames, missing = [have], []
    for k, d in enumerate(todo, 1):
        url = FINRA_URL.format(d=f"{d:%Y%m%d}")
        for attempt in range(3):
            try:
                r = sess.get(url, timeout=30)
                break
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        else:
            missing.append(d)
            continue
        if r.status_code != 200:
            missing.append(d)
        else:
            frames.append(parse_finra(r.text, keep))
        if k % 100 == 0 or k == len(todo):
            pd.concat(frames, ignore_index=True).to_parquet(FINRA_PATH, index=False)
            print(f"[finra] {k}/{len(todo)} · 없음 {len(missing)}", flush=True)
        time.sleep(0.2)
    if missing:
        print(f"[finra] 파일 없는 날 {len(missing)}: {[f'{d:%Y-%m-%d}' for d in missing[:20]]}", flush=True)


# ── 신호 표 ────────────────────────────────────────────────────────────────────────────


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


def xs_frame(signals: dict[str, pd.DataFrame], f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """월말 × 구성 종목: 신호값 + fwd_21·fwd_63 (다음날 시가 → h 거래일 뒤 시가)."""
    me = month_ends(f["Close"].index)
    mem = f["mem"].loc[me]
    key = mem.stack()
    key = key[key].index
    x = pd.DataFrame(index=key)
    for name, s in signals.items():
        x[name] = s.reindex(index=me, columns=f["Close"].columns).stack(future_stack=True).reindex(key)
    for h in HORIZONS:
        fwd = f["open_tr"].shift(-(1 + h)) / f["open_tr"].shift(-1) - 1
        x[f"fwd_{h}"] = fwd.loc[me].stack(future_stack=True).reindex(key)
    x.index.names = ["date", "티커"]
    return x.reset_index()


def build_events(f: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    c, v = f["Close"], f["Volume"]
    return {
        "A2-HVP": flag_events(hvp_flags(c, v), HVP_COOLDOWN_W),
        "A3-DRYBO": flag_events(drybo_flags(c, v), EVENT_COOLDOWN),
        "A4-BDIV": flag_events(bdiv_flags(f["High"], f["Low"], c, v), EVENT_COOLDOWN),
        "A4-BDIV-대조": flag_events(bdiv_flags(f["High"], f["Low"], c, v, sign=-1), EVENT_COOLDOWN),
    }


def in_window(ev: pd.DataFrame, mem: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """기간 안 + 그날 구성 종목인 사건만."""
    e = ev[(ev["date"] >= a) & (ev["date"] <= b)]
    ok = mem.stack()
    ok = ok[ok].index
    return e[pd.MultiIndex.from_arrays([e["date"], e["티커"]]).isin(ok)]


def event_stats(ev: pd.DataFrame, ar: pd.Series, h: int, bar_days: int = 1) -> dict:
    key = pd.MultiIndex.from_arrays([ev["date"], ev["티커"]])
    s = ar.reindex(key)
    s.index = key
    return stats(s, math.ceil(h / bar_days))


def placebo(ev: pd.DataFrame, ar: pd.Series, mem: pd.DataFrame, h: int, bar_days: int, seed: int) -> dict:
    """같은 날 같은 수만큼 구성 종목을 무작위로 뽑은 가짜 사건 (누수·코딩 오류 점검 — t ≈ 0 이어야 한다)."""
    rng = np.random.default_rng(seed)
    cols = mem.columns.to_numpy()
    rows = []
    for d, n in ev.groupby("date").size().items():
        pool = cols[mem.loc[d].to_numpy()]
        if len(pool):
            rows.append(pd.DataFrame({"date": d, "티커": rng.choice(pool, size=min(n, len(pool)), replace=False)}))
    return event_stats(pd.concat(rows, ignore_index=True), ar, h, bar_days) if rows else {}


# ── 실행 ───────────────────────────────────────────────────────────────────────────────


def _log() -> pd.DataFrame:
    return pd.read_csv(TRIAL_LOG, encoding="utf-8-sig") if TRIAL_LOG.exists() else pd.DataFrame()


def _registered(family: str) -> pd.Series | None:
    log = _log()
    if log.empty:
        return None
    hit = log[(log["family"] == family) & (log["판정"] == "등록")]
    return hit.iloc[-1] if len(hit) else None


def prepare() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    t0 = time.time()
    f = load_data()
    print(f"[data] {f['Close'].shape[1]}종목 × {f['Close'].shape[0]}일 ({time.time() - t0:.0f}s)", flush=True)
    sigs = {"A1-UDV50": udv_ratio(f["Close"], f["Volume"])}
    if FINRA_PATH.exists():
        sv, tv = load_finra(f["Close"].index, f["Close"].columns)
        sigs["B1-SVR20"], sigs["B2-DSVR"] = svr_signals(sv, tv)
    x = xs_frame(sigs, f)
    ev = build_events(f)
    print(f"[signals] 준비 끝 ({time.time() - t0:.0f}s)", flush=True)
    return f, x, ev


def count_table(f: dict[str, pd.DataFrame], x: pd.DataFrame, ev: dict[str, pd.DataFrame]) -> pd.DataFrame:
    a, b = PERIODS["dev"]
    rows = []
    for name, kind, _, h, _, grp, start in TESTS:
        if kind == "ev":
            e = in_window(ev[name], f["mem"], max(a, start), b)
            per_year = e.groupby(e["date"].dt.year).size()
            rows.append({"test_id": test_id(name, h), "n_events": len(e), "n_days": e["date"].nunique(),
                         "per_year": ";".join(f"{y}:{n}" for y, n in per_year.items()),
                         "family": len(e) >= MIN_EVENTS and e["date"].nunique() >= MIN_DAYS})
        else:
            d = x[(x["date"] >= start) & (x["date"] <= b)].dropna(subset=[name])
            months = d.groupby("date").size()
            rows.append({"test_id": test_id(name, h), "n_events": int(months.sum()), "n_days": int((months >= 50).sum()),
                         "per_year": ";".join(f"{y}:{n:.0f}" for y, n in
                                              months.groupby(months.index.year).mean().items()),
                         "family": int((months >= 50).sum()) >= 24})
    return pd.DataFrame(rows)


def coverage_table(f: dict[str, pd.DataFrame], x: pd.DataFrame) -> pd.DataFrame:
    """연도별: 그날 구성 종목 수(명단) · 가격·거래량 있는 구성 종목 수 · 신호 있는 비율."""
    mlist = membership()
    rows = []
    for y in range(2006, DEV_END.year + 1):
        d = f["Close"].index[f["Close"].index.year == y]
        mid = d[len(d) // 2]
        listed = len(mlist.on(f"{mid:%Y-%m-%d}"))
        have = int((f["mem"].loc[mid] & f["Close"].loc[mid].notna() & f["Volume"].loc[mid].notna()).sum())
        xy = x[x["date"].dt.year == y]
        rows.append({"연도": y, "명단": listed, "가격있음": have,
                     **{k: float(xy[k].notna().mean()) if k in xy and len(xy) else np.nan
                        for k in ("A1-UDV50", "B1-SVR20", "B2-DSVR")}})
    return pd.DataFrame(rows)


def do_count() -> None:
    f, x, ev = prepare()
    ct = count_table(f, x, ev)
    ct.to_csv(TIER3_DIR / "vol_flow_counts.csv", index=False, encoding="utf-8-sig")
    cov = coverage_table(f, x)
    cov.to_csv(TIER3_DIR / "vol_flow_coverage.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(ct.to_string(index=False))
    print(cov.round(2).to_string(index=False))
    n = int(ct["family"].sum())
    print(f"\nfamily {n} / {len(ct)} · Bonferroni z = {bonferroni_z(n):.2f}")


def do_register() -> None:
    if _registered(FAMILY) is not None:
        raise SystemExit(f"{FAMILY} 은 이미 등록됐다")
    ct = pd.read_csv(TIER3_DIR / "vol_flow_counts.csv", encoding="utf-8-sig")
    fam = ct[ct["family"]]
    z = bonferroni_z(len(fam))
    log_trials(STEP, FAMILY, len(fam), [{"test_id": "(등록)", "판정": "등록", "메모": (
        f"{len(fam)}건 z={z:.2f}: " + ",".join(fam["test_id"]) +
        " | 정의 scripts/vol_flow_research.py 머리말 (UDV50 50일·유효40 월말 상위20% / HVP 주봉 10주 최대·|r|≤3%·쿨다운4주 / "
        "DRYBO MA5/MA50≤0.6 직전10일·20일 고점·1.5×MA50·>SMA50 / BDIV 60일저점×1.05·<SMA200·CMF20≥0.05 / "
        "SVR20 20일 Σsv/Σtv 상위20% 피하기 / DSVR SVR5−SVR60 상위20% 피하기). 개발 2016-01~2025-09, "
        "A 후보만 2006~2015 확인(같은 부호·t≥1.96·비용 후>0). B 는 확인 구간 없음 → 최대 관찰")}])
    print(f"등록: {FAMILY} {len(fam)}건 z={z:.2f} · 누적 시도 {total_trials()}")


def _row(name: str, h: int, kind: str, direction: str, grp: str, period: str, r: dict, verdict: str) -> dict:
    mean = r.get("mean_ar", r.get("mean_ex", np.nan)) if r else np.nan
    return {"test_id": test_id(name, h), "name": name, "kind": kind, "direction": direction, "묶음": grp,
            "period": period, "h": h, "n_obs": r.get("n_days", r.get("n_obs", np.nan)) if r else np.nan,
            "n_events": r.get("n_events", r.get("n_avg", np.nan)) if r else np.nan, "mean": mean,
            "mean_net": r.get("mean_net", np.nan) if r else np.nan, "t_nw": r.get("t_nw", np.nan) if r else np.nan,
            "hit": r.get("hit", np.nan) if r else np.nan, "years_pos": r.get("years_pos", np.nan) if r else np.nan,
            "n_years": r.get("n_years", np.nan) if r else np.nan,
            "first_half": r.get("first_half", np.nan) if r else np.nan,
            "second_half": r.get("second_half", np.nan) if r else np.nan,
            "yearly": r.get("yearly", "") if r else "", "판정": verdict}


def _verdict_ev(r: dict, z: float, direction: str) -> str:
    if direction == "buy":
        return grade(r, z)
    return judge({**r, "mean_ex": r.get("mean_ar", np.nan)}, z, direction) if r else "표본 없음"


def confirm_grade(r: dict) -> str:
    if not r or not np.isfinite(r.get("t_nw", np.nan)):
        return "표본 없음"
    return "확인 통과" if r["t_nw"] >= CONFIRM_Z and r["mean_net"] > 0 else "확인 실패"


def do_run() -> None:
    reg = _registered(FAMILY)
    if reg is None:
        raise SystemExit("등록 먼저 (--count → --register)")
    family = set(str(reg["메모"]).split(": ", 1)[1].split(" | ", 1)[0].split(","))
    z = bonferroni_z(len(family))
    f, x, ev = prepare()
    mem = f["mem"]
    ars = {h: excess_matrix(f["open_tr"], mem, h).stack() for h in HORIZONS}

    rows, conf, plc = [], [], []
    for name, kind, direction, h_main, h_ref, grp, start in TESTS:
        for period in ("dev", "confirm"):
            if period == "confirm" and grp == "B":
                continue
            a, b = PERIODS[period]
            a = max(a, start) if period == "dev" else a
            for h in (h_main, h_ref):
                if kind == "xs":
                    r, _ = xs_test(x, name, h, a, pd.Timestamp(b))
                    r = r or {}
                    v_buy = judge(r, z, direction) if r else "표본 없음"
                else:
                    e = in_window(ev[name], mem, a, b)
                    r = event_stats(e, ars[h], h, 5 if name == "A2-HVP" else 1)
                    v_buy = _verdict_ev(r, z, direction)
                tid = test_id(name, h)
                main = h == h_main and tid in family
                if period == "dev":
                    rows.append(_row(name, h, kind, direction, grp, period, r, v_buy if main else "참고"))
                else:
                    conf.append(_row(name, h, kind, direction, grp, period, r, confirm_grade(r) if h == h_main else "참고"))
                if kind == "ev" and h == h_main:
                    ts = [placebo(e, ars[h], mem.loc[a:b], h, 5 if name == "A2-HVP" else 1, s).get("t_nw", np.nan)
                          for s in range(N_PLACEBO)]
                    plc.append({"test_id": tid, "period": period, "t_mean": float(np.nanmean(ts)),
                                "t_sd": float(np.nanstd(ts)), "t_max_abs": float(np.nanmax(np.abs(ts)))})
            print(f"[{period}] {test_id(name, h_main)} 끝", flush=True)

    # 참고: A4 대조군 (같은 바닥권, CMF20 ≤ −0.05)
    ref = []
    for period in ("dev", "confirm"):
        a, b = PERIODS[period]
        for nm in ("A4-BDIV", "A4-BDIV-대조"):
            r = event_stats(in_window(ev[nm], mem, a, b), ars[63], 63)
            ref.append(_row(nm, 63, "ev", "buy", "A", period, r, "참고"))

    res = pd.DataFrame(rows)
    cf = pd.DataFrame(conf)
    cands = res[(res["판정"] == "후보") & (res["묶음"] == "A")]["test_id"].tolist()
    cf.loc[(cf["판정"] != "참고") & ~cf["test_id"].isin(cands), "판정"] = "참고(개발 후보 아님)"
    res.to_csv(TIER3_DIR / "vol_flow_tests.csv", index=False, encoding="utf-8-sig")
    cf.to_csv(TIER3_DIR / "vol_flow_confirm.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(plc).to_csv(TIER3_DIR / "vol_flow_placebo.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ref).to_csv(TIER3_DIR / "vol_flow_bdiv_control.csv", index=False, encoding="utf-8-sig")
    cov = coverage_table(f, x)
    cov.to_csv(TIER3_DIR / "vol_flow_coverage.csv", index=False, encoding="utf-8-sig")

    judged = res[res["판정"] != "참고"]
    log_trials(STEP, FAMILY, len(family), [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}" if pd.notna(r.t_nw) else "",
                                            "판정": r["판정"]} for _, r in judged.iterrows()])
    if cands:
        log_trials(STEP, FAMILY_CONFIRM, len(cands), [{"test_id": "(등록)", "판정": "등록", "메모": (
            f"개발 후보 {len(cands)}건 2006-01~2015-12 확인: 같은 부호·t≥{CONFIRM_Z}·비용 후>0")}])
        ok = cf[cf["test_id"].isin(cands)]
        log_trials(STEP, FAMILY_CONFIRM, len(cands), [{"test_id": r.test_id, "t": f"{r.t_nw:.2f}", "판정": r["판정"]}
                                                       for _, r in ok.iterrows()])
    write_report(res, cf, pd.DataFrame(plc), pd.DataFrame(ref), cov, z, len(family), cands)


def do_report() -> None:
    """저장된 표로 보고서만 다시 쓴다 (TRIAL_LOG 안 건드림)."""
    reg = _registered(FAMILY)
    family = str(reg["메모"]).split(": ", 1)[1].split(" | ", 1)[0].split(",")
    rd = {k: pd.read_csv(TIER3_DIR / f"vol_flow_{k}.csv", encoding="utf-8-sig")
          for k in ("tests", "confirm", "placebo", "bdiv_control", "coverage")}
    res = rd["tests"]
    cands = res[(res["판정"] == "후보") & (res["묶음"] == "A")]["test_id"].tolist()
    write_report(res, rd["confirm"], rd["placebo"], rd["bdiv_control"], rd["coverage"], bonferroni_z(len(family)),
                 len(family), cands)


# ── 보고서 ─────────────────────────────────────────────────────────────────────────────


def _p(v: float, signed: bool = True) -> str:
    return "–" if v is None or not np.isfinite(v) else (f"{v:+.2%}" if signed else f"{v:.0%}")


def _t(v: float) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:+.2f}"


def _table(df: pd.DataFrame) -> list[str]:
    out = ["| 판정 | test | 내용 | 관측 | 사건 | 평균 초과 | 비용 후 | 적중률 | t | 연도+ | 앞/뒤 반 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        unit = "달" if r.kind == "xs" else "일"
        ev = f"{r.n_events:,.0f}" if pd.notna(r.n_events) else "–"
        net = _p(r.mean_net) if r.direction == "buy" else "– (피하기)"
        yp = f"{r.years_pos:.0f}/{r.n_years:.0f}" if pd.notna(r.years_pos) else "–"
        out.append(f"| {r['판정']} | {r.test_id} | {NAMES.get(r['name'], r['name'])} | "
                   f"{r.n_obs:.0f}{unit} | {ev} | {_p(r['mean'])} | {net} | {_p(r.hit, False)} | {_t(r.t_nw)} | {yp} | "
                   f"{_p(r.first_half)} / {_p(r.second_half)} |")
    return out


def write_report(res: pd.DataFrame, cf: pd.DataFrame, plc: pd.DataFrame, ref: pd.DataFrame, cov: pd.DataFrame,
                 z: float, n_fam: int, cands: list[str]) -> None:
    main = res[res["판정"] != "참고"]
    counts = " · ".join(f"{k} {v}" for k, v in main["판정"].value_counts().items())
    lines = [
        "# 거래량·수급 신호 연구 — 상승/하락 거래량 · 고거래량 · 고갈 뒤 돌파 · 바닥 매집 · 일일 공매도 (vol_flow_v1)", "",
        f"- 실행 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 정의·사전 등록은 `scripts/vol_flow_research.py` 머리말",
        f"- 개발 2016-01-04 ~ 2025-09-30 (B 는 FINRA 파일이 있는 2018-09/11 ~) · 확인 2006-01-03 ~ 2015-12-31 (A 만) · "
        f"홀드아웃 안 씀",
        f"- 그날 S&P 500 구성 종목 · t 종가 신호 → t+1 시가 진입 → h 거래일 뒤 시가 · 같은 구간 구성 종목 동일가중 대비 · "
        f"비용 왕복 0.2% (피하기는 비용 없음)",
        f"- 사전 등록 {n_fam}건 → **Bonferroni z = {z:.2f}** · 누적 시도 {total_trials()}건 · 판정: {counts}", "",
        "## 쉬운 말 요약", "", *(SUMMARY or ["(실행 뒤 채움)"]), "",
        "표 읽는 법: '평균 초과'·'비용 후'·'적중률' = 사건(종목) 하나하나의 평균. 't'·'연도+'·'앞/뒤 반' = 날짜별 평균을 낸 뒤의 값 "
        "(같은 날 사건이 몰려도 하루로 센다). 둘이 엇갈리면 사건이 특정 며칠에 몰린 것이다.", "",
        "## ① 개발 구간 (주 보유기간 = 판정, 나머지 = 참고)", ""]
    lines += _table(res)
    lines += ["", "## ② 확인 구간 2006~2015 (A 만 — 개발 '후보'만 판정, 나머지는 참고)", "",
              f"개발 후보: {', '.join(cands) if cands else '없음'}", ""]
    lines += _table(cf)
    lines += ["", "## ③ A4 바닥권: 매집(CMF20 ≥ +0.05) vs 분산(CMF20 ≤ −0.05) — 참고", ""]
    lines += _table(ref.assign(**{"판정": ref["name"].map({"A4-BDIV": "매집", "A4-BDIV-대조": "분산(대조)"}) + " · " + ref["period"]}))
    lines += ["", "## ④ 무작위 사건 점검 (같은 날 같은 수를 구성 종목에서 무작위로, 20회) — t 가 0 근처면 계산이 정상", "",
              "| test | 구간 | 평균 t | t 표준편차 | 최대 \\|t\\| |", "|---|---|---|---|---|"]
    lines += [f"| {r.test_id} | {r.period} | {r.t_mean:+.2f} | {r.t_sd:.2f} | {r.t_max_abs:.2f} |" for _, r in plc.iterrows()]
    lines += ["", "## 연도별 평균 초과수익 (개발 · 주 보유기간)", "", "| test | 연도별 |", "|---|---|"]
    lines += [f"| {r.test_id} | {r.yearly} |" for _, r in res[res["판정"] != "참고"].iterrows()]
    lines += ["", "## 커버리지 (연중 가운데 날 기준)", "",
              "| 연도 | 명단 | 가격·거래량 있음 | UDV50 있음 | SVR20 있음 | DSVR 있음 |", "|---|---|---|---|---|---|"]
    lines += [f"| {int(r['연도'])} | {int(r['명단'])} | {int(r['가격있음'])} | {_p(r['A1-UDV50'], False)} | "
              f"{_p(r['B1-SVR20'], False)} | {_p(r['B2-DSVR'], False)} |" for _, r in cov.iterrows()]
    lines += ["", "## 한계", "",
              "- 가격은 지금 야후에서 받을 수 있는 종목만 → 상장폐지된 회사가 빠진다. 2006~2015 는 구성 종목의 약 60~75% 만 있다 "
              "(생존 편향 — 특히 A4 바닥권은 망한 회사가 빠져 좋게 나올 수 있다).",
              "- FINRA 일일 공매도 거래량은 FINRA 에 보고된 장외 거래만이다 (거래소 거래 제외). 2018-08 이전 파일이 없어 확인 구간이 없다.",
              "- '후보'는 같은 날 구성 종목보다 낫다는 뜻일 뿐이다. SPY·PT-1 을 이기는지는 PT-1 결합 검정을 따로 해야 한다.",
              "- 지표 상수(50일·0.6·1.5배·0.05 등)는 결과를 보기 전에 정했고 바꾸지 않는다."]
    (TIER3_DIR / "VOL_FLOW_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch-finra", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--register", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true", help="저장된 표로 보고서만 다시 쓰기")
    args = ap.parse_args()
    if args.fetch_finra:
        fetch_finra()
    elif args.count:
        do_count()
    elif args.register:
        do_register()
    elif args.report:
        do_report()
    else:
        do_run()


if __name__ == "__main__":
    main()
