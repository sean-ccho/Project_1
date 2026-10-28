"""차트와 같은 선으로 교차 라벨 — TradingView SeanEMA 의 EMA10(주황) · EMA20(보라).

2026-10-26 사용자 결정 (docs/goldencross.md): 라벨 · 계산 · TradingView 차트가 같은 선을 보게 한다.
예전 라벨은 차트에 없는 선(일봉 EMA20/50 · 주봉 SMA10/40 · 월봉 SMA3/10, 배당 조정 가격, 진행 중인 봉)이라
같은 '골든크로스'가 차트와 반대인 경우가 많았다 (2026-10-16: 주·월봉 42개 중 24개).

- 선: 일·주·월봉 모두 EMA10 · EMA20 (Pine ta.ema — 첫 n봉 평균에서 시작, screener.ma_cross.pine_ema)
- 가격: 분할만 반영 (TradingView 기본, ADJ 꺼짐). 받은 가격이 배당 조정(auto_adjust)이면 Dividends 칸으로 되돌린다
- 봉: 끝난 봉 기준 (ma_cross.bar_complete: 주봉 = 그 주 금요일, 월봉 = 그 달 마지막 평일 장이 끝난 봉). 진행 중인 봉은 '예고'로 따로
- 교차: 마지막 끝난 봉에서 최근 recent_window 봉 안의 마지막 교차 (ma_cross.TF_RULES: 일 5 · 주 5 · 월 3)
- 골든 임박: ma_cross 의 GI+ — 갭 −5%~0 · 좁혀지는 중 · 단기선이 계속 아래 · (주·월) 단기선 오름 · (일) 종가 > EMA10 ·
  종가 > 단기선
- 데드 임박: 그 거울 — 갭 0~+5% · 좁혀지는 중 · 단기선이 계속 위 · (주·월) 단기선 내림 · 종가 < 단기선

5년치 가격으로도 주봉·월봉 EMA10/20 상태가 전체 기간 계산과 같다 (2026-10-26, S&P 554종목: 주봉 0개 · 월봉 1개 다름).

2026-10-27 사용자 결정: 교차 라벨뿐 아니라 모든 패턴(가격 패턴 · 캔들 · 정배열 · 차트반전)을 같은 바탕으로 본다.
- 차트 가격 묶음 ChartPrices: 분할만 반영한 일봉 OHLCV + 상장 이후 전체 주·월봉 (밤 실행은 period="max" 로 따로 받는다,
  백테스트는 같은 방식으로 받아 얼린 파일). 주·월봉 EMA100/200 은 긴 기록이 있어야 TradingView 값과 맞는다
  (월봉 EMA200 = 200개월 ≈ 17년, Pine ta.ema 는 첫 봉부터 이어지는 계산)
- chart_view: 그 시점에 보이는 일봉(최근 5년) · 주봉 · 월봉 (끝난 봉 + 진행 중인 봉 하나, complete 칸)
- 정배열 3단계: 차트 4선 EMA10/20/100/200 (uptrend_stage). 아직 없는 선은 빼고 있는 선으로만 본다
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener.ma_cross import IMMINENT_GAP, TF_RULES, Timeframe, bar_complete, cross_frame, last_weekday, pine_ema

SHORT, LONG = 10, 20
LINES = (10, 20, 100, 200)   # 사용자 차트(SeanEMA)에 보이는 선: EMA10 주황 · EMA20 보라 · EMA100 노랑 · EMA200 파랑
OHLCV = ("Open", "High", "Low", "Close", "Volume")
DAILY_YEARS = 5              # 일봉은 최근 5년 (밤 실행 fetch_ohlcv(period="5y") · 백테스트 FEATURE_HISTORY_YEARS 와 같음)
# 교차 라벨은 선을 전체 기록으로 계산한 뒤 마지막 60봉에서만 교차 · 임박을 본다 (ma_cross.cross_frame tail) — 판정이 보는 창은
# 최대 약 25봉(최근 교차 5봉 · 단기선이 계속 아래 10봉 · 간격 비교 9봉 전)이라 값이 같다. 상장 이후 전체 주봉(수천 봉)마다
# 규칙을 다 돌리면 백테스트가 크게 느려져서 (2026-10-27)
CROSS_TAIL = 60
LABELS = ("golden_cross", "death_cross", "golden_cross_imminent", "death_cross_imminent")


@dataclass(frozen=True)
class CrossLabel:
    label: str      # golden_cross · death_cross · golden_cross_imminent · death_cross_imminent · none
    gap: float      # (EMA10 − EMA20) / EMA20, 마지막 끝난 봉 (선이 없으면 nan)
    preview: str    # 진행 중인 봉으로 보면 달라지는 라벨 (예고) — 없으면 ""


NONE = CrossLabel("none", float("nan"), "")


def split_only_close(df: pd.DataFrame) -> pd.Series:
    """배당 조정 종가 + Dividends → 분할만 반영한 종가 (모양이 같다 — 상수배까지).

    배당 조정: A_t = C_t × F_t, F_t = Π(배당락일 j > t) (1 − D_j / C_(j−1)) (Yahoo · yfinance 방식).
    뒤에서부터 F 를 되짚는다: 배당락일 j 에서 F_(j−1) = F_j × A_(j−1) / (A_(j−1) + D_j × F_j).
    받은 날이 데이터 끝이어야 정확하다 (라이브). 나중에 받은 가격을 기준일까지 잘라 쓰면(백테스트) 기준일 뒤 배당이 섞여
    0.1~0.2% 어긋나므로, 백테스트는 전체 기간으로 미리 되돌린 CloseSplit 칸을 붙여 둔다 (_prepare_price_map) — 있으면 그 칸.
    Dividends 칸이 없거나 배당이 없으면 종가 그대로.
    """
    if "CloseSplit" in df.columns and df["CloseSplit"].notna().any():
        return df["CloseSplit"].astype("float64")
    close = df["Close"].astype("float64")
    if "Dividends" not in df.columns:
        return close
    div = df["Dividends"].astype("float64").fillna(0.0).to_numpy()
    if not (div > 0).any():
        return close
    a = close.to_numpy()
    f = np.ones(len(a))
    cur = 1.0
    for t in range(len(a) - 1, 0, -1):
        f[t] = cur
        d = div[t]
        prev = a[t - 1]
        if d > 0 and np.isfinite(prev) and prev > 0:
            cur = cur * prev / (prev + d * cur)
    if len(a):
        f[0] = cur
    return pd.Series(a / f, index=close.index, name=close.name)


def split_only_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """배당 조정 OHLC + Dividends → 분할만 반영한 OHLCV. 시가 · 고가 · 저가는 그날 종가와 같은 배당 계수로 나눈다.

    배당 조정은 하루 안의 네 값에 같은 계수를 곱하므로 (Yahoo), 종가로 구한 계수 = 그날 계수다. Volume 은 그대로.
    """
    close = split_only_close(df)
    factor = (df["Close"].astype("float64") / close).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out = pd.DataFrame(index=df.index)
    for col in ("Open", "High", "Low"):
        out[col] = df[col].astype("float64") / factor if col in df.columns else close
    out["Close"] = close
    out["Volume"] = df["Volume"].astype("float64") if "Volume" in df.columns else np.nan
    return out


def ohlcv_bars(daily: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    """일봉 OHLCV → 주·월봉 OHLCV (patterns.get_weekly_data / get_monthly_data 와 같은 값).

    날짜 = 그 봉의 마지막 거래일 (ma_cross.to_bars 와 같음), end = 그 봉 기간의 마지막 날 (주봉 금요일 · 월봉 말일).
    종가가 없는 날(거래 없음)은 뺀다. 일봉은 그대로 (end = 그날).
    """
    d = daily.loc[daily["Close"].notna(), list(OHLCV)] if len(daily) else daily.reindex(columns=list(OHLCV))
    if tf == "D" or d.empty:
        out = d.astype("float64").copy()
        out["end"] = out.index.normalize() if len(out) else pd.DatetimeIndex([])
        return out
    freq = "W-FRI" if tf == "W" else "M"
    key = d.index.to_period(freq)
    g = d.groupby(key)
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum(min_count=1)})
    out["end"] = out.index.end_time.normalize()
    out.index = pd.DatetimeIndex(d.index.to_series().groupby(key).max().to_numpy())
    return out


@dataclass(frozen=True)
class ChartPrices:
    """한 종목의 차트 가격 (분할만 반영 = TradingView 기본). daily = 일봉 OHLCV, weekly · monthly = 상장 이후 전체 봉."""
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame


def chart_prices(daily_split: pd.DataFrame, daily_since: pd.Timestamp | None = None) -> ChartPrices:
    """분할만 반영한 일봉(상장 이후 전체)으로 묶음을 만든다. daily_since 를 주면 일봉은 그날부터만 남긴다 (주·월봉은 전체)."""
    d = daily_split.loc[daily_split["Close"].notna(), list(OHLCV)].astype("float64")
    w, m = ohlcv_bars(d, "W"), ohlcv_bars(d, "M")
    if daily_since is not None:
        d = d.loc[pd.Timestamp(daily_since):]
    return ChartPrices(d, w, m)


def extend_chart(cp: ChartPrices, recent: pd.DataFrame) -> ChartPrices:
    """차트 가격에 아직 없는 최근 날을 받은 일봉(분할만 반영으로 되돌린 것)으로 채운다.

    상장 이후 전체(period="max")는 가장 최근 봉이 하루 늦게 들어오기도 한다 (2026-10-27: 5년치엔 10-27 봉이 있는데
    전체 기록은 10-26 까지). 채운 날이 들어가는 주·월봉만 다시 만든다. 두 가격의 분할 기준이 다르면(백테스트에서
    나중에 받은 파일) 겹치는 날 비율로 맞춘다. 채울 날이 없으면 그대로 돌려준다.
    """
    if cp.daily.empty or recent.empty or recent.index[-1] <= cp.daily.index[-1]:
        return cp
    last = cp.daily.index[-1]
    extra = recent.loc[recent.index > last, list(OHLCV)].astype("float64")
    both = cp.daily["Close"].reindex(recent.index).dropna().tail(20)
    ratio = float(np.median(both / recent["Close"].reindex(both.index))) if len(both) else 1.0
    if np.isfinite(ratio) and abs(ratio - 1.0) > 1e-3:
        extra[["Open", "High", "Low", "Close"]] *= ratio
        extra["Volume"] /= ratio
    daily = pd.concat([cp.daily, extra])
    first = extra.index[0]

    def rebuild(bars: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
        start = first.to_period("W-FRI" if tf == "W" else "M").start_time
        keep = bars.loc[bars.index < start]
        return pd.concat([keep, ohlcv_bars(daily.loc[daily.index >= start], tf)])

    return ChartPrices(daily, rebuild(cp.weekly, "W"), rebuild(cp.monthly, "M"))


def _view_bars(bars: pd.DataFrame, daily: pd.DataFrame, tf: Timeframe, as_of: pd.Timestamp) -> pd.DataFrame:
    """as_of 에 보이는 주·월봉: 끝난 봉(그 기간 마지막 평일 ≤ as_of, ma_cross.bar_complete 와 같은 규칙)은 미리 만든 봉에서,
    그 뒤(진행 중인 봉)는 as_of 까지 일봉으로 새로 만든다.

    미리 만든 봉의 마지막 봉은 as_of 뒤 날짜까지 합친 값일 수 있어서(백테스트) 그대로 쓰면 미래를 본다.
    """
    # 봉은 기간 순서라 '마지막 평일 ≤ as_of' 는 앞에서부터 이어진다 → 경계만 찾는다. 끝난 봉은 정의상 complete
    k = int(last_weekday(pd.DatetimeIndex(bars["end"])).searchsorted(as_of, side="right")) if len(bars) else 0
    done = bars.iloc[:k]
    after = daily.loc[:as_of]
    if len(done):
        after = after.loc[after.index > done["end"].iloc[-1]]
    rest = ohlcv_bars(after, tf)
    done = done.assign(complete=np.ones(len(done), dtype=bool))
    if not len(rest):
        return done
    rest = rest.assign(complete=bar_complete(pd.DatetimeIndex(rest.index), tf, as_of))
    return pd.concat([done, rest])


def chart_view(cp: ChartPrices, as_of: pd.Timestamp,
               daily_years: int = DAILY_YEARS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """as_of 시점의 차트: 일봉(최근 daily_years 년) · 주봉 · 월봉 (끝난 봉 + 진행 중인 봉, complete 칸)."""
    as_of = pd.Timestamp(as_of)
    daily = cp.daily.loc[as_of - pd.DateOffset(years=daily_years):as_of]
    return daily, _view_bars(cp.weekly, cp.daily, "W", as_of), _view_bars(cp.monthly, cp.daily, "M", as_of)


def pine_ema_valid(s: pd.Series, n: int) -> pd.Series:
    """Pine ta.ema 를 앞쪽 빈 값(na)이 있는 시리즈에 — 값이 생긴 봉부터 n봉 평균으로 시작한다 (예: MACD 신호선)."""
    v = s.dropna()
    return pine_ema(v, n).reindex(s.index) if len(v) else pd.Series(np.nan, index=s.index, name=s.name)


def pine_rma(s: pd.Series, n: int) -> pd.Series:
    """Pine ta.rma: 처음 n봉 평균에서 시작, alpha = 1/n (RSI · ATR 의 평균)."""
    v = s.dropna().astype("float64")
    if len(v) < n:
        return pd.Series(np.nan, index=s.index, name=s.name)
    x = v.copy()
    x.iloc[: n - 1] = np.nan
    x.iloc[n - 1] = float(v.iloc[:n].mean())
    return x.ewm(alpha=1.0 / n, adjust=False).mean().reindex(s.index)


def pine_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Pine ta.rsi: rma(오른 폭) / rma(내린 폭). 내린 폭 평균이 0 이면 100, 오른 폭 평균이 0 이면 0."""
    ch = close.astype("float64").diff()
    up, down = pine_rma(ch.clip(lower=0), n), pine_rma(-ch.clip(upper=0), n)
    rsi = 100 - 100 / (1 + up / down.replace(0, np.nan))
    return rsi.mask(down == 0, 100.0).mask((up == 0) & (down != 0), 0.0)


def finished(bars: pd.DataFrame) -> pd.DataFrame:
    """끝난 봉만 (complete 칸이 없으면 그대로)."""
    return bars.loc[bars["complete"].to_numpy(bool)] if "complete" in bars.columns else bars


def uptrend_stage(close: pd.Series) -> str:
    """정배열 3단계 — 차트 4선 EMA10 · 20 · 100 · 200 (Pine ta.ema, 끝난 봉 종가). "full" · "pullback" · "forming" · "".

    완전: 종가 > EMA10 > EMA20 > EMA100 > EMA200 · 눌림목: 네 선 정렬 + EMA10 ≥ 종가 > EMA20 ·
    형성중: EMA10 > EMA20 + 종가 > EMA10 (아래 선은 아직 안 맞음). 아직 없는 선(봉 수 < 기간)은 빼고 있는 선으로만 본다 —
    차트에도 안 그려진다. EMA100 도 없으면 '형성중'까지만 (2026-10-27 사용자 결정).
    """
    c = close.dropna().astype("float64")
    if len(c) < LONG:
        return ""
    ema = {n: float(pine_ema(c, n).iloc[-1]) if len(c) >= n else float("nan") for n in LINES}
    price, e10, e20 = float(c.iloc[-1]), ema[10], ema[20]
    lower = [ema[n] for n in (100, 200) if np.isfinite(ema[n])]
    chain = [e10, e20, *lower]
    stacked = bool(lower) and all(a > b for a, b in zip(chain, chain[1:]))
    if stacked and price > e10:
        return "full"
    if stacked and e10 >= price > e20:
        return "pullback"
    if not stacked and e10 > e20 and price > e10:
        return "forming"
    return ""


def _dead_imminent(fr: pd.DataFrame, tf: Timeframe) -> pd.Series:
    """ma_cross 의 골든 임박(GI+)을 거꾸로: 단기선이 장기선 위에서 내려와 5% 안으로 좁혀지는 중 + 종가 < 단기선."""
    rule = TF_RULES[tf]
    c, ms, ml, gap = fr["close"], fr["ma_s"], fr["ma_l"], fr["gap"]
    valid = ms.notna() & ml.notna()
    lag = rule.slope_lookback - 1
    cross = (fr["gc"] | fr["dc"]).astype(float)
    recent_cross = cross.rolling(rule.recent_window, min_periods=1).max() > 0
    above = ((ms > ml) | ~valid).astype(float).rolling(SHORT, min_periods=1).min() > 0
    any_valid = valid.astype(float).rolling(SHORT, min_periods=1).max() > 0
    narrowing = gap.abs() < gap.shift(lag).abs()
    ok = valid & ~recent_cross & (gap > 0) & (gap < IMMINENT_GAP) & above & any_valid & narrowing & (c < ms)
    if rule.short_slope_confirm:
        ok &= ms < ms.shift(lag)
    if rule.close_filter_ema > 0:   # 전체 기록으로 계산한 필터 EMA (tail 로 자른 프레임이면 close_ema 칸)
        ok &= c < (fr["close_ema"] if "close_ema" in fr.columns else pine_ema(c, rule.close_filter_ema))
    return ok.fillna(False).astype(bool)


def _label_last(fr: pd.DataFrame, dead_imm: pd.Series, tf: Timeframe) -> str:
    """프레임의 마지막 봉 기준 라벨."""
    if fr.empty or not (pd.notna(fr["ma_s"].iloc[-1]) and pd.notna(fr["ma_l"].iloc[-1])):
        return "none"
    win = fr.iloc[-TF_RULES[tf].recent_window:]
    crosses = win[win["gc"] | win["dc"]]
    if len(crosses):
        last = crosses.iloc[-1]
        if bool(last["gc"]) and bool(fr["golden"].iloc[-1]):
            return "golden_cross"
        if bool(last["dc"]) and not bool(fr["golden"].iloc[-1]):
            return "death_cross"
    if bool(fr["gi_plus"].iloc[-1]):
        return "golden_cross_imminent"
    if bool(dead_imm.iloc[-1]):
        return "death_cross_imminent"
    return "none"


def cross_label(close: pd.Series, tf: Timeframe, as_of: pd.Timestamp | None = None, bars: bool = False) -> CrossLabel:
    """분할만 반영한 일봉 종가 → 그 봉의 교차 라벨 (끝난 봉) + 진행 중인 봉 예고.

    bars: close 가 이미 주·월봉 종가(chart_view 의 봉)면 True — 일봉 → 봉 변환을 건너뛴다.
    """
    close = close.dropna()
    if close.size < 2:
        return NONE
    fr = cross_frame(close, SHORT, LONG, "ema", tf, as_of, bars=bars, tail=CROSS_TAIL)
    if fr.empty:
        return NONE
    dead_imm = _dead_imminent(fr, tf)
    done = fr["complete"].to_numpy(bool)
    if not done.any():
        return CrossLabel("none", float("nan"), "")
    n_done = int(np.flatnonzero(done)[-1]) + 1          # 끝난 봉은 앞에서부터 이어진다
    label = _label_last(fr.iloc[:n_done], dead_imm.iloc[:n_done], tf)
    gap = float(fr["gap"].iloc[n_done - 1]) if pd.notna(fr["gap"].iloc[n_done - 1]) else float("nan")
    preview = ""
    if n_done < len(fr):                                 # 진행 중인 봉이 있으면 그 봉으로도 본다
        live = _label_last(fr, dead_imm, tf)
        if live != label and live != "none":
            preview = live
    return CrossLabel(label, gap, preview)
