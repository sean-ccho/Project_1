"""실적 발표 때 종목이 얼마나 움직였나 (표시·알림 전용) — 2026-10-29.

계기: 보유 종목이 실적 발표 다음 날 크게 떨어졌는데, 밤 메일 '실적 7일 이내' 알림에는 숫자(이 종목이 실적 때 보통 얼마나
움직이는지)도, 해야 할 결정도 없었다. 연구: docs/quant_improvement/tier3/EARNINGS_GUARD_REPORT.md
(사용자 스타일 계좌에서 실적 2주 안에 안 사기·실적 구간 비우기 → 수익은 그대로, 한 종목 최악 거래는 절반. Sharpe 판정은 미달).

- 실적 발표 시각 = SEC 8-K Item 2.02 접수 시각 (submissions JSON — 공시 경고와 같은 요청, 정정 8-K/A 제외).
  16:00 ET 이후 접수 → 다음 거래일 반응, 그 전 → 그날 (휴장이면 다음 거래일). 5일 안 중복은 첫 것, 45일 안에 몰리면
  마지막 것만 (실적 경고·잠정치 빼기 — scripts/build_sec_data.py · paper_trading.earnings_calendar.scheduled_only 와 같은 규칙).
  SEC 에 없는 종목(캐나다 등)은 yfinance 실적일로 대신한다 (내계좌처럼 종목이 적을 때만).
- 반응 = 반응일 종가 ÷ 전 거래일 종가 − 1 (밤사이 갭 + 그날 움직임).
- 요약 = 최근 8번 평균 |움직임| · 최악 · 최고 · 보통 시간(장 전 / 장 마감 후).
- 나쁜 경우 = max(최근 8번 최악 하락, 평균 × 3), 최대 50%. S&P 500 2015~2026 실적 25,723건에서 다음 실적 하락이 이보다
  컸던 비율 2.6% (약 40번에 1번). 매수 창·알림의 '나쁜 경우' 금액과 계좌 1~2% 기준 수량에 쓴다.
매매 판단(CCS)에는 쓰지 않는다.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

ET = "America/New_York"
LAST_N = 8                 # 요약에 쓰는 최근 실적 수
MIN_EVENTS = 2             # 이보다 적으면 요약 없음
DEDUPE_DAYS = 5            # 같은 발표의 중복 8-K
CLUSTER_DAYS = 45          # 예정 발표 전에 낸 실적 경고·잠정치를 뺀다
KEEP_ACCEPTS = 14          # 스냅샷에 남기는 최근 접수 시각 수 (8번 + 묶음·중복 여유)
BAD_MULT = 3.0             # 나쁜 경우 = max(최악, 평균 × 3)
BAD_CAP = 0.5
WINDOW_DAYS = 14           # 실적까지 이 달력일 안이면 '실적 임박' (연구 E1 = 10거래일 ≈ 14일)
TIMING_KO = {"amc": "장 마감 후", "bmo": "장 전"}
TIMING_FROM_KO = {v: k for k, v in TIMING_KO.items()}
DECISIONS = ("팔기", "절반", "그대로")
_WEEKDAY_KO = "월화수목금토일"


# ---------------------------------------------------------------------------
# 미국 증시 휴장일 (NYSE 정규 규칙 — 국가 애도일 같은 임시 휴장은 모른다)
# ---------------------------------------------------------------------------

def _easter(y: int) -> date:
    """부활절 (그레고리력, 익명 알고리즘)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = (h + el - 7 * m + 114) % 31 + 1
    return date(y, month, day)


def _nth_weekday(y: int, month: int, weekday: int, n: int) -> date:
    d = date(y, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(y: int, month: int, weekday: int) -> date:
    d = date(y + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """토요일 휴일 → 금요일, 일요일 → 월요일."""
    return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d


@lru_cache(maxsize=None)
def nyse_holidays(year: int) -> frozenset[date]:
    days = {
        _nth_weekday(year, 1, 0, 3),           # 마틴 루서 킹 데이
        _nth_weekday(year, 2, 0, 3),           # 대통령의 날
        _easter(year) - timedelta(days=2),     # 성금요일
        _last_weekday(year, 5, 0),             # 메모리얼 데이
        _observed(date(year, 7, 4)),           # 독립기념일
        _nth_weekday(year, 9, 0, 1),           # 노동절
        _nth_weekday(year, 11, 3, 4),          # 추수감사절
        _observed(date(year, 12, 25)),         # 크리스마스
    }
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:                # 1월 1일이 토요일이면 쉬는 날이 따로 없다 (전해 12-31 은 연다)
        days.add(_observed(new_year))
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # 준틴스
    return frozenset(days)


def is_session(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def add_sessions(d: date, k: int) -> date:
    """d 에서 거래일 k 개 뒤(k<0 이면 앞). d 가 휴장일이어도 된다."""
    step = 1 if k >= 0 else -1
    left = abs(k)
    while left:
        d += timedelta(days=step)
        if is_session(d):
            left -= 1
    return d


def sessions_until(today: date, day: date) -> int:
    """today 다음 날부터 day 까지(포함) 거래일 수. day == today 면 0, 지났으면 음수."""
    if day == today:
        return 0
    lo, hi, sign = (today, day, 1) if day > today else (day, today, -1)
    return sign * sum(1 for i in range(1, (hi - lo).days + 1) if is_session(lo + timedelta(days=i)))


def reaction_date(earn: date, timing: Optional[str]) -> date:
    """실적이 가격에 처음 반영되는 거래일 — 장 마감 후면 다음 거래일, 장 전·모르면 그날 (휴장이면 다음 거래일)."""
    if timing == "amc":
        return add_sessions(earn, 1)
    return earn if is_session(earn) else add_sessions(earn, 1)


def decision_deadline(earn: date, timing: Optional[str]) -> date:
    """팔지 말지 정할 마지막 거래일 = 반응일 바로 전 거래일 (시간을 모르면 장 전으로 보고 더 이르게)."""
    return add_sessions(reaction_date(earn, timing), -1)


# ---------------------------------------------------------------------------
# 실적 발표 시각 → 반응
# ---------------------------------------------------------------------------

def accepts_from_filings(blocks: Iterable[Mapping[str, Any]]) -> list[str]:
    """submissions JSON 의 filings 블록(recent · 옛 파일) → 실적 8-K(2.02) 접수 시각 (UTC 글자, 오래된 순)."""
    out: set[str] = set()
    for b in blocks:
        for form, items, ts in zip(b.get("form", []), b.get("items", []), b.get("acceptanceDateTime", [])):
            if form == "8-K" and "2.02" in {x.strip() for x in str(items or "").split(",")} and ts:
                out.add(str(ts))
    return sorted(out)


def scheduled_events(accepted: Iterable[Any]) -> list[pd.Timestamp]:
    """접수 시각 (UTC 글자 또는 시간대 있는 시각) → ET 시각. 5일 안 중복은 첫 것, 45일 안 묶음은 마지막 것만."""
    raw = pd.to_datetime(pd.Series(list(accepted), dtype=object), utc=True, errors="coerce").dropna()
    ts = raw.dt.tz_convert(ET).sort_values().tolist()
    kept: list[pd.Timestamp] = []
    for t in ts:
        if kept and (t - kept[-1]) <= pd.Timedelta(days=DEDUPE_DAYS):
            continue
        kept.append(t)
    return [t for i, t in enumerate(kept)
            if i + 1 == len(kept) or (kept[i + 1].normalize() - t.normalize()).days > CLUSTER_DAYS]


def _timing(t: pd.Timestamp) -> str:
    if t.hour >= 16:
        return "amc"
    return "bmo" if (t.hour, t.minute) < (9, 30) else "day"


def reaction_moves(accepted: Iterable[Any], close: pd.Series) -> list[dict[str, Any]]:
    """실적 한 번마다 {date: 반응일, move: 반응일 종가 ÷ 전날 종가 − 1, timing: amc·bmo·day} (오래된 순)."""
    c = pd.to_numeric(close, errors="coerce").dropna()
    if c.empty:
        return []
    idx = pd.DatetimeIndex(c.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    keep = ~idx.duplicated(keep="last")
    c, idx = c[keep], idx[keep]
    vals = c.to_numpy(float)
    out = []
    for t in scheduled_events(accepted):
        d0 = pd.Timestamp(t.date())
        timing = _timing(t)
        pos = int(idx.searchsorted(d0, side="right" if timing == "amc" else "left"))
        if pos <= 0 or pos >= len(idx) or (idx[pos] - d0).days > 5:
            continue
        prev, cur = vals[pos - 1], vals[pos]
        if not (np.isfinite(prev) and np.isfinite(cur) and prev > 0):
            continue
        out.append({"date": idx[pos].date().isoformat(), "move": float(cur / prev - 1), "timing": timing})
    return out


def move_stats(moves: list[dict[str, Any]], n: int = LAST_N) -> Optional[dict[str, Any]]:
    """최근 n번 요약 {avg, worst, best, n, timing, last: [[반응일, 움직임], …]}. 실적이 모자라면 None."""
    last = moves[-n:]
    if len(last) < MIN_EVENTS:
        return None
    m = [x["move"] for x in last]
    tim = [x["timing"] for x in last]
    timing = next((k for k in ("amc", "bmo") if tim.count(k) * 2 > len(tim)), None)
    return {"avg": round(float(np.mean(np.abs(m))), 4), "worst": round(min(m), 4), "best": round(max(m), 4),
            "n": len(m), "timing": timing, "last": [[x["date"], round(x["move"], 4)] for x in last]}


def bad_case(avg: Optional[float], worst: Optional[float]) -> Optional[float]:
    """나쁜 경우 하락률 (양수) = max(최근 최악 하락, 평균 × 3), 최대 50%. 평균을 모르면 None."""
    if avg is None or not np.isfinite(avg) or avg <= 0:
        return None
    w = -worst if worst is not None and np.isfinite(worst) and worst < 0 else 0.0
    return float(min(BAD_CAP, max(BAD_MULT * avg, w)))


def earnings_block(accepts: Mapping[str, list[str]], prices: Mapping[str, Any],
                   symbol: Callable[[str], str] = lambda t: t) -> dict[str, dict[str, Any]]:
    """티커 → move_stats (웹 TickerInfo.earnings_move). prices = {심볼: OHLCV 또는 종가}. 계산 못 하면 빠진다."""
    out: dict[str, dict[str, Any]] = {}
    for t, acc in accepts.items():
        f = prices.get(symbol(t))
        if f is None or not acc:
            continue
        close = f["Close"] if isinstance(f, pd.DataFrame) else f
        try:
            st = move_stats(reaction_moves(acc, close))
        except Exception:  # noqa: BLE001 — 표시용이라 한 종목 실패는 건너뛴다
            st = None
        if st:
            out[t] = st
    return out


def yf_accepts(ticker: str, limit: int = 16) -> list[str]:
    """SEC 에 없는 종목용 — yfinance 지난 실적 시각 (UTC 글자). 실패하면 []."""
    try:
        import yfinance as yf

        ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception:  # noqa: BLE001
        return []
    if ed is None or ed.empty:
        return []
    now = pd.Timestamp.now(tz="UTC")
    when = pd.to_datetime(pd.Series(ed.index), utc=True, errors="coerce").dropna()
    return [t.strftime("%Y-%m-%dT%H:%M:%S.000Z") for t in when if t < now]


# ---------------------------------------------------------------------------
# 시트 · 알림 글자
# ---------------------------------------------------------------------------

def sheet_values(st: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """내계좌 리스크 칸 — '실적 평균' 8.8% · '실적 최악' -17.7% · '실적 시간' 장 마감 후 (모르면 빈칸)."""
    if not st:
        return {"실적 평균": "", "실적 최악": "", "실적 시간": ""}
    return {"실적 평균": f"{st['avg'] * 100:.1f}%", "실적 최악": f"{st['worst'] * 100:+.1f}%",
            "실적 시간": TIMING_KO.get(st.get("timing") or "", "")}


def _pct_cell(v: Any) -> Optional[float]:
    s = str(v or "").replace("%", "").replace("+", "").replace("−", "-").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    return x / 100 if np.isfinite(x) else None


def stats_from_sheet(rec: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """시트 한 줄의 '실적 평균'·'실적 최악'·'실적 시간' → {avg, worst, timing} (평균이 없으면 None)."""
    avg = _pct_cell(rec.get("실적 평균"))
    if avg is None:
        return None
    return {"avg": avg, "worst": _pct_cell(rec.get("실적 최악")),
            "timing": TIMING_FROM_KO.get(str(rec.get("실적 시간", "")).strip())}


def decision_of(cell: Any, earn: Optional[str]) -> Optional[str]:
    """'실적대응' 칸 '절반 · 2026-11-04' → 그 실적일에 대한 결정이면 '절반'. 날짜가 다르면(지난 실적) None."""
    s = str(cell or "").strip()
    if not s or not earn:
        return None
    word, _, day = s.partition("·")
    word, day = word.strip(), day.strip()
    return word if word in DECISIONS and day == earn[:10] else None


def _md(d: date) -> str:
    return f"{d.month}/{d.day}({_WEEKDAY_KO[d.weekday()]})"


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def deadline_words(today: date, deadline: date) -> str:
    left = sessions_until(today, deadline)
    return "오늘" if left == 0 else "다음 거래일" if left == 1 else f"거래일 {left}일 남음" if left > 1 else "지남"


def alert_text(ticker: str, earn: str, st: Optional[Mapping[str, Any]], *, shares: Optional[float],
               price: Optional[float], account: Optional[float], decision: Optional[str], today: date) -> str:
    """보유 종목 실적 알림 한 줄 — 언제 · 결정 마감 · 이 종목 실적 때 움직임 · 내 수량이면 얼마 · 대응."""
    d = date.fromisoformat(earn[:10])
    timing = st.get("timing") if st else None
    dl = decision_deadline(d, timing)
    when = f"{_md(d)} {TIMING_KO[timing]}" if timing in TIMING_KO else f"{_md(d)} (시간 모름 — 장 전으로 봄)"
    parts = [f"{ticker} 실적 {when}", f"결정 마감 {_md(dl)} 장 마감 전 ({deadline_words(today, dl)})"]
    if st and st.get("avg") is not None:
        worst = st.get("worst")
        n = f" {st['n']}번" if st.get("n") else ""
        parts.append(f"최근{n} 실적 평균 ±{st['avg']:.1%}" + (f" · 최악 {worst:+.1%}" if worst is not None else ""))
        bad = bad_case(st["avg"], worst)
        if shares and price and bad:
            value = shares * price
            tail = f" (계좌의 {value * bad / account:.1%})" if account else ""
            parts.append(f"{shares:g}주 {_usd(value)} → 보통 ±{_usd(value * st['avg'])} · 나쁜 경우 −{_usd(value * bad)}{tail}")
    else:
        parts.append("지난 실적 움직임 모름")
    parts.append(f"대응: {decision} (적어 둠)" if decision else "대응: 아직 안 정함 → 팔기·절반·그대로 중 정해서 웹 내계좌에 적기")
    return " — ".join(parts[:2]) + " · " + " · ".join(parts[2:])


def alert_stage(today: date, earn: str, timing: Optional[str]) -> Optional[str]:
    """푸시할 날이면 'd2'(결정 마감 2거래일 전 = 반응 3거래일 전) · 'd0'(결정 마감 당일), 아니면 None."""
    try:
        dl = decision_deadline(date.fromisoformat(earn[:10]), timing)
    except ValueError:
        return None
    left = sessions_until(today, dl)
    return "d0" if left == 0 else "d2" if left == 2 else None


def days_to(earn: Optional[str], today: date) -> Optional[int]:
    """실적까지 달력일 (모르면 None)."""
    try:
        return (date.fromisoformat(str(earn)[:10]) - today).days
    except (TypeError, ValueError):
        return None


def now_et() -> datetime:
    return pd.Timestamp.now(tz=ET).to_pydatetime()
