"""OHLCV 데이터 및 포지션 파일 무결성 검증 모듈.

프로덕션 파이프라인에서 데이터 품질 이슈를 조기에 발견하기 위한 검증 유틸리티.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Yahoo 가 아직 반영 안 한 액면분할 보정 (2026-11-18) ─────────────
#
# Yahoo 는 분할 기록(Stock Splits)은 바로 주면서 그 전 가격은 하루 이틀 분할 전 그대로 둘 때가 있다.
# 11-17 밤: EMBJ 4:1 (11-16) 이 82 → 18.9 로 남아 오늘 탭 '하락 상위' −76%, 52주 $19~$83 '저점 0%', 1년 −70%,
# KALA 1:30 역분할 +2,368%. 분할 기록이 있고 그 직후 첫 종가 ÷ 직전 종가가 분할 비율로 설명될 때만 그 전 봉을 나눈다 —
# Yahoo 가 이미 반영했으면 비율이 1 근처라 건드리지 않고, 분할 기록 없는 실제 급등락(SLS +156%)도 그대로.

SPLIT_COL = "Stock Splits"
_PRICE_COLS = ("Open", "High", "Low", "Close", "Adj Close")
# 이보다 작은 분할은 고치지 않는다 — Yahoo 는 1~10% 주식배당(SCCO 1.01:1 · LEN 1.033:1 …)도 분할로 적어서, 그날 우연히
# 비슷하게 움직이면 과거 가격을 몇 % 틀리게 고친다 (11-18 펄스 기준값 2년에서 47건 중 22건이 이런 것). 5:4 · 6:5 부터.
MIN_SPLIT_LOG = math.log(1.2)


@dataclass(frozen=True)
class SplitFix:
    """보정한 분할 하나 — ratio 는 Yahoo 값 (4.0 = 4:1 분할, 0.0333 = 1:30 병합)."""

    ticker: str
    first_bar: pd.Timestamp   # 분할 뒤 첫 종가 날짜 — 이 날 전 봉을 나눴다
    ratio: float
    raw_move: float           # 보정 전 그날 등락 (예: −0.76)
    fixed_move: float         # 보정 뒤 그날 등락 (예: −0.04)


def split_label(ratio: float) -> str:
    """4.0 → '4:1', 0.0333 → '1:30', 1.5 → '3:2'."""
    if ratio >= 1:
        for den in (1, 2, 3, 4, 5, 10):
            num = ratio * den
            if abs(num - round(num)) < 1e-6:
                return f"{int(round(num))}:{den}"
        return f"{ratio:g}:1"
    inv = 1 / ratio
    for num in (1, 2, 3, 4, 5, 10):
        den = inv * num
        if abs(den - round(den)) < 1e-3:
            return f"{num}:{int(round(den))}"
    return f"1:{inv:.3g}"


def split_explains_jump(prev_close: float, next_close: float, ratio: float) -> bool:
    """분할 직전 종가 → 직후 첫 종가의 점프가 분할 비율로 설명되는가 (= Yahoo 가 아직 반영 안 함).

    보정하면 그날 등락(로그)이 분할 크기의 절반 넘게 줄어야 한다. 이미 반영된 가격이면 보정이 오히려 점프를 만든다.
    """
    if not (prev_close > 0 and next_close > 0 and ratio > 0) or abs(math.log(ratio)) < MIN_SPLIT_LOG - 1e-9:
        return False
    raw = abs(math.log(next_close / prev_close))
    fixed = abs(math.log(next_close * ratio / prev_close))
    return fixed < raw and raw - fixed > 0.5 * abs(math.log(ratio))


def repair_unapplied_splits_frame(df: pd.DataFrame, ticker: str = "") -> tuple[pd.DataFrame, list[SplitFix]]:
    """한 종목 일봉(Open…Close · Volume · Stock Splits) — 반영 안 된 분할이 있으면 그 전 가격 ÷비율 · 거래량 ×비율.

    Stock Splits 칸이 없거나 분할이 없으면 받은 표를 그대로 돌려준다 (복사하지 않음).
    """
    if SPLIT_COL not in df.columns or "Close" not in df.columns:
        return df, []
    sp = pd.to_numeric(df[SPLIT_COL], errors="coerce")
    events = sp[(sp > 0) & ((sp - 1).abs() > 1e-9)]
    if events.empty:
        return df, []
    out: pd.DataFrame | None = None
    fixes: list[SplitFix] = []
    for when, ratio in events.items():
        cur = (out if out is not None else df)["Close"]
        close = pd.to_numeric(cur, errors="coerce").dropna()
        close = close[close > 0]
        post = close[close.index >= when]
        if post.empty:
            continue
        first = post.index[0]
        pre = close[close.index < first]
        if pre.empty:
            continue
        prev_c, next_c = float(pre.iloc[-1]), float(post.iloc[0])
        if not split_explains_jump(prev_c, next_c, float(ratio)):
            continue
        if out is None:
            out = df.copy()
            if "Volume" in out.columns:
                out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").astype(float)
        mask = out.index < first
        for col in _PRICE_COLS:
            if col in out.columns:
                out.loc[mask, col] = pd.to_numeric(out.loc[mask, col], errors="coerce") / float(ratio)
        if "Volume" in out.columns:
            out.loc[mask, "Volume"] = out.loc[mask, "Volume"] * float(ratio)
        fixes.append(SplitFix(ticker, pd.Timestamp(first), float(ratio), next_c / prev_c - 1,
                              next_c * float(ratio) / prev_c - 1))
    return (out if out is not None else df), fixes


def repair_unapplied_splits(data: pd.DataFrame) -> tuple[pd.DataFrame, list[SplitFix]]:
    """(티커, 필드) 멀티인덱스 일봉 전체 — 분할 기록이 있는 종목만 repair_unapplied_splits_frame.

    fetch_ohlcv · fetch_chart_prices 가 yf.download(actions=True) 로 받은 표에 쓴다. 바꾼 종목이 없으면 그대로 돌려준다.
    """
    if data is None or data.empty or not isinstance(data.columns, pd.MultiIndex):
        return data, []
    if SPLIT_COL not in data.columns.get_level_values(1):
        return data, []
    sp = data.xs(SPLIT_COL, axis=1, level=1).apply(pd.to_numeric, errors="coerce")
    cand = [t for t in sp.columns[((sp > 0) & ((sp - 1).abs() > 1e-9)).any()]]
    if not cand:
        return data, []
    fixes: list[SplitFix] = []
    changed: dict[str, pd.DataFrame] = {}
    for t in cand:
        new, f = repair_unapplied_splits_frame(data[t], str(t))
        if f:
            changed[t] = new
            fixes += f
    if not changed:
        return data, []
    out = data.copy()
    for t, new in changed.items():
        for col in new.columns:
            out[(t, col)] = new[col]   # 칸을 통째로 바꾼다 (거래량은 정수 → 실수)
    return out, fixes


def summarize_split_fixes(fixes: list[SplitFix], limit: int = 40) -> list[str]:
    """보정 로그 한 줄 — '[분할 보정] 3종목 — Yahoo 가 아직 반영 안 한 분할: EMBJ 4:1 (11-17 −76% → −4%), …'."""
    if not fixes:
        return []
    parts = [f"{f.ticker} {split_label(f.ratio)} ({f.first_bar:%m-%d} {f.raw_move:+.0%} → {f.fixed_move:+.0%})"
             for f in fixes[:limit]]
    more = f" 외 {len(fixes) - limit}건" if len(fixes) > limit else ""
    return [f"[분할 보정] {len(fixes)}건 — Yahoo 가 아직 반영 안 한 분할을 그 전 가격에 반영: {', '.join(parts)}{more}"]


# ── OHLCV 데이터 검증 ────────────────────────────────────────


class DataValidationWarning:
    """검증 경고를 담는 경량 컨테이너.

    group·detail 은 요약 출력용: 같은 group 끼리 한 줄로 묶고, 종목마다 detail(짧은 값)만 붙인다.
    """

    def __init__(self, level: str, ticker: str, message: str, group: str = "", detail: str = ""):
        self.level = level  # "warning" | "error"
        self.ticker = ticker
        self.message = message
        self.group = group or message
        self.detail = detail

    def __repr__(self) -> str:
        return f"[{self.level.upper()}] {self.ticker}: {self.message}"


RECENT_BARS = 252  # 급등락·거래량 0 은 최근 252 거래일(≈1년)만 본다 — 지금 신호에 쓰이는 구간


def validate_ohlcv(
    data: pd.DataFrame,
    *,
    max_nan_ratio: float = 0.30,
    max_price_change: float = 0.50,
    max_zero_volume_streak: int = 5,
) -> list[DataValidationWarning]:
    """OHLCV 데이터의 품질을 검증하고 경고 목록을 반환 (정보용 — 데이터는 고치거나 빼지 않는다).

    2026-10-11 실행에서 NASDAQ/NYSE 스캔 경고 707건 중 208건이 NaN 비율(대부분 최근 상장 종목의 상장 전 빈 값),
    310건이 1년도 더 지난 급등락이었다. 그래서 NaN 비율은 상장 이후만, 급등락은 최근 1년만 본다.

    Args:
        data: (ticker, field) 멀티인덱스 DataFrame
        max_nan_ratio: 상장 이후 Close NaN 비율 경고 기준 (기본 30%)
        max_price_change: 최근 1년 전일 대비 가격 변동률 경고 기준 (기본 50%)
        max_zero_volume_streak: 최근 1년 연속 거래량 0 경고 기준 (기본 5일)

    Returns:
        DataValidationWarning 리스트
    """
    warnings_list: list[DataValidationWarning] = []

    if data.empty:
        warnings_list.append(DataValidationWarning("error", "ALL", "OHLCV 데이터가 비어있음"))
        return warnings_list

    # 멀티인덱스가 아닌 경우 스킵
    if not isinstance(data.columns, pd.MultiIndex):
        return warnings_list

    # 티커 목록 추출
    tickers = data.columns.get_level_values(0).unique().tolist()

    for ticker in tickers:
        if ticker not in data.columns.get_level_values(0):
            continue

        ticker_data = data[ticker]

        # 1. NaN 비율 체크 — 첫 종가 이후만 (여러 종목을 한 번에 받으면 최근 상장 종목은 상장 전 날짜가 NaN 으로 채워진다)
        if "Close" in ticker_data.columns:
            close = ticker_data["Close"]
            first = close.first_valid_index()
            listed = close.loc[first:] if first is not None else close
            nan_ratio = float(listed.isna().mean()) if len(listed) > 0 else 0.0
            if nan_ratio > max_nan_ratio:
                warnings_list.append(DataValidationWarning(
                    "warning", ticker,
                    f"Close NaN 비율 {nan_ratio:.1%} (상장 이후, 기준: {max_nan_ratio:.0%})",
                    group=f"상장 이후 종가 빈 값 {max_nan_ratio:.0%} 넘음",
                    detail=f"{nan_ratio:.0%}",
                ))

        # 2. 가격 이상치 감지 (최근 1년) — 대부분 실제 급등락이고, 스플릿 미반영·잘못된 틱일 수도 있다
        if "Close" in ticker_data.columns:
            close = ticker_data["Close"].dropna().tail(RECENT_BARS + 1)
            if len(close) > 1:
                pct_changes = close.pct_change().dropna()
                extreme = pct_changes[pct_changes.abs() > max_price_change]
                if len(extreme) > 0:
                    worst_date = extreme.abs().idxmax()
                    worst_val = float(extreme.loc[worst_date])
                    warnings_list.append(DataValidationWarning(
                        "warning", ticker,
                        f"비정상 가격 변동 {worst_val:+.1%} ({worst_date.date()}) — 스플릿/데이터 오류 가능",
                        group=f"최근 1년 하루 ±{max_price_change:.0%} 넘는 변동 (대부분 실제 급등락, 스플릿/데이터 오류 가능)",
                        detail=f"{worst_val:+.0%}({worst_date.date()})",
                    ))

        # 3. 거래량 0 연속일 체크 (최근 1년만 검사 — 합병 전/상장 전 데이터 오탐 방지)
        if "Volume" in ticker_data.columns:
            vol = ticker_data["Volume"].dropna()
            vol = vol.tail(RECENT_BARS)
            if len(vol) > 0:
                zero_streaks = (vol == 0).astype(int)
                # 연속 0 카운트
                streak = 0
                max_streak = 0
                for v in zero_streaks:
                    if v == 1:
                        streak += 1
                        max_streak = max(max_streak, streak)
                    else:
                        streak = 0
                if max_streak >= max_zero_volume_streak:
                    warnings_list.append(DataValidationWarning(
                        "warning", ticker,
                        f"거래량 0 연속 {max_streak}일 (기준: {max_zero_volume_streak}일) — 상장폐지/거래정지 가능",
                        group=f"최근 1년 거래량 0 이 {max_zero_volume_streak}일 이상 연속 (상장폐지/거래정지 가능)",
                        detail=f"{max_streak}일",
                    ))

    # 출력은 부르는 쪽(fetch_ohlcv)에서 summarize_warnings 로 한 번만 — 여기서도 찍으면 같은 경고가 두 번 나온다
    for w in warnings_list:
        logger.debug(str(w))

    return warnings_list


def summarize_warnings(warnings_list: list[DataValidationWarning]) -> list[str]:
    """경고를 종류별로 한 줄씩 묶은 출력 줄 — 수백 건이어도 몇 줄로 보이게."""
    if not warnings_list:
        return []
    groups: dict[str, list[DataValidationWarning]] = {}
    for w in warnings_list:
        groups.setdefault(w.group, []).append(w)
    lines = [f"[DataValidator] 데이터 품질 경고 {len(warnings_list)}건 — 정보용, 데이터는 그대로 씀"]
    for group, items in groups.items():
        listed = ", ".join(f"{w.ticker} {w.detail}".rstrip() for w in items)
        lines.append(f"  {group} — {len(items)}종목: {listed}")
    return lines


# ── 포지션 파일 검증 ──────────────────────────────────────────

_REQUIRED_POSITION_FIELDS = [
    "ticker", "entry_price", "entry_date", "strategy",
    "star_rating", "ccs_score", "highest_price",
]


def validate_positions(positions: list[dict[str, Any]]) -> list[DataValidationWarning]:
    """positions.json 스키마 검증.

    Returns:
        경고 리스트 (비어있으면 정상)
    """
    warnings_list: list[DataValidationWarning] = []

    if not isinstance(positions, list):
        warnings_list.append(DataValidationWarning(
            "error", "ALL", "positions가 리스트가 아님"
        ))
        return warnings_list

    for i, pos in enumerate(positions):
        if not isinstance(pos, dict):
            warnings_list.append(DataValidationWarning(
                "error", f"index_{i}", "포지션이 dict가 아님"
            ))
            continue

        ticker = pos.get("ticker", f"index_{i}")

        # 필수 필드 존재 확인
        for field in _REQUIRED_POSITION_FIELDS:
            if field not in pos:
                warnings_list.append(DataValidationWarning(
                    "error", ticker, f"필수 필드 누락: {field}"
                ))

        # 가격 유효성
        entry_price = pos.get("entry_price", 0)
        if not isinstance(entry_price, (int, float)) or entry_price <= 0:
            warnings_list.append(DataValidationWarning(
                "error", ticker, f"유효하지 않은 entry_price: {entry_price}"
            ))

        highest = pos.get("highest_price", 0)
        if isinstance(highest, (int, float)) and isinstance(entry_price, (int, float)):
            if highest > 0 and highest < entry_price * 0.5:
                warnings_list.append(DataValidationWarning(
                    "warning", ticker,
                    f"highest_price({highest:.2f})가 entry_price({entry_price:.2f})의 50% 미만"
                ))

    return warnings_list


def validate_positions_file(path: Path | str) -> list[DataValidationWarning]:
    """positions.json 파일을 로드하고 검증.

    Returns:
        경고 리스트
    """
    path = Path(path)
    if not path.exists():
        return []  # 파일 없음 = 빈 포지션 (정상)

    try:
        positions = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [DataValidationWarning("error", "ALL", f"positions.json 파싱 실패: {e}")]

    return validate_positions(positions)
