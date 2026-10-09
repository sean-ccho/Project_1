"""페이퍼 트레이딩 로직 그대로 과거 데이터로 시뮬레이션하는 백테스터.

실제 paper trading과 동일한 로직:
- select_best_candidate() — CCS 7단계 필터 + 5 서브스코어
- check_sell_conditions() — 목표가/손절/트레일링/장기보유
- should_replace() — 교체 판단 (CCS 마진 0.05)
- 최대 포지션 3개
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hashlib

from data.fetch import fetch_ohlcv
from screener.backtest import _compute_ranked_snapshot, _prepare_price_map
from screener.cache import cache_meta_valid, count_cached_snapshots, write_cache_meta
from screener import config as _cfg
from screener.config import (
    BACKTEST_COST_PER_SIDE,
    BACKTEST_FUNDAMENTALS_PIT_SAFE,
    SECTOR_ETFS,
    TICKERS,
)
from paper_trading.candidate_selector import select_top_candidates
from paper_trading.earnings_calendar import EarningsCalendar
from paper_trading.structure import bear_liquidate, slot_allocation
from paper_trading.engine import (
    _activate_tight_trail,
    _get_row_for_ticker,
    _should_defer_sell,
    check_sell_conditions,
    current_ccs,
    intraday_stop_exit,
    should_replace,
)
from paper_trading.portfolio import get_worst_position

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git_info() -> dict[str, Any]:
    """현재 HEAD의 커밋 SHA / 브랜치 / dirty 여부를 반환. 실패 시 None 값."""
    def _run(args: list[str]) -> str:
        return subprocess.check_output(
            args, cwd=_PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()

    try:
        return {
            "sha": _run(["git", "rev-parse", "HEAD"])[:12],
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_run(["git", "status", "--porcelain"])),
        }
    except Exception:
        return {"sha": None, "branch": None, "dirty": None}


_HASHED_SOURCES = (
    "screener/features.py", "screener/alpha_model.py", "screener/patterns.py",
    "screener/fundamentals.py", "screener/processing.py", "screener/signals.py",
    "screener/backtest.py", "screener/config.py",
)
_SECRET_MARKERS = ("PASSWORD", "SECRET", "TOKEN", "CREDENTIAL", "EMAIL", "SPREADSHEET", "DRIVE", "GITHUB", "KEY")
_SNAPSHOT_SKIP = {"TICKERS", "SECTOR_MAP", "COMPANY_NAME_MAP"}


def _code_hash() -> str:
    """피처 계산에 영향을 주는 소스의 지문. 파일을 고칠 때만 바뀜 (Optuna 런타임 오버라이드는 캐시를 깨지 않음)."""
    src_root = Path(__file__).resolve().parents[1]
    h = hashlib.md5()
    for rel in _HASHED_SOURCES:
        p = src_root / rel
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:8]


def _config_snapshot() -> dict[str, Any]:
    """JSON으로 저장 가능한 config 값 전체. meta.json은 공개 리포에 올라가므로 비밀·개인정보 키는 제외."""
    import screener.config as cfg

    snap: dict[str, Any] = {}
    for k in dir(cfg):
        if not k.isupper() or k in _SNAPSHOT_SKIP or any(m in k for m in _SECRET_MARKERS):
            continue
        v = getattr(cfg, k)
        try:
            json.dumps(v)
        except TypeError:
            continue
        snap[k] = v
    return snap


def default_universe() -> list[str]:
    """기본 백테스트 유니버스 (고정 순서).

    TICKERS는 set으로 만들어져 프로세스마다 순서가 달라서, max_tickers로 자르면
    실행할 때마다 다른 종목이 뽑혔다. 정렬 후 고정 시드로 섞어 항상 같은 종목을 쓴다.
    """
    universe = sorted(TICKERS)
    random.Random(42).shuffle(universe)
    return universe


def _pit_universe_tickers(membership: Any, period: str) -> list[str]:
    """기간 중 한 번이라도 S&P 500이었던 종목 (고정 순서). 나중에 빠진 종목까지 넣어야 생존 편향이 줄어든다."""
    years = int(period[:-1]) if period.endswith("y") and period[:-1].isdigit() else 10
    today = date.today()
    start = date(today.year - years, today.month, min(today.day, 28))
    universe = sorted(membership.union_between(str(start), str(today)))
    random.Random(42).shuffle(universe)
    return universe


def _equity_now(positions: list["BtPosition"], closes: pd.DataFrame, date_ts: pd.Timestamp, cash: float) -> float:
    """현금 + 보유 종목 오늘 종가 평가액 (종가가 없으면 매수가)."""
    value = cash
    for p in positions:
        px = float(closes.at[date_ts, p.ticker]) if p.ticker in closes.columns else float("nan")
        value += p.shares * (px if not np.isnan(px) and px > 0 else p.entry_price)
    return value


def _next_open(opens: pd.DataFrame, next_date_ts: pd.Timestamp, ticker: str) -> float | None:
    """다음날 시가 (없거나 0이면 None)."""
    if ticker not in opens.columns:
        return None
    price = float(opens.at[next_date_ts, ticker])
    return None if np.isnan(price) or price == 0 else price


@dataclass
class BtPosition:
    """백테스트용 인메모리 포지션."""
    ticker: str
    entry_price: float
    entry_date: str
    strategy: str
    star_rating: str
    ccs_score: float
    sector: str
    highest_price: float = field(default=0.0)
    shares: float = field(default=0.0)  # 보유 주식 수 (자본 추적 모드)
    entry_features: dict[str, Any] = field(default_factory=dict)
    # Hold-Winners 재평가 상태
    defer_count: int = 0
    trailing_stop_override: float | None = None
    last_defer_date: str | None = None
    defer_debug: dict | None = None

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date,
            "strategy": self.strategy,
            "star_rating": self.star_rating,
            "ccs_score": self.ccs_score,
            "sector": self.sector,
            "highest_price": self.highest_price,
            "shares": self.shares,
            "defer_count": self.defer_count,
            "trailing_stop_override": self.trailing_stop_override,
            "last_defer_date": self.last_defer_date,
            "defer_debug": self.defer_debug,
        }


def _close_position(
    positions: list[BtPosition],
    trades: list[dict],
    ticker: str,
    exit_price: float,
    exit_date: str,
    reason: str,
) -> tuple[dict, float] | tuple[None, float]:
    """포지션 청산 및 거래 기록 추가.

    Returns:
        (trade_dict, proceeds) — proceeds는 매도대금 (자본 추적 모드 시 사용)
    """
    for i, pos in enumerate(positions):
        if pos.ticker == ticker:
            positions.pop(i)
            entry_price = pos.entry_price
            entry_eff = entry_price * (1 + BACKTEST_COST_PER_SIDE)
            exit_eff = exit_price * (1 - BACKTEST_COST_PER_SIDE)
            return_pct = (exit_eff - entry_eff) / entry_eff if entry_price else 0.0
            entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
            exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
            holding_days = (exit_dt - entry_dt).days
            proceeds = pos.shares * exit_eff if pos.shares > 0 else 0.0
            dollar_pnl = pos.shares * (exit_eff - entry_eff) if pos.shares > 0 else 0.0
            trade = {
                "ticker": ticker,
                "entry_date": pos.entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": round(return_pct, 4),
                "holding_days": holding_days,
                "strategy": pos.strategy,
                "star_rating": pos.star_rating,
                "ccs_score": pos.ccs_score,
                "sector": pos.sector,
                "exit_reason": reason,
                "shares": pos.shares,
                "dollar_pnl": round(dollar_pnl, 2),
                "entry_features": pos.entry_features,
                "defer_count": pos.defer_count,
                "last_defer_date": pos.last_defer_date,
            }
            trades.append(trade)
            return trade, proceeds
    return None, 0.0


def _calculate_metrics(
    trades: list[dict],
    spy_returns: pd.Series | None = None,
    initial_capital: float = 0.0,
    final_cash: float = 0.0,
    equity_series: pd.Series | None = None,
) -> dict[str, Any]:
    """성과 지표 계산."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    returns = df["return_pct"]
    total = len(df)
    wins = (returns > 0).sum()
    win_rate = wins / total if total else 0

    avg_return = returns.mean()
    avg_win = returns[returns > 0].mean() if wins > 0 else 0
    avg_loss = returns[returns <= 0].mean() if (total - wins) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    avg_holding = df["holding_days"].mean() if total else 0

    # MDD — 달러 추적 모드면 equity curve 기반, 아니면 순차 복리 기반
    # inf/-inf, 극단값 제거 후 cumprod (-99%~+1000% 클리핑) → RuntimeWarning 방지
    safe_returns = returns.dropna().replace([float("inf"), float("-inf")], float("nan")).dropna().clip(-0.99, 10.0)
    cumulative = (1 + safe_returns).cumprod()
    if equity_series is not None and len(equity_series) > 1 and initial_capital > 0:
        eq = equity_series.dropna()
        rolling_max = eq.expanding().max()
        mdd = float(((eq - rolling_max) / rolling_max).min())
    else:
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        mdd = drawdown.min()

    # Sharpe
    if len(returns) > 1 and returns.std() > 0:
        trades_per_year = 252 / max(avg_holding, 1)
        sharpe = (avg_return * trades_per_year) / (returns.std() * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # 매도사유 분포
    exit_dist = df["exit_reason"].value_counts().to_dict()

    # 전략별 성과
    strategy_perf: dict[str, dict] = {}
    for strat, grp in df.groupby("strategy"):
        key = str(strat).replace("📉 ", "").replace("📈 ", "").replace("🔄 ", "")
        g_ret = grp["return_pct"]
        strategy_perf[key] = {
            "건수": len(grp),
            "승률": round((g_ret > 0).mean(), 4),
            "평균수익률": round(g_ret.mean(), 4),
        }

    summary: dict[str, Any] = {
        "총거래수": int(total),
        "승률": round(win_rate, 4),
        "평균수익률": round(avg_return, 4),
        "평균승리": round(avg_win, 4),
        "평균손실": round(avg_loss, 4),
        "승패비율": round(win_loss_ratio, 2),
        "평균보유일": round(avg_holding, 1),
        "MDD": round(mdd, 4),
        "Sharpe": round(sharpe, 2),
        "전략별": strategy_perf,
        "매도사유": exit_dist,
    }

    if spy_returns is not None and len(spy_returns) > 0:
        spy_total = float((1 + spy_returns).prod() - 1)
        if initial_capital > 0 and final_cash > 0:
            # 달러 추적 모드: 실제 포트폴리오 수익률 사용
            strat_total = (final_cash - initial_capital) / initial_capital
        else:
            strat_total = float(cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0
        summary["SPY수익률"] = round(spy_total, 4)
        summary["전략총수익률"] = round(strat_total, 4)
        summary["SPY초과수익"] = round(strat_total - spy_total, 4)

    # 자본 추적 모드 달러 통계
    if initial_capital > 0 and "dollar_pnl" in df.columns:
        total_dollar_pnl = df["dollar_pnl"].sum()
        summary["초기자본"] = round(initial_capital, 2)
        summary["최종자본"] = round(final_cash, 2)
        summary["총수익금"] = round(total_dollar_pnl, 2)
        summary["총수익률_자본기준"] = round((final_cash - initial_capital) / initial_capital, 4)

    if equity_series is not None and len(equity_series) > 20 and initial_capital > 0:
        eq = equity_series.dropna()
        daily = eq.pct_change().dropna()
        if daily.std() > 0:
            summary["Sharpe_일간"] = round(float(daily.mean() / daily.std() * np.sqrt(252)), 2)
            down = daily[daily < 0].std()
            summary["Sortino_일간"] = round(float(daily.mean() / down * np.sqrt(252)), 2) if down > 0 else None
        years = len(eq) / 252
        if years > 0 and eq.iloc[0] > 0:
            cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
            summary["CAGR"] = round(cagr, 4)
            summary["Calmar"] = round(cagr / abs(mdd), 2) if mdd else None
    if spy_returns is not None and len(spy_returns) > 20 and spy_returns.std() > 0:
        summary["SPY_Sharpe_일간"] = round(float(spy_returns.mean() / spy_returns.std() * np.sqrt(252)), 2)

    return summary


def _safe_val(v: Any) -> Any:
    """NaN/inf를 None으로 변환 (JSON 직렬화 안전)."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    except (TypeError, ValueError):
        return v


def _extract_entry_features(
    ranked_df: pd.DataFrame,
    ticker: str,
    candidate: dict,
    cand_debug: dict,
) -> dict[str, Any]:
    """진입 시점 피처를 ranked_df에서 추출.

    Returns:
        기술지표, 알파팩터, 전략점수, CCS 서브스코어, 시장상태, 섹터 피처 dict
    """
    row_mask = ranked_df["티커"] == ticker if "티커" in ranked_df.columns else ranked_df.index == ticker
    rows = ranked_df[row_mask]
    row: dict = rows.iloc[0].to_dict() if not rows.empty else {}

    def g(key: str, default: Any = None) -> Any:
        return _safe_val(row.get(key, default))

    feats: dict[str, Any] = {
        # 기술지표
        "RSI": g("RSI"),
        "adx": g("adx"),
        "macd_hist": g("macd_hist"),
        "bollinger_pband": g("bollinger_pband"),
        "atr_pct": g("ATR%"),
        "vol_z20": g("거래량Z(20)"),
        "obv_z20": g("obv_z20"),
        "cmf_20": g("cmf_20"),
        "pos_52w": g("52주포지션"),
        "ret_5d": g("5일수익률"),
        "ret_20d": g("20일수익률"),
        "ema_gap_20_50": g("ema_gap_20_50"),
        "ema_gap_50_200": g("ema_gap_50_200"),
        "volatility_contraction": g("변동성압축"),
        "buy_support_count": g("buy_support_count"),
        # 알파팩터
        "alpha_score": g("알파점수"),
        "factor_momentum": g("팩터_모멘텀"),
        "factor_trend": g("팩터_추세"),
        "factor_volume": g("팩터_거래량"),
        "factor_volatility": g("팩터_변동성"),
        "factor_mean_reversion": g("팩터_평균회귀"),
        # 전략점수
        "bottom_reversal_fit": g("바닥반등_적합도"),
        "momentum_fit": g("모멘텀_적합도"),
        "reversal_score": g("반등스코어"),
        "low_prob": g("저점확률"),
        # 섹터
        "sector": g("섹터"),
        "in_strong_sector": g("in_strong_sector"),
        "sector_relative_strength": g("섹터상대강도"),
        # 시장상태
        "regime": cand_debug.get("regime"),
        "spy_close_to_ema200": _safe_val(row.get("close_to_ema200_pct")),
    }

    # CCS 서브스코어
    ccs_breakdown: dict = candidate.get("ccs_breakdown", {})
    for k in ("strategy_fit", "timing", "alpha", "risk", "confluence", "penalty"):
        feats[f"ccs_{k}"] = _safe_val(ccs_breakdown.get(k))
    # v1·v2 점수를 모두 남겨 같은 거래로 IC를 비교한다
    feats["ccs_v1"] = _safe_val(candidate.get("ccs_v1"))
    feats["ccs_v2"] = _safe_val(candidate.get("ccs_v2"))
    for k, v in (candidate.get("ccs_v2_breakdown") or {}).items():
        if k != "strategy_fit":
            feats[f"ccs_{k}"] = _safe_val(v)

    return feats


def run_paper_trading_backtest(
    tickers: list[str] | None = None,
    *,
    period: str = "1y",
    rebalance_every: int = 5,
    include_fundamentals: bool = False,
    min_history_days: int = 220,
    max_tickers: int | None = 100,
    initial_capital: float = 0.0,
    no_cache: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    save_run: bool = True,
    pit_universe: bool | None = None,
) -> dict[str, Any]:
    """페이퍼 트레이딩 로직을 과거 데이터로 시뮬레이션.

    Args:
        tickers: 분석할 종목 리스트 (None이면 default_universe())
        no_cache: True이면 디스크 캐시를 무시하고 처음부터 전체 재계산
        period: 데이터 기간 ('1y', '2y' 등)
        rebalance_every: 후보 선정 주기 (거래일 기준)
        include_fundamentals: 펀더멘탈 데이터 포함 여부 (느림)
        min_history_days: 기술적 지표 계산에 필요한 최소 히스토리 일수
        max_tickers: 최대 티커 수 제한 (None이면 전체)
        initial_capital: 초기 자본금 (0이면 수익률 모드, >0이면 달러 추적 모드)
        start_date/end_date: 시뮬레이션 구간 YYYY-MM-DD (walk-forward·홀드아웃용)
        save_run: False면 output/ 에 파일을 남기지 않는다 (Optuna 반복 실행용)
        pit_universe: True면 그날의 S&P 500 구성종목만 후보로 (None이면 config.BACKTEST_PIT_UNIVERSE)

    Returns:
        {"trades": list[dict], "summary": dict, "equity_curve": pd.Series}
    """
    membership = None
    if _cfg.BACKTEST_PIT_UNIVERSE if pit_universe is None else pit_universe:
        from paper_trading.universe import Membership, load_membership
        membership = Membership(load_membership(_PROJECT_ROOT / _cfg.SP500_MEMBERSHIP_PATH))
        if tickers is None:
            tickers = _pit_universe_tickers(membership, period)
            print(f"[백테스트] PIT 유니버스: 기간 중 S&P 500 이었던 {len(tickers)}개 종목")
    earnings_cal = EarningsCalendar.load(_PROJECT_ROOT / "data" / "research" / "sec_earnings_dates.parquet")
    if tickers is None:
        tickers = default_universe()

    # SPY 추가 (레짐 감지 + 벤치마크)
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers

    if max_tickers is not None:
        non_spy = [t for t in tickers if t != "SPY"]
        tickers = ["SPY"] + non_spy[:max_tickers]

    # NaN이 포함된 주가 데이터의 rolling/cumulative 연산에서 발생하는 numpy 경고 억제
    warnings.filterwarnings("ignore", message="invalid value encountered in accumulate")

    logger.info(f"[BtPaper] 데이터 다운로드: {len(tickers)}개 종목, 기간={period}")
    print(f"[백테스트] 데이터 다운로드 중... ({len(tickers)}개 종목, {period})")

    raw = fetch_ohlcv(tickers, period=period, force_download=no_cache)
    if raw.empty:
        raise RuntimeError("OHLCV 데이터를 가져오지 못했습니다.")

    closes = raw.xs("Close", level=1, axis=1)
    opens = raw.xs("Open", level=1, axis=1)
    lows = raw.xs("Low", level=1, axis=1)
    dates = closes.index.tolist()

    if len(dates) <= min_history_days + 5:
        raise ValueError(f"분석 기간이 너무 짧습니다. (필요: {min_history_days+5}일, 실제: {len(dates)}일)")

    # SPY 벤치마크
    spy_closes = closes.get("SPY")
    spy_returns: pd.Series | None = None
    if spy_closes is not None:
        spy_returns = spy_closes.pct_change().dropna()

    # OHLCV 해시 계산 (캐시 키)
    ohlcv_key = ",".join(sorted(tickers)) + "|" + period + f"|fund={int(include_fundamentals)}|code={_code_hash()}"
    ohlcv_hash = hashlib.md5(ohlcv_key.encode()).hexdigest()[:12]

    # Feature 캐시 유효성 확인
    use_feature_cache = not no_cache and cache_meta_valid(
        ohlcv_hash, period, include_fundamentals
    )
    if use_feature_cache:
        cached_count = count_cached_snapshots(ohlcv_hash)
        print(f"[백테스트] Feature 캐시 감지: {cached_count}개 날짜 로드 예정 (빠른 모드)")
    else:
        if no_cache:
            # --no-cache: 기존 feature/IC 캐시 삭제 후 새로 생성
            import shutil
            from screener.cache import feature_cache_dir as _fcd, _IC_CACHE_DIR
            old_dir = _fcd(ohlcv_hash)
            if old_dir.exists():
                shutil.rmtree(old_dir)
                print(f"[백테스트] 기존 feature 캐시 삭제: {old_dir}")
            for ic_file in _IC_CACHE_DIR.glob(f"{ohlcv_hash}_*.json"):
                ic_file.unlink()
        write_cache_meta(ohlcv_hash, period, include_fundamentals)

    # Fundamentals 사전 로드 (시뮬레이션 시작 전 한 번만)
    prefetched_fundamentals: "pd.DataFrame | None" = None
    if include_fundamentals:
        from screener.fundamentals import fetch_fundamental_snapshots
        non_spy_tickers = [t for t in tickers if t != "SPY"]
        print(f"[백테스트] Fundamentals 수집 중... ({len(non_spy_tickers)}개 종목)")
        prefetched_fundamentals = fetch_fundamental_snapshots(
            non_spy_tickers, use_cache=(not no_cache)
        )

    if prefetched_fundamentals is not None and BACKTEST_FUNDAMENTALS_PIT_SAFE:
        keep = [c for c in ("티커", "fund_sector") if c in prefetched_fundamentals.columns]
        prefetched_fundamentals = prefetched_fundamentals[keep]

    price_map = _prepare_price_map(raw)
    # 섹터 강도 계산용 섹터 ETF 일봉 — 따로 받아 price_map에만 넣는다 (tickers·OHLCV 해시는 그대로)
    # 예전엔 ETF가 없어 강한 섹터 = ∅ → buy_signal 상시 False (실거래와 다른 전략을 쟀다)
    etf_tickers = [t for t in SECTOR_ETFS.values() if t not in price_map]
    if etf_tickers:
        try:
            etf_raw = fetch_ohlcv(etf_tickers, period=period, force_download=no_cache)
            if not etf_raw.empty:
                price_map.update(_prepare_price_map(etf_raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[BtPaper] 섹터 ETF 다운로드 실패 — 섹터 필터 없이 진행: {exc}")
    ranked_cache: dict[pd.Timestamp, pd.DataFrame] = {}
    ic_weights_cache: dict = {}  # IC 가중치 캐시 (20거래일마다 재계산)

    use_capital = initial_capital > 0
    cash: float = initial_capital

    positions: list[BtPosition] = []
    trades: list[dict] = []
    equity_points: list[tuple[str, float]] = []

    start_idx = min_history_days
    last_idx = len(dates) - 1
    if start_date:
        start_idx = max(start_idx, next((i for i, d in enumerate(dates) if str(d.date()) >= start_date), last_idx))
    if end_date:
        last_idx = max((i for i, d in enumerate(dates) if str(d.date()) <= end_date), default=start_idx)
    if last_idx - start_idx < 2:
        raise ValueError(f"시뮬레이션 구간이 너무 짧습니다 (start={start_date}, end={end_date})")
    sim_start_date = str(dates[start_idx].date())
    sim_end_date = str(dates[last_idx].date())

    total_days = last_idx - start_idx
    sim_wall_start = datetime.now()
    print(f"[백테스트] 시뮬레이션 시작: {sim_start_date} ~ {sim_end_date} ({total_days}거래일)")

    for idx in range(start_idx, last_idx):
        date_ts = dates[idx]
        next_date_ts = dates[idx + 1]
        today_str = str(date_ts.date())
        next_str = str(next_date_ts.date())

        # ── 0. 장중 손절 (PT1_STOP_INTRADAY): 오늘 저가가 손절/트레일링선을 건드리면 그 가격에 오늘 청산 ──
        if _cfg.PT1_STOP_INTRADAY:
            for pos in list(positions):
                if pos.ticker not in lows.columns or pos.entry_date > today_str:
                    continue
                hit = intraday_stop_exit(pos.to_dict(), float(opens.at[date_ts, pos.ticker]),
                                         float(lows.at[date_ts, pos.ticker]))
                if hit is None:
                    continue
                fill, reason = hit
                _, proceeds = _close_position(positions, trades, pos.ticker, fill, today_str, reason)
                if use_capital:
                    cash += proceeds

        # ── 1. 현재가로 highest_price 업데이트 ──────────────────
        for pos in positions:
            if pos.ticker in closes.columns:
                cur = float(closes.at[date_ts, pos.ticker])
                if not np.isnan(cur) and cur > pos.highest_price:
                    pos.highest_price = cur

        # ── 2. 매도 판단 (매일, Hold-Winners 재평가 포함) ────────
        sells_to_exec: list[tuple[str, float, str]] = []
        _defer_ranked_df: pd.DataFrame | None = None  # profit-trigger 시에만 지연 계산
        for pos in positions:
            if pos.ticker not in closes.columns:
                continue
            cur = float(closes.at[date_ts, pos.ticker])
            if np.isnan(cur):
                continue
            pos_dict = pos.to_dict()
            should_sell, reason = check_sell_conditions(pos_dict, cur, today_str)
            if not should_sell:
                continue

            profit_driven = reason.startswith("목표가") or reason.startswith("시간익절")
            if profit_driven:
                if _defer_ranked_df is None:
                    _defer_ranked_df = _compute_ranked_snapshot(
                        price_map, date_ts,
                        include_fundamentals=include_fundamentals,
                        cache=ranked_cache,
                        ic_weights_cache=ic_weights_cache,
                        ohlcv_hash=ohlcv_hash,
                        fundamentals_df=prefetched_fundamentals,
                    )
                row = _get_row_for_ticker(_defer_ranked_df, pos.ticker)
                defer, debug = _should_defer_sell(pos_dict, row, cur)
                if defer:
                    # 포지션 객체에 defer 상태 반영
                    _activate_tight_trail(pos_dict, cur, today_str, debug)
                    pos.defer_count = pos_dict["defer_count"]
                    pos.trailing_stop_override = pos_dict["trailing_stop_override"]
                    pos.last_defer_date = pos_dict["last_defer_date"]
                    pos.defer_debug = pos_dict["defer_debug"]
                    pos.highest_price = pos_dict["highest_price"]
                    logger.debug(
                        f"[BtPaper] DEFER {pos.ticker} @ {cur:.2f} "
                        f"({reason} → tight trail, defer {pos.defer_count})"
                    )
                    continue
                else:
                    # 실패 이유 로그 (engine.py와 동일한 패턴)
                    failed = [k for k, v in debug.get("checks", {}).items() if not v]
                    if failed:
                        logger.debug(
                            f"[BtPaper] DEFER SKIP {pos.ticker}: {', '.join(failed)} "
                            f"(RSI={debug.get('RSI', 0):.1f}, ADX={debug.get('adx', 0):.1f}, "
                            f"vol_mult={debug.get('vol_mult', 0):.2f}, "
                            f"ret_5d={debug.get('ret_5d', 0):+.2%}, "
                            f"bb_pband={debug.get('bb_pband', 0):.2f})"
                        )

            # 다음날 시가로 청산
            exit_price = float(opens.at[next_date_ts, pos.ticker]) if pos.ticker in opens.columns else cur
            if np.isnan(exit_price):
                exit_price = cur
            sells_to_exec.append((pos.ticker, exit_price, reason))

        for ticker, exit_price, reason in sells_to_exec:
            _, proceeds = _close_position(positions, trades, ticker, exit_price, next_str, reason)
            if use_capital:
                cash += proceeds
            logger.debug(f"[BtPaper] SELL {ticker} @ {exit_price:.2f} ({reason}) on {next_str}")

        # ── 3. 후보 선정 (리밸런스 주기마다) ────────────────────
        elapsed = idx - start_idx
        progress = elapsed / total_days * 100
        step_marker = elapsed % 50 == 0
        wall_elapsed = datetime.now() - sim_wall_start
        wall_str = f"{int(wall_elapsed.total_seconds() // 60):02d}:{int(wall_elapsed.total_seconds() % 60):02d}"
        pos_summary = ", ".join(p.ticker for p in positions) if positions else "없음"
        line = (
            f"  [{progress:5.1f}%] {today_str}"
            f" | 보유: {len(positions)}개 ({pos_summary})"
            f" | 거래: {len(trades)}건"
            f" | {wall_str}"
        )
        if step_marker:
            sys.stdout.write(line + "\n")
        else:
            sys.stdout.write(line + "\r")
        sys.stdout.flush()

        if elapsed % max(1, rebalance_every) != 0:
            # 리밸런스 아닌 날: 에쿼티 기록만
            _record_equity(equity_points, positions, closes, date_ts, today_str, cash=cash, use_capital=use_capital)
            continue

        ranked_df = _compute_ranked_snapshot(
            price_map,
            date_ts,
            include_fundamentals=include_fundamentals,
            cache=ranked_cache,
            ic_weights_cache=ic_weights_cache,
            ohlcv_hash=ohlcv_hash,
            fundamentals_df=prefetched_fundamentals,
        )
        if membership is not None and "티커" in ranked_df.columns:
            ranked_df = ranked_df[ranked_df["티커"].isin(membership.on(today_str) | {"SPY"})]
        ranked_df = earnings_cal.annotate(ranked_df, date_ts)  # 어닝 회피 필터용 (실거래와 같게)

        if ranked_df.empty:
            _record_equity(equity_points, positions, closes, date_ts, today_str, cash=cash, use_capital=use_capital)
            continue

        pos_dicts = [p.to_dict() for p in positions]
        candidates, cand_debug = select_top_candidates(
            ranked_df, pos_dicts, k=max(1, int(_cfg.PAPER_TRADING_MAX_DAILY_BUY)),
        )

        # 약세장 최대 포지션 축소
        bt_regime = cand_debug.get("regime", "neutral")
        max_positions = int(_cfg.PAPER_TRADING_MAX_POSITIONS)
        effective_max = max_positions if bt_regime != "bear" else max(1, max_positions - 1)

        # 약세장 보유 정리 (PT1_BEAR_MODE=liquidate): 다음날 시가에 전부 매도
        if bt_regime == "bear" and bear_liquidate() and positions:
            for pos in list(positions):
                exit_price = _next_open(opens, next_date_ts, pos.ticker)
                if exit_price is None and pos.ticker in closes.columns:
                    cur = float(closes.at[date_ts, pos.ticker])
                    exit_price = cur if not np.isnan(cur) and cur > 0 else None
                if exit_price is None:
                    continue
                _, proceeds = _close_position(positions, trades, pos.ticker, exit_price, next_str, "약세장정리")
                if use_capital:
                    cash += proceeds

        # ── 4. 매수/교체 판단 ───────────────────────────────────
        if len(positions) < effective_max:
            # 빈 슬롯 → 신규 매수 (하루 최대 PAPER_TRADING_MAX_DAILY_BUY개)
            for candidate in candidates:
                if len(positions) >= effective_max:
                    break
                cand_ticker = candidate["ticker"]
                entry_price = _next_open(opens, next_date_ts, cand_ticker)
                if entry_price is None:
                    continue
                if use_capital:
                    empty_slots = effective_max - len(positions)
                    allocation = slot_allocation(cash, _equity_now(positions, closes, date_ts, cash), effective_max,
                                                 empty_slots, ranked_df, cand_ticker)
                    shares = allocation / (entry_price * (1 + BACKTEST_COST_PER_SIDE))
                    cash -= allocation
                else:
                    shares = 0.0
                positions.append(BtPosition(
                    ticker=cand_ticker,
                    entry_price=entry_price,
                    entry_date=next_str,
                    strategy=candidate["strategy"],
                    star_rating=candidate["star_rating"],
                    ccs_score=candidate["ccs_score"],
                    sector=candidate["sector"],
                    highest_price=entry_price,
                    shares=shares,
                    entry_features=_extract_entry_features(ranked_df, cand_ticker, candidate, cand_debug),
                ))
                logger.debug(f"[BtPaper] BUY {cand_ticker} @ {entry_price:.2f} (CCS={candidate['ccs_score']:.4f}) on {next_str}")

        elif candidates:
            # 풀슬롯 → 교체 검토 (하루 1건)
            candidate = candidates[0]
            cand_ticker = candidate["ticker"]
            entry_price = _next_open(opens, next_date_ts, cand_ticker)
            cur_prices = {
                p.ticker: float(closes.at[date_ts, p.ticker])
                for p in positions
                if p.ticker in closes.columns and not np.isnan(float(closes.at[date_ts, p.ticker]))
            }
            worst = get_worst_position(pos_dicts, cur_prices, ranked_df) if entry_price is not None else None
            if worst and should_replace(candidate["ccs_score"], current_ccs(worst, cand_debug)):
                worst_ticker = worst["ticker"]
                worst_price = cur_prices.get(worst_ticker, worst["entry_price"])
                _, proceeds = _close_position(positions, trades, worst_ticker, worst_price, next_str, f"교체→{cand_ticker}")
                if use_capital:
                    cash += proceeds
                    allocation = slot_allocation(cash, _equity_now(positions, closes, date_ts, cash), effective_max,
                                                 effective_max - len(positions), ranked_df, cand_ticker)
                    shares = allocation / (entry_price * (1 + BACKTEST_COST_PER_SIDE))
                    cash -= allocation
                else:
                    shares = 0.0
                positions.append(BtPosition(
                    ticker=cand_ticker,
                    entry_price=entry_price,
                    entry_date=next_str,
                    strategy=candidate["strategy"],
                    star_rating=candidate["star_rating"],
                    ccs_score=candidate["ccs_score"],
                    sector=candidate["sector"],
                    highest_price=entry_price,
                    shares=shares,
                    entry_features=_extract_entry_features(ranked_df, cand_ticker, candidate, cand_debug),
                ))
                logger.debug(f"[BtPaper] REPLACE {worst_ticker} → {cand_ticker} on {next_str}")

        _record_equity(equity_points, positions, closes, date_ts, today_str, cash=cash, use_capital=use_capital)

    # ── 잔여 포지션 강제 청산 ────────────────────────────────────
    final_date_ts = dates[last_idx]
    final_str = str(final_date_ts.date())
    for pos in list(positions):
        if pos.ticker in closes.columns:
            exit_price = float(closes.at[final_date_ts, pos.ticker])
            if not np.isnan(exit_price) and exit_price > 0:
                _, proceeds = _close_position(positions, trades, pos.ticker, exit_price, final_str, "기간종료")
                if use_capital:
                    cash += proceeds

    # ── 청산 후 가격 변동 추가 (손절 회복 / 익절 잔여수익 진단) ──────
    date_to_idx: dict[str, int] = {str(d.date()): i for i, d in enumerate(dates)}
    for trade in trades:
        exit_d = trade.get("exit_date", "")
        exit_idx = date_to_idx.get(exit_d)
        tkr = trade.get("ticker", "")
        if exit_idx is None or tkr not in closes.columns:
            trade["post_exit_return_5d"] = None
            trade["post_exit_return_10d"] = None
            trade["post_exit_return_20d"] = None
            continue
        ep = trade.get("exit_price") or 0.0
        for lag, key in ((5, "post_exit_return_5d"), (10, "post_exit_return_10d"), (20, "post_exit_return_20d")):
            future_idx = exit_idx + lag
            if future_idx < len(dates) and ep > 0:
                future_price = float(closes.at[dates[future_idx], tkr])
                trade[key] = round((future_price - ep) / ep, 4) if not np.isnan(future_price) else None
            else:
                trade[key] = None

    # ── 성과 지표 계산 (파일 저장보다 먼저 수행) ──────────────
    if spy_returns is not None:
        spy_in_range = spy_returns[(spy_returns.index >= dates[start_idx]) & (spy_returns.index <= dates[last_idx])]
    else:
        spy_in_range = None

    equity_curve = pd.Series(
        {pd.Timestamp(d): v for d, v in equity_points},
        name="equity_curve",
    ) if equity_points else pd.Series(dtype=float)

    summary = _calculate_metrics(
        trades, spy_in_range,
        initial_capital=initial_capital, final_cash=cash,
        equity_series=equity_curve if not equity_curve.empty else None,
    )
    summary["시뮬레이션_시작"] = sim_start_date
    summary["시뮬레이션_종료"] = sim_end_date
    if use_capital and summary.get("총거래수"):
        from paper_trading.benchmarks import benchmark_summary
        summary.update(benchmark_summary(
            equity_curve, closes,
            top_n=int(_cfg.BENCHMARK_MOMENTUM_TOP_N),
            cost_per_side=BACKTEST_COST_PER_SIDE,
            members_on=membership.on if membership is not None else None,
        ))
        summary["PIT_유니버스"] = membership is not None

    if not save_run:
        sys.stdout.write("\n")
        return {"trades": trades, "summary": summary, "equity_curve": equity_curve, "enhanced_trades_path": None}

    # ── 향상된 거래 로그 저장 ────────────────────────────────────
    import json, os, csv
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    enhanced_json_path = os.path.join(output_dir, "backtest_trades_enhanced.json")
    with open(enhanced_json_path, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2, default=str)

    # CSV: entry_features 플랫화
    flat_trades = []
    if trades:
        for t in trades:
            row_flat = {k: v for k, v in t.items() if k != "entry_features"}
            for feat_k, feat_v in (t.get("entry_features") or {}).items():
                row_flat[f"feat_{feat_k}"] = feat_v
            flat_trades.append(row_flat)
        enhanced_csv_path = os.path.join(output_dir, "backtest_trades_enhanced.csv")
        all_keys = list(flat_trades[0].keys())
        with open(enhanced_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_trades)
        print(f"[백테스트] 향상된 거래 로그 저장: {enhanced_json_path}, {enhanced_csv_path}")

    # ── 버전 run 폴더 자동 저장 ────────────────────────────────────
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(output_dir, "runs", run_ts)
    os.makedirs(run_dir, exist_ok=True)

    # JSON/CSV 복사
    import shutil
    shutil.copy(enhanced_json_path, os.path.join(run_dir, "backtest_trades_enhanced.json"))
    if flat_trades:
        shutil.copy(enhanced_csv_path, os.path.join(run_dir, "backtest_trades_enhanced.csv"))

    # meta.json: 성과 수치 + 파라미터 스냅샷 + 플래그 + git + 벤치마크
    try:
        from screener.config import (
            ALPHA_DEFAULT_WEIGHTS, CANDIDATE_REGIME_WEIGHTS,
            ADX_BUY_MIN, EXIT_PARAMS, EXIT_PARAMS_DEFAULT,
            HOLD_WINNERS_MIN_CHECKS, HOLD_WINNERS_DEFER_FREEZE_DAYS,
            HOLD_WINNERS_MAX_DEFERS,
        )
        _df_ret = [t["return_pct"] for t in trades if "return_pct" in t]
        _wins = [r for r in _df_ret if r > 0]
        _losses = [r for r in _df_ret if r <= 0]
        _n = len(_df_ret)

        _by_strat: dict = {}
        for t in trades:
            sk = str(t.get("strategy", "")).replace("📉 ", "").replace("📈 ", "").replace("🔄 ", "").strip()
            if sk not in _by_strat:
                _by_strat[sk] = {"n": 0, "wins": 0, "total_ret": 0.0}
            _by_strat[sk]["n"] += 1
            _by_strat[sk]["wins"] += int(t.get("return_pct", 0) > 0)
            _by_strat[sk]["total_ret"] += float(t.get("return_pct", 0))

        _strat_summary = {
            k: {
                "n": v["n"],
                "win_rate": round(v["wins"] / v["n"], 4) if v["n"] > 0 else 0,
                "avg_return": round(v["total_ret"] / v["n"], 4) if v["n"] > 0 else 0,
            }
            for k, v in _by_strat.items()
        }

        _defer_triggered = sum(1 for t in trades if int(t.get("defer_count", 0) or 0) > 0)
        _defer_total = sum(int(t.get("defer_count", 0) or 0) for t in trades)

        meta = {
            "run_label": run_ts,
            "code_hash": _code_hash(),
            "config_hash": hashlib.md5(
                json.dumps(_config_snapshot(), sort_keys=True, ensure_ascii=False, default=str).encode()
            ).hexdigest()[:8],
            "equity_metrics": {
                k: summary.get(k)
                for k in ("CAGR", "Sharpe_일간", "Sortino_일간", "Calmar", "SPY_Sharpe_일간")
            },
            "baselines": {
                k: v for k, v in summary.items()
                if k.startswith(("기준_", "알파_", "베타_", "모멘텀대비_", "PIT_"))
            },
            "cost_per_side": BACKTEST_COST_PER_SIDE,
            "fundamentals_pit_safe": BACKTEST_FUNDAMENTALS_PIT_SAFE,
            "config_full": _config_snapshot(),
            "period": period,
            "capital": initial_capital,
            "flags": {
                "include_fundamentals": bool(include_fundamentals),
                "no_cache": bool(no_cache),
                "rebalance_every": int(rebalance_every),
                "max_positions": int(_cfg.PAPER_TRADING_MAX_POSITIONS),
                "max_daily_buy": int(_cfg.PAPER_TRADING_MAX_DAILY_BUY),
                "pit_universe": membership is not None,
                "min_history_days": int(min_history_days),
            },
            "git": _git_info(),
            "simulation": {
                "start_date": sim_start_date,
                "end_date": sim_end_date,
                "trading_days": int(total_days),
                "ticker_count": len(tickers),
            },
            "benchmark": {
                "spy_return": summary.get("SPY수익률"),
                "strategy_return": summary.get("전략총수익률"),
                "excess_return": summary.get("SPY초과수익"),
                "mdd": summary.get("MDD"),
                "sharpe": summary.get("Sharpe"),
                "initial_capital": summary.get("초기자본"),
                "final_capital": summary.get("최종자본"),
            },
            "hold_winners": {
                "defer_triggered": _defer_triggered,
                "defer_total_count": _defer_total,
                "MIN_CHECKS": HOLD_WINNERS_MIN_CHECKS,
                "DEFER_FREEZE_DAYS": HOLD_WINNERS_DEFER_FREEZE_DAYS,
                "MAX_DEFERS": HOLD_WINNERS_MAX_DEFERS,
            },
            "performance": {
                "total_trades": _n,
                "win_rate": round(sum(1 for r in _df_ret if r > 0) / _n, 4) if _n > 0 else 0,
                "avg_return": round(sum(_df_ret) / _n, 4) if _n > 0 else 0,
                "avg_win": round(sum(_wins) / len(_wins), 4) if _wins else 0,
                "avg_loss": round(sum(_losses) / len(_losses), 4) if _losses else 0,
                "by_strategy": _strat_summary,
            },
            "params": {
                "ALPHA_DEFAULT_WEIGHTS": ALPHA_DEFAULT_WEIGHTS,
                "CANDIDATE_REGIME_WEIGHTS": CANDIDATE_REGIME_WEIGHTS,
                "ADX_BUY_MIN": ADX_BUY_MIN,
                "EXIT_PARAMS": EXIT_PARAMS,
                "EXIT_PARAMS_DEFAULT": EXIT_PARAMS_DEFAULT,
            },
        }
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        print(f"[백테스트] 버전 저장: {run_dir}/")
    except Exception as _e:
        print(f"[백테스트] 버전 저장 meta.json 실패 (무시): {_e}")

    sys.stdout.write("\n")
    sys.stdout.flush()
    print(f"[백테스트] 완료: {len(trades)}건 거래")

    return {
        "trades": trades,
        "summary": summary,
        "equity_curve": equity_curve,
        "enhanced_trades_path": enhanced_json_path if trades else None,
    }


def _record_equity(
    equity_points: list,
    positions: list[BtPosition],
    closes: pd.DataFrame,
    date_ts: pd.Timestamp,
    today_str: str,
    cash: float = 0.0,
    use_capital: bool = False,
) -> None:
    """현재 포트폴리오 에쿼티를 기록.

    use_capital=True: 달러 기준 (cash + 보유 종목 시가)
    use_capital=False: 평균 수익률 비율 기준
    """
    if use_capital:
        market_value = sum(
            pos.shares * float(closes.at[date_ts, pos.ticker])
            for pos in positions
            if pos.ticker in closes.columns
            and not np.isnan(float(closes.at[date_ts, pos.ticker]))
        )
        equity_points.append((today_str, cash + market_value))
        return

    # 수익률 비율 모드 (기존 동작)
    if not positions:
        equity_points.append((today_str, 1.0))
        return
    returns = []
    for pos in positions:
        if pos.ticker in closes.columns:
            cur = float(closes.at[date_ts, pos.ticker])
            if not np.isnan(cur) and pos.entry_price > 0:
                returns.append(cur / pos.entry_price)
    if returns:
        equity_points.append((today_str, float(np.mean(returns))))
    else:
        equity_points.append((today_str, 1.0))


def print_summary(summary: dict[str, Any], trades: list[dict]) -> None:
    """백테스트 결과를 터미널에 출력."""
    if not summary:
        print("거래 없음 — 결과 없음")
        return

    start = summary.get("시뮬레이션_시작", "")
    end = summary.get("시뮬레이션_종료", "")
    print(f"\n{'='*55}")
    print(f"  백테스트 결과  ({start} ~ {end})")
    print(f"{'='*55}")
    print(f"  총거래:   {summary.get('총거래수', 0)}건")
    print(f"  승률:     {summary.get('승률', 0):.1%}")
    print(f"  평균수익: {summary.get('평균수익률', 0):+.2%}")
    print(f"  평균승리: {summary.get('평균승리', 0):+.2%}  |  평균손실: {summary.get('평균손실', 0):+.2%}")
    print(f"  승패비율: {summary.get('승패비율', 0):.2f}  |  평균보유: {summary.get('평균보유일', 0):.1f}일")
    print(f"  Sharpe:   {summary.get('Sharpe', 0):.2f}  |  MDD: {summary.get('MDD', 0):.1%}")

    if "초기자본" in summary:
        init = summary["초기자본"]
        final = summary["최종자본"]
        pnl = summary["총수익금"]
        total_ret = summary["총수익률_자본기준"]
        print(f"\n  초기자본:   ${init:,.2f}")
        print(f"  최종자본:   ${final:,.2f}  ({total_ret:+.2%})")
        print(f"  총수익금:   ${pnl:+,.2f}")

    if "SPY수익률" in summary:
        print(f"\n  SPY수익률:  {summary['SPY수익률']:+.2%}")
        print(f"  전략수익률: {summary.get('전략총수익률', 0):+.2%}")
        print(f"  초과수익:   {summary.get('SPY초과수익', 0):+.2%}")

    if "기준_모멘텀_Sharpe" in summary:
        n = summary.get("기준_모멘텀_N", 20)
        print(f"\n  쉬운 방법과 비교 (Sharpe / CAGR / MDD){'  [PIT 유니버스]' if summary.get('PIT_유니버스') else ''}")
        print(f"    이 전략        {summary.get('Sharpe_일간', 0):5.2f} / {summary.get('CAGR', 0):+.1%} / {summary.get('MDD', 0):.1%}")
        print(f"    SPY 보유       {summary['기준_SPY_Sharpe']:5.2f} / {summary['기준_SPY_CAGR']:+.1%} / {summary['기준_SPY_MDD']:.1%}")
        print(f"    모멘텀 상위{n:<3} {summary['기준_모멘텀_Sharpe']:5.2f} / {summary['기준_모멘텀_CAGR']:+.1%} / {summary['기준_모멘텀_MDD']:.1%}")
        if "알파_연" in summary:
            print(f"    알파(시장·모멘텀 제외) {summary['알파_연']:+.1%}/년 (t={summary['알파_t']:.2f}, 2 이상이면 의미)")
        if "모멘텀대비_판정" in summary:
            print(f"    모멘텀 대비 ΔSharpe {summary['모멘텀대비_ΔSharpe']:+.2f} {summary['모멘텀대비_95%']} → {summary['모멘텀대비_판정']}")

    strat_perf = summary.get("전략별", {})
    if strat_perf:
        print(f"\n  전략별 성과:")
        for strat, perf in strat_perf.items():
            print(
                f"    {strat:<12} 승률 {perf['승률']:.1%}  "
                f"평균 {perf['평균수익률']:+.2%}  ({perf['건수']}건)"
            )

    exit_dist = summary.get("매도사유", {})
    if exit_dist:
        # 카테고리별 집계
        cat_order = ["시간익절", "트레일링", "장기보유", "손절", "교체", "목표가", "기간종료"]
        cats: dict[str, int] = {}
        for reason, cnt in exit_dist.items():
            for cat in cat_order:
                if cat in reason:
                    cats[cat] = cats.get(cat, 0) + cnt
                    break
            else:
                cats[reason] = cats.get(reason, 0) + cnt
        total = sum(cats.values())
        print(f"\n  매도사유:")
        for cat in cat_order:
            if cat in cats:
                print(f"    {cat:<10} {cats[cat]}건 ({cats[cat]/total:.1%})")
        for cat, cnt in cats.items():
            if cat not in cat_order:
                print(f"    {cat:<10} {cnt}건 ({cnt/total:.1%})")

    print(f"{'='*55}\n")

    # ── 거래 상세 로그 ────────────────────────────────────
    if trades:
        print(f"  {'#':<3}  {'티커':<6}  {'진입일':<12}  {'청산일':<12}  {'진입가':>8}  {'청산가':>8}  {'수익률':>8}  {'보유':>4}  {'전략':<10}  매도사유")
        print(f"  {'-'*3}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*4}  {'-'*10}  {'-'*20}")
        for i, t in enumerate(trades, 1):
            strat = str(t.get("strategy", "")).replace("📉 ", "").replace("📈 ", "").replace("🔄 ", "")
            ret = t.get("return_pct", 0)
            ret_str = f"{ret:+.1%}"
            win_mark = "✓" if ret > 0 else "✗"
            print(
                f"  {i:<3}  {t['ticker']:<6}  {t['entry_date']:<12}  {t['exit_date']:<12}"
                f"  {t['entry_price']:>8.2f}  {t['exit_price']:>8.2f}"
                f"  {ret_str:>8}  {t['holding_days']:>3}일"
                f"  {strat:<10}  {t['exit_reason']}  {win_mark}"
            )
        print()
