"""개선된 백테스트 모듈.

진입/청산 전략, 리스크 관리, 성과 지표가 강화된 버전.
- Entry Score 시스템: buy_signal + 저점확률 + 반등스코어 + 패턴 + 섹터 통합 평가
- 다중 청산 조건: 수익 목표, 트레일링 스탑, 손절, 고점확률, sell_signal
- 리스크 관리: ATR 기반 포지션 사이징, 일일/종목별 손실 제한
- 성과 지표: MDD, Sharpe Ratio, 벤치마크 비교 등
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Iterable

import numpy as np
import pandas as pd

from screener.config import (
    TICKERS,
    BACKTEST_ENTRY_SCORE_MIN,
    BACKTEST_LOW_PROB_THRESHOLD,
    BACKTEST_REVERSAL_SCORE_MIN,
    BACKTEST_BUY_SIGNAL_WEIGHT,
    BACKTEST_LOW_PROB_WEIGHT,
    BACKTEST_REVERSAL_WEIGHT,
    BACKTEST_PATTERN_WEIGHT,
    BACKTEST_SECTOR_WEIGHT,
    BACKTEST_TREND_SCORE_WEIGHT,
    SECTOR_ETFS,
    SECTOR_ROTATION_ENABLED,
    BACKTEST_PROFIT_TARGET,
    BACKTEST_TRAILING_STOP,
    BACKTEST_STOP_LOSS,
    BACKTEST_HIGH_PROB_THRESHOLD,
    BACKTEST_MIN_HOLDING_DAYS,
    BACKTEST_SELL_SIGNAL_ENABLED,
    BACKTEST_USE_ATR_SIZING,
    DEFAULT_EQUITY,
    RISK_PER_TRADE,
    ATR_POSITION_MULTIPLE,
)
from data.fetch import fetch_ohlcv
from screener.features import compute_features_snapshot
from screener.processing import apply_neutralization, liquidity_filter
from screener.signals import attach_signals_and_sort
from screener.sector_rotation import get_strong_sectors
from screener.config import SECTOR_ROTATION_ENABLED


class ExitReason(Enum):
    """청산 사유 열거형."""
    PROFIT_TARGET = "수익목표달성"
    TRAILING_STOP = "트레일링스탑"
    STOP_LOSS = "손절"
    HIGH_PROB = "고점확률"
    SELL_SIGNAL = "매도신호"
    PATTERN_EXIT = "패턴청산"
    END_OF_PERIOD = "기간종료"


@dataclass
class SignalTrade:
    """거래 기록을 담는 데이터 클래스."""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    position_size: float
    entry_score: float
    exit_reason: str
    holding_days: int
    max_gain: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class Position:
    """활성 포지션 정보."""
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    position_size: float
    entry_score: float
    highest_price: float = field(default=0.0)
    sell_signal_window: Deque[bool] = field(default_factory=lambda: deque(maxlen=5))  # 5일로 확장

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price


def _prepare_price_map(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """티커별로 OHLCV 데이터를 분리."""
    return {ticker: raw[ticker].dropna(how="all") for ticker in raw.columns.levels[0]}


def _compute_ranked_snapshot(
    price_map: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    *,
    include_fundamentals: bool,
    cache: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    """주어진 시점까지의 데이터로 특징 계산 및 시그널 부착."""
    if cutoff in cache:
        return cache[cutoff]

    snapshot: dict[str, pd.DataFrame] = {}
    for ticker, frame in price_map.items():
        history = frame.loc[:cutoff]
        if history.empty:
            continue
        snapshot[ticker] = history

    if not snapshot:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    features = compute_features_snapshot(
        snapshot, include_fundamentals=include_fundamentals
    )
    if features.empty:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    liquid = liquidity_filter(features)
    if liquid.empty:
        cache[cutoff] = pd.DataFrame()
        return cache[cutoff]

    neutral = apply_neutralization(liquid)

    # 섹터 로테이션 필터링 (메인 워크플로우와 동일하게)
    if SECTOR_ROTATION_ENABLED and "섹터" in neutral.columns:
        sector_etf_tickers = list(SECTOR_ETFS.values())
        etf_data = {t: price_map[t].loc[:cutoff] for t in sector_etf_tickers if t in price_map}
        spy_data = price_map.get("SPY", pd.DataFrame()).loc[:cutoff]
        
        if not spy_data.empty:
            strong_sectors = get_strong_sectors(etf_data, spy_data)
            neutral["in_strong_sector"] = neutral["섹터"].isin(strong_sectors)
        else:
            neutral["in_strong_sector"] = True
    else:
        neutral["in_strong_sector"] = True

    ranked = attach_signals_and_sort(neutral)
    cache[cutoff] = ranked
    return ranked


def _lookup(rowset: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """DataFrame을 티커 → 행 딕셔너리로 변환."""
    if rowset.empty or "티커" not in rowset.columns:
        return {}
    index_df = rowset.assign(티커=rowset["티커"].astype(str)).set_index("티커")
    return index_df.to_dict("index")


def _compute_entry_score(row: dict[str, Any]) -> float:
    """
    진입 스코어를 계산한다.
    
    구성 요소:
    - buy_signal: 기본 매수 신호 (가중치 2.0)
    - 저점확률: ML 모델 기반 저점 예측 (가중치 1.5)
    - 반등스코어: 기술적 반등 지표 (가중치 1.5)
    - 상승 패턴: 차트 패턴 (가중치 1.0)
    - 강한 섹터: 섹터 로테이션 (가중치 0.5)
    - 트렌드점수: 높은 모멘텀 (가중치 1.0) - 대체 조건
    - RSI 기반 가점/감점
    - 급등/급락 감점/가점
    """
    score = 0.0
    
    # buy_signal (시장 필터에 의해 False일 수 있음)
    if row.get("buy_signal"):
        score += BACKTEST_BUY_SIGNAL_WEIGHT
    
    # 저점확률
    low_prob = row.get("저점확률", 0)
    if pd.notna(low_prob) and low_prob >= BACKTEST_LOW_PROB_THRESHOLD:
        score += BACKTEST_LOW_PROB_WEIGHT
    
    # 반등스코어
    reversal_score = row.get("반등스코어", 0)
    if pd.notna(reversal_score) and reversal_score >= BACKTEST_REVERSAL_SCORE_MIN:
        score += BACKTEST_REVERSAL_WEIGHT
    
    # 상승 패턴 (더블바텀, 역헤드숄더, 컵핸들, 상승삼각형, 하락쐐기)
    bullish_patterns = [
        row.get("패턴_더블", "") == "더블바텀",
        row.get("패턴_헤드숄더", "") == "역헤드앤숄더",
        row.get("패턴_컵핸들", False),
        row.get("패턴_삼각형", "") == "상승삼각형",
        row.get("패턴_쐐기", "") == "하락쐐기",
        str(row.get("패턴_캔들", "")) in ["강세잉걸핑", "모닝스타"],
    ]
    if any(bullish_patterns):
        score += BACKTEST_PATTERN_WEIGHT
    
    # 강한 섹터
    if row.get("in_strong_sector", True):
        score += BACKTEST_SECTOR_WEIGHT
    
    # 트렌드점수 (buy_signal이 비활성화된 경우 대체 조건)
    trend_score = row.get("트렌드점수_최종", row.get("트렌드점수", 0))
    if pd.notna(trend_score) and trend_score > 0.1:  # 상위 모멘텀
        score += BACKTEST_TREND_SCORE_WEIGHT
    
    # RSI 기반 점수 (과매도 가점 / 과매수 감점)
    rsi = row.get("RSI", 50)
    if pd.notna(rsi):
        if rsi < 30:
            score += 1.5    # 과매도 - 반등 기회
        elif rsi < 40:
            score += 1.0    # 과매도 탈출 구간
        elif rsi <= 45:
            score += 0.5    # 중립 하단
        elif rsi > 80:
            score -= 2.0    # 극심한 과매수 - 위험
        elif rsi > 70:
            score -= 1.0    # 과매수 - 주의 (완화)
    
    # EMA 정배열 확인 (추가 가점)
    ema20 = row.get("ema20", 0)
    ema50 = row.get("ema50", 0)
    if pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50:
        score += 0.5
    
    # 급등 감점 (5일 수익률 > 20%) - 기준 완화
    ret_5d = row.get("5일수익률", 0)
    if pd.notna(ret_5d) and ret_5d > 0.20:
        score -= 1.0    # 급등 후 조정 위험
    
    # 볼린저밴드 상단 돌파 감점
    bollinger_pband = row.get("bollinger_pband", 0.5)
    if pd.notna(bollinger_pband) and bollinger_pband > 0.95:
        score -= 1.0    # 상단 밴드 돌파 - 과열
    
    # 급락 가점 (5일 수익률 < -8%) - 기준 완화
    if pd.notna(ret_5d) and ret_5d < -0.08:
        score += 0.5    # 급락 후 반등 기회
    
    return score


def _check_exit_conditions(
    position: Position,
    current_price: float,
    row: dict[str, Any],
    current_date: pd.Timestamp,
) -> tuple[bool, ExitReason | None]:
    """
    청산 조건을 체크한다.
    
    Returns:
        (should_exit, exit_reason)
    """
    entry_price = position.entry_price
    current_return = (current_price / entry_price) - 1.0 if entry_price > 0 else 0
    
    # 최소 보유일 체크 (단기 변동 무시)
    holding_days = (current_date - position.entry_date).days
    if holding_days < BACKTEST_MIN_HOLDING_DAYS:
        # 최소 보유일 이전에는 손절만 허용 (큰 손실 방지)
        if current_return <= -BACKTEST_STOP_LOSS:
            return True, ExitReason.STOP_LOSS
        return False, None
    
    # 1. 수익 목표 달성
    if current_return >= BACKTEST_PROFIT_TARGET:
        return True, ExitReason.PROFIT_TARGET
    
    # 2. 손절
    if current_return <= -BACKTEST_STOP_LOSS:
        return True, ExitReason.STOP_LOSS
    
    # 3. 트레일링 스탑 (고점 대비 하락)
    if current_price > position.highest_price:
        position.highest_price = current_price
    
    drawdown_from_high = (current_price / position.highest_price) - 1.0
    if drawdown_from_high <= -BACKTEST_TRAILING_STOP:
        return True, ExitReason.TRAILING_STOP
    
    # 4. 고점확률 청산
    high_prob = row.get("고점확률", 0)
    if pd.notna(high_prob) and high_prob >= BACKTEST_HIGH_PROB_THRESHOLD:
        return True, ExitReason.HIGH_PROB
    
    # 5. 하락 패턴 감지
    bearish_patterns = [
        row.get("패턴_더블", "") == "더블탑",
        row.get("패턴_헤드숄더", "") == "헤드앤숄더",
        row.get("패턴_삼각형", "") == "하락삼각형",
        row.get("패턴_쐐기", "") == "상승쐐기",
        row.get("패턴_캔들", "") in ["약세잉걸핑", "이브닝스타"],
    ]
    if any(bearish_patterns):
        return True, ExitReason.PATTERN_EXIT
    
    # 6. 매도 신호 (5일 중 3회 - 설정으로 비활성화 가능)
    if BACKTEST_SELL_SIGNAL_ENABLED:
        sell_signal = bool(row.get("sell_signal", False))
        position.sell_signal_window.append(sell_signal)
        if sum(position.sell_signal_window) >= 3:
            return True, ExitReason.SELL_SIGNAL
    
    return False, None


def _calculate_position_size(
    row: dict[str, Any],
    equity: float,
    current_price: float,
) -> float:
    """ATR 기반 포지션 사이징 또는 균등 배분."""
    if not BACKTEST_USE_ATR_SIZING:
        return 1.0
    
    atr_value = row.get("atr_value", np.nan)
    if pd.isna(atr_value) or atr_value <= 0:
        return 1.0
    
    # 위험 금액 / (ATR * 배수) = 주식 수
    risk_amount = equity * RISK_PER_TRADE
    stop_distance = atr_value * ATR_POSITION_MULTIPLE
    shares = risk_amount / stop_distance if stop_distance > 0 else 0
    
    # 포지션 가치 비율로 정규화
    position_value = shares * current_price
    return min(position_value / equity, 0.2)  # 최대 20% 단일 포지션


def _calculate_metrics(
    trades_df: pd.DataFrame,
    spy_returns: pd.Series | None = None,
) -> dict[str, Any]:
    """성과 지표를 계산한다."""
    if trades_df.empty:
        return {}
    
    returns = trades_df["return_pct"]
    position_sizes = trades_df["position_size"]
    
    # 기본 지표
    total_trades = len(trades_df)
    wins = trades_df[returns > 0]
    losses = trades_df[returns <= 0]
    win_rate = len(wins) / total_trades if total_trades else 0
    
    # 수익률 지표
    avg_return = returns.mean()
    weighted_return = (returns * position_sizes).sum() / position_sizes.sum() if position_sizes.sum() > 0 else 0
    median_return = returns.median()
    
    # 승패 비율
    avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["return_pct"].mean()) if len(losses) > 0 else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # 평균 보유일수
    avg_holding_days = trades_df["holding_days"].mean() if "holding_days" in trades_df.columns else 0
    
    # MDD 계산 (누적 수익률 기반)
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (연율화, 무위험 수익률 0 가정)
    if len(returns) > 1 and returns.std() > 0:
        # 평균 보유일수 기반 연율화
        trades_per_year = 252 / max(avg_holding_days, 1) if avg_holding_days > 0 else 12
        sharpe = (avg_return * trades_per_year) / (returns.std() * np.sqrt(trades_per_year))
    else:
        sharpe = 0
    
    # 청산 사유 분석
    exit_reasons = trades_df["exit_reason"].value_counts().to_dict() if "exit_reason" in trades_df.columns else {}
    
    summary = {
        "총거래수": int(total_trades),
        "승률": round(win_rate, 4),
        "평균수익률": round(avg_return, 6),
        "가중평균수익률": round(weighted_return, 6),
        "중앙값수익률": round(median_return, 6),
        "평균승리": round(avg_win, 6),
        "평균손실": round(-avg_loss, 6),
        "승패비율": round(win_loss_ratio, 2),
        "평균보유일수": round(avg_holding_days, 1),
        "최대낙폭(MDD)": round(max_drawdown, 4),
        "Sharpe Ratio": round(sharpe, 2),
    }
    
    # 청산 사유 추가
    for reason, count in exit_reasons.items():
        summary[f"청산_{reason}"] = int(count)
    
    # 벤치마크 비교 (SPY)
    if spy_returns is not None and len(spy_returns) > 0:
        spy_total_return = (1 + spy_returns).prod() - 1
        strategy_total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        summary["SPY수익률"] = round(spy_total_return, 4)
        summary["전략총수익률"] = round(strategy_total_return, 4)
        summary["초과수익"] = round(strategy_total_return - spy_total_return, 4)
    
    return summary


def run_backtest(
    tickers: Iterable[str] | None = None,
    *,
    period: str = "1y",
    include_fundamentals: bool = True,
    max_positions: int = 5,
    max_tickers: int | None = 100,
    min_history_days: int = 220,
    rebalance_every: int = 20,
) -> dict[str, Any]:
    """
    개선된 백테스트를 실행한다.
    
    진입: Entry Score 시스템 (buy_signal + 저점확률 + 반등스코어 + 패턴 + 섹터)
    청산: 다중 조건 (수익목표, 트레일링스탑, 손절, 고점확률, 패턴, 매도신호)
    """
    if tickers is None:
        tickers = TICKERS.copy()
    else:
        tickers = list(dict.fromkeys(tickers))

    # 섹터 ETF 및 SPY 추가
    needed = ["SPY"]
    if SECTOR_ROTATION_ENABLED:
        needed.extend(list(SECTOR_ETFS.values()))
    
    for t in needed:
        if t not in tickers:
            tickers.append(t)

    if max_tickers is not None:
        # 주요 지수/ETF는 제외하고 개수 제한
        base_tickers = [t for t in tickers if t not in needed]
        tickers = needed + base_tickers[:max_tickers]

    raw = fetch_ohlcv(tickers, period=period)
    if raw.empty:
        raise RuntimeError("OHLCV 데이터를 가져오지 못했습니다.")

    closes = raw.xs("Close", level=1, axis=1)
    opens = raw.xs("Open", level=1, axis=1)
    dates = closes.index.to_list()
    if len(dates) <= min_history_days + 5:
        raise ValueError("분석 기간이 너무 짧습니다.")

    # SPY 수익률 (벤치마크)
    spy_closes = closes.get("SPY")
    spy_returns = spy_closes.pct_change().dropna() if spy_closes is not None else None

    price_map = _prepare_price_map(raw)
    ranked_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    start_index = min_history_days
    trades: list[SignalTrade] = []
    active_positions: dict[str, Position] = {}
    portfolio_equity = DEFAULT_EQUITY

    for idx in range(start_index, len(dates) - 1):
        date = dates[idx]
        next_date = dates[idx + 1]

        ranked = _compute_ranked_snapshot(
            price_map, date, include_fundamentals=include_fundamentals, cache=ranked_cache
        )
        lookup = _lookup(ranked)

        # ===== 1) 기존 포지션 청산 여부 확인 =====
        next_active: dict[str, Position] = {}
        for ticker, position in active_positions.items():
            if ticker not in closes.columns:
                continue
                
            current_price = float(closes.at[date, ticker])
            if np.isnan(current_price):
                next_active[ticker] = position
                continue
            
            meta = lookup.get(ticker, {})
            should_exit, exit_reason = _check_exit_conditions(position, current_price, meta, date)
            
            if should_exit and exit_reason:
                exit_price = float(opens.at[next_date, ticker])
                entry_price = position.entry_price
                
                if np.isnan(exit_price) or np.isnan(entry_price) or entry_price == 0:
                    next_active[ticker] = position
                    continue
                
                return_pct = exit_price / entry_price - 1.0
                holding_days = (next_date - position.entry_date).days
                max_gain = (position.highest_price / entry_price) - 1.0
                max_dd = (current_price / position.highest_price) - 1.0 if position.highest_price > 0 else 0
                
                trades.append(
                    SignalTrade(
                        ticker=ticker,
                        entry_date=position.entry_date,
                        exit_date=next_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=return_pct,
                        position_size=position.position_size,
                        entry_score=position.entry_score,
                        exit_reason=exit_reason.value,
                        holding_days=holding_days,
                        max_gain=max_gain,
                        max_drawdown=max_dd,
                    )
                )
            else:
                next_active[ticker] = position
        
        active_positions = next_active

        # ===== 2) 신규 진입 – 리밸런스 주기마다 =====
        if (idx - start_index) % max(1, rebalance_every) != 0:
            continue

        # Entry Score가 높은 종목 필터링
        candidates = []
        for _, row in ranked.iterrows():
            row_dict = row.to_dict()
            entry_score = _compute_entry_score(row_dict)
            if entry_score >= BACKTEST_ENTRY_SCORE_MIN:
                candidates.append((entry_score, row_dict))
        
        # Entry Score 내림차순 정렬
        candidates.sort(key=lambda x: x[0], reverse=True)

        available_slots = max_positions - len(active_positions)
        if available_slots <= 0:
            continue

        for entry_score, row_dict in candidates:
            ticker = str(row_dict.get("티커", ""))
            if not ticker or ticker in active_positions or ticker == "SPY":
                continue

            if ticker not in opens.columns:
                continue
                
            entry_price = float(opens.at[next_date, ticker])
            if np.isnan(entry_price) or entry_price == 0:
                continue

            position_size = _calculate_position_size(row_dict, portfolio_equity, entry_price)

            active_positions[ticker] = Position(
                ticker=ticker,
                entry_date=next_date,
                entry_price=entry_price,
                position_size=position_size,
                entry_score=entry_score,
                highest_price=entry_price,
            )

            available_slots -= 1
            if available_slots <= 0:
                break

    # ===== 잔여 포지션 강제 청산 =====
    final_date = dates[-1]
    for ticker, position in active_positions.items():
        if ticker not in closes.columns:
            continue
            
        exit_price = float(closes.at[final_date, ticker])
        entry_price = position.entry_price
        if np.isnan(exit_price) or np.isnan(entry_price) or entry_price == 0:
            continue
        
        return_pct = exit_price / entry_price - 1.0
        holding_days = (final_date - position.entry_date).days
        max_gain = (position.highest_price / entry_price) - 1.0
        
        trades.append(
            SignalTrade(
                ticker=ticker,
                entry_date=position.entry_date,
                exit_date=final_date,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=return_pct,
                position_size=position.position_size,
                entry_score=position.entry_score,
                exit_reason=ExitReason.END_OF_PERIOD.value,
                holding_days=holding_days,
                max_gain=max_gain,
                max_drawdown=0.0,
            )
        )

    if not trades:
        return {"trades": pd.DataFrame(), "summary": {}}

    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
    trades_df.sort_values("exit_date", inplace=True)

    # 성과 지표 계산
    summary = _calculate_metrics(trades_df, spy_returns)

    # 컬럼명 한글화
    trades_df = trades_df.rename(
        columns={
            "ticker": "티커",
            "entry_date": "진입일",
            "exit_date": "청산일",
            "entry_price": "진입가",
            "exit_price": "청산가",
            "return_pct": "수익률",
            "position_size": "포지션사이즈",
            "entry_score": "진입스코어",
            "exit_reason": "청산사유",
            "holding_days": "보유일수",
            "max_gain": "최대상승",
            "max_drawdown": "최대하락",
        }
    )

    # 수익률 퍼센트 변환
    for col in ["수익률", "최대상승", "최대하락"]:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].map(lambda x: round(x * 100, 2))
    
    if "포지션사이즈" in trades_df.columns:
        trades_df["포지션사이즈"] = trades_df["포지션사이즈"].map(lambda x: round(x, 4))

    return {"trades": trades_df, "summary": summary}
