"""쉬운 방법과 비교 (같은 기간·같은 비용).

- SPY 그냥 보유
- 12-1 모멘텀 상위 N개 매달 교체: "최근 1년간 많이 오른 종목 N개를 매달 산다"
복잡한 전략이 이 둘을 못 이기면 복잡함이 값을 못 하는 것이다.
"""

from __future__ import annotations

from typing import Callable, Sequence

import pandas as pd

from paper_trading.evaluation import alpha_beta, cagr, max_drawdown, paired_bootstrap, sharpe, verdict

MembersOn = Callable[[str], "frozenset[str]"]


def momentum_topn_returns(
    closes: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    *,
    top_n: int = 20,
    lookback: int = 252,
    skip: int = 21,
    cost_per_side: float = 0.001,
    members_on: MembersOn | None = None,
    exclude: tuple[str, ...] = ("SPY",),
) -> pd.Series:
    """12-1 모멘텀(1년 수익률, 최근 1달 제외) 상위 N개 동일가중의 일간 수익률.

    첫날과 매달 마지막 거래일 종가로 고르고 다음 거래일 종가에 바꾼다 (하루 지연).
    교체 비용 = 회전율 × 편도 비용. members_on이 있으면 그날 구성종목 중에서만 고른다.
    """
    px = closes.drop(columns=[c for c in exclude if c in closes.columns])
    rets = px.pct_change(fill_method=None)
    mom = px.shift(skip) / px.shift(lookback) - 1.0
    dates = list(dates)
    weights = pd.Series(dtype=float)
    pending: pd.Series | None = None
    out: list[float] = []
    for i, d in enumerate(dates):
        r = float((weights * rets.loc[d, weights.index].fillna(0.0)).sum()) if len(weights) else 0.0
        if pending is not None:
            r -= cost_per_side * float(pending.sub(weights, fill_value=0.0).abs().sum())
            weights, pending = pending, None
        out.append(r)
        if i == 0 or (i + 1 < len(dates) and dates[i + 1].month != d.month):
            score = mom.loc[d].dropna()
            if members_on is not None:
                score = score[score.index.isin(members_on(str(pd.Timestamp(d).date())))]
            top = score.nlargest(top_n).index
            if len(top):
                pending = pd.Series(1.0 / len(top), index=top)
    return pd.Series(out, index=pd.DatetimeIndex(dates), name=f"momentum_top{top_n}")


def benchmark_summary(
    equity: pd.Series,
    closes: pd.DataFrame,
    *,
    top_n: int = 20,
    cost_per_side: float = 0.001,
    members_on: MembersOn | None = None,
) -> dict[str, float | str]:
    """전략 에쿼티 vs SPY 보유·모멘텀 상위 N (같은 날짜).

    알파는 전략 일간 수익률을 [시장(SPY), 모멘텀−시장]에 회귀한 절편이다.
    즉 "SPY와 쉬운 모멘텀으로 설명되고 남는 수익". 0 근처면 쉬운 방법을 비싸게 하는 셈.
    """
    strat = equity.pct_change().dropna()
    if len(strat) < 60 or "SPY" not in closes.columns:
        return {}
    if getattr(closes.index, "tz", None) is not None:
        closes = closes.tz_localize(None)
    spy = closes["SPY"].pct_change(fill_method=None).reindex(strat.index).fillna(0.0)
    mom = momentum_topn_returns(
        closes, list(equity.index), top_n=top_n, cost_per_side=cost_per_side, members_on=members_on,
    ).reindex(strat.index).fillna(0.0)

    s, m, p = strat.tolist(), spy.tolist(), mom.tolist()
    out: dict[str, float | str] = {
        "기준_SPY_Sharpe": round(sharpe(m), 2),
        "기준_SPY_CAGR": round(cagr(m), 4),
        "기준_SPY_MDD": round(max_drawdown(m), 4),
        "기준_모멘텀_N": top_n,
        "기준_모멘텀_Sharpe": round(sharpe(p), 2),
        "기준_모멘텀_CAGR": round(cagr(p), 4),
        "기준_모멘텀_MDD": round(max_drawdown(p), 4),
    }
    ab = alpha_beta(s, {"시장": m, "모멘텀": [pi - mi for pi, mi in zip(p, m)]})
    out.update({k: round(v, 4 if k == "알파_연" else 2) for k, v in ab.items()})

    vs = paired_bootstrap(s, p)
    if vs:
        out["모멘텀대비_ΔSharpe"] = round(vs["ΔSharpe"], 2)
        out["모멘텀대비_95%"] = f"[{vs['ΔSharpe_하한']:+.2f}, {vs['ΔSharpe_상한']:+.2f}]"
        out["모멘텀대비_판정"] = verdict(vs, max_drawdown(s), max_drawdown(p))
    return out
