"""성과 비교 통계 (순수 파이썬, pandas 없이 테스트 가능).

- sharpe / cagr / max_drawdown: 일간 수익률 리스트 기준
- alpha_beta: 일간 수익률을 팩터(시장·모멘텀)에 회귀 → 쉬운 방법으로 설명 안 되는 연 알파
- paired_bootstrap: 같은 날짜의 두 전략을 날짜 블록 단위로 같이 재표본 → Sharpe 차이가 운인지
- deflated_sharpe: 여러 번 시도한 뒤 고른 Sharpe가 운으로 기대되는 최고치보다 클 확률 (DSR)
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Sequence

TRADING_DAYS = 252
_EULER_GAMMA = 0.5772156649015329


def sharpe(returns: Sequence[float], periods: int = TRADING_DAYS) -> float:
    """연율화 Sharpe (무위험수익률 0)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return mean / math.sqrt(var) * math.sqrt(periods) if var > 0 else 0.0


def cagr(returns: Sequence[float], periods: int = TRADING_DAYS) -> float:
    """일간 수익률 → 연 복리 수익률."""
    if not returns:
        return 0.0
    growth = 1.0
    for r in returns:
        growth *= 1.0 + r
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(returns)) - 1.0


def max_drawdown(returns: Sequence[float]) -> float:
    """최대 낙폭 (음수, 예: -0.25)."""
    equity, peak, mdd = 1.0, 1.0, 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
    return mdd


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """연립방정식 a·x = b (가우스 소거, 부분 피벗). 특이행렬이면 ValueError."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-15:
            raise ValueError("특이행렬")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def alpha_beta(
    returns: Sequence[float],
    factors: dict[str, Sequence[float]],
    periods: int = TRADING_DAYS,
) -> dict[str, float]:
    """returns = α + Σ β·factor + ε 최소제곱.

    Returns:
        {"알파_연": 연율화 α, "알파_t": α의 t값, "베타_<팩터명>": β ...}. 데이터 부족·특이행렬이면 {}.
    """
    names = list(factors)
    n, k = len(returns), len(names) + 1
    if n <= k + 1 or any(len(factors[f]) != n for f in names):
        return {}
    x = [[1.0] + [float(factors[f][i]) for f in names] for i in range(n)]
    xtx = [[sum(row[p] * row[q] for row in x) for q in range(k)] for p in range(k)]
    xty = [sum(x[i][p] * returns[i] for i in range(n)) for p in range(k)]
    try:
        coef = _solve(xtx, xty)
        inv00 = _solve(xtx, [1.0] + [0.0] * (k - 1))[0]
    except ValueError:
        return {}
    resid = [returns[i] - sum(c * v for c, v in zip(coef, x[i])) for i in range(n)]
    s2 = sum(e * e for e in resid) / (n - k)
    se_alpha = math.sqrt(s2 * inv00) if s2 > 0 and inv00 > 0 else 0.0
    out = {
        "알파_연": coef[0] * periods,
        "알파_t": coef[0] / se_alpha if se_alpha > 0 else 0.0,
    }
    out.update({f"베타_{name}": coef[i + 1] for i, name in enumerate(names)})
    return out


def fold_wins(a: Sequence[float], b: Sequence[float], n_folds: int = 3) -> int:
    """기간을 n_folds 등분했을 때 a의 Sharpe가 b보다 높은 구간 수."""
    n = len(a)
    wins = 0
    for f in range(n_folds):
        lo, hi = n * f // n_folds, n * (f + 1) // n_folds
        if hi - lo >= 2 and sharpe(a[lo:hi]) > sharpe(b[lo:hi]):
            wins += 1
    return wins


def _block_indices(n: int, block: int, rng: random.Random) -> list[int]:
    idx: list[int] = []
    while len(idx) < n:
        start = rng.randrange(n - block + 1) if n > block else 0
        idx.extend(range(start, min(start + block, n)))
    return idx[:n]


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    block: int = 20,
    n_boot: int = 1000,
    seed: int = 42,
    n_folds: int = 3,
) -> dict[str, float]:
    """같은 날짜로 맞춘 두 일간 수익률(a=변형, b=기준)의 Sharpe 차이가 운으로 설명되는지.

    20거래일 블록을 통째로 뽑아 a·b를 같이 재표본한다 (같은 시장 국면을 공유한 채 비교, 자기상관 보존).

    Returns:
        ΔSharpe, 95% 구간(하한·상한), p_개선아님(재표본 중 Δ≤0 비율), 구간_개선(n_folds 중 a가 이긴 구간 수).
        데이터가 블록 2개보다 짧으면 {}.
    """
    n = len(a)
    if n != len(b) or n < 2 * block:
        return {}
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        diffs.append(sharpe([a[i] for i in idx]) - sharpe([b[i] for i in idx]))
    diffs.sort()
    return {
        "ΔSharpe": sharpe(a) - sharpe(b),
        "ΔSharpe_하한": diffs[int(0.025 * n_boot)],
        "ΔSharpe_상한": diffs[min(n_boot - 1, int(0.975 * n_boot))],
        "p_개선아님": sum(1 for d in diffs if d <= 0) / n_boot,
        "구간_개선": fold_wins(a, b, n_folds),
        "구간_수": n_folds,
    }


def verdict(
    stats: dict[str, float],
    mdd_variant: float | None = None,
    mdd_base: float | None = None,
    mdd_tolerance: float = 0.02,
) -> str:
    """채택 후보 = 95% 구간 하한 > 0 + 구간 대부분 개선 + MDD가 tolerance 이상 나빠지지 않음."""
    if not stats:
        return "데이터 부족"
    worse_mdd = mdd_variant is not None and mdd_base is not None and mdd_variant < mdd_base - mdd_tolerance
    if stats["ΔSharpe_하한"] > 0 and stats["구간_개선"] >= stats["구간_수"] - 1 and not worse_mdd:
        return "채택 후보"
    if stats["ΔSharpe_상한"] < 0:
        return "기각"
    return "운과 구분 안 됨"


def deflated_sharpe(sr: float, sr_trials: Sequence[float], n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014). sr·sr_trials 는 기간당(일간) Sharpe.

    시도 N개의 Sharpe 분산으로 "운으로 기대되는 최고 Sharpe" SR0 를 구하고, sr 이 그보다 클 확률을 낸다.
    시도가 2개 미만이거나 관측이 3개 미만이면 nan.
    """
    n = len(sr_trials)
    if n < 2 or n_obs < 3:
        return float("nan")
    mean = sum(sr_trials) / n
    var = sum((s - mean) ** 2 for s in sr_trials) / (n - 1)
    nd = NormalDist()
    sr0 = math.sqrt(var) * ((1 - _EULER_GAMMA) * nd.inv_cdf(1 - 1 / n)
                            + _EULER_GAMMA * nd.inv_cdf(1 - 1 / (n * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return nd.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denom)
