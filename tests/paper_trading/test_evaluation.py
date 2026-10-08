"""성과 비교 통계(evaluation) 테스트 — 순수 파이썬."""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_trading.evaluation import (
    alpha_beta,
    cagr,
    deflated_sharpe,
    fold_wins,
    max_drawdown,
    paired_bootstrap,
    sharpe,
    verdict,
)


def test_basic_metrics():
    assert sharpe([0.01]) == 0.0
    assert sharpe([0.01, 0.01]) == 0.0  # 변동 0
    r = [0.01, -0.005] * 50
    mean = sum(r) / len(r)
    std = math.sqrt(sum((x - mean) ** 2 for x in r) / (len(r) - 1))
    assert math.isclose(sharpe(r), mean / std * math.sqrt(252))
    assert math.isclose(cagr([0.0] * 10), 0.0)
    assert math.isclose(cagr([0.001] * 252), 1.001 ** 252 - 1)
    assert math.isclose(max_drawdown([0.1, -0.5, 0.2]), -0.5)


def test_alpha_beta_recovers_known_coefficients():
    rng = random.Random(1)
    m = [rng.gauss(0.0005, 0.01) for _ in range(500)]
    f = [rng.gauss(0.0, 0.005) for _ in range(500)]
    y = [0.0004 + 1.2 * a + 0.5 * b + rng.gauss(0, 0.001) for a, b in zip(m, f)]
    out = alpha_beta(y, {"시장": m, "모멘텀": f})
    assert abs(out["베타_시장"] - 1.2) < 0.02
    assert abs(out["베타_모멘텀"] - 0.5) < 0.05
    assert abs(out["알파_연"] - 0.0004 * 252) < 0.03
    assert out["알파_t"] > 3


def test_alpha_beta_degenerate_inputs():
    assert alpha_beta([0.01] * 3, {"x": [0.1, 0.2, 0.3]}) == {}  # 표본 부족
    assert alpha_beta([0.01] * 50, {"x": [0.0] * 50}) == {}  # 팩터가 0뿐 → 특이행렬
    assert alpha_beta([0.01] * 50, {"x": [0.0] * 49}) == {}  # 길이 불일치


def _series(seed: int, n: int = 750) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0003, 0.01) for _ in range(n)]


def test_paired_bootstrap_detects_real_improvement():
    base = _series(7)
    rng = random.Random(8)
    better = [b + 0.001 + rng.gauss(0, 0.001) for b in base]  # 매일 +10bp
    s = paired_bootstrap(better, base, n_boot=300)
    assert s["ΔSharpe"] > 0 and s["ΔSharpe_하한"] > 0
    assert s["구간_개선"] == 3
    assert s["p_개선아님"] == 0.0
    assert verdict(s) == "채택 후보"


def test_paired_bootstrap_noise_is_not_improvement():
    base = _series(9)
    rng = random.Random(10)
    eps = [rng.gauss(0, 0.004) for _ in range(len(base) // 2)]
    noise = [e for x in eps for e in (x, -x)]  # 평균이 정확히 0인 잡음
    noisy = [b + e for b, e in zip(base, noise)]
    s = paired_bootstrap(noisy, base, n_boot=300)
    assert s["ΔSharpe_하한"] < 0 < s["ΔSharpe_상한"]
    assert verdict(s) == "운과 구분 안 됨"


def test_paired_bootstrap_rejects_clearly_worse():
    base = _series(11)
    rng = random.Random(12)
    worse = [b - 0.001 + rng.gauss(0, 0.001) for b in base]
    assert verdict(paired_bootstrap(worse, base, n_boot=300)) == "기각"


def test_verdict_guards():
    s = {"ΔSharpe_하한": 0.1, "ΔSharpe_상한": 0.5, "구간_개선": 3, "구간_수": 3}
    assert verdict(s, -0.30, -0.20) == "운과 구분 안 됨"  # MDD 10%p 악화
    assert verdict(s, -0.21, -0.20) == "채택 후보"
    assert verdict({**s, "구간_개선": 1}) == "운과 구분 안 됨"  # 한 구간에서만 좋음
    assert verdict({}) == "데이터 부족"
    assert paired_bootstrap([0.01] * 10, [0.0] * 10) == {}  # 블록 2개보다 짧음


def test_fold_wins_counts_each_period():
    a = [0.01, 0.0] * 30 + [0.0, 0.0] * 30 + [0.01, 0.0] * 30
    b = [0.0, 0.0] * 90
    assert fold_wins(a, b, 3) == 2


def test_deflated_sharpe_penalizes_many_trials():
    rng = random.Random(0)
    one = deflated_sharpe(0.1, [0.1, 0.1], n_obs=756)  # 분산 0 → PSR(0) ≈ 0.997
    many = deflated_sharpe(0.1, [rng.gauss(0.0, 0.05) for _ in range(200)], n_obs=756)
    assert 0.0 <= many < one <= 1.0
    assert math.isnan(deflated_sharpe(0.1, [0.1], n_obs=756))
