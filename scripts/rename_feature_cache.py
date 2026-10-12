"""10년 피처 캐시 이름을 지금 코드의 캐시 키로 맞춘다 (계산 결과는 그대로, 이름만 바꾼다).

피처 캐시 키는 피처 계산 코드(config.py 포함)의 지문으로 만든다. 피처 계산과 상관없는 줄을 고쳐도
키가 바뀌어 백테스트가 캐시를 못 찾고 몇 시간짜리 재계산을 하게 되는데, 그럴 때 이 스크립트로 이름만 맞춘다.
피처 계산을 실제로 바꾼 경우(지표·패턴 로직 수정)에는 쓰면 안 된다 — 그때는 새로 계산해야 맞다.

사용법 (프로젝트 루트에서):
    PYTHONPATH=.:src .venv/bin/python scripts/rename_feature_cache.py          # 확인만 (아무것도 안 바꿈)
    PYTHONPATH=.:src .venv/bin/python scripts/rename_feature_cache.py --apply  # 실제로 이름 바꾸기
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import paper_trading.backtest as bt  # noqa: E402
import screener.config as cfg  # noqa: E402
from paper_trading.universe import Membership, load_membership  # noqa: E402
from screener.cache import FEATURE_VERSION  # noqa: E402

FEATURES = ROOT / "data" / "cache" / "features"
IC_WEIGHTS = ROOT / "data" / "cache" / "ic_weights"


def current_key(period: str) -> str:
    """run_paper_trading_backtest 가 PIT 유니버스로 만드는 캐시 키와 같은 값."""
    membership = Membership(load_membership(ROOT / cfg.SP500_MEMBERSHIP_PATH))
    tickers = bt._pit_universe_tickers(membership, period)
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers
    key = ",".join(sorted(tickers)) + "|" + period + f"|fund=0|code={bt._code_hash()}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def caches_for(period: str) -> list[tuple[str, int]]:
    """같은 기간의 기존 캐시 (키, 스냅샷 개수) — 스냅샷이 많은 순."""
    found = []
    for meta_path in FEATURES.glob(f"*_{FEATURE_VERSION}/meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("period") == period and not meta.get("include_fundamentals"):
            found.append((meta["ohlcv_hash"], len(list(meta_path.parent.glob("????-??-??.parquet")))))
    return sorted(found, key=lambda kv: -kv[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", default="13y", help="백테스트 기간 (10년 캐시는 13y)")
    parser.add_argument("--from-key", help="바꿀 기존 캐시 키 (안 주면 스냅샷이 가장 많은 캐시)")
    parser.add_argument("--apply", action="store_true", help="실제로 이름을 바꾼다 (없으면 확인만)")
    args = parser.parse_args()

    new = current_key(args.period)
    existing = caches_for(args.period)
    print(f"지금 코드의 캐시 키: {new}")
    if any(key == new for key, _ in existing):
        print("이미 맞는 이름의 캐시가 있습니다 — 할 일 없음.")
        return
    if not existing:
        print(f"{args.period} 캐시가 없습니다 — 백테스트가 처음부터 계산합니다.")
        return
    old = args.from_key or existing[0][0]
    snapshots = dict(existing).get(old)
    if snapshots is None:
        sys.exit(f"{old} 캐시를 찾을 수 없습니다. 있는 캐시: {existing}")
    ic_files = sorted(IC_WEIGHTS.glob(f"{old}_*.json"))
    print(f"바꿀 캐시: {old} (스냅샷 {snapshots}개, IC 가중치 {len(ic_files)}개) → {new}")
    if not args.apply:
        print("확인만 했습니다. 바꾸려면 --apply 를 붙여 다시 실행하세요.")
        return

    (FEATURES / f"{old}_{FEATURE_VERSION}").rename(FEATURES / f"{new}_{FEATURE_VERSION}")
    meta_path = FEATURES / f"{new}_{FEATURE_VERSION}" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["ohlcv_hash"] = new
    meta_path.write_text(json.dumps(meta, indent=2))
    for f in ic_files:
        f.rename(IC_WEIGHTS / f"{new}_{f.name[len(old) + 1:]}")
    print(f"완료: 캐시 폴더·meta.json·IC 가중치 {len(ic_files)}개 이름을 {new} 로 바꿨습니다.")


if __name__ == "__main__":
    main()
