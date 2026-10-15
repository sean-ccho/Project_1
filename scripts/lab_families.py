#!/usr/bin/env python3
"""연구소(lab) 전략군 A~E — 목표 비중 규칙, 탐색 격자, 이웃 정의 (결과 보기 전에 고정).

모든 신호는 t 종가까지의 수익률만 쓴다. 자산은 수익 기록이 ELIGIBLE_DAYS 일 이상 쌓인 뒤부터 쓴다.
모멘텀은 같은 기간 무위험 수익을 뺀 초과 모멘텀이다.

- A 지수 위험관리 (자산 1개 "EQ"): 추세 필터 × 변동성 목표
- B 자산배분 추세추종: 상승 추세인 자산만, 동일/역변동성 비중, 나머지 현금
- C 상대·듀얼 모멘텀: 위험자산 상위 k 개, 초과 모멘텀 ≤ 0 이면 그 몫은 대피 자산
- D 위험 균등 정적 배분: 동일/역변동성 고정 배분
- E 달력 효과 (자산 1개 "EQ"): 월말 k 거래일 + 월초 j 거래일만 보유
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab_backtest import SCHEDULE_KO, SCHEDULE_ORDER, TRADING_DAYS  # noqa: E402

ELIGIBLE_DAYS = 253          # 12개월 모멘텀 + 1일
MENUS: dict[str, list[str]] = {
    "G4": ["US", "DEV", "LTT", "ITT"],
    "G8": ["US", "DEV", "EM", "LTT", "ITT", "REIT", "GOLD", "CMDTY"],
    "SB2": ["US", "ITT"],
    "AW": ["US", "LTT", "ITT", "GOLD", "CMDTY"],
    "R3": ["US", "DEV", "EM"],
    "R6": ["US", "DEV", "EM", "REIT", "GOLD", "CMDTY"],
}
MENU_KO = {"G4": "4자산(미국·선진국·장기채·중기채)", "G8": "8자산(+신흥국·리츠·금·원자재)", "SB2": "주식+중기채",
           "AW": "올웨더형(미국·장기채·중기채·금·원자재)", "R3": "주식 3(미국·선진국·신흥국)",
           "R6": "위험자산 6(+리츠·금·원자재)"}
BLEND_LOOKBACKS = (21, 63, 126, 252)


# ── 자료 묶음 ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class Universe:
    """날짜 × 자산 일간 총수익(생기기 전 NaN)과 무위험 수익. 신호 계산 결과는 안에 캐시한다."""

    ret: pd.DataFrame
    rf: pd.Series
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.rf = self.rf.reindex(self.ret.index).fillna(0.0)
        valid = self.ret.notna()
        self.tri = (1.0 + self.ret.fillna(0.0)).cumprod().where(valid.cummax())
        self.eligible = valid.cumsum() >= ELIGIBLE_DAYS
        self.rfi = (1.0 + self.rf).cumprod()

    def truncate(self, end: pd.Timestamp) -> "Universe":
        return Universe(self.ret.loc[:end], self.rf.loc[:end])

    def _memo(self, key: tuple, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def mom(self, m: int) -> pd.DataFrame:
        """m 거래일 초과 모멘텀 (총수익 − 무위험)."""
        return self._memo(("mom", m), lambda: (self.tri / self.tri.shift(m) - 1.0).sub(
            self.rfi / self.rfi.shift(m) - 1.0, axis=0))

    def sma(self, n: int) -> pd.DataFrame:
        return self._memo(("sma", n), lambda: self.tri.rolling(n).mean())

    def vol(self, n: int) -> pd.DataFrame:
        """최근 n 일 일간 수익 표준편차 × √252."""
        return self._memo(("vol", n), lambda: self.ret.rolling(n).std() * np.sqrt(TRADING_DAYS))


def _invalid_rows(w: pd.DataFrame, ok: pd.Series) -> pd.DataFrame:
    """ok 가 아닌 날은 결정하지 않는다 (행 전체 NaN)."""
    out = w.copy()
    out.loc[~ok.reindex(out.index, fill_value=False).astype(bool)] = np.nan
    return out


def base_weights(u: Universe, menu: list[str], wkind: str, wl: int) -> pd.DataFrame:
    """쓸 수 있는 메뉴 자산에 동일(EW) 또는 역변동성(IV, 최근 wl 일) 비중. 쓸 자산이 없으면 그날 NaN."""
    elig = u.eligible[menu]
    if wkind == "EW":
        raw = elig.astype(float)
    else:
        raw = (1.0 / u.vol(wl)[menu]).where(elig, 0.0).fillna(0.0)
    tot = raw.sum(axis=1)
    return raw.div(tot.where(tot > 0), axis=0)


# ── 전략군 정의 ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Family:
    """전략군 하나: 격자, 이웃 사다리, 목표 비중, 설명."""

    code: str
    title: str
    kind: str                                   # "index" (자산 EQ 하나) | "multi" (여러 자산)
    grid: Callable[[], list[dict[str, Any]]]
    ladders: Callable[[dict[str, Any]], dict[str, list[Any]]]
    target: Callable[[Universe, dict[str, Any]], pd.DataFrame]
    describe: Callable[[dict[str, Any]], str]

    def key(self, cfg: dict[str, Any]) -> str:
        return f"{self.code}:" + ",".join(f"{k}={v}" for k, v in cfg.items())

    def neighbors(self, cfg: dict[str, Any], keys: set[str]) -> list[str]:
        """한 손잡이만 사다리에서 한 칸 옮긴 조합 (격자에 있는 것만)."""
        out = []
        for p, ladder in self.ladders(cfg).items():
            if cfg[p] not in ladder:
                continue
            i = ladder.index(cfg[p])
            for j in (i - 1, i + 1):
                if 0 <= j < len(ladder):
                    k = self.key({**cfg, p: ladder[j]})
                    if k in keys:
                        out.append(k)
        return out


def schedule_of(cfg: dict[str, Any]) -> str:
    """조합의 매매 주기 (달력 효과 E 는 주기 손잡이가 없어 매일·폭 없음)."""
    return str(cfg.get("sched", "D0"))


# A ─ 지수 위험관리 ────────────────────────────────────────────────────────────────────────────

A_TREND_N = {"none": [0], "sma": [50, 100, 150, 200, 250], "mom": [63, 126, 189, 252], "xma": ["20/100", "50/200"]}
A_SIGMA = [0.10, 0.12, 0.15, 0.18, 0.20]
A_VOL_L = [21, 63, 126]


def grid_a() -> list[dict[str, Any]]:
    out = []
    for kind, ns in A_TREND_N.items():
        for n in ns:
            for sig, vl in [(0.0, 0)] + [(s, l) for s in A_SIGMA for l in A_VOL_L]:
                for sched in SCHEDULE_ORDER:
                    out.append({"trend": kind, "n": n, "sigma": sig, "vl": vl, "sched": sched})
    return out


def ladders_a(cfg: dict[str, Any]) -> dict[str, list[Any]]:
    lad: dict[str, list[Any]] = {"n": A_TREND_N[cfg["trend"]], "sched": SCHEDULE_ORDER}
    if cfg["sigma"] > 0:
        lad["sigma"], lad["vl"] = A_SIGMA, A_VOL_L
    return lad


def trend_signal(u: Universe, col: str, kind: str, n: Any) -> pd.Series:
    """추세 신호 1/0: sma = 총수익지수 > n일 평균, mom = n일 초과 모멘텀 > 0, xma = 짧은 평균 > 긴 평균."""
    if kind == "none":
        return pd.Series(1.0, index=u.ret.index)
    if kind == "sma":
        return (u.tri[col] > u.sma(int(n))[col]).astype(float)
    if kind == "mom":
        return (u.mom(int(n))[col] > 0).astype(float)
    f, s = (int(x) for x in str(n).split("/"))
    return (u.sma(f)[col] > u.sma(s)[col]).astype(float)


def target_a(u: Universe, cfg: dict[str, Any]) -> pd.DataFrame:
    w = trend_signal(u, "EQ", cfg["trend"], cfg["n"])
    if cfg["sigma"] > 0:
        w = w * np.minimum(1.0, cfg["sigma"] / u.vol(int(cfg["vl"]))["EQ"])
    return _invalid_rows(w.to_frame("EQ"), u.eligible["EQ"] & w.notna())


def describe_a(cfg: dict[str, Any]) -> str:
    t = {"none": "추세 필터 없음", "sma": f"총수익지수 > {cfg['n']}일 평균일 때만 보유",
         "mom": f"{cfg['n']}일 초과 모멘텀 > 0 일 때만 보유", "xma": f"{cfg['n']}일 이평 골든크로스 동안 보유"}[cfg["trend"]]
    v = "변동성 목표 없음" if cfg["sigma"] == 0 else f"비중 = min(1, {cfg['sigma']:.0%} ÷ 최근 {cfg['vl']}일 변동성)"
    return f"{t} · {v} · {SCHEDULE_KO[cfg['sched']]}"


# B ─ 자산배분 추세추종 ────────────────────────────────────────────────────────────────────────

B_SIG_N = {"mom": [63, 126, 189, 252], "sma": [100, 150, 200], "blend": [0]}
IV_L = [21, 63, 126]


def grid_b() -> list[dict[str, Any]]:
    out = []
    for menu in ("G4", "G8"):
        for sk, ns in B_SIG_N.items():
            for n in ns:
                for wk, wl in [("EW", 0)] + [("IV", l) for l in IV_L]:
                    for sched in SCHEDULE_ORDER:
                        out.append({"menu": menu, "sig": sk, "n": n, "wk": wk, "wl": wl, "sched": sched})
    return out


def ladders_b(cfg: dict[str, Any]) -> dict[str, list[Any]]:
    lad: dict[str, list[Any]] = {"n": B_SIG_N[cfg["sig"]], "sched": SCHEDULE_ORDER}
    if cfg["wk"] == "IV":
        lad["wl"] = IV_L
    return lad


def asset_signal(u: Universe, menu: list[str], kind: str, n: int) -> pd.DataFrame:
    """자산별 추세 신호 0~1: mom = n일 초과 모멘텀 > 0, sma = 총수익지수 > n일 평균, blend = 1·3·6·12개월 중 플러스 비율."""
    if kind == "mom":
        return (u.mom(n)[menu] > 0).astype(float)
    if kind == "sma":
        return (u.tri[menu] > u.sma(n)[menu]).astype(float)
    return sum((u.mom(m)[menu] > 0).astype(float) for m in BLEND_LOOKBACKS) / len(BLEND_LOOKBACKS)


def target_b(u: Universe, cfg: dict[str, Any]) -> pd.DataFrame:
    menu = MENUS[cfg["menu"]]
    base = base_weights(u, menu, cfg["wk"], int(cfg["wl"]))
    w = base * asset_signal(u, menu, cfg["sig"], int(cfg["n"]))
    return _invalid_rows(w.fillna(0.0), base.notna().any(axis=1))


def describe_b(cfg: dict[str, Any]) -> str:
    s = {"mom": f"{cfg['n']}일 초과 모멘텀 > 0", "sma": f"총수익지수 > {cfg['n']}일 평균",
         "blend": "1·3·6·12개월 초과 모멘텀 중 플러스 비율만큼"}[cfg["sig"]]
    wt = "동일 비중" if cfg["wk"] == "EW" else f"역변동성 비중({cfg['wl']}일)"
    return f"{MENU_KO[cfg['menu']]} 중 {s}인 자산만 · {wt} · 나머지 현금 · {SCHEDULE_KO[cfg['sched']]}"


# C ─ 상대·듀얼 모멘텀 ─────────────────────────────────────────────────────────────────────────

C_SCORE_N = {"mom": [63, 126, 252], "w13612": [0]}
C_K = [1, 2, 3]


def grid_c() -> list[dict[str, Any]]:
    out = []
    for menu in ("R3", "R6"):
        for safe in ("CASH", "ITT", "LTT"):
            for sk, ns in C_SCORE_N.items():
                for n in ns:
                    for k in C_K:
                        for sched in SCHEDULE_ORDER:
                            out.append({"menu": menu, "safe": safe, "score": sk, "n": n, "k": k, "sched": sched})
    return out


def ladders_c(cfg: dict[str, Any]) -> dict[str, list[Any]]:
    return {"n": C_SCORE_N[cfg["score"]], "k": C_K, "sched": SCHEDULE_ORDER}


def momentum_score(u: Universe, menu: list[str], kind: str, n: int) -> pd.DataFrame:
    """mom = n일 초과 모멘텀, w13612 = 12×1개월 + 4×3개월 + 2×6개월 + 1×12개월 초과 모멘텀 (VAA 방식)."""
    if kind == "mom":
        return u.mom(n)[menu]
    return 12 * u.mom(21)[menu] + 4 * u.mom(63)[menu] + 2 * u.mom(126)[menu] + u.mom(252)[menu]


def target_c(u: Universe, cfg: dict[str, Any]) -> pd.DataFrame:
    menu, k, safe = MENUS[cfg["menu"]], int(cfg["k"]), cfg["safe"]
    sc = momentum_score(u, menu, cfg["score"], int(cfg["n"])).where(u.eligible[menu])
    rank = sc.rank(axis=1, ascending=False, method="first")
    pick = (rank <= k) & (sc > 0)
    w = pick.astype(float) / k
    cols = menu + ([safe] if safe != "CASH" else [])
    out = w.reindex(columns=cols, fill_value=0.0)
    if safe != "CASH":
        rest = 1.0 - w.sum(axis=1)
        out[safe] = rest.where(u.eligible[safe], 0.0)
    ok = u.eligible[menu].any(axis=1)
    return _invalid_rows(out, ok)


def describe_c(cfg: dict[str, Any]) -> str:
    s = f"{cfg['n']}일 초과 모멘텀" if cfg["score"] == "mom" else "1·3·6·12개월 가중 모멘텀(12·4·2·1)"
    safe = {"CASH": "현금", "ITT": "중기국채", "LTT": "장기국채"}[cfg["safe"]]
    return (f"{MENU_KO[cfg['menu']]} 중 {s} 상위 {cfg['k']}개 · 모멘텀 ≤ 0 이면 그 몫은 {safe} · "
            f"{SCHEDULE_KO[cfg['sched']]}")


# D ─ 위험 균등 정적 배분 ──────────────────────────────────────────────────────────────────────

D_IV_L = [21, 63, 126, 252]


def grid_d() -> list[dict[str, Any]]:
    return [{"menu": m, "wk": wk, "wl": wl, "sched": s} for m in ("G4", "G8", "SB2", "AW")
            for wk, wl in [("EW", 0)] + [("IV", l) for l in D_IV_L] for s in SCHEDULE_ORDER]


def ladders_d(cfg: dict[str, Any]) -> dict[str, list[Any]]:
    lad: dict[str, list[Any]] = {"sched": SCHEDULE_ORDER}
    if cfg["wk"] == "IV":
        lad["wl"] = D_IV_L
    return lad


def target_d(u: Universe, cfg: dict[str, Any]) -> pd.DataFrame:
    base = base_weights(u, MENUS[cfg["menu"]], cfg["wk"], int(cfg["wl"]))
    return _invalid_rows(base.fillna(0.0), base.notna().any(axis=1))


def describe_d(cfg: dict[str, Any]) -> str:
    wt = "동일 비중" if cfg["wk"] == "EW" else f"역변동성 비중({cfg['wl']}일)"
    return f"{MENU_KO[cfg['menu']]} 항상 보유 · {wt} · {SCHEDULE_KO[cfg['sched']]}"


# E ─ 달력 효과 ──────────────────────────────────────────────────────────────────────────────

E_K = [1, 2, 3, 4, 5]
E_J = [1, 2, 3, 4, 5]


def grid_e() -> list[dict[str, Any]]:
    return [{"k": k, "j": j} for k in E_K for j in E_J]


def ladders_e(cfg: dict[str, Any]) -> dict[str, list[Any]]:
    return {"k": E_K, "j": E_J}


def month_window(index: pd.DatetimeIndex, k: int, j: int) -> pd.Series:
    """수익일 d 가 그 달 마지막 k 거래일 또는 첫 j 거래일이면 True (거래 달력은 미리 안다)."""
    ym = pd.Series(index.year * 12 + index.month, index=index)
    pos_start = ym.groupby(ym).cumcount() + 1
    pos_end = ym.groupby(ym).cumcount(ascending=False) + 1
    return (pos_end <= k) | (pos_start <= j)


def target_e(u: Universe, cfg: dict[str, Any]) -> pd.DataFrame:
    """종가 체결(결정 → 다음 날 종가 체결 → 그다음 날 수익)이라 이틀 앞 수익일의 창으로 정한다."""
    win = month_window(u.ret.index, int(cfg["k"]), int(cfg["j"])).astype(float).shift(-2)
    return _invalid_rows(win.to_frame("EQ"), u.eligible["EQ"] & win.notna())


def describe_e(cfg: dict[str, Any]) -> str:
    return f"월말 마지막 {cfg['k']}거래일 + 월초 첫 {cfg['j']}거래일만 보유, 나머지 현금"


FAMILIES: dict[str, Family] = {
    "A": Family("A", "지수 위험관리", "index", grid_a, ladders_a, target_a, describe_a),
    "B": Family("B", "자산배분 추세추종", "multi", grid_b, ladders_b, target_b, describe_b),
    "C": Family("C", "상대·듀얼 모멘텀", "multi", grid_c, ladders_c, target_c, describe_c),
    "D": Family("D", "위험 균등 정적 배분", "multi", grid_d, ladders_d, target_d, describe_d),
    "E": Family("E", "달력 효과", "index", grid_e, ladders_e, target_e, describe_e),
}


def target_key(fam: Family, cfg: dict[str, Any]) -> str:
    """목표 비중은 매매 주기와 무관하다 — 주기만 다른 조합은 같은 목표를 쓴다 (계산 재사용)."""
    return fam.key({k: v for k, v in cfg.items() if k != "sched"})


def grid_sizes() -> dict[str, int]:
    return {c: len(f.grid()) for c, f in FAMILIES.items()}


def _check_grids() -> None:
    for c, f in FAMILIES.items():
        g = f.grid()
        keys = [f.key(x) for x in g]
        assert len(keys) == len(set(keys)), c


_check_grids()
__all__ = ["FAMILIES", "MENUS", "Universe", "Family", "schedule_of", "target_key", "grid_sizes", "month_window",
           "base_weights"]
