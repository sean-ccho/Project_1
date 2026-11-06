"""캐나다 상장 보통주 목록 — TMX 회사 디렉터리 (TSX · TSX Venture, 2026-11-05).

전체분석 '캐나다' 탭(run_universe_scan.py --market ca)이 쓴다. 표시 전용 — 실전 스캔 · 매매와 무관하다.
TMX 가 tsx.com 회사 찾기 화면에 쓰는 공개 JSON (키 없음, 거래소마다 한 번):
    https://www.tsx.com/json/company-directory/search/{tsx|tsxv}/%5E*
    → {"results": [{"symbol": "BBD", "name": "Bombardier Inc.", "instruments": [{"symbol": "BBD.B", ...}, ...]}, ...]}

남기는 것: 보통주 · 종류주(BBD.B · CTC.A) · 신탁 유닛(REI.UN — 캐나다 리츠는 유닛으로 거래된다).
빼는 것 (2026-11-05 디렉터리: TSX 2,343사 → 약 690, TSXV 1,419사 → 약 1,270):
- 회사 이름으로: ETF · CDR(미국 주식 예탁증서 — BMW.TO 같은 것) · 펀드 · 인덱스 · 포트폴리오 · 풀 · 분할주 회사 · 채권,
  실물 보관 신탁(Sprott Physical … · 조폐국 금 · 은 보관), 뮤추얼펀드 법인 클래스(… Class), 이름에 Fund · Split 이 없는
  폐쇄형 펀드 · 분할주 회사 (2026-11-05 전체 실행에서 찾은 것 — _NAME_CLOSED_END_RE)
- 종목 접미사로: 우선주(.PR.A · .PF.A) · 전환사채(.DB) · 워런트(.WT) · 노트(.NT) · 미국 달러 거래(.U — 같은 주식의 USD 줄) ·
  ETF 시리즈(.F 등) · 권리 · 수령증
- TSXV 의 .P = 상장용 껍데기 회사 (Capital Pool Company — 사업이 없고 거래도 거의 없다)
"""

from __future__ import annotations

import json
import re
import urllib.request

import pandas as pd

TMX_URL = "https://www.tsx.com/json/company-directory/search/{board}/%5E*"
_USER_AGENT = "Mozilla/5.0 (compatible; StockScreener/1.0)"
# (디렉터리 이름, 거래소 표시 · TradingView 접두어, 야후 접미사)
BOARDS = (("tsx", "TSX", ".TO"), ("tsxv", "TSXV", ".V"))
# 이름으로 거르는 비주식 상품. 'ETF' 는 대문자 그대로만 (ETFTM 같은 상표 표기 포함, 'Getfit' 같은 말은 안 걸리게)
_NAME_ETF_RE = re.compile(r"ETF")
_NAME_FUND_RE = re.compile(r"\b(CDRs?|Funds?|Index|Portfolios?|Pool|Split|Bonds?)\b", re.I)
# 위 말이 이름에 없는 펀드 — 실물 보관 신탁 · 조폐국 보관증 · 뮤추얼펀드 법인 클래스(이름이 'Class' 로 끝남) · 폐쇄형 펀드 ·
# 분할주 회사 (2026-11-05 디렉터리에서 22개, 모두 펀드인 것을 확인. 'Decisive Dividend Corp' 같은 사업 회사는 안 걸리게 이름째로)
_NAME_CLOSED_END_RE = re.compile(
    r"\bPhysical\b|Royal Canadian Mint|\bClass\s*$|Dividend Select|Premium Income Corp|Prime Dividend Corp"
    r"|Canadian Banc Corp|Income Financial Trust|Precious Metals and Mining Trust|Primary Metals Corp|Real Assets\s*$",
    re.I,
)
# 남기는 접미사 — 없음(보통주) · 신탁 유닛 · 종류주 한 글자 (U = 미국 달러 줄, F = ETF 시리즈라 뺀다)
_KEEP_SUFFIX = {"", "UN", "A", "B", "C", "D", "E", "X", "Y", "Z"}


def is_fund_name(name: str) -> bool:
    """ETF · CDR · 펀드 같은 비주식 상품 이름이면 True."""
    s = str(name or "")
    return bool(_NAME_ETF_RE.search(s) or _NAME_FUND_RE.search(s) or _NAME_CLOSED_END_RE.search(s))


def keep_symbol(symbol: str, board: str) -> bool:
    """TMX 종목 심볼(BBD.B · REI.UN · ABC.P)이 보통주 · 종류주 · 신탁 유닛이면 True."""
    s = str(symbol or "").strip().upper()
    if not s:
        return False
    base, _, suffix = s.partition(".")
    if not base.isalnum():
        return False
    if board == "tsxv" and suffix == "P":
        return False
    return suffix in _KEEP_SUFFIX


def yahoo_ticker(symbol: str, yahoo_suffix: str) -> str:
    """TMX 심볼 → 야후 티커 (BBD.B → BBD-B.TO, REI.UN → REI-UN.TO, ABC → ABC.V)."""
    return str(symbol).strip().upper().replace(".", "-") + yahoo_suffix


def parse_directory(payload: dict, board: str, exchange: str, yahoo_suffix: str) -> pd.DataFrame:
    """디렉터리 JSON 한 개 → 표 (ticker, name, tv, exchange). 순서는 디렉터리 순서."""
    rows = []
    for company in payload.get("results") or []:
        name = str(company.get("name") or "").strip()
        if is_fund_name(name):
            continue
        for inst in company.get("instruments") or []:
            sym = str(inst.get("symbol") or "").strip().upper()
            if keep_symbol(sym, board):
                rows.append({"ticker": yahoo_ticker(sym, yahoo_suffix), "name": name, "tv": f"{exchange}:{sym}",
                             "exchange": exchange})
    return pd.DataFrame(rows, columns=["ticker", "name", "tv", "exchange"])


def _fetch_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_ca_listings(fetch_json=None) -> pd.DataFrame:
    """TSX + TSXV 상장 보통주 → 표 (ticker 야후 형식, name, tv TradingView 심볼, exchange 'TSX'·'TSXV').

    TSX 먼저, 같은 티커는 앞의 것. 한 거래소라도 못 받으면 예외 — 반쪽 목록으로 시장을 나누면 숫자가 틀리므로.
    """
    fetch_json = fetch_json or _fetch_json
    frames = [parse_directory(fetch_json(TMX_URL.format(board=board)), board, exch, suffix)
              for board, exch, suffix in BOARDS]
    out = pd.concat(frames, ignore_index=True).drop_duplicates("ticker", keep="first").reset_index(drop=True)
    counts = out["exchange"].value_counts().to_dict()
    print(f"[캐나다] 상장 목록: TSX {counts.get('TSX', 0)} · TSXV {counts.get('TSXV', 0)}")
    return out
