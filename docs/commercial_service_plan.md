# 퀀트 시스템 → 상업 웹 서비스: 셋업 진단 + 전환 계획

> 작성: 2026-10-14 (온타리오 기준) · 법률 자문 아님 — 유료 전환 전 변호사 확인 필요

## Context
사용자는 이 퀀트 시스템(스크리너 + 페이퍼 트레이딩)을 웹 서비스로 만들고 싶어 함.
- 참고 화면: 인스타 "ARC AI Trading"처럼 캐릭터들이 사무실에서 일하는 게임형 UI
- 핵심 기능:
  1. 필터로 종목 찾기 (매출 성장, 낮은 RSI, MACD 등)
  2. 페이퍼 트레이딩 결과 공개
- 사용자 답변:
  - 대상: 한국 + 북미 (한/영)
  - 수익: 무료 베타로 시작해 나중에 유료
  - 데이터 예산: 월 $100 이하
  - 이번 범위: **정리 계획까지만** (코드 변경 없음)
- 1순위 질문: "지금 셋업으로 상업화해도 나중에 문제가 없나, 무엇을 바꿔야 하나"

## 결론
- **서비스로 만드는 것은 가능하다.** CI가 매일 `data/paper_trading/*_ranked.parquet`(124개 컬럼)와 PT JSON을 이미 만들고 있어서, 웹 화면의 원천 데이터는 거의 준비돼 있다.
- **지금 셋업 그대로 공개·유료화하면 안 된다.** 데이터 출처 3곳이 상업 이용 불가이거나 약관 위반이다.
  - Yahoo: 개인용 전용
  - openinsider: HTML 스크래핑
  - TradingView: 로그인 쿠키로 자동 캡처
- 저장소가 공개 상태라서, 그 데이터를 이미 공개 재배포하고 있다.
- 차트는 생각보다 쉽다. TradingView 무료 위젯을 출처 표기와 함께 임베드하면 합법적으로 넣을 수 있다.

## 1. 현재 셋업 진단

| 항목 | 지금 | 상업 이용 | 바꿀 것 |
|---|---|---|---|
| 가격(OHLCV) | yfinance (`src/data/fetch.py:76`) | ❌ Yahoo는 개인용 전용 | 표시 권한을 서면으로 받은 유료 EOD 공급사 (§3) |
| 재무(매출·마진·부채 등) | yfinance `get_info` (`src/screener/fundamentals.py:120-160`) | ❌ | **SEC EDGAR companyfacts** (미국 정부 데이터, 무료·상업 OK). 이미 만든 `scripts/build_sec_data.py`, `rev_yoy_quarterly.parquet` 로직을 재사용 |
| 공매도%·기관보유%·다음 실적일 | yfinance | ❌ | v1에서는 뺀다. 실적일은 지난 8-K 2.02 날짜 + 1년을 "예상"으로 표시하는 방법을 검토 |
| 내부자 매수 | openinsider 스크래핑 (`src/screener/insider.py:77`) | ❌ 제3자 사이트 스크래핑 | **SEC Form 4 직접 사용**. `scripts/fetch_insider_data.py`, `insider_first_owner.parquet` 재사용 |
| 차트 이미지 | Playwright + `TRADINGVIEW_SESSIONID`로 자동 캡처 (`charts/tradingview_capture.py`) | ❌ TV 약관 3조는 자동 접근을 금지(계정 정지 사유). 개인 용도로도 위험 | 서비스에서는 **TradingView 무료 위젯 임베드**(출처 표기 유지)로 대체 |
| 섹터 | Wikipedia의 GICS (`scripts/update_sp500_tickers.py`) | ⚠️ GICS는 MSCI/S&P 독점 분류. Wikipedia는 CC BY-SA | SEC SIC 코드(submissions API)로 직접 분류 |
| 유니버스 | Nasdaq Trader 심볼 목록, Wikipedia S&P 500 | ⚠️ 낮음 | "S&P 500" 상표를 쓸지 변호사에게 확인한다. 쓰지 않는다면 "미국 대형주 상위 500"(자체 시총 순위)로 정의 |
| ML 모델 | `data/ml_models/*.pkl`(Yahoo 데이터로 학습) | ⚠️ 회색지대 | 라이선스 받은 데이터로 재학습한 뒤 사용 |
| ISB 지표 | `pinescript/ISB.pine` | ✅ 사용자 본인 코드 (goldenEntry 저장소와 동일) | v3.0 최초 버전을 직접 처음부터 쓴 게 맞는지만 본인이 확인. 검증에서 탈락했으므로 상품에는 넣지 않음 |
| 저장소 | `sean-ccho/Project_1` **공개**, LICENSE 없음 | ❌ 코드와 매일 결과가 공짜로 공개됨 | private 전환 (§4) |
| Google Sheets | 결과 기록 + 이메일 | ⚠️ 위법은 아니지만 공개 서비스 백엔드로는 부적합 (분당 읽기 300회, 서비스 계정 1개로 공유, SLA 없음) | **운영자(본인) 전용 화면으로 유지**. 웹은 JSON을 읽음 |
| `내계좌` 탭 | 본인 보유 종목 | — | 서비스에는 절대 넣지 않음 (개인 맞춤 조언 = 등록 대상) |

덤으로 발견한 것: `liquidity_filter`(`src/screener/processing.py:48-51`)는 상위 25%만 남긴다. 그런데 설정 주석(`config.py:119`)에는 "하위 25% 제거"라고 적혀 있다. 스크리너 상품으로 쓰면 종목 범위가 좁아지므로, 나중에 따로 확인할 것.

## 2. 규제: 나라별로 지킬 선

**공통 원칙**
- 누구에게나 같은 정보를 보여 준다. 개인 맞춤 조언은 하지 않는다.
- 1:1 상담·채팅·댓글로 종목 이야기를 하지 않는다.
- 사용자 계좌를 연결하지 않는다.
- 자동매매나 돈을 대신 굴리는 기능을 만들지 않는다.

| 지역 | 근거 | 지금 계획의 위치 | 할 일 |
|---|---|---|---|
| 캐나다 (사용자 거주지, 온타리오) | NI 31-103 s.8.25 "일반 조언" 등록 면제: 개인 맞춤이 아닌 조언 | 필터 도구와 공개 PT 기록은 면제 범위 안으로 보임 | 보유 종목 이해관계 고지 문구 넣기. 유료 전환 전 온타리오 증권 변호사 1회 상담 |
| 미국 | 투자자문업법 publisher 예외 (*Lowe v. SEC*): 비개인적·정기적·홍보 목적이 아닌 출판물 | 같음 | 본인이 가진 종목을 띄우지 않기. 성과는 허위·과장 없이 표시 (사기 금지 조항은 항상 적용) |
| 한국 | 자본시장법 **유사투자자문업** (대가를 받고 불특정 다수에게 정보 제공 → 금융위 신고, 금융투자협회 교육 이수). 2024-08-14부터 **양방향 소통이 있으면 투자자문업 등록** 대상 | 무료 베타는 "대가"가 없어 대상 밖일 가능성이 큼. **유료 전환 시 신고 필요 가능성 높음** | 유료 전환 전에, 해외 거주자가 운영해도 신고 대상인지 한국 변호사·행정사에게 확인. 유료 기능은 일방향만 (커뮤니티·채팅 없음) |

**광고·표현 규칙** (한국 광고 규제 + 미국·캐나다 사기 금지 조항)
- 금지:
  - 수익 보장
  - 허위·미실현 수익률
  - 금융회사로 오인하게 하는 표현
- 필수 고지:
  - 정식 투자자문업자가 아님
  - 개인 상담 불가
  - 원금 손실 가능성과 손실은 투자자 본인 책임

## 3. 성과 표시: 정직이 곧 법적 방어
- 우리 연구의 결론은 "PT-1은 SPY 대비 엣지 없음"이다. 10년 PIT 검증에서 Sharpe 0.70 vs SPY 0.89였다.
  - 따라서 "수익 내는 AI"로 팔면 허위·과장 광고가 된다.
  - 미국 SEC는 2024년 'AI 워싱'(AI를 과장한 광고)을 실제로 제재했다.
- 상품의 정체성: **"종목을 쉽게 찾는 검색 도구 + 투명한 모의투자 실험 기록"**
- 성과 화면 규칙:
  - PT-1과 PT-SPY를 같은 그래프에 그린다
  - 손실 거래를 포함한 전체 거래를 보여 준다
  - 시작일을 명시한다
  - "모의투자(실제 돈 아님)"라고 표시한다
- 백테스트 수치(+147% 등)는 절대 쓰지 않는다
- "AI 팀"이라는 이름은 규칙 기반인 실제 구조에 맞게 "퀀트 팀" 등으로 바꾼다

## 4. 바뀐 셋업 (월 $100 이하 기준)

| 구성 | 선택 | 비용 |
|---|---|---|
| EOD 가격 | 1순위 Tiingo (상업 플랜 약 $50/월). 기본은 "내부 사용"이라서 **"지표만 계산해 공개 사이트에 파생 신호를 표시하고, 원 가격은 표시하지 않음"** 조건으로 서면 허락을 받는다. 거절당하면 다른 공급사 2~3곳에 같은 조건으로 문의한다 | ≈$50 |
| 재무·내부자·섹터 | SEC EDGAR (User-Agent 헤더 필수, 초당 10회 이하) | $0 |
| 화면 속 가격·차트 | TradingView 무료 위젯 (로고·링크 유지). 유료 전환 전에 TV에 상업 사이트 사용 서면 확인 | $0 |
| 웹 호스팅 | Cloudflare Pages (무료 플랜도 상업 이용 가능). ※ Vercel Hobby는 상업 이용 금지 | $0 |
| 저장소 | private 전환. 비공개 저장소는 Actions 무료 한도가 월 2,000분인데 매일 실행이 몇 시간씩 걸려 초과함 → **Mac mini를 self-hosted runner로** (항상 켜져 있음) | $0 |
| 회원·결제 (유료 단계) | Supabase(인증) + Paddle 같은 Merchant of Record. MoR이 국가별 부가세를 대신 처리하는데, 한국 해외 전자용역 VAT도 포함되는지 확인 | 수수료만 |
| 도메인 | — | 약 $15/년 |

- 예산 $100 안에서 Tiingo를 포함해 표시 권한을 주는 곳이 하나도 없다면, 정직한 답은 "예산을 $300~500으로 올려야 한다"이다 (Intrinio·EODHD 상업 플랜 수준).
- 이 경우 계획을 다시 조정한다.
- 저장소를 private로 바꾸면 시트의 `=IMAGE()` 차트가 깨진다. 다만 TV 캡처를 어차피 걷어내므로 함께 정리한다.

## 5. 웹 서비스 구조 (나중에 구현할 때)

```
매일 CI (Mac mini runner)
  ├─ 기존 파이프라인 (스크리너 → PT-1/PT-1S/PT-SPY → 헬스체크)
  ├─ Google Sheets + 이메일  ← 운영자(본인) 전용, 그대로 유지
  └─ [신규] 공개용 내보내기 → screener.json / pt.json / pipeline_status.json
        (허용 컬럼만 화이트리스트로) → Cloudflare Pages
웹 (정적 사이트, 한/영)
  ├─ 스크리너: 브라우저 안에서 필터 (최대 수천 행이라 서버 불필요)
  ├─ 모의투자: PT-1 vs SPY 곡선, 보유 종목, 전체 거래 기록
  ├─ 종목 상세: TradingView 위젯 차트 + 우리 지표
  └─ "퀀트 팀 사무실": pipeline_status.json으로 캐릭터 상태 표시
```

**필터 v1** (모두 허락받은 소스에서 나옴)
- 가격 파생 지표: RSI, MACD 히스토그램·크로스, EMA20/50/200 정배열, 52주 위치, 5일·20일 수익률, 거래량 Z, ATR%, 일·주·월봉 패턴, 점수
- SEC 데이터: 매출 YoY, 이익 성장, 마진, 부채비율, 현금흐름, 90일 내부자 매수, 섹터(SIC)

**사무실 화면**: 캐릭터마다 실제 파이프라인 단계를 하나씩 맡는다.

| 캐릭터 | 맡는 단계 | 근거 코드 |
|---|---|---|
| 수집 | 가격·재무·내부자 데이터 수집 | — |
| 분석 | 지표 계산 | `features.py` |
| 패턴 | 차트 패턴 판별 | `patterns.py` |
| 점수 | 점수·전략 구분 | `signals.py` |
| 시장판정 | 강세/약세 → 슬롯 수 | — |
| 트레이더 | PT-1 | — |
| 심판 | PT-SPY와 비교 | — |
| 감시 | 헬스체크 | — |

- 말풍선에는 실제 로그를 띄운다. 예: "후보 882→227", "오늘 매수 0건(약세장)".
- 그림은 직접 만들거나 CC0 에셋(예: Kenney 아이소메트릭 팩)을 쓴다.
- ARC의 캐릭터·이름·배치는 베끼지 않는다. "아이소메트릭 사무실"이라는 콘셉트 자체는 보호 대상이 아니다.

**나중에 손댈 파일**
- 데이터 교체: `src/data/fetch.py`, `src/screener/fundamentals.py`, `src/screener/insider.py`
- 차트·유니버스: `charts/tradingview_capture.py`, `scripts/update_sp500_tickers.py`
- CI: `.github/workflows/run-screener.yml`
- 신규 공개 내보내기 모듈

## 6. 로드맵

**0단계: 정리** (공개 전 필수, 코드 작업)
1. 저장소 private 전환 + Mac mini self-hosted runner
2. TV 자동 캡처 끄기 (`CHARTS_ENABLED=False`)
3. 내부자 데이터: openinsider → SEC Form 4
4. 재무 데이터: yfinance → SEC companyfacts
5. 섹터: GICS → SIC
6. 가격 공급사에 서면 문의 (표시·파생 데이터 조건)
7. 가격 소스 교체 후 PT 결과가 크게 달라지지 않는지 비교 검증

**1단계: 무료 베타**
1. 공개용 JSON 내보내기
2. 정적 웹사이트 (한/영)
3. 면책 고지·이용약관·개인정보처리방침 (PIPEDA + 한국 개인정보보호법). 이메일을 받을 때만 필요
4. 서비스 이름 상표 검색 (KIPRIS / CIPO / USPTO)

**2단계: 유료 전환 전**
1. 변호사 상담 2건: 온타리오 증권 변호사, 한국 금융규제 변호사·행정사
2. 한국 유사투자자문업 신고 여부 결정
3. TradingView 상업 사용 서면 확인
4. 온타리오 사업자명 등록, HST 등록 (매출이 4분기 합계 $30k를 넘을 때), MoR 결제 연결

## 검증 방법 (이 계획이 맞는지 확인하는 법)
- 데이터 공급사 답장(이메일)으로 "공개 웹 표시 + 파생 지표" 허용 범위를 문서로 확보한다.
- TradingView 위젯의 상업 사용 여부를 TV의 서면 답으로 확보한다.
- 변호사 상담에서 이 문서의 §1, §2 표를 그대로 보여 주고 확인받는다.
- 0단계를 마친 뒤 `grep -rn "yfinance\|openinsider\|tradingview_capture" src/`로 서비스 경로에 남은 사용처가 없는지 확인한다.
- 같은 기간 PT 백테스트를 새 가격 소스로 다시 돌려, 거래 목록 차이가 작은지 확인한다.

## 참고 출처 (요약에 사용)
- yfinance 자체 안내 "Yahoo API는 개인용": https://pypi.org/project/yfinance/
- TradingView 약관 (자동 접근 금지, 위젯 출처 표기): https://in.tradingview.com/policies , 위젯 문서: https://kr.tradingview.com/widget-docs/getting-started/
- OSC NI 31-103: https://www.osc.gov.on.ca/documents/en/Securities-Category3/ni_20190612_31-103_unofficial-consolidation.pdf , 일반 조언 면제 해설: https://nssc.novascotia.ca/node/1147975
- Lowe v. SEC / Seeking Alpha 판결 해설: https://www.gtlaw.com/en/insights/2024/8/no-need-for-seeking-alpha-to-seek-registration
- 한국 자본시장법 개정 (2024-08-14, 양방향 금지): https://www.daeryunlaw.com/newsletter/news/53 , https://www.shinkim.com/kor/media/newsletter/2998
- 데이터 공급사 라이선스: https://www.tiingo.com/about/pricing , https://site.financialmodelingprep.com/insights/platform/can-you-use-fmp-data-in-a-public-app-website-or-client-dashboard , https://massive.com/terms/market_data_terms.pdf
- Google Sheets API 한도: https://developers.google.com/sheets/api/limits
- Vercel Hobby 상업 금지: https://conductatlas.com/platform/vercel/vercel-terms-of-service/provision/CA-P-048642/hobby-plan-personal-non-commercial-use-only/
- ※ 법률 자문이 아님. 가격·약관은 바뀔 수 있으니 실행 시점에 각 공식 페이지로 재확인.
