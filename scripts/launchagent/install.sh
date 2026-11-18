#!/bin/bash
# 야간 · 장중 신호 LaunchAgent 설치 — Mac mini 에서 한 번 실행: bash scripts/launchagent/install.sh
# (신호 스크립트를 ~/account-watch 로 복사하고 plist 를 ~/Library/LaunchAgents 에 넣어 등록한다 — LaunchAgent 는
#  ~/Documents 를 못 읽어서 복사본을 쓴다)
#   nightly-canada                 매일 6:00 PM ET 캐나다 전체분석 (TSX 는 저녁 Yahoo 빈 칸 문제가 없음)
#   nightly-us · nightly-universe  매일 10:00 · 10:30 PM ET 야간 작업 (2026-11-17 6:00 · 6:30 PM 에서 옮김 —
#                                  평일 밤 20:00~21:30쯤 Yahoo 미국 종가가 빈 칸이라 그 뒤로, 보내기 전에 yahoo_bar_ready.py 로 확인)
#   market-pulse                   월~금 09:45~15:45 15분마다 + 16:20 사이트 장중 펄스 (2026-11-09)
# 같은 날 저녁 6시~10시 반 사이에 설치하면 그날 밤 작업이 두 번 돌 수 있다 — 아침 · 낮에 설치한다.
set -eu
SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/account-watch/logs" "$HOME/Library/LaunchAgents"
cp "$SRC/nightly_dispatch.sh" "$HOME/account-watch/nightly_dispatch.sh"
cp "$SRC/yahoo_bar_ready.py" "$HOME/account-watch/yahoo_bar_ready.py"
cp "$SRC/market_pulse_dispatch.sh" "$HOME/account-watch/market_pulse_dispatch.sh"
for label in com.seancho.nightly-us com.seancho.nightly-canada com.seancho.nightly-universe com.seancho.market-pulse; do
  plist="$HOME/Library/LaunchAgents/$label.plist"
  launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
  cp "$SRC/$label.plist" "$plist"
  launchctl bootstrap "gui/$(id -u)" "$plist"
  echo "✅ $label 등록"
done
echo "시험 (실행 신호는 안 보냄): DRY_RUN=1 bash ~/account-watch/nightly_dispatch.sh run-screener.yml  (로그: ~/account-watch/logs/nightly_dispatch.log)"
echo "미국 종가만 확인: /usr/bin/python3 ~/account-watch/yahoo_bar_ready.py"

