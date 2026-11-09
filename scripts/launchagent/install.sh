#!/bin/bash
# 야간 · 장중 신호 LaunchAgent 설치 — Mac mini 에서 한 번 실행: bash scripts/launchagent/install.sh
# (신호 스크립트를 ~/account-watch 로 복사하고 plist 를 ~/Library/LaunchAgents 에 넣어 등록한다 — LaunchAgent 는
#  ~/Documents 를 못 읽어서 복사본을 쓴다)
#   nightly-us · nightly-universe  매일 6:00 · 6:30 PM ET 야간 작업
#   market-pulse                   월~금 09:45~15:45 15분마다 + 16:20 사이트 장중 펄스 (2026-11-09)
set -eu
SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/account-watch/logs" "$HOME/Library/LaunchAgents"
cp "$SRC/nightly_dispatch.sh" "$HOME/account-watch/nightly_dispatch.sh"
cp "$SRC/market_pulse_dispatch.sh" "$HOME/account-watch/market_pulse_dispatch.sh"
for label in com.seancho.nightly-us com.seancho.nightly-universe com.seancho.market-pulse; do
  plist="$HOME/Library/LaunchAgents/$label.plist"
  launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
  cp "$SRC/$label.plist" "$plist"
  launchctl bootstrap "gui/$(id -u)" "$plist"
  echo "✅ $label 등록"
done
echo "지금 바로 시험: bash ~/account-watch/nightly_dispatch.sh canada-scan.yml  (로그: ~/account-watch/logs/nightly_dispatch.log)"

