#!/bin/bash
# 야간 신호 LaunchAgent 설치 — Mac mini 에서 한 번 실행: bash scripts/launchagent/install.sh
# (nightly_dispatch.sh 를 ~/account-watch 로 복사하고 plist 를 ~/Library/LaunchAgents 에 넣어 등록한다)
set -eu
SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/account-watch/logs" "$HOME/Library/LaunchAgents"
cp "$SRC/nightly_dispatch.sh" "$HOME/account-watch/nightly_dispatch.sh"
for label in com.seancho.nightly-us com.seancho.nightly-universe; do
  plist="$HOME/Library/LaunchAgents/$label.plist"
  launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
  cp "$SRC/$label.plist" "$plist"
  launchctl bootstrap "gui/$(id -u)" "$plist"
  echo "✅ $label 등록"
done
echo "지금 바로 시험: bash ~/account-watch/nightly_dispatch.sh canada-scan.yml  (로그: ~/account-watch/logs/nightly_dispatch.log)"

