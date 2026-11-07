#!/bin/bash
# 야간 파이프라인 실행 신호기 — Mac mini LaunchAgent com.seancho.nightly-*
#
# GitHub 예약(schedule) 실행은 몇 시간씩 늦게 시작해서, 이 Mac 이 정해진 시각에 workflow_dispatch API 로
# 워크플로를 실행시킨다. 실제 작업은 GitHub 에서 돈다 (~/account-watch/trigger.sh 와 같은 방식).
#
# 사용: bash nightly_dispatch.sh <워크플로.yml> [<워크플로.yml> ...]
#   run-screener.yml 은 nightly=true 를 같이 보낸다 (헬스체크를 정기 실행처럼 엄격하게).
# 인증: ~/account-watch/token 이 있으면 그 토큰, 없으면 이 Mac 의 git 키체인 인증.
# 로그: ~/account-watch/logs/nightly_dispatch.log
set -u
DIR="$HOME/account-watch"
LOG="$DIR/logs/nightly_dispatch.log"
REPO="sean-ccho/Project_1"
mkdir -p "$DIR/logs"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

if [ "$#" -eq 0 ]; then
  log "❌ 워크플로 이름 없음"
  exit 2
fi

if [ -s "$DIR/token" ]; then
  TOKEN=$(tr -d '[:space:]' < "$DIR/token")
else
  TOKEN=$(printf "protocol=https\nhost=github.com\n\n" \
    | GIT_TERMINAL_PROMPT=0 /opt/homebrew/bin/git credential fill 2>/dev/null | sed -n 's/^password=//p')
fi
if [ -z "$TOKEN" ]; then
  log "❌ 토큰 없음 (~/account-watch/token 도, git 키체인 인증도 없음)"
  exit 1
fi

fail=0
for wf in "$@"; do
  body='{"ref":"main"}'
  if [ "$wf" = "run-screener.yml" ]; then body='{"ref":"main","inputs":{"nightly":"true"}}'; fi
  ok=0
  for attempt in 1 2 3; do
    code=$(/usr/bin/curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X POST \
      -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/$REPO/actions/workflows/$wf/dispatches" -d "$body")
    if [ "$code" = "204" ]; then
      log "✅ $wf 실행 신호 보냄"
      ok=1
      break
    fi
    log "⚠️ $wf HTTP $code (시도 $attempt/3)"
    sleep 20
  done
  if [ "$ok" -ne 1 ]; then
    fail=1
    /usr/bin/osascript -e "display notification \"$wf 실행 신호 실패 — ~/account-watch/logs/nightly_dispatch.log\" with title \"야간 파이프라인\"" 2>/dev/null
  fi
done
exit "$fail"

