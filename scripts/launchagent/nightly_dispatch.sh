#!/bin/bash
# 야간 파이프라인 실행 신호기 — Mac mini LaunchAgent com.seancho.nightly-*
#
# GitHub 예약(schedule) 실행은 몇 시간씩 늦게 시작해서, 이 Mac 이 정해진 시각에 workflow_dispatch API 로
# 워크플로를 실행시킨다. 실제 작업은 GitHub 에서 돈다 (~/account-watch/trigger.sh 와 같은 방식).
#
# 사용: bash nightly_dispatch.sh <워크플로.yml> [<워크플로.yml> ...]
#   run-screener.yml 은 nightly=true 를 같이 보낸다 (헬스체크를 정기 실행처럼 엄격하게).
# 미국 종가 확인 (2026-11-17): run-screener · universe-scan 은 보내기 전에 Yahoo 미국 일봉의 가장 최근 종가가 다
#   들어왔는지 yahoo_bar_ready.py 로 본다 — 평일 밤 20:00~21:30쯤 Yahoo 가 그날 종가를 빈 칸으로 준다.
#   빈 칸이면 10분마다 다시 보고, 새벽 3시(ET)까지 안 들어오면 보내지 않고 알림만 남긴다 (예비 실행 gate 도 같은 확인).
#   확인 자체가 3번 연속 안 되면(네트워크 · Yahoo 차단) 확인 없이 보낸다. BAR_CHECK=0 이면 확인을 건너뛴다.
# 같은 워크플로가 이미 대기 · 실행 중이면 보내지 않는다 (저녁에 다시 돌린 것과 겹쳐 두 번 돌지 않게).
# 인증: ~/account-watch/token 이 있으면 그 토큰, 없으면 이 Mac 의 git 키체인 인증.
# 로그: ~/account-watch/logs/nightly_dispatch.log
# 시험: DRY_RUN=1 이면 실행 신호만 빼고 다 한다 (DISPATCH_LOG · BAR_SCRIPT · BAR_POLL_SEC · ACTIVE_RECHECK_SEC 로 바꿔 볼 수 있음).
set -u
DIR="$HOME/account-watch"
LOG="${DISPATCH_LOG:-$DIR/logs/nightly_dispatch.log}"
BAR_SCRIPT="${BAR_SCRIPT:-$DIR/yahoo_bar_ready.py}"
POLL="${BAR_POLL_SEC:-600}"
REPO="sean-ccho/Project_1"
mkdir -p "$DIR/logs"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
notify() { /usr/bin/osascript -e "display notification \"$1\" with title \"야간 파이프라인\"" 2>/dev/null; }

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

# 같은 워크플로의 대기 · 실행 중 개수 ('?' = 못 읽음)
active_runs() {
  /usr/bin/curl -s --max-time 30 -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/actions/workflows/$1/runs?per_page=10" | /usr/bin/python3 -c '
import json, sys
try:
    runs = json.load(sys.stdin)["workflow_runs"]
except Exception:
    print("?")
    sys.exit()
print(sum(1 for r in runs if r.get("status") in ("queued", "in_progress", "waiting", "pending", "requested")))'
}

# ── 미국 종가 확인 (run-screener · universe-scan 만) ──
needs_bar=0
for wf in "$@"; do
  case "$wf" in run-screener.yml|universe-scan.yml) needs_bar=1 ;; esac
done
if [ "$needs_bar" -eq 1 ] && [ "${BAR_CHECK:-1}" != "0" ]; then
  if [ ! -f "$BAR_SCRIPT" ]; then
    log "⚠️ $BAR_SCRIPT 없음 — 미국 종가 확인 없이 보냄 (bash scripts/launchagent/install.sh 로 설치)"
  else
    # 마감 = 새벽 3시 ET (정오 이후에 시작했으면 다음 날 새벽 3시) — 이보다 늦게 보내면 작업이 장 시작 전에 안 끝난다
    today=$(TZ=America/Toronto date +%Y-%m-%d)
    until_ts=$(TZ=America/Toronto date -j -f '%Y-%m-%d %H:%M:%S' "$today 03:00:00" +%s)
    if [ "$(TZ=America/Toronto date +%H)" -ge 12 ]; then until_ts=$((until_ts + 86400)); fi
    errs=0
    while :; do
      res=$(/usr/bin/python3 "$BAR_SCRIPT" 2>&1)
      rc=$?
      log "미국 종가 확인: $res"
      [ "$rc" -eq 0 ] && break
      if [ "$rc" -eq 1 ]; then errs=0; else errs=$((errs + 1)); fi
      if [ "$errs" -ge 3 ]; then
        log "⚠️ 종가 확인이 3번 연속 안 됨 — 확인 없이 보냄"
        break
      fi
      if [ "$(date +%s)" -ge "$until_ts" ]; then
        log "❌ 새벽 3시(ET)까지 미국 종가가 안 들어옴 — $* 보내지 않음"
        notify "새벽 3시까지 Yahoo 미국 종가가 안 들어와 오늘 밤 작업을 보내지 않았어요 — ~/account-watch/logs/nightly_dispatch.log"
        exit 1
      fi
      sleep "$POLL"
    done
  fi
fi

fail=0
for wf in "$@"; do
  # 이미 대기 · 실행 중이면 건너뜀 — 예비 실행 gate 처럼 몇 초 만에 끝나는 것은 잠깐 기다려 다시 본다
  a=$(active_runs "$wf")
  if [ "$a" != "?" ] && [ "${a:-0}" -gt 0 ]; then
    sleep "${ACTIVE_RECHECK_SEC:-120}"
    a=$(active_runs "$wf")
  fi
  if [ "$a" != "?" ] && [ "${a:-0}" -gt 0 ]; then
    log "⏭️ $wf 이미 대기 · 실행 중인 것이 있어 보내지 않음"
    continue
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then
    log "(시험) $wf 실행 신호 보낼 차례 — 실제로는 안 보냄"
    continue
  fi
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
    notify "$wf 실행 신호 실패 — ~/account-watch/logs/nightly_dispatch.log"
  fi
done
exit "$fail"
