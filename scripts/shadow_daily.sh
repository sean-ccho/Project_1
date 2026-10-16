#!/usr/bin/env bash
# 그림자 추적 매일 기록 (scripts/shadow_track.py). 같은 일봉 날짜는 한 번만 기록한다. 일요일 아침엔 진행 보고서도 만든다.
# 보통은 ~/shadow-track/run.sh 가 이 리포의 얕은 복제를 origin/main 으로 맞춘 뒤 부른다 (LaunchAgent com.seancho.shadow-track,
# 07:30·19:30 — 저녁은 CI 가 늦게 끝난 날의 재시도). macOS 가 예약 작업의 ~/Documents 접근을 막아서 복제본에서 돈다.
# 환경변수: SHADOW_DIR (기록 폴더), SHADOW_PY (파이썬, 기본 .venv/bin/python), SHADOW_LOG_DIR (로그, 기본 output/logs)
cd "$(dirname "$0")/.." || exit 1
PY="${SHADOW_PY:-.venv/bin/python}"
LOG_DIR="${SHADOW_LOG_DIR:-output/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/shadow_track_$(date +%Y%m%d_%H%M%S).log"
export PYTHONPATH=.:src
"$PY" -u scripts/shadow_track.py --record >> "$LOG" 2>&1
if [ "$(date +%u)" = "7" ] && [ "$(date +%H)" -lt 12 ]; then
  "$PY" -u scripts/shadow_track.py --report >> "$LOG" 2>&1
fi
