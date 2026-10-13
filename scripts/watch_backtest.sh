#!/usr/bin/env bash
# 백그라운드 백테스트 진행 상황을 10초마다 한 줄씩 보여준다. 종료: Ctrl+C (백테스트는 계속 돈다)
# 사용법: bash scripts/watch_backtest.sh [로그파일]
cd "$(dirname "$0")/.." || exit 1
LOG="${1:-$(ls -t output/logs/*.log 2>/dev/null | head -1)}"
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
  echo "로그 파일이 없습니다 (output/logs/*.log)"; exit 1
fi
echo "로그: $LOG  (Ctrl+C로 보기만 종료)"
# 로그 이름 = <스크립트>_<날짜>_<시각>.log → 그 스크립트가 돌고 있으면 '실행 중' (새 실험 스크립트도 따로 등록할 필요 없음)
NAME=$(basename "$LOG" .log)
SCRIPT=${NAME%_*_*}
while true; do
  STAGE=$(grep -a "^=====" "$LOG" | tail -1)
  LAST=$(tr '\r' '\n' < "$LOG" | grep -a "%\]" | tail -1)
  if pgrep -f "scripts/${SCRIPT}.py" >/dev/null \
     || pgrep -f "run_hypothesis_ab|warm_feature_cache|ccs_placebo|earnings_filter_check|validate_upside|fetch_insider_data|pt1_baseline_equity|idea_batch_research|wfo_structure" >/dev/null; then
    STATE="실행 중"
  else
    STATE="종료됨"
  fi
  echo "$(date +%H:%M:%S) [$STATE] ${STAGE#===== } |${LAST}"
  if [ "$STATE" = "종료됨" ]; then
    echo; tr '\r' '\n' < "$LOG" | grep -av "^Warning\|%\]" | tail -40; break
  fi
  sleep 10
done
