#!/bin/bash
# 사이트 장중 펄스 신호기 — Mac mini LaunchAgent com.seancho.market-pulse 가 매시 :00 · :15 · :30 · :45 와 16:20 에 부른다.
# 월~금 09:45~15:45 (15분마다 25번)와 16:20 (마감 펄스)에만 market-pulse.yml 을 실행시킨다 (nightly_dispatch.sh 재사용).
# 마감 펄스를 16:20 보다 당기지 말 것 — NYSE 마감 동시호가 거래량이 Yahoo 일봉에 16:14~16:18 사이에 들어온다 (11-09 측정).
# 휴장일 · 조기 폐장은 GitHub 쪽 펄스가 알아서 건너뛴다. 로그: ~/account-watch/logs/nightly_dispatch.log
set -u
read -r dow hm < <(TZ=America/Toronto date '+%u %H%M')
if [ "$dow" -gt 5 ]; then exit 0; fi
t=$((10#$hm))
if (( (t >= 944 && t <= 1550) || (t >= 1619 && t <= 1630) )); then
  exec /bin/bash "$HOME/account-watch/nightly_dispatch.sh" market-pulse.yml
fi
exit 0
