#!/bin/bash
# gpu0：教师 logprob 服务（固定 S1F）。用法：UTIL=0.30 PORT=8100 bash teacher_server.sh
set -euo pipefail; S="$(dirname "$0")/.."; UTIL=${UTIL:-0.30} PORT=${PORT:-8100} MAXLEN=${MAXLEN:-6144} GPU=${GPU:-0} bash "$S/teacher_server.sh" "$@"
