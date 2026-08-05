#!/usr/bin/env bash
# Stage-What controls C01/C02 (protocol 8.2 / amendment A-4): the <color> segment
# generated with NO <where> reasoning in front of it.  Same driver, MODE flipped.
#
#   usage:  genctx_forced_color_dual.sh <checkpoint_path>
#
# Scope (WT-J10 + the code that consumes it):
#   run_what.py      --split train      -> <root>/train/forced_color
#   run_what.py      --eval-split V_what-> <root>/V_what/forced_color
#   evaluate_what.py --split V_what     -> the same root
# so `train` and `V_what` only; V_where is a Where-B split and Where-B never
# reads the forced-prefix generation.
#
# This is a SEPARATE full pass over the same 160k prompts, i.e. a second job of
# the same order as the two_segment one -- see NOTES.md §4 before scheduling it.
# It is needed by 2 of Stage-What's 12 arms; the other 10 (and all 8 Where-B
# arms) need only the two_segment pass.
#
# D-20 submission (same four steps as genctx_dual.sh):
#   cd /home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/genctx_jobs
#   rm -f logs/driver_forced_color.log
#   nohup setsid bash genctx_forced_color_dual.sh <ckpt> > logs/driver_forced_color.log 2>&1 &
#   echo $! > logs/driver_forced_color.pid
#   ps -p "$(cat logs/driver_forced_color.pid)" -o pid,etime,cmd --no-headers
#   grep -q PROBE-OK logs/driver_forced_color.log && tail -n 40 logs/driver_forced_color.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODE=forced_color
export SPLITS="${SPLITS:-V_what train}"
exec bash "${HERE}/genctx_dual.sh" "$@"
