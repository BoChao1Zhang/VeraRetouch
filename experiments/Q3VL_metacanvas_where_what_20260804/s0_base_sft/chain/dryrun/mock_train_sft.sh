#!/usr/bin/env bash
# Stand-in for a Base SFT rank process.  Its command line contains
# "mock_train_sft" so the chain's `ps -p <pid> -o args= | grep -F` liveness test
# has something real to match, including the PID-reuse guard.
#   mock_train_sft.sh <seconds>
sleep "${1:-6}"
