#!/bin/bash

SCRIPT="07_Production_Run_Analysis.jl"
LOG="run_$(date +%Y%m%d_%H%M%S).log"

if [ $# -eq 0 ]; then
    CMD="julia --project $SCRIPT"
else
    CMD="JULIA_NUM_THREADS=$1 julia --project $SCRIPT"
fi

{
    echo "$(date '+%F %T') CMD: $CMD"

    eval "$CMD"

    echo "$(date '+%F %T') DONE"
} 2>&1 | tee "$LOG"