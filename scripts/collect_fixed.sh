#!/bin/bash
set -euo pipefail

RUN_ID=$(date +"%Y%m%d_%H%M%S")
TRACE_COUNT=${1:-100}
SAMPLES_PER_TRACE=${2:-100}
CORE_MODE=${3:-ecore}
WORKLOAD_SECONDS=${4:-1.5}
BASE_DIR="$(dirname "$0")/../data/fixed_$RUN_ID"
TARGET_SCRIPT="$(dirname "$0")/../target/target.py"
FIXED_INPUT="afsghJklafdh1237"
#Avjbeiuh48323032
mkdir -p "$BASE_DIR"

echo "====================================="
echo "Collecting FIXED traces"
echo "Trace count: $TRACE_COUNT"
echo "Samples per trace: $SAMPLES_PER_TRACE"
echo "Core mode: $CORE_MODE"
echo "Workload seconds: $WORKLOAD_SECONDS"
echo "Saving to: $BASE_DIR"
echo "====================================="

run_target() {
  local input="$1"

  case "$CORE_MODE" in
    ecore)
      # Strictly bias workload to E-cores using background class.
      taskpolicy -c background python3 "$TARGET_SCRIPT" "$input" "$WORKLOAD_SECONDS"
      ;;
    pcore)
      # Foreground/default class to favor P-cores.
      taskpolicy -c default python3 "$TARGET_SCRIPT" "$input" "$WORKLOAD_SECONDS"
      ;;
    *)
      echo "❌ Invalid core mode: $CORE_MODE (use: ecore|pcore)"
      exit 1
      ;;
  esac
}

for i in $(seq 1 "$TRACE_COUNT")
do
  echo "Fixed Trace $i/$TRACE_COUNT"
  echo "$FIXED_INPUT" >> "$BASE_DIR/inputs.txt"

  sudo powermetrics --samplers cpu_power -i 10 -n "$SAMPLES_PER_TRACE" > "$BASE_DIR/trace_$i.txt" &
  PID=$!

  run_target "$FIXED_INPUT"

  wait $PID
done

echo "✅ Fixed trace collection complete!"
