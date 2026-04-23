#!/bin/bash
set -euo pipefail

RUN_ID=$(date +"%Y%m%d_%H%M%S")
TRACE_COUNT=${1:-100}
BASE_DIR="$(dirname "$0")/../data/fixed_$RUN_ID"
TARGET_SCRIPT="$(dirname "$0")/../target/target.py"
FIXED_INPUT="AAAAAAAAAAAAAAAA"

mkdir -p "$BASE_DIR"

echo "====================================="
echo "Collecting FIXED traces"
echo "Trace count: $TRACE_COUNT"
echo "Saving to: $BASE_DIR"
echo "====================================="

for i in $(seq 1 "$TRACE_COUNT")
do
  echo "Fixed Trace $i/$TRACE_COUNT"
  echo "$FIXED_INPUT" >> "$BASE_DIR/inputs.txt"

  sudo powermetrics --samplers cpu_power -i 10 -n 50 > "$BASE_DIR/trace_$i.txt" &
  PID=$!

  python3 "$TARGET_SCRIPT" "$FIXED_INPUT"

  wait $PID
done

echo "✅ Fixed trace collection complete!"
