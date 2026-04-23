#!/bin/bash
set -euo pipefail

RUN_ID=$(date +"%Y%m%d_%H%M%S")
TRACE_COUNT=${1:-100}
SAMPLES_PER_TRACE=${2:-50}
BASE_DIR="$(dirname "$0")/../data/random_$RUN_ID"
TARGET_SCRIPT="$(dirname "$0")/../target/target.py"

mkdir -p "$BASE_DIR"

echo "====================================="
echo "Collecting RANDOM traces"
echo "Trace count: $TRACE_COUNT"
echo "Samples per trace: $SAMPLES_PER_TRACE"
echo "Saving to: $BASE_DIR"
echo "====================================="

for i in $(seq 1 "$TRACE_COUNT")
do
  echo "Random Trace $i/$TRACE_COUNT"

  INPUT=$(openssl rand -hex 16)
  echo "$INPUT" >> "$BASE_DIR/inputs.txt"

  sudo powermetrics --samplers cpu_power -i 10 -n "$SAMPLES_PER_TRACE" > "$BASE_DIR/trace_$i.txt" &
  PID=$!

  python3 "$TARGET_SCRIPT" "$INPUT"

  wait $PID
done

echo "✅ Random trace collection complete!"
