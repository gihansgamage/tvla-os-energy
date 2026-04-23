#!/bin/bash

# Create unique experiment folder
RUN_ID=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="../data/fixed_$RUN_ID"

mkdir -p "$BASE_DIR"

echo "====================================="
echo "Collecting FIXED traces"
echo "Saving to: $BASE_DIR"
echo "====================================="

FIXED_INPUT="AAAAAAAAAAAAAAAA"

for i in {1..100}
do
  echo "Fixed Trace $i"

  # Save input (same every time)
  echo "$FIXED_INPUT" >> "$BASE_DIR/inputs.txt"

  sudo powermetrics --samplers cpu_power -i 10 -n 50 > "$BASE_DIR/trace_$i.txt" &
  PID=$!

  python3 ../target/target.py "$FIXED_INPUT"

  wait $PID
done

echo "✅ Fixed trace collection complete!"