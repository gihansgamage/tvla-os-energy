#!/bin/bash

# Create unique experiment folder
RUN_ID=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="../data/random_$RUN_ID"

mkdir -p "$BASE_DIR"

echo "====================================="
echo "Collecting RANDOM traces"
echo "Saving to: $BASE_DIR"
echo "====================================="

for i in {1..100}
do
  echo "Random Trace $i"

  # Generate random input
  INPUT=$(openssl rand -hex 16)

  # Save input for reproducibility
  echo "$INPUT" >> "$BASE_DIR/inputs.txt"

  sudo powermetrics --samplers cpu_power -i 10 -n 50 > "$BASE_DIR/trace_$i.txt" &
  PID=$!

  python3 ../target/target.py "$INPUT"

  wait $PID
done

echo "✅ Random trace collection complete!"