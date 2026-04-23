# TVLA OS Energy Research Workflow

This repository now includes an end-to-end workflow to:
1. Collect fixed and random power traces.
2. Apply multiple signal filters.
3. Save all outputs separately (raw, filtered, plots, summary).

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Collect traces

Run these on macOS with `powermetrics` permissions.

```bash
./scripts/collect_fixed.sh 100
./scripts/collect_random.sh 100
```

- The optional argument is trace count (default `100`).
- Data is saved in:
  - `data/fixed_<timestamp>/`
  - `data/random_<timestamp>/`

## 3) Analyze and apply filters

```bash
python3 scripts/analyze_traces.py
```

By default, this uses the most recent `data/fixed_*` and `data/random_*` folders.
You can also select explicit folders:

```bash
python3 scripts/analyze_traces.py \
  --fixed-dir data/fixed_YYYYMMDD_HHMMSS \
  --random-dir data/random_YYYYMMDD_HHMMSS
```

## 4) Results (saved separately)

Each run creates:

`results/analysis_<timestamp>/`

Inside that folder:

- `raw/`
  - `fixed_trace_means.csv`
  - `random_trace_means.csv`
- `filtered/fixed/`
  - `raw.csv`
  - `moving_average.csv`
  - `median.csv`
  - `lowpass.csv`
- `filtered/random/`
  - `raw.csv`
  - `moving_average.csv`
  - `median.csv`
  - `lowpass.csv`
- `plots/`
  - `fixed_filters.png`
  - `random_filters.png`
  - `raw_comparison.png`
- `summary.json`

## 5) Filter details

- **Moving average**: smooths short spikes.
- **Median filter**: robust against outlier samples.
- **Low-pass Butterworth**: removes high-frequency noise.

## 6) Research-friendly step-by-step use

1. Collect fixed traces.
2. Collect random traces.
3. Run analysis script.
4. Compare `plots/raw_comparison.png`.
5. Inspect filter-specific CSVs in `filtered/fixed/` and `filtered/random/`.
6. Report Welch t-test from `summary.json`.
