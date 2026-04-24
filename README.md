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
./scripts/collect_fixed.sh 1000 50
./scripts/collect_random.sh 1000 50
```

- First optional argument is trace count (default `100`).
- Second optional argument is samples per trace passed to `powermetrics -n` (default `50`).
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
  - `fixed_trace_summary.csv` (one row per fixed trace)
  - `random_trace_summary.csv` (one row per random trace)
  - `fixed_traces_aligned.csv` (all fixed traces, one row per trace)
  - `random_traces_aligned.csv` (all random traces, one row per trace)
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
  - `trace_means_all_traces.png` (shows all traces, e.g. all 1000 points)
  - `migration_events_per_trace.png`
- `summary.json`

## 5) Filter details

- **Moving average**: smooths short spikes.
- **Median filter**: robust against outlier samples.
- **Low-pass Butterworth**: removes high-frequency noise.

## 6) Task migration detection

The analyzer now includes a heuristic detector for migration-like events:
- It computes first-derivative jumps (`diff(trace)`).
- It marks large robust outliers using a MAD-based z-score threshold.
- It reports candidate events per trace into `raw/migration_events.csv`.

This is a signal-level heuristic (not scheduler ground truth), useful for
flagging traces likely to contain task migration or abrupt execution changes.

## 7) How to do the t-test (step by step)

After running:

```bash
python3 scripts/analyze_traces.py
```

check these files in the latest `results/analysis_<timestamp>/`:

1. **Overall Welch t-test** (single statistic on per-trace means):
   - `summary.json` → `welch_t_test_on_trace_means`
2. **Point-wise Welch t-test** (TVLA style across time samples):
   - `raw/pointwise_t_stat.csv`
   - `raw/pointwise_p_value.csv`
   - `plots/pointwise_t_stat.png`
3. **Threshold check**:
   - `summary.json` → `welch_t_test_pointwise.samples_exceeding_threshold`
   - default threshold is `|t| >= 4.5`

Interpretation:
- If many points exceed `|t| >= 4.5`, fixed/random traces are likely distinguishable.
- If almost none exceed the threshold, leakage evidence is weaker.
- If you collect `1000` traces, `fixed_trace_summary.csv` and
  `random_trace_summary.csv` should each contain `1000` rows (plus header).

## 8) Research-friendly step-by-step use

1. Collect fixed traces.
2. Collect random traces.
3. Run analysis script.
4. Compare `plots/raw_comparison.png`.
5. Inspect all-trace files: `raw/fixed_trace_summary.csv`, `raw/random_trace_summary.csv`,
   and `plots/trace_means_all_traces.png`.
6. Inspect filter-specific CSVs in `filtered/fixed/` and `filtered/random/`.
7. Inspect migration candidates:
   - `raw/migration_events.csv` (event index + power jump per trace)
   - `plots/migration_events_per_trace.png`
   - `summary.json` → `task_migration_detection`
8. Report Welch t-test values and migration summary from `summary.json`.
