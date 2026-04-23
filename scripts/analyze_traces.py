#!/usr/bin/env python3
"""Analyze power traces for fixed vs random TVLA-style experiments.

This script parses powermetrics outputs, applies multiple filters, and saves
all intermediate/final artifacts in separate folders for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import ttest_ind

POWER_PATTERNS = [
    re.compile(r"CPU Power[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)", re.IGNORECASE),
    re.compile(r"Package Power[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)", re.IGNORECASE),
]


@dataclass
class ExperimentData:
    label: str
    traces: list[np.ndarray]



def parse_trace_file(path: Path) -> np.ndarray:
    values: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        for pattern in POWER_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit == "w":
                value *= 1000.0
            values.append(value)
            break

    if not values:
        raise ValueError(f"No power values parsed from {path}")
    return np.array(values, dtype=float)



def load_experiment(folder: Path, label: str) -> ExperimentData:
    traces: list[np.ndarray] = []
    for trace_path in sorted(folder.glob("trace_*.txt")):
        try:
            traces.append(parse_trace_file(trace_path))
        except ValueError:
            continue

    if not traces:
        raise RuntimeError(f"No valid trace files were parsed in {folder}")

    return ExperimentData(label=label, traces=traces)



def average_trace(traces: list[np.ndarray]) -> np.ndarray:
    min_len = min(len(t) for t in traces)
    aligned = np.array([t[:min_len] for t in traces])
    return aligned.mean(axis=0)



def moving_average(signal: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return signal.copy()
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")



def lowpass(signal: np.ndarray, cutoff_ratio: float = 0.2, order: int = 3) -> np.ndarray:
    b, a = butter(order, cutoff_ratio, btype="low")
    return filtfilt(b, a, signal)



def apply_filters(signal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "raw": signal,
        "moving_average": moving_average(signal, window=5),
        "median": medfilt(signal, kernel_size=5),
        "lowpass": lowpass(signal, cutoff_ratio=0.2, order=3),
    }



def save_csv(path: Path, values: Iterable[float], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", header])
        for i, v in enumerate(values):
            writer.writerow([i, float(v)])



def plot_signals(path: Path, title: str, series: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(10, 5))
    for name, arr in series.items():
        plt.plot(arr, label=name)
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Power (mW)")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()



def select_latest_pair(data_root: Path) -> tuple[Path, Path]:
    fixed_dirs = sorted(data_root.glob("fixed_*"))
    random_dirs = sorted(data_root.glob("random_*"))
    if not fixed_dirs or not random_dirs:
        raise RuntimeError(
            f"Could not find fixed_*/random_* folders in {data_root}. "
            "Run collection scripts first."
        )
    return fixed_dirs[-1], random_dirs[-1]



def trace_means(traces: list[np.ndarray]) -> np.ndarray:
    return np.array([float(t.mean()) for t in traces], dtype=float)



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze TVLA power traces")
    p.add_argument("--fixed-dir", type=Path, help="Directory with fixed traces")
    p.add_argument("--random-dir", type=Path, help="Directory with random traces")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--results-root", type=Path, default=Path("results"))
    return p



def main() -> None:
    args = build_parser().parse_args()

    if args.fixed_dir and args.random_dir:
        fixed_dir, random_dir = args.fixed_dir, args.random_dir
    else:
        fixed_dir, random_dir = select_latest_pair(args.data_root)

    fixed = load_experiment(fixed_dir, "fixed")
    random = load_experiment(random_dir, "random")

    fixed_avg = average_trace(fixed.traces)
    random_avg = average_trace(random.traces)

    fixed_filtered = apply_filters(fixed_avg)
    random_filtered = apply_filters(random_avg)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = args.results_root / f"analysis_{timestamp}"

    # Save raw per-trace stats
    fixed_means = trace_means(fixed.traces)
    random_means = trace_means(random.traces)
    save_csv(out / "raw" / "fixed_trace_means.csv", fixed_means, "power_mw")
    save_csv(out / "raw" / "random_trace_means.csv", random_means, "power_mw")

    # Save each filtered signal separately
    for name, arr in fixed_filtered.items():
        save_csv(out / "filtered" / "fixed" / f"{name}.csv", arr, "power_mw")
    for name, arr in random_filtered.items():
        save_csv(out / "filtered" / "random" / f"{name}.csv", arr, "power_mw")

    # Save figures (shown separately + combined)
    plot_signals(out / "plots" / "fixed_filters.png", "Fixed traces: raw vs filters", fixed_filtered)
    plot_signals(
        out / "plots" / "random_filters.png", "Random traces: raw vs filters", random_filtered
    )
    plot_signals(
        out / "plots" / "raw_comparison.png",
        "Raw average comparison",
        {"fixed_raw": fixed_filtered["raw"], "random_raw": random_filtered["raw"]},
    )

    # TVLA-style Welch t-test on per-trace means
    t_stat, p_value = ttest_ind(fixed_means, random_means, equal_var=False)
    summary = {
        "fixed_dir": str(fixed_dir),
        "random_dir": str(random_dir),
        "trace_counts": {"fixed": len(fixed.traces), "random": len(random.traces)},
        "mean_power_mw": {
            "fixed": float(fixed_means.mean()),
            "random": float(random_means.mean()),
        },
        "welch_t_test_on_trace_means": {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Analysis complete. Results saved in: {out}")


if __name__ == "__main__":
    main()
