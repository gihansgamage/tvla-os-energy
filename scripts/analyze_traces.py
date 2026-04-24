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


def align_traces(traces: list[np.ndarray]) -> np.ndarray:
    """Trim traces to a common length and stack into a 2D array."""
    min_len = min(len(t) for t in traces)
    return np.array([t[:min_len] for t in traces], dtype=float)



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


def save_trace_matrix(path: Path, matrix: np.ndarray) -> None:
    """Save one row per trace so all collected traces are preserved in output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["trace_index"] + [f"sample_{i}" for i in range(matrix.shape[1])]
        writer.writerow(header)
        for idx, row in enumerate(matrix):
            writer.writerow([idx] + [float(x) for x in row])


def save_trace_summary(path: Path, traces: list[np.ndarray], label: str) -> None:
    """Save per-trace descriptive stats (one output row per trace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trace_index", "label", "num_samples", "mean_mw", "std_mw", "min_mw", "max_mw"])
        for idx, t in enumerate(traces):
            writer.writerow(
                [
                    idx,
                    label,
                    len(t),
                    float(np.mean(t)),
                    float(np.std(t)),
                    float(np.min(t)),
                    float(np.max(t)),
                ]
            )


def detect_migration_events(
    trace: np.ndarray, z_threshold: float = 3.5, min_gap: int = 2
) -> list[tuple[int, float]]:
    """
    Heuristic migration detector using large first-derivative spikes.

    Returns a list of (sample_index, delta_mw) candidate events.
    """
    if len(trace) < 3:
        return []

    diffs = np.diff(trace)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    robust_scale = 1.4826 * mad if mad > 0 else np.std(diffs)
    if robust_scale == 0:
        return []

    z = np.abs(diffs) / robust_scale
    candidate_idxs = np.where(z >= z_threshold)[0] + 1
    if len(candidate_idxs) == 0:
        return []

    merged: list[int] = [int(candidate_idxs[0])]
    for idx in candidate_idxs[1:]:
        if int(idx) - merged[-1] > min_gap:
            merged.append(int(idx))

    return [(idx, float(trace[idx] - trace[idx - 1])) for idx in merged]


def save_migration_report(
    path: Path,
    fixed_traces: np.ndarray,
    random_traces: np.ndarray,
    z_threshold: float,
) -> dict[str, object]:
    """
    Save per-trace migration candidates for both classes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed_counts: list[int] = []
    random_counts: list[int] = []

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "trace_index", "event_index", "delta_mw"])

        for label, traces, counts in (
            ("fixed", fixed_traces, fixed_counts),
            ("random", random_traces, random_counts),
        ):
            for trace_idx, trace in enumerate(traces):
                events = detect_migration_events(trace, z_threshold=z_threshold)
                counts.append(len(events))
                for event_idx, delta in events:
                    writer.writerow([label, trace_idx, event_idx, delta])

    def stats(values: list[int]) -> dict[str, float]:
        arr = np.array(values, dtype=float)
        return {
            "mean_events_per_trace": float(arr.mean()) if len(arr) else 0.0,
            "max_events_in_single_trace": float(arr.max()) if len(arr) else 0.0,
            "traces_with_any_event": int(np.sum(arr > 0)) if len(arr) else 0,
        }

    return {
        "z_threshold": z_threshold,
        "fixed": stats(fixed_counts),
        "random": stats(random_counts),
    }



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


def plot_trace_means(path: Path, fixed_means: np.ndarray, random_means: np.ndarray) -> None:
    """Plot one point per trace so user can inspect all traces (e.g., 1000 points)."""
    plt.figure(figsize=(11, 5))
    plt.plot(fixed_means, ".", alpha=0.7, label=f"fixed ({len(fixed_means)} traces)")
    plt.plot(random_means, ".", alpha=0.7, label=f"random ({len(random_means)} traces)")
    plt.title("Per-trace mean power (all collected traces)")
    plt.xlabel("Trace index")
    plt.ylabel("Mean power (mW)")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def plot_migration_counts(path: Path, fixed_traces: np.ndarray, random_traces: np.ndarray) -> None:
    fixed_counts = [len(detect_migration_events(t)) for t in fixed_traces]
    random_counts = [len(detect_migration_events(t)) for t in random_traces]

    plt.figure(figsize=(11, 5))
    plt.plot(fixed_counts, ".", alpha=0.7, label=f"fixed ({len(fixed_counts)} traces)")
    plt.plot(random_counts, ".", alpha=0.7, label=f"random ({len(random_counts)} traces)")
    plt.title("Detected migration-like events per trace")
    plt.xlabel("Trace index")
    plt.ylabel("Event count")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def migration_density_by_sample(traces: np.ndarray, z_threshold: float = 3.5) -> np.ndarray:
    """
    For each sample index, count how many traces have a migration-like event at that index.
    """
    if traces.size == 0:
        return np.array([], dtype=float)
    density = np.zeros(traces.shape[1], dtype=float)
    for t in traces:
        for idx, _ in detect_migration_events(t, z_threshold=z_threshold):
            if 0 <= idx < len(density):
                density[idx] += 1.0
    return density


def plot_tstat_vs_migration(
    path: Path,
    t_stat: np.ndarray,
    fixed_density: np.ndarray,
    random_density: np.ndarray,
    t_threshold: float = 4.5,
) -> None:
    """
    Overlay leakage indicator (|t|) with migration density to interpret confounding.
    """
    x = np.arange(len(t_stat))
    combined_density = fixed_density + random_density
    # Normalize to [0, 1] for visual comparison with |t|
    norm_density = combined_density / combined_density.max() if combined_density.max() > 0 else combined_density

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(x, np.abs(t_stat), label="|t-stat|", color="tab:blue")
    ax1.axhline(t_threshold, linestyle="--", color="tab:red", label=f"TVLA threshold {t_threshold}")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("|t-stat|")

    ax2 = ax1.twinx()
    ax2.plot(x, norm_density, color="tab:orange", alpha=0.8, label="Migration density (normalized)")
    ax2.set_ylabel("Normalized migration density")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Leakage indicator vs migration activity")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def migration_leakage_overlap_score(
    pointwise_t: np.ndarray, fixed_density: np.ndarray, random_density: np.ndarray, t_threshold: float = 4.5
) -> dict[str, float]:
    """
    Quantify overlap between high-|t| points and high migration-activity points.
    """
    abs_t = np.abs(pointwise_t)
    high_t_mask = abs_t >= t_threshold
    combined_density = fixed_density + random_density
    high_migration_mask = (
        combined_density >= np.percentile(combined_density, 75)
        if len(combined_density)
        else np.zeros_like(abs_t, dtype=bool)
    )

    if len(abs_t) == 0:
        return {"high_t_points": 0.0, "overlap_points": 0.0, "overlap_ratio": 0.0}

    overlap = np.logical_and(high_t_mask, high_migration_mask)
    high_t_points = int(np.sum(high_t_mask))
    overlap_points = int(np.sum(overlap))
    overlap_ratio = float(overlap_points / high_t_points) if high_t_points else 0.0
    return {
        "high_t_points": float(high_t_points),
        "overlap_points": float(overlap_points),
        "overlap_ratio": overlap_ratio,
    }



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

    fixed_aligned = align_traces(fixed.traces)
    random_aligned = align_traces(random.traces)

    common_len = min(fixed_aligned.shape[1], random_aligned.shape[1])
    fixed_aligned = fixed_aligned[:, :common_len]
    random_aligned = random_aligned[:, :common_len]

    fixed_avg = fixed_aligned.mean(axis=0)
    random_avg = random_aligned.mean(axis=0)

    fixed_filtered = apply_filters(fixed_avg)
    random_filtered = apply_filters(random_avg)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = args.results_root / f"analysis_{timestamp}"

    # Save raw per-trace stats
    fixed_means = trace_means(fixed.traces)
    random_means = trace_means(random.traces)
    save_csv(out / "raw" / "fixed_trace_means.csv", fixed_means, "power_mw")
    save_csv(out / "raw" / "random_trace_means.csv", random_means, "power_mw")
    save_trace_summary(out / "raw" / "fixed_trace_summary.csv", fixed.traces, "fixed")
    save_trace_summary(out / "raw" / "random_trace_summary.csv", random.traces, "random")
    save_trace_matrix(out / "raw" / "fixed_traces_aligned.csv", fixed_aligned)
    save_trace_matrix(out / "raw" / "random_traces_aligned.csv", random_aligned)

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
    plot_trace_means(out / "plots" / "trace_means_all_traces.png", fixed_means, random_means)
    plot_migration_counts(out / "plots" / "migration_events_per_trace.png", fixed_aligned, random_aligned)

    # Welch t-test on per-trace means (overall summary)
    t_stat, p_value = ttest_ind(fixed_means, random_means, equal_var=False)

    # TVLA-style point-wise Welch t-test across aligned trace samples
    pointwise_t, pointwise_p = ttest_ind(fixed_aligned, random_aligned, axis=0, equal_var=False)
    save_csv(out / "raw" / "pointwise_t_stat.csv", pointwise_t, "t_stat")
    save_csv(out / "raw" / "pointwise_p_value.csv", pointwise_p, "p_value")
    plot_signals(
        out / "plots" / "pointwise_t_stat.png",
        "Point-wise Welch t-statistic (fixed vs random)",
        {"t_stat": pointwise_t},
    )

    tvla_threshold = 4.5
    exceed_count = int(np.sum(np.abs(pointwise_t) >= tvla_threshold))
    fixed_mig_density = migration_density_by_sample(fixed_aligned, z_threshold=3.5)
    random_mig_density = migration_density_by_sample(random_aligned, z_threshold=3.5)
    save_csv(out / "raw" / "fixed_migration_density_by_sample.csv", fixed_mig_density, "event_count")
    save_csv(out / "raw" / "random_migration_density_by_sample.csv", random_mig_density, "event_count")
    plot_tstat_vs_migration(
        out / "plots" / "tstat_vs_migration_density.png",
        pointwise_t,
        fixed_mig_density,
        random_mig_density,
        t_threshold=tvla_threshold,
    )
    overlap_stats = migration_leakage_overlap_score(
        pointwise_t,
        fixed_mig_density,
        random_mig_density,
        t_threshold=tvla_threshold,
    )
    migration_summary = save_migration_report(
        out / "raw" / "migration_events.csv",
        fixed_aligned,
        random_aligned,
        z_threshold=3.5,
    )
    summary = {
        "fixed_dir": str(fixed_dir),
        "random_dir": str(random_dir),
        "trace_counts": {"fixed": len(fixed.traces), "random": len(random.traces)},
        "notes": {
            "trace_count_meaning": "Number of files parsed (trace_*.txt).",
            "sample_count_meaning": "Number of power samples inside each trace after alignment.",
        },
        "mean_power_mw": {
            "fixed": float(fixed_means.mean()),
            "random": float(random_means.mean()),
        },
        "welch_t_test_on_trace_means": {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        },
        "welch_t_test_pointwise": {
            "sample_count": int(common_len),
            "threshold_abs_t": tvla_threshold,
            "samples_exceeding_threshold": exceed_count,
        },
        "task_migration_detection": migration_summary,
        "migration_vs_tstat_interpretation": {
            "overlap_stats": overlap_stats,
            "rule_of_thumb": (
                "High overlap_ratio suggests t-stat peaks may be influenced by migration-related transients; "
                "low overlap_ratio suggests leakage-like effects are less correlated with migration events."
            ),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Analysis complete. Results saved in: {out}")


if __name__ == "__main__":
    main()