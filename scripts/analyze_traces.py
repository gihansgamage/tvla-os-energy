"""
Advanced TVLA-Style Power Trace Analysis Framework
==================================================

Features:
---------
✓ Automatic powermetrics parsing
✓ Fixed vs Random trace analysis
✓ Multiple filtering techniques
✓ Savitzky-Golay filtering
✓ Wavelet denoising
✓ Regression-based filtering
✓ FFT frequency analysis
✓ TVLA Welch t-test
✓ Task migration detection
✓ Migration-vs-leakage correlation
✓ Full CSV/plot export
✓ Reproducible experiment outputs

Author:
-------
Research Framework for OS-Level TVLA Leakage Analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pywt

from scipy.signal import (
    butter,
    filtfilt,
    medfilt,
    savgol_filter,
)

from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

# =========================================================
# REGEX PATTERNS
# =========================================================

POWER_PATTERNS = [
    re.compile(
        r"CPU Power[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)",
        re.IGNORECASE
    ),

    re.compile(
        r"Package Power[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)",
        re.IGNORECASE
    ),
]

FREQ_PATTERN = re.compile(
    r"CPU \d+ frequency:\s*([0-9]+)\s*MHz",
    re.IGNORECASE
)

# =========================================================
# DATA CLASS
# =========================================================

@dataclass
class ExperimentData:
    label: str
    traces: list[np.ndarray]

# =========================================================
# PARSERS
# =========================================================

def parse_trace_file(path: Path) -> np.ndarray:

    values = []

    text = path.read_text(errors="ignore")

    for line in text.splitlines():

        for pattern in POWER_PATTERNS:

            match = pattern.search(line)

            if match:

                value = float(match.group(1))
                unit = match.group(2).lower()

                if unit == "w":
                    value *= 1000.0

                values.append(value)
                break

    if not values:
        raise ValueError(f"No power values found in {path}")

    return np.array(values, dtype=float)


def parse_frequency_trace(path: Path) -> np.ndarray:

    freqs = []

    text = path.read_text(errors="ignore")

    for line in text.splitlines():

        match = FREQ_PATTERN.search(line)

        if match:
            freqs.append(float(match.group(1)))

    if not freqs:
        raise ValueError(f"No frequency values found in {path}")

    return np.array(freqs, dtype=float)

# =========================================================
# LOADING
# =========================================================

def load_experiment(folder: Path, label: str) -> ExperimentData:

    traces = []

    for trace_path in sorted(folder.glob("trace_*.txt")):

        try:
            traces.append(parse_trace_file(trace_path))

        except ValueError:
            continue

    if not traces:
        raise RuntimeError(f"No valid traces in {folder}")

    return ExperimentData(label=label, traces=traces)


def average_trace(traces: list[np.ndarray]) -> np.ndarray:

    min_len = min(len(t) for t in traces)

    aligned = np.array([
        t[:min_len]
        for t in traces
    ])

    return aligned.mean(axis=0)


def align_traces(traces: list[np.ndarray]) -> np.ndarray:

    min_len = min(len(t) for t in traces)

    return np.array([
        t[:min_len]
        for t in traces
    ])


def average_frequency_trace(folder: Path) -> np.ndarray:

    freq_traces = []

    for trace_path in sorted(folder.glob("trace_*.txt")):

        try:
            freq = parse_frequency_trace(trace_path)
            freq_traces.append(freq)

        except ValueError:
            continue

    if not freq_traces:
        raise RuntimeError(
            f"No frequency traces in {folder}"
        )

    return average_trace(freq_traces)

# =========================================================
# FILTERS
# =========================================================

def moving_average(
    signal: np.ndarray,
    window: int = 5
) -> np.ndarray:

    kernel = np.ones(window) / window

    return np.convolve(
        signal,
        kernel,
        mode="same"
    )


def lowpass(
    signal: np.ndarray,
    cutoff_ratio: float = 0.2,
    order: int = 3
) -> np.ndarray:

    b, a = butter(
        order,
        cutoff_ratio,
        btype="low"
    )

    return filtfilt(b, a, signal)


def savgol_denoise(
    signal: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> np.ndarray:

    if window_length % 2 == 0:
        window_length += 1

    if len(signal) <= window_length:
        return signal.copy()

    return savgol_filter(
        signal,
        window_length=window_length,
        polyorder=polyorder
    )


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 2
) -> np.ndarray:

    coeffs = pywt.wavedec(
        signal,
        wavelet,
        mode="per"
    )

    sigma = np.median(
        np.abs(coeffs[-1])
    ) / 0.6745

    threshold = sigma * np.sqrt(
        2 * np.log(len(signal))
    )

    denoised = [coeffs[0]]

    for c in coeffs[1:]:

        denoised.append(
            pywt.threshold(
                c,
                threshold,
                mode="soft"
            )
        )

    reconstructed = pywt.waverec(
        denoised,
        wavelet,
        mode="per"
    )

    return reconstructed[:len(signal)]


def regression_filter(
    power_signal: np.ndarray,
    freq_signal: np.ndarray
):

    min_len = min(
        len(power_signal),
        len(freq_signal)
    )

    y = power_signal[:min_len]

    X = freq_signal[:min_len].reshape(-1, 1)

    model = LinearRegression()

    model.fit(X, y)

    predicted = model.predict(X)

    residual = y - predicted

    return residual, predicted


def apply_filters(
    signal: np.ndarray,
    freq_signal: np.ndarray | None = None
):

    results = {

        "raw":
            signal,

        "moving_average":
            moving_average(signal),

        "median":
            medfilt(signal, kernel_size=5),

        "lowpass":
            lowpass(signal),

        "savitzky_golay":
            savgol_denoise(signal),

        "wavelet":
            wavelet_denoise(signal),
    }

    if freq_signal is not None:

        residual, predicted = regression_filter(
            signal,
            freq_signal
        )

        results["regression_residual"] = residual
        results["regression_predicted"] = predicted

    return results

# =========================================================
# TVLA
# =========================================================

def compute_tvla(
    fixed: np.ndarray,
    random: np.ndarray
):

    t_stat, p_val = ttest_ind(
        fixed,
        random,
        axis=0,
        equal_var=False
    )

    return t_stat, p_val

# =========================================================
# MIGRATION DETECTION
# =========================================================

def detect_migration_events(
    trace: np.ndarray,
    z_threshold: float = 3.5
):

    diffs = np.diff(trace)

    mad = np.median(
        np.abs(diffs - np.median(diffs))
    )

    scale = 1.4826 * mad

    if scale == 0:
        return []

    z = np.abs(diffs) / scale

    idxs = np.where(z >= z_threshold)[0]

    return idxs.tolist()

# =========================================================
# FFT
# =========================================================

def frequency_spectrum(signal):

    centered = signal - np.mean(signal)

    fft = np.fft.rfft(centered)

    freqs = np.fft.rfftfreq(
        len(centered),
        d=1.0
    )

    return freqs, np.abs(fft)

# =========================================================
# CSV SAVING
# =========================================================

def save_csv(
    path: Path,
    values: Iterable[float],
    header: str
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open("w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "index",
            header
        ])

        for i, v in enumerate(values):

            writer.writerow([
                i,
                float(v)
            ])

# =========================================================
# PLOTTING
# =========================================================

def plot_signals(
    path: Path,
    title: str,
    series: dict
):

    plt.figure(figsize=(12, 5))

    for name, arr in series.items():

        plt.plot(arr, label=name)

    plt.title(title)

    plt.xlabel("Sample")

    plt.ylabel("Power (mW)")

    plt.legend()

    plt.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(path)

    plt.close()


def plot_tvla(
    path: Path,
    t_stat: np.ndarray
):

    plt.figure(figsize=(12, 5))

    plt.plot(np.abs(t_stat))

    plt.axhline(
        4.5,
        linestyle="--"
    )

    plt.title("TVLA |t-statistic|")

    plt.xlabel("Sample")

    plt.ylabel("|t|")

    plt.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(path)

    plt.close()

# =========================================================
# ARGUMENTS
# =========================================================

def build_parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data")
    )

    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results")
    )

    return p

# =========================================================
# FIND DATASETS
# =========================================================

def select_latest_pair(data_root: Path):

    fixed_dirs = sorted(
        data_root.glob("fixed_*")
    )

    random_dirs = sorted(
        data_root.glob("random_*")
    )

    if not fixed_dirs or not random_dirs:

        raise RuntimeError(
            "No fixed/random datasets found."
        )

    return fixed_dirs[-1], random_dirs[-1]

# =========================================================
# MAIN
# =========================================================

def main():

    args = build_parser().parse_args()

    fixed_dir, random_dir = select_latest_pair(
        args.data_root
    )

    print("Loading traces...")

    fixed = load_experiment(
        fixed_dir,
        "fixed"
    )

    random = load_experiment(
        random_dir,
        "random"
    )

    fixed_aligned = align_traces(
        fixed.traces
    )

    random_aligned = align_traces(
        random.traces
    )

    common_len = min(
        fixed_aligned.shape[1],
        random_aligned.shape[1]
    )

    fixed_aligned = fixed_aligned[:, :common_len]

    random_aligned = random_aligned[:, :common_len]

    fixed_avg = fixed_aligned.mean(axis=0)

    random_avg = random_aligned.mean(axis=0)

    fixed_freq = average_frequency_trace(
        fixed_dir
    )

    random_freq = average_frequency_trace(
        random_dir
    )

    print("Applying filters...")

    fixed_filtered = apply_filters(
        fixed_avg,
        fixed_freq
    )

    random_filtered = apply_filters(
        random_avg,
        random_freq
    )

    print("Running TVLA...")

    t_stat, p_val = compute_tvla(
        fixed_aligned,
        random_aligned
    )

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d_%H%M%S"
    )

    out = args.results_root / f"analysis_{timestamp}"

    # =====================================================
    # SAVE FILTERED SIGNALS
    # =====================================================

    for name, arr in fixed_filtered.items():

        save_csv(
            out / "filtered/fixed" / f"{name}.csv",
            arr,
            "power_mw"
        )

    for name, arr in random_filtered.items():

        save_csv(
            out / "filtered/random" / f"{name}.csv",
            arr,
            "power_mw"
        )

    # =====================================================
    # SAVE TVLA
    # =====================================================

    save_csv(
        out / "tvla_t_stat.csv",
        t_stat,
        "t_stat"
    )

    save_csv(
        out / "tvla_p_value.csv",
        p_val,
        "p_value"
    )

    # =====================================================
    # PLOTS
    # =====================================================

    plot_signals(
        out / "plots/fixed_filters.png",
        "Fixed Filters",
        fixed_filtered
    )

    plot_signals(
        out / "plots/random_filters.png",
        "Random Filters",
        random_filtered
    )

    plot_tvla(
        out / "plots/tvla.png",
        t_stat
    )

    # =====================================================
    # MIGRATION ANALYSIS
    # =====================================================

    fixed_migrations = [
        len(detect_migration_events(t))
        for t in fixed_aligned
    ]

    random_migrations = [
        len(detect_migration_events(t))
        for t in random_aligned
    ]

    summary = {

        "fixed_traces":
            len(fixed_aligned),

        "random_traces":
            len(random_aligned),

        "tvla_threshold":
            4.5,

        "samples_exceeding_threshold":
            int(np.sum(np.abs(t_stat) >= 4.5)),

        "mean_fixed_migration_events":
            float(np.mean(fixed_migrations)),

        "mean_random_migration_events":
            float(np.mean(random_migrations)),
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print("===================================")
    print("ANALYSIS COMPLETE")
    print("===================================")
    print(f"Results saved to:\n{out}")
    print("===================================")


if __name__ == "__main__":
    main()