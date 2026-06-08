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

ELAPSED_TIME_PATTERN = re.compile(
    r"\*\*\* Sampled system activity .* \(([0-9.]+)(ms|s) elapsed\) \*\*\*",
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

def parse_elapsed_time(path: Path) -> np.ndarray:
    values = []
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        match = ELAPSED_TIME_PATTERN.search(line)
        if match:
            val = float(match.group(1))
            unit = match.group(2).lower()
            if unit == 's':
                val *= 1000.0
            values.append(val)
    if not values:
        raise ValueError(f"No elapsed time values found in {path}")
    return np.array(values, dtype=float)

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


def load_frequency_traces(folder: Path) -> list[np.ndarray]:

    freq_traces = []

    for trace_path in sorted(folder.glob("trace_*.txt")):

        try:
            freq_traces.append(
                parse_frequency_trace(trace_path)
            )
        except ValueError:
            continue

    if not freq_traces:
        raise RuntimeError(
            f"No frequency traces in {folder}"
        )

    return freq_traces

def load_elapsed_traces(folder: Path) -> list[np.ndarray]:
    elapsed_traces = []
    for trace_path in sorted(folder.glob("trace_*.txt")):
        try:
            elapsed_traces.append(parse_elapsed_time(trace_path))
        except ValueError:
            continue
    if not elapsed_traces:
        raise RuntimeError(f"No elapsed times in {folder}")
    return elapsed_traces

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

    if window_length < 3:
        window_length = 3

    if window_length % 2 == 0:
        window_length += 1

    if len(signal) <= window_length:
        window_length = max(3, len(signal) - 1)
        if window_length % 2 == 0:
            window_length -= 1

    if window_length <= polyorder or window_length < 3:
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
    freq_signal: np.ndarray | None = None,
    lowpass_cutoff: float = 0.2,
    savgol_window: int = 11,
    median_window: int = 5,
    moving_average_window: int = 5
):

    results = {

        "raw":
            signal,

        "moving_average":
            moving_average(signal, window=moving_average_window),

        "median":
            medfilt(signal, kernel_size=median_window),

        "lowpass":
            lowpass(
                signal,
                cutoff_ratio=lowpass_cutoff
            ),

        "savitzky_golay":
            savgol_denoise(
                signal,
                window_length=savgol_window
            ),

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


def _smoothness_objective(
    raw: np.ndarray,
    filtered: np.ndarray,
    fidelity_weight: float = 0.15
) -> float:

    min_len = min(len(raw), len(filtered))

    raw = raw[:min_len]
    filtered = filtered[:min_len]

    roughness = np.std(np.diff(filtered))

    fidelity_penalty = np.sqrt(
        np.mean((raw - filtered) ** 2)
    )

    return float(
        roughness + fidelity_weight * fidelity_penalty
    )


def tune_filter_params(
    signal: np.ndarray
) -> dict[str, float | int]:

    cutoff_candidates = [
        0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35
    ]

    window_candidates = [
        5, 7, 9, 11, 13, 15, 17
    ]

    best_cutoff = cutoff_candidates[0]
    best_cutoff_score = float("inf")

    for cutoff in cutoff_candidates:

        try:
            filtered = lowpass(
                signal,
                cutoff_ratio=cutoff
            )
            score = _smoothness_objective(
                signal,
                filtered
            )
        except ValueError:
            continue

        if score < best_cutoff_score:
            best_cutoff_score = score
            best_cutoff = cutoff

    best_window = window_candidates[0]
    best_window_score = float("inf")

    for window in window_candidates:

        filtered = savgol_denoise(
            signal,
            window_length=window
        )

        score = _smoothness_objective(
            signal,
            filtered
        )

        if score < best_window_score:
            best_window_score = score
            best_window = window

    return {
        "lowpass_cutoff": best_cutoff,
        "savgol_window": best_window,
    }

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


def compute_power_snr(
    fixed: np.ndarray,
    random: np.ndarray
) -> float:

    common_len = min(
        fixed.shape[1],
        random.shape[1]
    )

    fixed = fixed[:, :common_len]
    random = random[:, :common_len]

    signal = np.mean(random, axis=0) - np.mean(fixed, axis=0)

    noise = (
        np.std(fixed, axis=0)
        +
        np.std(random, axis=0)
    )

    ratios = np.divide(
        signal,
        noise,
        out=np.zeros_like(signal, dtype=float),
        where=noise != 0
    )

    return float(np.mean(ratios))


def compute_mean_power_difference(
    fixed: np.ndarray,
    random: np.ndarray
) -> float:

    common_len = min(
        fixed.shape[1],
        random.shape[1]
    )

    fixed = fixed[:, :common_len]
    random = random[:, :common_len]

    signal = np.mean(random, axis=0) - np.mean(fixed, axis=0)

    return float(np.mean(np.abs(signal)))


def tvla_quantitative_metrics(
    t_stat: np.ndarray,
    fixed: np.ndarray,
    random: np.ndarray,
    threshold: float = 4.5
) -> dict[str, float | int]:

    abs_t = np.abs(t_stat)
    samples = len(abs_t)
    samples_exceeding = int(
        np.sum(abs_t >= threshold)
    )

    if samples == 0:
        exceedance_rate = 0.0
        max_abs_t = 0.0
    else:
        exceedance_rate = samples_exceeding / samples
        max_abs_t = float(np.max(abs_t))

    return {
        "samples":
            samples,
        "samples_exceeding_threshold":
            samples_exceeding,
        "exceedance_rate":
            float(exceedance_rate),
        "exceedance_percent":
            float(exceedance_rate * 100.0),
        "max_abs_t_statistic":
            max_abs_t,
        "power_snr":
            compute_power_snr(fixed, random),
        "mean_power_difference_mw":
            compute_mean_power_difference(fixed, random),
    }

# =========================================================
# DECISION MATRIX
# =========================================================


def load_summary_metrics(
    path: Path,
    experiment: str = "regression_residual"
) -> dict[str, float | int]:

    summary = json.loads(
        path.read_text()
    )

    return summary["quantitative_metrics"][experiment]


def compare_filter_metrics(
    quantitative_metrics: dict[str, dict[str, float | int]]
) -> dict[str, dict[str, float | str]]:

    raw_rate = float(
        quantitative_metrics["raw"]["exceedance_rate"]
    )
    comparisons = {}

    for experiment in [
        "median",
        "moving_average",
        "wavelet",
        "regression_residual",
        "savitzky_golay"
    ]:

        filtered_rate = float(
            quantitative_metrics[experiment]["exceedance_rate"]
        )
        delta = filtered_rate - raw_rate

        if delta > 0:
            decision = (
                "filter isolated leakage from OS noise; "
                "standard TVLA benefits from this enhancement"
            )
        elif delta < 0:
            decision = (
                "filter reduced the detectable leakage; "
                "it may be removing signal with noise"
            )
        else:
            decision = (
                "filter did not change the exceedance rate"
            )

        comparisons[experiment] = {
            "raw_exceedance_rate":
                raw_rate,
            "filtered_exceedance_rate":
                filtered_rate,
            "delta_exceedance_rate":
                float(delta),
            "decision":
                decision,
        }

    return comparisons


def migration_alignment_metrics(
    t_stat: np.ndarray,
    fixed_profile: np.ndarray,
    random_profile: np.ndarray
) -> dict[str, float | int]:

    common_len = min(
        len(t_stat),
        len(fixed_profile),
        len(random_profile)
    )

    if common_len <= 1:
        return {
            "samples":
                common_len,
            "correlation_abs_t_vs_migration_gap":
                0.0,
            "max_migration_rate_gap":
                0.0,
        }

    abs_t = np.abs(t_stat[:common_len])
    migration_gap = np.abs(
        fixed_profile[:common_len]
        -
        random_profile[:common_len]
    )

    if np.std(abs_t) == 0 or np.std(migration_gap) == 0:
        correlation = 0.0
    else:
        correlation = float(
            np.corrcoef(abs_t, migration_gap)[0, 1]
        )

    return {
        "samples":
            common_len,
        "correlation_abs_t_vs_migration_gap":
            correlation,
        "max_migration_rate_gap":
            float(np.max(migration_gap)),
    }


def build_decision_matrix(
    quantitative_metrics: dict[str, dict[str, float | int]],
    migration_alignment: dict[str, float | int],
    control_run: bool = False,
    core_mode: str = "unknown",
    pinned_summary: Path | None = None,
    unpinned_summary: Path | None = None,
    ecore_summary: Path | None = None,
    pcore_summary: Path | None = None
) -> dict:

    raw_metrics = quantitative_metrics["raw"]
    matrix = {
        "decision_1_environment_control": {
            "applicable":
                control_run,
            "rule":
                (
                    "If exceedance rate > 1-2% and max |t| > 4.5, "
                    "the environment is too noisy. If max |t| < 4.5, "
                    "the environment is valid."
                ),
        },
        "decision_2_filter_rq5": {
            "comparisons":
                compare_filter_metrics(quantitative_metrics),
        },
        "decision_3_data_vs_migration_rq1_rq4": {
            "core_mode":
                core_mode,
            "migration_alignment":
                migration_alignment,
            "rule":
                (
                    "Compare pinned vs unpinned summaries. If pinned TVLA "
                    "exceedance drops to ~0%, migration was the source. "
                    "If pinned TVLA still detects leakage, data processing "
                    "leaks independently of migration."
                ),
        },
        "decision_4_big_vs_little_rq3": {
            "rule":
                (
                    "Compare E-core vs P-core summaries. If SNR(E-core) > "
                    "SNR(P-core), LITTLE cores are more vulnerable. If "
                    "Max_t(P-core) > Max_t(E-core), big cores leak more "
                    "absolute power."
                ),
        },
    }

    if control_run:
        raw_exceedance_rate = float(
            raw_metrics["exceedance_rate"]
        )
        raw_max_t = float(
            raw_metrics["max_abs_t_statistic"]
        )

        if raw_exceedance_rate > 0.02 and raw_max_t > 4.5:
            verdict = (
                "environment too noisy; reduce OS jitter or increase traces"
            )
        elif raw_max_t < 4.5:
            verdict = (
                "environment valid; proceed to fixed-vs-random analysis"
            )
        else:
            verdict = (
                "borderline control; inspect trace count and background load"
            )

        matrix["decision_1_environment_control"].update({
            "raw_exceedance_rate":
                raw_exceedance_rate,
            "raw_max_abs_t_statistic":
                raw_max_t,
            "verdict":
                verdict,
        })
    else:
        matrix["decision_1_environment_control"]["verdict"] = (
            "not evaluated; rerun with --control-run on fixed-vs-fixed "
            "or random-vs-random traces"
        )

    if pinned_summary is not None and unpinned_summary is not None:
        pinned = load_summary_metrics(pinned_summary)
        unpinned = load_summary_metrics(unpinned_summary)
        pinned_rate = float(pinned["exceedance_rate"])
        unpinned_rate = float(unpinned["exceedance_rate"])
        pinned_max_t = float(pinned["max_abs_t_statistic"])

        if pinned_rate <= 0.01 and unpinned_rate > pinned_rate:
            verdict = (
                "leakage primarily caused by OS scheduler migration"
            )
        elif pinned_rate > 0.01 or pinned_max_t > 4.5:
            verdict = (
                "data-dependent power remains visible when pinned"
            )
        else:
            verdict = (
                "pinned and unpinned comparison is inconclusive"
            )

        matrix["decision_3_data_vs_migration_rq1_rq4"].update({
            "pinned_exceedance_rate":
                pinned_rate,
            "unpinned_exceedance_rate":
                unpinned_rate,
            "pinned_max_abs_t_statistic":
                pinned_max_t,
            "verdict":
                verdict,
        })
    else:
        matrix["decision_3_data_vs_migration_rq1_rq4"]["verdict"] = (
            "requires --pinned-summary and --unpinned-summary for final "
            "scheduler-vs-data conclusion"
        )

    if ecore_summary is not None and pcore_summary is not None:
        ecore = load_summary_metrics(ecore_summary)
        pcore = load_summary_metrics(pcore_summary)
        ecore_snr = float(ecore["power_snr"])
        pcore_snr = float(pcore["power_snr"])
        ecore_max_t = float(ecore["max_abs_t_statistic"])
        pcore_max_t = float(pcore["max_abs_t_statistic"])

        matrix["decision_4_big_vs_little_rq3"].update({
            "ecore_snr":
                ecore_snr,
            "pcore_snr":
                pcore_snr,
            "ecore_max_abs_t_statistic":
                ecore_max_t,
            "pcore_max_abs_t_statistic":
                pcore_max_t,
            "snr_verdict":
                (
                    "LITTLE/E-cores have higher SNR"
                    if ecore_snr > pcore_snr
                    else "P-cores have equal or higher SNR"
                ),
            "max_t_verdict":
                (
                    "Big/P-cores leak more absolute power"
                    if pcore_max_t > ecore_max_t
                    else "E-cores have equal or higher max |t|"
                ),
        })
    else:
        matrix["decision_4_big_vs_little_rq3"]["verdict"] = (
            "requires --ecore-summary and --pcore-summary for final "
            "big-vs-LITTLE comparison"
        )

    return matrix

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


def save_metrics_csv(
    path: Path,
    metrics: dict[str, dict[str, float | int]]
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "experiment",
        "samples",
        "samples_exceeding_threshold",
        "exceedance_rate",
        "exceedance_percent",
        "max_abs_t_statistic",
        "power_snr",
        "mean_power_difference_mw",
    ]

    with path.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()

        for experiment, values in metrics.items():

            row = {
                "experiment":
                    experiment,
            }
            row.update(values)
            writer.writerow(row)


def _comparison_row(
    condition: str,
    summary: dict,
    experiment: str,
    insight: str
) -> dict[str, str | float]:

    metrics = summary["quantitative_metrics"][experiment]

    return {
        "experiment_condition":
            condition,
        "max_abs_t_statistic":
            float(metrics["max_abs_t_statistic"]),
        "tvla_exceedance_rate_percent":
            float(metrics["exceedance_percent"]),
        "mean_power_difference_mw":
            float(metrics["mean_power_difference_mw"]),
        "decision_insight":
            insight,
    }


def build_thesis_comparison_table(
    current_summary: dict,
    control_summary: dict | None = None,
    baseline_summary: dict | None = None,
    ecore_summary: dict | None = None,
    pcore_summary: dict | None = None
) -> list[dict[str, str | float]]:

    rows = []

    control_source = control_summary
    if current_summary.get("decision_matrix", {}).get(
        "decision_1_environment_control", {}
    ).get("applicable"):
        control_source = current_summary

    baseline_source = baseline_summary
    current_core_mode = current_summary.get(
        "decision_matrix", {}
    ).get(
        "decision_3_data_vs_migration_rq1_rq4", {}
    ).get("core_mode")

    if baseline_source is None and current_core_mode in [
        "unknown",
        "unpinned"
    ]:
        baseline_source = current_summary

    if ecore_summary is None and current_core_mode == "ecore":
        ecore_summary = current_summary

    if pcore_summary is None and current_core_mode == "pcore":
        pcore_summary = current_summary

    if control_source is not None:
        rows.append(_comparison_row(
            "1. Control: Fixed vs Fixed",
            control_source,
            "raw",
            "Validates setup / estimates false positives"
        ))

    if baseline_source is not None:
        rows.append(_comparison_row(
            "2. Baseline: Fixed vs Random (Unpinned)",
            baseline_source,
            "raw",
            "OS scheduling may amplify leakage"
        ))

    if ecore_summary is not None:
        rows.append(_comparison_row(
            "3. Isolated: Fixed vs Random (Pinned E-core)",
            ecore_summary,
            "raw",
            "Pure data leakage without migration pressure on E-cores"
        ))

    if pcore_summary is not None:
        rows.append(_comparison_row(
            "4. Isolated: Fixed vs Random (Pinned P-core)",
            pcore_summary,
            "raw",
            "Big-core leakage under pinned/core-mode collection"
        ))

    enhanced_source = baseline_source or current_summary
    rows.append(_comparison_row(
        "5. Enhanced: Fixed vs Random (Wavelet TVLA)",
        enhanced_source,
        "wavelet",
        "Shows whether wavelet filtering improves leakage detection"
    ))

    return rows


def save_comparison_table_csv(
    path: Path,
    rows: list[dict[str, str | float]]
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "experiment_condition",
        "max_abs_t_statistic",
        "tvla_exceedance_rate_percent",
        "mean_power_difference_mw",
        "decision_insight",
    ]

    with path.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_table_markdown(
    path: Path,
    rows: list[dict[str, str | float]]
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lines = [
        "| Experiment Condition | Max |t-stat| | TVLA Exceedance Rate (%) | Mean Power Difference (mW) | Decision / Insight |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]

    for row in rows:

        lines.append(
            "| {condition} | {max_t:.4f} | {rate:.4f} | {power:.4f} | {insight} |".format(
                condition=row["experiment_condition"],
                max_t=float(row["max_abs_t_statistic"]),
                rate=float(row["tvla_exceedance_rate_percent"]),
                power=float(row["mean_power_difference_mw"]),
                insight=row["decision_insight"]
            )
        )

    path.write_text(
        "\n".join(lines) + "\n"
    )

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

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(path)

    plt.close()


def average_migration_profile(
    aligned_traces: np.ndarray
) -> np.ndarray:

    if aligned_traces.size == 0:
        return np.array([], dtype=float)

    profile_len = aligned_traces.shape[1] - 1

    if profile_len <= 0:
        return np.array([], dtype=float)

    migration_hits = np.zeros(
        profile_len,
        dtype=float
    )

    for trace in aligned_traces:

        event_idxs = detect_migration_events(trace)

        for idx in event_idxs:

            if 0 <= idx < profile_len:
                migration_hits[idx] += 1.0

    return migration_hits / aligned_traces.shape[0]


def plot_migration_effect(
    path: Path,
    fixed_profile: np.ndarray,
    random_profile: np.ndarray
):

    common_len = min(
        len(fixed_profile),
        len(random_profile)
    )

    fixed_profile = fixed_profile[:common_len]
    random_profile = random_profile[:common_len]

    x = np.arange(common_len)

    plt.figure(figsize=(12, 5))
    plt.plot(
        x,
        fixed_profile,
        marker="o",
        linestyle="-",
        alpha=0.85,
        label="fixed avg migration rate"
    )
    plt.plot(
        x,
        random_profile,
        marker="o",
        linestyle="-",
        alpha=0.85,
        label="random avg migration rate"
    )

    plt.title("Migration Effect per Sample Index (Average Across Traces)")
    plt.xlabel("Sample Index")
    plt.ylabel("Average Migration Event Rate")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    path.parent.mkdir(parents=True, exist_ok=True)
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


def plot_tvla_migration_overlay(
    path: Path,
    t_stat: np.ndarray,
    fixed_profile: np.ndarray,
    random_profile: np.ndarray
):

    tvla_len = len(t_stat)
    migration_len = min(
        len(fixed_profile),
        len(random_profile)
    )
    common_len = min(tvla_len, migration_len)

    if common_len <= 0:
        return

    t_abs = np.abs(t_stat[:common_len])
    fixed_profile = fixed_profile[:common_len]
    random_profile = random_profile[:common_len]
    migration_delta = np.abs(
        fixed_profile - random_profile
    )
    x = np.arange(common_len)

    fig, ax_tvla = plt.subplots(
        figsize=(12, 5)
    )

    line_tvla = ax_tvla.plot(
        x,
        t_abs,
        color="tab:blue",
        label="|t-statistic|"
    )[0]
    thr_tvla = ax_tvla.axhline(
        4.5,
        color="tab:blue",
        linestyle="--",
        alpha=0.8,
        label="TVLA threshold (4.5)"
    )
    ax_tvla.set_xlabel("Sample Index")
    ax_tvla.set_ylabel(
        "|t-statistic|",
        color="tab:blue"
    )
    ax_tvla.tick_params(
        axis="y",
        labelcolor="tab:blue"
    )

    ax_migration = ax_tvla.twinx()
    line_fixed = ax_migration.plot(
        x,
        fixed_profile,
        color="tab:green",
        alpha=0.8,
        label="fixed migration rate"
    )[0]
    line_random = ax_migration.plot(
        x,
        random_profile,
        color="tab:orange",
        alpha=0.8,
        label="random migration rate"
    )[0]
    line_delta = ax_migration.plot(
        x,
        migration_delta,
        color="tab:red",
        alpha=0.8,
        linestyle=":",
        label="|fixed-random| migration gap"
    )[0]
    ax_migration.set_ylabel(
        "Migration Event Rate",
        color="tab:red"
    )
    ax_migration.tick_params(
        axis="y",
        labelcolor="tab:red"
    )

    handles = [
        line_tvla,
        thr_tvla,
        line_fixed,
        line_random,
        line_delta
    ]
    labels = [
        h.get_label()
        for h in handles
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=True
    )
    plt.title(
        "Overlay: TVLA vs Migration Rate"
    )
    fig.subplots_adjust(bottom=0.24)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    plt.savefig(
        path,
        bbox_inches="tight"
    )
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
        "--all-traces",
        action="store_true",
        help="Analyze all traces found in data-root instead of just the latest pair."
    )

    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results")
    )

    p.add_argument(
        "--control-summary",
        type=Path,
        help="summary.json from a fixed-vs-fixed or random-vs-random control run."
    )

    p.add_argument(
        "--baseline-summary",
        type=Path,
        help="summary.json from an unpinned fixed-vs-random baseline run."
    )

    p.add_argument(
        "--control-run",
        action="store_true",
        help="Interpret this run as a fixed-vs-fixed or random-vs-random control."
    )

    p.add_argument(
        "--core-mode",
        choices=["unknown", "unpinned", "ecore", "pcore"],
        default="unknown",
        help="Annotate this analysis with the collection core mode."
    )

    p.add_argument(
        "--pinned-summary",
        type=Path,
        help="summary.json from a pinned/core-restricted run for Decision 3."
    )

    p.add_argument(
        "--unpinned-summary",
        type=Path,
        help="summary.json from an unpinned run for Decision 3."
    )

    p.add_argument(
        "--ecore-summary",
        type=Path,
        help="summary.json from an E-core run for Decision 4."
    )

    p.add_argument(
        "--pcore-summary",
        type=Path,
        help="summary.json from a P-core run for Decision 4."
    )

    p.add_argument(
        "--median-window",
        type=int,
        default=5,
        help="Window size for median filter (must be odd, default: 5)"
    )

    p.add_argument(
        "--moving-average-window",
        type=int,
        default=5,
        help="Window size for moving average filter (default: 5)"
    )

    p.add_argument(
        "--savgol-window",
        type=int,
        help="Override auto-tuned window size for Savitzky-Golay filter (must be odd)"
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

    if args.all_traces:
        fixed_dirs = sorted(args.data_root.glob("fixed_*"))
        random_dirs = sorted(args.data_root.glob("random_*"))
        if not fixed_dirs or not random_dirs:
            raise RuntimeError("No datasets found for --all-traces.")
        
        fixed_traces_all = []
        fixed_freq_traces_all = []
        fixed_elapsed_traces_all = []
        for d in fixed_dirs:
            fixed_traces_all.extend(load_experiment(d, "fixed").traces)
            fixed_freq_traces_all.extend(load_frequency_traces(d))
            fixed_elapsed_traces_all.extend(load_elapsed_traces(d))
            
        random_traces_all = []
        random_freq_traces_all = []
        random_elapsed_traces_all = []
        for d in random_dirs:
            random_traces_all.extend(load_experiment(d, "random").traces)
            random_freq_traces_all.extend(load_frequency_traces(d))
            random_elapsed_traces_all.extend(load_elapsed_traces(d))

        fixed = ExperimentData(label="fixed", traces=fixed_traces_all)
        random = ExperimentData(label="random", traces=random_traces_all)
        
        fixed_freq_traces = fixed_freq_traces_all
        random_freq_traces = random_freq_traces_all
        fixed_elapsed_traces = fixed_elapsed_traces_all
        random_elapsed_traces = random_elapsed_traces_all
        
        fixed_freq = average_trace(fixed_freq_traces)
        random_freq = average_trace(random_freq_traces)

        dataset_timestamp = fixed_dirs[-1].name.replace("fixed_", "")
        folder_suffix = "_all"
    else:
        fixed_dir, random_dir = select_latest_pair(
            args.data_root
        )

        fixed = load_experiment(
            fixed_dir,
            "fixed"
        )

        random = load_experiment(
            random_dir,
            "random"
        )
        
        fixed_freq = average_frequency_trace(
            fixed_dir
        )
    
        random_freq = average_frequency_trace(
            random_dir
        )
    
        fixed_freq_traces = load_frequency_traces(
            fixed_dir
        )
        random_freq_traces = load_frequency_traces(
            random_dir
        )
        
        fixed_elapsed_traces = load_elapsed_traces(
            fixed_dir
        )
        random_elapsed_traces = load_elapsed_traces(
            random_dir
        )

        dataset_timestamp = fixed_dir.name.replace("fixed_", "")
        folder_suffix = ""

    fixed_input_value = "Unknown"
    
    if args.all_traces:
        if len(fixed_dirs) > 0:
            inputs_path = fixed_dirs[-1] / "inputs.txt"
            if inputs_path.exists():
                fixed_input_value = inputs_path.read_text().strip().split('\n')[0]
    else:
        inputs_path = fixed_dir / "inputs.txt"
        if inputs_path.exists():
            fixed_input_value = inputs_path.read_text().strip().split('\n')[0]

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

    print("Applying filters...")

    fixed_tuned = tune_filter_params(
        fixed_avg
    )
    random_tuned = tune_filter_params(
        random_avg
    )

    tuned_lowpass_cutoff = float(np.mean([
        fixed_tuned["lowpass_cutoff"],
        random_tuned["lowpass_cutoff"],
    ]))

    tuned_savgol_window = int(np.round(np.mean([
        fixed_tuned["savgol_window"],
        random_tuned["savgol_window"],
    ])))

    if tuned_savgol_window % 2 == 0:
        tuned_savgol_window += 1

    if args.savgol_window is not None:
        tuned_savgol_window = args.savgol_window

    fixed_filtered = apply_filters(
        fixed_avg,
        fixed_freq,
        lowpass_cutoff=tuned_lowpass_cutoff,
        savgol_window=tuned_savgol_window,
        median_window=args.median_window,
        moving_average_window=args.moving_average_window
    )

    random_filtered = apply_filters(
        random_avg,
        random_freq,
        lowpass_cutoff=tuned_lowpass_cutoff,
        savgol_window=tuned_savgol_window,
        median_window=args.median_window,
        moving_average_window=args.moving_average_window
    )

    print("Running TVLA...")

    t_stat_raw, p_val_raw = compute_tvla(
        fixed_aligned,
        random_aligned
    )

    print("Computing TVLA for Median and Moving Average...")

    fixed_median = align_traces([
        medfilt(t, kernel_size=args.median_window)
        for t in fixed.traces
    ])[:, :common_len]

    random_median = align_traces([
        medfilt(t, kernel_size=args.median_window)
        for t in random.traces
    ])[:, :common_len]

    t_stat_median, p_val_median = compute_tvla(
        fixed_median,
        random_median
    )

    fixed_moving_average = align_traces([
        moving_average(t, window=args.moving_average_window)
        for t in fixed.traces
    ])[:, :common_len]

    random_moving_average = align_traces([
        moving_average(t, window=args.moving_average_window)
        for t in random.traces
    ])[:, :common_len]

    t_stat_moving_average, p_val_moving_average = compute_tvla(
        fixed_moving_average,
        random_moving_average
    )

    fixed_wavelet = align_traces([
        wavelet_denoise(t)
        for t in fixed.traces
    ])[:, :common_len]

    random_wavelet = align_traces([
        wavelet_denoise(t)
        for t in random.traces
    ])[:, :common_len]

    t_stat_wavelet, p_val_wavelet = compute_tvla(
        fixed_wavelet,
        random_wavelet
    )

    fixed_savitzky_golay = align_traces([
        savgol_denoise(t, window_length=tuned_savgol_window)
        for t in fixed.traces
    ])[:, :common_len]

    random_savitzky_golay = align_traces([
        savgol_denoise(t, window_length=tuned_savgol_window)
        for t in random.traces
    ])[:, :common_len]

    t_stat_savitzky_golay, p_val_savitzky_golay = compute_tvla(
        fixed_savitzky_golay,
        random_savitzky_golay
    )

    fixed_residual_traces = []
    random_residual_traces = []

    for p_trace, f_trace in zip(
        fixed.traces,
        fixed_freq_traces
    ):
        residual, _ = regression_filter(
            p_trace,
            f_trace
        )
        fixed_residual_traces.append(residual)

    for p_trace, f_trace in zip(
        random.traces,
        random_freq_traces
    ):
        residual, _ = regression_filter(
            p_trace,
            f_trace
        )
        random_residual_traces.append(residual)

    fixed_residual = align_traces(
        fixed_residual_traces
    )
    random_residual = align_traces(
        random_residual_traces
    )

    residual_len = min(
        fixed_residual.shape[1],
        random_residual.shape[1]
    )
    fixed_residual = fixed_residual[:, :residual_len]
    random_residual = random_residual[:, :residual_len]

    t_stat_regression_residual, p_val_regression_residual = compute_tvla(
        fixed_residual,
        random_residual
    )

    quantitative_metrics = {
        "raw":
            tvla_quantitative_metrics(
                t_stat_raw,
                fixed_aligned,
                random_aligned
            ),

        "median":
            tvla_quantitative_metrics(
                t_stat_median,
                fixed_median,
                random_median
            ),

        "moving_average":
            tvla_quantitative_metrics(
                t_stat_moving_average,
                fixed_moving_average,
                random_moving_average
            ),

        "wavelet":
            tvla_quantitative_metrics(
                t_stat_wavelet,
                fixed_wavelet,
                random_wavelet
            ),

        "savitzky_golay":
            tvla_quantitative_metrics(
                t_stat_savitzky_golay,
                fixed_savitzky_golay,
                random_savitzky_golay
            ),

        "regression_residual":
            tvla_quantitative_metrics(
                t_stat_regression_residual,
                fixed_residual,
                random_residual
            ),
    }

    out = args.results_root / f"analysis_{dataset_timestamp}{folder_suffix}"

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
        t_stat_raw,
        "t_stat"
    )

    save_csv(
        out / "tvla_p_value.csv",
        p_val_raw,
        "p_value"
    )

    save_csv(
        out / "tvla_t_stat_median.csv",
        t_stat_median,
        "t_stat_median"
    )

    save_csv(
        out / "tvla_p_value_median.csv",
        p_val_median,
        "p_value_median"
    )

    save_csv(
        out / "tvla_t_stat_moving_average.csv",
        t_stat_moving_average,
        "t_stat_moving_average"
    )

    save_csv(
        out / "tvla_p_value_moving_average.csv",
        p_val_moving_average,
        "p_value_moving_average"
    )

    save_csv(
        out / "tvla_t_stat_wavelet.csv",
        t_stat_wavelet,
        "t_stat_wavelet"
    )

    save_csv(
        out / "tvla_p_value_wavelet.csv",
        p_val_wavelet,
        "p_value_wavelet"
    )

    save_csv(
        out / "tvla_t_stat_savitzky_golay.csv",
        t_stat_savitzky_golay,
        "t_stat_savitzky_golay"
    )

    save_csv(
        out / "tvla_p_value_savitzky_golay.csv",
        p_val_savitzky_golay,
        "p_value_savitzky_golay"
    )

    save_csv(
        out / "tvla_t_stat_regression_residual.csv",
        t_stat_regression_residual,
        "t_stat_regression_residual"
    )

    save_csv(
        out / "tvla_p_value_regression_residual.csv",
        p_val_regression_residual,
        "p_value_regression_residual"
    )

    save_metrics_csv(
        out / "quantitative_metrics.csv",
        quantitative_metrics
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
        t_stat_raw
    )

    plot_tvla(
        out / "plots/tvla_median.png",
        t_stat_median
    )

    plot_tvla(
        out / "plots/tvla_moving_average.png",
        t_stat_moving_average
    )

    plot_tvla(
        out / "plots/tvla_wavelet.png",
        t_stat_wavelet
    )

    plot_tvla(
        out / "plots/tvla_savitzky_golay.png",
        t_stat_savitzky_golay
    )

    plot_tvla(
        out / "plots/tvla_regression_residual.png",
        t_stat_regression_residual
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

    fixed_migration_profile = average_migration_profile(
        fixed_aligned
    )

    random_migration_profile = average_migration_profile(
        random_aligned
    )

    migration_alignment = migration_alignment_metrics(
        t_stat_regression_residual,
        fixed_migration_profile,
        random_migration_profile
    )

    decision_matrix = build_decision_matrix(
        quantitative_metrics,
        migration_alignment,
        control_run=args.control_run,
        core_mode=args.core_mode,
        pinned_summary=args.pinned_summary,
        unpinned_summary=args.unpinned_summary,
        ecore_summary=args.ecore_summary,
        pcore_summary=args.pcore_summary
    )

    plot_migration_effect(
        out / "plots/migration_effect.png",
        fixed_migration_profile,
        random_migration_profile
    )

    plot_tvla_migration_overlay(
        out / "plots/tvla_migration_overlay.png",
        t_stat_regression_residual,
        fixed_migration_profile,
        random_migration_profile
    )

    save_csv(
        out / "migration_fixed.csv",
        fixed_migration_profile,
        "migration_rate"
    )

    save_csv(
        out / "migration_random.csv",
        random_migration_profile,
        "migration_rate"
    )

    # Save Elapsed Times CSV
    try:
        elapsed_csv_path = out / "tvla_elapsed_times.csv"
        anomalies_csv_path = out / "tvla_elapsed_anomalies.csv"
        
        # We find the max length to create columns
        max_len = 0
        for e in fixed_elapsed_traces + random_elapsed_traces:
            max_len = max(max_len, len(e))
        
        headers = ["trace_type", "trace_index"] + [f"sample_{i}" for i in range(max_len)]
        
        anomalies_rows = [["trace_type", "trace_index", "sample_index", "elapsed_time_ms"]]
        
        with open(elapsed_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for idx, e in enumerate(fixed_elapsed_traces):
                row = ["fixed", idx] + list(e) + [""] * (max_len - len(e))
                writer.writerow(row)
                for s_idx, val in enumerate(e):
                    if val < 10.0 or val > 15.0:
                        anomalies_rows.append(["fixed", idx, s_idx, val])
                        
            for idx, e in enumerate(random_elapsed_traces):
                row = ["random", idx] + list(e) + [""] * (max_len - len(e))
                writer.writerow(row)
                for s_idx, val in enumerate(e):
                    if val < 10.0 or val > 15.0:
                        anomalies_rows.append(["random", idx, s_idx, val])
                        
        with open(anomalies_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(anomalies_rows)
            
    except Exception as ex:
        print(f"Warning: Could not save elapsed times CSV: {ex}")

    control_summary = (
        json.loads(args.control_summary.read_text())
        if args.control_summary is not None
        else None
    )
    baseline_summary = (
        json.loads(args.baseline_summary.read_text())
        if args.baseline_summary is not None
        else None
    )
    ecore_summary = (
        json.loads(args.ecore_summary.read_text())
        if args.ecore_summary is not None
        else None
    )
    pcore_summary = (
        json.loads(args.pcore_summary.read_text())
        if args.pcore_summary is not None
        else None
    )

    current_summary_for_comparison = {
        "quantitative_metrics":
            quantitative_metrics,
        "decision_matrix":
            decision_matrix,
    }

    thesis_comparison_table = build_thesis_comparison_table(
        current_summary_for_comparison,
        control_summary=control_summary,
        baseline_summary=baseline_summary,
        ecore_summary=ecore_summary,
        pcore_summary=pcore_summary
    )

    summary = {

        "fixed_input": fixed_input_value,

        "fixed_traces":
            len(fixed_aligned),

        "random_traces":
            len(random_aligned),

        "tvla_threshold":
            4.5,

        "samples_exceeding_threshold":
            quantitative_metrics["raw"][
                "samples_exceeding_threshold"
            ],

        "samples_exceeding_threshold_median":
            quantitative_metrics["median"][
                "samples_exceeding_threshold"
            ],

        "samples_exceeding_threshold_moving_average":
            quantitative_metrics["moving_average"][
                "samples_exceeding_threshold"
            ],

        "samples_exceeding_threshold_wavelet":
            quantitative_metrics["wavelet"][
                "samples_exceeding_threshold"
            ],

        "samples_exceeding_threshold_savitzky_golay":
            quantitative_metrics["savitzky_golay"][
                "samples_exceeding_threshold"
            ],

        "samples_exceeding_threshold_regression_residual":
            quantitative_metrics["regression_residual"][
                "samples_exceeding_threshold"
            ],

        "quantitative_metrics":
            quantitative_metrics,

        "decision_matrix":
            decision_matrix,

        "migration_alignment":
            migration_alignment,

        "thesis_comparison_table":
            thesis_comparison_table,

        "max_migration_rate_gap":
            float(
                np.max(np.abs(
                    fixed_migration_profile[:min(
                        len(fixed_migration_profile),
                        len(random_migration_profile)
                    )]
                    -
                    random_migration_profile[:min(
                        len(fixed_migration_profile),
                        len(random_migration_profile)
                    )]
                ))
            ),

        "mean_fixed_migration_events":
            float(np.mean(fixed_migrations)),

        "mean_random_migration_events":
            float(np.mean(random_migrations)),

        "auto_tuned_parameters": {
            "lowpass_cutoff_ratio":
                tuned_lowpass_cutoff,
            "savitzky_golay_window":
                tuned_savgol_window,
        },

        "filter_parameters": {
            "median_window": args.median_window,
            "moving_average_window": args.moving_average_window,
            "savitzky_golay_window": tuned_savgol_window,
            "savitzky_golay_auto_tuned": args.savgol_window is None,
        },
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    (out / "decision_matrix.json").write_text(
        json.dumps(decision_matrix, indent=2)
    )

    save_comparison_table_csv(
        out / "thesis_comparison_table.csv",
        thesis_comparison_table
    )

    save_comparison_table_markdown(
        out / "thesis_comparison_table.md",
        thesis_comparison_table
    )

    print()
    print("===================================")
    print("ANALYSIS COMPLETE")
    print("===================================")
    print(f"Results saved to:\n{out}")
    print("===================================")


if __name__ == "__main__":
    main()
