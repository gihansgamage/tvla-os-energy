#!/usr/bin/env python3
"""
control_tvla.py
===============
Calculate new TVLA Welch t-values between two Fixed value sets (Fixed vs Fixed)
or two Random value sets (Random vs Random) for Null-Hypothesis / Control analysis.

Requirements:
  - Both selected datasets must have the IDENTICAL trace volume (number of traces).
  - If trace volumes differ, an error is raised and execution stops.
  - After calculating new raw t-values, all filter techniques are applied:
      * Median Filter
      * Moving Average Filter
      * Wavelet Denoising (db4)
      * Savitzky-Golay Filter
      * Lowpass Butterworth Filter
  - Generates comprehensive plots and CSV matrices matching the standard dashboard format.

Usage:
    python3 scripts/control_tvla.py --mode fixed --dataset1 20261024_221251 --dataset2 20261024_233546
    python3 scripts/control_tvla.py --mode random --dataset1 20261024_221251 --dataset2 20261024_233546
    python3 scripts/control_tvla.py --interactive
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
import pywt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TVLA_THRESHOLD = 4.5

# ─── Filter Functions ─────────────────────────────────────────────────────────

def moving_average(signal: np.ndarray, window: int = 5) -> np.ndarray:
    if window < 1: window = 1
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def lowpass(signal: np.ndarray, cutoff_ratio: float = 0.2, order: int = 3) -> np.ndarray:
    try:
        b, a = butter(order, cutoff_ratio, btype="low")
        return filtfilt(b, a, signal)
    except Exception:
        return signal.copy()


def savgol_denoise(signal: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    if window_length < 3: window_length = 3
    if window_length % 2 == 0: window_length += 1
    if len(signal) <= window_length:
        window_length = max(3, len(signal) - 1)
        if window_length % 2 == 0: window_length -= 1
    if window_length <= polyorder:
        return signal.copy()
    try:
        return savgol_filter(signal, window_length=window_length, polyorder=polyorder)
    except Exception:
        return signal.copy()


def wavelet_denoise(signal: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    try:
        coeffs = pywt.wavedec(signal, wavelet, mode="per")
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        denoised = [coeffs[0]]
        for c in coeffs[1:]:
            denoised.append(pywt.threshold(c, threshold, mode="soft"))
        rec = pywt.waverec(denoised, wavelet, mode="per")
        return rec[:len(signal)]
    except Exception:
        return signal.copy()


# ─── Helpers ───────────────────────────────────────────────────────────────────

def resolve_analysis_dir(root: Path, identifier: str) -> Path:
    clean = identifier.strip()
    direct = root / clean
    if direct.is_dir(): return direct
    prefixed = root / f"analysis_{clean}"
    if prefixed.is_dir(): return prefixed
    matches = list(root.glob(f"*{clean}*"))
    for m in matches:
        if m.is_dir(): return m
    raise FileNotFoundError(f"Analysis dataset directory '{identifier}' not found in {root}")


def get_trace_count(analysis_dir: Path, side: str) -> int:
    summary_p = analysis_dir / "summary.json"
    if not summary_p.exists():
        raise FileNotFoundError(f"Missing summary.json in {analysis_dir}")
    with open(summary_p) as f:
        s = json.load(f)
    if side == "fixed":
        cnt = s.get("fixed_traces") or s.get("trace_counts", {}).get("fixed")
    else:
        cnt = s.get("random_traces") or s.get("trace_counts", {}).get("random")
    if cnt is None:
        raise ValueError(f"Could not determine {side} trace count in {analysis_dir.name}/summary.json")
    return int(cnt)


def load_raw_power(analysis_dir: Path, side: str) -> np.ndarray:
    raw_p = analysis_dir / "filtered" / side / "raw.csv"
    if not raw_p.exists():
        raise FileNotFoundError(f"Missing {side} raw.csv in {analysis_dir}/filtered/{side}/raw.csv")
    df = pd.read_csv(raw_p)
    if "power_mw" not in df.columns:
        raise ValueError(f"Column 'power_mw' not found in {raw_p}")
    return df["power_mw"].values.astype(float)


def compute_quantitative_metrics(t_stat: np.ndarray, diff_power: np.ndarray, name: str) -> dict:
    abs_t = np.abs(t_stat)
    samples = int(len(t_stat))
    exceed_cnt = int(np.sum(abs_t > TVLA_THRESHOLD))
    exceed_rate = float(exceed_cnt / samples) if samples > 0 else 0.0
    max_t = float(np.max(abs_t)) if samples > 0 else 0.0
    mean_diff = float(np.mean(np.abs(diff_power))) if samples > 0 else 0.0

    noise = float(np.std(diff_power))
    signal = float(np.mean(diff_power))
    power_snr = float(signal / noise) if noise > 1e-6 else 0.0

    return {
        "experiment": name,
        "samples": samples,
        "samples_exceeding_threshold": exceed_cnt,
        "exceedance_rate": exceed_rate,
        "exceedance_percent": round(exceed_rate * 100.0, 2),
        "max_abs_t_statistic": max_t,
        "power_snr": power_snr,
        "mean_power_difference_mw": mean_diff,
        "verdict": "CONTROL_PASSED" if max_t <= TVLA_THRESHOLD else "NOISE_OR_LEAKAGE"
    }


# ─── Main Compute Function ────────────────────────────────────────────────────

def run_control_tvla(
    dataset1_id: str,
    dataset2_id: str,
    mode: str = "fixed",
    results_root: Path = Path("results"),
    out_dir: Path = None,
    median_window: int = 5,
    moving_average_window: int = 5,
    savgol_window: int = 15,
) -> dict:
    """
    Execute Null-Hypothesis TVLA comparison between two Fixed or two Random value sets.
    """
    mode = mode.lower().strip()
    if mode not in ("fixed", "random"):
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'fixed' or 'random'.")

    dir1 = resolve_analysis_dir(results_root, dataset1_id)
    dir2 = resolve_analysis_dir(results_root, dataset2_id)

    name1 = dir1.name.replace("analysis_", "")
    name2 = dir2.name.replace("analysis_", "")

    # 1. Trace volume validation (MUST BE EQUAL)
    n1 = get_trace_count(dir1, mode)
    n2 = get_trace_count(dir2, mode)

    if n1 != n2:
        error_msg = (
            f"❌ Trace Volume Mismatch Error: Dataset 1 ({name1}) has {n1} traces, "
            f"while Dataset 2 ({name2}) has {n2} traces. "
            f"For {mode.capitalize()}-vs-{mode.capitalize()} TVLA calculation, "
            f"the number of traces MUST be identical!"
        )
        print(f"\n{error_msg}\n", file=sys.stderr)
        raise ValueError(error_msg)

    print(f"\n{'═'*72}")
    print(f"  TVLA Control Analysis  ▸  {mode.upper()}-vs-{mode.upper()}")
    print(f"  Dataset 1 : {name1} ({n1} traces)")
    print(f"  Dataset 2 : {name2} ({n2} traces)")
    print(f"  Trace Volume Check: PASSED (N = {n1})")
    print(f"{'═'*72}\n")

    # 2. Load average power traces
    p1 = load_raw_power(dir1, mode)
    p2 = load_raw_power(dir2, mode)

    common_len = min(len(p1), len(p2))
    p1 = p1[:common_len]
    p2 = p2[:common_len]

    # Load original fixed-vs-random t-stats & standard errors to establish baseline noise
    t1_df = pd.read_csv(dir1 / "tvla_t_stat.csv") if (dir1 / "tvla_t_stat.csv").exists() else None
    t2_df = pd.read_csv(dir2 / "tvla_t_stat.csv") if (dir2 / "tvla_t_stat.csv").exists() else None

    # Load random from run 1 to calibrate standard error
    r1_raw = dir1 / "filtered" / ("random" if mode == "fixed" else "fixed") / "raw.csv"
    r2_raw = dir2 / "filtered" / ("random" if mode == "fixed" else "fixed") / "raw.csv"

    def estimate_se(dir_path, main_power):
        other_mode = "random" if mode == "fixed" else "fixed"
        other_path = dir_path / "filtered" / other_mode / "raw.csv"
        t_path = dir_path / "tvla_t_stat.csv"
        if other_path.exists() and t_path.exists():
            other_p = pd.read_csv(other_path)["power_mw"].values[:common_len]
            t_vals = pd.read_csv(t_path)["t_stat"].values[:common_len]
            diff = np.abs(main_power - other_p)
            return np.where(np.abs(t_vals) > 1e-4, diff / np.abs(t_vals), np.std(main_power) / np.sqrt(n1))
        return np.full_like(main_power, np.std(main_power) / np.sqrt(n1))

    se1 = estimate_se(dir1, p1)
    se2 = estimate_se(dir2, p2)

    # Standard error for difference between two independent samples of size N
    se_diff = np.sqrt(0.5 * se1**2 + 0.5 * se2**2)
    min_se = max(1e-4, 0.005 * np.std(p1) / np.sqrt(n1))
    se_diff = np.maximum(se_diff, min_se)

    # 3. Compute New TVLA Welch t-statistic & p-value
    diff_power = p1 - p2
    t_stat_raw = diff_power / se_diff
    df_val = 2 * n1 - 2
    p_val_raw = 2.0 * t.sf(np.abs(t_stat_raw), df=df_val)
    p_val_raw = np.clip(p_val_raw, 1e-300, 1.0)

    # 4. Apply all TVLA filters to the new t-values and power signals
    t_stat_median = medfilt(t_stat_raw, kernel_size=median_window)
    t_stat_moving_avg = moving_average(t_stat_raw, window=moving_average_window)
    t_stat_wavelet = wavelet_denoise(t_stat_raw)
    t_stat_savgol = savgol_denoise(t_stat_raw, window_length=savgol_window, polyorder=3)
    t_stat_lowpass = lowpass(t_stat_raw, cutoff_ratio=0.2, order=3)

    p_val_median = 2.0 * t.sf(np.abs(t_stat_median), df=df_val)
    p_val_moving_avg = 2.0 * t.sf(np.abs(t_stat_moving_avg), df=df_val)
    p_val_wavelet = 2.0 * t.sf(np.abs(t_stat_wavelet), df=df_val)
    p_val_savgol = 2.0 * t.sf(np.abs(t_stat_savgol), df=df_val)
    p_val_lowpass = 2.0 * t.sf(np.abs(t_stat_lowpass), df=df_val)

    # 5. Output directory setup
    if out_dir is None:
        run_name = f"control_{mode}_{name1}_vs_{name2}"
        out_dir = results_root / "_control_tvla" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save all TVLA CSV files
    idx = np.arange(common_len)

    def save_csv_col(fname, vals, col_name):
        pd.DataFrame({"index": idx, col_name: vals}).to_csv(out_dir / fname, index=False)

    save_csv_col("tvla_t_stat.csv", t_stat_raw, "t_stat")
    save_csv_col("tvla_p_value.csv", p_val_raw, "p_value")
    save_csv_col("tvla_t_stat_median.csv", t_stat_median, "t_stat_median")
    save_csv_col("tvla_p_value_median.csv", p_val_median, "p_value_median")
    save_csv_col("tvla_t_stat_moving_average.csv", t_stat_moving_avg, "t_stat_moving_average")
    save_csv_col("tvla_p_value_moving_average.csv", p_val_moving_avg, "p_value_moving_average")
    save_csv_col("tvla_t_stat_wavelet.csv", t_stat_wavelet, "t_stat_wavelet")
    save_csv_col("tvla_p_value_wavelet.csv", p_val_wavelet, "p_value_wavelet")
    save_csv_col("tvla_t_stat_savitzky_golay.csv", t_stat_savgol, "t_stat_savitzky_golay")
    save_csv_col("tvla_p_value_savitzky_golay.csv", p_val_savgol, "p_value_savitzky_golay")
    save_csv_col("tvla_t_stat_lowpass.csv", t_stat_lowpass, "t_stat_lowpass")

    # Save compared power traces matrix CSV
    df_power = pd.DataFrame({
        "index": idx,
        f"{mode}_{name1}": p1,
        f"{mode}_{name2}": p2,
        "delta_power_mw": diff_power,
    })
    df_power.to_csv(out_dir / f"power_traces_{mode}.csv", index=False)

    # Save all t-stats into one master CSV
    df_master_t = pd.DataFrame({
        "index": idx,
        "t_raw": t_stat_raw,
        "t_median": t_stat_median,
        "t_moving_average": t_stat_moving_avg,
        "t_wavelet": t_stat_wavelet,
        "t_savitzky_golay": t_stat_savgol,
        "t_lowpass": t_stat_lowpass
    })
    df_master_t.to_csv(out_dir / "all_t_stat_filters.csv", index=False)

    # 7. Compute Quantitative Metrics for all filters
    metrics_list = [
        compute_quantitative_metrics(t_stat_raw, diff_power, "raw"),
        compute_quantitative_metrics(t_stat_median, diff_power, "median"),
        compute_quantitative_metrics(t_stat_moving_avg, diff_power, "moving_average"),
        compute_quantitative_metrics(t_stat_wavelet, diff_power, "wavelet"),
        compute_quantitative_metrics(t_stat_savgol, diff_power, "savitzky_golay"),
        compute_quantitative_metrics(t_stat_lowpass, diff_power, "lowpass"),
    ]
    pd.DataFrame(metrics_list).to_csv(out_dir / "quantitative_metrics.csv", index=False)

    # 8. Summary JSON
    raw_max_t = float(np.max(np.abs(t_stat_raw)))
    control_passed = raw_max_t <= TVLA_THRESHOLD
    summary_dict = {
        "mode": f"{mode}_vs_{mode}",
        "dataset_1": name1,
        "dataset_2": name2,
        "trace_volume": n1,
        "tvla_threshold": TVLA_THRESHOLD,
        "control_test_passed": control_passed,
        "verdict": (
            "Environment valid (control passed); no significant baseline noise bias detected."
            if control_passed else
            "Environment noisy (control failed); baseline noise exceeded threshold 4.5 between identical inputs."
        ),
        "raw_max_abs_t_statistic": raw_max_t,
        "raw_samples_exceeding_threshold": int(np.sum(np.abs(t_stat_raw) > TVLA_THRESHOLD)),
        "quantitative_metrics": {m["experiment"]: m for m in metrics_list},
        "mean_power_difference_mw": float(np.mean(np.abs(diff_power))),
        "filter_parameters": {
            "median_window": median_window,
            "moving_average_window": moving_average_window,
            "savgol_window": savgol_window
        }
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    # 9. Draw and save all plots
    BG_DARK = "#0F1117"
    BG_PANEL = "#1A1D2E"

    def style_ax(ax):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#333344")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    # Plot 1: TVLA Raw Plot
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=BG_DARK)
    style_ax(ax)
    ax.fill_between(idx, t_stat_raw, 0, where=(np.abs(t_stat_raw) >= TVLA_THRESHOLD),
                    color="#FF4C4C", alpha=0.4, label="Exceedance (|t| > 4.5)")
    ax.plot(idx, t_stat_raw, color="#3B82F6", lw=1.5, label="Raw TVLA t-stat")
    ax.axhline(TVLA_THRESHOLD, color="#FF4C4C", lw=1.2, ls="--", label=f"±{TVLA_THRESHOLD} threshold")
    ax.axhline(-TVLA_THRESHOLD, color="#FF4C4C", lw=1.2, ls="--")
    ax.axhline(0, color="#666666", lw=0.6)
    ax.set_title(f"TVLA Control {mode.capitalize()} vs {mode.capitalize()}: {name1} vs {name2} (N={n1})",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample Index", color="white")
    ax.set_ylabel("t-statistic", color="white")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor=BG_PANEL)
    plt.tight_layout()
    plt.savefig(plots_dir / "tvla.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Plot 2: TVLA Filters Overlay
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_DARK)
    style_ax(ax)
    ax.plot(idx, t_stat_raw, color="#3B82F6", lw=1.2, alpha=0.6, label="Raw")
    ax.plot(idx, t_stat_median, color="#8B5CF6", lw=1.4, label="Median")
    ax.plot(idx, t_stat_moving_avg, color="#06B6D4", lw=1.4, label="Moving Avg")
    ax.plot(idx, t_stat_wavelet, color="#F59E0B", lw=1.5, label="Wavelet")
    ax.plot(idx, t_stat_savgol, color="#EC4899", lw=1.4, label="Savitzky-Golay")
    ax.axhline(TVLA_THRESHOLD, color="#FF4C4C", lw=1.2, ls="--", label=f"±{TVLA_THRESHOLD}")
    ax.axhline(-TVLA_THRESHOLD, color="#FF4C4C", lw=1.2, ls="--")
    ax.axhline(0, color="#666666", lw=0.6)
    ax.set_title(f"TVLA Filter Comparison ({mode.capitalize()} vs {mode.capitalize()})",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Index", color="white")
    ax.set_ylabel("t-statistic", color="white")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor=BG_PANEL, ncol=3)
    plt.tight_layout()
    plt.savefig(plots_dir / "tvla_filters_overlay.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Plot 3: Power Signals Overlay
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=BG_DARK)
    style_ax(ax)
    ax.plot(idx, p1, color="#3B82F6", lw=1.5, label=f"{name1} ({n1} traces)")
    ax.plot(idx, p2, color="#F59E0B", lw=1.5, alpha=0.85, label=f"{name2} ({n2} traces)")
    ax.set_title(f"Average Power Traces ({mode.capitalize()} vs {mode.capitalize()})",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample Index", color="white")
    ax.set_ylabel("Power (mW)", color="white")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor=BG_PANEL)
    plt.tight_layout()
    plt.savefig(plots_dir / "power_overlay.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  ✅  Calculated new TVLA t-values:")
    print(f"      Raw Max |t|       : {raw_max_t:.4f}")
    print(f"      Exceedance Samples: {summary_dict['raw_samples_exceeding_threshold']}")
    print(f"      Control Verdict   : {'✓ PASSED (Clean)' if control_passed else '⚠ FAILED (Noisy)'}")
    print(f"  ✅  Saved CSV files to: {out_dir}")
    print(f"  ✅  Saved plots to    : {plots_dir}")

    return {
        "status": "success",
        "out_dir": str(out_dir),
        "summary": summary_dict,
        "metrics": metrics_list,
        "plots": [f.name for f in plots_dir.glob("*.png")],
        "csvs": [f.name for f in out_dir.glob("*.csv")]
    }


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def interactive_picker(results_root: Path):
    dirs = sorted([d for d in results_root.glob("analysis_*") if d.is_dir()])
    if not dirs:
        print("❌ No analysis directories found in results/"); sys.exit(1)

    print("\nSelect Mode:")
    print("  [1] fixed-vs-fixed   (Null hypothesis control on 2 fixed datasets)")
    print("  [2] random-vs-random (Null hypothesis control on 2 random datasets)")
    choice = input("  Choice [1/2, default: 1] > ").strip()
    mode = "random" if choice == "2" else "fixed"

    # Gather volumes
    valid_dirs = []
    for d in dirs:
        raw_p = d / "filtered" / mode / "raw.csv"
        if raw_p.exists():
            try:
                cnt = get_trace_count(d, mode)
                valid_dirs.append((d, cnt))
            except Exception:
                pass

    if len(valid_dirs) < 2:
        print(f"❌ Need at least 2 datasets with filtered/{mode}/raw.csv"); sys.exit(1)

    print(f"\nAvailable datasets for {mode.upper()}-vs-{mode.upper()}:")
    print("─" * 60)
    for i, (d, cnt) in enumerate(valid_dirs):
        print(f"  [{i:>2}]  {d.name.replace('analysis_', '')}   (Trace Volume = {cnt})")
    print("─" * 60)

    raw1 = input("  Enter index for Dataset 1 > ").strip()
    raw2 = input("  Enter index for Dataset 2 > ").strip()

    try:
        idx1 = int(raw1)
        idx2 = int(raw2)
        ds1 = valid_dirs[idx1][0].name
        ds2 = valid_dirs[idx2][0].name
    except (ValueError, IndexError):
        print("❌ Invalid selection"); sys.exit(1)

    run_control_tvla(ds1, ds2, mode=mode, results_root=results_root)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate TVLA t-values between 2 Fixed or 2 Random value sets (Control Run)."
    )
    parser.add_argument("--mode", choices=["fixed", "random"], default="fixed",
                        help="Comparison mode (fixed vs fixed, or random vs random)")
    parser.add_argument("--dataset1", default=None, help="Name or timestamp of first dataset")
    parser.add_argument("--dataset2", default=None, help="Name or timestamp of second dataset")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent.parent / "results")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--moving-average-window", type=int, default=5)
    parser.add_argument("--savgol-window", type=int, default=15)
    parser.add_argument("--interactive", action="store_true", help="Launch interactive dataset picker")

    args = parser.parse_args()

    results_root = args.results_dir.resolve()
    if not results_root.exists():
        print(f"❌ results dir not found: {results_root}"); sys.exit(1)

    if args.interactive or (args.dataset1 is None and args.dataset2 is None and sys.stdin.isatty()):
        interactive_picker(results_root)
        return

    if not args.dataset1 or not args.dataset2:
        print("❌ Error: Both --dataset1 and --dataset2 are required in non-interactive mode.")
        print("Usage: python3 scripts/control_tvla.py --mode fixed --dataset1 <id1> --dataset2 <id2>")
        sys.exit(1)

    try:
        run_control_tvla(
            args.dataset1,
            args.dataset2,
            mode=args.mode,
            results_root=results_root,
            out_dir=args.out_dir,
            median_window=args.median_window,
            moving_average_window=args.moving_average_window,
            savgol_window=args.savgol_window
        )
    except ValueError as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
