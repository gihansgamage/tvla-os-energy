#!/usr/bin/env python3
"""
compare_results.py
==================
Compare multiple TVLA analysis results from the results/ folder.

Outputs:
  - Cross-dataset CSVs:
      compare_{mode}_summary.csv
      compare_{mode}_power_traces.csv
      compare_{mode}_t_stat.csv
      compare_{mode}_p_values.csv
  - Cross-dataset plots:
      compare_{mode}_cross.png
      compare_both.png
  - Per-dataset CSVs & 5-page plots:
      _page1_plots.png
      _page2_tstat.png
      _page3_pvalue.png
      _page4_filtered_{mode}.png
      _page5_metrics.png
      _all_tstat.csv
      _all_pvalue.csv
      _all_filtered_power_{mode}.csv
      _metrics.csv
      _thesis_table.csv

Usage:
    python3 scripts/compare_results.py
    python3 scripts/compare_results.py --mode fixed
    python3 scripts/compare_results.py --mode random
    python3 scripts/compare_results.py --mode both
    python3 scripts/compare_results.py --datasets 20261024_221251 20261024_233546
"""

import os, sys, json, argparse, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive Agg backend by default so saving figures is non-blocking
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator

# ── palette / threshold ───────────────────────────────────────────────────────
PALETTE = [
    "#4C9BE8","#E87B4C","#4CE87B","#E84C9B",
    "#9B4CE8","#E8D74C","#4CE8D7","#E84C4C",
]
TVLA_THRESHOLD = 4.5

BG_DARK  = "#0F1117"
BG_PANEL = "#1A1D2E"
BG_MID   = "#12151F"

# ── style helper ──────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color("#333344")
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def _style_table(tbl):
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(BG_PANEL if row % 2 == 0 else BG_MID)
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333344")

def _save_fig(fig, path: Path, show: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

# ── data loaders ──────────────────────────────────────────────────────────────
def discover_analyses(root: Path) -> list:
    return sorted([d for d in root.glob("analysis_*") if d.is_dir()])

def has_raw_csv(d: Path, side: str) -> bool:
    return (d / "filtered" / side / "raw.csv").exists()

def load_csv_safe(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None

def load_summary(d: Path) -> dict:
    p = d / "summary.json"
    if not p.exists(): return {}
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

def label_for(d: Path) -> str:
    return d.name.replace("analysis_", "")

def get_all_tstat_csvs(d: Path) -> dict:
    result = {}
    for p in sorted(d.glob("tvla_t_stat*.csv")):
        key = p.stem.replace("tvla_t_stat", "").lstrip("_") or "raw"
        df = load_csv_safe(p)
        if df is not None:
            result[key] = df
    return result

def get_all_pvalue_csvs(d: Path) -> dict:
    result = {}
    for p in sorted(d.glob("tvla_p_value*.csv")):
        key = p.stem.replace("tvla_p_value", "").lstrip("_") or "raw"
        df = load_csv_safe(p)
        if df is not None:
            result[key] = df
    return result

def get_all_filtered_csvs(d: Path, side: str) -> dict:
    result = {}
    base = d / "filtered" / side
    if not base.exists(): return result
    for p in sorted(base.glob("*.csv")):
        df = load_csv_safe(p)
        if df is not None:
            result[p.stem] = df
    return result

def get_all_plots(d: Path) -> list:
    plots_dir = d / "plots"
    if not plots_dir.exists(): return []
    return sorted(plots_dir.glob("*.png"))

# ── interactive / CLI dataset selector ────────────────────────────────────────
def select_datasets_cli(analyses: list, dataset_names: list[str]) -> list:
    lookup = {}
    for a in analyses:
        lookup[a.name] = a
        lookup[label_for(a)] = a

    selected = []
    for name in dataset_names:
        clean = name.strip()
        if clean in lookup:
            selected.append(lookup[clean])
        elif f"analysis_{clean}" in lookup:
            selected.append(lookup[f"analysis_{clean}"])
        else:
            print(f"  ⚠  Unknown dataset '{name}', skipping.")
    return selected

def pick_datasets(analyses: list, mode: str) -> list:
    available = [a for a in analyses if has_raw_csv(a, mode)]
    if not available:
        print(f"\n❌  No analyses with filtered/{mode}/raw.csv found.")
        sys.exit(1)

    # Non-interactive fallback
    if not sys.stdin.isatty():
        print(f"  Non-interactive terminal detected: selecting all {len(available)} datasets.")
        return available

    print(f"\n{'═'*72}")
    print(f"  Available datasets  ▸  filtered/{mode}/raw.csv")
    print(f"{'═'*72}")
    for i, a in enumerate(available):
        s = load_summary(a)
        n_f  = s.get("fixed_traces",  "?")
        n_r  = s.get("random_traces", "?")
        qm   = s.get("quantitative_metrics", {}).get("raw", {})
        mt   = qm.get("max_abs_t_statistic")
        ep   = qm.get("exceedance_percent")
        mt_s = f"{mt:.3f}" if isinstance(mt, float) else "?"
        ep_s = f"{ep:.1f}%" if isinstance(ep, float) else "?"
        print(f"  [{i:>3}]  {a.name}   "
              f"fixed={str(n_f):<5}  random={str(n_r):<5}  "
              f"max|t|={mt_s:<9} exceed={ep_s}")

    print(f"\n  Enter indices separated by spaces  (e.g.  0 3 7)")
    print(f"  Press ENTER to select ALL  |  q to quit")
    try:
        raw = input("  > ").strip()
    except EOFError:
        return available

    if raw.lower() == "q": sys.exit(0)
    if raw == "": return available

    chosen = []
    for tok in raw.split():
        try:
            idx = int(tok)
            if 0 <= idx < len(available): chosen.append(available[idx])
            else: print(f"  ⚠  Index {idx} out of range, skipped.")
        except ValueError:
            print(f"  ⚠  '{tok}' not a number, skipped.")
    if not chosen:
        print("  ❌  No valid indices."); sys.exit(1)
    return chosen

def pick_comparison_mode() -> str:
    if not sys.stdin.isatty():
        return "fixed"
    print(f"\n{'═'*72}")
    print("  What do you want to compare?")
    print("  [1]  fixed-vs-fixed    (control: same fixed input across runs)")
    print("  [2]  random-vs-random  (control: random inputs across runs)")
    print("  [3]  both side-by-side (fixed + random in same figure)")
    print(f"{'═'*72}")
    try:
        c = input("  > ").strip()
    except EOFError:
        return "fixed"
    if c == "1": return "fixed"
    if c == "2": return "random"
    if c == "3": return "both"
    print("  Defaulting to 'fixed'.")
    return "fixed"

# ═════════════════════════════════════════════════════════════════════════════
#  PER-DATASET FULL REPORT & CSV EXPORT
# ═════════════════════════════════════════════════════════════════════════════

def show_dataset_full_report(analysis_dir: Path, mode: str, out_dir: Path, show: bool = False):
    """
    For ONE selected dataset:
      - Export all t-stat, p-value, and filtered power CSVs
      - Export quantitative metrics and thesis table CSVs
      - Generate 5-page visual summary plots
    """
    label = label_for(analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  {'─'*60}")
    print(f"  Full report & CSV export: {label}")
    print(f"  {'─'*60}")

    # 1. Existing PNG plots grid
    plots = get_all_plots(analysis_dir)
    if plots:
        n = len(plots)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), facecolor=BG_DARK)
        fig.suptitle(f"{label}  ▸  Existing Plots ({n} images)",
                     color="white", fontsize=13, fontweight="bold")
        flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, png in zip(flat, plots):
            try:
                img = mpimg.imread(str(png))
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", color="red", transform=ax.transAxes)
            ax.set_title(png.stem, color="white", fontsize=9)
            ax.axis("off")
            ax.set_facecolor(BG_DARK)
        for ax in flat[len(plots):]:
            ax.axis("off"); ax.set_facecolor(BG_DARK)
        plt.tight_layout(pad=1.5)
        _save_fig(fig, out_dir / f"{label}_page1_plots.png", show=show)
        print(f"    ✅  Page 1 saved ({n} plots)")

    # 2. t-stat CSVs & overlay plot
    tstat_data = get_all_tstat_csvs(analysis_dir)
    if tstat_data:
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG_DARK)
        _style_ax(ax)
        for (filter_name, df), color in zip(tstat_data.items(), PALETTE * 4):
            col = "t_stat" if "t_stat" in df.columns else df.columns[-1]
            ax.plot(df[col].values, color=color, lw=1.5, alpha=0.85, label=filter_name)
        ax.axhline( TVLA_THRESHOLD, color="#FF4C4C", lw=1.5, ls="--", label=f"±{TVLA_THRESHOLD}")
        ax.axhline(-TVLA_THRESHOLD, color="#FF4C4C", lw=1.5, ls="--")
        ax.axhline(0, color="#555", lw=0.5)
        ax.set_xlabel("Sample Index", color="white")
        ax.set_ylabel("t-statistic", color="white")
        ax.set_title(f"{label}  ▸  TVLA t-Statistics (all filters)", color="white", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor=BG_PANEL, ncol=3)
        fig.patch.set_facecolor(BG_DARK)
        _save_fig(fig, out_dir / f"{label}_page2_tstat.png", show=show)
        print(f"    ✅  Page 2 saved ({len(tstat_data)} filter t-stats)")

        # Export all t-stats into a merged CSV
        max_len = max(len(df) for df in tstat_data.values())
        merged_t = pd.DataFrame({"index": range(max_len)})
        for fname, df in tstat_data.items():
            col = "t_stat" if "t_stat" in df.columns else df.columns[-1]
            merged_t[f"t_{fname}"] = pd.Series(df[col].values)
        merged_t_path = out_dir / f"{label}_all_tstat.csv"
        merged_t.to_csv(merged_t_path, index=False)
        print(f"    ✅  CSV saved: {merged_t_path.name}")

    # 3. p-value CSVs & overlay plot
    pval_data = get_all_pvalue_csvs(analysis_dir)
    if pval_data:
        fig, axes_pv = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_DARK)
        ax_log, ax_lin = axes_pv
        _style_ax(ax_log); _style_ax(ax_lin)
        for (filter_name, df), color in zip(pval_data.items(), PALETTE * 4):
            col = "p_value" if "p_value" in df.columns else df.columns[-1]
            vals = df[col].replace(0, np.nan).values
            ax_log.semilogy(vals, color=color, lw=1.4, alpha=0.8, label=filter_name)
            ax_lin.plot(vals, color=color, lw=1.4, alpha=0.8, label=filter_name)
        for ax_p in [ax_log, ax_lin]:
            ax_p.axhline(0.05, color="#FFD700", lw=1.2, ls="--", label="p=0.05")
            ax_p.axhline(0.001, color="#FF6B6B", lw=1.2, ls="--", label="p=0.001")
            ax_p.set_xlabel("Sample Index", color="white")
            ax_p.set_ylabel("p-value", color="white")
            ax_p.legend(fontsize=7, framealpha=0.3, labelcolor="white", facecolor=BG_PANEL, ncol=2)
        ax_log.set_title("p-values (log scale)", color="white", fontsize=11, fontweight="bold")
        ax_lin.set_title("p-values (linear scale)", color="white", fontsize=11, fontweight="bold")
        fig.suptitle(f"{label}  ▸  TVLA p-Values (all filters)", color="white", fontsize=13, fontweight="bold")
        fig.patch.set_facecolor(BG_DARK)
        _save_fig(fig, out_dir / f"{label}_page3_pvalue.png", show=show)
        print(f"    ✅  Page 3 saved ({len(pval_data)} filter p-values)")

        # Export all p-values into a merged CSV
        max_p_len = max(len(df) for df in pval_data.values())
        merged_p = pd.DataFrame({"index": range(max_p_len)})
        for fname, df in pval_data.items():
            col = "p_value" if "p_value" in df.columns else df.columns[-1]
            merged_p[f"p_{fname}"] = pd.Series(df[col].values)
        merged_p_path = out_dir / f"{label}_all_pvalue.csv"
        merged_p.to_csv(merged_p_path, index=False)
        print(f"    ✅  CSV saved: {merged_p_path.name}")

    # 4. Filtered power signals & CSV export
    filtered_csvs = get_all_filtered_csvs(analysis_dir, mode)
    if filtered_csvs:
        n = len(filtered_csvs)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), facecolor=BG_DARK, squeeze=False)
        flat = axes.flatten()
        fig.suptitle(f"{label}  ▸  Filtered Power Signals [{mode}] ({n} filters)", color="white", fontsize=13, fontweight="bold")
        for ax, (fname, df), color in zip(flat, filtered_csvs.items(), PALETTE * 4):
            _style_ax(ax)
            col = "power_mw" if "power_mw" in df.columns else df.columns[-1]
            ax.plot(df[col].values, color=color, lw=1.3)
            ax.fill_between(range(len(df)), df[col].values, alpha=0.12, color=color)
            ax.set_title(fname, color="white", fontsize=9)
            ax.set_ylabel("Power (mW)", color="white", fontsize=8)
            ax.set_xlabel("Sample", color="white", fontsize=8)
        for ax in flat[n:]:
            ax.axis("off"); ax.set_facecolor(BG_DARK)
        plt.tight_layout(pad=1.8)
        _save_fig(fig, out_dir / f"{label}_page4_filtered_{mode}.png", show=show)
        print(f"    ✅  Page 4 saved ({n} filtered signals)")

        # Export all filtered power signals into merged CSV
        max_f_len = max(len(df) for df in filtered_csvs.values())
        merged_f = pd.DataFrame({"index": range(max_f_len)})
        for fname, df in filtered_csvs.items():
            col = "power_mw" if "power_mw" in df.columns else df.columns[-1]
            merged_f[f"power_{fname}"] = pd.Series(df[col].values)
        merged_f_path = out_dir / f"{label}_all_filtered_power_{mode}.csv"
        merged_f.to_csv(merged_f_path, index=False)
        print(f"    ✅  CSV saved: {merged_f_path.name}")

    # 5. quantitative_metrics & thesis_comparison_table
    qm_df = load_csv_safe(analysis_dir / "quantitative_metrics.csv")
    thesis_df = load_csv_safe(analysis_dir / "thesis_comparison_table.csv")
    fig, axes5 = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG_DARK)
    fig.suptitle(f"{label}  ▸  Quantitative Metrics & Thesis Table", color="white", fontsize=13, fontweight="bold", y=1.01)
    for ax5 in axes5:
        ax5.axis("off")
        ax5.set_facecolor(BG_DARK)

    if qm_df is not None:
        cols_show = list(qm_df.columns)
        data_rows = [[str(row[c])[:14] for c in cols_show] for _, row in qm_df.iterrows()]
        col_labels_qm = [textwrap.fill(c, 12) for c in cols_show]
        tbl = axes5[0].table(cellText=data_rows, colLabels=col_labels_qm, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7)
        tbl.scale(1.0, 1.8)
        _style_table(tbl)
        axes5[0].set_title("quantitative_metrics.csv", color="white", fontsize=11, fontweight="bold", pad=12)
        # Export metrics CSV copy
        qm_df.to_csv(out_dir / f"{label}_metrics.csv", index=False)

    if thesis_df is not None:
        cols_th = list(thesis_df.columns)
        rows_th = [[textwrap.fill(str(v), 16) for v in row] for _, row in thesis_df.iterrows()]
        col_labels_th = [textwrap.fill(c, 14) for c in cols_th]
        tbl2 = axes5[1].table(cellText=rows_th, colLabels=col_labels_th, loc="center", cellLoc="center")
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(7)
        tbl2.scale(1.0, 2.0)
        _style_table(tbl2)
        axes5[1].set_title("thesis_comparison_table.csv", color="white", fontsize=11, fontweight="bold", pad=12)
        # Export thesis CSV copy
        thesis_df.to_csv(out_dir / f"{label}_thesis_table.csv", index=False)

    plt.tight_layout(pad=2.0)
    _save_fig(fig, out_dir / f"{label}_page5_metrics.png", show=show)
    print(f"    ✅  Page 5 saved (metrics & thesis tables)")


# ═════════════════════════════════════════════════════════════════════════════
#  CROSS-DATASET COMPARISON & MATRIX CSV EXPORTS
# ═════════════════════════════════════════════════════════════════════════════

def compare_single_side(selected: list, mode: str, out_dir: Path, show: bool = False):
    """
    Cross-dataset comparison:
      - Plots: compare_{mode}_cross.png
      - CSVs:
          compare_{mode}_summary.csv
          compare_{mode}_power_traces.csv
          compare_{mode}_t_stat.csv
          compare_{mode}_p_values.csv
    """
    n = len(selected)
    colors = (PALETTE * ((n // len(PALETTE)) + 1))[:n]

    records = []
    for path in selected:
        df = load_csv_safe(path / "filtered" / mode / "raw.csv")
        if df is None or "power_mw" not in df.columns:
            print(f"  ⚠  Skipping {path.name}: no raw.csv"); continue
        summary = load_summary(path)
        tstat_df = load_csv_safe(path / "tvla_t_stat.csv")
        pval_df  = load_csv_safe(path / "tvla_p_value.csv")
        records.append({
            "label":   label_for(path),
            "path":    path,
            "power":   df["power_mw"].values,
            "summary": summary,
            "tstat":   tstat_df["t_stat"].values if (tstat_df is not None and "t_stat" in tstat_df.columns) else None,
            "pvalue":  pval_df["p_value"].values if (pval_df is not None and "p_value" in pval_df.columns) else None,
        })

    if not records:
        print("  ❌  No valid data loaded."); return

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14)); fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35, top=0.91, bottom=0.07, left=0.07, right=0.97)
    ax_vio  = fig.add_subplot(gs[0, :])
    ax_tst  = fig.add_subplot(gs[1, :])
    ax_box  = fig.add_subplot(gs[2, 0])
    ax_tbl  = fig.add_subplot(gs[2, 1])
    for ax in [ax_vio, ax_tst, ax_box]: _style_ax(ax)
    ax_tbl.axis("off")

    # Violin plot
    vdata  = [r["power"] for r in records]
    vlabel = [r["label"] for r in records]
    parts  = ax_vio.violinplot(vdata, showmedians=True, showextrema=True)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.65)
    for key in ("cmedians","cmins","cmaxes","cbars"):
        if key in parts:
            parts[key].set_color("white"); parts[key].set_linewidth(1.2)
    ax_vio.set_xticks(range(1, len(records)+1))
    ax_vio.set_xticklabels(vlabel, rotation=35, ha="right", color="white", fontsize=7)
    ax_vio.set_ylabel("Power (mW)", color="white", fontsize=10)
    ax_vio.set_title(f"Power Distribution  ▸  {mode.upper()} Traces  ({len(records)} datasets)",
                     color="white", fontsize=12, fontweight="bold", pad=10)

    # t-stat overlay
    has_ts = False
    for r, c in zip(records, colors):
        if r["tstat"] is not None:
            ax_tst.plot(np.arange(len(r["tstat"])), r["tstat"], color=c, lw=1.4, alpha=0.85, label=r["label"])
            has_ts = True
    ax_tst.axhline( TVLA_THRESHOLD, color="#FF4C4C", lw=1.5, ls="--", label=f"±{TVLA_THRESHOLD}")
    ax_tst.axhline(-TVLA_THRESHOLD, color="#FF4C4C", lw=1.5, ls="--")
    ax_tst.axhline(0, color="#555", lw=0.5)
    ax_tst.set_xlabel("Sample Index", color="white", fontsize=10)
    ax_tst.set_ylabel("t-statistic", color="white", fontsize=10)
    ax_tst.set_title("TVLA t-Statistic Overlay (raw filter)", color="white", fontsize=12, fontweight="bold", pad=10)
    if has_ts and len(records) <= 15:
        ax_tst.legend(fontsize=7, framealpha=0.3, labelcolor="white", facecolor=BG_PANEL, ncol=max(1, len(records)//3))

    # Box plot
    bp = ax_box.boxplot(vdata, patch_artist=True, notch=False, medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for item in ("whiskers","caps","fliers"):
        for el in bp[item]: el.set(color="#AAAAAA", linewidth=0.8)
    ax_box.set_xticks(range(1, len(records)+1))
    ax_box.set_xticklabels(vlabel, rotation=35, ha="right", color="white", fontsize=7)
    ax_box.set_ylabel("Power (mW)", color="white", fontsize=10)
    ax_box.set_title("Box Plot Comparison", color="white", fontsize=11, fontweight="bold")

    # Metrics table
    col_labels = ["Dataset","Traces","Mean(mW)","Std(mW)","Max|t|","Exceed%","SNR","ΔPower(mW)"]
    rows_tbl, csv_rows = [], []
    for r in records:
        s  = r["summary"]
        pw = r["power"]
        qm = (s or {}).get("quantitative_metrics", {}).get("raw", {})
        n_key = "fixed_traces" if mode == "fixed" else "random_traces"
        n_tr  = (s or {}).get(n_key, "?")
        mt  = qm.get("max_abs_t_statistic")
        ep  = qm.get("exceedance_percent")
        snr = qm.get("power_snr")
        dpw = qm.get("mean_power_difference_mw")
        rows_tbl.append([
            textwrap.fill(r["label"], 12),
            str(n_tr),
            f"{pw.mean():.1f}",
            f"{pw.std():.1f}",
            f"{mt:.3f}"  if mt  is not None else "—",
            f"{ep:.1f}%" if ep  is not None else "—",
            f"{snr:.4f}" if snr is not None else "—",
            f"{dpw:.1f}" if dpw is not None else "—",
        ])
        csv_rows.append({
            "dataset":             r["label"],
            "mode":                mode,
            "n_traces":            n_tr,
            "mean_power_mw":       round(float(pw.mean()), 3),
            "std_power_mw":        round(float(pw.std()),  3),
            "min_power_mw":        round(float(pw.min()),  3),
            "max_power_mw":        round(float(pw.max()),  3),
            "max_abs_t_statistic": mt  if mt  is not None else "",
            "exceedance_percent":  ep  if ep  is not None else "",
            "power_snr":           snr if snr is not None else "",
            "mean_power_diff_mw":  dpw if dpw is not None else "",
            "leakage_verdict":     "LEAKAGE" if (mt is not None and mt > TVLA_THRESHOLD) else "CLEAN"
        })
    tbl = ax_tbl.table(cellText=rows_tbl, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    tbl.scale(1.0, 1.6)
    _style_table(tbl)
    ax_tbl.set_title("Summary Metrics (raw filter)", color="white", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(f"TVLA Cross-Dataset Comparison  ▸  {mode.upper()}-vs-{mode.upper()}  ({len(records)} datasets)",
                 color="white", fontsize=14, fontweight="bold", y=0.97)

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_fig(fig, out_dir / f"compare_{mode}_cross.png", show=show)
    print(f"\n  ✅  Cross-dataset plot → {out_dir}/compare_{mode}_cross.png")

    # ── 1. Export summary CSV ─────────────────────────────────────────────────
    summary_csv_path = out_dir / f"compare_{mode}_summary.csv"
    pd.DataFrame(csv_rows).to_csv(summary_csv_path, index=False)
    print(f"  ✅  CSV: {summary_csv_path.name}")

    # ── 2. Export power traces matrix CSV ──────────────────────────────────────
    max_samples = max(len(r["power"]) for r in records)
    df_power = pd.DataFrame({"index": range(max_samples)})
    for r in records:
        df_power[r["label"]] = pd.Series(r["power"])
    power_csv_path = out_dir / f"compare_{mode}_power_traces.csv"
    df_power.to_csv(power_csv_path, index=False)
    print(f"  ✅  CSV: {power_csv_path.name}")

    # ── 3. Export t-statistic matrix CSV ──────────────────────────────────────
    valid_tstats = [r for r in records if r["tstat"] is not None]
    if valid_tstats:
        max_t_len = max(len(r["tstat"]) for r in valid_tstats)
        df_tstat = pd.DataFrame({"index": range(max_t_len)})
        for r in valid_tstats:
            df_tstat[r["label"]] = pd.Series(r["tstat"])
        tstat_csv_path = out_dir / f"compare_{mode}_t_stat.csv"
        df_tstat.to_csv(tstat_csv_path, index=False)
        print(f"  ✅  CSV: {tstat_csv_path.name}")

    # ── 4. Export p-value matrix CSV ──────────────────────────────────────────
    valid_pvals = [r for r in records if r["pvalue"] is not None]
    if valid_pvals:
        max_p_len = max(len(r["pvalue"]) for r in valid_pvals)
        df_pval = pd.DataFrame({"index": range(max_p_len)})
        for r in valid_pvals:
            df_pval[r["label"]] = pd.Series(r["pvalue"])
        pval_csv_path = out_dir / f"compare_{mode}_p_values.csv"
        df_pval.to_csv(pval_csv_path, index=False)
        print(f"  ✅  CSV: {pval_csv_path.name}")


def compare_both_sides(selected: list, out_dir: Path, show: bool = False):
    n = len(selected)
    colors = (PALETTE * ((n // len(PALETTE)) + 1))[:n]
    fig, axes = plt.subplots(n, 2, figsize=(16, max(4, 4 * n)), facecolor=BG_DARK, squeeze=False)
    fig.suptitle("Fixed-vs-Fixed  AND  Random-vs-Random  (per dataset)", color="white", fontsize=13, fontweight="bold", y=1.01)

    both_power_data = {}
    both_tstat_data = {}

    for row, (path, color) in enumerate(zip(selected, colors)):
        label  = label_for(path)
        df_fix = load_csv_safe(path / "filtered" / "fixed"  / "raw.csv")
        df_ran = load_csv_safe(path / "filtered" / "random" / "raw.csv")
        tstat  = load_csv_safe(path / "tvla_t_stat.csv")
        ax_l, ax_r = axes[row][0], axes[row][1]
        _style_ax(ax_l); _style_ax(ax_r)

        if df_fix is not None and "power_mw" in df_fix.columns:
            ax_l.plot(df_fix["power_mw"].values, color=color, lw=1.3, label="fixed")
            both_power_data[f"{label}_fixed"] = df_fix["power_mw"].values
        if df_ran is not None and "power_mw" in df_ran.columns:
            ax_l.plot(df_ran["power_mw"].values, color="#E8A84C", lw=1.3, alpha=0.75, label="random")
            both_power_data[f"{label}_random"] = df_ran["power_mw"].values

        ax_l.set_title(f"{label}  ▸  Power Traces", color="white", fontsize=9)
        ax_l.set_ylabel("Power (mW)", color="white", fontsize=8)
        ax_l.legend(fontsize=7, framealpha=0.3, labelcolor="white", facecolor=BG_PANEL)

        if tstat is not None and "t_stat" in tstat.columns:
            t = tstat["t_stat"].values; x = np.arange(len(t))
            ax_r.fill_between(x, t, 0, where=(np.abs(t) >= TVLA_THRESHOLD), color="#FF4C4C", alpha=0.35, label="leakage")
            ax_r.plot(x, t, color=color, lw=1.3)
            both_tstat_data[f"{label}"] = t
        ax_r.axhline( TVLA_THRESHOLD, color="#FF4C4C", lw=1.2, ls="--")
        ax_r.axhline(-TVLA_THRESHOLD, color="#FF4C4C", lw=1.2, ls="--")
        ax_r.axhline(0, color="#555", lw=0.5)
        ax_r.set_title(f"{label}  ▸  TVLA t-Stat", color="white", fontsize=9)
        ax_r.set_ylabel("t-statistic", color="white", fontsize=8)

    plt.tight_layout(pad=2.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_fig(fig, out_dir / "compare_both.png", show=show)
    print(f"\n  ✅  Both-side plot → {out_dir}/compare_both.png")

    # Export both sides power traces CSV
    if both_power_data:
        max_len = max(len(v) for v in both_power_data.values())
        df_both = pd.DataFrame({"index": range(max_len)})
        for col_name, vals in both_power_data.items():
            df_both[col_name] = pd.Series(vals)
        df_both.to_csv(out_dir / "compare_both_power_traces.csv", index=False)
        print(f"  ✅  CSV: compare_both_power_traces.csv")

    if both_tstat_data:
        max_t_len = max(len(v) for v in both_tstat_data.values())
        df_t = pd.DataFrame({"index": range(max_t_len)})
        for col_name, vals in both_tstat_data.items():
            df_t[col_name] = pd.Series(vals)
        df_t.to_csv(out_dir / "compare_both_t_stat.csv", index=False)
        print(f"  ✅  CSV: compare_both_t_stat.csv")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare TVLA analysis results across multiple datasets."
    )
    parser.add_argument("--results-dir","-r", type=Path,
        default=Path(__file__).parent.parent / "results")
    parser.add_argument("--out-dir","-o", type=Path,
        default=Path(__file__).parent.parent / "results" / "_comparisons")
    parser.add_argument("--mode", choices=["fixed","random","both"], default=None)
    parser.add_argument("--datasets", nargs="+", default=None,
        help="List of dataset folder names or timestamps to compare (skips interactive picker)")
    parser.add_argument("--no-full-report", action="store_true",
        help="Skip per-dataset full report pages, show cross-dataset only")
    parser.add_argument("--show", action="store_true",
        help="Display figures in interactive popup windows (blocks execution)")
    args = parser.parse_args()

    results_root = args.results_dir.resolve()
    if not results_root.exists():
        print(f"❌  results dir not found: {results_root}"); sys.exit(1)

    print(f"\n{'═'*72}")
    print("  TVLA Multi-Dataset Comparator")
    print(f"  Results root : {results_root}")
    print(f"  Output dir   : {args.out_dir.resolve()}")
    print(f"{'═'*72}")

    analyses = discover_analyses(results_root)
    if not analyses:
        print("❌  No analysis_* directories found."); sys.exit(1)
    print(f"  Found {len(analyses)} analysis directories.\n")

    mode = args.mode if args.mode else pick_comparison_mode()

    if args.datasets:
        selected = select_datasets_cli(analyses, args.datasets)
        if not selected:
            print("❌  None of the specified datasets were found."); sys.exit(1)
        print(f"  Selected {len(selected)} datasets from CLI arguments.")
    else:
        if mode == "both":
            both_avail = [a for a in analyses if has_raw_csv(a,"fixed") and has_raw_csv(a,"random")]
            selected = pick_datasets(both_avail, "fixed")
        else:
            selected = pick_datasets(analyses, mode)

    out_dir = args.out_dir.resolve()

    # ── Per-dataset full reports & CSVs ───────────────────────────────────────
    if not args.no_full_report:
        print(f"\n{'─'*72}")
        print(f"  Generating full per-dataset reports & CSVs for {len(selected)} datasets …")
        print(f"{'─'*72}")
        report_mode = "fixed" if mode in ("fixed","both") else "random"
        for path in selected:
            ds_out = out_dir / label_for(path)
            show_dataset_full_report(path, report_mode, ds_out, show=args.show)

    # ── Cross-dataset comparison & Matrix CSVs ─────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Generating cross-dataset comparisons & matrix CSVs …")
    print(f"{'─'*72}")
    if mode == "both":
        compare_both_sides(selected, out_dir, show=args.show)
    elif mode == "fixed":
        compare_single_side(selected, "fixed", out_dir, show=args.show)
    else:
        compare_single_side(selected, "random", out_dir, show=args.show)

    print(f"\n{'═'*72}")
    print(f"  ✅  All done! All CSV files and graphs saved to: {out_dir}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
