"""Generate all publish-ready figures for the SpatialPathDB paper."""

import os
import json
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from spdb import config

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "Mono":    "#d62728",
    "Mono-T":  "#ff7f0e",
    "Mono-C":  "#e377c2",
    "SO":      "#2ca02c",
    "SO-C":    "#8c564b",
    "SPDB":    "#1f77b4",
    "SPDB-Z":  "#9467bd",
    "Hilbert": "#1f77b4",
    "Z-order": "#9467bd",
}

MARKERS = {
    "Mono": "v", "Mono-T": "^", "Mono-C": "D",
    "SO": "s", "SO-C": "p", "SPDB": "o", "SPDB-Z": "X",
}


def _load_raw_latencies(name, cfg):
    path = os.path.join(config.RAW_DIR, f"{name}_{cfg}.csv")
    lats = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(float(row["latency_ms"]))
    return np.array(lats)


def _load_json(name):
    path = os.path.join(config.RAW_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def _savefig(fig, name):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(config.FIGURES_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


# ---------- Figure 1: Q1 Latency CDF ----------

def fig_q1_latency_cdf():
    """CDF of Q1 viewport latency across all configurations."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for cfg in ["Mono", "Mono-T", "Mono-C", "SO", "SO-C", "SPDB"]:
        try:
            lats = _load_raw_latencies("q1_viewport", cfg)
            sorted_lats = np.sort(lats)
            cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
            ax.plot(sorted_lats, cdf, label=cfg,
                    color=COLORS.get(cfg), linewidth=1.5)
        except FileNotFoundError:
            pass

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Q1 Viewport Latency Distribution")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    _savefig(fig, "q1_latency_cdf")


# ---------- Figure 2: Viewport Sensitivity ----------

def fig_viewport_sensitivity():
    try:
        data = _load_json("viewport_sensitivity")
    except FileNotFoundError:
        print("  Skipping viewport_sensitivity (no data)")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))

    fracs = sorted([float(f) for f in data.keys()])
    for cfg in ["Mono", "Mono-C", "SO", "SO-C", "SPDB"]:
        try:
            p50s = [data[str(f)][cfg]["p50"] for f in fracs]
            ax.plot([f * 100 for f in fracs], p50s, "o-",
                    label=cfg, color=COLORS.get(cfg),
                    marker=MARKERS.get(cfg, "o"), linewidth=1.5, markersize=5)
        except KeyError:
            pass

    ax.set_xlabel("Viewport Size (% of slide)")
    ax.set_ylabel("p50 Latency (ms)")
    ax.set_title("Viewport Size Sensitivity")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, "viewport_sensitivity")


# ---------- Figure 3: Hilbert Order Sensitivity ----------

def fig_hilbert_sensitivity():
    try:
        data = _load_json("hilbert_order_sensitivity")
    except FileNotFoundError:
        print("  Skipping hilbert_sensitivity (no data)")
        return

    fig, ax = plt.subplots(figsize=(4, 3))

    orders = sorted([int(p) for p in data.keys()])
    p50s = [data[str(p)]["p50"] for p in orders]
    p95s = [data[str(p)]["p95"] for p in orders]

    x = np.arange(len(orders))
    width = 0.35
    ax.bar(x - width / 2, p50s, width, label="p50", color=COLORS["SPDB"])
    ax.bar(x + width / 2, p95s, width, label="p95", color=COLORS["SPDB"], alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in orders])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Hilbert Order Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "hilbert_sensitivity")


# ---------- Figure 4: Hilbert vs Z-order ----------

def fig_hilbert_vs_zorder():
    try:
        data = _load_json("hilbert_vs_zorder")
    except FileNotFoundError:
        print("  Skipping hilbert_vs_zorder (no data)")
        return

    fig, ax = plt.subplots(figsize=(3.5, 3))
    names = ["Hilbert", "Z-order"]
    p50s = [data[n]["p50"] for n in names]
    p95s = [data[n]["p95"] for n in names]

    x = np.arange(len(names))
    width = 0.3
    ax.bar(x - width / 2, p50s, width, label="p50",
           color=[COLORS["Hilbert"], COLORS["Z-order"]])
    ax.bar(x + width / 2, p95s, width, label="p95",
           color=[COLORS["Hilbert"], COLORS["Z-order"]], alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Hilbert vs Z-order")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "hilbert_vs_zorder")


# ---------- Figure 5: Concurrency Throughput ----------

def fig_concurrency():
    try:
        data = _load_json("concurrent_throughput")
    except FileNotFoundError:
        print("  Skipping concurrency (no data)")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for cfg in ["Mono", "Mono-C", "SO", "SO-C", "SPDB"]:
        if cfg not in data:
            continue
        levels = sorted([int(n) for n in data[cfg].keys()])
        qps = [data[cfg][str(n)]["qps"] for n in levels]
        ax.plot(levels, qps, "o-", label=cfg, color=COLORS.get(cfg),
                marker=MARKERS.get(cfg, "o"), linewidth=1.5, markersize=5)

    ax.set_xlabel("Concurrent Clients")
    ax.set_ylabel("Queries/sec")
    ax.set_title("Concurrent Throughput (Q1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, "concurrency_throughput")


# ---------- Figure 6: kNN k-sweep ----------

def fig_knn_sweep():
    try:
        data = _load_json("knn_k_sweep")
    except FileNotFoundError:
        print("  Skipping knn_sweep (no data)")
        return

    fig, ax = plt.subplots(figsize=(4, 3))
    ks = sorted([int(k) for k in data.keys()])
    p50s = [data[str(k)]["p50"] for k in ks]
    p95s = [data[str(k)]["p95"] for k in ks]

    ax.plot(ks, p50s, "o-", label="p50", color=COLORS["SPDB"], linewidth=1.5)
    ax.plot(ks, p95s, "s--", label="p95", color=COLORS["SPDB"], alpha=0.6, linewidth=1)
    ax.set_xlabel("k (nearest neighbors)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("kNN Latency vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "knn_k_sweep")


# ---------- Figure 7: Cold vs Warm Cache ----------

def fig_cold_warm():
    try:
        cold = _load_json("cold_cache")
    except FileNotFoundError:
        print("  Skipping cold_warm (no data)")
        return
    try:
        warm = _load_json("q1_viewport")
    except FileNotFoundError:
        print("  Skipping cold_warm (no warm data)")
        return

    fig, ax = plt.subplots(figsize=(5, 3.2))

    cfgs = [c for c in ["Mono", "Mono-C", "SO", "SO-C", "SPDB"]
            if c in cold and c in warm.get("configs", {})]
    x = np.arange(len(cfgs))
    width = 0.3

    warm_p50 = [warm["configs"].get(c, {}).get("p50", 0) for c in cfgs]
    cold_p50 = [cold.get(c, {}).get("p50", 0) for c in cfgs]

    ax.bar(x - width / 2, warm_p50, width, label="Warm", color=COLORS["SPDB"])
    ax.bar(x + width / 2, cold_p50, width, label="Cold", color=COLORS["Mono"])
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, rotation=15)
    ax.set_ylabel("p50 Latency (ms)")
    ax.set_title("Warm vs Cold Cache")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "cold_warm_cache")


# ---------- Figure 8: Density Distribution ----------

def fig_density():
    try:
        data = _load_json("density_analysis")
    except FileNotFoundError:
        print("  Skipping density (no data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    densities = [v["density_per_mpx"] for v in data.values()]
    objects = [v["n_objects"] for v in data.values()]

    ax1.hist(densities, bins=15, color=COLORS["SPDB"], edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Nuclei per Mpx")
    ax1.set_ylabel("Number of Slides")
    ax1.set_title("Object Density Distribution")

    ax2.hist([o / 1e6 for o in objects], bins=15, color=COLORS["SO"], edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Objects (millions)")
    ax2.set_ylabel("Number of Slides")
    ax2.set_title("Objects per Slide")

    fig.tight_layout()
    _savefig(fig, "density_distribution")


# ---------- Figure 9: Storage Overhead ----------

def fig_storage():
    try:
        data = _load_json("storage_overhead")
    except FileNotFoundError:
        print("  Skipping storage (no data)")
        return

    fig, ax = plt.subplots(figsize=(6, 3.2))

    cfgs = [c for c in ["Mono", "Mono-T", "Mono-C", "SO", "SO-C", "SPDB", "SPDB-Z"]
            if c in data]
    table_mb = [data[c]["table_mb"] for c in cfgs]
    index_mb = [data[c]["index_mb"] for c in cfgs]

    x = np.arange(len(cfgs))
    width = 0.5
    ax.bar(x, table_mb, width, label="Table", color=COLORS["SPDB"])
    ax.bar(x, index_mb, width, bottom=table_mb, label="Indexes",
           color=COLORS["Mono"], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, rotation=20)
    ax.set_ylabel("Size (MB)")
    ax.set_title("Storage Overhead")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "storage_overhead")


# ---------- Figure 10: I/O Decomposition (NEW) ----------

def fig_io_decomposition():
    """Stacked bar chart: planning time vs execution time, hit vs read blocks."""
    try:
        data = _load_json("io_decomposition")
    except FileNotFoundError:
        print("  Skipping io_decomposition (no data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    cfgs = [c for c in ["Mono", "Mono-T", "Mono-C", "SO", "SO-C", "SPDB"]
            if c in data]
    x = np.arange(len(cfgs))
    width = 0.5

    plan_ms = [data[c]["planning_time"] for c in cfgs]
    exec_ms = [data[c]["execution_time"] for c in cfgs]

    ax1.bar(x, plan_ms, width, label="Planning", color="#2ca02c")
    ax1.bar(x, exec_ms, width, bottom=plan_ms, label="Execution", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfgs, rotation=20)
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Latency Decomposition")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    hits = [data[c]["shared_hit"] for c in cfgs]
    reads = [data[c]["shared_read"] for c in cfgs]

    ax2.bar(x, hits, width, label="Buffer Hits", color="#1f77b4")
    ax2.bar(x, reads, width, bottom=hits, label="Disk Reads", color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfgs, rotation=20)
    ax2.set_ylabel("Blocks")
    ax2.set_title("I/O Decomposition")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _savefig(fig, "io_decomposition")


# ---------- Figure 11: Cost Model Validation (NEW) ----------

def fig_cost_model_validation():
    """Predicted vs observed latency scatter plot."""
    try:
        data = _load_json("cost_model_validation")
    except FileNotFoundError:
        print("  Skipping cost_model_validation (no data)")
        return

    fig, ax = plt.subplots(figsize=(4.5, 4))

    validation = data.get("validation", {})
    for cfg, metrics in validation.items():
        ax.scatter(metrics["mean_predicted"], metrics["mean_observed"],
                   color=COLORS.get(cfg, "#333"),
                   marker=MARKERS.get(cfg, "o"),
                   s=80, label=f"{cfg} (R²={metrics['r_squared']:.2f})",
                   zorder=3)

    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Predicted Latency (ms)")
    ax.set_ylabel("Observed Latency (ms)")
    ax.set_title("Cost Model Validation")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    _savefig(fig, "cost_model_validation")


# ---------- Figure 12: Cost Surface (NEW) ----------

def fig_cost_surface():
    """Cost surface C(p, T) showing the optimal parameter region."""
    try:
        from spdb.cost_model_analytical import generate_cost_surface_data
    except ImportError:
        print("  Skipping cost_surface (cost_model_analytical not available)")
        return

    data = generate_cost_surface_data(n_objects=1_260_000, viewport_frac=0.05)
    T_values = np.array(data["T_values"])
    p_values = np.array(data["p_values"])
    cost_grid = data["cost_grid"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Contour plot: Cost(T) for selected p values
    for i, p in enumerate(p_values):
        if p in [6, 8, 10, 12]:
            ax1.plot(T_values / 1000, cost_grid[i, :], label=f"p={p}",
                     linewidth=1.5)

    opt = data["optimal"]
    ax1.axvline(opt["T_star"] / 1000, color="red", linestyle="--",
                alpha=0.5, label=f"T*={opt['T_star']//1000}K")
    ax1.set_xlabel("Bucket Target T (thousands)")
    ax1.set_ylabel("Estimated Cost (ms)")
    ax1.set_title("Cost vs Bucket Size")
    ax1.legend(fontsize=8)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    # Pruning rate surface
    pruning_grid = data["pruning_grid"]
    for i, p in enumerate(p_values):
        if p in [6, 8, 10, 12]:
            ax2.plot(T_values / 1000, pruning_grid[i, :] * 100,
                     label=f"p={p}", linewidth=1.5)
    ax2.set_xlabel("Bucket Target T (thousands)")
    ax2.set_ylabel("Pruning Rate (%)")
    ax2.set_title("Pruning Rate vs Bucket Size")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, "cost_surface")


# ---------- Figure 13: Architecture Diagram (NEW) ----------

def fig_architecture():
    """SPDB two-level partitioning architecture diagram."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Level 0: Parent table
    ax.add_patch(plt.Rectangle((3.5, 4.0), 3, 0.7, fill=True,
                                facecolor="#1f77b4", alpha=0.3, edgecolor="black"))
    ax.text(5, 4.35, "objects_spdb\n(Parent)", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Level 1: Slide partitions
    slide_colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    slide_labels = ["Slide A\n(1.4M obj)", "Slide B\n(2.1M obj)", "Slide C\n(0.8M obj)"]
    for i, (color, label) in enumerate(zip(slide_colors, slide_labels)):
        x = 1.0 + i * 3.2
        ax.add_patch(plt.Rectangle((x, 2.5), 2.6, 0.7, fill=True,
                                    facecolor=color, alpha=0.2, edgecolor="black"))
        ax.text(x + 1.3, 2.85, label, ha="center", va="center", fontsize=7)
        ax.annotate("", xy=(x + 1.3, 3.2), xytext=(5, 4.0),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # Level 2: Hilbert buckets (for Slide B)
    bucket_colors = ["#d62728", "#1f77b4", "#1f77b4", "#1f77b4",
                     "#bcbd22", "#1f77b4", "#1f77b4"]
    bucket_labels = ["H0", "H1", "H2", "H3", "H4", "H5", "H6"]
    bucket_alphas = [0.15, 0.15, 0.4, 0.4, 0.15, 0.4, 0.15]  # touched=darker

    for j in range(7):
        x = 0.3 + j * 1.3
        color = "#1f77b4" if bucket_alphas[j] > 0.2 else "#cccccc"
        alpha = 0.6 if bucket_alphas[j] > 0.2 else 0.2
        ax.add_patch(plt.Rectangle((x, 0.5), 1.0, 0.7, fill=True,
                                    facecolor=color, alpha=alpha, edgecolor="black",
                                    linewidth=0.5))
        label = bucket_labels[j]
        if bucket_alphas[j] > 0.2:
            label += "\n(scan)"
        else:
            label += "\n(pruned)"
        ax.text(x + 0.5, 0.85, label, ha="center", va="center", fontsize=6)

    # Arrow from Slide B to buckets
    for j in range(7):
        x = 0.3 + j * 1.3 + 0.5
        ax.annotate("", xy=(x, 1.2), xytext=(4.5, 2.5),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    # Labels
    ax.text(0.1, 4.35, "Level 0", fontsize=8, color="gray", style="italic")
    ax.text(0.1, 2.85, "Level 1\n(LIST by slide)", fontsize=7, color="gray", style="italic")
    ax.text(0.1, 0.85, "Level 2\n(RANGE by\nhilbert_key)", fontsize=6, color="gray", style="italic")

    # Pruning annotation
    ax.annotate("89% pruned", xy=(5, 0.2), fontsize=10, fontweight="bold",
                color="#d62728", ha="center")

    fig.tight_layout()
    _savefig(fig, "architecture")


# ---------- Master function ----------

def generate_all_figures():
    """Generate all paper figures."""
    print("Generating figures...")
    fig_q1_latency_cdf()
    fig_viewport_sensitivity()
    fig_hilbert_sensitivity()
    fig_hilbert_vs_zorder()
    fig_concurrency()
    fig_knn_sweep()
    fig_cold_warm()
    fig_density()
    fig_storage()
    fig_io_decomposition()
    fig_cost_model_validation()
    fig_cost_surface()
    fig_architecture()
    print(f"All figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
