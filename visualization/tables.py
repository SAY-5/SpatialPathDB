"""Generate LaTeX tables from benchmark results."""

import os
import json

from spdb import config


def _load_json(name):
    path = os.path.join(config.RAW_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def _save_table(latex, name):
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    path = os.path.join(config.TABLES_DIR, f"{name}.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved {name}.tex")
    return path


def table_q1_master():
    """Table: Q1 viewport latency across all 7 configs."""
    data = _load_json("q1_viewport")
    cfgs = data["configs"]

    latex = r"""\begin{table}[t]
\centering
\caption{Q1 viewport latency (ms) across storage layouts -- TCGA BLCA.}
\label{tab:q1_master}
\begin{tabular}{@{}lrrrrl@{}}
\toprule
\textbf{Config} & \textbf{p50} & \textbf{p95} & \textbf{mean} & \textbf{std} & \textbf{vs.\ SPDB} \\
\midrule
"""
    spdb_p50 = cfgs.get("SPDB", {}).get("p50", 1)
    for name in ["Mono", "Mono-T", "Mono-C", "SO", "SO-C", "SPDB"]:
        if name in cfgs:
            s = cfgs[name]
            if name == "SPDB":
                prefix = r"\textbf{SPDB}"
                vs = "---"
                latex += (f"{prefix} & \\textbf{{{s['p50']:.0f}}} & "
                         f"\\textbf{{{s['p95']:.0f}}} & {s['mean']:.0f} & "
                         f"{s['std']:.0f} & \\textbf{{{vs}}} \\\\\n")
            else:
                ratio = s["p50"] / spdb_p50 if spdb_p50 > 0 else 0
                vs = f"${ratio:.2f}\\times$"
                latex += (f"{name} & {s['p50']:.0f} & {s['p95']:.0f} & "
                         f"{s['mean']:.0f} & {s['std']:.0f} & {vs} \\\\\n")
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return _save_table(latex, "q1_master")


def table_all_queries():
    """Table: All query types performance for SPDB."""
    tables = {}
    for name in ["q1_viewport", "q2_knn_k50", "q3_aggregation", "q4_spatial_join"]:
        try:
            d = _load_json(name)
            if "configs" in d and "SPDB" in d["configs"]:
                tables[name] = d["configs"]["SPDB"]
        except FileNotFoundError:
            pass

    latex = r"""\begin{table}[t]
\centering
\caption{SPDB query performance across all types.}
\label{tab:allqueries}
\begin{tabular}{@{}llrrr@{}}
\toprule
& \textbf{Query} & \textbf{p50 (ms)} & \textbf{p95 (ms)} & \textbf{$n$} \\
\midrule
"""
    query_names = {
        "q1_viewport": ("Q1", "Viewport (range)"),
        "q2_knn_k50": ("Q2", "kNN ($k{=}50$)"),
        "q3_aggregation": ("Q3", "Aggregation"),
        "q4_spatial_join": ("Q4", "Spatial join"),
    }
    for key, (qid, qlabel) in query_names.items():
        if key in tables:
            s = tables[key]
            latex += f"{qid} & {qlabel} & {s['p50']:.0f} & {s['p95']:.0f} & {s['n']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return _save_table(latex, "all_queries")


def table_concurrent():
    """Table: Concurrent throughput (expanded)."""
    try:
        data = _load_json("concurrent_throughput")
    except FileNotFoundError:
        print("  Skipping concurrent table (no data)")
        return

    cfgs_present = [c for c in ["Mono", "Mono-C", "SO", "SO-C", "SPDB"] if c in data]
    ncols = len(cfgs_present)

    header = " & ".join([f"\\textbf{{{c} QPS}}" for c in cfgs_present])
    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Concurrent Q1 throughput (30-sec windows).}}
\\label{{tab:throughput}}
\\begin{{tabular}}{{@{{}}l{"r" * ncols}@{{}}}}
\\toprule
\\textbf{{Clients}} & {header} \\\\
\\midrule
"""
    first_cfg = cfgs_present[0]
    levels = sorted([int(n) for n in data.get(first_cfg, {}).keys()])
    for n in levels:
        vals = []
        for c in cfgs_present:
            qps = data.get(c, {}).get(str(n), {}).get("qps", 0)
            if c == "SPDB":
                vals.append(f"\\textbf{{{qps:.1f}}}")
            else:
                vals.append(f"{qps:.1f}")
        latex += f"{n} & {' & '.join(vals)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return _save_table(latex, "concurrent_throughput")


def table_hilbert_sensitivity():
    """Table: Hilbert order sensitivity."""
    try:
        data = _load_json("hilbert_order_sensitivity")
    except FileNotFoundError:
        print("  Skipping hilbert sensitivity table (no data)")
        return

    latex = r"""\begin{table}[t]
\centering
\caption{Hilbert order sensitivity: Q1 viewport latency.}
\label{tab:hilbert}
\begin{tabular}{@{}rrrrl@{}}
\toprule
$p$ & \textbf{Grid} & \textbf{p50 (ms)} & \textbf{p95 (ms)} & \textbf{vs.\ $p{=}8$} \\
\midrule
"""
    orders = sorted([int(p) for p in data.keys()])
    p8_p50 = data.get("8", {}).get("p50", 1)
    for p in orders:
        s = data[str(p)]
        grid = f"{(1 << (2*p)):,}".replace(",", "{,}")
        if p == 8:
            vs = "---"
            latex += (f"\\textbf{{{p}}} & \\textbf{{{grid}}} & "
                     f"\\textbf{{{s['p50']:.0f}}} & \\textbf{{{s['p95']:.0f}}} & "
                     f"\\textbf{{{vs}}} \\\\\n")
        else:
            ratio = s["p50"] / p8_p50
            vs = f"${ratio:.2f}\\times$" + (" slower" if ratio > 1 else " faster")
            latex += f"{p} & {grid} & {s['p50']:.0f} & {s['p95']:.0f} & {vs} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return _save_table(latex, "hilbert_sensitivity")


def table_storage():
    """Table: Storage overhead comparison -- all configs."""
    try:
        data = _load_json("storage_overhead")
    except FileNotFoundError:
        print("  Skipping storage table (no data)")
        return

    latex = r"""\begin{table}[t]
\centering
\caption{Storage overhead across storage layouts.}
\label{tab:storage}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Config} & \textbf{Rows} & \textbf{Table (MB)} & \textbf{Index (MB)} & \textbf{Total (MB)} & \textbf{B/row} \\
\midrule
"""
    for name in ["Mono", "Mono-T", "Mono-C", "SO", "SO-C", "SPDB", "SPDB-Z"]:
        if name in data:
            d = data[name]
            latex += (f"{name} & {d['row_count']:,} & {d['table_mb']:.0f} & "
                     f"{d['index_mb']:.0f} & {d['total_mb']:.0f} & "
                     f"{d['bytes_per_row']:.0f} \\\\\n")

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return _save_table(latex, "storage_overhead")


def table_io_decomposition():
    """Table: I/O decomposition across configs (NEW)."""
    try:
        data = _load_json("io_decomposition")
    except FileNotFoundError:
        print("  Skipping io_decomposition table (no data)")
        return

    latex = r"""\begin{table}[t]
\centering
\caption{I/O decomposition: Q1 viewport query (EXPLAIN BUFFERS average).}
\label{tab:io_decomposition}
\scriptsize
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Config} & \textbf{Plan (ms)} & \textbf{Exec (ms)} & \textbf{Buf Hits} & \textbf{Disk Reads} & \textbf{Hit\%} \\
\midrule
"""
    for name in ["Mono", "Mono-T", "Mono-C", "SO", "SO-C", "SPDB"]:
        if name in data:
            d = data[name]
            hr = d["hit_ratio"] * 100
            latex += (f"{name} & {d['planning_time']:.1f} & {d['execution_time']:.1f} & "
                     f"{d['shared_hit']:.0f} & {d['shared_read']:.0f} & "
                     f"{hr:.0f}\\% \\\\\n")

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return _save_table(latex, "io_decomposition")


def generate_all_tables():
    """Generate all LaTeX tables."""
    print("Generating LaTeX tables...")
    table_q1_master()
    table_all_queries()
    table_concurrent()
    table_hilbert_sensitivity()
    table_storage()
    table_io_decomposition()
    print(f"All tables saved to {config.TABLES_DIR}")


if __name__ == "__main__":
    generate_all_tables()
