"""Generate publication-quality figures for HCCI VLDB Research Track paper.

Reads results from results/raw/ and outputs to results/figures/.
All figures use consistent VLDB-style formatting.

Usage:
    python results/generate_hcci_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'results', 'raw')
FIG_DIR = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_json(name):
    path = os.path.join(RAW_DIR, name)
    with open(path) as f:
        return json.load(f)

tcga = load_json('hcci_benchmark.json')
osm = load_json('osm_hcci_benchmark.json')
cold = load_json('hcci_cold_cache.json')
osm_stor = load_json('osm_storage.json')
path_stor = load_json('storage_overhead.json')

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
HCCI_COLOR = '#2563eb'      # blue
GIST_COLOR = '#dc2626'      # red
BBOX_COLOR = '#f59e0b'      # amber
OSM_HCCI = '#059669'        # emerald
OSM_GIST = '#d97706'        # orange
COLD_HCCI = '#7c3aed'      # violet
COLD_GIST = '#be185d'      # pink

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  [{name}] saved')


# ===================================================================
# Figure 1: HCCI vs GiST Speedup — All 6 TCGA Queries
# ===================================================================
def fig1_tcga_speedup():
    queries = tcga['queries']
    qids = ['A', 'B', 'C', 'D', 'E', 'F']
    labels = [
        'A: Tumor\n(16.4%)',
        'B: Lymph\n(11.4%)',
        'C: T+L\n(27.8%)',
        'D: Epi\n(42.5%)',
        'E: All\n(100%)',
        'F: Tumor\n(1% vp)',
    ]
    hcci_p50 = [queries[q]['hcci']['p50'] for q in qids]
    gist_p50 = [queries[q]['gist']['p50'] for q in qids]
    speedups = [queries[q]['speedup_p50'] for q in qids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [3, 2]})

    # Left: grouped bar chart
    x = np.arange(len(qids))
    w = 0.35
    bars_h = ax1.bar(x - w/2, hcci_p50, w, label='HCCI', color=HCCI_COLOR, edgecolor='black', linewidth=0.4)
    bars_g = ax1.bar(x + w/2, gist_p50, w, label='GiST+Filter', color=GIST_COLOR, edgecolor='black', linewidth=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('p50 Latency (ms)')
    ax1.set_title('(a) TCGA Pathology: HCCI vs GiST (195M nuclei, 500 trials)')
    ax1.set_yscale('log')
    ax1.set_ylim(1, 600)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.text(5.3, 110, '100 ms', color='gray', fontsize=7, va='bottom')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Right: speedup bar chart
    colors = [HCCI_COLOR if s > 5 else '#60a5fa' for s in speedups]
    bars_s = ax2.barh(x, speedups, color=colors, edgecolor='black', linewidth=0.4)
    for i, (bar, sp) in enumerate(zip(bars_s, speedups)):
        ax2.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                 f'{sp:.0f}x', va='center', fontsize=9, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels([f'Q{q}' for q in qids])
    ax2.set_xlabel('Speedup (x)')
    ax2.set_title('(b) Speedup (p50)')
    ax2.set_xlim(0, max(speedups) * 1.25)
    ax2.axvline(x=1, color='gray', linestyle='-', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_yaxis()

    fig.tight_layout(w_pad=3)
    save(fig, 'hcci_tcga_speedup')


# ===================================================================
# Figure 2: Latency CDF for Query A — HCCI vs GiST vs BBox
# ===================================================================
def fig2_latency_cdf():
    """CDF from summary statistics (simulated from mean/std/p50/p95)."""
    queries = tcga['queries']
    bbox = tcga['bbox_comparison']

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Draw CDFs from summary stats using log-normal approximation
    for label, stats, color, ls in [
        ('HCCI', queries['A']['hcci'], HCCI_COLOR, '-'),
        ('GiST+ST_Intersects', bbox['gist_st_intersects'], GIST_COLOR, '-'),
        ('GiST+BBox', bbox['gist_bbox'], BBOX_COLOR, '--'),
    ]:
        mu = stats['mean']
        sigma = stats['std']
        p50 = stats['p50']
        p95 = stats['p95']

        # Generate synthetic CDF points
        xs = np.linspace(max(0.1, stats['min'] * 0.8), stats['max'] * 1.1, 500)
        # Use log-normal fit
        if p50 > 0 and sigma > 0:
            log_mu = np.log(p50)
            log_sigma = np.log(p95 / p50) / 1.645
            from scipy.stats import lognorm
            cdf_vals = lognorm.cdf(xs, s=log_sigma, scale=np.exp(log_mu))
        else:
            cdf_vals = np.zeros_like(xs)

        ax.plot(xs, cdf_vals, color=color, linestyle=ls, linewidth=1.8, label=label)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Query A: Tumor Viewport (5%, 200 trials)')
    ax.set_xscale('log')
    ax.set_xlim(0.5, 800)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.3)
    ax.text(0.6, 0.51, 'p50', color='gray', fontsize=7)
    ax.text(0.6, 0.96, 'p95', color='gray', fontsize=7)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.tight_layout()
    save(fig, 'hcci_latency_cdf')


# ===================================================================
# Figure 3: I/O Decomposition — Buffer Breakdown
# ===================================================================
def fig3_io_decomposition():
    io = tcga['io_decomposition']
    fig, ax = plt.subplots(figsize=(6, 3.5))

    queries_with_io = ['A', 'C']
    labels_map = {'A': 'Q-A: Tumor', 'C': 'Q-C: T+L'}

    x = np.arange(len(queries_with_io))
    w = 0.35

    for i, q in enumerate(queries_with_io):
        h_hit = io[q]['hcci']['shared_hit']
        h_read = io[q]['hcci']['shared_read']
        g_hit = io[q]['gist']['shared_hit']
        g_read = io[q]['gist']['shared_read']

        # HCCI bars
        ax.bar(i - w/2, h_hit, w, color=HCCI_COLOR, alpha=0.7, edgecolor='black', linewidth=0.4, label='HCCI hit' if i == 0 else '')
        ax.bar(i - w/2, h_read, w, bottom=h_hit, color=HCCI_COLOR, alpha=0.3, edgecolor='black', linewidth=0.4, hatch='///', label='HCCI read' if i == 0 else '')

        # GiST bars
        ax.bar(i + w/2, g_hit, w, color=GIST_COLOR, alpha=0.7, edgecolor='black', linewidth=0.4, label='GiST hit' if i == 0 else '')
        ax.bar(i + w/2, g_read, w, bottom=g_hit, color=GIST_COLOR, alpha=0.3, edgecolor='black', linewidth=0.4, hatch='///', label='GiST read' if i == 0 else '')

        # Reduction annotation
        total_hcci = h_hit + h_read
        total_gist = g_hit + g_read
        reduction = io[q]['buffer_reduction_pct']
        ax.annotate(f'{reduction:.0f}% fewer\nbuffers',
                    xy=(i, total_gist * 0.6), fontsize=8, ha='center',
                    fontweight='bold', color='#16a34a')

    ax.set_xticks(x)
    ax.set_xticklabels([labels_map[q] for q in queries_with_io])
    ax.set_ylabel('Buffer Accesses (pages)')
    ax.set_title('I/O Decomposition: HCCI Index-Only vs GiST+Heap')
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, 'hcci_io_decomposition')


# ===================================================================
# Figure 4: Cold vs Warm Cache Comparison
# ===================================================================
def fig4_cold_warm():
    warm_a = tcga['queries']['A']
    warm_c = tcga['queries']['C']
    cold_a = cold['query_A']
    cold_c = cold['query_C']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # --- Left: Latency comparison ---
    labels = ['Q-A\nWarm', 'Q-A\nCold', 'Q-C\nWarm', 'Q-C\nCold']
    hcci_vals = [warm_a['hcci']['p50'], cold_a['hcci']['p50'],
                 warm_c['hcci']['p50'], cold_c['hcci']['p50']]
    gist_vals = [warm_a['gist']['p50'], cold_a['gist']['p50'],
                 warm_c['gist']['p50'], cold_c['gist']['p50']]

    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w/2, hcci_vals, w, color=HCCI_COLOR, edgecolor='black', linewidth=0.4, label='HCCI')
    ax1.bar(x + w/2, gist_vals, w, color=GIST_COLOR, edgecolor='black', linewidth=0.4, label='GiST')

    # Speedup annotations
    for i, (h, g) in enumerate(zip(hcci_vals, gist_vals)):
        sp = g / h
        ax1.text(i, max(h, g) + 50, f'{sp:.0f}x', ha='center', fontsize=8, fontweight='bold', color='#16a34a')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('p50 Latency (ms)')
    ax1.set_title('(a) Warm vs Cold Cache Latency')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Right: Cold cache buffer breakdown ---
    cold_labels = ['Q-A', 'Q-C']
    hcci_reads = [cold_a['buffer_breakdown']['hcci_avg_read'],
                  cold_c['buffer_breakdown']['hcci_avg_read']]
    gist_reads = [cold_a['buffer_breakdown']['gist_avg_read'],
                  cold_c['buffer_breakdown']['gist_avg_read']]

    x2 = np.arange(len(cold_labels))
    ax2.bar(x2 - w/2, hcci_reads, w, color=HCCI_COLOR, edgecolor='black', linewidth=0.4, label='HCCI disk reads')
    ax2.bar(x2 + w/2, gist_reads, w, color=GIST_COLOR, edgecolor='black', linewidth=0.4, label='GiST disk reads')

    for i, (hr, gr) in enumerate(zip(hcci_reads, gist_reads)):
        ratio = gr / hr if hr > 0 else 0
        ax2.text(i, gr * 0.5, f'{ratio:.0f}x fewer\ndisk reads', ha='center',
                 fontsize=8, fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#16a34a', alpha=0.8))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cold_labels)
    ax2.set_ylabel('Avg Disk Reads (pages)')
    ax2.set_title('(b) Cold Cache: Disk I/O')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=3)
    save(fig, 'hcci_cold_warm')


# ===================================================================
# Figure 5: OSM Cross-Domain Validation
# ===================================================================
def fig5_osm_validation():
    oq = osm['queries']
    qids = ['A', 'B', 'C', 'D', 'E', 'F']
    labels = [
        'A: garage\n(7.6%)',
        'B: apts\n(2.1%)',
        'C: 2-class\n(9.6%)',
        'D: bld:yes\n(67.7%)',
        'E: top-10\n(92.8%)',
        'F: garage\n(1% vp)',
    ]
    hcci_p50 = [oq[q]['hcci']['p50'] for q in qids]
    gist_p50 = [oq[q]['gist_st_intersects']['p50'] for q in qids]
    bbox_p50 = [oq[q]['gist_bbox']['p50'] for q in qids]
    speedups = [oq[q]['speedup_p50'] for q in qids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [3, 2]})

    x = np.arange(len(qids))
    w = 0.25
    ax1.bar(x - w, hcci_p50, w, color=OSM_HCCI, edgecolor='black', linewidth=0.4, label='HCCI')
    ax1.bar(x, gist_p50, w, color=OSM_GIST, edgecolor='black', linewidth=0.4, label='GiST+Intersects')
    ax1.bar(x + w, bbox_p50, w, color=BBOX_COLOR, edgecolor='black', linewidth=0.4, label='GiST+BBox')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel('p50 Latency (ms)')
    ax1.set_title('(a) NYC OSM: HCCI vs GiST (1.5M POIs, 500 trials)')
    ax1.set_yscale('log')
    ax1.set_ylim(1, 500)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=7.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

    # Right: speedup chart
    colors_sp = [OSM_HCCI if s > 3 else '#6ee7b7' for s in speedups]
    bars_s = ax2.barh(x, speedups, color=colors_sp, edgecolor='black', linewidth=0.4)
    for i, (bar, sp) in enumerate(zip(bars_s, speedups)):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{sp:.1f}x', va='center', fontsize=9, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels([f'Q{q}' for q in qids])
    ax2.set_xlabel('Speedup vs GiST+Intersects (x)')
    ax2.set_title('(b) Speedup (p50)')
    ax2.set_xlim(0, max(speedups) * 1.3)
    ax2.axvline(x=1, color='gray', linestyle='-', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_yaxis()

    fig.tight_layout(w_pad=3)
    save(fig, 'hcci_osm_validation')


# ===================================================================
# Figure 6: Storage Overhead
# ===================================================================
def fig6_storage():
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Pathology dataset
    p_heap = 8058.5  # MB (from storage_overhead.json, Mono table_mb)
    p_gist = 1957.8  # MB (Mono index_mb includes GiST)
    # HCCI adds about 8.1% overhead (SPDB total - Mono total) / Mono total
    p_hcci = path_stor['SPDB']['total_mb'] - path_stor['Mono']['total_mb']  # ~813 MB

    # OSM dataset
    o_heap = osm_stor['heap_mb']        # 644.3
    o_gist = osm_stor['gist_mb']        # 63.4
    o_hcci = osm_stor['hcci_mb']        # 120.0

    labels = ['Pathology\n(42M rows)', 'OSM NYC\n(1.5M rows)']
    heap = [p_heap, o_heap]
    gist = [p_gist, o_gist]
    hcci = [p_hcci, o_hcci]

    x = np.arange(len(labels))
    w = 0.5

    ax.bar(x, heap, w, label='Heap', color='#94a3b8', edgecolor='black', linewidth=0.4)
    ax.bar(x, gist, w, bottom=heap, label='GiST Index', color=GIST_COLOR, alpha=0.6, edgecolor='black', linewidth=0.4)
    ax.bar(x, hcci, w, bottom=[h+g for h, g in zip(heap, gist)], label='HCCI Index', color=HCCI_COLOR, alpha=0.6, edgecolor='black', linewidth=0.4)

    # Overhead percentages
    for i, (h, g, c) in enumerate(zip(heap, gist, hcci)):
        total_existing = h + g
        overhead_pct = c / total_existing * 100
        ax.text(i, h + g + c + 100, f'+{overhead_pct:.0f}% overhead',
                ha='center', fontsize=9, fontweight='bold', color=HCCI_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Storage (MB)')
    ax.set_title('HCCI Storage Overhead')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, 'hcci_storage_overhead')


# ===================================================================
# Figure 7: Cost Model Validation — Predicted vs Measured Speedup
# ===================================================================
def fig7_cost_model():
    cm = tcga['cost_model']
    fig, ax = plt.subplots(figsize=(4.5, 4))

    measured = {
        'Tumor': tcga['queries']['A']['speedup_p50'],
        'Lymphocyte': tcga['queries']['B']['speedup_p50'],
        'Tumor+Lymph': tcga['queries']['C']['speedup_p50'],
        'Epithelial': tcga['queries']['D']['speedup_p50'],
    }

    predicted = {k: v['speedup'] for k, v in cm.items()}
    selectivities = {k: v['class_selectivity'] for k, v in cm.items()}

    # Manual offsets to avoid overlap
    offsets = {
        'Tumor': (-5, -30),
        'Lymphocyte': (18, 5),
        'Tumor+Lymph': (15, 5),
        'Epithelial': (-18, -25),
    }
    for label in predicted:
        p = predicted[label]
        m = measured[label]
        s = selectivities[label]
        ax.scatter(p, m, s=80, zorder=5, color=HCCI_COLOR, edgecolor='black', linewidth=0.5)
        ox, oy = offsets.get(label, (12, -5))
        ax.annotate(f'{label}\n(s={s:.0%})', (p, m),
                    textcoords='offset points', xytext=(ox, oy), fontsize=7.5,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Perfect prediction line
    lim_max = max(max(predicted.values()), max(measured.values())) * 1.15
    ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, linewidth=1, label='Perfect prediction')

    ax.set_xlabel('Predicted Speedup (cost model)')
    ax.set_ylabel('Measured Speedup (p50)')
    ax.set_title('Cost Model Validation')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, 'hcci_cost_model')


# ===================================================================
# Figure 8: Cross-Domain Comparison — Pathology vs OSM
# ===================================================================
def fig8_cross_domain():
    """Side-by-side comparison of speedups across both datasets."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Pathology queries (selected)
    p_labels = ['Tumor\n(16.4%)', 'Lymph\n(11.4%)', 'T+L\n(27.8%)', 'Epi\n(42.5%)', 'All\n(100%)']
    p_speedups = [tcga['queries'][q]['speedup_p50'] for q in ['A', 'B', 'C', 'D', 'E']]

    # OSM queries (selected)
    o_labels = ['garage\n(7.6%)', 'apts\n(2.1%)', '2-class\n(9.6%)', 'bld:yes\n(67.7%)', 'top-10\n(92.8%)']
    o_speedups = [osm['queries'][q]['speedup_p50'] for q in ['A', 'B', 'C', 'D', 'E']]

    x = np.arange(5)
    w = 0.35
    ax.bar(x - w/2, p_speedups, w, color=HCCI_COLOR, edgecolor='black', linewidth=0.4, label='Pathology (195M)')
    ax.bar(x + w/2, o_speedups, w, color=OSM_HCCI, edgecolor='black', linewidth=0.4, label='OSM NYC (1.5M)')

    # Selectivity labels
    sel_labels = ['~8-16%', '~2-11%', '~10-28%', '~43-68%', '~93-100%']
    for i, sl in enumerate(sel_labels):
        ax.text(i, max(p_speedups[i], o_speedups[i]) + 2, sl,
                ha='center', fontsize=7, color='gray')

    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{chr(65+i)}' for i in range(5)])
    ax.set_ylabel('Speedup over GiST (x)')
    ax.set_title('Cross-Domain HCCI Speedup: Pathology vs OSM')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, 'hcci_cross_domain')


# ===================================================================
# Main
# ===================================================================
def main():
    print('Generating HCCI figures...')
    fig1_tcga_speedup()
    fig2_latency_cdf()
    fig3_io_decomposition()
    fig4_cold_warm()
    fig5_osm_validation()
    fig6_storage()
    fig7_cost_model()
    fig8_cross_domain()
    print(f'\nAll figures saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
