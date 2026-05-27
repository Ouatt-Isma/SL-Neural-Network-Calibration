"""
Comprehensive calibration-based trust analysis.

This module implements and visualises three evaluations requested in the
chapter TODO:

  1. ECE comparison  – ECE plotted alongside SL trust metrics over training
                       epochs, illustrating the richer information provided by
                       the SL framework.

  2. Cluster variation – how the global trust opinion changes as the number of
                        probability clusters M varies, for a fixed (final)
                        epoch.

  3. Dynamic assessment – per-prediction trust scores obtained by looking up
                          pre-computed cluster opinions at inference time.

Usage:
    python analysis.py --mnist_dir  data/MNIST_PRED  \
                       --cifar_dir  data/CIFAR10_PRED \
                       --outdir     img/cal

All figures are saved as PDF files ready for LaTeX inclusion.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')        # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker

from trustopinion import TrustOpinion

# ─────────────────────────────────────────────────────────────────────────────
# Colour / style constants
# ─────────────────────────────────────────────────────────────────────────────
COLORS_10 = cm.viridis(np.linspace(0, 1, 10))
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 13

matplotlib.rcParams.update({'font.size': LABEL_FONTSIZE})


# ═════════════════════════════════════════════════════════════════════════════
# Core calibration / trust utilities
# ═════════════════════════════════════════════════════════════════════════════

def make_representatives(n_clusters: int):
    """Return (representatives, half_step) for a uniform partition of [0,1]."""
    step = 1.0 / n_clusters
    half_step = step / 2.0
    reps = [round(half_step + i * step, 6) for i in range(n_clusters)]
    return reps, half_step


def compute_opinion_for_class(class_value: int,
                               reps: list,
                               half_step: float,
                               data: pd.DataFrame) -> TrustOpinion:
    """
    Compute the fused SL trust opinion for *class_value* using cumulative
    fusion over all probability clusters.

    Returns
    -------
    TrustOpinion
        The fused opinion (b, d, u, a).
    """
    arr_l = data['True Label'].values
    arr_p = data[f'Class_{class_value}_Probability'].values

    total_pos, total_neg = 0, 0
    for rp in reps:
        mask = (arr_p >= rp - half_step) & (arr_p < rp + half_step)
        n_i = mask.sum()
        t_i = int((arr_l[mask] == class_value).sum())
        total_pos += t_i
        total_neg += int(round(abs(t_i - n_i * rp)))

    return TrustOpinion.ev2tdu(total_pos, total_neg)


def compute_global_opinion(data: pd.DataFrame,
                            n_clusters: int = 10) -> TrustOpinion:
    """
    Compute the global trust opinion for a NN by fusing per-class opinions.

    Iterates over all 10 classes, computes each class opinion, then fuses
    them cumulatively.

    Parameters
    ----------
    data : pd.DataFrame
        CSV with columns Class_0_Probability … Class_9_Probability, True Label
    n_clusters : int
        Number of probability clusters M.

    Returns
    -------
    TrustOpinion
    """
    reps, half_step = make_representatives(n_clusters)
    n_classes = sum(1 for c in data.columns if c.startswith('Class_') and
                    c.endswith('_Probability'))

    fused = None
    for c in range(n_classes):
        op_c = compute_opinion_for_class(c, reps, half_step, data)
        fused = op_c if fused is None else TrustOpinion.cumFuse(fused, op_c)
    return fused


def build_cluster_lookup(data: pd.DataFrame,
                          n_clusters: int = 10) -> dict:
    """
    Build a lookup table: {(class_value, cluster_rep) -> TrustOpinion}.

    Used for dynamic (per-prediction) trust assessment.
    """
    reps, half_step = make_representatives(n_clusters)
    n_classes = sum(1 for c in data.columns if c.startswith('Class_') and
                    c.endswith('_Probability'))
    lookup = {}
    for c in range(n_classes):
        arr_l = data['True Label'].values
        arr_p = data[f'Class_{c}_Probability'].values
        for rp in reps:
            mask = (arr_p >= rp - half_step) & (arr_p < rp + half_step)
            n_i = mask.sum()
            t_i = int((arr_l[mask] == c).sum())
            neg_ev = int(round(abs(t_i - n_i * rp)))
            lookup[(c, rp)] = TrustOpinion.ev2tdu(t_i, neg_ev)
    return lookup


def get_per_prediction_trust(data: pd.DataFrame,
                               lookup: dict,
                               n_clusters: int = 10) -> dict:
    """
    Assign a trust opinion to every prediction using the pre-computed lookup.

    For each sample, finds the predicted class and its confidence, maps it to
    the nearest cluster representative, and retrieves the stored opinion.

    Returns
    -------
    dict with keys 'belief', 'disbelief', 'uncertainty', 'confidence',
    'correct' (bool array).
    """
    reps, half_step = make_representatives(n_clusters)
    reps_arr = np.array(reps)

    n_classes = sum(1 for c in data.columns if c.startswith('Class_') and
                    c.endswith('_Probability'))
    prob_cols = [f'Class_{i}_Probability' for i in range(n_classes)]
    probs = data[prob_cols].values          # (N, C)
    true_labels = data['True Label'].values

    pred_classes = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(probs)), pred_classes]  # max prob

    beliefs, disbeliefs, uncertainties = [], [], []
    for i, (pred_c, conf) in enumerate(zip(pred_classes, confidences)):
        # Find nearest cluster representative
        nearest_rp = reps_arr[np.argmin(np.abs(reps_arr - conf))]
        op = lookup.get((pred_c, nearest_rp),
                        TrustOpinion.vacuous())
        beliefs.append(op.t)
        disbeliefs.append(op.d)
        uncertainties.append(op.u)

    return {
        'belief':      np.array(beliefs),
        'disbelief':   np.array(disbeliefs),
        'uncertainty': np.array(uncertainties),
        'confidence':  confidences,
        'correct':     (pred_classes == true_labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ECE utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(data: pd.DataFrame, n_bins: int = 10) -> float:
    """
    Compute the Expected Calibration Error (ECE) from a predictions DataFrame.

    Uses the maximum predicted probability as the confidence.

    Parameters
    ----------
    data : pd.DataFrame
        Columns Class_0_Probability … Class_C_Probability, True Label.
    n_bins : int
        Number of confidence bins (default 10, matching the standard 10-bin
        ECE used in calibration literature).

    Returns
    -------
    float
    """
    n_classes = sum(1 for c in data.columns if c.startswith('Class_') and
                    c.endswith('_Probability'))
    prob_cols = [f'Class_{i}_Probability' for i in range(n_classes)]
    probs = data[prob_cols].values
    true_labels = data['True Label'].values

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == true_labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(true_labels)
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═════════════════════════════════════════════════════════════════════════════

def list_epoch_files(directory: str) -> list:
    """
    Return sorted list of (epoch, bef_path, aft_path) tuples for all
    checkpoints found in *directory*.
    """
    pattern = re.compile(r'^bef_(\d+)\.csv$')
    epochs = []
    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if m:
            e = int(m.group(1))
            bef = os.path.join(directory, f'bef_{e}.csv')
            aft = os.path.join(directory, f'aft_{e}.csv')
            if os.path.isfile(aft):
                epochs.append((e, bef, aft))
    return sorted(epochs, key=lambda x: x[0])


def load_loss_history(directory: str):
    """Load loss_history.npy if present, else return None."""
    path = os.path.join(directory, 'loss_history.npy')
    if os.path.isfile(path):
        return np.load(path, allow_pickle=True).item()
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 – Trust metrics + ECE over training epochs
# ═════════════════════════════════════════════════════════════════════════════

def plot_trust_and_ece(directory: str, outdir: str,
                        dataset_name: str,
                        n_clusters: int = 10) -> None:
    """
    Generate a 2×4 grid figure:
      - Row 0: Before calibration  (belief, disbelief, uncertainty, ECE)
      - Row 1: After  calibration  (belief, disbelief, uncertainty, ECE)

    Each of the first three columns shows the per-label curves; the fourth
    column shows the scalar ECE over epochs.

    The figure is saved to <outdir>/<dataset_name>_ECE.pdf.
    """
    epoch_data = list_epoch_files(directory)
    if not epoch_data:
        print(f"[plot_trust_and_ece] No files found in {directory}")
        return

    reps, half_step = make_representatives(n_clusters)

    epochs = []
    metrics_bef = {'b': [], 'd': [], 'u': [], 'ece': []}
    metrics_aft = {'b': [], 'd': [], 'u': [], 'ece': []}
    # Per-class beliefs (list-of-lists, one inner list per class)
    n_classes = 10
    per_class_bef = {k: [[] for _ in range(n_classes)] for k in 'bdu'}
    per_class_aft = {k: [[] for _ in range(n_classes)] for k in 'bdu'}

    for epoch, bef_path, aft_path in epoch_data:
        d_bef = pd.read_csv(bef_path)
        d_aft = pd.read_csv(aft_path)

        op_bef = compute_global_opinion(d_bef, n_clusters)
        op_aft = compute_global_opinion(d_aft, n_clusters)

        metrics_bef['b'].append(op_bef.t)
        metrics_bef['d'].append(op_bef.d)
        metrics_bef['u'].append(op_bef.u)
        metrics_bef['ece'].append(compute_ece(d_bef))

        metrics_aft['b'].append(op_aft.t)
        metrics_aft['d'].append(op_aft.d)
        metrics_aft['u'].append(op_aft.u)
        metrics_aft['ece'].append(compute_ece(d_aft))

        for c in range(n_classes):
            op_c_bef = compute_opinion_for_class(c, reps, half_step, d_bef)
            op_c_aft = compute_opinion_for_class(c, reps, half_step, d_aft)
            per_class_bef['b'][c].append(op_c_bef.t)
            per_class_bef['d'][c].append(op_c_bef.d)
            per_class_bef['u'][c].append(op_c_bef.u)
            per_class_aft['b'][c].append(op_c_aft.t)
            per_class_aft['d'][c].append(op_c_aft.d)
            per_class_aft['u'][c].append(op_c_aft.u)

        epochs.append(epoch)

    # ── Plotting ──────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 4, figsize=(16, 6),
                             gridspec_kw={'wspace': 0.35, 'hspace': 0.45})

    col_labels = ['Belief (Trust)', 'Disbelief', 'Uncertainty', 'ECE']
    row_labels = ['Before Calibration', 'After Calibration']
    metric_keys_per_class = ['b', 'd', 'u']

    for row, (metrics, per_class, row_lbl) in enumerate([
            (metrics_bef, per_class_bef, 'Before Calibration'),
            (metrics_aft, per_class_aft, 'After Calibration'),
    ]):
        for col, (mk, col_lbl) in enumerate(zip(metric_keys_per_class,
                                                 col_labels[:3])):
            ax = axs[row][col]
            for c in range(n_classes):
                ax.plot(epochs, per_class[mk][c], color=COLORS_10[c],
                        linewidth=1.0, alpha=0.9)
            ax.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE)
            ax.set_ylabel(col_lbl, fontsize=LABEL_FONTSIZE)
            if row == 0:
                ax.set_title(col_lbl, fontsize=TITLE_FONTSIZE)

        # ECE column (col 3)
        ax_ece = axs[row][3]
        ax_ece.plot(epochs, metrics['ece'], color='steelblue',
                    linewidth=2.0, marker='o', markersize=3)
        ax_ece.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE)
        ax_ece.set_ylabel('ECE', fontsize=LABEL_FONTSIZE)
        ax_ece.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))
        if row == 0:
            ax_ece.set_title('ECE', fontsize=TITLE_FONTSIZE)

        # Row label on the left
        axs[row][0].annotate(row_lbl, xy=(-0.35, 0.5),
                              xycoords='axes fraction', rotation=90,
                              va='center', ha='center',
                              fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Shared colorbar for class labels
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=0, vmax=9))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, location='right', fraction=0.015,
                        pad=0.01)
    cbar.set_label('Class', fontsize=LABEL_FONTSIZE)
    cbar.set_ticks(np.arange(0, 10))

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'{dataset_name}_ECE.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {dataset_name}_ECE.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 – Effect of varying the number of clusters M
# ═════════════════════════════════════════════════════════════════════════════

def plot_cluster_variation(directory: str, outdir: str,
                            dataset_name: str,
                            m_values: list = None) -> None:
    """
    Show how the global trust opinion (b, d, u) changes as M varies,
    evaluated at the *last* available epoch for both before/after calibration.

    Two side-by-side sub-figures (before | after) each show three lines
    (belief, disbelief, uncertainty) as functions of M.

    Saved to <outdir>/<dataset_name>_clusters.pdf.
    """
    if m_values is None:
        m_values = [2, 3, 5, 8, 10, 15, 20, 30, 50]

    epoch_data = list_epoch_files(directory)
    if not epoch_data:
        print(f"[plot_cluster_variation] No files found in {directory}")
        return

    # Use the last available checkpoint
    last_epoch, bef_path, aft_path = epoch_data[-1]
    d_bef = pd.read_csv(bef_path)
    d_aft = pd.read_csv(aft_path)

    def sweep_m(data):
        bs, ds, us = [], [], []
        for m in m_values:
            op = compute_global_opinion(data, n_clusters=m)
            bs.append(op.t)
            ds.append(op.d)
            us.append(op.u)
        return np.array(bs), np.array(ds), np.array(us)

    bs_bef, ds_bef, us_bef = sweep_m(d_bef)
    bs_aft, ds_aft, us_aft = sweep_m(d_aft)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4),
                             sharey=False, gridspec_kw={'wspace': 0.3})

    for ax, (bs, ds, us), title in zip(
            axs,
            [(bs_bef, ds_bef, us_bef), (bs_aft, ds_aft, us_aft)],
            ['Before Calibration', 'After Calibration']):
        ax.plot(m_values, bs, 'o-', color='steelblue',  label='Belief',
                linewidth=1.8, markersize=5)
        ax.plot(m_values, ds, 's--', color='firebrick', label='Disbelief',
                linewidth=1.8, markersize=5)
        ax.plot(m_values, us, '^:', color='seagreen',   label='Uncertainty',
                linewidth=1.8, markersize=5)
        ax.set_xlabel('Number of clusters $M$', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Opinion component', fontsize=LABEL_FONTSIZE)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.set_xticks(m_values)
        ax.set_xticklabels([str(m) for m in m_values], rotation=45)
        ax.legend(fontsize=LABEL_FONTSIZE - 1)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle(
        f'{dataset_name} — Trust opinion vs. number of clusters '
        f'(epoch {last_epoch})',
        fontsize=TITLE_FONTSIZE,
    )
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'{dataset_name}_clusters.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {dataset_name}_clusters.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 – Dynamic (per-prediction) trust assessment
# ═════════════════════════════════════════════════════════════════════════════

def plot_dynamic_assessment(directory: str, outdir: str,
                              dataset_name: str,
                              n_clusters: int = 10) -> None:
    """
    Illustrate the dynamic (per-prediction) trust assessment mechanism.

    For the last available epoch, after calibration:
      - Panel A (left):  Scatter plot — predicted confidence vs. belief,
                         coloured by correctness.
      - Panel B (middle): Histograms of per-prediction belief for correct
                          vs. incorrect predictions.
      - Panel C (right):  Mean opinion components (b, d, u) binned by
                          predicted confidence — a calibration-reliability
                          diagram for the trust framework.

    Saved to <outdir>/<dataset_name>_dynamic.pdf.
    """
    epoch_data = list_epoch_files(directory)
    if not epoch_data:
        print(f"[plot_dynamic_assessment] No files found in {directory}")
        return

    last_epoch, _, aft_path = epoch_data[-1]
    d_aft = pd.read_csv(aft_path)

    # Build lookup table from this epoch's data
    lookup = build_cluster_lookup(d_aft, n_clusters=n_clusters)
    result = get_per_prediction_trust(d_aft, lookup, n_clusters=n_clusters)

    belief      = result['belief']
    disbelief   = result['disbelief']
    uncertainty = result['uncertainty']
    confidence  = result['confidence']
    correct     = result['correct']

    # ── Panel A: confidence vs belief scatter (sample for speed) ─────────
    rng = np.random.default_rng(42)
    idx = rng.choice(len(belief),
                     size=min(2000, len(belief)),
                     replace=False)

    # ── Panel C: mean opinion per confidence bin ──────────────────────────
    n_bins = n_clusters
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_b, mean_d, mean_u = [], [], []
    for i in range(n_bins):
        mask = (confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1])
        if mask.sum() > 0:
            mean_b.append(belief[mask].mean())
            mean_d.append(disbelief[mask].mean())
            mean_u.append(uncertainty[mask].mean())
        else:
            mean_b.append(np.nan)
            mean_d.append(np.nan)
            mean_u.append(np.nan)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={'wspace': 0.35})

    # --- Panel A: scatter ---
    ax = axs[0]
    colors_scatter = np.where(correct[idx], '#2196F3', '#F44336')
    ax.scatter(confidence[idx], belief[idx], c=colors_scatter,
               s=8, alpha=0.4, rasterized=True)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
               markersize=7, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
               markersize=7, label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, fontsize=LABEL_FONTSIZE - 1)
    ax.set_xlabel('Predicted confidence', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Trust belief $b$', fontsize=LABEL_FONTSIZE)
    ax.set_title('Confidence vs. Trust Belief', fontsize=TITLE_FONTSIZE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(linestyle='--', alpha=0.3)

    # --- Panel B: histogram of belief ---
    ax = axs[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(belief[correct],   bins=bins, density=True, alpha=0.6,
            color='#2196F3', label='Correct', edgecolor='none')
    ax.hist(belief[~correct],  bins=bins, density=True, alpha=0.6,
            color='#F44336', label='Incorrect', edgecolor='none')
    ax.set_xlabel('Trust belief $b$ per prediction', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Density', fontsize=LABEL_FONTSIZE)
    ax.set_title('Distribution of Per-Prediction Belief', fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LABEL_FONTSIZE - 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Panel C: mean opinion per confidence bin ---
    ax = axs[2]
    ax.plot(bin_centres, mean_b, 'o-',   color='steelblue',  linewidth=1.8,
            markersize=5, label='Belief')
    ax.plot(bin_centres, mean_d, 's--',  color='firebrick',  linewidth=1.8,
            markersize=5, label='Disbelief')
    ax.plot(bin_centres, mean_u, '^:',   color='seagreen',   linewidth=1.8,
            markersize=5, label='Uncertainty')
    ax.set_xlabel('Predicted confidence', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Mean opinion component', fontsize=LABEL_FONTSIZE)
    ax.set_title('Mean Trust Opinion per Confidence Bin', fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LABEL_FONTSIZE - 1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(linestyle='--', alpha=0.3)

    fig.suptitle(
        f'{dataset_name} — Dynamic trust assessment at inference '
        f'(epoch {last_epoch}, after calibration)',
        fontsize=TITLE_FONTSIZE,
    )
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'{dataset_name}_dynamic.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {dataset_name}_dynamic.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 – Original trust-metrics figure (before / after, all classes)
# Reproduces MNIST_ALL.pdf / CIFAR_ALL.pdf referenced in the LaTeX chapter.
# ═════════════════════════════════════════════════════════════════════════════

def plot_trust_all(directory: str, outdir: str,
                   dataset_name: str,
                   n_clusters: int = 10) -> None:
    """
    Reproduce the original 2×3 trust-metrics figure (before/after calibration)
    with per-label curves colour-coded by viridis.

    Row 0 = before calibration, Row 1 = after.
    Columns: belief, disbelief, uncertainty.
    """
    epoch_data = list_epoch_files(directory)
    if not epoch_data:
        print(f"[plot_trust_all] No files found in {directory}")
        return

    reps, half_step = make_representatives(n_clusters)
    n_classes = 10
    epochs = []
    per_class_bef = {k: [[] for _ in range(n_classes)] for k in 'bdu'}
    per_class_aft = {k: [[] for _ in range(n_classes)] for k in 'bdu'}

    for epoch, bef_path, aft_path in epoch_data:
        d_bef = pd.read_csv(bef_path)
        d_aft = pd.read_csv(aft_path)
        for c in range(n_classes):
            op_b = compute_opinion_for_class(c, reps, half_step, d_bef)
            op_a = compute_opinion_for_class(c, reps, half_step, d_aft)
            per_class_bef['b'][c].append(op_b.t)
            per_class_bef['d'][c].append(op_b.d)
            per_class_bef['u'][c].append(op_b.u)
            per_class_aft['b'][c].append(op_a.t)
            per_class_aft['d'][c].append(op_a.d)
            per_class_aft['u'][c].append(op_a.u)
        epochs.append(epoch)

    fig, axs = plt.subplots(2, 3, figsize=(13, 5))

    col_map = [('b', 'Trust'), ('d', 'DisTrust'), ('u', 'Uncertainty')]

    for row, (per_class, row_lbl) in enumerate([
            (per_class_bef, 'Before Calibration'),
            (per_class_aft, 'After Calibration'),
    ]):
        for col, (mk, col_lbl) in enumerate(col_map):
            ax = axs[row][col]
            for c in range(n_classes):
                ax.plot(epochs, per_class[mk][c],
                        color=COLORS_10[c], linewidth=1.0)
            ax.set_ylabel(col_lbl)
            if row == 1:
                ax.set_xlabel('Epochs')
            if col == 2:
                ax.ticklabel_format(axis='y', style='sci',
                                    scilimits=(-2, -2))

    # Synchronise y-limits between before/after rows per column
    for col in range(3):
        ylims = [axs[r][col].get_ylim() for r in range(2)]
        ymin = min(y[0] for y in ylims)
        ymax = max(y[1] for y in ylims)
        for r in range(2):
            axs[r][col].set_ylim(ymin, ymax)
            axs[r][col].set_yticks(np.around(np.linspace(ymin, ymax, 5),
                                              3 if col == 2 else 1))

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=1, vmax=10))
    sm.set_array([])
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.005, 0.7])
    fig.colorbar(sm, cax=cbar_ax)

    fig.text(0.5, 0.94, 'Before Calibration', ha='center', fontsize=14)
    fig.text(0.5, 0.47, 'After Calibration',  ha='center', fontsize=14)
    line1 = plt.Line2D([0.05, 0.93], [0.48, 0.48], color='black',
                        linewidth=1.2, transform=fig.transFigure)
    line2 = plt.Line2D([0.05, 0.93], [0.96, 0.96], color='black',
                        linewidth=1.2, transform=fig.transFigure)
    fig.add_artist(line1)
    fig.add_artist(line2)

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'{dataset_name}_ALL.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {dataset_name}_ALL.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 – Training / validation loss (overfitting)
# Reproduces MNIST_OV.pdf / CIFAR_OV.pdf referenced in the LaTeX chapter.
# ═════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(directory: str, outdir: str,
                     dataset_name: str) -> None:
    """
    Plot training vs. validation loss curves from the saved loss_history.npy.
    """
    history = load_loss_history(directory)
    if history is None:
        print(f"[plot_loss_curves] loss_history.npy not found in {directory}")
        return

    train_loss = history['train']
    val_loss   = history['val']
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, label='Training loss',   color='steelblue',
            linewidth=1.8)
    ax.plot(epochs, val_loss,   label='Validation loss', color='firebrick',
            linewidth=1.8, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title(f'{dataset_name} — Training and Validation Loss')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'{dataset_name}_OV.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {dataset_name}_OV.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# ECE summary table
# ═════════════════════════════════════════════════════════════════════════════

def print_ece_table(directories: dict) -> None:
    """
    Print a LaTeX-ready ECE table for the last epoch of each dataset.

    Parameters
    ----------
    directories : dict
        {dataset_name: directory_path}
    """
    print("\n% ECE summary table")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{ECE (pre-calibration)} "
          r"& \textbf{ECE (post-calibration)} \\")
    print(r"\midrule")
    for ds_name, directory in directories.items():
        epoch_data = list_epoch_files(directory)
        if not epoch_data:
            continue
        _, bef_path, aft_path = epoch_data[-1]
        d_bef = pd.read_csv(bef_path)
        d_aft = pd.read_csv(aft_path)
        ece_bef = compute_ece(d_bef)
        ece_aft = compute_ece(d_aft)
        print(f"{ds_name:12s} & {ece_bef:.3f} & {ece_aft:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate all calibration trust analysis figures.')
    p.add_argument('--mnist_dir',  default='data/MNIST_PRED',
                   help='Directory with MNIST predictions (default: data/MNIST_PRED)')
    p.add_argument('--cifar_dir',  default='data/CIFAR10_PRED',
                   help='Directory with CIFAR-10 predictions (default: data/CIFAR10_PRED)')
    p.add_argument('--outdir',     default='img/cal',
                   help='Output directory for PDF figures (default: img/cal)')
    p.add_argument('--n_clusters', type=int, default=10,
                   help='Number of probability clusters M (default: 10)')
    p.add_argument('--m_values', nargs='+', type=int,
                   default=[2, 3, 5, 8, 10, 15, 20, 30, 50],
                   help='M values for cluster-variation plot')
    p.add_argument('--skip_mnist',  action='store_true')
    p.add_argument('--skip_cifar',  action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    datasets = {}
    if not args.skip_mnist and os.path.isdir(args.mnist_dir):
        datasets['MNIST'] = args.mnist_dir
    if not args.skip_cifar and os.path.isdir(args.cifar_dir):
        datasets['CIFAR'] = args.cifar_dir

    if not datasets:
        print("No prediction directories found.  Run train_models.py first.")
        raise SystemExit(1)

    for ds_name, directory in datasets.items():
        print(f"\n{'═'*60}")
        print(f"  Dataset: {ds_name}  |  directory: {directory}")
        print('═' * 60)

        # 1. Original trust-metrics figure (MNIST_ALL / CIFAR_ALL)
        plot_trust_all(directory, args.outdir, ds_name,
                       n_clusters=args.n_clusters)

        # 2. Training loss curves (MNIST_OV / CIFAR_OV)
        plot_loss_curves(directory, args.outdir, ds_name)

        # 3. NEW: Trust metrics + ECE comparison
        plot_trust_and_ece(directory, args.outdir, ds_name,
                            n_clusters=args.n_clusters)

        # 4. NEW: Cluster variation
        plot_cluster_variation(directory, args.outdir, ds_name,
                                m_values=args.m_values)

        # 5. NEW: Dynamic assessment
        plot_dynamic_assessment(directory, args.outdir, ds_name,
                                 n_clusters=args.n_clusters)

    # 6. Print ECE table for LaTeX
    print_ece_table(datasets)
