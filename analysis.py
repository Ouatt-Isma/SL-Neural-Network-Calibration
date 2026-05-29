"""
Calibration-based trust analysis using Subjective Logic (SL).

This module implements and visualises four evaluations:

  1. Trust metrics      – per-class belief, disbelief, and uncertainty curves
                          over training epochs, before and after calibration.

  2. ECE comparison     – ECE plotted alongside SL trust metrics over training
                          epochs, illustrating the richer information provided
                          by the SL framework.

  3. Cluster variation  – how the global trust opinion changes as the number of
                          probability clusters M varies at the final epoch.

  4. Dynamic assessment – per-prediction trust scores obtained by looking up
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
            ax.set_ylim(0, 1)
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
        m_values = [2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]

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
        ax.set_ylim(0, 1)

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
                              n_clusters: int = 10,
                              class_id: int = None) -> None:
    """
    Illustrate the dynamic (per-prediction) trust assessment mechanism.

    2×3 grid: Row 0 = before calibration, Row 1 = after calibration.
    Columns: scatter (confidence vs belief), stacked proportion bar,
             mean opinion per confidence bin.
    """
    epoch_data = list_epoch_files(directory)
    if not epoch_data:
        print(f"[plot_dynamic_assessment] No files found in {directory}")
        return

    last_epoch, bef_path, aft_path = epoch_data[-1]

    def _get_result(path):
        d = pd.read_csv(path)
        lookup = build_cluster_lookup(d, n_clusters=n_clusters)
        res = get_per_prediction_trust(d, lookup, n_clusters=n_clusters)
        if class_id is not None:
            n_cls = sum(1 for c in d.columns
                        if c.startswith('Class_') and c.endswith('_Probability'))
            pred_cls = d[[f'Class_{i}_Probability' for i in range(n_cls)]].values.argmax(axis=1)
            m = pred_cls == class_id
            res = {k: v[m] for k, v in res.items()}
        return res

    results = [_get_result(bef_path), _get_result(aft_path)]
    row_labels = ['Before Calibration', 'After Calibration']

    n_bins = n_clusters
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_bins   = np.linspace(0, 1, n_clusters + 1)
    hist_centres= 0.5 * (hist_bins[:-1] + hist_bins[1:])
    bar_width   = (hist_bins[1] - hist_bins[0]) * 0.85

    fig, axs = plt.subplots(2, 3, figsize=(15, 9),
                             gridspec_kw={'wspace': 0.35, 'hspace': 0.45})

    for row, (res, row_lbl) in enumerate(zip(results, row_labels)):
        belief      = res['belief']
        disbelief   = res['disbelief']
        uncertainty = res['uncertainty']
        confidence  = res['confidence']
        correct     = res['correct']

        def _stacked_bar(ax, values, correct_mask, xlabel, title, row):
            total_c, _   = np.histogram(values, bins=hist_bins)
            correct_c, _ = np.histogram(values[correct_mask], bins=hist_bins)
            incorrect_c  = total_c - correct_c
            safe  = np.maximum(total_c, 1)
            p_cor = np.where(total_c > 0, correct_c   / safe, 0)
            p_inc = np.where(total_c > 0, incorrect_c / safe, 0)
            ax.bar(hist_centres, p_cor, width=bar_width, color='#2196F3',
                   alpha=0.7, label='Correct')
            ax.bar(hist_centres, p_inc, width=bar_width, color='#F44336',
                   alpha=0.7, label='Incorrect', bottom=p_cor)
            for center, total in zip(hist_centres, total_c):
                if total > 0:
                    ax.text(center, 1.02, str(int(total)),
                            ha='center', va='bottom', fontsize=6, rotation=90)
            ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
            ax.set_ylabel('Proportion within bin', fontsize=LABEL_FONTSIZE)
            ax.set_ylim(0, 1.4)
            if row == 0:
                ax.set_title(title, fontsize=TITLE_FONTSIZE)
            ax.legend(fontsize=LABEL_FONTSIZE - 1)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        # --- Panel A: proportion by predicted confidence ---
        _stacked_bar(axs[row][0], confidence, correct,
                     'Predicted confidence', 'Distribution of Predicted Confidence', row)

        # --- Panel B: proportion by trust belief ---
        _stacked_bar(axs[row][1], belief, correct,
                     'Trust belief $b$', 'Distribution of Trust Belief', row)

        # --- Panel C: mean opinion per confidence bin ---
        ax = axs[row][2]
        mean_b, mean_d, mean_u = [], [], []
        for i in range(n_bins):
            m = (confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1])
            if m.sum() > 0:
                mean_b.append(belief[m].mean())
                mean_d.append(disbelief[m].mean())
                mean_u.append(uncertainty[m].mean())
            else:
                mean_b.append(np.nan); mean_d.append(np.nan); mean_u.append(np.nan)
        ax.plot(bin_centres, mean_b, 'o-',  color='steelblue', linewidth=1.8,
                markersize=5, label='Belief')
        ax.plot(bin_centres, mean_d, 's--', color='firebrick', linewidth=1.8,
                markersize=5, label='Disbelief')
        ax.plot(bin_centres, mean_u, '^:',  color='seagreen',  linewidth=1.8,
                markersize=5, label='Uncertainty')
        ax.set_xlabel('Predicted confidence', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Mean opinion component', fontsize=LABEL_FONTSIZE)
        if row == 0:
            ax.set_title('Mean Trust Opinion per Confidence Bin', fontsize=TITLE_FONTSIZE)
        ax.legend(fontsize=LABEL_FONTSIZE - 1)
        ax.set_xlim(-0.02, 1.02);  ax.set_ylim(0, 1)
        ax.grid(linestyle='--', alpha=0.3)

        # Row label on the left
        axs[row][0].annotate(row_lbl, xy=(-0.35, 0.5), xycoords='axes fraction',
                              rotation=90, va='center', ha='center',
                              fontsize=TITLE_FONTSIZE, fontweight='bold')

    class_label = f'class {class_id}' if class_id is not None else 'all classes'
    fig.suptitle(
        f'{dataset_name} — Dynamic trust assessment at inference '
        f'(epoch {last_epoch}, {class_label})',
        fontsize=TITLE_FONTSIZE,
    )
    os.makedirs(outdir, exist_ok=True)
    fname = (f'{dataset_name}_dynamic_class{class_id}.pdf'
             if class_id is not None else f'{dataset_name}_dynamic.pdf')
    fig.savefig(os.path.join(outdir, fname), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


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

    fig, axs = plt.subplots(2, 3, figsize=(13, 6))

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

    # Fix y-axis to [0, 1] for all three columns
    for col in range(3):
        for r in range(2):
            axs[r][col].set_ylim(0, 1)
            axs[r][col].set_yticks(np.linspace(0, 1, 6))

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=1, vmax=10))
    sm.set_array([])
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.005, 0.7])
    fig.colorbar(sm, cax=cbar_ax)

    fig.text(0.5, 0.9, 'Before Calibration', ha='center', fontsize=14)
    fig.text(0.5, 0.47, 'After Calibration',  ha='center', fontsize=14)
    # line1 = plt.Line2D([0.05, 0.93], [0.48, 0.48], color='black',
    #                     linewidth=1.2, transform=fig.transFigure)
    # line2 = plt.Line2D([0.05, 0.93], [0.96, 0.96], color='black',
    #                     linewidth=1.2, transform=fig.transFigure)
    # fig.add_artist(line1)
    # fig.add_artist(line2)

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
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Accuracy} "
          r"& \textbf{ECE (pre-calibration)} & \textbf{ECE (post-calibration)} \\")
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
        n_classes = sum(1 for c in d_bef.columns if c.startswith('Class_') and
                        c.endswith('_Probability'))
        prob_cols = [f'Class_{i}_Probability' for i in range(n_classes)]
        true_labels = d_bef['True Label'].values
        acc_bef = (d_bef[prob_cols].values.argmax(axis=1) == true_labels).mean()
        print(f"{ds_name:12s} & {acc_bef:.3f} & {ece_bef:.3f} & {ece_aft:.3f} \\\\")
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
                   default=[2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 1000],
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
                                #  n_clusters=args.n_clusters, 
                                #  class_id=0
                                 )

    # 6. Print ECE table for LaTeX
    print_ece_table(datasets)
