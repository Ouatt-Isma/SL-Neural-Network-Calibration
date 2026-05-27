"""
Standalone demonstration of the three new analyses using synthetically
generated prediction data.

Run this to verify the analysis pipeline without having to train the full
MNIST / CIFAR-10 models:

    python demo_analysis.py

All figures are saved to img/cal_demo/.
"""

import os
import numpy as np
import pandas as pd

from analysis import (
    plot_trust_all,
    plot_trust_and_ece,
    plot_cluster_variation,
    plot_dynamic_assessment,
    plot_loss_curves,
    print_ece_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

def _dirichlet_probs(n: int, n_classes: int, concentration: float) -> np.ndarray:
    """Sample probability vectors from a symmetric Dirichlet distribution."""
    alpha = np.full(n_classes, concentration)
    return np.random.default_rng(0).dirichlet(alpha, size=n)


def _miscalibrated_probs(n: int, n_classes: int,
                          overconfidence: float = 2.0) -> np.ndarray:
    """
    Generate overconfident probabilities by sharpening Dirichlet samples with
    a temperature < 1 (mimics an uncalibrated NN).
    """
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(n_classes) * 0.5, size=n)
    # Sharpen: raise each prob to 1/T, renormalise
    T = 1.0 / overconfidence
    sharpened = probs ** (1.0 / T)
    sharpened /= sharpened.sum(axis=1, keepdims=True)
    return sharpened


def _calibrated_probs(probs_bef: np.ndarray,
                       temperature: float = 2.5) -> np.ndarray:
    """Soften probabilities to simulate temperature scaling."""
    log_p = np.log(probs_bef + 1e-12)
    scaled = log_p / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def generate_synthetic_dataset(n: int = 3000,
                                n_classes: int = 10,
                                accuracy: float = 0.82,
                                overconfidence: float = 3.0,
                                calibration_temperature: float = 2.8,
                                seed: int = 42) -> tuple:
    """
    Return (probs_bef, probs_aft, true_labels) numpy arrays.

    The 'before' probabilities are intentionally overconfident;
    'after' probabilities are softened by temperature scaling.
    True labels are generated such that the model achieves roughly *accuracy*.
    """
    rng = np.random.default_rng(seed)
    true_labels = rng.integers(0, n_classes, size=n)

    # Build before-calibration probabilities: overconfident
    probs_bef = _miscalibrated_probs(n, n_classes, overconfidence)

    # Force the model to be 'accurate' by biasing probabilities toward the
    # true label for a fraction of samples
    correct_mask = rng.random(n) < accuracy
    for i in np.where(correct_mask)[0]:
        # Assign most mass to the true class
        probs_bef[i] = rng.dirichlet(
            np.where(np.arange(n_classes) == true_labels[i], 10.0, 0.3))

    # After calibration: soften the distribution
    probs_aft = _calibrated_probs(probs_bef, temperature=calibration_temperature)
    return probs_bef, probs_aft, true_labels


def make_predictions_df(probs: np.ndarray,
                         true_labels: np.ndarray) -> pd.DataFrame:
    n_classes = probs.shape[1]
    df = pd.DataFrame(probs, columns=[f'Class_{i}_Probability'
                                       for i in range(n_classes)])
    df['True Label'] = true_labels
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Write CSV files that the analysis functions expect
# ─────────────────────────────────────────────────────────────────────────────

def create_demo_directory(outdir: str,
                           n_epochs: int = 30,
                           n_samples: int = 3000,
                           n_classes: int = 10) -> None:
    """
    Write synthetic bef_E.csv / aft_E.csv for epochs 1…n_epochs,
    and a loss_history.npy.

    Later epochs are more accurate (and the uncalibrated version becomes more
    overconfident), mimicking actual training dynamics.
    """
    os.makedirs(outdir, exist_ok=True)
    milestones = list(range(1, 10)) + list(range(10, n_epochs + 1, 10))

    train_losses, val_losses = [], []

    for epoch in milestones:
        # As training progresses: accuracy increases, overconfidence increases
        progress = epoch / n_epochs
        accuracy = 0.40 + 0.50 * progress                 # 40% → 90%
        overconfidence = 1.0 + 4.0 * progress              # slight → very OC
        # Loss curves: training drops, validation diverges (overfitting)
        for _ in range(1):  # one synthetic step per milestone epoch
            train_losses.append(1.5 * np.exp(-2.0 * progress) + 0.05)
            val_losses.append(
                0.6 + 0.3 * progress + 0.1 * np.random.randn())

        probs_bef, probs_aft, labels = generate_synthetic_dataset(
            n=n_samples, n_classes=n_classes,
            accuracy=accuracy,
            overconfidence=overconfidence,
            calibration_temperature=max(1.2, 0.8 + 2.5 * progress),
            seed=epoch,
        )
        df_bef = make_predictions_df(probs_bef, labels)
        df_aft = make_predictions_df(probs_aft, labels)
        df_bef.to_csv(os.path.join(outdir, f'bef_{epoch}.csv'), index=False)
        df_aft.to_csv(os.path.join(outdir, f'aft_{epoch}.csv'), index=False)

    # Pad loss history to length n_epochs
    full_train = np.interp(np.arange(1, n_epochs + 1),
                           np.array(milestones), train_losses)
    full_val   = np.interp(np.arange(1, n_epochs + 1),
                           np.array(milestones), val_losses)
    np.save(os.path.join(outdir, 'loss_history.npy'),
            {'train': full_train.tolist(), 'val': full_val.tolist()})
    print(f"Created synthetic dataset in {outdir} "
          f"({len(milestones)} epoch checkpoints)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    demo_dir = 'demo_data/DEMO_PRED'
    out_dir  = 'img/cal_demo'
    ds_name  = 'DEMO'
    m_values = [2, 3, 5, 8, 10, 15, 20, 30, 50]

    print("═" * 60)
    print("  Generating synthetic prediction data …")
    print("═" * 60)
    create_demo_directory(demo_dir, n_epochs=30)

    print("\n═" * 60)
    print("  Running analyses …")
    print("═" * 60)

    # 1. Original trust-metrics figure
    plot_trust_all(demo_dir, out_dir, ds_name, n_clusters=10)

    # 2. Training loss curves
    plot_loss_curves(demo_dir, out_dir, ds_name)

    # 3. NEW: Trust metrics + ECE comparison
    plot_trust_and_ece(demo_dir, out_dir, ds_name, n_clusters=10)

    # 4. NEW: Cluster variation
    plot_cluster_variation(demo_dir, out_dir, ds_name, m_values=m_values)

    # 5. NEW: Dynamic assessment
    plot_dynamic_assessment(demo_dir, out_dir, ds_name, n_clusters=10)

    # 6. ECE table
    print_ece_table({ds_name: demo_dir})

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
