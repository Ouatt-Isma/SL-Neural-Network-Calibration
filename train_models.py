"""
Efficient incremental training script for MNIST and CIFAR-10.

Instead of retraining from scratch at every checkpoint (as in the original scripts),
this script trains each model incrementally, saving predictions at each milestone.
Predictions are saved in CSV format compatible with the analysis pipeline.

Usage:
    python train_models.py --dataset mnist  --outdir data/MNIST_PRED
    python train_models.py --dataset cifar10 --outdir data/CIFAR10_PRED

Each output directory will contain files like:
    bef_1.csv, aft_1.csv, bef_10.csv, aft_10.csv, ...
as well as a loss history file loss_history.npy.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def softmax_with_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling followed by softmax."""
    exp_logits = np.exp((logits - logits.max(axis=1, keepdims=True)) / T)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def nll_with_temperature(T, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood used to find the optimal temperature."""
    T = float(T)
    probs = softmax_with_temperature(logits, T)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12))
    return nll


def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find the temperature T that minimises NLL on the validation set."""
    result = minimize(nll_with_temperature, x0=1.0,
                      args=(logits, labels), bounds=[(0.05, 20.0)],
                      method='L-BFGS-B')
    return float(result.x[0])


def save_predictions(probs: np.ndarray, labels: np.ndarray,
                     path: str) -> None:
    """Save class probabilities and true labels to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_classes = probs.shape[1]
    df = pd.DataFrame(probs, columns=[f'Class_{i}_Probability'
                                       for i in range(n_classes)])
    df['True Label'] = labels
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_mnist_model() -> tf.keras.Model:
    """Fully-connected network for MNIST (as in the chapter)."""
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10),        # logits — no softmax here
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def build_cifar10_model() -> tf.keras.Model:
    """Small CNN for CIFAR-10 (as in the chapter: 32→64→64 filters)."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),        # logits
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save(dataset: str, outdir: str,
                   val_split: float = 0.1,
                   batch_size: int = 32,
                   total_epochs: int = 100) -> None:
    """
    Train a model incrementally and save predictions at milestone epochs.

    Milestones: epochs 1–9 (every epoch) + 10, 20, …, 100.
    Calibration is done on the *training* validation split; predictions are
    saved for the *test* set.
    """
    # ── Load data ─────────────────────────────────────────────────────────
    if dataset == 'mnist':
        (x_train_full, y_train_full), (x_test, y_test) = \
            tf.keras.datasets.mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        build_fn = build_mnist_model
    elif dataset == 'cifar10':
        (x_train_full, y_train_full), (x_test, y_test) = \
            tf.keras.datasets.cifar10.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()
        build_fn = build_cifar10_model
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ── Train / val split (fixed) ─────────────────────────────────────────
    n_val = int(len(x_train_full) * val_split)
    x_val, y_val = x_train_full[:n_val], y_train_full[:n_val]
    x_train, y_train = x_train_full[n_val:], y_train_full[n_val:]

    os.makedirs(outdir, exist_ok=True)

    # ── Milestone epochs ───────────────────────────────────────────────────
    milestones = list(range(1, 10)) + list(range(10, total_epochs + 1, 10))
    milestone_set = set(milestones)

    model = build_fn()
    loss_history = {'train': [], 'val': []}

    cumulative_epoch = 0

    for milestone in milestones:
        epochs_to_run = milestone - cumulative_epoch
        if epochs_to_run <= 0:
            continue

        history = model.fit(
            x_train, y_train,
            epochs=epochs_to_run,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=1,
        )

        loss_history['train'].extend(history.history['loss'])
        loss_history['val'].extend(history.history['val_loss'])
        cumulative_epoch = milestone

        # ── Raw logits on test set ─────────────────────────────────────
        logits_test = model.predict(x_test, verbose=0)

        # ── Calibration temperature (fitted on validation set) ─────────
        logits_val = model.predict(x_val, verbose=0)
        T_opt = find_optimal_temperature(logits_val, y_val)
        print(f"[epoch {milestone:3d}] optimal T = {T_opt:.4f}")

        # ── Probabilities before / after calibration ───────────────────
        probs_bef = tf.nn.softmax(logits_test, axis=1).numpy()
        probs_aft = softmax_with_temperature(logits_test, T_opt)

        # ── Save CSVs ──────────────────────────────────────────────────
        save_predictions(probs_bef, y_test,
                         os.path.join(outdir, f'bef_{milestone}.csv'))
        save_predictions(probs_aft, y_test,
                         os.path.join(outdir, f'aft_{milestone}.csv'))

        print(f"  → saved predictions for epoch {milestone}")

    # ── Save loss history ──────────────────────────────────────────────────
    np.save(os.path.join(outdir, 'loss_history.npy'), loss_history)
    print(f"\nTraining complete. Loss history saved to {outdir}/loss_history.npy")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train MNIST/CIFAR-10 and save epoch-wise predictions.')
    p.add_argument('--dataset', choices=['mnist', 'cifar10'],
                   default='mnist')
    p.add_argument('--outdir', default=None,
                   help='Output directory (default: data/<DATASET>_PRED)')
    p.add_argument('--epochs', type=int, default=100,
                   help='Total training epochs (default: 100)')
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    outdir = args.outdir or f'data/{args.dataset.upper()}_PRED'
    train_and_save(
        dataset=args.dataset,
        outdir=outdir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        total_epochs=args.epochs,
    )
