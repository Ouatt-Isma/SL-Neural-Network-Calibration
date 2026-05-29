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
    """Apply temperature scaling followed by softmax (matches calibratedMNIST.py)."""
    exp_logits = np.exp(logits / T)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def nll_with_temperature(T, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood used to find the optimal temperature."""
    probs = softmax_with_temperature(logits, T)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    return nll


def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find the temperature T that minimises NLL on the given set."""
    result = minimize(nll_with_temperature, x0=1.0,
                      args=(logits, labels), bounds=[(0.1, 10)])
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


def build_cifar10_model(total_epochs: int = 100) -> tf.keras.Model:
    """WideResNet-28-4 for CIFAR-10, targeting ~94-95% test accuracy.

    Architecture: pre-activation residual blocks, width factor 4,
    depth 28 (4 blocks × 3 stages). Data augmentation (random horizontal
    flip + translation) is embedded as model layers and is automatically
    disabled at inference time. Trained with SGD + Nesterov momentum and
    cosine learning-rate decay.
    """
    depth, width = 28, 4
    n = (depth - 4) // 6   # 4 residual blocks per stage
    wd = 5e-4
    reg = tf.keras.regularizers.l2(wd)

    def res_block(x, filters, stride=1):
        shortcut = x
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if stride != 1 or int(shortcut.shape[-1]) != filters:
            shortcut = layers.Conv2D(
                filters, 1, strides=stride, padding='same',
                use_bias=False, kernel_regularizer=reg)(x)
        x = layers.Conv2D(
            filters, 3, strides=stride, padding='same',
            use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(
            filters, 3, padding='same',
            use_bias=False, kernel_regularizer=reg)(x)
        return layers.Add()([x, shortcut])

    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Augmentation: active only during training
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomTranslation(0.125, 0.125, fill_mode='reflect')(x)

    # Stem
    x = layers.Conv2D(16, 3, padding='same', use_bias=False,
                      kernel_regularizer=reg)(x)

    # Stage 1 — 16×width = 64 filters
    for _ in range(n):
        x = res_block(x, 16 * width)

    # Stage 2 — 32×width = 128 filters, stride 2
    x = res_block(x, 32 * width, stride=2)
    for _ in range(n - 1):
        x = res_block(x, 32 * width)

    # Stage 3 — 64×width = 256 filters, stride 2
    x = res_block(x, 64 * width, stride=2)
    for _ in range(n - 1):
        x = res_block(x, 64 * width)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, kernel_regularizer=reg)(x)

    model = tf.keras.Model(inputs, outputs)

    # Cosine LR decay: 0.1 → ~0 over `total_epochs` (assumes batch=32, n_train≈45k)
    steps_per_epoch = 45000 // 32
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1,
        decay_steps=total_epochs * steps_per_epoch,
        alpha=1e-6,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9, nesterov=True),
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
    Train one model incrementally and save predictions at each milestone.

    The model is built once and trained continuously; at each milestone epoch
    predictions are saved before and after temperature scaling.
    Milestones: epochs 1–9 (every epoch) + 10, 20, …, 100.
    """
    # ── Load data ─────────────────────────────────────────────────────────
    if dataset == 'mnist':
        (x_train_full, y_train_full), (x_test, y_test) = \
            tf.keras.datasets.mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        model = build_mnist_model()
    elif dataset == 'cifar10':
        (x_train_full, y_train_full), (x_test, y_test) = \
            tf.keras.datasets.cifar10.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()
        model = build_cifar10_model(total_epochs=total_epochs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ── Train / val split (fixed) ─────────────────────────────────────────
    n_val = int(len(x_train_full) * val_split)
    x_val, y_val = x_train_full[:n_val], y_train_full[:n_val]
    x_train, y_train = x_train_full[n_val:], y_train_full[n_val:]

    os.makedirs(outdir, exist_ok=True)

    milestones = list(range(1, 10)) + list(range(10, total_epochs + 1, 10))
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

        # ── Calibration temperature ────────────────────────────────────
        T_opt = find_optimal_temperature(logits_test, y_test)
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
