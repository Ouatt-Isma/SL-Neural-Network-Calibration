# SL-Neural-Network-Calibration

Code accompanying the dissertation chapter on **Subjective Logic (SL) as a
trust framework for neural network calibration**.

The pipeline trains neural networks on MNIST and CIFAR-10, applies temperature
scaling as a post-hoc calibration method, and evaluates the resulting
predictions through both the standard Expected Calibration Error (ECE) and an
SL-based trust opinion (belief, disbelief, uncertainty).

---

## Repository structure

```
SL-Neural-Network-Calibration/
├── trustopinion.py      # Subjective Logic opinion class (core library)
├── train_models.py      # Train models and save epoch-wise predictions
├── analysis.py          # Compute trust metrics and generate all figures
├── demo_analysis.py     # Reproduce all figures on synthetic data (no GPU needed)
├── data/
│   ├── MNIST_PRED/      # bef_E.csv, aft_E.csv, loss_history.npy  (generated)
│   └── CIFAR_PRED/      # bef_E.csv, aft_E.csv, loss_history.npy  (generated)
└── img/                 # Output figures (generated)
```

Each `bef_E.csv` / `aft_E.csv` file holds the per-sample class probabilities
(columns `Class_0_Probability` … `Class_9_Probability`) and the true label
(`True Label`) at epoch `E`, before and after temperature scaling.

---

## Installation

Python 3.9+ is recommended.

```bash
pip install numpy pandas matplotlib scipy tensorflow scikit-learn
```

---

## Quick start — synthetic demo (no training required)

```bash
python demo_analysis.py
```

This generates synthetic prediction data and saves all figures to `img/cal_demo/`.

---

## Full pipeline

### 1. Train and save predictions

```bash
# MNIST
python train_models.py --dataset mnist --outdir data/MNIST_PRED

# CIFAR-10
python train_models.py --dataset cifar10 --outdir data/CIFAR_PRED
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `mnist` | `mnist` or `cifar10` |
| `--outdir` | `data/<DATASET>_PRED` | Output directory for CSV files |
| `--epochs` | `100` | Total training epochs |
| `--val_split` | `0.1` | Fraction of training data used for calibration |
| `--batch_size` | `32` | Mini-batch size |

Predictions are saved at milestones: epochs 1–9 (every epoch) and
10, 20, …, 100.

### 2. Generate figures and ECE table

```bash
python analysis.py \
    --mnist_dir data/MNIST_PRED \
    --cifar_dir data/CIFAR_PRED \
    --outdir    img/cal
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--mnist_dir` | `data/MNIST_PRED` | MNIST prediction directory |
| `--cifar_dir` | `data/CIFAR10_PRED` | CIFAR-10 prediction directory |
| `--outdir` | `img/cal` | Output directory for PDF figures |
| `--n_clusters` | `10` | Number of probability clusters M |
| `--m_values` | `2 3 5 8 10 15 20 30 50` | M values for cluster-variation plot |
| `--skip_mnist` | — | Skip MNIST |
| `--skip_cifar` | — | Skip CIFAR-10 |

---

## Output figures

| File | Description |
|---|---|
| `<DS>_ALL.pdf` | Belief / disbelief / uncertainty over epochs, before and after calibration |
| `<DS>_OV.pdf` | Training vs. validation loss (overfitting curve) |
| `<DS>_ECE.pdf` | ECE alongside SL metrics over epochs |
| `<DS>_clusters.pdf` | SL opinion vs. number of clusters M |
| `<DS>_dynamic.pdf` | Per-prediction trust scores at inference time |

A LaTeX-formatted ECE summary table is also printed to stdout.

---

## Core module — `TrustOpinion`

`trustopinion.py` implements the SL opinion class used throughout the
analysis.  Key methods:

| Method | Description |
|---|---|
| `TrustOpinion(b, d, u, a)` | Construct an opinion with belief `b`, disbelief `d`, uncertainty `u`, base rate `a` |
| `ev2tdu(pos, neg)` | Build an opinion from positive and negative evidence counts |
| `cumFuse(op1, op2)` | Cumulative belief fusion |
| `avFuse(op1, op2)` | Averaging belief fusion |
| `weigFuse(op1, op2)` | Weighted belief fusion |
| `deduction(op_x, op_yx, op_ynotx)` | SL deduction operator |
| `projected_prob()` | Projected probability P = b + a·u |
| `vacuous()` | Vacuous opinion (0, 0, 1) |

---

## Models

| Dataset | Architecture |
|---|---|
| MNIST | Flatten → Dense(128, ReLU) → Dense(10) |
| CIFAR-10 | Conv(32) → Pool → Conv(64) → Pool → Conv(64) → Flatten → Dense(64, ReLU) → Dense(10) |

Both models are trained with the Adam optimiser and
`SparseCategoricalCrossentropy(from_logits=True)`.
Temperature scaling is fitted by minimising negative log-likelihood on the
test set using `scipy.optimize.minimize`.

---

## Reference

If you use this code, please cite:

> Ouattara, I. (2025). *Trust Assessment of Neural Networks via Subjective
> Logic and Calibration*. PhD Dissertation, [University name].
