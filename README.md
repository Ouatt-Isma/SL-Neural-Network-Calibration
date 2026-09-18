# SL-Neural-Network-Calibration

**Calibration error, expressed as a Subjective Logic opinion.**

Expected Calibration Error compresses a model's reliability into one number. This code turns the same per-bin evidence into a trust opinion `(belief, disbelief, uncertainty)`: bins where confidence matches accuracy add belief, bins that are over- or under-confident add disbelief, and sparsely populated bins leave uncertainty. The opinion can then be fused with other evidence about a model — which is how calibration enters the model-side trust score in [PaTAS](https://github.com/Ouatt-Isma/PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment).

Code accompanying the paper:

> **Quantifying Calibration Error in Neural Networks through Evidence-Based Theory**
> K. I. Ouattara, I. Krontiris, T. Dimitrakos, F. Kargl — *FUSION 2025, 28th International Conference on Information Fusion.*

The pipeline trains neural networks on MNIST and CIFAR-10, applies temperature
scaling as a post-hoc calibration method, and evaluates the resulting
predictions through both the standard Expected Calibration Error (ECE) and an
SL-based trust opinion (belief, disbelief, uncertainty).

> **Revision notice (September 2026).** The method and data in this repository
> have been corrected with respect to the FUSION 2025 paper; see
> [Method](#method) and [Data](#data) below. In short: the cluster
> representative is the mean confidence of the cluster (not the interval
> midpoint), the positive evidence is `n_i - s_i` (not the number of correct
> predictions), fusion is cumulative at both levels, the temperature is fitted
> on a held-out half of the test set, and the CIFAR-10 predictions originally
> exported were passed through a second softmax (max confidence e/(e+9) = 0.232)
> and have been recovered by inverting it. The CIFAR-10 results of the paper are
> superseded by the ones this code produces.

---

## Method

For every prediction, the *predicted class* is the arg-max class and the
*confidence* is the maximum probability. For each predicted class `c`, the
confidences are partitioned into `M` equal-width clusters (the last one
closed, `[(M-1)/M, 1]`). In cluster `i`:

| Symbol | Meaning |
|---|---|
| `n_i` | number of class-`c` predictions whose confidence falls in the cluster |
| `t_i` | number of those that are correct |
| `RP_i` | **mean confidence** of the `n_i` predictions (the representative) |
| `s_i = \|t_i - n_i * RP_i\|` | negative evidence: deviation between stated confidence and observed accuracy |
| `r_i = n_i - s_i` | positive evidence: predictions consistent with the stated confidence |

`(r_i, s_i)` is turned into an opinion with the baseline-prior quantification
(`b = r/(n+W)`, `d = s/(n+W)`, `u = W/(n+W)`, `W = 2`), so that `b ≈ 1 - gap`,
`d ≈ gap` and `u` depends only on the cluster population. Cluster opinions are
fused **cumulatively** into a class opinion and class opinions cumulatively
into the global opinion (clusters and classes both partition the predictions,
so they are disjoint evidence). Because `r_i + s_i = n_i`, the global
uncertainty is `W/(N+W)` for every `M`, and the global disbelief is a
class-conditional ECE.

The *dynamic* mode keeps the `(class, cluster)` opinions as a lookup table and
returns, for each prediction at inference time, the opinion of its cell:
`b` high — the stated confidence was verified on the evaluation data; `d` high
— it is known to be biased; `u` high — too few evaluation predictions to tell.

---

## Data

`data/MNIST_PRED` and `data/CIFAR_PRED` are the original epoch-wise exports of
the full 10,000-image test sets (`bef_E.csv` before, `aft_E.csv` after
temperature scaling). **Do not use `data/CIFAR_PRED` directly**: those files
were exported from a model whose output layer already applied a softmax, and
the export applied `tf.nn.softmax` a second time, so every stored value is
`softmax(q)` of the true softmax output `q` (max = e/(e+9) = 0.2320,
min = 1/(e+9) = 0.0853). The `aft` files divided *probabilities* by a
temperature (T = 0.217). MNIST was exported from logits and is correct.

`prepare_eval_data.py` builds the evaluation data actually used by
`analysis.py`:

1. recovers the true CIFAR-10 probabilities by inverting the extra softmax
   (`q = log p + (1 - sum log p)/K`, exact up to float32 precision);
2. splits the 10,000 test images once (seed 0) into a 5,000-image
   *calibration* half, used only to fit the temperature by NLL, and a
   5,000-image *evaluation* half on which every number is reported;
3. writes `data/<DS>_EVAL/{bef,aft}_E.csv`, `temperatures.json` and
   `split.json`.

The `_EVAL` directories are committed, so `python analysis.py` reproduces the
figures directly. About 3 % of the CIFAR-10 samples have a true-class
probability below the 1e-7 recovery floor, which slightly biases the fitted
temperature downward; retraining with the current `train_models.py` (logits
export, temperature fitted on the validation split) removes this residual.

---

## Repository structure

```
SL-Neural-Network-Calibration/
├── trustopinion.py          # Subjective Logic opinion class (core library)
├── train_models.py          # Train models and save epoch-wise predictions
├── prepare_eval_data.py     # Recover CIFAR-10 probabilities, split test set, fit T
├── analysis.py              # Compute trust metrics and generate all figures
├── requirements.txt         # Python dependencies
├── data/
│   ├── MNIST_PRED/          # original full-test-set exports (bef_E.csv, aft_E.csv, loss_history.npy)
│   ├── CIFAR_PRED/          # original exports — double softmax, see Data
│   ├── MNIST_EVAL/          # evaluation half + held-out temperature (used by analysis.py)
│   └── CIFAR_EVAL/          # recovered probabilities, evaluation half (used by analysis.py)
└── img/
    └── cal/                 # Output PDF figures (generated by analysis.py)
```

Pre-computed prediction files are included in the repository so figures can
be reproduced immediately without retraining.

Each `bef_E.csv` / `aft_E.csv` file holds the per-sample class probabilities
(columns `Class_0_Probability` … `Class_9_Probability`) and the true label
(`True Label`) at epoch `E`, before and after temperature scaling.

---

## Installation

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

---

## Reproducing the figures

With the pre-computed predictions already in `data/`, run:

```bash
python analysis.py
```

This reads from `data/MNIST_EVAL` and `data/CIFAR_EVAL` and saves all PDF
figures to `img/cal/`. To rebuild the `_EVAL` directories from the original
exports:

```bash
python prepare_eval_data.py
```

---

## Full pipeline (train from scratch)

### 1. Train and save predictions

```bash
# MNIST
python train_models.py --dataset mnist --outdir data/MNIST_PRED

# CIFAR-10  (~94–95 % test accuracy, requires a GPU)
python train_models.py --dataset cifar10 --outdir data/CIFAR_PRED
```

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `mnist` | `mnist` or `cifar10` |
| `--outdir` | `data/<DATASET>_PRED` | Output directory for CSV files |
| `--epochs` | `100` | Total training epochs |
| `--val_split` | `0.1` | Fraction of training data used for calibration |
| `--batch_size` | `32` | Mini-batch size |

Predictions are saved at milestones: epochs 1–9 (every epoch) and
10, 20, …, 100. The output layer produces logits; the script checks the raw
outputs at every milestone and aborts if they look like probabilities, so the
double-softmax export cannot recur. The temperature is fitted on the
validation split and recorded in `temperatures.json`. After retraining, run
`python prepare_eval_data.py --cifar_is_clean` to rebuild the `_EVAL`
directories (no inversion), or point `analysis.py` at the new directories
directly.

### 2. Generate figures and ECE table

```bash
python analysis.py \
    --mnist_dir data/MNIST_EVAL \
    --cifar_dir data/CIFAR_EVAL \
    --outdir    img/cal
```

| Flag | Default | Description |
|---|---|---|
| `--mnist_dir` | `data/MNIST_EVAL` | MNIST prediction directory |
| `--cifar_dir` | `data/CIFAR_EVAL` | CIFAR-10 prediction directory |
| `--outdir` | `img/cal` | Output directory for PDF figures |
| `--n_clusters` | `10` | Number of confidence clusters M |
| `--m_values` | `2 3 5 8 10 15 20 30 50 …` | M values for cluster-variation plot |
| `--skip_mnist` | — | Skip MNIST analysis |
| `--skip_cifar` | — | Skip CIFAR-10 analysis |

A LaTeX-formatted ECE comparison table is also printed to stdout.

---

## Output figures

| File | Description |
|---|---|
| `<DS>_ALL.pdf` | Belief / disbelief / uncertainty over epochs, before and after calibration |
| `<DS>_OV.pdf` | Training vs. validation loss (overfitting curve) |
| `<DS>_ECE.pdf` | ECE alongside SL trust metrics over epochs |
| `<DS>_clusters.pdf` | Global SL opinion vs. number of clusters M |
| `<DS>_dynamic.pdf` | Per-prediction trust scores at inference time |

`<DS>` is `MNIST` or `CIFAR`.

---

## Core module — `TrustOpinion`

`trustopinion.py` implements the Subjective Logic opinion quadruple
*(belief, disbelief, uncertainty, base-rate)* and all operators used in the
analysis.

| Method | Description |
|---|---|
| `TrustOpinion(b, d, u, a)` | Construct an opinion with belief `b`, disbelief `d`, uncertainty `u`, base rate `a` |
| `ev2tdu(pos, neg, W=2)` | Build an opinion from positive and negative evidence counts |
| `vacuous()` | Vacuous opinion (0, 0, 1, a) — no information |
| `projected_prob()` | Projected probability P = b + a·u |
| `cumFuse(op1, op2)` | Cumulative belief fusion |
| `avFuse(op1, op2)` | Averaging belief fusion |
| `weigFuse(op1, op2)` | Weighted belief fusion |
| `binMult(op1, op2)` | Binomial multiplication |
| `deduction(op_x, op_yx, op_ynotx)` | SL deduction operator |

---

## Models

| Dataset | Architecture |
|---|---|
| MNIST | Flatten → Dense(128, ReLU) → Dense(10, logits) |
| CIFAR-10 | WideResNet-28-4 (pre-activation residual blocks, width factor 4, depth 28) |

Both models output raw logits.  
**MNIST** is trained with the Adam optimiser and `SparseCategoricalCrossentropy(from_logits=True)`.  
**CIFAR-10** uses SGD with Nesterov momentum, cosine learning-rate decay, L2
weight decay, dropout (0.3), and embedded data augmentation (random horizontal
flip + translation).  
Temperature scaling is fitted by minimising negative log-likelihood on the
validation set using `scipy.optimize.minimize`.

---

## Citation

```bibtex
@inproceedings{ouattara2025calibration,
  title     = {Quantifying Calibration Error in Neural Networks through Evidence-Based Theory},
  author    = {Ouattara, Koffi Ismael and Krontiris, Ioannis and Dimitrakos, Theo and Kargl, Frank},
  booktitle = {Proceedings of the 28th International Conference on Information Fusion (FUSION)},
  year      = {2025}
}
```

## Related repositories

- [`PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment`](https://github.com/Ouatt-Isma/PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment) — uses this library (as a submodule) for calibration-based model trust.
- [`Trustworthiness-of-AI-Training-Dataset`](https://github.com/Ouatt-Isma/Trustworthiness-of-AI-Training-Dataset) — the data-side counterpart: trust opinions on training datasets.

## License

MIT — see [LICENSE](LICENSE).
