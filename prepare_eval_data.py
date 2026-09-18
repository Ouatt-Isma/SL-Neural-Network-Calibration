"""
Prepare evaluation data with a held-out temperature-scaling split.

Why this script exists
----------------------
1. The CIFAR-10 predictions originally exported to ``data/CIFAR_PRED`` were
   produced by a model whose output layer already applied a softmax, and the
   export applied ``tf.nn.softmax`` a second time.  Every stored "probability"
   is therefore ``softmax(q)`` where ``q`` is the true softmax output
   (max = e/(e+9) = 0.2320, min = 1/(e+9) = 0.0853).  The "after calibration"
   files then divided *probabilities* (not logits) by T.  Both are wrong.
   Because ``q`` sums to one, the extra softmax is exactly invertible:
   ``q = log(p) + (1 - sum(log p)) / K``.  We recover ``q`` here (float32
   precision, ~1e-7 absolute).  MNIST was exported from logits and is used as
   stored.

2. Temperature scaling was originally fitted on the full test set and
   evaluated on the same set.  Here the 10,000 test samples are split once
   (fixed seed) into a 5,000-sample calibration half, used only to fit T, and a
   5,000-sample evaluation half on which every reported number is computed.

Outputs (same layout as the original directories so ``analysis.py`` runs
unchanged):

    data/<DS>_EVAL/bef_<E>.csv     true probabilities, evaluation half
    data/<DS>_EVAL/aft_<E>.csv     softmax(logits / T_E), evaluation half
    data/<DS>_EVAL/temperatures.json
    data/<DS>_EVAL/split.json
    data/<DS>_EVAL/loss_history.npy (copied if present)

Usage:
    python prepare_eval_data.py
"""

import os
import re
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

SEED = 0
CIFAR_EPS = 1e-7        # recovery precision floor for the inverted softmax
MNIST_EPS = 1e-45       # float32 underflow floor (stored values are float32)


def prob_cols(df):
    return [c for c in df.columns if c.startswith('Class_') and c.endswith('_Probability')]


def invert_softmax(p):
    """Recover q from p = softmax(q) given that q sums to one."""
    lp = np.log(p)
    c = (1.0 - lp.sum(axis=1, keepdims=True)) / p.shape[1]
    q = lp + c
    return np.clip(q, 0.0, 1.0)


def nll(T, logits, labels):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(labels)), labels].mean()


def fit_temperature(logits, labels):
    res = minimize_scalar(nll, bounds=(0.05, 20.0), args=(logits, labels), method='bounded')
    return float(res.x)


def softmax_T(logits, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sanity_report(name, P):
    """Print what a raw-output check should show for genuine softmax outputs."""
    print(f'  [{name}] range=[{P.min():.4g}, {P.max():.4g}]  row-sum=[{P.sum(1).min():.6f}, {P.sum(1).max():.6f}]  '
          f'mean max={P.max(1).mean():.4f}')


def prepare(src_dir, dst_dir, double_softmax, eps):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted((int(m.group(1)), f) for f in os.listdir(src_dir)
                   for m in [re.match(r'^bef_(\d+)\.csv$', f)] if m)
    rng = np.random.default_rng(SEED)
    perm = None
    temps = {}
    for epoch, fname in files:
        df = pd.read_csv(os.path.join(src_dir, fname))
        cols = prob_cols(df)
        P = df[cols].values.astype(np.float64)
        y = df['True Label'].values.astype(int)
        if perm is None:
            perm = rng.permutation(len(df))
            cal_idx, ev_idx = np.sort(perm[:len(df) // 2]), np.sort(perm[len(df) // 2:])
            json.dump({'seed': SEED, 'calibration_idx': cal_idx.tolist(),
                       'evaluation_idx': ev_idx.tolist()},
                      open(os.path.join(dst_dir, 'split.json'), 'w'))
        if double_softmax:
            if epoch == files[-1][0]:
                print('  stored (double-softmax) values:'); sanity_report(f'bef_{epoch} stored', P)
            P = invert_softmax(P)
        if epoch == files[-1][0]:
            sanity_report(f'bef_{epoch} true probabilities', P)
        logits = np.log(np.clip(P, eps, 1.0))
        T = fit_temperature(logits[cal_idx], y[cal_idx])
        temps[epoch] = T
        n_floor = int((P[np.arange(len(y)), y] < eps).sum())
        P_aft = softmax_T(logits, T)
        for tag, arr in [('bef', P), ('aft', P_aft)]:
            out = pd.DataFrame(arr[ev_idx], columns=cols)
            out['True Label'] = y[ev_idx]
            out.to_csv(os.path.join(dst_dir, f'{tag}_{epoch}.csv'), index=False)
        acc = (P[ev_idx].argmax(1) == y[ev_idx]).mean()
        print(f'  epoch {epoch:3d}: T = {T:.4f}   eval acc = {acc:.4f}   '
              f'true-class prob below floor (NLL clipped): {n_floor}')
    json.dump(temps, open(os.path.join(dst_dir, 'temperatures.json'), 'w'), indent=1)
    lh = os.path.join(src_dir, 'loss_history.npy')
    if os.path.isfile(lh):
        shutil.copy(lh, dst_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--cifar_is_clean', action='store_true',
                    help='CIFAR-10 exports come from the corrected train_models.py '
                         '(single softmax): skip the inversion.')
    a = ap.parse_args()
    print('MNIST (stored from logits; used as is)')
    prepare(os.path.join(a.data, 'MNIST_PRED'), os.path.join(a.data, 'MNIST_EVAL'),
            double_softmax=False, eps=MNIST_EPS)
    print('CIFAR-10 (clean export)' if a.cifar_is_clean
          else 'CIFAR-10 (stored with a second softmax; inverted)')
    prepare(os.path.join(a.data, 'CIFAR_PRED'), os.path.join(a.data, 'CIFAR_EVAL'),
            double_softmax=not a.cifar_is_clean,
            eps=MNIST_EPS if a.cifar_is_clean else CIFAR_EPS)
