"""rf_protocol.py: random-forest importance protocol (Piotrowska+2022, Bluck+2020).

Shared helper for 04 (method validation) and 05 (the bar test). It follows the
class-balanced, random-control protocol of those papers, with permutation as
well as Gini importance; it is not an exact re-implementation of their
settings. The science-specific feature lists and verdicts live in the calling
scripts.

The recipe and WHY each step matters:

  * balanced_sample : subsample the majority class to 50/50 on the target. An
    imbalanced set lets a classifier score well by always predicting the majority
    class; balancing makes AUC and importances reflect real structure.

  * add_random_control : a fresh U(0,1) column drawn EVERY repeat. A physical
    feature is only meaningful if it beats this pure-noise control. (Gini
    importance is biased toward continuous, high-cardinality features, so the
    control is continuous on purpose, so it sets the noise floor for the kind of
    variable that Gini importance favours.)

  * tune_leaf : grow min_samples_leaf until the forest stops overfitting, i.e.
    AUC_train - AUC_test <= AUC_OVERFIT_TOL. Importances read off an overfit
    forest are not trustworthy.

  * run_rf_importance : many repeats, each a fresh balanced draw + fresh control
    + fresh 50/50 split. Record BOTH Gini (mean decrease in impurity, from the
    fitted trees) AND permutation importance (drop in TEST-set AUC when a column
    is shuffled; the less biased measure). Report medians with 5th/95th
    percentiles, plus the median test AUC (an importance ranking from a forest at
    AUC ~ 0.5 is noise and must be flagged).

Permutation importance measures predictive reliance on a fitted model.
Correlated predictors affect that reliance; the ranking does not give a
general upper or lower bound on a feature's physical causal importance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

RANDOM = "random_control"


def add_random_control(df, rng):
    """Return a copy of df with a fresh flat U(0,1) noise column."""
    out = df.copy()
    out[RANDOM] = rng.uniform(0.0, 1.0, size=len(out))
    return out


def balanced_sample(df, target, rng):
    """Subsample the majority class so the two `target` classes are 50/50."""
    y = df[target].to_numpy(bool)
    idx_pos = np.flatnonzero(y)
    idx_neg = np.flatnonzero(~y)
    n = min(idx_pos.size, idx_neg.size)
    keep = np.concatenate([rng.choice(idx_pos, n, replace=False),
                           rng.choice(idx_neg, n, replace=False)])
    rng.shuffle(keep)
    return df.iloc[keep].reset_index(drop=True)


def _fit_rf(x_tr, y_tr, leaf, seed):
    rf = RandomForestClassifier(
        n_estimators=config.N_TREES,
        max_features=config.MAX_FEATURES,     # all features considered at each split
        min_samples_leaf=leaf,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x_tr, y_tr)
    return rf


def _split(bs, features, target, seed):
    x = bs[features].to_numpy(float)
    y = bs[target].to_numpy(bool)
    return train_test_split(x, y, test_size=config.TEST_SIZE,
                            random_state=seed, stratify=y)


def tune_leaf(df, features, target, rng, n_draws=3):
    """Pick min_samples_leaf maximising test AUC subject to the overfit tolerance.

    Averages train/test AUC over a few balanced draws per leaf for stability.
    Falls back to the smallest train-test gap if no leaf meets the tolerance.
    Returns (best_leaf, {leaf: (auc_train, auc_test)}).
    """
    table = {}
    for leaf in config.LEAF_GRID:
        tr, te = [], []
        for _ in range(n_draws):
            bs = balanced_sample(add_random_control(df, rng), target, rng)
            seed = int(rng.integers(1_000_000_000))
            x_tr, x_te, y_tr, y_te = _split(bs, features, target, seed)
            rf = _fit_rf(x_tr, y_tr, leaf, seed)
            tr.append(roc_auc_score(y_tr, rf.predict_proba(x_tr)[:, 1]))
            te.append(roc_auc_score(y_te, rf.predict_proba(x_te)[:, 1]))
        table[leaf] = (float(np.mean(tr)), float(np.mean(te)))

    ok = {l: te for l, (tr, te) in table.items()
          if (tr - te) <= config.AUC_OVERFIT_TOL}
    if ok:
        best = max(ok, key=ok.get)                      # highest test AUC within tol
    else:
        best = min(table, key=lambda l: table[l][0] - table[l][1])  # smallest gap
    return best, table


def run_rf_importance(df, features, target, n_repeats, leaf, rng,
                      perm_repeats=5, log=None, progress_every=25):
    """Repeat the balanced-draw / fit / importance cycle and summarise.

    Returns a dict with, per feature, the median and 5th/95th percentiles of both
    Gini and permutation importance, plus AUC summary and bookkeeping.
    """
    n_feat = len(features)
    gini = np.empty((n_repeats, n_feat))
    perm = np.empty((n_repeats, n_feat))
    aucs = np.empty(n_repeats)

    for i in range(n_repeats):
        bs = balanced_sample(add_random_control(df, rng), target, rng)
        seed = int(rng.integers(1_000_000_000))
        x_tr, x_te, y_tr, y_te = _split(bs, features, target, seed)
        rf = _fit_rf(x_tr, y_tr, leaf, seed)

        gini[i] = rf.feature_importances_
        aucs[i] = roc_auc_score(y_te, rf.predict_proba(x_te)[:, 1])
        pi = permutation_importance(rf, x_te, y_te, scoring="roc_auc",
                                    n_repeats=perm_repeats, random_state=seed,
                                    n_jobs=1)
        perm[i] = pi.importances_mean

        if log and progress_every and (i + 1) % progress_every == 0:
            log(f"    repeat {i + 1:>3d}/{n_repeats}   "
                f"AUC(median so far)={np.median(aucs[:i + 1]):.3f}")

    def _summ(arr):
        return dict(median=np.median(arr, axis=0),
                    p5=np.percentile(arr, 5, axis=0),
                    p95=np.percentile(arr, 95, axis=0))

    return dict(
        features=list(features),
        gini=_summ(gini),
        perm=_summ(perm),
        auc_median=float(np.median(aucs)),
        auc_p5=float(np.percentile(aucs, 5)),
        auc_p95=float(np.percentile(aucs, 95)),
        n_repeats=int(n_repeats),
        leaf=int(leaf),
        n_input=int(len(df)),
    )


def importance_table(res):
    """Tidy per-feature DataFrame from a run_rf_importance result dict."""
    f = res["features"]
    return pd.DataFrame({
        "feature": f,
        "perm_median": res["perm"]["median"],
        "perm_p5": res["perm"]["p5"],
        "perm_p95": res["perm"]["p95"],
        "gini_median": res["gini"]["median"],
        "gini_p5": res["gini"]["p5"],
        "gini_p95": res["gini"]["p95"],
        "auc_median": res["auc_median"],
        "auc_p5": res["auc_p5"],
        "auc_p95": res["auc_p95"],
        "n_input": res["n_input"],
        "leaf": res["leaf"],
        "n_repeats": res["n_repeats"],
    })
