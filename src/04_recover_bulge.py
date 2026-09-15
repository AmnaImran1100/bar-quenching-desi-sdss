"""04_recover_bulge.py: method validation on a known quenching hierarchy.

Before the bar analysis, the random-forest procedure is checked against
results from Bluck et al. (2020) and Piotrowska et al. (2022):

  MAIN  : among disc galaxies, bulge prominence and stellar mass should both
          outperform a random control, on a forest with AUC well above 0.5.
  SIGMA : on the sigma-clean tier, central velocity dispersion sigma_c should
          be the leading predictor of quenching, ahead of bulge and mass
          (the central result of Bluck et al. 2020).

This is a validation step, not a headline measurement.

Scope note for the MAIN test. GZ DESI asks the bulge-size question only for
featured (disc) galaxies, so `bulge_prominence`, and hence this test, is
defined only for that population; the quenched ellipticals that drive the
full-sample "bulge over mass" result are absent. With a coarse five-category
bulge vote compared against a precise stellar mass, the disc-only sample is
expected to rank mass first. In the merged catalogue, bulge prominence
remains associated with quenching at fixed mass (at 10.4 < log M* < 10.8 the
quenched fraction rises from ~0.04 to ~0.25 across bulge tertiles), so bulge is
a useful control variable; no bulge>mass ordering is imposed. This is a
within-disc method check: the upstream friendly-catalogue mask makes this sample
identical to the conventional-scheme disc sample. The quenched fraction of the
eligible sample is low (~0.17), which is why balancing matters.

Reads : data/merged_catalog.parquet
Writes: results/04_bulge_importance.csv, results/04_sigma_importance.csv,
        results/04_summary.txt, figures/04_recover_bulge.png

Run:  py src/04_recover_bulge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_protocol as rf  # noqa: E402


class Audit:
    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self, path):
        Path(path).write_text("\n".join(self.lines), encoding="utf-8")


LOG = Audit()

# Human-readable feature labels for the figure.
NICE = {
    "log_mass": r"$\log M_*$",
    "bulge_prominence": "bulge prominence",
    "sigma_c": r"$\sigma_c$",
    rf.RANDOM: "random control",
}


# ---------------------------------------------------------------------------
# Build the RF input frame (shared column prep).
# ---------------------------------------------------------------------------

def build_frame():
    df = pd.read_parquet(config.MERGED)
    m = df[df["in_main_sample"].to_numpy(bool)].copy()
    m["log_mass"] = pd.to_numeric(m["lgm_tot_p50"], errors="coerce")
    m["bulge_prominence"] = pd.to_numeric(m["bulge_prominence"], errors="coerce")
    m["sigma_c"] = pd.to_numeric(m["v_disp"], errors="coerce")
    return m


def rf_eligible(m, extra_cols=()):
    """Main-sample galaxies with a defined bulge, green valley removed.

    NOTE this validation sample is NOT gated on the comparison-scheme
    featured/face-on thresholds used by 05's fiducial science sample: the only
    disc-ward selection is that bulge_prominence is finite. Because the friendly
    catalogue masks the bulge question outside its applicability thresholds,
    this selects exactly the conventional-scheme disc sample (83,929 galaxies
    after removing the green valley), slightly broader than the fiducial one.

    Target = quenched_dsfr (clearly quenched vs clearly star-forming). Requires
    every listed feature finite so the forest never sees a NaN.
    """
    need = ["log_mass", "bulge_prominence", *extra_cols]
    ok = np.ones(len(m), dtype=bool)
    for c in need:
        ok &= np.isfinite(pd.to_numeric(m[c], errors="coerce").to_numpy(float))
    ok &= ~m["green_valley"].to_numpy(bool)
    return m.loc[ok].reset_index(drop=True)


# ---------------------------------------------------------------------------
# One RF experiment (tune + run + report).
# ---------------------------------------------------------------------------

def experiment(name, df, features, target, rng):
    LOG("=" * 72)
    LOG(f"{name}")
    LOG("=" * 72)
    n_pos = int(df[target].sum())
    LOG(f"  input galaxies ................. {len(df):>10,}")
    LOG(f"  quenched / star-forming ....... {n_pos:>10,} / {len(df) - n_pos:,}"
        f"   (quenched frac {df[target].mean():.3f})")
    LOG(f"  features: {features}")

    leaf, table = rf.tune_leaf(df, features, target, rng)
    LOG("  leaf tuning (min_samples_leaf -> AUC_train, AUC_test, gap):")
    for l, (tr, te) in table.items():
        flag = "  <- chosen" if l == leaf else ""
        tol = "" if (tr - te) <= config.AUC_OVERFIT_TOL else "  (over tol)"
        LOG(f"      {l:>4d}   {tr:.3f}  {te:.3f}   gap={tr - te:+.3f}{tol}{flag}")
    LOG(f"  tuned min_samples_leaf = {leaf}")

    res = rf.run_rf_importance(df, features, target, config.N_RF_REPEATS,
                               leaf, rng, log=LOG)
    LOG(f"  median test AUC = {res['auc_median']:.3f}  "
        f"[5th {res['auc_p5']:.3f}, 95th {res['auc_p95']:.3f}]")

    # rank by permutation importance (the less-biased measure)
    order = np.argsort(res["perm"]["median"])[::-1]
    LOG("  feature ranking (by permutation importance):")
    LOG("      feature            perm[med (5,95)]        gini[med]")
    for j in order:
        f = res["features"][j]
        LOG(f"      {f:18s} {res['perm']['median'][j]:+.4f} "
            f"({res['perm']['p5'][j]:+.4f},{res['perm']['p95'][j]:+.4f})   "
            f"{res['gini']['median'][j]:.4f}")
    return res


# ---------------------------------------------------------------------------
# Verdicts.
# ---------------------------------------------------------------------------

def _perm(res, feat):
    j = res["features"].index(feat)
    return res["perm"]["median"][j], res["perm"]["p5"][j], res["perm"]["p95"][j]


def verdict_main(res):
    """Validate the machinery on the within-disc sample.

    The physically defensible 'known result' to recover here is NOT the
    full-sample 'bulge >> mass' ordering (that is driven by the ellipticals we
    have necessarily excluded, and uses a finer bulge measure than a 5-category
    GZ vote). It is the weaker, robust statement that BOTH structural drivers of
    quenching (bulge prominence and stellar mass) carry real, independent
    predictive power and comfortably beat a random control, on a forest that
    actually learns (AUC well above 0.5). The bulge/mass ordering is reported
    descriptively, with that caveat.
    """
    b_med, b_p5, _ = _perm(res, "bulge_prominence")
    m_med, m_p5, _ = _perm(res, "log_mass")
    r_med, _, r_p95 = _perm(res, rf.RANDOM)
    beats_rand = (b_p5 > r_p95) and (m_p5 > r_p95)
    auc_ok = res["auc_median"] > 0.60
    passed = beats_rand and auc_ok
    leader = "bulge" if b_med > m_med else "mass"
    LOG("")
    LOG("  MAIN verdict (within-disc method check):")
    LOG(f"    bulge beats random ........... {'YES' if b_p5 > r_p95 else 'NO'} "
        f"(perm {b_med:+.4f} vs random {r_med:+.4f})")
    LOG(f"    mass  beats random ........... {'YES' if m_p5 > r_p95 else 'NO'} "
        f"(perm {m_med:+.4f})")
    LOG(f"    forest learns (AUC > 0.60) ... {'YES' if auc_ok else 'NO'} "
        f"({res['auc_median']:.3f})")
    LOG(f"    (descriptive) leading axis ... {leader} "
        f"(bulge {b_med:+.4f} vs mass {m_med:+.4f}; expected mass-leaning in the")
    LOG("                    disc-only regime with a coarse ordinal bulge proxy)")
    LOG(f"    ==> {'PASS' if passed else 'REVIEW'} "
        "(both structural drivers informative, forest discriminates)")
    return passed


def verdict_sigma(res):
    s_med, s_p5, _ = _perm(res, "sigma_c")
    others = [f for f in res["features"] if f != "sigma_c"]
    top = all(s_med > _perm(res, f)[0] for f in others)
    beats_rand = s_p5 > _perm(res, rf.RANDOM)[2]
    auc_ok = res["auc_median"] > 0.60
    passed = top and beats_rand and auc_ok
    LOG("")
    LOG("  SIGMA verdict:")
    LOG(f"    sigma_c is top feature ....... {'YES' if top else 'NO'} "
        f"(perm {s_med:+.4f})")
    LOG(f"    sigma_c beats random ......... {'YES' if beats_rand else 'NO'}")
    LOG(f"    AUC > 0.60 ................... {'YES' if auc_ok else 'NO'} "
        f"({res['auc_median']:.3f})")
    LOG(f"    ==> {'PASS' if passed else 'REVIEW'}")
    return passed


# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------

def _panel(ax, res, title):
    feats = res["features"]
    order = np.argsort(res["perm"]["median"])          # ascending -> best on top
    y = np.arange(len(feats))
    med = res["perm"]["median"][order]
    lo = med - res["perm"]["p5"][order]
    hi = res["perm"]["p95"][order] - med
    colors = ["#8e44ad" if feats[i] == rf.RANDOM else "#2471a3" for i in order]
    ax.barh(y, med, xerr=[lo, hi], color=colors, edgecolor="0.2",
            capsize=3, height=0.6)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([NICE.get(feats[i], feats[i]) for i in order])
    ax.set_xlabel("permutation importance (drop in test AUC)")
    ax.set_title(f"{title}\nmedian AUC = {res['auc_median']:.3f}")


def make_figure(res_main, res_sigma):
    plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.autolayout": True})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    _panel(axes[0], res_main, "MAIN: within-disc quenching")
    _panel(axes[1], res_sigma, r"SIGMA-clean: add $\sigma_c$")
    png = config.FIGURES / "04_recover_bulge.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(config.FIGURES / "04_recover_bulge.pdf", bbox_inches="tight")
    plt.close(fig)
    LOG(f"  saved -> {png.name} + 04_recover_bulge.pdf")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

INTERP = ("Permutation importance measures predictive reliance; correlated "
          "features affect rankings. It provides no general upper or lower "
          "bound on physical causation. Split scatter is not total uncertainty.")


def main():
    rng = np.random.default_rng(config.RANDOM_STATE)
    LOG("04_recover_bulge.py -- method-validation gate\n")

    m = build_frame()

    # ---- MAIN: bulge vs mass vs random, within-disc ----
    df_main = rf_eligible(m)
    res_main = experiment("MAIN  bulge vs mass vs random (within-disc)",
                          df_main,
                          ["log_mass", "bulge_prominence", rf.RANDOM],
                          "quenched_dsfr", rng)
    pass_main = verdict_main(res_main)

    # ---- SIGMA: add sigma_c on the sigma-clean tier ----
    df_sig = rf_eligible(m[m["sigma_clean"].to_numpy(bool)], extra_cols=["sigma_c"])
    res_sigma = experiment("SIGMA  add sigma_c on the sigma-clean tier",
                           df_sig,
                           ["log_mass", "bulge_prominence", "sigma_c", rf.RANDOM],
                           "quenched_dsfr", rng)
    pass_sigma = verdict_sigma(res_sigma)

    # ---- write tidy CSVs ----
    rf.importance_table(res_main).to_csv(config.RESULTS / "04_bulge_importance.csv",
                                         index=False)
    rf.importance_table(res_sigma).to_csv(config.RESULTS / "04_sigma_importance.csv",
                                          index=False)
    LOG("")
    LOG("  wrote results/04_bulge_importance.csv, results/04_sigma_importance.csv")

    make_figure(res_main, res_sigma)

    LOG("")
    LOG("=" * 72)
    LOG("METHOD-VALIDATION VERDICT")
    LOG("=" * 72)
    LOG(f"  MAIN  (bulge & mass both beat random, AUC>0.6): "
        f"{'PASS' if pass_main else 'REVIEW'}")
    LOG(f"  SIGMA (sigma_c is the leading predictor)     : "
        f"{'PASS' if pass_sigma else 'REVIEW'}")
    LOG("")
    LOG("  " + INTERP)
    if pass_main and pass_sigma:
        LOG("  Machinery reproduces the known hierarchy -> cleared for the bar test.")
    else:
        LOG("  One or both checks need review before trusting a bar result "
            "(note the within-disc scope above).")

    out = config.RESULTS / "04_summary.txt"
    LOG.save(out)
    print(f"\nSAVED summary -> {config.rel(out)}")


if __name__ == "__main__":
    main()
