"""06_result_figures.py: turn results/05*.csv into the paper figures.

Step 06 is presentation only: it reads the frozen 05 CSVs and renders the
scientific figures. No new statistics are computed here; every plotted number
comes from results/05*.csv.

Fiducial scheme = "comparison" (the featured/face-on disc definition used for
the headline). The "conventional" scheme lives in the CSVs for the robustness
appendix but is not drawn here.

Figures (figures/, PNG 200 dpi + PDF):
  06a_feature_importance  RF perm + Gini importances, both tiers, random floor
  06b_controlled_binning  Delta q over the (mass, bulge) grid
  06c_partial_correlation corr(bar, dSFR) declining across control sets
  06d_fibre_ssfr          central SF contrasts: statistics and within-sample composition
  06e_bpt_selection       BPT composition and within-class contrasts (no mediation test)

Reads : results/05a_rf_importance_comparison.csv
        results/05b_controlled_binning_comparison_all.csv
        results/05b_controlled_binning_comparison_strong_only.csv
        results/05c_partial_correlation.csv
        results/05d_fibre_ssfr.csv
        results/05e_bpt_breakdown.csv
Writes: figures/06*.png, figures/06*.pdf, results/06_summary.txt

Run:  py src/06_result_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless: write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402


# ---------------------------------------------------------------------------
# Logging helper (same contract as 03/04): everything printed is also saved.
# ---------------------------------------------------------------------------

class Audit:
    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def save(self, path):
        Path(path).write_text("\n".join(self.lines), encoding="utf-8")


LOG = Audit()

SCHEME = "comparison"            # fiducial scheme for every figure here

# Shared palette (matches 03/04 house style).
BLUE = "#2471a3"                # primary / strong bars
PURPLE = "#8e44ad"             # secondary / strong-only
RED = "#c0392b"                # more-quenched / excess
GREY = "#7f8c8d"               # reference / null / random floor
GREEN = "#27ae60"              # star-forming reference

NICE = {
    "log_mass": "stellar mass",
    "bulge_prominence": "bulge",
    "bar_strength": "bar strength",
    "spiral": "spiral arms",
    "sigma_c": r"central $\sigma$",
    "random_control": "random control",
}


# ---------------------------------------------------------------------------
# Shared plotting style + saver (identical contract to 03_validation_figures).
# ---------------------------------------------------------------------------

def set_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.autolayout": True,
    })


def save_fig(fig, name):
    """Write PNG (200 dpi) and PDF to figures/, then close."""
    png = config.FIGURES / f"{name}.png"
    pdf = config.FIGURES / f"{name}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    LOG(f"  saved -> {png.name} + {pdf.name}")


def load(name):
    return pd.read_csv(config.RESULTS / name)


def ivw_delta_q(cells):
    """Inverse-variance-weighted <Delta q> over the valid cells of a Method-B
    CSV, replicating _summarize_cells in 05_bar_test.py exactly (same SE-from-
    interval and IVW helpers), so the figure annotation matches the pipeline's
    headline to the last digit rather than recomputing a different statistic.
    """
    v = cells[cells["valid"].astype(bool)]
    d = v["delta_q"].to_numpy(float)
    se = su.diff_se_from_interval(v["delta_q_lo"].to_numpy(float),
                                  v["delta_q_hi"].to_numpy(float))
    good = np.isfinite(se) & (se > 0)
    mean, semean = su.inverse_variance_weighted_mean(d[good], se[good])
    return mean, semean, int(good.sum())


def re_delta_q(cells):
    """DerSimonian-Laird random-effects <Delta q> over the same valid cells,
    matching the RE numbers in 05_summary.txt (the pipeline's own summary says
    the FE SE understates the uncertainty at the measured I^2)."""
    v = cells[cells["valid"].astype(bool)]
    d = v["delta_q"].to_numpy(float)
    se = su.diff_se_from_interval(v["delta_q_lo"].to_numpy(float),
                                  v["delta_q_hi"].to_numpy(float))
    good = np.isfinite(se) & (se > 0)
    re = su.random_effects_mean(d[good], se[good])
    return re["mean"], re["se"], re["I2"]


# ---------------------------------------------------------------------------
# 06a: RF feature importance (Method A)
# ---------------------------------------------------------------------------

def fig_feature_importance():
    """Permutation + Gini importance for the main and sigma tiers.

    Two rows (tiers) x two columns (permutation, Gini). The random_control bar
    is the noise floor: any feature not clearly above it is uninformative.
    Importance is predictive, depends on correlated features, and does not
    bound physical causation. These forests do not include redshift.
    """
    LOG("=" * 72)
    LOG("06a  RF feature importance (perm + Gini, both tiers)")
    LOG("=" * 72)

    df = load(f"05a_rf_importance_{SCHEME}.csv")
    tiers = [("comparison:main", "within-disc (mass, bulge, bar, spiral)"),
             ("comparison:sigma", r"$\sigma$-clean tier (adds central $\sigma_c$)")]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    for r, (tier, title) in enumerate(tiers):
        sub = df[df["sample"] == tier].copy()
        sub = sub.sort_values("perm_median", ascending=True)   # barh: top = largest
        feats = sub["feature"].tolist()
        ypos = np.arange(len(feats))
        labels = [NICE.get(f, f) for f in feats]
        auc = float(sub["auc_median"].iloc[0])

        for c, (col, lo, hi, name) in enumerate([
                ("perm_median", "perm_p5", "perm_p95", "permutation importance"),
                ("gini_median", "gini_p5", "gini_p95", "Gini importance")]):
            ax = axes[r, c]
            med = sub[col].to_numpy(float)
            err = np.vstack([med - sub[lo].to_numpy(float),
                             sub[hi].to_numpy(float) - med])
            colours = [GREY if f == "random_control" else
                       (RED if f == "bar_strength" else BLUE) for f in feats]
            ax.barh(ypos, med, xerr=err, color=colours, alpha=0.9,
                    error_kw=dict(ecolor="0.3", lw=1.0, capsize=2.5))
            # random-control floor as a vertical guide
            rc = sub[sub["feature"] == "random_control"][col]
            if len(rc):
                ax.axvline(float(rc.iloc[0]), color=GREY, ls=":", lw=1.2, zorder=0)
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels if c == 0 else [])
            ax.set_xlabel(name)
            ax.axvline(0, color="0.6", lw=0.8)
            if c == 0:
                ax.set_ylabel(title, fontsize=10)
            if r == 0:
                ax.set_title(name)
        axes[r, 1].text(0.97, 0.05, f"test AUC = {auc:.3f}",
                        transform=axes[r, 1].transAxes, ha="right", va="bottom",
                        fontsize=10, color="0.35")

    handles = [Patch(color=BLUE, label="structural feature"),
               Patch(color=RED, label="bar strength"),
               Patch(color=GREY, label="random control (noise floor)")]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    # Title derived from the loaded numbers, not asserted: the bar is judged
    # against the random control by the same test the summaries use (its 5th
    # percentile vs the control's 95th), and the wording follows that verdict.
    _above = []
    for tier, _ in tiers:
        _s = df[df["sample"] == tier].set_index("feature")
        if {"bar_strength", "random_control"} <= set(_s.index):
            _above.append(float(_s.loc["bar_strength", "perm_p5"])
                          > float(_s.loc["random_control", "perm_p95"]))
    if _above and all(_above):
        _verdict = ("bar strength is small but resolved above the noise floor")
    elif any(_above):
        _verdict = ("bar strength clears the noise floor in only some tiers")
    else:
        _verdict = "bar strength is consistent with the noise floor"
    fig.suptitle(f"Random-forest importance: {_verdict}", y=1.06, fontsize=13)
    save_fig(fig, "06a_feature_importance")
    for tier, _ in tiers:
        sub = df[df["sample"] == tier].set_index("feature")
        top = sub["perm_median"].idxmax()
        bar_p = float(sub.loc["bar_strength", "perm_median"]) \
            if "bar_strength" in sub.index else np.nan
        floor = float(sub.loc["random_control", "perm_p95"]) \
            if "random_control" in sub.index else np.nan
        LOG(f"  {tier}: top feature = {top} "
            f"(perm {float(sub.loc[top, 'perm_median']):.3f}); "
            f"bar_strength perm = {bar_p:.3f} vs random-control p95 = {floor:.3f}"
            f" -> {'above' if bar_p > floor else 'at/below'} the noise floor.")


# ---------------------------------------------------------------------------
# 06b: controlled (mass, bulge) binning
# ---------------------------------------------------------------------------

def _grid_from(sub):
    """Return (Z, N, valid, mass_centres, bulge_centres) as (n_bulge, n_mass)."""
    nm = int(sub["mass_bin"].max()) + 1
    nb = int(sub["bulge_bin"].max()) + 1
    Z = np.full((nb, nm), np.nan)
    N = np.full((nb, nm), np.nan)
    V = np.zeros((nb, nm), dtype=bool)
    S = np.zeros((nb, nm), dtype=bool)          # significant (interval excludes 0)
    mc = np.full(nm, np.nan)
    bc = np.full(nb, np.nan)
    for _, row in sub.iterrows():
        i, j = int(row["bulge_bin"]), int(row["mass_bin"])
        valid = bool(row["valid"])
        V[i, j] = valid
        mc[j] = 0.5 * (row["mass_lo"] + row["mass_hi"])
        bc[i] = 0.5 * (row["bulge_lo"] + row["bulge_hi"])
        if valid:
            Z[i, j] = row["delta_q"]
            N[i, j] = min(row["n_barred"], row["n_unbarred"])
            S[i, j] = (row["delta_q_lo"] > 0) or (row["delta_q_hi"] < 0)
    return Z, N, V, S, mc, bc


def fig_controlled_binning():
    """Delta q = q(barred) - q(unbarred) at fixed (mass, bulge), two bar samples.

    Diverging colormap centred at zero (RdBu_r): red = barred MORE quenched.
    Invalid cells (< 30 per class) are hatched. A ring marks cells whose 95%
    Newcombe interval excludes zero. The heatmaps are the redshift-uncontrolled
    projection; the annotated summaries are the redshift-stratified estimates.
    """
    LOG("=" * 72)
    LOG("06b  controlled (mass, bulge) binning")
    LOG("=" * 72)

    panels = [("all", "all barred (strong + weak)"),
              ("strong_only", "strong-barred only")]
    grids = {}
    headline = {}
    vlim = 0.0
    for key, _ in panels:
        # These CSVs stack the full ('all') and 'agn_clean' sub-runs on shared
        # bin indices, so select the fiducial full sample; otherwise the grid
        # would plot the AGN-clean cells written last.
        sub = load(f"05b_controlled_binning_{SCHEME}_{key}.csv")
        sub = sub[sub["sample"] == "all"]
        # The heatmap is the z-UNCONTROLLED projection (a single mass x bulge
        # grid); it is the intuitive visual but not the fiducial number.
        grids[key] = _grid_from(sub)
        vlim = max(vlim, np.nanmax(np.abs(grids[key][0])))
        # The FIDUCIAL headline holds z fixed too: recompute the IVW over the
        # z-sliced cells so the annotated number matches the stage-05 summary.
        # The DL random-effects mean is annotated alongside because the cells
        # are heterogeneous (I^2 ~ 0.6): FE alone would look overconfident.
        zsub = load(f"05b_controlled_binning_zsliced_{SCHEME}_{key}.csv")
        zvalid = zsub[zsub["sample"] == "all"]
        zc, zc_se, _ = ivw_delta_q(zvalid)
        re_m, re_se, re_i2 = re_delta_q(zvalid)
        nz, _, _ = ivw_delta_q(sub)
        headline[key] = (zc, zc_se, re_m, re_se, re_i2, nz)
    vlim = float(np.ceil(vlim * 100) / 100)     # tidy symmetric limit

    # constrained_layout places the shared colorbar without overlap; disable the
    # global autolayout (tight_layout), which is incompatible with a multi-axes
    # colorbar and triggers the "results might be incorrect" warning.
    with plt.rc_context({"figure.autolayout": False}):
        return _draw_controlled_binning(panels, grids, vlim, headline)


def _draw_controlled_binning(panels, grids, vlim, headline):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    im = None
    for ax, (key, title) in zip(axes, panels):
        Z, N, V, S, mc, bc = grids[key]
        nb, nm = Z.shape
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vlim, vmax=vlim)
        # hatch invalid cells
        for i in range(nb):
            for j in range(nm):
                if not V[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 fill=False, hatch="///", edgecolor="0.6",
                                 lw=0.0, zorder=2))
                    continue
                # annotate delta q in percentage points + significance ring
                txt = f"{Z[i, j]*100:+.1f}"
                ax.text(j, i + 0.13, txt, ha="center", va="center", fontsize=8.5,
                        color="black", zorder=3)
                ax.text(j, i - 0.24, f"n={int(N[i, j])}", ha="center", va="center",
                        fontsize=9, color="0.25", zorder=3)
                if S[i, j]:
                    ax.add_patch(plt.Circle((j, i), 0.42, fill=False,
                                 edgecolor="black", lw=1.3, zorder=4))
        ax.set_xticks(np.arange(nm))
        ax.set_xticklabels([f"{m:.1f}" for m in mc], fontsize=9)
        ax.set_yticks(np.arange(nb))
        ax.set_yticklabels([f"{b:.2f}" for b in bc], fontsize=9)
        ax.set_xlabel(r"$\log\,M_\star / M_\odot$  (bin centre)")
        ax.set_ylabel("bulge prominence (bin centre)")
        ax.set_title(title)
        ax.grid(False)
        # Annotate the fiducial (z-controlled) FE and DL random-effects means,
        # and label the grid as the z-uncontrolled projection.
        zc, zc_se, re_m, re_se, re_i2, nz = headline[key]
        ax.text(0.5, -0.28,
                r"fiducial $z$-ctl: FE $\langle\Delta q\rangle$ = "
                f"{zc*100:+.2f}$\\pm${zc_se*100:.2f} pp,  "
                f"RE = {re_m*100:+.2f}$\\pm${re_se*100:.2f} pp "
                f"($I^2$={re_i2:.2f})\n"
                f"(heatmap shows the $z$-uncontrolled projection: {nz*100:+.2f} pp)",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                color="0.2")

    cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.02)
    cbar.set_label(r"$\Delta q = q_{\rm barred}-q_{\rm unbarred}$  (red = more quenched)")
    handles = [Patch(facecolor="white", edgecolor="0.6", hatch="///",
                     label="invalid cell (< 30 / class)"),
               plt.Line2D([0], [0], marker="o", mfc="none", mec="black", ls="",
                          ms=10, label="95% interval excludes 0")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.12), fontsize=9)
    # The title states what the loaded numbers actually show rather than
    # asserting the conclusion unconditionally:
    # the sign and significance are read from the fiducial z-controlled
    # random-effects mean of the STRONG-bar panel at render time, so a rerun
    # that changed the result would change the title with it.
    _, _, re_strong, re_strong_se, _, _ = headline["strong_only"]
    if np.isfinite(re_strong_se) and re_strong_se > 0:
        n_sig = abs(re_strong) / re_strong_se
    else:
        n_sig = 0.0
    if n_sig < 2:
        verdict = ("No strong-bar contrast resolved at fixed mass, bulge "
                   "and redshift")
    elif re_strong > 0:
        verdict = ("Strong-barred discs are more quenched at fixed mass, "
                   "bulge and redshift")
    else:
        verdict = ("Strong-barred discs are less quenched at fixed mass, "
                   "bulge and redshift")
    fig.suptitle("Quenched-fraction contrasts: heatmaps omit redshift control\n"
                 f"Fiducial strong-bar random effects: {re_strong*100:+.2f} pp "
                 f"({n_sig:.1f} nominal SE)", fontsize=13)
    save_fig(fig, "06b_controlled_binning")
    LOG("  grid = z-uncontrolled projection; annotated fiducial is z-controlled.")
    for key, (zc, zc_se, re_m, re_se, re_i2, nz) in headline.items():
        LOG(f"    {key:11s}: z-controlled FE <dq>={zc:+.4f}+/-{zc_se:.4f}  "
            f"RE <dq>={re_m:+.4f}+/-{re_se:.4f} (I2={re_i2:.3f})  "
            f"(z-uncontrolled {nz:+.4f})")


# ---------------------------------------------------------------------------
# 06c: rank partial correlation, declining across control sets (Method C).
# ---------------------------------------------------------------------------

def fig_partial_correlation():
    """Control sensitivity on separate main and sigma-clean samples."""
    LOG("=" * 72)
    LOG("06c  partial correlation corr(bar, dSFR) vs controls")
    LOG("=" * 72)

    df = load("05c_partial_correlation.csv")
    df = df[(df["scheme"] == SCHEME) & (df["sample"] == "all")]
    order = ["none", "mass", "mass+bulge", "mass+bulge+z"]
    labels = ["none", "mass", "mass +\nbulge", "mass +\nbulge + z"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, tier in zip(axes, ("main", "sigma")):
        levels = order if tier == "main" else order + ["mass+bulge+sigma+z"]
        ticklabels = labels if tier == "main" else labels + ["mass + bulge\n+ z + sigma"]
        for offset, (subset, colour, label) in zip((-.09, .09), [
                ("strong+weak", BLUE, "all discs"),
                ("strong_only", PURPLE, "strong + unbarred")]):
            sub = df[(df["subset"] == subset) & (df["tier"] == tier)].set_index("controls")
            point = np.array([sub.loc[c, "pcc"] for c in levels])
            lo = np.array([sub.loc[c, "lo"] for c in levels])
            hi = np.array([sub.loc[c, "hi"] for c in levels])
            ax.errorbar(np.arange(len(levels)) + offset, point,
                        yerr=[point-lo, hi-point], marker="o", ls="--",
                        color=colour, capsize=3,
                        label=f"{label} (n={int(sub.iloc[0]['n']):,})")
            LOG(f"  {tier}/{subset}: " + ", ".join(f"{c}={sub.loc[c, 'pcc']:+.5f}" for c in levels))
        ax.axhline(0, color="0.4", lw=.8)
        ax.set_xticks(np.arange(len(levels)), ticklabels, fontsize=9)
        ax.set_title("Main sample" if tier == "main" else "Sigma-clean sample (fixed across levels)")
        ax.set_xlabel("rank controls")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), fontsize=10)
    axes[0].set_ylabel(r"rank partial correlation of bar score and $\Delta$SFR")
    fig.suptitle("Redshift sensitivity of the bar-SFR association", fontsize=13)
    save_fig(fig, "06c_partial_correlation")


# ---------------------------------------------------------------------------
# 06d: central fibre sSFR: suppression + enhancement vs Liu & Zhou.
# ---------------------------------------------------------------------------

def fig_fibre_ssfr():
    """Three panels of central-SF contrasts, compared with Liu & Zhou's results.

    L&Z quote MEAN fibre sSFR (mass+colour-matched controls); medians and means
    legitimately disagree here because the population is bimodal (SF vs
    quenched) and the 9.5-10.5 band mixes two opposite-sign mass regimes.
    LEFT   : strong-unbarred contrast at z<0.05 under BOTH statistics, for the
             full band and the 9.5-10.0 / 10.0-10.5 halves. The full-band mean
             is compatible with zero; the upper-half mean is negative and the
             median contrast is positive, largest at 9.5-10.0. L&Z use
             different matching and measurements.
    MIDDLE : within-sample composition: non-quenched members have a
             positive median contrast, quenched members a negative one, and
             strong bars have a higher in-band quenched fraction (ratio annotated
             from the qfrac_inband row of 05d). The split is the exhaustive
             two-way quenched / non-quenched partition of Section 2.6, so the
             non-quenched group also contains the green valley; see script 09.
    RIGHT  : low-z 9.0-9.5 slice, positive central contrast (same sign as
             L&Z's low-mass result, without bulge matching). No L&Z reference
             level is drawn: they quote no amplitude for this regime.
    """
    LOG("=" * 72)
    LOG("06d  central fibre sSFR (statistics and within-sample composition)")
    LOG("=" * 72)

    df = load("05d_fibre_ssfr.csv")
    df = df[(df["scheme"] == SCHEME) & (df["sample"] == "all")]
    sup = df[df["panel"] == "suppression"]

    def row(control, stat=None):
        r = sup[sup["control"] == control]
        if stat is not None:
            r = r[r["stat"] == stat]
        r = r.iloc[0]
        return float(r["delta"]), float(r["delta_lo"]), float(r["delta_hi"])

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(13, 5.2),
                                        gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # ---- LEFT: median vs mean, band and mass split, all at z<0.05 ----
    groups = [("band\n9.5-10.5",
               row("mass_only_zslice:LZ_z<0.05"),
               row("lz_band_mean_z<0.05", "mean")),
              ("9.5-10.0",
               row("mass_split:9.5-10.0_z<0.05", "median"),
               row("mass_split:9.5-10.0_z<0.05", "mean")),
              ("10.0-10.5",
               row("mass_split:10.0-10.5_z<0.05", "median"),
               row("mass_split:10.0-10.5_z<0.05", "mean"))]
    xx = np.arange(len(groups))
    w = 0.36
    for off, idx, col, lbl in ((-w / 2, 1, BLUE, "median (this work)"),
                               (+w / 2, 2, PURPLE, "mean (this work)")):
        pts = np.array([g[idx][0] for g in groups])
        lo = np.array([g[idx][1] for g in groups])
        hi = np.array([g[idx][2] for g in groups])
        axL.bar(xx + off, pts, w, yerr=np.vstack([pts - lo, hi - pts]),
                color=col, alpha=0.9, label=lbl,
                error_kw=dict(ecolor="0.3", lw=1.0, capsize=4))
    axL.axhline(-0.20, color=GREY, ls="--", lw=1.1, zorder=1,
                label="L&Z $-$0.2 dex (mean, matched controls)")
    axL.axhline(0, color="0.5", lw=0.9)
    axL.set_xticks(xx)
    axL.set_xticklabels([g[0] for g in groups], fontsize=9)
    axL.set_xlabel(r"$\log M_\star$ range at $z<0.05$")
    axL.set_ylabel(r"$\Delta\,\log\,{\rm sSFR_{fib}}$  (strong $-$ unbarred)")
    axL.set_title("Mass-band contrasts")
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.31), fontsize=10)

    # ---- MIDDLE: SF / quenched decomposition (medians, band, z<0.05) ----
    dsf = row("decomp_nonquenched_z<0.05")
    dq = row("decomp_Q_only_z<0.05")
    pts = np.array([dsf[0], dq[0]])
    lo = np.array([dsf[1], dq[1]])
    hi = np.array([dsf[2], dq[2]])
    axM.bar([0, 1], pts, 0.5, yerr=np.vstack([pts - lo, hi - pts]),
            color=[GREEN, RED], alpha=0.9,
            error_kw=dict(ecolor="0.3", lw=1.0, capsize=4))
    axM.axhline(0, color="0.5", lw=0.9)
    axM.set_xticks([0, 1])
    axM.set_xticklabels(["non-quenched\nmembers", "quenched\nmembers"])
    axM.set_ylabel(r"$\Delta\,$median $\log\,{\rm sSFR_{fib}}$")
    axM.set_title("One bimodal population")
    qrow = sup[sup["control"] == "qfrac_inband_z<0.05"].iloc[0]
    q_s, q_u = float(qrow["med_strong"]), float(qrow["med_unbarred"])
    qratio = q_s / q_u if q_u > 0 else np.nan
    axM.text(0.5, -0.31,
             f"Quenched fraction: {q_s:.3f} vs {q_u:.3f}\n"
             f"(strong / unbarred = {qratio:.1f})\n"
             "Non-quenched includes green valley",
             transform=axM.transAxes, ha="center", va="top", fontsize=10,
             color="0.25", clip_on=False)

    # ---- RIGHT: low-z enhancement + L&Z reference ----
    enh = df[df["panel"] == "enhancement"].set_index("control")
    ek = ["mass_only:ssfr_fib", "mass_only:sfr_fib"]
    elbl = [r"$\log\,{\rm sSFR_{fib}}$", r"$\log\,{\rm SFR_{fib}}$"]
    emed = np.array([enh.loc[k, "delta"] for k in ek])
    elo = np.array([enh.loc[k, "delta_lo"] for k in ek])
    ehi = np.array([enh.loc[k, "delta_hi"] for k in ek])
    eerr = np.vstack([emed - elo, ehi - emed])
    ex = np.arange(len(ek))
    # NO L&Z reference level is drawn on this panel. Their ~0.2 dex figure is a
    # DEFICIT in fibre sSFR at 9.5 < log M* < 10.5 (marked on the left panel);
    # for this low-mass regime they report the enhancement qualitatively and
    # quote no amplitude at all. Drawing +0.2 dex here would attribute a number
    # to them with the wrong sign in the wrong mass range.
    axR.bar(ex, emed, 0.5, yerr=eerr, color=GREEN, alpha=0.9,
            error_kw=dict(ecolor="0.3", lw=1.0, capsize=4))
    axR.axhline(0, color="0.5", lw=0.9)
    axR.set_xticks(ex)
    axR.set_xticklabels(elbl)
    axR.set_ylabel(r"$\Delta\,$median  (strong $-$ unbarred)")
    axR.set_title(r"$9.0<\log M_\star<9.5$: enhancement")

    fig.suptitle("Central star formation: statistics and population composition", y=1.02)
    save_fig(fig, "06d_fibre_ssfr")
    bm = row("lz_band_mean_z<0.05", "mean")
    hm = row("mass_split:10.0-10.5_z<0.05", "mean")
    LOG(f"  same-statistic (MEAN), different matching band contrast @ z<0.05: {bm[0]:+.3f} "
        f"[{bm[1]:+.3f},{bm[2]:+.3f}]; 10.0-10.5 half (mean) {hm[0]:+.3f}; "
        f"non-quenched {dsf[0]:+.3f}, Q-only {dq[0]:+.3f}; in-band qfrac "
        f"{q_s:.3f} vs {q_u:.3f} ({qratio:.1f}x); low-z sSFR enhancement "
        f"{float(emed[0]):+.2f} dex. This sample does not reproduce the L&Z full-band suppression.")


# ---------------------------------------------------------------------------
# 06e: BPT composition and within-class contrasts.
# ---------------------------------------------------------------------------

def fig_bpt_selection():
    """Show how the BPT 1-2 (agn_clean) selection changes the population.

    MPA-JHU galSpecExtra BPTCLASS convention: -1/0 = unclassifiable (no usable
    lines), 1 = SF, 2 = low-S/N SF, 3 = composite, 4 = AGN excluding LINERs
    (Seyfert-like), 5 = low-S/N LINER.

    LEFT : fraction of each bar class REMOVED by agn_clean (which keeps only
           classes 1,2), decomposed into no-line, composite, AGN and low-S/N
           LINER. Strong bars lose a far larger share than unbarred discs (the
           percentages are read from 05e), so the compared population changes.
    RIGHT: within-class comparison. Quenched fraction strong vs unbarred among
           BPT AGN hosts (class 4) and, alongside, among composites (class 3).
           Conditioning on current emission-line class does not identify or
           exclude AGN mediation.
           The mass-controlled <Delta q> (Newcombe/IVW over mass bins, from
           05e) is annotated for each population; the verdict in the title is
           computed from whether the AGN-host statistic is within 2 sigma of
           zero.
    """
    LOG("=" * 72)
    LOG("06e  BPT composition / selection effect")
    LOG("=" * 72)

    df = load("05e_bpt_breakdown.csv")
    main = df[df["bar_class"].isin(["strong", "weak", "unbarred"])].set_index("bar_class")
    classes = ["strong", "weak", "unbarred"]

    def pop(tag):
        """(qfrac_strong, n_strong, qfrac_unbarred, n_unbarred, wm, wse, nbins)
        for one BPT population's rows in 05e."""
        s = df[df["bar_class"] == f"{tag}:strong"].iloc[0]
        u = df[df["bar_class"] == f"{tag}:unbarred"].iloc[0]
        m = df[df["bar_class"] == f"{tag}:mass_controlled"].iloc[0]
        return (float(s["qfrac"]), int(s["n"]), float(u["qfrac"]), int(u["n"]),
                float(m["qfrac"]), float(m["qfrac_se"]),
                int(m["n"]))

    agn = pop("agn_only")
    cmp_ = pop("composite_only")
    agn_null = not (np.isfinite(agn[5]) and agn[5] > 0
                    and abs(agn[4]) > 2 * agn[5])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- LEFT: stacked removed fraction (classes -1/0, 3, 4, 5) ----
    removed = np.array([main.loc[c, "frac_removed_agn_clean"] for c in classes])
    noline = np.array([main.loc[c, "frac_noline"] for c in classes])
    compf = np.array([main.loc[c, "frac_composite"] for c in classes])
    agnf = np.array([main.loc[c, "frac_agn"] for c in classes])
    linerf = np.array([main.loc[c, "frac_liner_lowsn"] for c in classes])
    x = np.arange(len(classes))
    w = 0.6
    bottom = np.zeros(len(classes))
    for vals, colour, lbl in ((noline, GREY, "no usable lines ($-1$, 0)"),
                              (compf, "#d6b656", "composite (3)"),
                              (agnf, RED, "AGN excl. LINERs (4)"),
                              (linerf, PURPLE, "low-S/N LINER (5)")):
        axL.bar(x, vals, w, bottom=bottom, color=colour, alpha=0.9, label=lbl)
        bottom += vals
    for j in range(len(classes)):
        axL.text(j, removed[j] + 0.012, f"{removed[j]*100:.0f}% cut",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axL.set_xticks(x)
    axL.set_xticklabels(classes)
    axL.set_ylabel("fraction removed by agn_clean (keeps BPT 1, 2)")
    axL.set_ylim(0, max(removed) * 1.30)
    axL.set_title("BPT 1-2 selection removes bar classes unequally")
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10)

    # ---- RIGHT: quenched fraction among AGN hosts and composites ----
    groups = [("BPT AGN\n(class 4)", agn), ("composite\n(class 3)", cmp_)]
    ex = np.arange(len(groups))
    wb = 0.36
    for off, idx_q, idx_n, colour, lbl in ((-wb / 2, 0, 1, BLUE, "strong bars"),
                                           (+wb / 2, 2, 3, GREY, "unbarred")):
        qv = np.array([g[1][idx_q] for g in groups])
        nv = [g[1][idx_n] for g in groups]
        bars = axR.bar(ex + off, qv, wb, color=colour, alpha=0.9, label=lbl)
        for b, qi, ni in zip(bars, qv, nv):
            axR.text(b.get_x() + b.get_width() / 2, qi + 0.002,
                     f"{qi:.3f}\n(n={ni})", ha="center", va="bottom",
                     fontsize=8)
    axR.set_xticks(ex)
    axR.set_xticklabels([g[0] for g in groups])
    axR.set_ylabel("quenched fraction within BPT population")
    top = max(max(g[1][0], g[1][2]) for g in groups)
    axR.set_ylim(0, top * 1.45)
    axR.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10)
    axR.set_title("Within-BPT contrast: "
                  + ("no difference detected"
                     if agn_null else "difference detected"))
    axR.text(0.5, 0.90,
             f"mass-controlled $\\langle\\Delta q\\rangle$ "
             f"(AGN): {agn[4]:+.4f}$\\pm${agn[5]:.4f} "
             f"({agn[6]} bins)\n"
             f"(composite): {cmp_[4]:+.4f}$\\pm${cmp_[5]:.4f} "
             f"({cmp_[6]} bins)",
             transform=axR.transAxes, ha="center", va="top", fontsize=9,
             color="0.3")

    fig.suptitle("BPT selection changes the population; AGN mediation remains unresolved",
                 y=1.02)
    save_fig(fig, "06e_bpt_selection")
    r_s = float(main.loc["strong", "frac_removed_agn_clean"])
    r_u = float(main.loc["unbarred", "frac_removed_agn_clean"])
    LOG(f"  removed by agn_clean: strong {r_s:.0%} vs unbarred {r_u:.0%}.")
    LOG(f"  AGN-only (BPT 4) qfrac: strong {agn[0]:.3f} (n={agn[1]}) vs "
        f"unbarred {agn[2]:.3f} (n={agn[3]}); mass-controlled "
        f"{agn[4]:+.4f}+/-{agn[5]:.4f} over {agn[6]} bins "
        f"({'not detected' if agn_null else 'detected'}).")
    LOG(f"  composite-only (BPT 3) qfrac: strong {cmp_[0]:.3f} vs unbarred "
        f"{cmp_[2]:.3f}; mass-controlled {cmp_[4]:+.4f}+/-{cmp_[5]:.4f} "
        f"over {cmp_[6]} bins.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    set_style()
    config.FIGURES.mkdir(parents=True, exist_ok=True)

    LOG("06_result_figures.py: figures from the stage-05 CSVs")
    LOG(f"  fiducial scheme = {SCHEME}")
    LOG("")

    fig_feature_importance()
    fig_controlled_binning()
    fig_partial_correlation()
    fig_fibre_ssfr()
    fig_bpt_selection()

    LOG("")
    LOG("  wrote figures/06a..06e (PNG + PDF)")
    LOG.save(config.RESULTS / "06_summary.txt")
    print(f"\nwrote {config.rel(config.RESULTS / '06_summary.txt')}")


if __name__ == "__main__":
    main()
