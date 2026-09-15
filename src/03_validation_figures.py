"""03_validation_figures.py: sample validation figures.

These checks run before any bar analysis. If the sSFR distribution is not
bimodal, the script stops, because a sample without a distinct quenched
population cannot test quenching.

The merged catalogue contains no photometry (52 columns of MPA-JHU spectroscopy
and GZ votes), so a g-r colour bimodality cannot be computed. The log sSFR
bimodality is used instead; it reflects the same star-forming/quenched division,
on the quantity the Delta SFR definition is built from.

Figures (figures/, PNG >=150 dpi + PDF):
  03a_ssfr_bimodality   sSFR histogram + 2-component GMM   [gate]
  03b_sfms              logM* vs logSFR, fitted MS ridge overplotted, DeltaSFR-coded
  03c_dsfr_distribution DeltaSFR hist (with -1.1/-0.5) + sSFR hist (with -11)
  03d_barfraction_mass  strong / weak / total bar fraction vs logM*, Wilson errors
  03e_match_separation  cross-match separation histogram (supplementary)
  03f_mass_completeness logM* vs z, showing flux-limit incompleteness

Reads : data/merged_catalog.parquet
Writes: figures/03*.png, figures/03*.pdf, results/03_validation_summary.txt

Run:  py src/03_validation_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless: write files, never open a window
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import wilson_interval  # noqa: E402


# ---------------------------------------------------------------------------
# Logging helper (same contract as 02): everything printed is also saved.
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

# Column aliases (physics-readable).
LOGM = "lgm_tot_p50"
LOGSFR = "sfr_tot_p50"
LOGSSFR = "specsfr_tot_p50"
DSFR = "delta_sfr"


# ---------------------------------------------------------------------------
# Shared plotting style + saver.
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
    """Write PNG (>=150 dpi) and PDF to figures/, then close."""
    png = config.FIGURES / f"{name}.png"
    pdf = config.FIGURES / f"{name}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    LOG(f"  saved -> {png.name} + {pdf.name}")


def _col(df, name):
    return pd.to_numeric(df[name], errors="coerce").to_numpy(float)


# ---------------------------------------------------------------------------
# 03a: sSFR bimodality (gate)
# ---------------------------------------------------------------------------

def fig_ssfr_bimodality(main):
    """log sSFR histogram + 2-component Gaussian mixture. The bimodality gate.

    Passes if the two GMM components are separated by > 0.8 dex and the minority
    component carries > 15% of the weight, i.e. a distinct second population
    rather than a shoulder. Returns a dict recording the verdict.
    """
    LOG("=" * 72)
    LOG("03a  sSFR bimodality  [gate]")
    LOG("=" * 72)

    s = _col(main, LOGSSFR)
    s = s[np.isfinite(s)]
    LOG(f"  main-sample galaxies with finite log sSFR ..... {s.size:>12,}")

    gmm = GaussianMixture(n_components=2, random_state=config.RANDOM_STATE,
                          n_init=5).fit(s.reshape(-1, 1))
    order = np.argsort(gmm.means_.ravel())          # low sSFR (quenched) first
    mus = gmm.means_.ravel()[order]
    sigs = np.sqrt(gmm.covariances_.ravel())[order]
    wts = gmm.weights_.ravel()[order]

    separation = float(mus[1] - mus[0])
    minority_w = float(wts.min())
    passed = (separation > 0.8) and (minority_w > 0.15)

    LOG(f"  GMM component 1 (quenched):  mu={mus[0]:+.3f}  sigma={sigs[0]:.3f}  w={wts[0]:.3f}")
    LOG(f"  GMM component 2 (star-forming): mu={mus[1]:+.3f}  sigma={sigs[1]:.3f}  w={wts[1]:.3f}")
    LOG(f"  peak separation ...... {separation:.3f} dex   (require > 0.80)")
    LOG(f"  minority weight ...... {minority_w:.3f}       (require > 0.15)")
    LOG(f"  GATE: {'PASS: two populations present' if passed else 'FAIL: bimodality absent, stopping'}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    lo, hi = np.percentile(s, [0.2, 99.8])
    bins = np.linspace(lo, hi, 80)
    ax.hist(s, bins=bins, density=True, color="0.80", edgecolor="0.55",
            linewidth=0.4, label="main sample")

    xs = np.linspace(lo, hi, 500)
    comp_labels = ["quenched", "star-forming"]
    comp_colors = ["#c0392b", "#2471a3"]
    total = np.zeros_like(xs)
    for i in range(2):
        pdf = wts[i] * _gauss(xs, mus[i], sigs[i])
        total += pdf
        ax.plot(xs, pdf, color=comp_colors[i], lw=2,
                label=f"{comp_labels[i]}  ($\\mu$={mus[i]:.2f})")
    ax.plot(xs, total, color="black", lw=1.5, ls="--", label="GMM total")

    ax.axvline(config.SSFR_QUENCHED_CUT, color="green", ls=":", lw=1.8,
               label=f"sSFR cut = {config.SSFR_QUENCHED_CUT:.1f}")
    ax.set_xlabel(r"$\log\,\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")
    ax.set_ylabel("normalised density")
    verdict = "PASS" if passed else "FAIL"
    ax.set_title(f"sSFR bimodality gate: {verdict}  "
                 f"($\\Delta\\mu$ = {separation:.2f} dex)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=2, fontsize=10)
    save_fig(fig, "03a_ssfr_bimodality")

    return dict(passed=passed, separation=separation, minority_w=minority_w,
                mus=mus.tolist(), wts=wts.tolist())


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------------------
# 03b: star-forming main sequence with fitted ridge
# ---------------------------------------------------------------------------

def _recover_ridge(main):
    """Recover the MS ridge line log SFR_MS = a*logM + b exactly.

    DeltaSFR was defined in 02 as logSFR - logSFR_MS with a linear ridge, so
    (logSFR - DeltaSFR) is that ridge evaluated per galaxy: one OLS recovers a,b.
    """
    m = _col(main, LOGM)
    ridge = _col(main, LOGSFR) - _col(main, DSFR)
    ok = np.isfinite(m) & np.isfinite(ridge)
    a, b = np.polyfit(m[ok], ridge[ok], 1)
    return float(a), float(b)


def fig_sfms(main):
    LOG("=" * 72)
    LOG("03b  star-forming main sequence + fitted ridge")
    LOG("=" * 72)

    m = _col(main, LOGM)
    sfr = _col(main, LOGSFR)
    d = _col(main, DSFR)
    ok = np.isfinite(m) & np.isfinite(sfr) & np.isfinite(d)
    m, sfr, d = m[ok], sfr[ok], d[ok]

    a, b = _recover_ridge(main)
    LOG(f"  recovered ridge:  log SFR_MS = {a:.3f} * logM + ({b:.3f})")
    LOG(f"  Renzini & Peng 2015 reference:  0.760 * logM + (-7.640)")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    hb = ax.hexbin(m, sfr, C=d, reduce_C_function=np.median,
                   gridsize=55, cmap="RdBu", vmin=-1.5, vmax=1.5,
                   mincnt=5, linewidths=0.0)
    hb.set_rasterized(True)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r"median $\Delta$SFR in cell [dex]")

    xs = np.linspace(np.percentile(m, 0.5), np.percentile(m, 99.5), 100)
    ax.plot(xs, a * xs + b, "k-", lw=2.2, label="fitted MS ridge")
    ax.plot(xs, 0.760 * xs - 7.640, color="darkorange", lw=2.0, ls="--",
            label="Renzini & Peng 2015")
    ax.plot(xs, a * xs + b + config.DSFR_QUENCHED, "k:", lw=1.4,
            label=f"quenched line ($\\Delta$SFR={config.DSFR_QUENCHED})")

    ax.set_xlabel(r"$\log\,M_*\ [M_\odot]$")
    ax.set_ylabel(r"$\log\,\mathrm{SFR}\ [M_\odot\,\mathrm{yr}^{-1}]$")
    ax.set_title("Star-forming main sequence (main sample)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=10)
    save_fig(fig, "03b_sfms")


# ---------------------------------------------------------------------------
# 03c: DeltaSFR and sSFR distributions with the quenching cuts drawn
# ---------------------------------------------------------------------------

def fig_dsfr_distribution(main):
    LOG("=" * 72)
    LOG("03c  DeltaSFR + sSFR distributions with cut lines")
    LOG("=" * 72)

    d = _col(main, DSFR)
    s = _col(main, LOGSSFR)
    d = d[np.isfinite(d)]
    s = s[np.isfinite(s)]

    q_frac = float((d < config.DSFR_QUENCHED).mean())
    gv_frac = float(((d >= config.DSFR_QUENCHED) & (d < config.DSFR_GV_HI)).mean())
    LOG(f"  quenched fraction (DeltaSFR < {config.DSFR_QUENCHED}) ....... {q_frac:.3f}")
    LOG(f"  green-valley fraction ({config.DSFR_QUENCHED} to {config.DSFR_GV_HI}) . {gv_frac:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    lo, hi = np.percentile(d, [0.2, 99.8])
    ax.hist(d, bins=np.linspace(lo, hi, 80), color="0.80", edgecolor="0.55",
            linewidth=0.4)
    ax.axvline(config.DSFR_QUENCHED, color="#c0392b", ls="--", lw=1.8,
               label=f"quenched = {config.DSFR_QUENCHED}")
    ax.axvline(config.DSFR_GV_HI, color="green", ls="--", lw=1.8,
               label=f"green-valley hi = {config.DSFR_GV_HI}")
    ax.set_xlabel(r"$\Delta$SFR [dex]")
    ax.set_ylabel("galaxies")
    ax.set_title(r"$\Delta$SFR distribution")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=10)

    ax = axes[1]
    lo, hi = np.percentile(s, [0.2, 99.8])
    ax.hist(s, bins=np.linspace(lo, hi, 80), color="0.80", edgecolor="0.55",
            linewidth=0.4)
    ax.axvline(config.SSFR_QUENCHED_CUT, color="green", ls=":", lw=1.8,
               label=f"flat cut = {config.SSFR_QUENCHED_CUT}")
    ax.set_xlabel(r"$\log\,\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")
    ax.set_ylabel("galaxies")
    ax.set_title("sSFR distribution (robustness cut)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=10)

    save_fig(fig, "03c_dsfr_distribution")


# ---------------------------------------------------------------------------
# 03d: bar fraction vs stellar mass (fiducial 'comparison' scheme)
# ---------------------------------------------------------------------------

def fig_barfraction_vs_mass(df):
    """Strong / weak / total bar fraction vs logM* with Wilson binomial errors.

    Sample: the fiducial disc sample (in_sample_comparison), i.e. galaxies that pass
    the disc+orientation gate and carry usable bar votes. Binned over logM* down
    to 9.0 so the weak-bar peak (~10^9.5) and the rising strong-bar branch are
    both visible, cf. Liu & Zhou 2026 Fig. 1 / Geron 2021.
    """
    LOG("=" * 72)
    LOG("03d  bar fraction vs stellar mass  (comparison scheme)")
    LOG("=" * 72)

    disc = df[df["in_sample_comparison"].to_numpy(bool)].copy()
    m = _col(disc, LOGM)
    cls = disc["bar_class_comparison"].astype(str).to_numpy()
    ok = np.isfinite(m) & np.isin(cls, ["unbarred", "weak", "strong"])
    m, cls = m[ok], cls[ok]
    LOG(f"  classified disc galaxies ....................... {m.size:>12,}")

    edges = np.arange(9.0, 11.4 + 1e-9, 0.2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for i in range(len(centers)):
        sel = (m >= edges[i]) & (m < edges[i + 1])
        n = int(sel.sum())
        if n < 50:                                   # too few galaxies to bin
            rows.append((centers[i], n, np.nan, np.nan, np.nan))
            continue
        n_strong = int((cls[sel] == "strong").sum())
        n_weak = int((cls[sel] == "weak").sum())
        rows.append((centers[i], n, n_strong / n, n_weak / n,
                     (n_strong + n_weak) / n))

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    series = [("strong", 2, "#c0392b", "o"),
              ("weak", 3, "#e67e22", "s"),
              ("total", 4, "#2471a3", "^")]
    for label, idx, color, marker in series:
        xs, ys, los, his = [], [], [], []
        for r in rows:
            c, n, fs, fw, ft = r
            frac = r[idx]
            if not np.isfinite(frac):
                continue
            k = int(round(frac * n))
            lo, hi = wilson_interval(k, n)
            xs.append(c); ys.append(frac); los.append(frac - lo); his.append(hi - frac)
        ax.errorbar(xs, ys, yerr=[los, his], marker=marker, color=color,
                    lw=1.6, capsize=2.5, label=f"{label} bar")

    ax.set_xlabel(r"$\log\,M_*\ [M_\odot]$")
    ax.set_ylabel("bar fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Bar fraction vs stellar mass (comparison scheme)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, fontsize=10)
    save_fig(fig, "03d_barfraction_mass")

    # log the numbers so the trend can be checked without opening the PNG
    LOG("    logM   N     f_strong  f_weak   f_total")
    for c, n, fs, fw, ft in rows:
        if np.isfinite(fs):
            LOG(f"    {c:4.2f}  {n:6d}  {fs:7.3f}  {fw:7.3f}  {ft:7.3f}")
        else:
            LOG(f"    {c:4.2f}  {n:6d}   (skipped, n<50)")


# ---------------------------------------------------------------------------
# 03e: cross-match separation (supplementary)
# ---------------------------------------------------------------------------

def fig_match_separation(df):
    LOG("=" * 72)
    LOG("03e  cross-match separation")
    LOG("=" * 72)

    sep = _col(df, "sep_arcsec")
    sep = sep[np.isfinite(sep)]
    med = float(np.median(sep))
    LOG(f"  matched galaxies ............................... {sep.size:>12,}")
    LOG(f"  median separation .............................. {med:.3f} arcsec")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(sep, bins=np.linspace(0, config.MATCH_RADIUS_ARCSEC, 60),
            color="#2471a3", edgecolor="0.3", linewidth=0.3)
    ax.axvline(med, color="black", ls="--", lw=1.5, label=f"median = {med:.3f}\"")
    ax.axvline(config.MATCH_RADIUS_ARCSEC, color="#c0392b", ls=":", lw=1.8,
               label=f"match radius = {config.MATCH_RADIUS_ARCSEC}\"")
    ax.set_xlabel("separation [arcsec]")
    ax.set_ylabel("matched galaxies")
    ax.set_title("GZ DESI x MPA-JHU cross-match separation")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=10)
    save_fig(fig, "03e_match_separation")


# ---------------------------------------------------------------------------
# 03f: mass-redshift distribution
# ---------------------------------------------------------------------------

def fig_mass_completeness(main):
    """Empirical mass distribution, not a calibrated completeness limit.

    The fifth percentile describes an already selected sample. Re-forming
    bins in redshift slices does not recover galaxies missing from it.
    """
    LOG("=" * 72)
    LOG("03f  empirical mass-redshift lower envelope")
    LOG("=" * 72)

    z = _col(main, "z")
    m = _col(main, LOGM)
    ok = np.isfinite(z) & np.isfinite(m)
    z, m = z[ok], m[ok]

    edges = np.linspace(config.Z_MIN, config.Z_MAX, 14)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p05, p50 = [], []
    for i in range(len(centers)):
        sel = (z >= edges[i]) & (z < edges[i + 1])
        if sel.sum() < 30:
            p05.append(np.nan); p50.append(np.nan); continue
        p05.append(np.percentile(m[sel], 5))
        p50.append(np.percentile(m[sel], 50))
    p05 = np.array(p05); p50 = np.array(p50)

    # z beyond which the 5th-percentile mass DEPARTS from the main-sample floor.
    #
    # Note the simple test (first bin with p05 > LOGMASS_MIN) is uninformative here:
    # the main sample is defined by logM > 9.5, so its 5th percentile exceeds
    # 9.5 in EVERY bin by construction and the test always returns the first
    # bin. What is informative is where the envelope pulls away from the floor,
    # so we report the first bin exceeding it by more than MARGIN dex, plus the
    # actual run of the envelope.
    MARGIN = 0.10
    z_incomplete = np.nan
    above = np.isfinite(p05) & (p05 > config.LOGMASS_MIN + MARGIN)
    if above.any():
        z_incomplete = float(centers[above][0])
    LOG(f"  5th-pct logM* stays within {MARGIN:.2f} dex of the "
        f"{config.LOGMASS_MIN} floor out to z ~ "
        f"{z_incomplete if np.isfinite(z_incomplete) else float('nan'):.3f}, "
        f"then climbs")
    good = np.isfinite(p05)
    LOG(f"     (5th-pct logM* = {p05[good][0]:.2f} at z={centers[good][0]:.3f} "
        f"-> {p05[good][-1]:.2f} at z={centers[good][-1]:.3f})")
    LOG("  This lower envelope describes the selected sample; it does not")
    LOG("  measure a completeness probability. Redshift stratification does")
    LOG("  not recover missing low-mass galaxies or remove all selection effects.")

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    hb = ax.hexbin(z, m, gridsize=55, cmap="viridis", bins="log", mincnt=1)
    hb.set_rasterized(True)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r"$\log_{10}$ galaxies")
    ax.plot(centers, p05, "w-", lw=2.2, label="5th-pct $\\log M_*$ (5th percentile)")
    ax.plot(centers, p50, color="white", lw=1.4, ls="--", label="median $\\log M_*$")
    ax.axhline(config.LOGMASS_MIN, color="#c0392b", ls=":", lw=1.8,
               label=f"main-sample floor = {config.LOGMASS_MIN}")
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"$\log\,M_*\ [M_\odot]$")
    ax.set_title("Mass distribution versus redshift (selected sample)")
    # Dark framed legend: the white completeness/median lines and the red
    # floor line are only legible against a dark patch, and the frame stops
    # the floor line from striking through the legend text.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=10,
              frameon=True, facecolor="0.20",
              framealpha=0.85, edgecolor="none", labelcolor="white")
    save_fig(fig, "03f_mass_completeness")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    set_style()
    LOG("03_validation_figures.py: sample validation\n")

    df = pd.read_parquet(config.MERGED)
    LOG(f"merged catalog: {len(df):,} rows x {df.shape[1]} cols")
    main_df = df[df["in_main_sample"].to_numpy(bool)].copy()
    LOG(f"main sample (logM > {config.LOGMASS_MIN}): {len(main_df):,}\n")

    gate = fig_ssfr_bimodality(main_df)
    fig_sfms(main_df)
    fig_dsfr_distribution(main_df)
    fig_barfraction_vs_mass(df)
    fig_match_separation(df)
    fig_mass_completeness(main_df)

    LOG("")
    LOG("=" * 72)
    LOG("VALIDATION VERDICT")
    LOG("=" * 72)
    if gate["passed"]:
        LOG("  sSFR bimodality: PASS (a distinct quenched population is present).")
        LOG("  Proceed to steps 04 and 05.")
    else:
        LOG("  sSFR bimodality: FAIL. No second population; do not run")
        LOG("  the bar test until the sample/quenching definition is debugged.")

    out = config.RESULTS / "03_validation_summary.txt"
    LOG(f"\nSAVED summary -> {config.rel(out)}")
    LOG("=" * 72)
    LOG.save(out)

    if not gate["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
