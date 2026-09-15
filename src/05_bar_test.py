"""Methods A--E for conditional bar--quenching associations.

A: predictive forest importance; neither tier includes redshift.
B: quenched-fraction contrasts in mass--bulge cells within redshift slices,
   with Wilson/Newcombe intervals and fixed/random-effects summaries.
C: sequential rank controls including redshift, on fixed main/sigma samples.
D: fibre-sSFR means, medians and population decomposition in overlapping
   mass/redshift ranges. This is not the mass-and-colour matching of Liu & Zhou.
E: within-BPT contrasts and selection fractions; no causal mediation test.

All methods share data and systematic limitations. Statistical errors are
conditional on the analysis and exclude uncalibrated selection and measurement
systematics. Agreement between methods does not establish causation.

Reads : data/merged_catalog.parquet
Writes: results/05a_rf_importance_{scheme}.csv
        results/05b_controlled_binning_{scheme}_{barmode}.csv
        results/05c_partial_correlation.csv
        results/05d_fibre_ssfr.csv
        results/05e_bpt_breakdown.csv
        results/05_summary.txt

Run:  py src/05_bar_test.py
      py src/05_bar_test.py --force-recompute   # ignore any cached Method-A RF
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_protocol as rf          # noqa: E402
import stats_utils as su          # noqa: E402
import provenance

SCHEMES = ["comparison", "conventional"]

# Method-A caches require matching input, code, configuration and dependency
# fingerprints, plus a checksum of the cached CSV.
FORCE_RECOMPUTE = "--force-recompute" in sys.argv

# How the ordinal bar_class maps onto a barred / unbarred contrast. The
# unbarred class is the common control in every mode; only the "barred" side
# changes, which isolates whether a bar effect (if any) is carried by strong
# bars, weak bars, or both.
BARMODES = {
    "all":         (("weak", "strong"), ("unbarred",)),
    "strong_only": (("strong",),        ("unbarred",)),
    "weak_only":   (("weak",),          ("unbarred",)),
}

# Number of bootstrap resamples.
N_BOOT = 1000


class Audit:
    """Logger that flushes every line to disk immediately, so a crash or an
    external kill can never lose the log (or an eventual traceback)."""

    def __init__(self, path=None):
        self.lines = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.write_text("", encoding="utf-8")   # truncate at start

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")

    def save(self, path):
        Path(path).write_text("\n".join(self.lines), encoding="utf-8")


LOG = Audit()  # Importing analysis helpers must not truncate a result file.


# ===========================================================================
# Shared frame construction
# ===========================================================================

def build_frame(scheme):
    """Main-sample disc galaxies that pass `scheme`'s gate and carry usable
    bar+bulge votes, with analysis columns renamed to their physical meaning.

    Every science column below is non-null on this subset (verified in 02); the
    only optionally-missing feature is sigma_c (v_disp), used solely on the
    sigma-clean tier where finiteness is imposed explicitly.
    """
    df = pd.read_parquet(config.MERGED)
    m = df[df["in_main_sample"].to_numpy(bool)
           & df[f"in_sample_{scheme}"].to_numpy(bool)].copy()
    out = pd.DataFrame(index=m.index)
    out["z"] = pd.to_numeric(m["z"], errors="coerce")
    out["log_mass"] = pd.to_numeric(m["lgm_tot_p50"], errors="coerce")
    out["bulge_prominence"] = pd.to_numeric(m["bulge_prominence"], errors="coerce")
    out["bar_strength"] = pd.to_numeric(m["bar_strength"], errors="coerce")
    out["spiral"] = pd.to_numeric(m["has-spiral-arms_yes_fraction"], errors="coerce")
    out["sigma_c"] = pd.to_numeric(m["v_disp"], errors="coerce")
    out["delta_sfr"] = pd.to_numeric(m["delta_sfr"], errors="coerce")
    out["specsfr_fib"] = pd.to_numeric(m["specsfr_fib_p50"], errors="coerce")
    out["specsfr_tot"] = pd.to_numeric(m["specsfr_tot_p50"], errors="coerce")
    out["sfr_fib"] = pd.to_numeric(m["sfr_fib_p50"], errors="coerce")
    out["bptclass"] = pd.to_numeric(m["bptclass"], errors="coerce")
    out["quenched_dsfr"] = m["quenched_dsfr"].to_numpy(bool)
    out["green_valley"] = m["green_valley"].to_numpy(bool)
    # Flat-sSFR quenched flags for the quenched-definition robustness check
    # (Method B): log sSFR_tot < cut => quenched. specsfr_tot is finite on the
    # main sample (valid_ssfr required in 02), so NaN handling is moot.
    for cut in config.SSFR_QUENCHED_VARIANTS:
        out[f"quenched_ssfr_{cut}"] = out["specsfr_tot"] < cut
    out["bar_class"] = m[f"bar_class_{scheme}"].to_numpy()
    out["sigma_clean"] = m["sigma_clean"].to_numpy(bool)
    out["agn_clean"] = m["agn_clean"].to_numpy(bool)
    return out.reset_index(drop=True)


def build_lowz(scheme):
    """The low-mass enhancement slice (9.0<logM<9.5, 0.02<z<0.05) with the same
    gate. Kept separate from the main frame because its mass range does not
    overlap the main sample."""
    df = pd.read_parquet(config.MERGED)
    m = df[df["in_lowz_slice"].to_numpy(bool)
           & df[f"in_sample_{scheme}"].to_numpy(bool)].copy()
    out = pd.DataFrame(index=m.index)
    out["log_mass"] = pd.to_numeric(m["lgm_tot_p50"], errors="coerce")
    out["bulge_prominence"] = pd.to_numeric(m["bulge_prominence"], errors="coerce")
    out["specsfr_fib"] = pd.to_numeric(m["specsfr_fib_p50"], errors="coerce")
    out["sfr_fib"] = pd.to_numeric(m["sfr_fib_p50"], errors="coerce")
    out["bar_class"] = m[f"bar_class_{scheme}"].to_numpy()
    out["agn_clean"] = m["agn_clean"].to_numpy(bool)
    return out.reset_index(drop=True)


def bar_masks(bar_class, barmode):
    """Boolean (barred, unbarred) masks for a barmode over a bar_class array."""
    barred_lbls, unbarred_lbls = BARMODES[barmode]
    bc = np.asarray(bar_class)
    barred = np.isin(bc, barred_lbls)
    unbarred = np.isin(bc, unbarred_lbls)
    return barred, unbarred


# ===========================================================================
# Method A: RF feature importance (reported, not headlined)
# ===========================================================================

FEATURES_A = ["log_mass", "bulge_prominence", "bar_strength", "spiral", rf.RANDOM]


def rf_eligible(df, features):
    """Finite features + green valley removed (clearly-quenched vs clearly-SF),
    the same target definition validated in Step 04."""
    ok = np.ones(len(df), dtype=bool)
    for c in features:
        if c == rf.RANDOM:
            continue
        ok &= np.isfinite(pd.to_numeric(df[c], errors="coerce").to_numpy(float))
    ok &= ~df["green_valley"].to_numpy(bool)
    return df.loc[ok].reset_index(drop=True)


def method_a_one(name, df, features, rng):
    LOG("-" * 72)
    LOG(f"  Method A [{name}]")
    elig = rf_eligible(df, features)
    n_pos = int(elig["quenched_dsfr"].sum())
    LOG(f"    galaxies (green valley removed) . {len(elig):>9,}   "
        f"quenched {n_pos:,} ({elig['quenched_dsfr'].mean():.3f})")
    LOG(f"    features: {features}")
    leaf, _ = rf.tune_leaf(df=elig, features=features, target="quenched_dsfr", rng=rng)
    res = rf.run_rf_importance(elig, features, "quenched_dsfr",
                               config.N_RF_REPEATS, leaf, rng, log=LOG)
    LOG(f"    tuned leaf={leaf}   median test AUC={res['auc_median']:.3f} "
        f"[{res['auc_p5']:.3f},{res['auc_p95']:.3f}]")
    order = np.argsort(res["perm"]["median"])[::-1]
    LOG("    perm-importance ranking:")
    for j in order:
        f = res["features"][j]
        LOG(f"      {f:18s} perm={res['perm']['median'][j]:+.4f} "
            f"({res['perm']['p5'][j]:+.4f},{res['perm']['p95'][j]:+.4f})  "
            f"gini={res['gini']['median'][j]:.4f}")
    tbl = rf.importance_table(res)
    tbl.insert(0, "sample", name)
    # Where does the bar land, relative to the random-control noise floor?
    bj = res["features"].index("bar_strength")
    rj = res["features"].index(rf.RANDOM)
    bar_beats_rand = res["perm"]["p5"][bj] > res["perm"]["p95"][rj]
    LOG(f"    -> bar_strength above random-control floor: "
        f"{'YES' if bar_beats_rand else 'NO'} "
        f"(bar perm {res['perm']['median'][bj]:+.4f} vs "
        f"random {res['perm']['median'][rj]:+.4f})")
    return tbl, dict(sample=name, auc=res["auc_median"],
                     bar_perm=res["perm"]["median"][bj],
                     bar_beats_rand=bool(bar_beats_rand))


def _summary_from_table(tbl):
    """Rebuild the per-sample Method-A summary dicts from a written 05a CSV, so a
    completed (deterministic) RF result can be reused without recomputation."""
    out = []
    for sample, g in tbl.groupby("sample", sort=False):
        b = g[g["feature"] == "bar_strength"].iloc[0]
        r = g[g["feature"] == rf.RANDOM].iloc[0]
        out.append(dict(sample=sample, auc=float(b["auc_median"]),
                        bar_perm=float(b["perm_median"]),
                        bar_beats_rand=bool(b["perm_p5"] > r["perm_p95"])))
    return out


def method_a(frames, rng):
    LOG("=" * 72)
    LOG("METHOD A -- Random-Forest feature importance (report, do not headline)")
    LOG("=" * 72)
    summary = []
    for scheme_index, scheme in enumerate(SCHEMES):
        out = config.RESULTS / f"05a_rf_importance_{scheme}.csv"
        inputs = provenance.fingerprint(config.ROOT,
            [config.MERGED, config.ROOT / "config.py", Path(__file__),
             Path(rf.__file__), Path(provenance.__file__)],
            settings={"scheme": scheme, "seed": config.RANDOM_STATE + 1 + scheme_index})
        # Independent stream per scheme: partial cache hits cannot shift draws.
        scheme_rng = np.random.default_rng(config.RANDOM_STATE + 1 + scheme_index)
        if not FORCE_RECOMPUTE and provenance.cache_valid(out, inputs):
            tbl = pd.read_csv(out)
            LOG(f"  [reuse] {out.name} already present -> loaded "
                f"({tbl['sample'].nunique()} tiers, {len(tbl)} rows)")
            summary += _summary_from_table(tbl)
            continue
        df = frames[scheme]
        tables = []
        t, s = method_a_one(f"{scheme}:main", df, FEATURES_A, scheme_rng)
        tables.append(t); summary.append(s)
        # sigma tier: add sigma_c, restrict to sigma-clean with finite sigma_c
        sig = df[df["sigma_clean"].to_numpy(bool)].reset_index(drop=True)
        feats_sig = ["log_mass", "bulge_prominence", "bar_strength", "spiral",
                     "sigma_c", rf.RANDOM]
        t, s = method_a_one(f"{scheme}:sigma", sig, feats_sig, scheme_rng)
        tables.append(t); summary.append(s)
        out = config.RESULTS / f"05a_rf_importance_{scheme}.csv"
        # LF line endings on every platform keep the cached-output checksum
        # valid on a fresh clone.
        pd.concat(tables, ignore_index=True).to_csv(out, index=False,
                                                    lineterminator="\n")
        provenance.record_cache(out, inputs)
        LOG(f"    wrote {out.name}")
    return summary


# ===========================================================================
# Method B: controlled binning (HEADLINE)
# ===========================================================================

def quantile_edges(x, n_bins):
    """n_bins+1 quantile edges over finite x (equal-count bins). Nudge the top
    edge so the maximum lands in the last bin under a half-open digitize."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, q)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _summarize_cells(valid_rows):
    """Combine an already-filtered list of valid cells into a single summary.

    Reports fixed-effect and DerSimonian--Laird random-effects summaries,
    Cochran's Q/dof/I2 and descriptive sign counts. Under heterogeneity the
    summaries weight cells differently. Both errors are nominal and neither
    is a calibrated bound on total uncertainty; sign counts are not an
    independent significance test.
    Shared by the single-grid and z-sliced (pooled-cell) drivers so both report
    the identical statistics."""
    if valid_rows:
        d = np.array([r["delta_q"] for r in valid_rows])
        se = su.diff_se_from_interval(
            np.array([r["delta_q_lo"] for r in valid_rows]),
            np.array([r["delta_q_hi"] for r in valid_rows]))
        se = np.where(se > 0, se, np.nan)
        good = np.isfinite(se)
        wmean, wse = su.inverse_variance_weighted_mean(d[good], se[good])
        re = su.random_effects_mean(d[good], se[good])
        frac_pos = float(np.mean(d > 0))
        n_sig_pos = int(np.sum([r["delta_q_lo"] > 0 for r in valid_rows]))
        n_sig_neg = int(np.sum([r["delta_q_hi"] < 0 for r in valid_rows]))
        cell_sd = float(np.std(d, ddof=1)) if d.size > 1 else np.nan
        re_mean, re_se = re["mean"], re["se"]
        Q, dof, I2, tau2, pQ = (re["Q"], re["dof"], re["I2"], re["tau2"],
                                re["p_value"])
    else:
        wmean = wse = frac_pos = np.nan
        re_mean = re_se = cell_sd = Q = I2 = tau2 = pQ = np.nan
        dof = 0
        n_sig_pos = n_sig_neg = 0
    return dict(n_valid_cells=len(valid_rows), wmean_delta_q=wmean,
                wmean_delta_q_se=wse, frac_cells_pos=frac_pos,
                n_cells_sig_pos=n_sig_pos, n_cells_sig_neg=n_sig_neg,
                re_mean_delta_q=re_mean, re_mean_delta_q_se=re_se,
                cell_sd_delta_q=cell_sd, cochran_Q=Q, cochran_dof=dof,
                cochran_I2=I2, tau2=tau2, cochran_p=pQ)


def controlled_binning(df, mass_edges, bulge_edges, barmode,
                       quench_col="quenched_dsfr"):
    """Per-cell q_barred, q_unbarred, Delta q with Wilson/Newcombe intervals over the
    (mass, bulge) grid. A cell is valid iff it holds >= MIN_PER_CLASS_PER_CELL of
    BOTH classes. `quench_col` selects the quenched-flag column (fiducial
    delta_sfr, or a flat-sSFR variant for the robustness check). Returns
    (rows, summary)."""
    mass = df["log_mass"].to_numpy(float)
    bulge = df["bulge_prominence"].to_numpy(float)
    q = df[quench_col].to_numpy(bool)
    barred, unbarred = bar_masks(df["bar_class"].to_numpy(), barmode)

    mi = np.digitize(mass, mass_edges) - 1
    bi = np.digitize(bulge, bulge_edges) - 1
    n_m = len(mass_edges) - 1
    n_b = len(bulge_edges) - 1

    rows = []
    for i in range(n_m):
        for j in range(n_b):
            cell = (mi == i) & (bi == j)
            cb = cell & barred
            cu = cell & unbarred
            nb, nu = int(cb.sum()), int(cu.sum())
            valid = (nb >= config.MIN_PER_CLASS_PER_CELL
                     and nu >= config.MIN_PER_CLASS_PER_CELL)
            kb, ku = int(q[cb].sum()), int(q[cu].sum())
            row = dict(
                mass_bin=i, bulge_bin=j,
                mass_lo=mass_edges[i], mass_hi=mass_edges[i + 1],
                bulge_lo=bulge_edges[j], bulge_hi=bulge_edges[j + 1],
                n_barred=nb, n_unbarred=nu,
                k_barred=kb, k_unbarred=ku, valid=valid,
            )
            if valid:
                qb, ql, qh = kb / nb, *su.wilson_interval(kb, nb)
                qu, ul, uh = ku / nu, *su.wilson_interval(ku, nu)
                diff, dlo, dhi = su.newcombe_diff_interval(kb, nb, ku, nu)
                row.update(q_barred=qb, q_barred_lo=ql, q_barred_hi=qh,
                           q_unbarred=qu, q_unbarred_lo=ul, q_unbarred_hi=uh,
                           delta_q=diff, delta_q_lo=dlo, delta_q_hi=dhi)
            else:
                for k in ("q_barred", "q_barred_lo", "q_barred_hi",
                          "q_unbarred", "q_unbarred_lo", "q_unbarred_hi",
                          "delta_q", "delta_q_lo", "delta_q_hi"):
                    row[k] = np.nan
            rows.append(row)

    valid_rows = [r for r in rows if r["valid"]]
    return rows, _summarize_cells(valid_rows)


def controlled_binning_zsliced(base, sample, barmode, z_slices,
                               quench_col="quenched_dsfr"):
    """Redshift-controlled Method B. Within each z slice, recompute the
    (mass, bulge) quantile grid on the FULL in-sample population of that slice
    (edges from `base`, so the `all` and `agn_clean` runs share cells), bin the
    target `sample`, then pool every valid cell across all slices and
    inverse-variance-weight them. Each surviving cell fixes mass, bulge AND z.

    Returns (rows, summary); rows carry z_lo / z_hi for provenance.
    """
    zb = base["z"].to_numpy(float)
    zs = sample["z"].to_numpy(float)
    all_rows = []
    for zlo, zhi in z_slices:
        base_slice = base[(zb >= zlo) & (zb < zhi)]
        samp_slice = sample[(zs >= zlo) & (zs < zhi)]
        me = quantile_edges(base_slice["log_mass"], config.N_MASS_BINS)
        be = quantile_edges(base_slice["bulge_prominence"], config.N_BULGE_BINS)
        rows, _ = controlled_binning(samp_slice, me, be, barmode, quench_col)
        for r in rows:
            r["z_lo"], r["z_hi"] = zlo, zhi
        all_rows.extend(rows)
    valid_rows = [r for r in all_rows if r["valid"]]
    return all_rows, _summarize_cells(valid_rows)


def method_b(frames, rng):
    LOG("=" * 72)
    LOG("METHOD B -- controlled (mass, bulge) binning  [HEADLINE]")
    LOG(f"  grid {config.N_MASS_BINS} x {config.N_BULGE_BINS} quantile bins; "
        f"cell valid iff >= {config.MIN_PER_CLASS_PER_CELL} barred AND unbarred")
    LOG("=" * 72)
    headline = []
    for scheme in SCHEMES:
        base = frames[scheme]
        # One grid per scheme, fixed on the full in-sample population so that the
        # full and AGN-clean runs are compared on identical cells.
        mass_edges = quantile_edges(base["log_mass"], config.N_MASS_BINS)
        bulge_edges = quantile_edges(base["bulge_prominence"], config.N_BULGE_BINS)
        LOG(f"  [{scheme}] mass edges  = "
            + ", ".join(f"{e:.2f}" for e in mass_edges))
        LOG(f"  [{scheme}] bulge edges = "
            + ", ".join(f"{e:.2f}" for e in bulge_edges))
        for barmode in ("all", "strong_only", "weak_only"):
            all_rows = []
            z_rows = []
            for sample_name, sub in (("all", base),
                                     ("agn_clean",
                                      base[base["agn_clean"].to_numpy(bool)])):
                # (i) DIAGNOSTIC: single (mass, bulge) grid, no z control. Kept so
                # the redshift inflation is visible, but NOT the headline.
                rows, summ = controlled_binning(sub, mass_edges, bulge_edges,
                                                barmode)
                for r in rows:
                    r["sample"] = sample_name
                    r["scheme"] = scheme
                    r["barmode"] = barmode
                all_rows.extend(rows)
                LOG(f"    {scheme:12s} {barmode:11s} {sample_name:9s} [no-z] : "
                    f"{summ['n_valid_cells']:>2d} valid cells  "
                    f"<Delta q>_FE={summ['wmean_delta_q']:+.4f}"
                    f"+/-{summ['wmean_delta_q_se']:.4f}  "
                    f"<Delta q>_RE={summ['re_mean_delta_q']:+.4f}"
                    f"+/-{summ['re_mean_delta_q_se']:.4f}  "
                    f"Q/dof={summ['cochran_Q']:.0f}/{summ['cochran_dof']} "
                    f"I2={summ['cochran_I2']:.2f}  "
                    f"frac>0={summ['frac_cells_pos']:.2f}  "
                    f"sig +{summ['n_cells_sig_pos']}/-{summ['n_cells_sig_neg']}")
                headline.append(dict(scheme=scheme, barmode=barmode,
                                     sample=sample_name, zcontrol=False, **summ))

                # (ii) FIDUCIAL: (mass, bulge) grid recomputed inside each z slice,
                # cells pooled and inverse-variance-weighted -> mass, bulge AND z
                # held fixed.
                zrows, zsumm = controlled_binning_zsliced(
                    base, sub, barmode, config.Z_SLICES)
                for r in zrows:
                    r["sample"] = sample_name
                    r["scheme"] = scheme
                    r["barmode"] = barmode
                z_rows.extend(zrows)
                LOG(f"    {scheme:12s} {barmode:11s} {sample_name:9s} [z-ctl]: "
                    f"{zsumm['n_valid_cells']:>2d} valid cells  "
                    f"<Delta q>_FE={zsumm['wmean_delta_q']:+.4f}"
                    f"+/-{zsumm['wmean_delta_q_se']:.4f}  "
                    f"<Delta q>_RE={zsumm['re_mean_delta_q']:+.4f}"
                    f"+/-{zsumm['re_mean_delta_q_se']:.4f}  "
                    f"Q/dof={zsumm['cochran_Q']:.0f}/{zsumm['cochran_dof']} "
                    f"I2={zsumm['cochran_I2']:.2f}  "
                    f"frac>0={zsumm['frac_cells_pos']:.2f}  "
                    f"sig +{zsumm['n_cells_sig_pos']}/-{zsumm['n_cells_sig_neg']}")
                headline.append(dict(scheme=scheme, barmode=barmode,
                                     sample=sample_name, zcontrol=True, **zsumm))
            out = config.RESULTS / f"05b_controlled_binning_{scheme}_{barmode}.csv"
            pd.DataFrame(all_rows).to_csv(out, index=False)
            zout = (config.RESULTS
                    / f"05b_controlled_binning_zsliced_{scheme}_{barmode}.csv")
            pd.DataFrame(z_rows).to_csv(zout, index=False)
        LOG(f"    wrote 05b_controlled_binning_{scheme}_*.csv "
            f"(+ zsliced variants)")
    return headline


def method_b_quench_robustness(frames):
    """Does the redshift-controlled headline survive a change in the quenched
    DEFINITION? Rerun the fiducial z-controlled Method B (comparison scheme, full
    sample) under the fiducial Delta SFR cut and three flat log-sSFR cuts
    (-10.5, -11.0, -11.5). Reported as a robustness table, not a new headline."""
    LOG("=" * 72)
    LOG("METHOD B robustness -- quenched-DEFINITION stability (z-controlled, "
        "comparison, full sample)")
    LOG("=" * 72)
    base = frames["comparison"]
    quench_defs = ([("delta_sfr (fiducial)", "quenched_dsfr")]
                   + [(f"ssfr<{cut}", f"quenched_ssfr_{cut}")
                      for cut in config.SSFR_QUENCHED_VARIANTS])
    records = []
    for barmode in ("all", "strong_only"):
        for label, col in quench_defs:
            _, summ = controlled_binning_zsliced(
                base, base, barmode, config.Z_SLICES, quench_col=col)
            records.append(dict(barmode=barmode, quench_def=label, quench_col=col,
                                **summ))
            LOG(f"  {barmode:11s} {label:22s}: "
                f"FE <Delta q>={summ['wmean_delta_q']:+.4f}"
                f"+/-{summ['wmean_delta_q_se']:.4f}  "
                f"RE <Delta q>={summ['re_mean_delta_q']:+.4f}"
                f"+/-{summ['re_mean_delta_q_se']:.4f}  "
                f"({summ['n_valid_cells']} cells, "
                f"frac>0={summ['frac_cells_pos']:.2f}, "
                f"sig +{summ['n_cells_sig_pos']}/-{summ['n_cells_sig_neg']})")
    out = config.RESULTS / "05b_quench_robustness.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    LOG(f"    wrote {out.name}")
    return records


# ===========================================================================
# Method C: rank partial correlation
# ===========================================================================

# Ordered control sets, keyed by a short label used in the figure story.
CONTROL_SETS = [
    ("none",            []),
    ("mass",            ["log_mass"]),
    ("mass+bulge",      ["log_mass", "bulge_prominence"]),
    ("mass+bulge+z", ["log_mass", "bulge_prominence", "z"]),
    ("mass+bulge+sigma+z", ["log_mass", "bulge_prominence", "sigma_c", "z"]),
    ("mass+bulge+sigma", ["log_mass", "bulge_prominence", "sigma_c"]),
]


def _pcc_all_levels(rank_corr, cols):
    """Given a rank-correlation matrix over `cols` (0=bar_strength, 1=delta_sfr,
    then covariates), return the partial corr of (0,1) for each control set that
    is fully available in `cols`."""
    idx = {c: k for k, c in enumerate(cols)}
    out = {}
    for label, covars in CONTROL_SETS:
        if all(c in idx for c in covars):
            controls = [idx[c] for c in covars]
            out[label] = su.partial_corr_from_matrix(rank_corr, 0, 1, controls)
    return out


def _partial_corr_bootstrap(df, cols, rng, n_boot):
    """Point estimate + 16/84 band for every control level, ranking once per
    resample so all levels share the same ranks (fast and internally consistent)."""
    sub = df[cols].dropna().reset_index(drop=True)
    n = len(sub)
    ranks0 = sub.rank().to_numpy()
    point = _pcc_all_levels(np.corrcoef(ranks0, rowvar=False), cols)
    boot = {lab: np.empty(n_boot) for lab in point}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rk = sub.iloc[idx].rank().to_numpy()
        vals = _pcc_all_levels(np.corrcoef(rk, rowvar=False), cols)
        for lab in point:
            boot[lab][b] = vals[lab]
    rows = []
    for lab in point:
        lo, hi = np.percentile(boot[lab], [16.0, 84.0])
        rows.append(dict(controls=lab, pcc=point[lab], lo=lo, hi=hi, n=n))
    return rows


def method_c(frames, rng):
    LOG("=" * 72)
    LOG("METHOD C -- rank partial correlation  corr(bar_strength, Delta SFR)")
    LOG("  response = CONTINUOUS delta_sfr; controls added cumulatively.")
    LOG("=" * 72)
    records = []
    for scheme in SCHEMES:
        df0 = frames[scheme]
        # The agn_clean rerun keeps only BPT star-forming galaxies (bptclass 1:
        # SF, 2: low-S/N SF; MPA-JHU galSpecExtra convention). It is a
        # population selection, not an AGN-mechanism test: the removed classes
        # (unclassifiable, composite, AGN, low-S/N LINER) contain most of the
        # quenched discs, and they are removed unequally from the bar classes
        # (see Method E). A change in the bar->low-SFR signal on this tier is
        # therefore not evidence for or against AGN-mediated quenching.
        for sample_name, df in (("all", df0),
                                ("agn_clean",
                                 df0[df0["agn_clean"].to_numpy(bool)].reset_index(drop=True))):
            for subset in ("strong+weak", "strong_only"):
                if subset == "strong+weak":
                    d = df
                else:
                    keep = np.isin(df["bar_class"].to_numpy(), ("unbarred", "strong"))
                    d = df[keep].reset_index(drop=True)
                # Second-order (mass, bulge) on the frame; the sigma level is only
                # meaningful on the sigma-clean tier, computed separately so its
                # sample is stated explicitly.
                cols2 = ["bar_strength", "delta_sfr", "log_mass", "bulge_prominence", "z"]
                rows = _partial_corr_bootstrap(d, cols2, rng, N_BOOT)
                for r in rows:
                    r["tier"] = "main"
                dsig = d[d["sigma_clean"].to_numpy(bool)].reset_index(drop=True)
                cols3 = cols2 + ["sigma_c"]
                rows_sig = _partial_corr_bootstrap(dsig, cols3, rng, N_BOOT)
                # every control level is kept on the identical sigma-tier sample
                for r in rows_sig:
                    r["tier"] = "sigma"
                rows += rows_sig  # retain all levels on this identical sample
                LOG(f"  [{scheme} / {sample_name} / {subset}]  n={len(d):,} "
                    f"(sigma tier {len(dsig):,})")
                for r in rows:
                    r["scheme"] = scheme
                    r["sample"] = sample_name
                    r["subset"] = subset
                    records.append(r)
                    LOG(f"      controls={r['controls']:16s} "
                        f"pcc={r['pcc']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    out = config.RESULTS / "05c_partial_correlation.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    LOG(f"    wrote {out.name}")
    return records


# ===========================================================================
# Method D: fibre-sSFR comparison with Liu & Zhou
# ===========================================================================

def _boot_median(x, rng, n_boot):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    bs = np.empty(n_boot)
    for b in range(n_boot):
        bs[b] = np.median(x[rng.integers(0, x.size, size=x.size)])
    lo, hi = np.percentile(bs, [16.0, 84.0])
    return med, float(lo), float(hi)


def _median_diff(a, b, rng, n_boot):
    """median(a) - median(b) with a 16/84 bootstrap band on the difference."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan
    d = float(np.median(a) - np.median(b))
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = (np.median(a[rng.integers(0, a.size, size=a.size)])
                 - np.median(b[rng.integers(0, b.size, size=b.size)]))
    lo, hi = np.percentile(bs, [16.0, 84.0])
    return d, float(lo), float(hi)


def _mean_diff(a, b, rng, n_boot):
    """mean(a) - mean(b) with a 16/84 bootstrap band on the difference.

    Liu & Zhou quote MEAN fibre sSFR (their Fig. 6 uses standard errors of the
    mean), so any same-statistic comparison to their numbers must use this
    statistic, not the median."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan
    d = float(np.mean(a) - np.mean(b))
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = (np.mean(a[rng.integers(0, a.size, size=a.size)])
                 - np.mean(b[rng.integers(0, b.size, size=b.size)]))
    lo, hi = np.percentile(bs, [16.0, 84.0])
    return d, float(lo), float(hi)


def _controlled_median_diff(d, n_mass_sub, n_bulge, rng, n_boot):
    """Mean over (mass, bulge) sub-cells of the strong-minus-unbarred median-sSFR
    difference, WITHIN whatever mass range `d` already spans, with a paired
    bootstrap 16/84 band. Cells need >= MIN_PER_CLASS_PER_CELL of each class.

    Returns (point, lo, hi, n_cells, n_strong, n_unbarred).
    """
    m = d["log_mass"].to_numpy(float)
    bl = d["bulge_prominence"].to_numpy(float)
    ssfr = d["specsfr_fib"].to_numpy(float)
    strong, unb = bar_masks(d["bar_class"].to_numpy(), "strong_only")
    me = quantile_edges(m, n_mass_sub)
    be = quantile_edges(bl, n_bulge)
    mi = np.digitize(m, me) - 1
    bi = np.digitize(bl, be) - 1
    cells = []
    for i in range(n_mass_sub):
        for j in range(n_bulge):
            sel = (mi == i) & (bi == j)
            cs = ssfr[sel & strong]; cs = cs[np.isfinite(cs)]
            cu = ssfr[sel & unb]; cu = cu[np.isfinite(cu)]
            if cs.size >= config.MIN_PER_CLASS_PER_CELL and \
               cu.size >= config.MIN_PER_CLASS_PER_CELL:
                cells.append((cs, cu))
    if not cells:
        return np.nan, np.nan, np.nan, 0, 0, 0
    stat = lambda cl: float(np.mean([np.median(cs) - np.median(cu)
                                     for cs, cu in cl]))
    point = stat(cells)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        bs[b] = stat([(cs[rng.integers(0, cs.size, cs.size)],
                       cu[rng.integers(0, cu.size, cu.size)]) for cs, cu in cells])
    lo, hi = np.percentile(bs, [16.0, 84.0])
    ns = int(sum(cs.size for cs, cu in cells))
    nu = int(sum(cu.size for cs, cu in cells))
    return point, float(lo), float(hi), len(cells), ns, nu


def method_d(frames, lowz_frames, rng):
    LOG("=" * 72)
    LOG("METHOD D -- fibre-sSFR comparison with Liu & Zhou (strong-barred vs unbarred)")
    LOG("=" * 72)
    records = []

    # ---- Suppression panel: main sample, 9.5 < logM < 10.5 ----
    for scheme in SCHEMES:
        base = frames[scheme]
        for sample_name, src in (("all", base),
                                 ("agn_clean",
                                  base[base["agn_clean"].to_numpy(bool)])):
            d = src[(src["log_mass"] > 9.5) & (src["log_mass"] < 10.5)]
            strong, unb = bar_masks(d["bar_class"].to_numpy(), "strong_only")
            sb = d["specsfr_fib"].to_numpy()[strong]
            ub = d["specsfr_fib"].to_numpy()[unb]

            # (a) fixed-mass-only: single 9.5-10.5 band (direct L&Z-style contrast)
            diff, lo, hi = _median_diff(sb, ub, rng, N_BOOT)
            ms, mslo, mshi = _boot_median(sb, rng, N_BOOT)
            mu, mulo, muhi = _boot_median(ub, rng, N_BOOT)
            records.append(dict(panel="suppression", control="mass_only",
                                scheme=scheme, sample=sample_name,
                                n_strong=int(strong.sum()), n_unbarred=int(unb.sum()),
                                med_strong=ms, med_unbarred=mu,
                                delta=diff, delta_lo=lo, delta_hi=hi))
            LOG(f"  [suppression/{scheme}/{sample_name}] mass-only "
                f"(n_s={int(strong.sum())}, n_u={int(unb.sum())}): "
                f"d(median sSFR_fib)={diff:+.4f} [{lo:+.4f},{hi:+.4f}]")

            # (b) controlled WITHIN the panel's 9.5-10.5 band: sub-bin on mass and
            # bulge (so both confounds are held fixed at the L&Z mass regime), and
            # bootstrap the mean within-cell median difference.
            pt, lo2, hi2, ncell, ns, nu = _controlled_median_diff(
                d, n_mass_sub=3, n_bulge=config.N_BULGE_BINS, rng=rng, n_boot=N_BOOT)
            if ncell:
                records.append(dict(panel="suppression", control="mass+bulge_cells",
                                    scheme=scheme, sample=sample_name,
                                    n_strong=ns, n_unbarred=nu,
                                    med_strong=np.nan, med_unbarred=np.nan,
                                    delta=pt, delta_lo=lo2, delta_hi=hi2,
                                    n_cells=ncell))
                LOG(f"  [suppression/{scheme}/{sample_name}] mass+bulge cells "
                    f"({ncell} cells, band 9.5-10.5): within-cell "
                    f"d(median sSFR_fib)={pt:+.4f} [{lo2:+.4f},{hi2:+.4f}]")
            else:
                LOG(f"  [suppression/{scheme}/{sample_name}] mass+bulge cells: "
                    f"no cell with >=30 strong AND unbarred in band")

            # (c) REDSHIFT-SLICED within the 9.5-10.5 band. The 3" fibre samples a
            # different physical aperture at every z, so the full-band average of
            # (b) mixes physically different regimes. The z<0.05
            # slice overlaps Liu & Zhou's redshift regime. NOTE these rows are
            # MEDIAN-based; L&Z quote MEANS with mass+colour-matched controls, so
            # a sign difference against their -0.2 dex here is not a like-for-like
            # comparison; see block (e) for the mean contrast and the
            # within-sample quenched/non-quenched decomposition.
            zz = d["z"].to_numpy(float)
            d_zslices = ([(config.LOWZ_SLICE["z_min"], config.LOWZ_SLICE["z_max"],
                           "LZ_z<0.05")]
                         + [(lo, hi, f"z{lo:g}-{hi:g}") for lo, hi in config.Z_SLICES])
            for zlo, zhi, tag in d_zslices:
                zsel = (zz >= zlo) & (zz < zhi)
                sbz = d["specsfr_fib"].to_numpy()[strong & zsel]
                ubz = d["specsfr_fib"].to_numpy()[unb & zsel]
                dz, lz_, hz = _median_diff(sbz, ubz, rng, N_BOOT)
                records.append(dict(panel="suppression",
                                    control=f"mass_only_zslice:{tag}",
                                    scheme=scheme, sample=sample_name,
                                    z_lo=zlo, z_hi=zhi,
                                    n_strong=int((strong & zsel).sum()),
                                    n_unbarred=int((unb & zsel).sum()),
                                    med_strong=np.nan, med_unbarred=np.nan,
                                    delta=dz, delta_lo=lz_, delta_hi=hz))
                LOG(f"  [suppression/{scheme}/{sample_name}] z-slice {tag:10s} "
                    f"(n_s={int((strong & zsel).sum())}, "
                    f"n_u={int((unb & zsel).sum())}): "
                    f"d(median sSFR_fib)={dz:+.4f} [{lz_:+.4f},{hz:+.4f}]")

            # (d) SAME-STATISTIC mass+bulge control at z<0.05 (both confounds held
            # fixed inside the L&Z redshift regime).
            dlz = d[(zz >= config.LOWZ_SLICE["z_min"])
                    & (zz < config.LOWZ_SLICE["z_max"])]
            pt, lo3, hi3, ncell3, ns3, nu3 = _controlled_median_diff(
                dlz, n_mass_sub=2, n_bulge=3, rng=rng, n_boot=N_BOOT)
            if ncell3:
                records.append(dict(panel="suppression",
                                    control="mass+bulge_cells_LZ_z<0.05",
                                    scheme=scheme, sample=sample_name,
                                    z_lo=config.LOWZ_SLICE["z_min"],
                                    z_hi=config.LOWZ_SLICE["z_max"],
                                    n_strong=ns3, n_unbarred=nu3,
                                    med_strong=np.nan, med_unbarred=np.nan,
                                    delta=pt, delta_lo=lo3, delta_hi=hi3,
                                    n_cells=ncell3))
                LOG(f"  [suppression/{scheme}/{sample_name}] mass+bulge cells "
                    f"@ z<0.05 ({ncell3} cells): within-cell "
                    f"d(median sSFR_fib)={pt:+.4f} [{lo3:+.4f},{hi3:+.4f}]")
            else:
                LOG(f"  [suppression/{scheme}/{sample_name}] mass+bulge cells "
                    f"@ z<0.05: too thin for a >=30/class cell")

            # (e) MEAN contrast + composition decomposition @ z<0.05.
            # L&Z quote MEAN fibre sSFR against mass+colour-matched controls;
            # the rows above are medians. The 9.5-10.5 band also mixes two
            # opposite-sign mass regimes and two populations (star-forming vs
            # quenched), so mean and median contrasts can differ in sign.
            # Report (i) the band contrast with the mean statistic, (ii) the
            # mass split under both statistics, and (iii) the quenched /
            # non-quenched split. This describes our sample; it does not
            # identify the cause of the difference from L&Z, whose matching,
            # apertures and SFR estimators differ.
            lzsel = ((zz >= config.LOWZ_SLICE["z_min"])
                     & (zz < config.LOWZ_SLICE["z_max"]))
            fib = d["specsfr_fib"].to_numpy(float)
            qfl = d["quenched_dsfr"].to_numpy(bool)
            mvals = d["log_mass"].to_numpy(float)
            dm, dmlo, dmhi = _mean_diff(fib[strong & lzsel], fib[unb & lzsel],
                                        rng, N_BOOT)
            records.append(dict(panel="suppression",
                                control="lz_band_mean_z<0.05", stat="mean",
                                scheme=scheme, sample=sample_name,
                                z_lo=config.LOWZ_SLICE["z_min"],
                                z_hi=config.LOWZ_SLICE["z_max"],
                                n_strong=int((strong & lzsel).sum()),
                                n_unbarred=int((unb & lzsel).sum()),
                                med_strong=np.nan, med_unbarred=np.nan,
                                delta=dm, delta_lo=dmlo, delta_hi=dmhi))
            LOG(f"  [suppression/{scheme}/{sample_name}] band 9.5-10.5 @ z<0.05, "
                f"L&Z statistic (MEAN): d={dm:+.4f} [{dmlo:+.4f},{dmhi:+.4f}]")
            for mlo, mhi in ((9.5, 10.0), (10.0, 10.5)):
                msel = lzsel & (mvals >= mlo) & (mvals < mhi)
                a_ = fib[strong & msel]
                b_ = fib[unb & msel]
                for stat_name, fn in (("median", _median_diff),
                                      ("mean", _mean_diff)):
                    dd, dlo, dhi = fn(a_, b_, rng, N_BOOT)
                    records.append(dict(panel="suppression",
                                        control=f"mass_split:{mlo}-{mhi}_z<0.05",
                                        stat=stat_name,
                                        scheme=scheme, sample=sample_name,
                                        z_lo=config.LOWZ_SLICE["z_min"],
                                        z_hi=config.LOWZ_SLICE["z_max"],
                                        n_strong=int((strong & msel).sum()),
                                        n_unbarred=int((unb & msel).sum()),
                                        med_strong=np.nan, med_unbarred=np.nan,
                                        delta=dd, delta_lo=dlo, delta_hi=dhi))
                    LOG(f"      mass {mlo}-{mhi} @ z<0.05 {stat_name:6s}: "
                        f"d={dd:+.4f} [{dlo:+.4f},{dhi:+.4f}]  "
                        f"(n_s={int((strong & msel).sum())}, "
                        f"n_u={int((unb & msel).sum())})")
            # EXHAUSTIVE two-way split: quenched vs NOT quenched. Under the
            # three-class scheme of Section 2.6 the non-quenched group also
            # contains the green valley, so it is NOT the strictly star-forming
            # class. The composition argument needs this partition because only
            # an exhaustive split reconstructs the band mean; script 09 reports
            # the green-valley share and the strictly-SF contrast alongside.
            for tag, popsel in (("nonquenched", ~qfl), ("Q_only", qfl)):
                a_ = fib[strong & lzsel & popsel]
                b_ = fib[unb & lzsel & popsel]
                dd, dlo, dhi = _median_diff(a_, b_, rng, N_BOOT)
                records.append(dict(panel="suppression",
                                    control=f"decomp_{tag}_z<0.05", stat="median",
                                    scheme=scheme, sample=sample_name,
                                    z_lo=config.LOWZ_SLICE["z_min"],
                                    z_hi=config.LOWZ_SLICE["z_max"],
                                    n_strong=int((strong & lzsel & popsel).sum()),
                                    n_unbarred=int((unb & lzsel & popsel).sum()),
                                    med_strong=np.nan, med_unbarred=np.nan,
                                    delta=dd, delta_lo=dlo, delta_hi=dhi))
                LOG(f"      decomposition {tag:8s} @ z<0.05 (median): "
                    f"d={dd:+.4f} [{dlo:+.4f},{dhi:+.4f}]  "
                    f"(n_s={int((strong & lzsel & popsel).sum())}, "
                    f"n_u={int((unb & lzsel & popsel).sum())})")
            ns_band = int((strong & lzsel).sum())
            nu_band = int((unb & lzsel).sum())
            qs = float(qfl[strong & lzsel].mean())
            qu = float(qfl[unb & lzsel].mean())
            dq_b, dq_blo, dq_bhi = su.newcombe_diff_interval(
                int(qfl[strong & lzsel].sum()), ns_band,
                int(qfl[unb & lzsel].sum()), nu_band)
            records.append(dict(panel="suppression",
                                control="qfrac_inband_z<0.05", stat="fraction",
                                scheme=scheme, sample=sample_name,
                                z_lo=config.LOWZ_SLICE["z_min"],
                                z_hi=config.LOWZ_SLICE["z_max"],
                                n_strong=ns_band, n_unbarred=nu_band,
                                med_strong=qs, med_unbarred=qu,
                                delta=dq_b, delta_lo=dq_blo, delta_hi=dq_bhi))
            LOG(f"      quenched fraction in band @ z<0.05: strong={qs:.3f}  "
                f"unbarred={qu:.3f}  (the composition driving mean-vs-median)")

    # ---- Enhancement panel: low-z slice, 9.0 < logM < 9.5 ----
    for scheme in SCHEMES:
        lz = lowz_frames[scheme]
        for sample_name, src in (("all", lz),
                                 ("agn_clean",
                                  lz[lz["agn_clean"].to_numpy(bool)])):
            strong, unb = bar_masks(src["bar_class"].to_numpy(), "strong_only")
            for col, tag in (("specsfr_fib", "ssfr_fib"), ("sfr_fib", "sfr_fib")):
                sb = src[col].to_numpy()[strong]
                ub = src[col].to_numpy()[unb]
                diff, lo, hi = _median_diff(sb, ub, rng, N_BOOT)
                ms, _, _ = _boot_median(sb, rng, N_BOOT)
                mu, _, _ = _boot_median(ub, rng, N_BOOT)
                records.append(dict(panel="enhancement", control=f"mass_only:{tag}",
                                    scheme=scheme, sample=sample_name,
                                    n_strong=int(strong.sum()),
                                    n_unbarred=int(unb.sum()),
                                    med_strong=ms, med_unbarred=mu,
                                    delta=diff, delta_lo=lo, delta_hi=hi))
                LOG(f"  [enhancement/{scheme}/{sample_name}] {tag} "
                    f"(n_s={int(strong.sum())}, n_u={int(unb.sum())}): "
                    f"d(median)={diff:+.4f} [{lo:+.4f},{hi:+.4f}]")
            # bulge medians per group (cells are too thin for a 2D control here)
            bs_med = np.nanmedian(src["bulge_prominence"].to_numpy()[strong])
            bu_med = np.nanmedian(src["bulge_prominence"].to_numpy()[unb])
            LOG(f"      bulge_prominence median  strong={bs_med:.3f}  "
                f"unbarred={bu_med:.3f}  (2D control not possible at this N)")
            records.append(dict(panel="enhancement", control="bulge_median",
                                scheme=scheme, sample=sample_name,
                                n_strong=int(strong.sum()), n_unbarred=int(unb.sum()),
                                med_strong=bs_med, med_unbarred=bu_med,
                                delta=bs_med - bu_med, delta_lo=np.nan, delta_hi=np.nan))

    out = config.RESULTS / "05d_fibre_ssfr.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    LOG(f"    wrote {out.name}")
    return records


# ===========================================================================
# Method E: BPT composition per bar class and within-class contrasts
# ===========================================================================

def method_e(frames):
    """Composition of each bar class by MPA-JHU BPT class, computed so the
    summary numbers are traceable and the agn_clean selection is described explicitly.

    galSpecExtra BPTCLASS convention (official SDSS data model):
      -1 = unclassifiable (lines too weak/absent), 1 = star-forming,
       2 = low-S/N star-forming, 3 = composite, 4 = AGN excluding LINERs
       (i.e. Seyfert-like), 5 = low-S/N LINER. A value 0 also occurs in the
       catalogue (undocumented; treated as unclassifiable alongside -1).

    agn_clean keeps ONLY BPT star-forming galaxies (bptclass 1,2); it removes
    the unclassifiable/no-line systems (-1, 0), composites (3), AGN (4) and
    low-S/N LINERs (5), which contain most (~88 per cent) of the quenched
    comparison discs. A change in Delta q or the bar correlation on the
    agn_clean tier therefore reflects a changed population; it is not evidence
    for or against AGN-mediated quenching. For the fiducial comparison
    scheme we report, per bar class:
      * frac_removed_agn_clean : share with bptclass not in {1,2}
      * frac_composite         : share with bptclass == 3 (SF/AGN composite)
      * frac_agn               : share with bptclass == 4 (AGN excl. LINERs)
      * frac_liner_lowsn       : share with bptclass == 5 (low-S/N LINER)
      * frac_noline            : share with bptclass in {-1, 0} (no usable lines)
      * qfrac                  : quenched fraction
      * med_ssfr_fib_minus_tot : central-vs-global SF concentration
    Column semantics for the derived `*:mass_controlled` rows: `n` = number of
    contributing mass bins, `qfrac` = the mass-controlled <Delta qfrac>, and
    `qfrac_se` = its standard error (NaN on all other rows).
    plus a within-BPT AGN contrast: quenched fraction among BPT AGN (bptclass==4)
    only, strong vs unbarred, uncontrolled AND mass-controlled. Conditioning on
    current emission-line class does not measure past AGN activity, so this
    contrast neither establishes nor excludes AGN mediation. The same
    comparison on composites (bptclass==3) is reported alongside.
    """
    LOG("=" * 72)
    LOG("METHOD E -- BPT composition per bar class and within-class contrasts")
    LOG("=" * 72)
    df = frames["comparison"]
    bpt = df["bptclass"].to_numpy(float)
    bc = df["bar_class"].to_numpy()
    q = df["quenched_dsfr"].to_numpy(bool)
    mass = df["log_mass"].to_numpy(float)
    fib_m_tot = (df["specsfr_fib"].to_numpy(float)
                 - df["specsfr_tot"].to_numpy(float))
    records = []
    for cls in ("strong", "weak", "unbarred"):
        sel = bc == cls
        n = int(sel.sum())
        removed = float(np.mean(~np.isin(bpt[sel], (1, 2))))
        f_comp = float(np.mean(bpt[sel] == 3))
        f_agn = float(np.mean(bpt[sel] == 4))
        f_liner = float(np.mean(bpt[sel] == 5))
        f_noline = float(np.mean(np.isin(bpt[sel], (-1, 0))))
        qfrac = float(q[sel].mean())
        concentration = float(np.nanmedian(fib_m_tot[sel]))
        records.append(dict(bar_class=cls, n=n, qfrac=qfrac, qfrac_se=np.nan,
                            frac_removed_agn_clean=removed,
                            frac_composite=f_comp,
                            frac_agn=f_agn,
                            frac_liner_lowsn=f_liner,
                            frac_noline=f_noline,
                            med_ssfr_fib_minus_tot=concentration))
        LOG(f"  {cls:9s} n={n:6d}  qfrac={qfrac:.3f}  "
            f"removed(agn_clean)={removed:.3f}  composite={f_comp:.3f}  "
            f"AGN={f_agn:.3f}  low-S/N LINER={f_liner:.3f}  "
            f"no-line={f_noline:.3f}  med(sSFR_fib-tot)={concentration:+.4f}")

    def _population_test(tag, label, popsel):
        """Quenched-fraction contrast strong vs unbarred inside one BPT
        population, uncontrolled and mass-controlled (the two
        classes have different mass distributions inside any BPT population,
        so the uncontrolled difference alone is not a clean test). The
        mass-controlled version bins the population in mass quantiles, takes
        the Newcombe per-bin q difference where each class has
        >= MIN_PER_CLASS_PER_CELL, and inverse-variance-weights across bins
        (identical machinery to Method B)."""
        sel_s = (bc == "strong") & popsel
        sel_u = (bc == "unbarred") & popsel
        qs = float(q[sel_s].mean()) if sel_s.sum() else np.nan
        qu = float(q[sel_u].mean()) if sel_u.sum() else np.nan
        for sub, s, val in ((f"{tag}:strong", sel_s, qs),
                            (f"{tag}:unbarred", sel_u, qu)):
            records.append(dict(bar_class=sub, n=int(s.sum()), qfrac=val,
                                qfrac_se=np.nan,
                                frac_removed_agn_clean=np.nan,
                                frac_composite=np.nan, frac_agn=np.nan,
                                frac_liner_lowsn=np.nan, frac_noline=np.nan,
                                med_ssfr_fib_minus_tot=np.nan))
        LOG(f"  {label}, UNCONTROLLED: qfrac strong={qs:.3f} "
            f"(n={int(sel_s.sum())}) vs unbarred={qu:.3f} (n={int(sel_u.sum())})"
            f"  -> d={qs - qu:+.3f}")
        both = sel_s | sel_u
        med = quantile_edges(mass[both], config.N_MASS_BINS)
        mi = np.digitize(mass, med) - 1
        diffs, los, his = [], [], []
        for i in range(len(med) - 1):
            cs = sel_s & (mi == i)
            cu = sel_u & (mi == i)
            ns, nu = int(cs.sum()), int(cu.sum())
            if (ns >= config.MIN_PER_CLASS_PER_CELL
                    and nu >= config.MIN_PER_CLASS_PER_CELL):
                dd, dl, dh = su.newcombe_diff_interval(int(q[cs].sum()), ns,
                                                       int(q[cu].sum()), nu)
                diffs.append(dd); los.append(dl); his.append(dh)
        if diffs:
            se = su.diff_se_from_interval(np.array(los), np.array(his))
            good = se > 0
            wm, wse = su.inverse_variance_weighted_mean(
                np.array(diffs)[good], se[good])
        else:
            wm = wse = np.nan
        # NOTE on this row's schema: `n` holds the number of contributing mass
        # bins (not a galaxy count) and `qfrac` holds the mass-controlled
        # <Delta qfrac> (not a fraction). Its standard error goes in the
        # dedicated `qfrac_se` column.
        records.append(dict(bar_class=f"{tag}:mass_controlled",
                            n=len(diffs), qfrac=wm, qfrac_se=wse,
                            frac_removed_agn_clean=np.nan,
                            frac_composite=np.nan, frac_agn=np.nan,
                            frac_liner_lowsn=np.nan, frac_noline=np.nan,
                            med_ssfr_fib_minus_tot=np.nan))
        # "not detected" rather than "consistent with zero": this contrast is
        # also consistent with the full-sample excess, so it does not
        # discriminate between the two (see Section 6).
        verdict = ("no difference detected"
                   if not (np.isfinite(wse) and wse > 0 and abs(wm) > 2 * wse)
                   else f"DIFFERENCE DETECTED ({abs(wm)/wse:.1f} sigma)")
        LOG(f"  {label}, MASS-CONTROLLED: "
            f"<Delta qfrac>={wm:+.4f}+/-{wse:.4f} over {len(diffs)} mass bins "
            f"-- {verdict}.")

    _population_test("agn_only", "within-BPT AGN contrast (BPT AGN, class 4, only)",
                     bpt == 4)
    _population_test("composite_only", "composite test (BPT class 3 only)",
                     bpt == 3)
    out = config.RESULTS / "05e_bpt_breakdown.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    LOG(f"    wrote {out.name}")
    return records


# ===========================================================================
# main
# ===========================================================================

INTERP = ("Permutation importance measures predictive reliance, depends on "
          "correlated features, and does not bound physical causation. Method A "
          "omits redshift and is descriptive context. Method B stratifies mass, "
          "bulge and redshift; its nominal errors exclude selection and "
          "measurement systematics. Method C includes redshift sensitivity.")


def _story(headline_b, records_c, records_e):
    """Plain-language summary of Methods B, C and E. Every number is read from
    the computed records; the wording follows the manuscript's estimands."""
    comp = {r["bar_class"]: r for r in records_e}
    LOG("")
    LOG("=" * 72)
    LOG("SUMMARY (numbers computed above)")
    LOG("=" * 72)

    def _b(barmode, sample, zcontrol=True):
        g = [h for h in headline_b if h["scheme"] == "comparison"
             and h["barmode"] == barmode and h["sample"] == sample
             and h.get("zcontrol", False) == zcontrol]
        return g[0] if g else None

    # Fiducial: comparison scheme, full sample, mass+bulge grid rebuilt in each
    # redshift slice. The strong-bar random-effects mean is the principal
    # estimate; fixed-effect means weight cells differently and are reported
    # alongside, not as a competing estimate of the same quantity.
    LOG("  METHOD B (comparison scheme, full sample, mass+bulge+z stratified)")
    for barmode, label in (("strong_only", "strong bars"), ("all", "all bars"),
                           ("weak_only", "weak bars")):
        h, hraw = _b(barmode, "all"), _b(barmode, "all", zcontrol=False)
        if not h:
            continue
        LOG(f"    {label:11s}: RE <Delta q> = {h['re_mean_delta_q']:+.4f} +/- "
            f"{h['re_mean_delta_q_se']:.4f}   FE = {h['wmean_delta_q']:+.4f} +/- "
            f"{h['wmean_delta_q_se']:.4f}   ({h['n_valid_cells']} cells, "
            f"Q = {h['cochran_Q']:.1f} / {h['cochran_dof']} dof, "
            f"I^2 = {h['cochran_I2']:.2f})")
        LOG(f"                 descriptive sign census: {h['frac_cells_pos']:.0%} "
            f"positive; intervals excluding zero +{h['n_cells_sig_pos']} / "
            f"-{h['n_cells_sig_neg']}")
        if hraw:
            LOG(f"                 without redshift control: RE "
                f"{hraw['re_mean_delta_q']:+.4f}, FE {hraw['wmean_delta_q']:+.4f} "
                f"({hraw['n_valid_cells']} cells)")
    LOG("    Errors are nominal and conditional on the selection, bins and")
    LOG("    statistical model; they exclude selection and measurement systematics.")
    LOG("    The sign census is descriptive, not an independent significance test.")

    LOG("  BPT 1-2 (agn_clean) tier, Method B redshift-stratified:")
    for barmode in ("all", "strong_only"):
        hf, ha = _b(barmode, "all"), _b(barmode, "agn_clean")
        if hf and ha:
            LOG(f"    {barmode:11s}: RE full = {hf['re_mean_delta_q']:+.4f}"
                f"+/-{hf['re_mean_delta_q_se']:.4f}  ->  BPT 1-2 = "
                f"{ha['re_mean_delta_q']:+.4f}+/-{ha['re_mean_delta_q_se']:.4f}")
    LOG("    This tier changes the compared population (Method E); it is not")
    LOG("    evidence for or against AGN-mediated quenching.")

    dec = [r for r in records_c if r["scheme"] == "comparison"
           and r["subset"] == "strong+weak" and r["sample"] == "all"
           and r["tier"] == "main"]
    if dec:
        LOG("  METHOD C (comparison, all discs, main sample), corr(bar score, dSFR):")
        for lab in ("none", "mass", "mass+bulge", "mass+bulge+z"):
            r = [x for x in dec if x["controls"] == lab]
            if r:
                LOG(f"    | {lab:16s} : {r[0]['pcc']:+.4f} "
                    f"[{r[0]['lo']:+.4f}, {r[0]['hi']:+.4f}]")
        LOG("    The residual rank association is sensitive to adding redshift.")

    cs, cu = comp.get("strong", {}), comp.get("unbarred", {})
    agnmc = comp.get("agn_only:mass_controlled", {})
    cmpmc = comp.get("composite_only:mass_controlled", {})
    if cs and cu:
        LOG("  METHOD E (BPT composition, comparison scheme):")
        LOG(f"    BPT 1-2 selection removes {cs['frac_removed_agn_clean']:.0%} of "
            f"strong-barred and {cu['frac_removed_agn_clean']:.0%} of unbarred discs.")
        LOG(f"    Removed strong bars: composite {cs['frac_composite']:.0%}, "
            f"AGN {cs['frac_agn']:.0%}, low-S/N LINER {cs['frac_liner_lowsn']:.0%}, "
            f"unclassifiable {cs['frac_noline']:.0%} "
            f"(unbarred: {cu['frac_composite']:.0%} / {cu['frac_agn']:.0%} / "
            f"{cu['frac_liner_lowsn']:.0%} / {cu['frac_noline']:.0%}).")
        if agnmc and np.isfinite(agnmc.get("qfrac", np.nan)):
            LOG(f"    Mass-controlled contrast within BPT class 4: "
                f"{agnmc['qfrac']:+.4f} +/- {agnmc['qfrac_se']:.4f}; "
                f"within class 3: {cmpmc.get('qfrac', np.nan):+.4f} +/- "
                f"{cmpmc.get('qfrac_se', np.nan):.4f}.")
            LOG("    The class-4 contrast is compatible with both zero and the overall")
            LOG("    excess; these comparisons neither establish nor exclude AGN mediation.")
        LOG(f"    Median log sSFR_fib - log sSFR_tot: strong "
            f"{cs['med_ssfr_fib_minus_tot']:+.3f}, unbarred "
            f"{cu['med_ssfr_fib_minus_tot']:+.3f} dex (aperture- and")
        LOG("    estimator-dependent; gas inflow and depletion were not measured).")
    LOG("  METHOD D: the low-mass (9.0-9.5) fibre contrast has the same sign as the")
    LOG("    L&Z low-mass enhancement; the full 9.5-10.5 mean contrast at z<0.05 does")
    LOG("    not reproduce their approximately -0.2 dex suppression. Their mass-and-")
    LOG("    colour matching, apertures and SFR estimators differ, so these")
    LOG("    within-sample statistics do not identify the cause of the difference.")


def main():
    # Independent, fixed-seed streams per method: each method is reproducible on
    # its own, so reusing a cached Method-A result cannot shift B/C/D.
    global LOG
    LOG = Audit(config.RESULTS / "05_summary.txt")
    seed = config.RANDOM_STATE
    rng_a = np.random.default_rng(seed + 1)
    rng_c = np.random.default_rng(seed + 3)
    rng_d = np.random.default_rng(seed + 4)
    LOG("05_bar_test.py -- conditional bar-quenching associations (Methods A-E)\n")
    LOG(f"  Method-A cache: {'DISABLED (--force-recompute)' if FORCE_RECOMPUTE else 'enabled'}\n")

    frames = {s: build_frame(s) for s in SCHEMES}
    lowz_frames = {s: build_lowz(s) for s in SCHEMES}
    for s in SCHEMES:
        LOG(f"  frame[{s}]: {len(frames[s]):,} in-sample main discs; "
            f"low-z slice {len(lowz_frames[s]):,}")
    LOG("")

    a = method_a(frames, rng_a)
    b = method_b(frames, rng_c)          # binning is deterministic; rng unused
    method_b_quench_robustness(frames)   # quenched-definition stability
    c = method_c(frames, rng_c)
    d = method_d(frames, lowz_frames, rng_d)
    e = method_e(frames)                 # composition is deterministic; no rng

    _story(b, c, e)

    LOG("")
    LOG("  " + INTERP)
    LOG("")
    LOG(f"  wrote results/05a_rf_importance_*.csv, "
        f"05b_controlled_binning_*.csv, 05c_partial_correlation.csv, "
        f"05d_fibre_ssfr.csv, 05e_bpt_breakdown.csv")

    out = config.RESULTS / "05_summary.txt"
    LOG.save(out)
    print(f"\nSAVED summary -> {config.rel(out)}")


if __name__ == "__main__":
    main()
