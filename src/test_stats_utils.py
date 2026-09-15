"""Self-checks for stats_utils.py: run before trusting any science output.

These tests use synthetic data with known answers, so they need no catalogs.
Run:  py src/test_stats_utils.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import stats_utils as su


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_wilson_known_value():
    # k=0, n=10, 95%: Wilson upper bound = 0.27753 (NB: 0.3085 is the
    # Clopper-Pearson exact bound, which is a different interval).
    lo, hi = su.wilson_interval(0, 10)
    assert approx(lo, 0.0, 1e-12), f"lower should be 0, got {lo}"
    assert approx(hi, 0.27753, 1e-4), f"upper should be ~0.27753, got {hi}"
    # Symmetric case k=n: lower bound must be > 0 (normal SE would give 0).
    lo2, hi2 = su.wilson_interval(10, 10)
    assert approx(hi2, 1.0, 1e-12)
    assert lo2 > 0.6, f"Wilson lower at p=1 should be well above 0, got {lo2}"
    print("  [ok] wilson_interval known values + saturation behaviour")


def test_wilson_beats_normal_at_saturation():
    # The normal-approximation SE sqrt(p(1-p)/n) is 0 at p=1; Wilson is not.
    k, n = 40, 40
    naive_se = np.sqrt((k / n) * (1 - k / n) / n)
    lo, hi = su.wilson_interval(k, n)
    assert approx(naive_se, 0.0), "normal SE is degenerate at p=1 (as warned)"
    assert (hi - lo) > 0.05, "Wilson must retain finite width at p=1"
    print("  [ok] Wilson non-degenerate where normal SE collapses")


def test_newcombe_symmetry():
    # Swapping the two groups negates the difference and mirrors the interval.
    d1, lo1, hi1 = su.newcombe_diff_interval(30, 100, 10, 100)
    d2, lo2, hi2 = su.newcombe_diff_interval(10, 100, 30, 100)
    assert approx(d1, -d2, 1e-12)
    assert approx(lo1, -hi2, 1e-9)
    assert approx(hi1, -lo2, 1e-9)
    # A clear difference (0.30 vs 0.10) should exclude zero.
    assert lo1 > 0, f"interval should exclude 0 for a clear difference, got lo={lo1}"
    print("  [ok] newcombe_diff_interval symmetry + significance")


def test_ivw_mean():
    # Two measurements, equal SE -> plain mean; unequal -> pulled to precise one.
    m, se = su.inverse_variance_weighted_mean([0.2, 0.4], [0.1, 0.1])
    assert approx(m, 0.3, 1e-12)
    m2, _ = su.inverse_variance_weighted_mean([0.2, 0.4], [0.01, 1.0])
    assert m2 < 0.21, f"weighted mean should sit near the precise value, got {m2}"
    print("  [ok] inverse_variance_weighted_mean")


def test_partial_corr_two_methods_agree():
    # Random data: matrix and recursive PCC must give identical answers.
    rng = np.random.default_rng(0)
    n = 5000
    m = rng.normal(size=n)                     # mass-like driver
    bulge = 0.7 * m + rng.normal(size=n) * 0.5  # correlated with mass
    bar = 0.4 * bulge + rng.normal(size=n) * 0.8
    dsfr = -0.6 * bulge - 0.1 * bar + rng.normal(size=n) * 0.5
    df = pd.DataFrame(dict(bar=bar, dsfr=dsfr, mass=m, bulge=bulge))
    r_mat = su.spearman_partial_corr(df, "bar", "dsfr", ["mass", "bulge"], "matrix")
    r_rec = su.spearman_partial_corr(df, "bar", "dsfr", ["mass", "bulge"], "recursive")
    assert approx(r_mat, r_rec, 1e-8), f"matrix {r_mat} != recursive {r_rec}"
    print(f"  [ok] partial corr matrix==recursive  (rho={r_mat:+.4f})")


def test_partial_corr_removes_confound():
    # Construct bar with NO true effect on dsfr except through bulge.
    # Raw Spearman(bar, dsfr) should be non-zero; partialling out bulge -> ~0.
    rng = np.random.default_rng(1)
    n = 20000
    bulge = rng.normal(size=n)
    bar = 0.8 * bulge + rng.normal(size=n) * 0.3   # bar is driven by bulge
    dsfr = -0.9 * bulge + rng.normal(size=n) * 0.3  # dsfr driven by bulge only
    df = pd.DataFrame(dict(bar=bar, dsfr=dsfr, bulge=bulge))
    from scipy.stats import spearmanr
    raw = spearmanr(df.bar, df.dsfr).statistic
    partial = su.spearman_partial_corr(df, "bar", "dsfr", ["bulge"], "matrix")
    assert abs(raw) > 0.3, f"raw correlation should be sizeable, got {raw}"
    assert abs(partial) < 0.05, (
        f"partialling out the confounder should null the bar signal, got {partial}")
    print(f"  [ok] confound removal:  raw={raw:+.3f}  partial={partial:+.3f}")


def test_partial_corr_keeps_true_signal():
    # Now give the bar a genuine independent effect; it must survive.
    rng = np.random.default_rng(2)
    n = 20000
    bulge = rng.normal(size=n)
    bar = 0.5 * bulge + rng.normal(size=n) * 0.7
    dsfr = -0.6 * bulge - 0.5 * bar + rng.normal(size=n) * 0.3  # real bar term
    df = pd.DataFrame(dict(bar=bar, dsfr=dsfr, bulge=bulge))
    partial = su.spearman_partial_corr(df, "bar", "dsfr", ["bulge"], "matrix")
    assert partial < -0.2, f"genuine bar signal should survive, got {partial}"
    print(f"  [ok] genuine signal survives:  partial={partial:+.3f}")


def test_chi2_sf_matches_scipy():
    # The pure-NumPy chi^2 survival function must match SciPy across regimes
    # (both the series branch, x < a+1, and the continued-fraction branch).
    from scipy.stats import chi2
    for dof in (1, 2, 5, 74, 89):
        for x in (0.5 * dof, dof, 2.0 * dof, 3.0 * dof):
            got = su._chi2_sf(x, dof)
            ref = float(chi2.sf(x, dof))
            assert abs(got - ref) < 1e-6, (
                f"chi2_sf(x={x}, dof={dof}) = {got} vs scipy {ref}")
    print("  [ok] _chi2_sf matches scipy.stats.chi2.sf")


def test_cochran_homogeneous_vs_heterogeneous():
    # Homogeneous cells (all share one mean): Q ~ dof, I^2 ~ 0.
    rng = np.random.default_rng(3)
    k = 200
    se = np.full(k, 0.05)
    vals_homo = rng.normal(0.03, se)               # scatter == sampling error
    h = su.cochran_heterogeneity(vals_homo, se)
    assert abs(h["Q"] - h["dof"]) < 4 * np.sqrt(2 * h["dof"]), \
        f"homogeneous Q should be ~dof, got Q={h['Q']} dof={h['dof']}"
    assert h["I2"] < 0.2, f"homogeneous I^2 should be small, got {h['I2']}"
    # Heterogeneous: add real between-cell scatter tau=0.05 on top of se=0.05.
    vals_het = rng.normal(0.03, np.sqrt(se ** 2 + 0.05 ** 2))
    h2 = su.cochran_heterogeneity(vals_het, se)
    assert h2["I2"] > 0.4, f"heterogeneous I^2 should be sizeable, got {h2['I2']}"
    assert h2["p_value"] < 1e-3, "heterogeneous Q should be highly significant"
    print(f"  [ok] Cochran Q/I^2: homo I2={h['I2']:.2f}, het I2={h2['I2']:.2f}")


def test_random_effects_reduces_and_widens():
    # (a) Homogeneous -> tau^2 ~ 0 and RE mean/SE ~ fixed-effect.
    rng = np.random.default_rng(4)
    k = 300
    se = np.full(k, 0.05)
    vals_homo = rng.normal(0.03, se)
    fe_m, fe_se = su.inverse_variance_weighted_mean(vals_homo, se)
    re = su.random_effects_mean(vals_homo, se)
    assert re["tau2"] < 1e-4, f"homogeneous tau^2 should be ~0, got {re['tau2']}"
    assert approx(re["se"], fe_se, 1e-3), "RE SE should match FE when homogeneous"
    # (b) Heterogeneous -> RE SE strictly WIDER than FE SE (the whole point).
    vals_het = rng.normal(0.03, np.sqrt(se ** 2 + 0.06 ** 2))
    fe_m2, fe_se2 = su.inverse_variance_weighted_mean(vals_het, se)
    re2 = su.random_effects_mean(vals_het, se)
    assert re2["se"] > fe_se2 * 1.5, (
        f"RE SE should exceed FE SE under heterogeneity, got "
        f"RE={re2['se']:.4f} vs FE={fe_se2:.4f}")
    print(f"  [ok] random effects: homo SE~FE ({re['se']:.4f}), "
          f"het SE>FE ({re2['se']:.4f}>{fe_se2:.4f})")


def main():
    print("Running stats_utils self-checks...")
    for fn in (
        test_wilson_known_value,
        test_wilson_beats_normal_at_saturation,
        test_newcombe_symmetry,
        test_ivw_mean,
        test_partial_corr_two_methods_agree,
        test_partial_corr_removes_confound,
        test_partial_corr_keeps_true_signal,
        test_chi2_sf_matches_scipy,
        test_cochran_homogeneous_vs_heterogeneous,
        test_random_effects_reduces_and_widens,
    ):
        fn()
    print("All stats_utils checks passed.")


if __name__ == "__main__":
    main()
