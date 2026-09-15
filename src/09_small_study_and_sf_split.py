"""Precision-association diagnostic and fibre population decomposition.

The Egger standardized intercept is tested correctly, but its nominal p-value
does not calibrate bias in these binomial contrasts. Effects and SEs share
data; a non-significant count correlation does not establish absence of bias.
The exhaustive quenched/non-quenched mean decomposition is separate from
descriptive median contrasts; non-quenched includes green-valley galaxies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import stats_utils as su

Z975 = 1.959963984540054
N_BOOT = 1000

LINES = []


def LOG(msg=""):
    print(msg)
    LINES.append(str(msg))


# ---------------------------------------------------------------------------
# shared helpers (kept local so this script depends on nothing in 05)
# ---------------------------------------------------------------------------

def se_from_interval(lo, hi):
    """Symmetric SE from a Newcombe score interval, as used in Method B."""
    return (np.asarray(hi, float) - np.asarray(lo, float)) / (2.0 * Z975)


def random_effects(d, se):
    """DerSimonian-Laird random-effects mean and SE, matching stats_utils."""
    d = np.asarray(d, float)
    se = np.asarray(se, float)
    w = 1.0 / se ** 2
    fe = (w * d).sum() / w.sum()
    Q = (w * (d - fe) ** 2).sum()
    dof = d.size - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - dof) / C) if C > 0 else 0.0
    ws = 1.0 / (se ** 2 + tau2)
    return (ws * d).sum() / ws.sum(), ws.sum() ** -0.5


def spearman(x, y):
    """Spearman rho and its two-sided p-value, via the t approximation."""
    x = pd.Series(np.asarray(x, float)).rank().to_numpy()
    y = pd.Series(np.asarray(y, float)).rank().to_numpy()
    n = x.size
    rho = float(np.corrcoef(x, y)[0, 1])
    if abs(rho) >= 1.0 or n < 3:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1.0 - rho ** 2))
    return rho, float(2.0 * _t_sf(abs(t), n - 2))


def _t_sf(t, dof):
    """Upper-tail Student-t probability via the regularised incomplete beta."""
    x = dof / (dof + t * t)
    return 0.5 * _betainc(dof / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b), Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = np.exp(a * np.log(x) + b * np.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - np.exp(b * np.log(1.0 - x) + a * np.log(x) - lbeta) \
        * _betacf(b, a, 1.0 - x) / b


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _lgamma(x):
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    x = float(x)
    if x < 0.5:
        return np.log(np.pi / np.sin(np.pi * x)) - _lgamma(1.0 - x)
    x -= 1.0
    a = c[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += c[i] / (x + i)
    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


def median_diff(a, b, rng, n_boot=N_BOOT):
    """median(a) - median(b) with a 16/84 bootstrap band, as in Method D."""
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


# ---------------------------------------------------------------------------
# (A) small-study diagnostic on the 75 fiducial strong-bar cells
# ---------------------------------------------------------------------------

def part_a():
    LOG("=" * 72)
    LOG("(A) SMALL-STUDY DIAGNOSTIC  (fiducial z-controlled strong-bar cells)")
    LOG("=" * 72)

    path = (config.RESULTS
            / "05b_controlled_binning_zsliced_comparison_strong_only.csv")
    cells = pd.read_csv(path)
    v = cells[cells["valid"].to_numpy(bool)
              & (cells["sample"] == "all")].copy()
    v["se"] = se_from_interval(v["delta_q_lo"], v["delta_q_hi"])
    d = v["delta_q"].to_numpy(float)
    se = v["se"].to_numpy(float)
    nb = v["n_barred"].to_numpy(float)

    LOG(f"    source table ......................... {path.name}")
    LOG(f"    valid fiducial cells ................. {len(v)}")
    LOG("")

    egger = su.egger_precision_diagnostic(d, se)
    p_int = egger["p_value"]
    LOG("    Egger precision-association diagnostic (nominal Student-t test)")
    LOG("      model: Delta q / se = b + a / se, ordinary least squares")
    LOG(f"      standardized intercept b = {egger['intercept']:+.6f}")
    LOG(f"      SE(b) = {egger['intercept_se']:.6f}; a = {egger['slope']:+.6f}")
    LOG(f"      t = {egger['t']:+.5f} on {egger['dof']} dof; p = {p_int:.6e}")
    LOG("")

    r_se, p_se = spearman(se, d)
    r_n, p_n = spearman(nb, d)
    LOG("    Rank correlations of the per-cell excess")
    LOG(f"      Spearman(se, Delta q) ......... rho = {r_se:+.3f}, "
        f"p = {p_se:.3e}")
    LOG(f"      Spearman(n_barred, Delta q) ... rho = {r_n:+.3f}, "
        f"p = {p_n:.3f}")
    LOG("")

    med = float(np.median(nb))
    small = nb < med
    large = nb >= med
    m_s, s_s = random_effects(d[small], se[small])
    m_l, s_l = random_effects(d[large], se[large])
    LOG("    Random-effects mean split at the median barred count")
    LOG(f"      median n_barred per cell .......... {med:.1f}")
    LOG(f"      smaller half (n_barred <  median) . {int(small.sum())} cells, "
        f"<Delta q>_RE = {m_s:+.4f} +/- {s_s:.4f}")
    LOG(f"      larger  half (n_barred >= median) . {int(large.sum())} cells, "
        f"<Delta q>_RE = {m_l:+.4f} +/- {s_l:.4f}")
    LOG("")
    LOG("    The positive larger-cell estimate is a descriptive sensitivity check.")
    LOG("    Binomial effects and SEs are coupled; these are not published studies.")
    LOG("    Neither this diagnostic nor the count correlation establishes the")
    LOG("    presence or absence of bias, or bounds residual confounding.")
    LOG("")
    LOG("    PRECISION DIAGNOSTIC:")
    LOG(f"      Egger standardized-intercept nominal p = {p_int:.6e}")
    LOG(f"      Spearman(se, Delta q) rho = {r_se:+.2f}, p = {p_se:.1e}")
    LOG(f"      Spearman(n_barred, Delta q) rho = {r_n:+.2f}, p = {p_n:.2f}")
    LOG(f"      smaller half {m_s:+.4f} +/- {s_s:.4f}; "
        f"larger half {m_l:+.4f} +/- {s_l:.4f}")
    LOG("")


# ---------------------------------------------------------------------------
# (B) strictly star-forming split of the fibre decomposition
# ---------------------------------------------------------------------------

def part_b():
    LOG("=" * 72)
    LOG("(B) STRICT STAR-FORMING SPLIT  (fibre decomposition, Section 5.2)")
    LOG("=" * 72)

    df = pd.read_parquet(config.MERGED)
    m = df[df["in_main_sample"].to_numpy(bool)
           & df["in_sample_comparison"].to_numpy(bool)].copy()
    z = pd.to_numeric(m["z"], errors="coerce").to_numpy(float)
    mass = pd.to_numeric(m["lgm_tot_p50"], errors="coerce").to_numpy(float)
    fib = pd.to_numeric(m["specsfr_fib_p50"], errors="coerce").to_numpy(float)
    qfl = m["quenched_dsfr"].to_numpy(bool)
    gvl = m["green_valley"].to_numpy(bool)
    cls = m["bar_class_comparison"].to_numpy()

    band = ((z >= config.LOWZ_SLICE["z_min"]) & (z < 0.05)
            & (mass >= 9.5) & (mass < 10.5))
    strong = band & (cls == "strong")
    unb = band & (cls == "unbarred")

    LOG("    band: 9.5 <= log M* < 10.5, 0.02 <= z < 0.05, "
        "comparison scheme")
    LOG("")
    LOG("    Population shares by bar class")
    for name, sel in (("strong", strong), ("unbarred", unb)):
        n_tot = int(sel.sum())
        n_nq = int((sel & ~qfl).sum())
        n_gv = int((sel & ~qfl & gvl).sum())
        n_sf = int((sel & ~qfl & ~gvl).sum())
        n_q = int((sel & qfl).sum())
        LOG(f"      {name:9s}: N = {n_tot:5d}   quenched {n_q:5d}   "
            f"non-quenched {n_nq:5d}  (green valley {n_gv:4d} = "
            f"{n_gv / n_nq:.1%}, strictly SF {n_sf:5d})")
    LOG("")

    # Exact identity for means, using unbarred within-population means as
    # the reference. Median differences do not satisfy this identity.
    qs, qu = qfl[strong].mean(), qfl[unb].mean()
    msq, muq = fib[strong & qfl].mean(), fib[unb & qfl].mean()
    msn, mun = fib[strong & ~qfl].mean(), fib[unb & ~qfl].mean()
    composition = (qs - qu) * (muq - mun)
    within_q = qs * (msq - muq)
    within_nq = (1 - qs) * (msn - mun)
    mean_difference = fib[strong].mean() - fib[unb].mean()
    assert np.isclose(composition + within_q + within_nq, mean_difference, atol=1e-12)
    LOG(f"    Mean decomposition: composition {composition:+.6f}, "
        f"within-quenched {within_q:+.6f}, within-non-quenched {within_nq:+.6f}")
    LOG(f"    Sum = observed mean contrast {mean_difference:+.6f} dex")
    LOG(f"    Quenched-fraction ratio strong/unbarred = {qs/qu:.6f}")
    LOG("    This decomposition describes this sample, not the cause of another survey's result.")
    LOG("")

    # The two published contrasts are READ from the frozen Method-D table, not
    # re-bootstrapped here: a fresh RNG stream would shift their 16/84 bands in
    # the third decimal and manufacture a disagreement with the saved results.
    # Only the strictly-star-forming contrast is new, so only it is bootstrapped.
    fibre = pd.read_csv(config.RESULTS / "05d_fibre_ssfr.csv")
    pub = fibre[(fibre["scheme"] == "comparison")
                & (fibre["sample"] == "all")].set_index("control")
    nq_d, nq_lo, nq_hi = (float(pub.loc["decomp_nonquenched_z<0.05", c])
                          for c in ("delta", "delta_lo", "delta_hi"))
    q_d, q_lo, q_hi = (float(pub.loc["decomp_Q_only_z<0.05", c])
                       for c in ("delta", "delta_lo", "delta_hi"))

    rng = np.random.default_rng(config.RANDOM_STATE + 9)
    sf_d, sf_lo, sf_hi = median_diff(fib[strong & ~qfl & ~gvl],
                                     fib[unb & ~qfl & ~gvl], rng)

    LOG("    Delta median log sSFR_fib (strong - unbarred), "
        "16/84 bootstrap band")
    LOG(f"      non-quenched members (from 05d) ........... {nq_d:+.4f} "
        f"[{nq_lo:+.4f}, {nq_hi:+.4f}]")
    LOG(f"      strictly star-forming members ................ {sf_d:+.4f} "
        f"[{sf_lo:+.4f}, {sf_hi:+.4f}]")
    LOG(f"      quenched members (from 05d) ................ {q_d:+.4f} "
        f"[{q_lo:+.4f}, {q_hi:+.4f}]")
    LOG("")
    LOG("    Reading: the quenched / non-quenched split is the one used in the")
    LOG("    analysis, because the composition argument needs an EXHAUSTIVE "
        "two-way")
    LOG("    partition that reconstructs the band mean. The green valley "
        "therefore")
    LOG("    sits inside the non-quenched group, and it is a larger share of "
        "the")
    LOG("    strong-barred members than of the unbarred ones, so it DILUTES "
        "the")
    LOG("    central enhancement rather than creating it: removing it raises "
        "the")
    LOG(f"    contrast from {nq_d:+.3f} to {sf_d:+.3f} dex.")
    LOG("")
    LOG("    POPULATION COMPARISON:")
    LOG(f"      non-quenched contrast {nq_d:+.3f} "
        f"[{nq_lo:+.3f}, {nq_hi:+.3f}]")
    LOG(f"      strictly star-forming contrast {sf_d:+.2f} dex")
    LOG("")


def main():
    LOG("09_small_study_and_sf_split.py: provenance for two more number sets")
    LOG("")
    part_a()
    part_b()
    out = config.RESULTS / "09_small_study_and_sf_split.txt"
    out.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
