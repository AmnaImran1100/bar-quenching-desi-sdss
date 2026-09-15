"""Statistical kernels for the bar-quenching analysis.

Statistical functions used throughout the analysis. Two points matter most:

  1. Binomial errors on a quenched fraction must use the WILSON score interval,
     not the normal-approximation SE  sqrt(p(1-p)/N).  The normal SE collapses
     to zero when p -> 0 or 1, which is exactly the high-mass / high-bulge
     corner of the Method-B grid where quenched fractions saturate and where the
     bar signal (if any) lives.  A difference of two proportions (Delta q =
     q_barred - q_unbarred) is given a Newcombe (score-based) interval, which
     inherits the good tail behaviour of Wilson.

  2. A partial correlation is NOT scipy.stats.spearmanr of two variables.
     Controlling for two (or three) covariates requires a second- (third-)
     order partial correlation.  It is implemented here two independent ways
     that must agree: (a) the recursive first-order formula, and (b) inversion
     of the correlation (precision) matrix.  For Spearman partial correlations,
     rank-transform every column first, then call these on Pearson correlations
     of the ranks.

Everything is pure NumPy/pandas: no pingouin, no statsmodels dependency.
"""

from __future__ import annotations

import numpy as np


def egger_precision_diagnostic(values, ses):
    """OLS of effect/SE on precision; test the standardized intercept.

    Equivalent to WLS of effect = a + b*SE with weights 1/SE**2,
    testing b, not a. For binomial cell contrasts this is descriptive:
    the estimates and their SEs are coupled, so the nominal t probability
    is not a calibrated test for publication bias or residual confounding.
    """
    from scipy.stats import t

    d, se = np.asarray(values, float), np.asarray(ses, float)
    if (d.ndim != 1 or d.shape != se.shape or d.size < 3
            or not np.all(np.isfinite(d)) or not np.all(np.isfinite(se))
            or np.any(se <= 0)):
        raise ValueError("Egger diagnostic requires >=3 finite effects and positive SEs")
    x = np.column_stack([np.ones(d.size), 1.0 / se])
    if np.linalg.matrix_rank(x) != 2:
        raise ValueError("Egger diagnostic requires varying precision")
    y = d / se
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    dof = d.size - 2
    variance = np.sum((y - x @ beta) ** 2) / dof
    cov = variance * np.linalg.inv(x.T @ x)
    intercept_se = float(np.sqrt(cov[0, 0]))
    statistic = float(beta[0] / intercept_se)
    return dict(intercept=float(beta[0]), slope=float(beta[1]),
                intercept_se=intercept_se, t=statistic, dof=dof,
                p_value=float(2 * t.sf(abs(statistic), dof)))


# ---------------------------------------------------------------------------
# Binomial proportion intervals (Method B)
# ---------------------------------------------------------------------------

def wilson_interval(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : int or array-like
        Number of successes (e.g. quenched galaxies in a cell).
    n : int or array-like
        Number of trials (e.g. total galaxies of that class in a cell).
    z : float
        Standard-normal quantile.  Default 1.95996... gives a 95% interval.

    Returns
    -------
    (lower, upper) : tuple of float or ndarray
        The Wilson score interval.  Well-behaved at k = 0 and k = n, unlike the
        normal approximation.

    Notes
    -----
    center = (phat + z^2/2n) / (1 + z^2/n)
    half   = (z / (1 + z^2/n)) * sqrt( phat(1-phat)/n + z^2/(4 n^2) )
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    if np.any(n <= 0):
        raise ValueError("wilson_interval: n must be positive.")
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    lower = center - half
    upper = center + half
    # Clip to [0, 1] to guard tiny floating overshoots.
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    if lower.ndim == 0:
        return float(lower), float(upper)
    return lower, upper


def newcombe_diff_interval(k1, n1, k2, n2, z=1.959963984540054):
    """Newcombe (method 10) score interval for the difference of two proportions.

    Returns a confidence interval for  d = p1 - p2  built from the two Wilson
    intervals.  This is the correct interval for Delta q in Method B; the naive
    sqrt(se1^2 + se2^2) again misbehaves when either fraction saturates.

    Parameters
    ----------
    k1, n1 : successes / trials for group 1 (e.g. barred).
    k2, n2 : successes / trials for group 2 (e.g. unbarred).

    Returns
    -------
    (diff, lower, upper) : floats
        diff = p1 - p2, and its lower/upper confidence bounds.
    """
    p1 = k1 / n1
    p2 = k2 / n2
    l1, u1 = wilson_interval(k1, n1, z)
    l2, u2 = wilson_interval(k2, n2, z)
    diff = p1 - p2
    lower = diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, lower, upper


def diff_se_from_interval(lower, upper, z=1.959963984540054):
    """Approximate a symmetric standard error from a (possibly asymmetric)
    score interval, for inverse-variance weighting across cells.

    se ~= (upper - lower) / (2 z).  This is an approximation used only for the
    weighted-mean summary statistic; per-cell inference uses the full interval.
    """
    return (np.asarray(upper) - np.asarray(lower)) / (2.0 * z)


def inverse_variance_weighted_mean(values, ses):
    """Inverse-variance weighted (FIXED-EFFECT) mean and its standard error.

    Used to summarise Delta q across all valid Method-B cells. NOTE: this is a
    fixed-effect estimator: its SE propagates only the per-cell sampling error
    and assumes every cell measures one common Delta q. When the cells are
    heterogeneous (the true Delta q varies across the mass/bulge/z grid) the
    fixed-effect SE is an UNDER-estimate; use `random_effects_mean` /
    `cochran_heterogeneity` to quantify and absorb that between-cell scatter.
    """
    values = np.asarray(values, dtype=float)
    ses = np.asarray(ses, dtype=float)
    if np.any(ses <= 0):
        raise ValueError("inverse_variance_weighted_mean: SEs must be positive.")
    w = 1.0 / (ses ** 2)
    mean = np.sum(w * values) / np.sum(w)
    se = np.sqrt(1.0 / np.sum(w))
    return mean, se


def cochran_heterogeneity(values, ses):
    """Cochran's Q test for between-cell heterogeneity of a set of estimates.

    Given per-cell effect sizes and their standard errors, tests the null that
    all cells share one common effect (the fixed-effect assumption). Returns the
    heterogeneity statistic Q ~ chi^2 with (k-1) dof, the dof, the upper-tail
    p-value, and the I^2 fraction (share of total variance due to real
    cell-to-cell differences rather than sampling error).

    Q  = sum_i w_i (y_i - ybar_FE)^2 ,  w_i = 1/se_i^2
    I^2 = max(0, (Q - dof) / Q)

    Returns
    -------
    dict(Q, dof, p_value, I2)
    """
    values = np.asarray(values, dtype=float)
    ses = np.asarray(ses, dtype=float)
    if np.any(ses <= 0):
        raise ValueError("cochran_heterogeneity: SEs must be positive.")
    k = values.size
    if k < 2:
        return dict(Q=np.nan, dof=max(k - 1, 0), p_value=np.nan, I2=np.nan)
    w = 1.0 / (ses ** 2)
    ybar = np.sum(w * values) / np.sum(w)
    Q = float(np.sum(w * (values - ybar) ** 2))
    dof = k - 1
    # upper-tail chi^2 survival function without a SciPy dependency, via the
    # regularised upper incomplete gamma  Q_gamma(dof/2, Q/2).
    p_value = float(_chi2_sf(Q, dof))
    I2 = float(max(0.0, (Q - dof) / Q)) if Q > 0 else 0.0
    return dict(Q=Q, dof=dof, p_value=p_value, I2=I2)


def random_effects_mean(values, ses):
    """DerSimonian-Laird random-effects pooled mean and its standard error.

    Unlike `inverse_variance_weighted_mean`, this adds the estimated between-cell
    variance tau^2 to every cell's variance before weighting, so the pooled SE
    reflects genuine heterogeneity as well as sampling error. When the cells are
    homogeneous tau^2 -> 0 and this reduces to the fixed-effect result.

    tau^2 = max(0, (Q - dof) / C),   C = sum w_i - sum w_i^2 / sum w_i
    w*_i  = 1 / (se_i^2 + tau^2)

    Returns
    -------
    dict(mean, se, tau2, Q, dof, I2, p_value, k)
    """
    values = np.asarray(values, dtype=float)
    ses = np.asarray(ses, dtype=float)
    if np.any(ses <= 0):
        raise ValueError("random_effects_mean: SEs must be positive.")
    k = values.size
    het = cochran_heterogeneity(values, ses)
    w = 1.0 / (ses ** 2)
    sw = np.sum(w)
    if k < 2:
        mean = float(np.sum(w * values) / sw)
        return dict(mean=mean, se=float(np.sqrt(1.0 / sw)), tau2=0.0,
                    Q=het["Q"], dof=het["dof"], I2=het["I2"],
                    p_value=het["p_value"], k=int(k))
    C = sw - np.sum(w ** 2) / sw
    tau2 = max(0.0, (het["Q"] - het["dof"]) / C) if C > 0 else 0.0
    wstar = 1.0 / (ses ** 2 + tau2)
    mean = float(np.sum(wstar * values) / np.sum(wstar))
    se = float(np.sqrt(1.0 / np.sum(wstar)))
    return dict(mean=mean, se=se, tau2=float(tau2), Q=het["Q"], dof=het["dof"],
                I2=het["I2"], p_value=het["p_value"], k=int(k))


def _chi2_sf(x, dof):
    """Upper-tail probability P(X > x) for X ~ chi^2_dof, pure NumPy.

    Uses the regularised upper incomplete gamma function Q(a, z) with
    a = dof/2, z = x/2, evaluated by a series (small z) or the Lentz continued
    fraction (large z). Accurate to ~1e-10 over the range we need; avoids a
    SciPy dependency to keep this module NumPy-only (see module docstring).
    """
    if dof <= 0 or x <= 0:
        return 1.0
    a = dof / 2.0
    z = x / 2.0
    gln = _lgamma(a)
    if z < a + 1.0:
        # lower incomplete gamma via series -> P(a,z); return 1 - P.
        term = 1.0 / a
        summ = term
        n = a
        for _ in range(2000):
            n += 1.0
            term *= z / n
            summ += term
            if abs(term) < abs(summ) * 1e-14:
                break
        p = summ * np.exp(-z + a * np.log(z) - gln)
        return float(max(0.0, min(1.0, 1.0 - p)))
    # upper incomplete gamma via Lentz continued fraction -> Q(a,z) directly.
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = np.exp(-z + a * np.log(z) - gln) * h
    return float(max(0.0, min(1.0, q)))


def _lgamma(x):
    """log Gamma(x) via the Lanczos approximation (NumPy-only helper)."""
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


# ---------------------------------------------------------------------------
# Partial correlation (Method C)
# ---------------------------------------------------------------------------

def _pearson_corr_matrix(data):
    """Pearson correlation matrix of columns of a 2D array (rows = samples)."""
    data = np.asarray(data, dtype=float)
    return np.corrcoef(data, rowvar=False)


def partial_corr_from_matrix(corr, i, j, controls):
    """Partial correlation of variables i and j controlling for `controls`,
    computed by inverting the relevant submatrix (precision-matrix method).

    Parameters
    ----------
    corr : (p, p) ndarray
        A correlation matrix (Pearson of ranks -> Spearman partial correlation).
    i, j : int
        Indices of the two variables of interest.
    controls : sequence of int
        Indices of the variables to control for.

    Returns
    -------
    float : the partial correlation rho_{ij . controls}.
    """
    idx = [i, j] + list(controls)
    sub = corr[np.ix_(idx, idx)]
    precision = np.linalg.inv(sub)
    # partial corr between the first two entries of `idx`:
    return -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])


def _first_order_pcc(r_xy, r_xz, r_yz):
    """First-order partial correlation from three pairwise correlations."""
    return (r_xy - r_xz * r_yz) / np.sqrt((1.0 - r_xz ** 2) * (1.0 - r_yz ** 2))


def partial_corr_recursive(corr, x, y, controls):
    """Partial correlation via the recursive first-order formula.

    Independent implementation of `partial_corr_from_matrix`, used to
    cross-check it.  `controls` is an ordered sequence of covariate indices.

    r_{xy.z}   = (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    r_{xy.zw}  = (r_{xy.z} - r_{xw.z} r_{yw.z}) / sqrt((1-r_{xw.z}^2)(1-r_{yw.z}^2))
    ... and so on, peeling one covariate at a time.
    """
    controls = list(controls)

    def pcorr(a, b, given):
        if not given:
            return corr[a, b]
        z = given[-1]
        rest = given[:-1]
        r_ab = pcorr(a, b, rest)
        r_az = pcorr(a, z, rest)
        r_bz = pcorr(b, z, rest)
        return _first_order_pcc(r_ab, r_az, r_bz)

    return pcorr(x, y, controls)


def spearman_partial_corr(df, x, y, covars, method="matrix"):
    """Spearman partial correlation of columns x and y in `df`, controlling for
    the columns in `covars`.

    Rank-transforms every involved column, then computes the partial
    correlation on the Pearson correlations of the ranks.

    Parameters
    ----------
    df : pandas.DataFrame
    x, y : str
        Column names for the two variables of interest.
    covars : list of str
        Column names to control for.
    method : {"matrix", "recursive"}
        Two independent estimators; they must agree (see tests).

    Returns
    -------
    float
    """
    cols = [x, y] + list(covars)
    ranks = df[cols].rank().to_numpy()
    corr = _pearson_corr_matrix(ranks)
    control_idx = list(range(2, 2 + len(covars)))
    if method == "matrix":
        return partial_corr_from_matrix(corr, 0, 1, control_idx)
    elif method == "recursive":
        return partial_corr_recursive(corr, 0, 1, control_idx)
    raise ValueError(f"unknown method: {method!r}")


def bootstrap_partial_corr(df, x, y, covars, n_boot=1000, rng=None,
                           method="matrix"):
    """Bootstrap the Spearman partial correlation.

    Returns
    -------
    (point, lo, hi) : float
        Point estimate on the full sample, and the 16th/84th percentiles of the
        bootstrap distribution (1-sigma equivalent).
    """
    if rng is None:
        rng = np.random.default_rng()
    point = spearman_partial_corr(df, x, y, covars, method=method)
    n = len(df)
    vals = np.empty(n_boot)
    cols = [x, y] + list(covars)
    sub = df[cols].reset_index(drop=True)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = spearman_partial_corr(sub.iloc[idx], x, y, covars, method=method)
    lo, hi = np.percentile(vals, [16.0, 84.0])
    return point, lo, hi
