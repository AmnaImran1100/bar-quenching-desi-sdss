"""02_crossmatch.py: sky-match GZ DESI to MPA-JHU, apply cuts, derive columns, save.

An incorrect join would affect every later result, so each step prints its
surviving count and the main decisions are stated here:

  * We match from MPA-JHU galaxies (every analysis galaxy needs a spectrum)
    into the larger GZ morphology catalogue.
  * Non-galaxy and placeholder MPA rows are dropped before the match, so their
    placeholder coordinates cannot create false matches.
  * Repeat spectra of one object match the same GZ dr8_id; we keep the
    highest-S/N spectrum per dr8_id.
  * The mass floor for saved rows is 9.0, not 9.5: the main sample (>9.5) and
    the Liu & Zhou low-mass slice (9.0-9.5) are flags, so the low-mass regime
    is kept.
  * Bulge prominence is the normalised vote-weighted score (fractions need not
    sum exactly to 1 in the friendly catalogue).
  * Sample membership (comparison / conventional scheme) requires the relevant
    votes to exist; a galaxy that cannot be labelled is not in the sample.
  * A final guard asserts that no MPA-JHU -9999 sentinel reaches an analysis
    frame through the fibre columns, which Methods D and E read. Empirically
    none does, but only because such rows also fail the total-sSFR cut, so the
    invariant is checked rather than assumed.

Nothing scientific is hard-dropped beyond physical invalidity; schemes, AGN-clean
and sigma-clean tiers are boolean columns so one merged file serves every run.

Reads : raw catalogs (via the verified loaders in 01_load_data.py) + external z.
Writes: data/merged_catalog.parquet, results/02_crossmatch_log.txt

Run:  py src/02_crossmatch.py
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# --- import the *verified* loaders from 01 (module name starts with a digit) ---
def _import_by_path(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_L01 = _import_by_path("load_data01", Path(__file__).resolve().parent / "01_load_data.py")

# GZ morphology columns we carry into the merged table (from 01's verified list).
GZ_MORPH_COLS = _L01.GZ_MORPH_COLS

# Short, physics-readable aliases for the vote fractions we key definitions on.
FEATURED = "smooth-or-featured_featured-or-disk_fraction"
EDGEON_YES = "disk-edge-on_yes_fraction"
EDGEON_NO = "disk-edge-on_no_fraction"
BAR_STRONG = "bar_strong_fraction"
BAR_WEAK = "bar_weak_fraction"
BAR_NO = "bar_no_fraction"
SPIRAL = "has-spiral-arms_yes_fraction"
BULGE_F = {k: f"bulge-size_{k}_fraction" for k in
           ("none", "small", "moderate", "large", "dominant")}


# ---------------------------------------------------------------------------
# Small logging helper: everything printed is also written to the log file.
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


# ---------------------------------------------------------------------------
# Step 1: GZ morphology + permissive redshift pre-filter
# ---------------------------------------------------------------------------

def load_gz_prefiltered():
    """Load GZ science columns and apply the PERFORMANCE-only redshift pre-filter.

    The external catalog is row-aligned to the morphology catalog by dr8_id
    (verified: 0 mismatches), so the pre-filter mask is applied positionally.
    Keep NaN redshift and everything below PREFILTER_Z_MAX; science z comes from
    MPA-JHU after the match, never from here.
    """
    LOG("=" * 72)
    LOG("STEP 1  Load GZ DESI morphology + redshift pre-filter")
    LOG("=" * 72)

    gz = _L01.load_gz_desi()
    n0 = len(gz)
    LOG(f"  GZ DESI rows loaded ............................ {n0:>12,}")

    if not config.USE_EXTERNAL_Z_PREFILTER:
        LOG("  External-redshift pre-filter disabled: match the full catalogue.")
        LOG("")
        return gz

    ext = pd.read_parquet(config.GZ_EXTERNAL_FILE,
                          columns=["dr8_id", config.EXTERNAL_Z_COL])
    if len(ext) != n0 or not (ext["dr8_id"].values == gz["dr8_id"].values).all():
        raise ValueError("external catalog is NOT row-aligned to GZ by dr8_id; "
                         "cannot apply the pre-filter positionally.")
    z = pd.to_numeric(ext[config.EXTERNAL_Z_COL], errors="coerce").to_numpy(float)
    finite = np.isfinite(z)
    keep = (finite & (z > 0.0) & (z < config.PREFILTER_Z_MAX))
    if config.PREFILTER_KEEP_NAN:
        keep = keep | (~finite)

    gz = gz.loc[keep].reset_index(drop=True)
    LOG(f"  pre-filter: keep NaN={config.PREFILTER_KEEP_NAN}, "
        f"z<{config.PREFILTER_Z_MAX}")
    LOG(f"  GZ rows after pre-filter ....................... {len(gz):>12,}  "
        f"({len(gz)/n0:.1%} of parent)")
    LOG("")
    return gz


# ---------------------------------------------------------------------------
# Step 2: MPA-JHU galaxies only (drop placeholders / stars / QSOs)
# ---------------------------------------------------------------------------

def load_mpa_galaxies():
    """Load MPA-JHU and keep only real GALAXY spectra.

    The placeholder rows (spectrotype = nan, ra=0, plate/mjd/fiber=-1) and the
    STAR/QSO spectra are removed before matching so their coordinates cannot
    create false matches in GZ.
    """
    LOG("=" * 72)
    LOG("STEP 2  Load MPA-JHU, keep GALAXY spectra only")
    LOG("=" * 72)

    mpa = _L01.load_mpajhu()
    n0 = len(mpa)
    LOG(f"  MPA-JHU spectra loaded ......................... {n0:>12,}")

    stype = mpa["spectrotype"].astype(str).str.strip().str.upper()
    gal = mpa.loc[stype == "GALAXY"].reset_index(drop=True)
    LOG(f"  spectrotype == GALAXY .......................... {len(gal):>12,}")
    LOG(f"    (dropped: {n0 - len(gal):,} placeholder / STAR / QSO rows)")

    # A few GALAXY rows still carry sentinel coordinates (ra/dec = -9999): they
    # have no sky position and cannot be matched. Drop them before SkyCoord.
    ra = pd.to_numeric(gal["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(gal["dec"], errors="coerce").to_numpy(float)
    valid_coord = (np.isfinite(ra) & np.isfinite(dec) &
                   (ra >= 0.0) & (ra <= 360.0) & (dec >= -90.0) & (dec <= 90.0))
    n_bad = int((~valid_coord).sum())
    gal = gal.loc[valid_coord].reset_index(drop=True)
    LOG(f"  with valid sky coordinates ..................... {len(gal):>12,}"
        f"   (dropped {n_bad:,} sentinel/invalid ra|dec)")
    LOG("")
    return gal


# ---------------------------------------------------------------------------
# Step 3: sky cross-match MPA(galaxies) -> GZ, dedup many-to-one
# ---------------------------------------------------------------------------

def crossmatch(gz, mpa):
    """Match each MPA galaxy to its nearest GZ source; keep < MATCH_RADIUS.

    Then collapse many-to-one (several spectra -> one dr8_id) by keeping the
    highest-S/N spectrum, which also handles repeat observations.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    LOG("=" * 72)
    LOG("STEP 3  Sky cross-match  (MPA galaxies -> GZ, "
        f"r < {config.MATCH_RADIUS_ARCSEC}\")")
    LOG("=" * 72)

    # Promote before angular arithmetic: FITS coordinates may be float32.
    c_gz = SkyCoord(ra=gz["ra"].to_numpy(float) * u.deg,
                    dec=gz["dec"].to_numpy(float) * u.deg)
    c_mpa = SkyCoord(ra=mpa["ra"].to_numpy(float) * u.deg,
                     dec=mpa["dec"].to_numpy(float) * u.deg)
    idx, sep2d, _ = c_mpa.match_to_catalog_sky(c_gz)
    sep = sep2d.arcsec

    mpa = mpa.copy()
    mpa["sep_arcsec"] = sep
    mpa["_gz_row"] = idx

    within = mpa["sep_arcsec"] < config.MATCH_RADIUS_ARCSEC
    LOG(f"  MPA galaxies with a GZ source < {config.MATCH_RADIUS_ARCSEC}\" ... "
        f"{int(within.sum()):>12,}  ({within.mean():.1%})")
    # separation distribution (real matches cluster at small separations).
    for q in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
        LOG(f"    sep < {q:>4.2f}\": {int((sep < q).sum()):>10,}")
    matched = mpa.loc[within].reset_index(drop=True)

    # attach GZ morphology (positional take from gz by matched row index)
    gz_take = gz.iloc[matched["_gz_row"].to_numpy()].reset_index(drop=True)
    matched["dr8_id"] = gz_take["dr8_id"].to_numpy()
    matched["gz_ra"] = gz_take["ra"].to_numpy()
    matched["gz_dec"] = gz_take["dec"].to_numpy()
    for col in GZ_MORPH_COLS:
        matched[col] = gz_take[col].to_numpy()

    # many-to-one: keep highest sn_median per GZ galaxy
    n_pre = len(matched)
    dup_groups = matched["dr8_id"].duplicated(keep=False)
    n_multi = int(matched.loc[dup_groups, "dr8_id"].nunique())
    max_mult = int(matched["dr8_id"].value_counts().max()) if n_pre else 0
    matched = (matched.sort_values("sn_median", ascending=False, kind="stable")
               .drop_duplicates("dr8_id", keep="first")
               .reset_index(drop=True))
    LOG(f"  matched rows before dedup ...................... {n_pre:>12,}")
    LOG(f"  GZ galaxies claimed by >1 spectrum ............. {n_multi:>12,}  "
        f"(max multiplicity {max_mult})")
    LOG(f"  unique matched galaxies (kept highest S/N) ..... {len(matched):>12,}")
    matched = matched.drop(columns=["_gz_row"])
    LOG("")
    return matched


# ---------------------------------------------------------------------------
# Step 4: physical validity cuts (scheme-independent) + membership flags
# ---------------------------------------------------------------------------

def physical_cuts(df):
    """Drop physically invalid rows; flag main-sample and low-z-slice membership.

    Row-level survival requires: valid (non-sentinel) total mass, SFR and sSFR;
    a reliable redshift (z_warning == 0) inside [Z_MIN, Z_MAX]; and logM > 9.0
    (the LOWER floor, to preserve the low-mass slice). in_main_sample adds the
    9.5 floor; in_lowz_slice picks out the L&Z enhancement regime.
    """
    LOG("=" * 72)
    LOG("STEP 4  Physical validity cuts + sample-membership flags")
    LOG("=" * 72)
    n0 = len(df)

    m = pd.to_numeric(df["lgm_tot_p50"], errors="coerce").to_numpy(float)
    sfr = pd.to_numeric(df["sfr_tot_p50"], errors="coerce").to_numpy(float)
    ssfr = pd.to_numeric(df["specsfr_tot_p50"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(float)
    zw = pd.to_numeric(df["z_warning"], errors="coerce").to_numpy(float)

    valid_mass = _L01.valid_measurement(m, "lgm_tot_p50") & (m > 0.0)
    valid_sfr = _L01.valid_measurement(sfr, "sfr_tot_p50")
    valid_ssfr = _L01.valid_measurement(ssfr, "specsfr_tot_p50")
    LOG(f"  valid log SFR=-1 rows retained by validity rule: {int((valid_sfr & (sfr == -1)).sum())}")
    good_z = np.isfinite(z) & (zw == 0) & (z > config.Z_MIN) & (z < config.Z_MAX)
    mass_floor = m > config.LOWZ_SLICE["logmass_min"]  # 9.0

    def step(mask, label):
        LOG(f"  {label:44s} {int(mask.sum()):>12,}")

    step(valid_mass, "valid total mass")
    step(valid_mass & valid_sfr & valid_ssfr, "  + valid SFR & sSFR")
    step(valid_mass & valid_sfr & valid_ssfr & good_z,
         "  + z_warning==0 & 0.02<z<0.15")
    keep = valid_mass & valid_sfr & valid_ssfr & good_z & mass_floor
    step(keep, "  + logM > 9.0  (row-level floor)")

    df = df.loc[keep].reset_index(drop=True)
    m = m[keep]; z = z[keep]
    df["in_main_sample"] = m > config.LOGMASS_MIN
    lz = config.LOWZ_SLICE
    df["in_lowz_slice"] = ((z > lz["z_min"]) & (z < lz["z_max"]) &
                           (m > lz["logmass_min"]) & (m < lz["logmass_max"]))
    LOG(f"  --> rows retained .............................. {len(df):>12,}  "
        f"(from {n0:,})")
    LOG(f"      in_main_sample (logM>9.5) ................. "
        f"{int(df['in_main_sample'].sum()):>12,}")
    LOG(f"      in_lowz_slice (9.0-9.5, z<0.05) ........... "
        f"{int(df['in_lowz_slice'].sum()):>12,}")
    LOG("")
    return df


# ---------------------------------------------------------------------------
# Step 5: derived morphology columns (bulge prominence, bar labels, B_vol)
# ---------------------------------------------------------------------------

def _votes_present(df):
    edgeon = df[EDGEON_NO].notna() & df[EDGEON_YES].notna()
    bar = df[[BAR_NO, BAR_STRONG, BAR_WEAK]].notna().all(axis=1)
    bulge = df[[BULGE_F[k] for k in BULGE_F]].notna().all(axis=1)
    return edgeon, bar, bulge


def add_morphology(df):
    """Bulge prominence (normalised), continuous bar strength B_vol, and the
    per-scheme bar class. Requires the relevant votes to be present."""
    LOG("=" * 72)
    LOG("STEP 5  Derived morphology: B_prom, B_vol, bar labels")
    LOG("=" * 72)

    # --- normalised bulge prominence ---
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for k, w in config.BULGE_WEIGHTS.items():
        f = pd.to_numeric(df[BULGE_F[k]], errors="coerce").to_numpy(float)
        f = np.where(np.isfinite(f), f, 0.0)
        num += w * f
        den += f
    with np.errstate(invalid="ignore", divide="ignore"):
        df["bulge_prominence"] = np.where(den > 0, num / den, np.nan)

    # --- continuous bar strength B_vol = f_strong + 0.5 f_weak (this work; cf. Geron 2021) ---
    fs = pd.to_numeric(df[BAR_STRONG], errors="coerce").to_numpy(float)
    fw = pd.to_numeric(df[BAR_WEAK], errors="coerce").to_numpy(float)
    fn = pd.to_numeric(df[BAR_NO], errors="coerce").to_numpy(float)
    df["bar_strength"] = fs + 0.5 * fw   # NaN where votes absent

    # --- per-scheme categorical bar class ---
    # comparison (Liu & Zhou / Geron 2021 thresholds): barred iff bar_no < 0.5;
    #   strong iff f_strong > f_weak else weak. No ambiguous class. (Geron 2021
    #   assign exact ties to strong; no barred row has f_strong == f_weak here.)
    comp = np.full(len(df), "unclassified", dtype=object)
    barred_c = fn < 0.5
    strong_c = barred_c & (fs > fw)
    comp[barred_c & ~strong_c] = "weak"
    comp[strong_c] = "strong"
    comp[~barred_c & np.isfinite(fn)] = "unbarred"
    df["bar_class_comparison"] = comp

    # conventional (0.5 thresholds): unbarred iff f_no>0.5; barred iff
    #   f_strong+f_weak>0.5; else ambiguous. (Non-overlapping by construction.)
    conv = np.full(len(df), "unclassified", dtype=object)
    unbarred_v = fn > 0.5
    barred_v = (~unbarred_v) & ((fs + fw) > 0.5)
    ambiguous_v = np.isfinite(fn) & ~unbarred_v & ~barred_v
    conv[unbarred_v] = "unbarred"
    conv[barred_v & (fs > fw)] = "strong"
    conv[barred_v & (fs <= fw)] = "weak"
    conv[ambiguous_v] = "ambiguous"
    df["bar_class_conventional"] = conv

    for scheme, col in (("comparison", "bar_class_comparison"),
                        ("conventional", "bar_class_conventional")):
        vc = df[col].value_counts(dropna=False)
        LOG(f"  bar classes ({scheme}):")
        for k, v in vc.items():
            LOG(f"      {str(k):>14s}: {v:>10,}")
    LOG(f"  bulge_prominence: median={np.nanmedian(df['bulge_prominence']):.3f}, "
        f"NaN={df['bulge_prominence'].isna().mean():.2%}")
    LOG("")
    return df


# ---------------------------------------------------------------------------
# Step 6: selection-scheme membership flags
# ---------------------------------------------------------------------------

def add_scheme_flags(df):
    """in_sample_comparison / in_sample_conventional. A galaxy is in the sample
    only if it passes the disc+face-on gate AND has the bar & bulge votes needed
    to be labelled at all."""
    LOG("=" * 72)
    LOG("STEP 6  Selection-scheme membership flags")
    LOG("=" * 72)

    feat = pd.to_numeric(df[FEATURED], errors="coerce").to_numpy(float)
    eno = pd.to_numeric(df[EDGEON_NO], errors="coerce").to_numpy(float)
    eyes = pd.to_numeric(df[EDGEON_YES], errors="coerce").to_numpy(float)
    edgeon_ok, bar_ok, bulge_ok = _votes_present(df)
    labelled = (bar_ok & bulge_ok).to_numpy()

    comp = config.SCHEMES["comparison"]
    conv = config.SCHEMES["conventional"]

    # Operator asymmetry (>= vs >) is deliberate and follows the source
    # definitions: Geron (2021) / Liu & Zhou state their cuts as
    # p_featured-or-disk >= 0.27 and p_not-edge-on >= 0.68 (inclusive), whereas
    # the conventional scheme is the usual "majority vote", i.e. strictly more
    # than half. Vote fractions are small-denominator rationals so exact
    # equality does occur; keeping each scheme's own convention means neither
    # is shifted by one vote. In practice the choice has no effect here:
    # every galaxy carrying the bar and bulge votes needed for a label already
    # has featured > 0.5 (see src/08_detectability_overlap.py), so the two
    # featured thresholds select identically regardless of the operator.
    gate_comp = (feat >= comp["disc_min_featured"]) & (eno >= comp["faceon_min_edgeon_no"])
    gate_conv = (feat > conv["disc_min_featured"]) & (eyes < conv["faceon_max_edgeon_yes"])

    df["in_sample_comparison"] = gate_comp & edgeon_ok.to_numpy() & labelled
    df["in_sample_conventional"] = gate_conv & edgeon_ok.to_numpy() & labelled

    for scheme, gate, flag in (
            ("comparison", gate_comp, "in_sample_comparison"),
            ("conventional", gate_conv, "in_sample_conventional")):
        n_gate = int((gate & edgeon_ok.to_numpy()).sum())
        n_final = int(df[flag].sum())
        LOG(f"  {scheme}: passing gate={n_gate:,}  "
            f"with usable bar+bulge votes={n_final:,}  "
            f"(lost {n_gate - n_final:,} to missing votes)")
        # in-main-sample breakdown, and bar composition (the L&Z sanity check)
        sub = df[df[flag] & df["in_main_sample"]]
        LOG(f"     in main sample (logM>9.5): {len(sub):,}")
        if len(sub):
            bcol = f"bar_class_{scheme}"
            vc = sub[bcol].value_counts()
            for k in ("unbarred", "weak", "strong", "ambiguous"):
                if k in vc:
                    LOG(f"        {k:>10s}: {vc[k]:>8,}  ({vc[k]/len(sub):6.2%})")
    LOG("")
    return df


# ---------------------------------------------------------------------------
# Step 7: star-forming main sequence, Delta SFR, quenched labels
# ---------------------------------------------------------------------------

def fit_main_sequence(df):
    """Fit the SF main-sequence ridge on star-forming galaxies in the sample,
    by median-binned OLS, and sanity-check against Renzini & Peng (2015).

    Returns (alpha, beta) for log SFR_MS = alpha*logM + beta.
    """
    LOG("=" * 72)
    LOG("STEP 7  Main-sequence fit -> Delta SFR -> quenched labels")
    LOG("=" * 72)

    m = pd.to_numeric(df["lgm_tot_p50"], errors="coerce").to_numpy(float)
    sfr = pd.to_numeric(df["sfr_tot_p50"], errors="coerce").to_numpy(float)
    ssfr = pd.to_numeric(df["specsfr_tot_p50"], errors="coerce").to_numpy(float)

    lo, hi = config.MS_FIT_MASS_RANGE
    sf = (ssfr > config.MS_FIT_SSFR_MIN) & (m > lo) & (m < hi)
    LOG(f"  star-forming galaxies for the ridge "
        f"(logsSFR>{config.MS_FIT_SSFR_MIN}, {lo}<logM<{hi}): {int(sf.sum()):,}")

    edges = np.arange(lo, hi + 1e-9, 0.1)
    centers, med_sfr = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        inb = sf & (m >= a) & (m < b)
        if inb.sum() >= 20:
            centers.append(0.5 * (a + b))
            med_sfr.append(np.median(sfr[inb]))
    centers = np.asarray(centers); med_sfr = np.asarray(med_sfr)
    if len(centers) < 3:
        raise RuntimeError("too few populated mass bins to fit the main sequence")
    # Unweighted OLS on the BINNED MEDIANS (one point per populated 0.1-dex
    # mass bin), not on individual galaxies: the median is robust to the
    # low-SFR tail that survives the log sSFR cut, and equal weight per bin
    # stops the crowded 10.0-10.5 bins from setting the slope on their own.
    alpha, beta = np.polyfit(centers, med_sfr, 1)
    LOG(f"  populated 0.1-dex mass bins used in the fit: {len(centers)} "
        f"(>= 20 star-forming galaxies each)")
    LOG(f"  fitted MS ridge:  log SFR_MS = {alpha:.3f} * logM + ({beta:.3f})")
    LOG(f"  full-precision coefficients: alpha = {alpha!r}, beta = {beta!r}")
    LOG(f"  Renzini & Peng 2015 reference:            0.760 * logM + (-7.640)")
    rp = 0.76 * centers - 7.64
    LOG(f"  mean |fit - R&P| over fitted bins: "
        f"{np.mean(np.abs((alpha*centers+beta) - rp)):.3f} dex")
    if not (0.4 < alpha < 1.1):
        LOG(f"  ** WARNING: MS slope {alpha:.3f} is far from ~0.76 -- investigate.")

    df["delta_sfr"] = sfr - (alpha * m + beta)
    df["quenched_dsfr"] = df["delta_sfr"] < config.DSFR_QUENCHED
    df["green_valley"] = (df["delta_sfr"] >= config.DSFR_QUENCHED) & \
                         (df["delta_sfr"] < config.DSFR_GV_HI)
    df["quenched_ssfr"] = ssfr < config.SSFR_QUENCHED_CUT

    main = df[df["in_main_sample"]]
    LOG(f"  quenched fraction (dSFR<{config.DSFR_QUENCHED}) in main sample: "
        f"{main['quenched_dsfr'].mean():.3f}")
    LOG(f"  green-valley fraction in main sample: {main['green_valley'].mean():.3f}")
    LOG(f"  quenched fraction (logsSFR<{config.SSFR_QUENCHED_CUT}) main sample: "
        f"{main['quenched_ssfr'].mean():.3f}")
    LOG("")
    return df


# ---------------------------------------------------------------------------
# Step 8: supplementary-tier flags (sigma-clean, AGN-clean)
# ---------------------------------------------------------------------------

def add_tier_flags(df):
    LOG("=" * 72)
    LOG("STEP 8  Supplementary-tier flags (sigma-clean, AGN-clean)")
    LOG("=" * 72)

    v = pd.to_numeric(df["v_disp"], errors="coerce").to_numpy(float)
    verr = pd.to_numeric(df["v_disp_err"], errors="coerce").to_numpy(float)
    sn = pd.to_numeric(df["sn_median"], errors="coerce").to_numpy(float)
    rel = pd.to_numeric(df["reliable"], errors="coerce").to_numpy(float)
    bpt = pd.to_numeric(df["bptclass"], errors="coerce").to_numpy(float)

    df["sigma_clean"] = ((v > config.SIGMA_MIN) & (v < config.SIGMA_MAX) &
                         np.isfinite(verr) & (verr > 0) &
                         np.isfinite(sn) & (sn > config.SIGMA_MIN_SN) & (rel == 1))
    df["agn_clean"] = np.isin(bpt, [1, 2])

    main = df["in_main_sample"].to_numpy()
    LOG(f"  sigma-clean (70<sigma<420, positive finite error, S/N>10, reliable): {int(df['sigma_clean'].sum()):,}"
        f"   (main sample: {int((df['sigma_clean'] & main).sum()):,})")
    LOG(f"  AGN-clean (bptclass in 1,2): {int(df['agn_clean'].sum()):,}"
        f"   (main sample: {int((df['agn_clean'] & main).sum()):,})")
    LOG("")
    return df


# ---------------------------------------------------------------------------
# Step 9: fibre-column sentinel guard (Methods D and E)
# ---------------------------------------------------------------------------

# MPA-JHU marks undefined fibre quantities with -9999 (not NaN). Note that -1,
# the other MPA-JHU sentinel, is a PHYSICALLY VALID value for sfr_fib_p50
# (log SFR = -1 is 0.1 Msun/yr), so it must not be treated as a sentinel here.
FIBRE_COLS = ["specsfr_fib_p50", "sfr_fib_p50", "lgm_fib_p50"]
FIBRE_SENTINEL = -9999.0


def assert_fibre_clean(df):
    """Raise an error if any -9999 sentinel or non-finite value reaches an analysis
    frame through the fibre columns.

    Why this exists: ~29.4-29.5% of the raw fibre columns are -9999 sentinels
    and nothing in the pipeline cleans them explicitly. Empirically none of them
    survives, because rows with an undefined fibre sSFR also fail the
    total-sSFR validity cut in Step 4, but that protection is incidental.
    A single -9999 would distort Method D's mean contrast and Method E's
    fibre-minus-total concentration, so the condition is asserted here.

    The guard covers exactly the frames 05 builds: main-sample discs under each
    scheme, and the low-z enhancement slice. Rows outside every frame may still
    carry sentinels; they are reported but do not fail the check, because no
    analysis ever reads them.
    """
    LOG("=" * 72)
    LOG("STEP 9  Fibre-column sentinel guard (Methods D / E)")
    LOG("=" * 72)

    frames = {}
    for scheme in ("comparison", "conventional"):
        frames[f"main:{scheme}"] = (df[f"in_sample_{scheme}"].to_numpy(bool)
                                    & df["in_main_sample"].to_numpy(bool))
        frames[f"lowz:{scheme}"] = (df[f"in_sample_{scheme}"].to_numpy(bool)
                                    & df["in_lowz_slice"].to_numpy(bool))

    failures = []
    for name, mask in frames.items():
        for col in FIBRE_COLS:
            x = pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(float)
            n_sent = int((x == FIBRE_SENTINEL).sum())
            n_bad = int((~np.isfinite(x)).sum())
            LOG(f"  {name:22s} {col:18s} n={x.size:>7,}  "
                f"sentinels={n_sent}  non-finite={n_bad}  "
                f"min={np.nanmin(x) if x.size else float('nan'):.3f}")
            if n_sent or n_bad:
                failures.append(f"{name}/{col}: {n_sent} sentinel, {n_bad} non-finite")

    # Sentinels that exist in the file but never enter an analysis frame.
    any_frame = np.zeros(len(df), dtype=bool)
    for mask in frames.values():
        any_frame |= mask
    outside = 0
    for col in FIBRE_COLS:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        outside += int(((x == FIBRE_SENTINEL) & ~any_frame).sum())
    LOG(f"  sentinel fibre values OUTSIDE every analysis frame: {outside} "
        f"(harmless -- never read)")

    if failures:
        raise ValueError(
            "Fibre sentinel guard FAILED -- a -9999 or non-finite fibre value "
            "reached an analysis frame. Methods D/E would be corrupted:\n  "
            + "\n  ".join(failures))
    LOG("  GUARD PASSED: no sentinel or non-finite fibre value in any analysis "
        "frame.")
    LOG("")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    LOG("02_crossmatch.py -- GZ DESI x MPA-JHU merge\n")

    gz = load_gz_prefiltered()
    mpa = load_mpa_galaxies()
    df = crossmatch(gz, mpa)
    df = physical_cuts(df)
    df = add_morphology(df)
    df = add_scheme_flags(df)
    df = fit_main_sequence(df)
    df = add_tier_flags(df)
    df = assert_fibre_clean(df)

    df.to_parquet(config.MERGED, index=False)
    LOG("=" * 72)
    LOG(f"SAVED merged catalog -> {config.rel(config.MERGED)}")
    LOG(f"  final rows: {len(df):,}    columns: {df.shape[1]}")
    LOG("=" * 72)

    log_path = config.RESULTS / "02_crossmatch_log.txt"
    LOG.save(log_path)
    print(f"\nWrote audit -> {config.rel(log_path)}")


if __name__ == "__main__":
    main()
