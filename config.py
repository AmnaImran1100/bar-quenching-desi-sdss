"""Central configuration for the bar-quenching-desi-sdss project.

Imported by every script. Contains only constants (plus directory creation):
paths, sample cuts and analysis thresholds, each with a short justification.
"""

from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
for d in (RAW, FIGURES, RESULTS):
    d.mkdir(parents=True, exist_ok=True)


def rel(path):
    """Path relative to the repository root, for logging.

    Log files are committed to the repository, so they must never contain a
    machine-specific absolute path. Every script prints paths through this
    helper.
    """
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


GZ_DESI_FILE = RAW / "gz_desi_deep_learning_catalog_friendly.parquet"
GZ_EXTERNAL_FILE = RAW / "external_catalog.parquet"   # redshift, row-matched to morphology; PRE-FILTER ONLY
MPAJHU_EXTRA = RAW / "galSpecExtra-dr8.fits"
MPAJHU_INFO = RAW / "galSpecInfo-dr8.fits"
MERGED = DATA / "merged_catalog.parquet"

# ---- Sample cuts (main sample) ----
Z_MIN, Z_MAX = 0.02, 0.15
LOGMASS_MIN = 9.5
MATCH_RADIUS_ARCSEC = 1.5

# ---- External redshift pre-filter (PERFORMANCE ONLY; science z = MPA-JHU z) ----
# The external catalog's `redshift` column is photo-z DOMINATED: spec_z is
# present for only ~13% of rows, so `redshift` = spec_z where available else
# photo_z. Verified on the real file (01/inspection):
#   - 7.33% of rows have NaN redshift (but may still have an SDSS spectrum);
#   - a tight upper cut (the old 0.16) would discard real low-z galaxies whose
#     photo-z scattered high.
# A finite auxiliary-redshift cut can reject real counterparts with erroneous
# photo-z. The release rebuild therefore matches the full morphology catalogue.
# The optional pre-filter is retained only for explicitly labelled experiments;
# it has no guaranteed completeness, even with a generous margin and NaNs kept.
# (Supersedes an earlier PREFILTER_Z=(0.01,0.16) from before the column was
#  known to be photo-z rather than spectroscopic.)
EXTERNAL_Z_COL = "redshift"            # coalesced z (spec_z where available, else photo_z)
USE_EXTERNAL_Z_PREFILTER = False
PREFILTER_Z_MAX = 0.25                 # keep galaxies with redshift < this ...
PREFILTER_KEEP_NAN = True              # ... OR with NaN redshift (never drop on a missing photo-z)

# ---- Low-mass supplementary slice (Liu & Zhou enhancement regime) ----
LOWZ_SLICE = dict(z_min=0.02, z_max=0.05, logmass_min=9.0, logmass_max=9.5)

# ---- Selection schemes ----
# 'comparison'  = Liu & Zhou 2026 / Geron 2021 (FIDUCIAL)
# 'conventional'= 0.5 thresholds (robustness)
# NOTE: the two-scheme comparison is an edge-on-gate robustness check only.
# Verified on merged_catalog.parquet by src/08_detectability_overlap.py
# (results/08_detectability_overlap.txt):
#   1. of the 102,239 rows carrying all three bar votes, all have
#      f_strong+f_weak+f_no = 1 to within 1.7e-7, so the two barred-class masks
#      (f_no < 0.5) and (f_strong+f_weak > 0.5) disagree on 0 rows; the bar
#      rules are mathematically identical;
#   2. the friendly catalogue only gives bar/bulge votes on the featured branch,
#      so usable votes already imply featured > 0.5 (minimum over labelled
#      rows = 0.510), and the comparison scheme's 0.27 threshold has no effect;
#   3. the edge-on answers also sum to 1, so edge-on_no >= 0.68 implies
#      edge-on_yes <= 0.32 < 0.5, and the comparison edge-on gate is stricter.
# Hence 'comparison' is a strict subset of 'conventional' (95,227 of 95,447;
# Jaccard 99.77%), the only operative difference is the edge-on threshold, and
# this comparison does not probe the featured-gate selection bias.
SCHEMES = {
    "comparison": dict(
        disc_min_featured=0.27,        # applied as p_featured-or-disk >= 0.27
        faceon_min_edgeon_no=0.68,     # p_edge-on_no       >= 0.68
        bar_rule="no_lt_half",         # barred iff bar_no_fraction < 0.5; strong iff f_strong > f_weak
    ),
    "conventional": dict(
        disc_min_featured=0.50,        # applied as p_featured-or-disk > 0.50
                                       # (strict; 0 rows sit exactly on 0.50)
        faceon_max_edgeon_yes=0.50,
        bar_rule="sum_gt_half",        # barred iff f_strong+f_weak > 0.5; unbarred iff f_no > 0.5; else ambiguous
    ),
}

# ---- Quenching definitions ----
# The fiducial definition is the Delta SFR cut below (following Bluck et al.
# 2020), with the MS ridge fitted to this sample; `quenched_dsfr` is the target
# column in stages 04 and 05.
DSFR_QUENCHED = -1.1                   # quenched:  Delta SFR < -1.1 dex
DSFR_GV_HI = -0.5                      # green valley: -1.1 <= dSFR < -0.5 (excluded from RF training)
MS_FIT_SSFR_MIN = -10.5                # star-forming selection for the MS ridge fit
MS_FIT_MASS_RANGE = (9.0, 10.8)        # logM* range for the ridge fit on binned medians,
                                       # where star-forming galaxies are numerous; the
                                       # fitted line is compared with Renzini & Peng (2015)
SSFR_QUENCHED_CUT = -11.0              # robustness flat cut (also run -10.5, -11.5)
SSFR_QUENCHED_VARIANTS = (-10.5, -11.0, -11.5)   # flat-sSFR quenched cuts for the
                                       # quenched-definition robustness check (Method B);
                                       # log sSFR_tot < cut => quenched.

# ---- sigma_c quality cuts (galSpecInfo) ----
SIGMA_MIN, SIGMA_MAX = 70.0, 420.0     # km/s; SDSS resolution floor / pipeline ceiling
SIGMA_MIN_SN = 10.0                    # median spectrum S/N, applied as sn > 10
                                       # (strict; 0 rows sit exactly on 10.0).
                                       # Robustness variant at 5.0.

# ---- ML (Piotrowska+2022 protocol) ----
RANDOM_STATE = 42
N_TREES = 200
MAX_FEATURES = None                    # all features considered at every split
N_RF_REPEATS = 200                     # Piotrowska et al. (2022) used 500
TEST_SIZE = 0.5
AUC_OVERFIT_TOL = 0.01                 # require AUC_train - AUC_test <= this when tuning min_samples_leaf
LEAF_GRID = [20, 50, 100, 150, 250, 400]

# ---- Controlled binning ----
N_MASS_BINS = 6
N_BULGE_BINS = 5
MIN_PER_CLASS_PER_CELL = 30            # >=30 barred AND >=30 unbarred

# ---- Redshift control (FIDUCIAL for Method B / D) ----
# Bar detectability in GZ DESI declines steeply with z (f_strong at fixed
# logM 10.0-10.5 falls from 14.5% at z=0.02-0.05 to 2.6% at z=0.11-0.15; see
# results/08_detectability_overlap.txt), and quenched galaxies sit at lower z
# than star-forming ones at fixed mass. z is therefore controlled alongside
# mass and bulge in the fiducial measurement.
# Method B recomputes the (mass, bulge) quantile grid WITHIN each z slice (the
# mass distribution shifts with z) and inverse-variance-weights every valid cell
# across all slices, so each cell restricts mass, bulge AND z to finite ranges.
Z_SLICES = [(0.02, 0.06), (0.06, 0.10), (0.10, 0.15)]

# ---- Bulge prominence weights ----
BULGE_WEIGHTS = {"none": 0, "small": 1, "moderate": 2, "large": 3, "dominant": 4}
