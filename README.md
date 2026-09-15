# bar-quenching-desi-sdss

[![DOI](https://zenodo.org/badge/1371783912.svg)](https://doi.org/10.5281/zenodo.22774684)

**At fixed stellar mass, bulge prominence, and redshift, do barred galaxies have a higher quenched fraction than unbarred galaxies?**
A controlled-variable cross-check of Liu & Zhou (2026), using Galaxy Zoo DESI deep-learning morphology and SDSS MPA-JHU spectroscopic star-formation rates, masses and velocity dispersions.

---

## Summary

Bars can transport gas towards galaxy centres, influence central star formation and contribute to bulge growth. Their association with quenching is difficult to separate from differences in galaxy mass, structure and selection.

This project examines two related questions:

1. **Headline: the quenched-fraction contrast.** Compare barred and unbarred discs within mass–bulge cells and redshift slices, with supplementary predictive and correlation analyses.
2. **Central star formation: the Liu & Zhou comparison.** Compare SDSS fibre SFR and sSFR in overlapping mass ranges, while explicitly accounting for differences in sample matching and measurement methods.

**The strong-bar population has a small positive conditional quenched-fraction excess: +3.69 ± 0.54 percentage points.** These are nominal statistical errors, not total uncertainties. The weak-bar result is borderline. The full-band central-sSFR comparison does not reproduce Liu & Zhou's approximately −0.2 dex suppression.

The results describe associations; they do not identify a causal effect of bars or a quenching timescale. The morphology is shared with Liu & Zhou, and a different spectroscopic pipeline does not guarantee independent errors.

---

## Scientific Background

### What is quenching?

Galaxies form stars at different rates. The star-forming main sequence describes the relation between stellar mass and star-formation rate (SFR) among actively star-forming galaxies. Here, a galaxy is called **quenched** when its SFR lies sufficiently far below that relation. Specific SFR (sSFR) is SFR divided by stellar mass; it distinguishes rapid star formation relative to a galaxy's size from a high SFR simply associated with a large stellar mass.

### What predicts quenching?

Stellar mass, bulge structure and central velocity dispersion are associated with quenching. [Bluck et al. (2020)](https://doi.org/10.1093/mnras/staa2806) and [Piotrowska et al. (2022)](https://doi.org/10.1093/mnras/stab3673) motivate controlling these properties when examining an additional bar association. Velocity dispersion is a proxy for central structure, not a direct black-hole mass measurement.

### Where do bars come in?

Stellar bars can redistribute angular momentum and transport gas towards galaxy centres. Central star formation, bulge growth and gas depletion are therefore possible links between bars and quenching. However, bars also correlate with host-galaxy structure. An uncontrolled difference between barred and unbarred galaxies cannot establish which physical process produced it. The role of bars in secular evolution is reviewed by [Kormendy & Kennicutt (2004)](https://doi.org/10.1146/annurev.astro.42.053102.134024).

[Liu & Zhou (2026)](https://arxiv.org/abs/2605.04537) report enhanced central SFR in low-mass strong-barred galaxies and lower central sSFR at higher masses. Their unbarred controls are matched in stellar mass and colour. Here, we use the same morphology catalogue with SDSS spectroscopy to examine quenching after stratifying mass, bulge prominence and redshift.

### The core question

> At fixed stellar mass, bulge prominence and redshift, do barred galaxies have a higher quenched fraction than unbarred galaxies?

The measurements describe present-day populations. They do not determine whether a bar formed before or after quenching, or whether an unmeasured property influenced both.

---

## Data

| Dataset | Provides | Source | Approximate file size |
|---|---|---|---|
| Galaxy Zoo DESI | Predicted morphology vote fractions for 8.7 million galaxies | [Zenodo 8331338](https://zenodo.org/records/8331338), **v1.0.0** | 659 MB friendly catalogue; 1.62 GB auxiliary catalogue |
| SDSS MPA-JHU DR8 | Mass, SFR/sSFR, redshift, velocity dispersion and BPT class | [SDSS documentation](https://www.sdss4.org/dr17/spectro/galaxy_mpajhu/) | 737 MB across the two FITS files |

**Exact files:**

- Galaxy Zoo DESI friendly morphology catalogue, version 1.0.0: [Zenodo record 8331338](https://zenodo.org/records/8331338). Place `gz_desi_deep_learning_catalog_friendly.parquet` in `data/raw/`. The auxiliary `external_catalog.parquet` is used by the input inventory, but external redshifts do not prefilter the fiducial cross-match. Both files are byte-identical (same MD5 checksums) in the later version 1.0.1 ([Zenodo record 8360385](https://zenodo.org/records/8360385)); SHA-256 checksums of the files used are in `results/raw_input_provenance.json`.
- SDSS MPA-JHU DR8: [galSpecExtra-dr8.fits](https://data.sdss.org/sas/dr8/common/sdss-spectro/redux/galSpecExtra-dr8.fits) and [galSpecInfo-dr8.fits](https://data.sdss.org/sas/dr8/common/sdss-spectro/redux/galSpecInfo-dr8.fits), placed in `data/raw/`. See the [survey documentation](https://www.sdss4.org/dr17/spectro/galaxy_mpajhu/).

The pipeline matches eligible SDSS spectra to the full morphology table with double-precision angular arithmetic, retains separations below 1.5 arcsec, and resolves duplicate morphology counterparts by spectral S/N. This nearest-neighbour association has no calibrated completeness or contamination probability. Science redshifts come from SDSS.

The friendly catalogue already masks downstream questions: the nominal featured gate is inactive in the analysed sample. The conventional scheme primarily tests the edge-on gate, rather than an independent morphology selection. All cuts are specified in [config.py](config.py).

### Key columns

**Galaxy Zoo DESI (vote fractions):**

- `smooth-or-featured_featured-or-disk_fraction` and `disk-edge-on_no_fraction`: disc and orientation gates.
- `bar_strong_fraction`, `bar_weak_fraction`, `bar_no_fraction`: bar classes and the adopted continuous bar score.
- `bulge-size_{none,small,moderate,large,dominant}_fraction`: the bulge control.
- `has-spiral-arms_yes_fraction`: a supplementary forest predictor.

**MPA-JHU:**

- `lgm_tot_p50`, `sfr_tot_p50`, `specsfr_tot_p50`: total stellar mass, SFR and sSFR.
- `lgm_fib_p50`, `sfr_fib_p50`, `specsfr_fib_p50`: fibre-aperture estimates.
- `z`, `ra`, `dec`: spectroscopic redshift and coordinates.
- `v_disp`, `v_disp_err`, `sn_median`, `reliable`: dispersion and measurement-quality fields.
- `bptclass`: emission-line classification.

---

## Method

1. **Method A: random forests.** Class-balanced training, stratified half-sample splits, tuned leaf size and 200 repeats, with Gini and permutation importance plus a random feature. This method omits redshift and provides predictive context. Correlated predictors prevent causal interpretation of importance.
2. **Method B: controlled binning.** Mass–bulge grids rebuilt within three redshift slices, with at least 30 galaxies per class, Wilson/Newcombe intervals, and fixed/random-effects summaries. Alternative quenching definitions and morphology gates provide sensitivity checks.
3. **Method C: partial correlations.** Rank residuals with sequential mass, bulge and redshift controls, plus dispersion on its quality-selected sample. Control levels are compared within the same sample tier, with bootstrap intervals.
4. **Method D: fibre sSFR.** Low-redshift mass-band comparisons and population decomposition. Non-quenched includes the green valley; strict star-forming contrasts are reported separately.
5. **Method E: BPT composition.** Within-class contrasts and selection fractions. Restricting to BPT star-forming objects is not a neutral AGN removal; these comparisons cannot establish or exclude AGN mediation.

**Why several methods?** Forest importance measures predictive reliance; binning compares fractions transparently but coarsely; rank correlations assess continuous trends; fibre measurements and BPT classes describe central activity and population composition. These methods share data and systematic limitations, so agreement is not independent proof of causation.

### Definitions

Mass and SFR logarithms are base 10, with stellar mass in solar masses and SFR in solar masses per year. A difference of one dex is a factor of ten. A percentage-point difference in quenched fraction is distinct from a percentage change relative to another fraction.

- **Main sample:** 0.02 < z < 0.15; log stellar mass > 9.5; valid total mass, SFR and sSFR; `z_warning == 0`.
- **Fiducial comparison discs:** Featured/disc fraction ≥ 0.27 and not-edge-on fraction ≥ 0.68, with usable bar and bulge votes.
- **Conventional disc selection:** Featured/disc fraction > 0.5 and edge-on fraction < 0.5, with usable votes.
- **Bar classes:** Barred if `bar_no_fraction < 0.5`; among barred objects, strong if `f_strong > f_weak`, otherwise weak. Remaining finite no-bar votes give the fiducial unbarred class.
- **Continuous bar score:** `f_strong + 0.5 × f_weak`, an ordinal score adopted in this work.
- **Bulge prominence:** Normalized vote-weighted score with weights 0, 1, 2, 3 and 4 for none, small, moderate, large and dominant bulges.
- **Main-sequence residual, ΔSFR:** `log SFR − log SFR_MS`; the fitted ridge is approximately `log SFR_MS = 0.645518 × log M* − 6.335746`.
- **Quenched:** ΔSFR < −1.1 dex.
- **Green valley:** −1.1 ≤ ΔSFR < −0.5 dex.
- **Strictly star-forming:** ΔSFR ≥ −0.5 dex. Non-quenched includes both these galaxies and the green valley.
- **Alternative quenching cuts:** Total log sSFR < −10.5, −11.0 or −11.5, used as sensitivity checks.
- **Dispersion-quality tier:** 70 < σ < 420 km/s, finite positive dispersion uncertainty, finite spectral S/N > 10 and `reliable == 1`.
- **`agn_clean` tier:** BPT classes 1 or 2: star-forming or low-S/N star-forming. This is a population selection, not a neutral removal of AGN.
- **Low-mass comparison:** 9.0 < log stellar mass < 9.5 and 0.02 < z < 0.05, used for the central-star-formation comparison.

Column-specific missing-value rules retain physically valid log SFR = −1. The merged catalogue also contains auxiliary fields with missing values, so use the documented validity rules before calculating new quantities. See [data/COLUMNS.md](data/COLUMNS.md) for the full data dictionary.

---

## Repository Structure

```text
bar-quenching-desi-sdss/
├── README.md
├── .gitattributes                 # LF line endings, so recorded checksums match a clone
├── CITATION.cff                    # Repository citation metadata
├── LICENSE                        # MIT licence for code
├── LICENSE-DATA                   # CC BY 4.0 for derived products
├── config.py                      # Paths, cuts and analysis settings
├── requirements.txt               # Analysis dependencies
├── requirements-release.txt       # Direct dependency versions used
├── data/
│   ├── raw/                       # Upstream downloads; large files excluded
│   ├── COLUMNS.md                 # Merged-catalogue data dictionary
│   └── merged_catalog.parquet     # Analysis-ready catalogue
├── src/
│   ├── 01_load_data.py            # Inspect and load raw catalogues
│   ├── 02_crossmatch.py           # Sky match, selections and derived columns
│   ├── 03_validation_figures.py   # Sample distributions and diagnostics
│   ├── 04_recover_bulge.py        # Quenching-predictor validation
│   ├── 05_bar_test.py             # Methods A–E
│   ├── 06_result_figures.py       # Main scientific figures
│   ├── 07_cell_balance_check.py   # Within-cell covariate offsets
│   ├── 08_detectability_overlap.py # Redshift trends and selection overlap
│   ├── 09_small_study_and_sf_split.py # Precision and population diagnostics
│   ├── 10_release_tables.py      # Numbers and tables quoted in the manuscript
│   ├── rf_protocol.py            # Shared random-forest procedure
│   ├── stats_utils.py            # Intervals, pooling and correlations
│   ├── provenance.py             # Content hashes and RF cache validation
│   ├── test_stats_utils.py       # Statistical self-checks (synthetic data)
│   └── test_audit_regressions.py # Regression tests for corrected diagnostics
├── figures/                       # Scientific plots in PDF and PNG
└── results/                       # Tables, summaries and input provenance
```

---

## How to Reproduce

Use Python 3.11 or newer and install Python dependencies from `requirements.txt`. The tracked merged catalogue permits starting at stage 03; stages 01–02 require raw files. Run from the repository root:

```sh
git clone https://github.com/AmnaImran1100/bar-quenching-desi-sdss.git
cd bar-quenching-desi-sdss
python -m venv venv
# Linux/macOS: source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Statistical self-checks (no catalogues needed)
cd src && python test_stats_utils.py && python -m unittest test_audit_regressions && cd ..

# Put the four upstream files in data/raw/ to rebuild stages 01–02.
# With the supplied merged catalogue, start at stage 03.
python src/01_load_data.py
python src/02_crossmatch.py
python src/03_validation_figures.py
python src/04_recover_bulge.py
python src/05_bar_test.py
python src/06_result_figures.py
python src/07_cell_balance_check.py
python src/08_detectability_overlap.py
python src/09_small_study_and_sf_split.py
python src/10_release_tables.py   # manuscript numbers -> paper/ (not tracked)
```

All computation runs on CPU. Repeated forests and bootstraps can take many hours. Stage 05 reuses the Method-A forest results only when the catalogue, relevant code, configuration, exact package versions (`requirements-release.txt`, Python 3.14.6), settings and cached CSV hashes match their provenance sidecars; otherwise it recomputes them. `--force-recompute` bypasses that cache. Stages 07–10 read existing upstream products; rerun them after upstream changes.

---

## Results

**Headline: a positive conditional association for strong bars.**

The rebuilt catalogue contains 504,138 galaxies, including 476,396 above log stellar mass 9.5. The fiducial disc sample contains 95,227 galaxies: 12,074 strong-barred, 29,324 weak-barred and 53,829 unbarred.

Within redshift slices and mass–bulge cells, the strong-bar quenched-fraction excess is **+0.0369 ± 0.0054** using a DerSimonian–Laird random-effects summary. The fixed-effect summary is +0.0267 ± 0.0032. These are nominal statistical standard errors, conditional on selection and binning; they exclude measurement and selection systematics. The 75 valid cells contain 11,726 strong-barred and 46,580 unbarred galaxies. Their effects are heterogeneous (I² about 0.61); the two summaries weight cells differently.

The all-bar random-effects excess is +0.0124 ± 0.0022. The weak-bar result, +0.0036 ± 0.0018, is borderline at approximately two nominal standard errors. Relative to the raw strong-bar gap, the fixed-effect residual is about 84% smaller and the random-effects residual about 78% smaller. These comparisons are descriptive, not causal fractions explained.

The full 9.5–10.5 mass band at z < 0.05 does **not reproduce** the approximately −0.2 dex mean fibre-sSFR suppression reported by Liu & Zhou. Its mean contrast is about +0.010 dex, with a bootstrap interval spanning zero. The within-sample quenched/non-quenched **mean** decomposition closes algebraically; it does not explain differences between the studies, whose matching, apertures and SFR estimators differ.

The per-cell results and analysis summaries are in `results/`.

---

## Limitations

- **Association and temporal order.** These observations do not establish that bars cause quenching. Bar formation, quenching and a shared underlying cause can produce similar population-level patterns.
- **Morphology selection.** Upstream masking determines which galaxies have usable votes. Agreement between the two selection schemes does not test galaxies omitted by that masking or calibrate the sign and size of selection bias.
- **Bulge measurement and residual imbalance.** Vote-based bulge prominence is a coarse structural proxy. Galaxies within a cell can still differ in mass, bulge structure and redshift. The balance diagnostic describes median offsets; it does not bound residual confounding.
- **Detectability and sample completeness.** Redshift stratification reduces some population differences but cannot restore missed bars or missing galaxies. The fifth-percentile mass envelope is descriptive, not a completeness limit; no volume weighting is applied.
- **Apertures and SFR estimators.** Fibre measurements sample different physical scales with redshift. Total MPA-JHU SFRs include aperture corrections and spectral-class-dependent estimators. A shared zero-point shift cancels from ΔSFR only if the fitted ridge shifts identically.
- **Dispersion and environment.** The supplementary dispersion tier has its own quality selection and aperture dependence. Central and satellite galaxies are mixed, and environment is not controlled.
- **Statistical interpretation.** Reported errors are nominal and omit selection and measurement systematics. Heterogeneous cells give different fixed-effect and random-effects weights. The Egger-style diagnostic uses coupled binomial effects and uncertainties; its nominal probability is not a calibrated test of publication bias or freedom from bias.
- **Comparison between methods and studies.** Methods share galaxies and errors. Method A omits redshift, and correlated predictors complicate feature importance. The Liu & Zhou comparison uses different matching, mass estimates, SFR estimators and apertures. BPT contrasts do not establish or exclude AGN mediation.

---

## License

Code (`src/`, `config.py`) is MIT licensed: [LICENSE](LICENSE). Derived data, figures and results (`data/merged_catalog.parquet`, `figures/`, `results/`) are CC BY 4.0: [LICENSE-DATA](LICENSE-DATA). Parent survey measurements retain their own terms: Galaxy Zoo DESI is released under CC BY 4.0, and the SDSS and DESI Legacy Imaging Surveys request their standard acknowledgements in publications (see [LICENSE-DATA](LICENSE-DATA)).

---

## Citation & Acknowledgements

To cite this repository, use [CITATION.cff](CITATION.cff). Release v1.0.0 is archived on Zenodo with the version DOI [10.5281/zenodo.22774685](https://doi.org/10.5281/zenodo.22774685); the concept DOI [10.5281/zenodo.22774684](https://doi.org/10.5281/zenodo.22774684) always resolves to the latest version.

> Nagi, A. I. (2026). *bar-quenching-desi-sdss* (v1.0.0) [Software and data]. Zenodo. https://doi.org/10.5281/zenodo.22774685


Please also credit the source catalogues and the methods relevant to your reuse:

- **Galaxy Zoo morphology:** [Walmsley et al. (2023), Galaxy Zoo DESI](https://doi.org/10.1093/mnras/stad2919), and [Walmsley et al. (2022), Galaxy Zoo DECaLS](https://doi.org/10.1093/mnras/stab2093). The adopted DESI catalogue version is available in [Zenodo record 8331338](https://zenodo.org/records/8331338).
- **SDSS MPA-JHU physical quantities:** [Brinchmann et al. (2004)](https://doi.org/10.1111/j.1365-2966.2004.07881.x), [Kauffmann et al. (2003)](https://doi.org/10.1046/j.1365-8711.2003.06291.x), and the [SDSS catalogue documentation](https://www.sdss4.org/dr17/spectro/galaxy_mpajhu/).
- **Quenching analysis:** [Bluck et al. (2020)](https://doi.org/10.1093/mnras/staa2806) and [Piotrowska et al. (2022)](https://doi.org/10.1093/mnras/stab3673).
- **Bar measurements and comparison studies:** [Géron et al. (2021)](https://doi.org/10.1093/mnras/stab2064), [Géron et al. (2023)](https://doi.org/10.1093/mnras/stad501), [Géron et al. (2024)](https://doi.org/10.3847/1538-4357/ad66b7), and [Liu & Zhou (2026)](https://arxiv.org/abs/2605.04537).

This project uses publicly available data from Galaxy Zoo DESI, the DESI Legacy Imaging Surveys, SDSS and the MPA-JHU team. It benefits from the work of Galaxy Zoo volunteers, catalogue builders and survey teams. Computation uses NumPy, pandas, SciPy, Astropy, scikit-learn, Matplotlib and PyArrow. These acknowledgements do not imply endorsement of this analysis by those projects.

---

## Author

Amna Imran Nagi, [github.com/AmnaImran1100](https://github.com/AmnaImran1100)

ORCID: [0009-0007-3742-7883](https://orcid.org/0009-0007-3742-7883)

*Independent research project, 2026.*
