# Data dictionary: `merged_catalog.parquet`

504,138 rows x 52 columns. Produced by `src/02_crossmatch.py` from Galaxy Zoo
DESI morphology (Walmsley et al. 2023) cross-matched to the SDSS MPA-JHU
value-added catalogue (DR8 `galSpecExtra` + `galSpecInfo`) within 1.5 arcsec.

Every row is one unique galaxy that survived: `spectrotype = GALAXY`, a sky
match, de-duplication on the DESI source (highest median S/N spectrum kept),
a valid stellar mass, valid total SFR and sSFR, `z_warning = 0`,
0.02 < z < 0.15, and log10(M*/Msun) > 9.0.

Licence: CC BY 4.0, see `../LICENSE-DATA`. The derived columns are this work's;
the raw measurements belong to Galaxy Zoo DESI and MPA-JHU and must be cited
separately.

---

## READ THIS BEFORE YOU AVERAGE ANYTHING

**MPA-JHU uses `-9999` as a missing-value sentinel, and one such value survives
in this table.** `specsfr_fib_p50` has a minimum of `-9999`: exactly one row
carries it. That row falls outside every analysis frame used in this project
(it is not in any disc sample), so no result here is affected, and
`src/02_crossmatch.py` step 9 asserts this. But if you slice this table
differently, filter `specsfr_fib_p50 > -100` (or `> -30`) before taking any
mean, median or fit. Validity cuts are column-specific; a logarithmic SFR of -1 is valid. Auxiliary columns must still be checked for missing values before reuse.

`v_disp = 0` is also a not-measured placeholder, not a real dispersion. Use the
`sigma_clean` flag rather than filtering `v_disp` by hand.

---

## Identifiers and astrometry

| Column | Type | Meaning |
|---|---|---|
| `specobjid` | str | SDSS spectroscopic object ID. Unique per row. |
| `dr8_id` | str | Galaxy Zoo DESI / DESI Legacy Surveys DR8 source ID. Unique per row. |
| `ra`, `dec` | float32 | SDSS spectroscopic coordinates, degrees (J2000). |
| `gz_ra`, `gz_dec` | float64 | Galaxy Zoo DESI coordinates of the matched source, degrees. |
| `sep_arcsec` | float64 | Sky separation of the match, arcsec. Median 0.080; all < 1.5 by construction. |
| `plateid`, `mjd`, `fiberid` | int | SDSS plate / MJD / fibre of the spectrum. |

## MPA-JHU physical quantities

All are the catalogue's median (P50) estimates. Masses are log10(M/Msun); SFRs
are log10(SFR / Msun/yr); sSFRs are log10(sSFR / yr^-1). Kroupa (2001) IMF,
flat LCDM with H0 = 70 km/s/Mpc and Omega_m = 0.3.

| Column | Type | Meaning |
|---|---|---|
| `lgm_tot_p50` | float32 | Total stellar mass. The analysis variable `log M*`. |
| `lgm_fib_p50` | float32 | Stellar mass inside the 3-arcsec fibre. |
| `sfr_tot_p50` | float32 | Total SFR, aperture-corrected outside the fibre (Salim et al. 2007). |
| `sfr_fib_p50` | float32 | SFR inside the fibre. |
| `specsfr_tot_p50` | float32 | Total sSFR. |
| `specsfr_fib_p50` | float32 | Fibre sSFR. **See the sentinel warning above.** |
| `z` | float32 | SDSS spectroscopic redshift. All science redshifts come from here, never from the GZ external catalogue. |
| `z_err` | float32 | Redshift uncertainty. |
| `z_warning` | int16 | SDSS redshift warning flag. Always 0 in this table (it is a selection cut). |
| `v_disp` | float32 | Central velocity dispersion inside the fibre, km/s. The analysis variable `sigma_c`. 0 means not measured. |
| `v_disp_err` | float32 | Its uncertainty. Negative values are placeholders. |
| `sn_median` | float32 | Median S/N of the spectrum. |
| `reliable` | int16 | MPA-JHU reliability flag, 1 = reliable. |
| `bptclass` | int16 | Emission-line class: 1 = star-forming, 2 = low-S/N star-forming, 3 = composite, 4 = AGN excluding LINERs, 5 = low-S/N LINER, -1 = unclassifiable. A value of 0 also occurs; it is undocumented and treated here as unclassifiable. |
| `spectrotype` | str | Always `GALAXY` (a selection cut). |
| `subclass` | str | SDSS spectral subclass (`STARFORMING`, `BROADLINE`, ...). |

## Galaxy Zoo DESI vote fractions

Deep-learning predicted fractions of volunteers selecting each answer, in
[0, 1]. **NaN where the question was never reached**: the GZ decision tree only
asks the bar, bulge and spiral questions on the featured, not-edge-on branch,
so ~80 per cent of rows have NaN there. These missing values reflect upstream friendly-catalogue masking and limit the available morphology sample.
Within a question the fractions sum to 1 (verified to 1.7e-7).

| Column | Meaning |
|---|---|
| `smooth-or-featured_featured-or-disk_fraction` | Featured / disc vote fraction. Disc gate. |
| `disk-edge-on_yes_fraction`, `disk-edge-on_no_fraction` | Edge-on vote fractions. Face-on gate. |
| `bar_strong_fraction`, `bar_weak_fraction`, `bar_no_fraction` | Bar vote fractions. |
| `bulge-size_{none,small,moderate,large,dominant}_fraction` | Bulge-size vote fractions. |
| `has-spiral-arms_yes_fraction` | Spiral-arm vote fraction. Secondary disc feature in Method A. |

## Derived columns (this work)

| Column | Type | Definition |
|---|---|---|
| `bulge_prominence` | float64 | Normalised vote-weighted bulge score, `sum(w_k f_k) / sum(f_k)` with weights (0,1,2,3,4) for (none, small, moderate, large, dominant). Range 0–4; observed 0.27–3.25. The analysis variable `B`. NaN where bulge votes are absent. |
| `bar_strength` | float64 | Continuous bar score `B_vol = f_strong + 0.5 f_weak` (adopted ordinal score in this work). 0 and 1 are the unbarred and strong-bar endpoints, not calibrated certainty. NaN where bar votes are absent. |
| `bar_class_comparison` | str | Bar class under the fiducial scheme: `strong`, `weak`, `unbarred`, or `unclassified`. Barred iff `bar_no_fraction < 0.5`; strong iff `f_strong > f_weak`. |
| `bar_class_conventional` | str | Same, under the robustness scheme (`f_strong + f_weak > 0.5`). **Identical to `bar_class_comparison` on every row**: the two rules are mathematically equivalent because the fractions sum to 1. Kept to make that verifiable. |
| `in_sample_comparison` | bool | Passes the fiducial disc gate (featured >= 0.27 AND edge-on_no >= 0.68) and has usable bar+bulge votes. |
| `in_sample_conventional` | bool | Passes the robustness disc gate (featured > 0.5, edge-on_yes < 0.5) and has usable votes. A strict superset of the above (differs on 220 rows in the main sample). |
| `in_main_sample` | bool | log10(M*/Msun) > 9.5. The main sample of 476,396 galaxies. |
| `in_lowz_slice` | bool | 9.0 < log10(M*/Msun) < 9.5 and 0.02 < z < 0.05. The 19,883-galaxy low-mass slice used only for the Liu & Zhou enhancement test. |
| `delta_sfr` | float64 | `log SFR - log SFR_MS(M*)`, with the main-sequence ridge fitted to this sample: `log SFR_MS = 0.646 log10(M*/Msun) - 6.336`. |
| `quenched_dsfr` | bool | **Fiducial quenched label**: `delta_sfr < -1.1`. |
| `green_valley` | bool | `-1.1 <= delta_sfr < -0.5`. Excluded from Random Forest training, included everywhere else. |
| `quenched_ssfr` | bool | Robustness quenched label: `specsfr_tot_p50 < -11.0`. The -10.5 and -11.5 variants are recomputed on the fly in `src/05_bar_test.py`, not stored. |
| `sigma_clean` | bool | Reliable central dispersion: `70 < v_disp < 420` km/s AND finite, positive `v_disp_err` AND finite `sn_median > 10` AND `reliable == 1`. Use this wherever `sigma_c` is a feature or control. |
| `agn_clean` | bool | `bptclass in {1, 2}`, i.e. BPT star-forming only. **Not a neutral filter**: it removes 63 per cent of strong-barred but 25 per cent of unbarred discs, so it preferentially discards the quenched population. See the BPT composition results in `results/05e_bpt_breakdown.csv` before using it. |

---

## Reproducing the headline number from this file

```python
import pandas as pd
d = pd.read_parquet("data/merged_catalog.parquet")
disc = d[d.in_sample_comparison & d.in_main_sample]
q = disc.groupby("bar_class_comparison").quenched_dsfr.mean()
print(q)          # strong 0.286, weak 0.147, unbarred 0.118
```

That is the *raw, uncontrolled* contrast (+16.9 pp). The controlled result
(+3.7 pp random effects) requires the redshift-sliced (mass, bulge) grid; run
`python src/05_bar_test.py`, or read the per-cell tables in
`results/05b_controlled_binning_zsliced_comparison_*.csv`.
