"""01_load_data.py: load and verify the raw catalogues (no merging, no cuts).

This is a verification step, not a science step. It checks that:

  * the GZ DESI parquet has the vote-fraction columns used for the disc, bar
    and bulge definitions (column names can change between releases);
  * galSpecExtra and galSpecInfo are row-matched, so step 02 can join them by
    row index (checked here via SPECOBJID);
  * the column-specific MPA-JHU missing values are known, since they are not
    NaN and would bias medians if treated as data;
  * the fibre columns (Method D) and V_DISP (the sigma_c tier) exist.

A human-readable summary is written to results/01_data_summary.txt.

Run:  py src/01_load_data.py
"""

from __future__ import annotations

import sys
import difflib
from pathlib import Path

import numpy as np
import pandas as pd

# config.py lives at the repo root, one level up from src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Columns the pipeline depends on (checked against the schemas on disk).
# A missing column raises an error immediately.
# ---------------------------------------------------------------------------

# Galaxy Zoo DESI (Walmsley et al. 2023). Vote fractions in [0, 1]; the
# "friendly" catalog blanks (NaN) answers most volunteers were never asked.
GZ_ID_COL = "dr8_id"
GZ_COORD_COLS = ["ra", "dec"]
GZ_MORPH_COLS = [
    # disc / face-on gates (Geron 2021 / Liu & Zhou Eq. 1)
    "smooth-or-featured_featured-or-disk_fraction",
    "disk-edge-on_yes_fraction",
    "disk-edge-on_no_fraction",
    # bar vote fractions
    "bar_strong_fraction",
    "bar_weak_fraction",
    "bar_no_fraction",
    # bulge control (not controlled by Liu & Zhou)
    "bulge-size_none_fraction",
    "bulge-size_small_fraction",
    "bulge-size_moderate_fraction",
    "bulge-size_large_fraction",
    "bulge-size_dominant_fraction",
    # secondary disc feature
    "has-spiral-arms_yes_fraction",
]
GZ_REQUIRED = [GZ_ID_COL, *GZ_COORD_COLS, *GZ_MORPH_COLS]

# MPA-JHU galSpecExtra: masses, SFRs, sSFRs (total AND fibre), BPT class.
# Column names are UPPERCASE in the FITS; we lower-case on load for consistency
# with config's naming. specsfr_* is already log sSFR; do not take its log again.
MPA_EXTRA_COLS = [
    "SPECOBJID",       # row-match verification key
    "BPTCLASS",        # AGN-clean rerun
    "LGM_TOT_P50",     # log stellar mass (total)
    "LGM_FIB_P50",     # log stellar mass (fibre aperture)
    "SFR_TOT_P50",     # log SFR (total)
    "SFR_FIB_P50",     # log SFR (fibre), Method D enhancement panel
    "SPECSFR_TOT_P50", # log sSFR (total)  [already log]
    "SPECSFR_FIB_P50", # log sSFR (fibre)  [already log], Method D
]

# MPA-JHU galSpecInfo: coordinates, redshift, velocity dispersion, quality.
MPA_INFO_COLS = [
    "SPECOBJID",       # must match Extra row-for-row
    "RA", "DEC",       # sky match into GZ
    "Z", "Z_ERR", "Z_WARNING",
    "V_DISP", "V_DISP_ERR",   # sigma_c tier
    "SN_MEDIAN",       # spectrum S/N quality cut
    "RELIABLE",        # reliability flag (convention verified from data, below)
    "SPECTROTYPE",     # GALAXY / STAR / QSO; the 1.84M rows include non-galaxies
    "SUBCLASS",        # AGN / STARFORMING etc.
    "PLATEID", "MJD", "FIBERID",
]

# Column-specific missing values from the SDSS galSpecExtra data model.
# In particular, -1 is a valid logarithmic SFR (0.1 solar masses/year),
# and BPTCLASS=-1 is a meaningful classification, not a generic missing value.
INVALID_VALUES = {c.lower(): (-9999.0,) for c in MPA_EXTRA_COLS
                  if c not in ("SPECOBJID", "BPTCLASS")}


def valid_measurement(values, column):
    x = np.asarray(values, dtype=float)
    return np.isfinite(x) & ~np.isin(x, INVALID_VALUES.get(column.lower(), ()))


# ---------------------------------------------------------------------------
# Column verification
# ---------------------------------------------------------------------------

def _check_columns(available, required, catalog_name):
    """Raise a clear, actionable error if any required column is absent.

    Lists the closest available names for each missing column, so a renamed
    column in a new catalogue release is easy to identify.
    """
    available = list(available)
    missing = [c for c in required if c not in available]
    if not missing:
        return
    lines = [f"{catalog_name}: {len(missing)} required column(s) not found:"]
    for col in missing:
        near = difflib.get_close_matches(col, available, n=3, cutoff=0.5)
        hint = f"  closest: {near}" if near else "  (no similar names)"
        lines.append(f"  MISSING: {col}\n{hint}")
    raise KeyError("\n".join(lines))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gz_desi(path=config.GZ_DESI_FILE, columns=None):
    """Load the Galaxy Zoo DESI morphology catalog (needed columns only).

    The catalogue has ~8.7M rows and 41 columns, so only the needed ones are read. We read
    the parquet schema first, verify the required columns exist, then read only
    those. Returns a DataFrame of the science columns.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GZ DESI catalog not found: {path}")

    schema_cols = pq.ParquetFile(path).schema.names
    _check_columns(schema_cols, GZ_REQUIRED, "GZ DESI")

    if columns is None:
        columns = GZ_REQUIRED
    df = pd.read_parquet(path, columns=columns)
    return df


def load_external(path=config.GZ_EXTERNAL_FILE):
    """Load the external (non-morphology) catalog, row-matched to GZ by dr8_id.

    Used ONLY as a redshift pre-filter in step 02. It may be absent (it is not
    required to *verify* the science columns), so return None with a loud note
    rather than crashing the whole verification gate.

    When present, discover the redshift column by name rather than assuming it.
    """
    path = Path(path)
    if not path.exists():
        return None, (
            f"external catalog ABSENT: {config.rel(path)}\n"
            "  -> This is the step 02 redshift pre-filter (Zenodo 8331338).\n"
            "  -> Without it, step 02 must sky-match all 8.7M GZ rows (slower,\n"
            "     but not wrong). Download it, or accept the full-catalog match.\n"
        )

    import pyarrow.parquet as pq
    schema_cols = pq.ParquetFile(path).schema.names
    z_candidates = [c for c in schema_cols
                    if c.lower() in ("redshift", "z", "specz", "spec_z",
                                     "photoz", "photo_z", "z_spec")]
    note = [f"external catalog present: {config.rel(path)}",
            f"  columns ({len(schema_cols)}): {schema_cols}",
            f"  redshift-column candidates: {z_candidates or 'NONE FOUND'}"]
    # Load only the id + candidate z columns for the summary.
    want = [c for c in schema_cols if c == "dr8_id" or c in z_candidates]
    df = pd.read_parquet(path, columns=want) if want else None
    return df, "\n".join(note)


def _decode_bytes(df):
    """Decode any bytes/object columns that hold byte strings to str."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else None
            if isinstance(sample, (bytes, bytearray)):
                df[col] = df[col].str.decode("utf-8", errors="replace")
    return df


def load_mpajhu(extra_path=config.MPAJHU_EXTRA, info_path=config.MPAJHU_INFO):
    """Load galSpecExtra + galSpecInfo and join them by ROW INDEX.

    The two files are row-matched by construction; we VERIFY this via SPECOBJID
    before trusting the index join. Column names are lower-cased. Returns the
    combined per-spectrum DataFrame (still no cuts, no sky match).
    """
    from astropy.table import Table

    extra_path, info_path = Path(extra_path), Path(info_path)
    for p in (extra_path, info_path):
        if not p.exists():
            raise FileNotFoundError(f"MPA-JHU catalog not found: {p}")

    extra = Table.read(extra_path)
    info = Table.read(info_path)

    _check_columns(extra.colnames, MPA_EXTRA_COLS, "galSpecExtra")
    _check_columns(info.colnames, MPA_INFO_COLS, "galSpecInfo")

    if len(extra) != len(info):
        raise ValueError(
            f"galSpecExtra ({len(extra)}) and galSpecInfo ({len(info)}) have "
            "different row counts, so they are not row-matched. Aborting; a "
            "row-index join would silently misalign every spectrum.")

    extra_df = extra[MPA_EXTRA_COLS].to_pandas()
    info_df = info[MPA_INFO_COLS].to_pandas()
    extra_df = _decode_bytes(extra_df)
    info_df = _decode_bytes(info_df)

    # Verify the row-match empirically: SPECOBJID must agree row-for-row.
    #
    # NA handling matters here. ~20% of the DR8 rows are placeholders
    # with a MISSING SPECOBJID in BOTH files; those rows are still correctly
    # aligned, so "missing in both" must count as agreement. Filling with an
    # explicit sentinel before comparing keeps the test valid regardless of
    # whether the pandas version in use coerces NaN to the literal string
    # 'nan' (pandas 2.x) or propagates it as NA (pandas 3.x). Under the
    # latter, a naive .astype(str) comparison makes every placeholder row look
    # like a mismatch and this check fails on valid data.
    e_id = extra_df["SPECOBJID"].fillna("__MISSING__").astype(str).str.strip()
    i_id = info_df["SPECOBJID"].fillna("__MISSING__").astype(str).str.strip()
    n_missing_both = int(((e_id == "__MISSING__") & (i_id == "__MISSING__")).sum())
    mismatches = int((e_id.values != i_id.values).sum())
    if mismatches:
        raise ValueError(
            f"SPECOBJID disagrees on {mismatches} rows between Extra and Info: "
            "the files are not aligned row-for-row. Join by ID, not index.")
    extra_df.attrs["specobjid_missing_both"] = n_missing_both

    info_df = info_df.drop(columns=["SPECOBJID"])  # keep one copy
    df = pd.concat([extra_df.reset_index(drop=True),
                    info_df.reset_index(drop=True)], axis=1)
    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _numeric_summary(s):
    """min / median / max over FINITE, NON-SENTINEL values, plus bad-value
    fractions. Sentinels are excluded from the range so they don't masquerade
    as physical extrema."""
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    n = x.size
    nan_frac = float(np.isnan(x).mean()) if n else float("nan")
    sentinels = INVALID_VALUES.get(str(s.name).lower(), ())
    sent_frac = {v: float((x == v).mean()) for v in sentinels}
    good = x[valid_measurement(x, str(s.name))]
    if good.size:
        lo, med, hi = np.min(good), np.median(good), np.max(good)
        rng = f"[{lo:.4g}, {med:.4g} (med), {hi:.4g}]"
    else:
        rng = "[no finite non-sentinel values]"
    parts = [f"NaN={nan_frac:6.2%}"]
    for v in sentinels:
        if sent_frac[v] > 0:
            parts.append(f"({int(v)})={sent_frac[v]:.2%}")
    return f"{rng:42s} " + "  ".join(parts)


def summarise(df, name):
    """Human-readable summary block for one loaded catalogue."""
    lines = [f"{'=' * 72}", f"{name}", f"{'=' * 72}",
             f"rows: {len(df):,}    cols: {df.shape[1]}", ""]
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            lines.append(f"  {col:44s} {_numeric_summary(s)}")
        else:
            nun = s.nunique(dropna=True)
            examples = list(pd.unique(s.dropna()))[:4]
            lines.append(f"  {col:44s} {'str/obj':42s} "
                         f"n_unique={nun}  e.g. {examples}")
    lines.append("")
    return "\n".join(lines)


def _categorical_report(df, col, title):
    """value_counts for a flag/type column that is checked rather than assumed
    (SPECTROTYPE, RELIABLE, BPTCLASS)."""
    if col not in df.columns:
        return f"{title}: column '{col}' absent\n"
    vc = df[col].value_counts(dropna=False)
    out = [f"{title}  (column '{col}'):"]
    for val, cnt in vc.items():
        out.append(f"    {str(val):>12s} : {cnt:>10,}  ({cnt / len(df):6.2%})")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_path = config.RESULTS / "01_data_summary.txt"
    blocks = []

    def emit(text):
        print(text)
        blocks.append(text)

    emit("01_load_data.py: raw catalogue verification\n")

    # --- GZ DESI morphology ---
    emit(">> Loading Galaxy Zoo DESI morphology (science columns only)...")
    gz = load_gz_desi()
    n_dup_id = int(gz[GZ_ID_COL].duplicated().sum())
    emit(f"   dr8_id duplicates: {n_dup_id:,} "
         f"({'OK, unique' if n_dup_id == 0 else 'WARNING: repeated IDs'})")
    emit(summarise(gz, "GALAXY ZOO DESI (friendly, science columns)"))

    # --- external (pre-filter) catalog ---
    emit(">> Checking external (redshift pre-filter) catalog...")
    ext, ext_note = load_external()
    emit(ext_note)
    if ext is not None:
        emit(summarise(ext, "EXTERNAL CATALOG (pre-filter only)"))

    # --- MPA-JHU spectroscopy ---
    emit(">> Loading MPA-JHU galSpecExtra + galSpecInfo...")
    mpa = load_mpajhu()
    emit("   row-match verified: SPECOBJID agrees row-for-row (index join safe)")
    emit(summarise(mpa, "MPA-JHU (galSpecExtra x galSpecInfo, row-joined)"))

    # --- flags and types to confirm ---
    emit(">> Distributions of flags we must verify before the 02 cross-match cuts:\n")
    emit(_categorical_report(mpa, "spectrotype",
                             "Spectral type (GALAXY vs STAR/QSO; only GALAXY "
                             "is science)"))
    emit(_categorical_report(mpa, "reliable",
                             "RELIABLE flag (confirm convention vs MPA-JHU docs "
                             "before using reliable==1)"))
    emit(_categorical_report(mpa, "bptclass",
                             "BPTCLASS (AGN-clean rerun keeps {1,2}: SF / "
                             "low-S/N SF)"))

    out_path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"\nWrote audit -> {config.rel(out_path)}")


if __name__ == "__main__":
    main()
