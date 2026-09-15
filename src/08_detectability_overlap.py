"""08_detectability_overlap.py: provenance for two numbers used in the analysis.

Both numbers are regenerated from data/merged_catalog.parquet so they are
traceable to a logged output.

(A) BAR DETECTABILITY vs REDSHIFT: the reason for making the
    redshift control fiducial in Method B. At FIXED stellar mass
    (10.0 < logM < 10.5, so the well-known mass dependence of the bar fraction
    cannot masquerade as a redshift trend) we measure the strong / weak / total
    bar fraction of in-sample discs as a function of z, with Wilson intervals.
    The analysis reports the fall in the strong-bar fraction between the extreme
    bins; the weak-bar fraction is quoted as nearly flat, which is the
    signature of the classifier reassigning unresolved strong bars into the
    weak and no-bar classes rather than of a real evolution in bar content.

(B) SELECTION-SCHEME OVERLAP: the check behind the statement that the
    "two schemes" comparison is a DISC-GATE robustness test and not a
    bar-definition one. Two things are verified:
      1. the three bar vote fractions sum to unity, so the comparison rule
         (f_no < 0.5) and the conventional rule (f_strong + f_weak > 0.5) are
         mathematically the same rule, reported as the number of rows on
         which the two barred masks actually disagree (expected: zero);
      2. the overlap between the two in-main disc samples, reported under
         every sensible definition (intersection over each sample, and the
         Jaccard index) so the output can identify the comparison precisely.

Reads : data/merged_catalog.parquet
Writes: results/08_detectability_overlap.txt
        results/08_bar_fraction_vs_z.csv

Run:  py src/08_detectability_overlap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

# Fixed-mass window for the detectability test. Chosen because it is the mass
# range over which both bar classes are well populated at every redshift, so
# the comparison is not limited by counts at either end.
DET_MASS_LO, DET_MASS_HI = 10.0, 10.5

# Redshift bins. The first two are the bins quoted in the analysis; the
# remainder give the full monotonic trend, including the three fiducial
# Method-B slices, so the whole curve is visible, not just its ends.
DET_Z_BINS = [
    (0.02, 0.05, "quoted-low"),
    (0.11, 0.15, "quoted-high"),
    (0.02, 0.06, "Z_SLICES[0]"),
    (0.06, 0.10, "Z_SLICES[1]"),
    (0.10, 0.15, "Z_SLICES[2]"),
    (0.02, 0.04, "fine"),
    (0.04, 0.06, "fine"),
    (0.06, 0.08, "fine"),
    (0.08, 0.10, "fine"),
    (0.10, 0.12, "fine"),
    (0.12, 0.15, "fine"),
]

BAR_STRONG = "bar_strong_fraction"
BAR_WEAK = "bar_weak_fraction"
BAR_NO = "bar_no_fraction"


class Audit:
    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


LOG = Audit()


# ---------------------------------------------------------------------------
# (A) bar detectability vs redshift at fixed stellar mass
# ---------------------------------------------------------------------------

def detectability(df):
    """Strong / weak / total bar fraction vs z at fixed mass, with Wilson 68%
    intervals. Denominator = all classified in-sample discs in the cell, so the
    three fractions sum to one by construction."""
    LOG("=" * 72)
    LOG("(A) BAR DETECTABILITY vs REDSHIFT  (fiducial 'comparison' scheme)")
    LOG(f"    fixed stellar mass: {DET_MASS_LO} < logM* < {DET_MASS_HI}")
    LOG("    denominator = in-main-sample discs passing the comparison gate")
    LOG("=" * 72)

    sel = (df["in_sample_comparison"].to_numpy(bool)
           & df["in_main_sample"].to_numpy(bool))
    m = pd.to_numeric(df["lgm_tot_p50"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(float)
    bc = df["bar_class_comparison"].to_numpy()
    sel &= (m > DET_MASS_LO) & (m < DET_MASS_HI)
    LOG(f"    galaxies in the fixed-mass window ......... {int(sel.sum()):>8,}")
    LOG("")
    LOG(f"    {'z bin':>13s} {'tag':>12s} {'N':>7s}  "
        f"{'f_strong':>18s}  {'f_weak':>18s}  {'f_total':>8s}")

    rows = []
    for zlo, zhi, tag in DET_Z_BINS:
        cell = sel & (z >= zlo) & (z < zhi)
        n = int(cell.sum())
        if n == 0:
            continue
        n_s = int((bc[cell] == "strong").sum())
        n_w = int((bc[cell] == "weak").sum())
        f_s, f_w = n_s / n, n_w / n
        s_lo, s_hi = su.wilson_interval(n_s, n, z=1.0)     # 68% interval
        w_lo, w_hi = su.wilson_interval(n_w, n, z=1.0)
        rows.append(dict(z_lo=zlo, z_hi=zhi, tag=tag, n=n,
                         n_strong=n_s, n_weak=n_w,
                         f_strong=f_s, f_strong_lo=s_lo, f_strong_hi=s_hi,
                         f_weak=f_w, f_weak_lo=w_lo, f_weak_hi=w_hi,
                         f_total=f_s + f_w))
        LOG(f"    {zlo:5.2f}-{zhi:<6.2f} {tag:>12s} {n:>7,}  "
            f"{f_s:6.3f} [{s_lo:.3f},{s_hi:.3f}]  "
            f"{f_w:6.3f} [{w_lo:.3f},{w_hi:.3f}]  {f_s + f_w:8.3f}")

    out = pd.DataFrame(rows)
    q = {r["tag"]: r for r in rows if r["tag"].startswith("quoted")}
    LOG("")
    if "quoted-low" in q and "quoted-high" in q:
        lo, hi = q["quoted-low"], q["quoted-high"]
        LOG("    REDSHIFT DIAGNOSTIC ('Bar detectability and the "
            "redshift confound'):")
        LOG(f"      strong-bar fraction  z {lo['z_lo']}-{lo['z_hi']}: "
            f"{lo['f_strong']:.4f}  ({lo['f_strong']*100:.1f} per cent, "
            f"n={lo['n']:,})")
        LOG(f"      strong-bar fraction  z {hi['z_lo']}-{hi['z_hi']}: "
            f"{hi['f_strong']:.4f}  ({hi['f_strong']*100:.1f} per cent, "
            f"n={hi['n']:,})")
        LOG(f"      -> falls by a factor {lo['f_strong']/hi['f_strong']:.1f} "
            f"across the redshift range at FIXED mass.")
        LOG(f"      weak-bar fraction over the same two bins: "
            f"{lo['f_weak']:.3f} -> {hi['f_weak']:.3f} "
            f"(near-flat; range over ALL bins "
            f"{out['f_weak'].min():.3f}-{out['f_weak'].max():.3f})")
    LOG("")
    LOG("    Reading: at fixed stellar mass the STRONG-bar fraction collapses")
    LOG("    with redshift while the WEAK-bar fraction barely moves; the")
    LOG("    pattern is consistent with unresolved strong bars entering the weak")
    LOG("    and no-bar classes as angular resolution degrades; selection and")
    LOG("    population differences may also contribute. Because quenched galaxies")
    LOG("    sit at lower z at fixed mass in a flux-limited sample, z is a confound")
    LOG("    on the bar-quenching contrast and is controlled in the fiducial")
    LOG("    Method B measurement.")
    LOG("")
    return out


# ---------------------------------------------------------------------------
# (B) selection-scheme overlap and the bar-rule identity
# ---------------------------------------------------------------------------

def scheme_overlap(df):
    LOG("=" * 72)
    LOG("(B) SELECTION-SCHEME OVERLAP  (comparison vs conventional)")
    LOG("=" * 72)

    fs = pd.to_numeric(df[BAR_STRONG], errors="coerce").to_numpy(float)
    fw = pd.to_numeric(df[BAR_WEAK], errors="coerce").to_numpy(float)
    fn = pd.to_numeric(df[BAR_NO], errors="coerce").to_numpy(float)
    have = np.isfinite(fs) & np.isfinite(fw) & np.isfinite(fn)

    # 1. do the three bar vote fractions sum to unity?
    tot = fs[have] + fw[have] + fn[have]
    n_have = int(have.sum())
    within = int((np.abs(tot - 1.0) < 1e-6).sum())
    LOG(f"  rows carrying all three bar votes ............. {n_have:>8,}")
    LOG(f"  with f_strong+f_weak+f_no = 1 (tol 1e-6) ...... {within:>8,}  "
        f"({within / n_have:.4%})")
    LOG(f"  max |sum - 1| over those rows ................. "
        f"{np.max(np.abs(tot - 1.0)):.3e}")

    # 2. so the two barred rules must be identical; verify this.
    barred_comparison = fn < 0.5
    barred_conventional = (fs + fw) > 0.5
    disagree = int((have & (barred_comparison != barred_conventional)).sum())
    LOG(f"  rows where (f_no<0.5) != (f_strong+f_weak>0.5) . {disagree:>8,}")
    LOG("  -> the two schemes' BAR rules are mathematically identical, so the")
    LOG("     scheme comparison cannot be a bar-definition robustness check.")
    LOG("     What remains of the difference is examined next.")
    LOG("")

    # 3. WHY the two gates nest: two structural facts about the GZ tree.
    #    (i) the friendly catalog only asks the bar / bulge questions of
    #        galaxies the network places firmly on the FEATURED branch, so
    #        requiring usable bar+bulge votes already implies featured > 0.5 --
    #        which makes the comparison scheme's looser 0.27 threshold
    #        INOPERATIVE (it never admits a single extra galaxy);
    #    (ii) the edge-on question has two answers summing to unity, so
    #        edge-on_no >= 0.68  =>  edge-on_yes <= 0.32 < 0.5, i.e. the
    #        comparison edge-on gate is strictly STRICTER than the
    #        conventional one.
    #    Together these force comparison-sample subset-of conventional-sample.
    feat = pd.to_numeric(df["smooth-or-featured_featured-or-disk_fraction"],
                         errors="coerce").to_numpy(float)
    eno = pd.to_numeric(df["disk-edge-on_no_fraction"],
                        errors="coerce").to_numpy(float)
    eyes = pd.to_numeric(df["disk-edge-on_yes_fraction"],
                         errors="coerce").to_numpy(float)
    bulge_cols = [c for c in df.columns if c.startswith("bulge-size_")]
    labelled = (df[[BAR_STRONG, BAR_WEAK, BAR_NO]].notna().all(axis=1)
                & df[bulge_cols].notna().all(axis=1)).to_numpy()
    comp_cfg = config.SCHEMES["comparison"]
    conv_cfg = config.SCHEMES["conventional"]
    n_between = int((labelled
                     & (feat >= comp_cfg["disc_min_featured"])
                     & (feat <= conv_cfg["disc_min_featured"])).sum())
    LOG("  WHY the two gates nest (structural, not accidental):")
    LOG(f"    min featured-or-disk fraction among labelled rows ... "
        f"{np.nanmin(feat[labelled]):.4f}")
    LOG(f"    labelled rows with {comp_cfg['disc_min_featured']} <= featured "
        f"<= {conv_cfg['disc_min_featured']} ......... {n_between:>8,}")
    LOG("    -> the friendly release masks predictions outside its upstream")
    LOG("       question-applicability thresholds; usable bar+bulge votes here")
    LOG("       imply featured > 0.5: the comparison scheme's looser 0.27")
    LOG("       threshold is INOPERATIVE and admits no extra galaxy.")
    LOG(f"    max |f_edge-on,yes + f_edge-on,no - 1| ............. "
        f"{np.nanmax(np.abs(eyes[labelled] + eno[labelled] - 1.0)):.3e}")
    LOG(f"    labelled rows with edge-on_no >= "
        f"{comp_cfg['faceon_min_edgeon_no']} AND edge-on_yes >= "
        f"{conv_cfg['faceon_max_edgeon_yes']} ... "
        f"{int((labelled & (eno >= comp_cfg['faceon_min_edgeon_no']) & (eyes >= conv_cfg['faceon_max_edgeon_yes'])).sum()):>8,}")
    LOG("    -> the edge-on answers sum to unity, so edge-on_no >= 0.68")
    LOG("       implies edge-on_yes <= 0.32 < 0.5: the comparison edge-on gate")
    LOG("       is strictly STRICTER than the conventional one.")
    LOG("    => the comparison sample is a strict SUBSET of the conventional")
    LOG("       sample, and the only operative difference between the two")
    LOG("       schemes is the edge-on threshold.")
    LOG("")

    # 4. overlap of the two in-main disc samples
    main = df["in_main_sample"].to_numpy(bool)
    a = df["in_sample_comparison"].to_numpy(bool) & main
    b = df["in_sample_conventional"].to_numpy(bool) & main
    n_a, n_b = int(a.sum()), int(b.sum())
    inter = int((a & b).sum())
    union = int((a | b).sum())
    LOG(f"  comparison in-main discs ...................... {n_a:>8,}")
    LOG(f"  conventional in-main discs .................... {n_b:>8,}")
    LOG(f"  in BOTH (intersection) ........................ {inter:>8,}")
    LOG(f"  in EITHER (union) ............................. {union:>8,}")
    LOG(f"  only in comparison ............................ "
        f"{int((a & ~b).sum()):>8,}")
    LOG(f"  only in conventional .......................... "
        f"{int((b & ~a).sum()):>8,}")
    LOG("")
    LOG(f"  overlap, intersection / comparison ............ {inter / n_a:.4%}")
    LOG(f"  overlap, intersection / conventional .......... {inter / n_b:.4%}")
    LOG(f"  overlap, Jaccard (intersection / union) ....... "
        f"{inter / union:.4%}   <-- value quoted in the analysis")
    LOG("")
    LOG("  Reading: the two disc samples are nested and near-identical, so the")
    LOG("  agreement of the two schemes is a WEAK check by construction. The")
    LOG("  bar rules are identical, the featured gate is inoperative, and the")
    LOG("  only operative difference is the edge-on threshold. It is therefore")
    LOG("  an EDGE-ON-GATE robustness check. In particular it does NOT probe")
    LOG("  the featured-gate selection bias (a bar is itself a 'feature', so")
    LOG("  quenched unbarred discs could be preferentially voted smooth and lost),")
    LOG("  whose sign and magnitude have not been calibrated here.")
    LOG("")
    return dict(n_comparison=n_a, n_conventional=n_b, intersection=inter,
                union=union, jaccard=inter / union,
                frac_of_comparison=inter / n_a, frac_of_conventional=inter / n_b,
                bar_rule_disagreements=disagree, rows_with_bar_votes=n_have,
                rows_summing_to_unity=within,
                min_featured_labelled=float(np.nanmin(feat[labelled])),
                n_featured_between_thresholds=n_between,
                comparison_is_subset=bool(int((a & ~b).sum()) == 0))


def main():
    LOG("08_detectability_overlap.py: provenance for two quoted numbers")
    LOG("")
    df = pd.read_parquet(config.MERGED)
    LOG(f"merged catalog: {len(df):,} rows x {df.shape[1]} cols")
    LOG("")

    det = detectability(df)
    ov = scheme_overlap(df)

    det.to_csv(config.RESULTS / "08_bar_fraction_vs_z.csv", index=False)
    LOG("=" * 72)
    LOG("SUMMARY: the two numbers as quoted in the analysis")
    LOG("=" * 72)
    q = det.set_index("tag")
    if "quoted-low" in q.index and "quoted-high" in q.index:
        LOG(f"  strong-bar fraction at fixed {DET_MASS_LO}<logM<{DET_MASS_HI}: "
            f"{q.loc['quoted-low', 'f_strong'] * 100:.1f} per cent "
            f"(z {q.loc['quoted-low', 'z_lo']}-{q.loc['quoted-low', 'z_hi']}) "
            f"-> {q.loc['quoted-high', 'f_strong'] * 100:.1f} per cent "
            f"(z {q.loc['quoted-high', 'z_lo']}-{q.loc['quoted-high', 'z_hi']})")
        LOG(f"  weak-bar fraction over all z bins: "
            f"{det['f_weak'].min():.2f}-{det['f_weak'].max():.2f} (near-flat)")
    LOG(f"  scheme overlap (Jaccard): {ov['jaccard']:.2%}   "
        f"[{ov['intersection']:,} of {ov['union']:,}]")
    LOG(f"  bar-rule disagreements between schemes: "
        f"{ov['bar_rule_disagreements']}")
    LOG(f"  comparison sample is a strict subset of conventional: "
        f"{ov['comparison_is_subset']}  "
        f"(min featured among labelled rows = "
        f"{ov['min_featured_labelled']:.3f}, so the 0.27 gate is inoperative)")

    out = config.RESULTS / "08_detectability_overlap.txt"
    LOG.save(out)
    print(f"\nSAVED -> {out}")
    print(f"SAVED -> {config.RESULTS / '08_bar_fraction_vs_z.csv'}")


if __name__ == "__main__":
    main()
