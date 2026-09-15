"""Descriptive within-cell balance for the fiducial Method-B sample.

Reports barred-minus-unbarred median offsets and physical cell boundaries.
Quantile edges vary by redshift slice: matching bin indices across slices
cannot identify a redshift derivative at fixed mass and bulge. No gradient
correction or bound on residual confounding is inferred from these offsets.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

BARMODES = {
    "all": (("strong", "weak"), ("unbarred",)),
    "strong_only": (("strong",), ("unbarred",)),
    "weak_only": (("weak",), ("unbarred",)),
}

LINES = []


def LOG(msg=""):
    print(msg)
    LINES.append(str(msg))


def build_frame(scheme):
    df = pd.read_parquet(config.MERGED)
    m = df[df["in_main_sample"].to_numpy(bool)
           & df[f"in_sample_{scheme}"].to_numpy(bool)].copy()
    out = pd.DataFrame(index=m.index)
    out["z"] = pd.to_numeric(m["z"], errors="coerce")
    out["log_mass"] = pd.to_numeric(m["lgm_tot_p50"], errors="coerce")
    out["bulge_prominence"] = pd.to_numeric(m["bulge_prominence"], errors="coerce")
    out["quenched_dsfr"] = m["quenched_dsfr"].to_numpy(bool)
    out["bar_class"] = m[f"bar_class_{scheme}"].to_numpy()
    return out.reset_index(drop=True)


def quantile_edges(x, n_bins):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, q)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def cell_stats(scheme="comparison", barmode="strong_only"):
    base = build_frame(scheme)
    barred_lbls, unbarred_lbls = BARMODES[barmode]
    rows = []
    for k_z, (zlo, zhi) in enumerate(config.Z_SLICES):
        sl = base[(base["z"] >= zlo) & (base["z"] < zhi)]
        me = quantile_edges(sl["log_mass"], config.N_MASS_BINS)
        be = quantile_edges(sl["bulge_prominence"], config.N_BULGE_BINS)
        mass = sl["log_mass"].to_numpy(float)
        bulge = sl["bulge_prominence"].to_numpy(float)
        z = sl["z"].to_numpy(float)
        q = sl["quenched_dsfr"].to_numpy(bool)
        bc = sl["bar_class"].to_numpy()
        barred = np.isin(bc, barred_lbls)
        unbarred = np.isin(bc, unbarred_lbls)
        mi = np.digitize(mass, me) - 1
        bi = np.digitize(bulge, be) - 1
        for i in range(config.N_MASS_BINS):
            for j in range(config.N_BULGE_BINS):
                cell = (mi == i) & (bi == j)
                cb, cu = cell & barred, cell & unbarred
                nb, nu = int(cb.sum()), int(cu.sum())
                valid = (nb >= config.MIN_PER_CLASS_PER_CELL
                         and nu >= config.MIN_PER_CLASS_PER_CELL)
                row = dict(z_idx=k_z, z_lo=zlo, z_hi=zhi, mass_bin=i,
                           bulge_bin=j, mass_lo=me[i], mass_hi=me[i+1],
                           bulge_lo=be[j], bulge_hi=be[j+1], n_barred=nb, n_unbarred=nu, valid=valid)
                if valid:
                    row.update(
                        d_mass=float(np.median(mass[cb]) - np.median(mass[cu])),
                        d_bulge=float(np.median(bulge[cb]) - np.median(bulge[cu])),
                        d_z=float(np.median(z[cb]) - np.median(z[cu])),
                        q_unbarred=float(q[cu].mean()),
                        med_mass_u=float(np.median(mass[cu])),
                        med_bulge_u=float(np.median(bulge[cu])),
                        med_z_u=float(np.median(z[cu])),
                        delta_q=float(q[cb].mean() - q[cu].mean()),
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def report(barmode):
    LOG("=" * 72)
    LOG(f"WITHIN-CELL BALANCE: comparison scheme, {barmode}, z-controlled")
    LOG("=" * 72)
    df = cell_stats("comparison", barmode)
    v = df[df["valid"]].copy()
    LOG(f"  valid cells: {len(v)}")
    for var, unit in (("d_mass", "dex"), ("d_bulge", "score"), ("d_z", "z")):
        off = v[var]
        LOG(f"  {var:8s} barred-unbarred median offset: "
            f"mean={off.mean():+.4f}  median={off.median():+.4f}  "
            f"[p5,p95]=[{off.quantile(.05):+.4f},{off.quantile(.95):+.4f}] {unit}")
    LOG("  These offsets are descriptive, not a bias correction or a bound.")
    LOG("  Distributional differences and residual confounding remain possible.")
    return df


def main():
    LOG("07_cell_balance_check.py: descriptive within-cell covariate balance")
    LOG("Offsets compare medians within each valid Method-B cell.")
    LOG("Bins have different physical edges in different redshift slices.")
    LOG()
    frames = []
    for barmode in ("strong_only", "all"):
        frames.append(report(barmode).assign(barmode=barmode))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(config.RESULTS / "07_cell_balance.csv", index=False)
    (config.RESULTS / "07_cell_balance.txt").write_text(
        "\n".join(LINES) + "\n", encoding="utf-8")
    LOG()
    LOG(f"wrote results/07_cell_balance.csv + 07_cell_balance.txt")


if __name__ == "__main__":
    main()
