"""Generate the manuscript's numbers and tables from the catalogue and results.

Run after stages 04--09. Every number quoted in the paper through a macro is
computed here; none is hard-coded. Writes paper/numbers.tex,
paper/table_headline.tex and paper/table_robustness.tex (the paper/ directory
is created if absent and is not part of the repository), plus the
machine-readable summary results/10_release_numbers.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import stats_utils as su


def read(name):
    return pd.read_csv(config.RESULTS / name)


def cells(mode, scheme="comparison", zcontrol=True):
    tag = "zsliced_" if zcontrol else ""
    d = read(f"05b_controlled_binning_{tag}{scheme}_{mode}.csv")
    return d[(d["sample"] == "all") & d["valid"].astype(bool)].copy()


def summarize(d):
    effect = d["delta_q"].to_numpy(float)
    se = su.diff_se_from_interval(d["delta_q_lo"], d["delta_q_hi"])
    fe, fe_se = su.inverse_variance_weighted_mean(effect, se)
    re = su.random_effects_mean(effect, se)
    return dict(fe=float(fe), fe_se=float(fe_se), re=re["mean"], re_se=re["se"],
                Q=re["Q"], dof=re["dof"], I2=re["I2"], n=len(d),
                positive=int((effect > 0).sum()),
                sig_positive=int((d["delta_q_lo"] > 0).sum()),
                sig_negative=int((d["delta_q_hi"] < 0).sum()),
                n_barred=int(d["n_barred"].sum()), n_unbarred=int(d["n_unbarred"].sum()))


def main():
    paper = config.ROOT / "paper"
    paper.mkdir(exist_ok=True)
    values = {}
    def put(key, value, fmt=None):
        values[key] = format(value, fmt) if fmt else str(value)
    def integer(key, n):
        put(key, f"{int(n):,}".replace(",", r"\,"))
    def pair(key, v, err, decimals=4):
        put(key, f"{v:+.{decimals}f} \\pm {err:.{decimals}f}")
    def scientific(key, x):
        mantissa, exponent = f"{x:.3e}".split("e")
        put(key, rf"{mantissa}\times10^{{{int(exponent)}}}")

    cat = pd.read_parquet(config.MERGED)
    main_disc = cat[cat.in_main_sample & cat.in_sample_comparison]
    integer("NDisc", len(main_disc))
    integer("NStrong", (main_disc.bar_class_comparison == "strong").sum())
    q = main_disc.groupby("bar_class_comparison").quenched_dsfr.mean()
    raw = float(q["strong"] - q["unbarred"])
    put("RawGap", 100*raw, "+.2f")
    put("StrongQ", q["strong"], ".4f")
    put("UnbarredQ", q["unbarred"], ".4f")
    summaries = {mode: summarize(cells(mode)) for mode in ("strong_only", "all", "weak_only")}
    strong = summaries["strong_only"]
    for prefix, mode in (("Strong", "strong_only"), ("All", "all"), ("Weak", "weak_only")):
        r = summaries[mode]
        pair(prefix+"RE", r["re"], r["re_se"])
        pair(prefix+"FE", r["fe"], r["fe_se"])
        put(prefix+"Sigma", abs(r["re"])/r["re_se"], ".1f")
    for key, field in (("NStrongCells", "n"), ("NStrongPositive", "positive"),
                       ("NStrongSigPositive", "sig_positive"), ("NStrongSigNegative", "sig_negative"),
                       ("NStrongUsed", "n_barred"), ("NUnbarredUsed", "n_unbarred"),
                       ("StrongDOF", "dof")):
        integer(key, strong[field])
    put("StrongQHet", strong["Q"], ".2f")
    put("StrongISquared", strong["I2"], ".2f")
    put("AttenFE", 100*(1-strong["fe"]/raw), ".0f")
    put("AttenRE", 100*(1-strong["re"]/raw), ".0f")
    put("StrongFEValue", strong["fe"], "+.4f")
    noz_strong = summarize(cells("strong_only", zcontrol=False))
    noz_all = summarize(cells("all", zcontrol=False))
    put("StrongNoZ", noz_strong["fe"], "+.4f")
    put("StrongNoZRE", noz_strong["re"], "+.4f")
    put("StrongREValue", strong["re"], "+.4f")
    # Fractional change of each summary when redshift stratification is added.
    put("ZAttenStrongFE", 100*(1-strong["fe"]/noz_strong["fe"]), ".0f")
    put("ZAttenStrongRE", 100*(1-strong["re"]/noz_strong["re"]), ".0f")
    put("ZAttenAllFE", 100*(1-summaries["all"]["fe"]/noz_all["fe"]), ".0f")
    put("ZAttenAllRE", 100*(1-summaries["all"]["re"]/noz_all["re"]), ".0f")
    cv = summarize(cells("strong_only", "conventional"))
    pair("ConventionalRE", cv["re"], cv["re_se"])
    cs = cells("strong_only")
    for key, (_, group) in zip(("SliceOne", "SliceTwo", "SliceThree"), cs.groupby("z_lo", sort=True)):
        r = summarize(group)
        pair(key, r["fe"], r["fe_se"])

    gate_main = read("04_bulge_importance.csv").set_index("feature")
    gate_sig = read("04_sigma_importance.csv").set_index("feature")
    bar_rf = read("05a_rf_importance_comparison.csv")
    for label, frame in (("GateMain", gate_main), ("GateSigma", gate_sig)):
        integer("N"+label+"RF", frame.iloc[0].n_input)
        put(label+"AUC", frame.iloc[0].auc_median, ".3f")
    for key, col in (("GateSigmaImportance", "sigma_c"), ("GateSigmaMassImportance", "log_mass"),
                     ("GateSigmaBulgeImportance", "bulge_prominence")):
        put(key, gate_sig.loc[col, "perm_median"], ".3f")
    for label, sample in (("BarMain", "comparison:main"), ("BarSigma", "comparison:sigma")):
        r = bar_rf[(bar_rf["sample"] == sample) & (bar_rf.feature == "bar_strength")].iloc[0]
        integer("N"+label+"RF", r.n_input)
        put(label+"AUC", r.auc_median, ".3f")
        put(label+"Importance", r.perm_median, ".3f")

    balance = read("07_cell_balance.csv")
    balance = balance[(balance.barmode == "strong_only") & balance.valid.astype(bool)]
    for key, col in (("BalanceMass", "d_mass"), ("BalanceBulge", "d_bulge"), ("BalanceZ", "d_z")):
        put(key, balance[col].median(), "+.4f")
    se = su.diff_se_from_interval(cs.delta_q_lo, cs.delta_q_hi)
    egger = su.egger_precision_diagnostic(cs.delta_q, se)
    put("EggerIntercept", egger["intercept"], ".6f")
    put("EggerSE", egger["intercept_se"], ".6f")
    scientific("EggerP", egger["p_value"])
    integer("EggerDOF", egger["dof"])
    r_se = spearmanr(se, cs.delta_q)
    r_count = spearmanr(cs.n_barred, cs.delta_q)
    put("PrecisionRho", r_se.statistic, "+.3f")
    put("CountRho", r_count.statistic, "+.3f")
    put("CountP", r_count.pvalue, ".3f")
    small = cs.n_barred < cs.n_barred.median()
    for key, mask in (("SmallCellsRE", small), ("LargeCellsRE", ~small)):
        r = su.random_effects_mean(cs.loc[mask, "delta_q"], np.asarray(se)[mask])
        pair(key, r["mean"], r["se"])

    pcc = read("05c_partial_correlation.csv")
    pcc = pcc[(pcc.scheme == "comparison") & (pcc["sample"] == "all")]
    def correlation(tier, subset, control):
        r = pcc[(pcc.tier == tier) & (pcc.subset == subset) & (pcc.controls == control)]
        assert len(r) == 1
        return r.iloc[0]
    for key, tier, subset, control in (
        ("PCCMainMB", "main", "strong+weak", "mass+bulge"),
        ("PCCStrongMB", "main", "strong_only", "mass+bulge"),
        ("PCCMainMBZ", "main", "strong+weak", "mass+bulge+z"),
        ("PCCStrongMBZ", "main", "strong_only", "mass+bulge+z"),
        ("PCCSigmaMBS", "sigma", "strong+weak", "mass+bulge+sigma"),
        ("PCCSigmaStrongMBS", "sigma", "strong_only", "mass+bulge+sigma"),
        ("PCCSigmaMBSZ", "sigma", "strong+weak", "mass+bulge+sigma+z"),
        ("PCCSigmaStrongMBSZ", "sigma", "strong_only", "mass+bulge+sigma+z")):
        r = correlation(tier, subset, control)
        put(key, r.pcc, "+.5f")
    integer("NSigma", correlation("sigma", "strong+weak", "mass+bulge+sigma").n)
    integer("NSigmaStrong", correlation("sigma", "strong_only", "mass+bulge+sigma").n)

    fibre = read("05d_fibre_ssfr.csv")
    fibre = fibre[(fibre.scheme == "comparison") & (fibre["sample"] == "all")]
    def fibre_row(control, stat=None):
        rows = fibre[fibre.control == control]
        if stat:
            rows = rows[rows.stat == stat]
        assert len(rows) == 1, control
        return rows.iloc[0]
    for key, control, stat in (
        ("FibreMean", "lz_band_mean_z<0.05", "mean"),
        ("FibreMedian", "mass_only_zslice:LZ_z<0.05", None),
        ("FibreUpperMean", "mass_split:10.0-10.5_z<0.05", "mean"),
        ("FibreNonQ", "decomp_nonquenched_z<0.05", "median"),
        ("FibreQ", "decomp_Q_only_z<0.05", "median"),
        ("LowMassSFR", "mass_only:sfr_fib", None),
        ("LowMassSSFR", "mass_only:ssfr_fib", None)):
        put(key, fibre_row(control, stat).delta, "+.4f")
    fm = fibre_row("lz_band_mean_z<0.05")
    fu = fibre_row("mass_split:10.0-10.5_z<0.05", "mean")
    for prefix, row in (("FibreMean", fm), ("FibreUpper", fu)):
        put(prefix+"Lo", row.delta_lo, "+.4f")
        put(prefix+"Hi", row.delta_hi, "+.4f")
    low = fibre_row("mass_only:ssfr_fib")
    integer("NLowStrong", low.n_strong)
    integer("NLowUnbarred", low.n_unbarred)
    band = main_disc[(main_disc.z < .05) & (main_disc.lgm_tot_p50 >= 9.5) & (main_disc.lgm_tot_p50 < 10.5)]
    a = band[band.bar_class_comparison == "strong"]
    b = band[band.bar_class_comparison == "unbarred"]
    qa, qb = a.quenched_dsfr.mean(), b.quenched_dsfr.mean()
    mean = lambda d: d.specsfr_fib_p50.astype(float).mean()
    aq, bq = mean(a[a.quenched_dsfr]), mean(b[b.quenched_dsfr])
    an, bn = mean(a[~a.quenched_dsfr]), mean(b[~b.quenched_dsfr])
    terms = [(qa-qb)*(bq-bn), qa*(aq-bq), (1-qa)*(an-bn)]
    assert np.isclose(sum(terms), fm.delta, atol=1e-12)
    for key, value in zip(("CompositionTerm", "WithinQTerm", "WithinNonQTerm"), terms):
        put(key, value, "+.5f")
    put("FibreQRatio", qa/qb, ".1f")
    sf = lambda d: d[~d.quenched_dsfr & ~d.green_valley].specsfr_fib_p50.astype(float).median()
    put("FibreStrictSF", sf(a)-sf(b), "+.4f")
    bpt = read("05e_bpt_breakdown.csv").set_index("bar_class")
    for key, label in (("Strong", "strong"), ("Weak", "weak"), ("Unbarred", "unbarred")):
        put("Concentration"+key, bpt.loc[label, "med_ssfr_fib_minus_tot"], "+.3f")
        put("Removed"+key, 100*bpt.loc[label, "frac_removed_agn_clean"], ".0f")
    for key, label in (("AGNContrast", "agn_only:mass_controlled"), ("CompositeContrast", "composite_only:mass_controlled")):
        pair(key, bpt.loc[label, "qfrac"], bpt.loc[label, "qfrac_se"])

    text = "% Generated by src/10_release_tables.py; do not edit numbers by hand.\n"
    text += "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in values.items()) + "\n"
    (paper / "numbers.tex").write_text(text, encoding="utf-8")
    (config.RESULTS / "10_release_numbers.json").write_text(json.dumps(dict(macros=values,
        headline=summaries, egger=egger, mean_decomposition=terms), indent=2) + "\n", encoding="utf-8")

    tab = [r"\begin{table*}\centering", r"\caption{Fiducial redshift-stratified quenched-fraction contrasts. Errors are nominal standard errors; positive and significant sign counts are descriptive.}",
           r"\label{tab:headline}", r"\begin{tabular}{lcccccc}\toprule",
           r"Comparison & Cells & FE & RE & $I^2$ & Positive & Significant $+/-$ \\\midrule"]
    for label, mode in (("Strong bars", "strong_only"), ("All bars", "all"), ("Weak bars", "weak_only")):
        r = summaries[mode]
        tab.append(f"{label} & {r['n']} & ${r['fe']:+.4f} \\pm {r['fe_se']:.4f}$ & "
                   f"${r['re']:+.4f} \\pm {r['re_se']:.4f}$ & {r['I2']:.2f} & "
                   f"{r['positive']}/{r['n']} & {r['sig_positive']}/{r['sig_negative']} " + r"\\")
    tab += [r"\bottomrule\end{tabular}\end{table*}"]
    (paper/"table_headline.tex").write_text("\n".join(tab)+"\n", encoding="utf-8")
    tab = [r"\begin{table}\centering", r"\caption{Quenched-definition sensitivity, comparison scheme with redshift control.}",
           r"\label{tab:robust}", r"\begin{tabular}{lccc}\toprule",
           r"Cut & FE & RE & Significant $+/-$ \\\midrule"]
    robust = read("05b_quench_robustness.csv")
    for label, mode in (("Strong bars", "strong_only"), ("All bars", "all")):
        tab.append(r"\multicolumn{4}{l}{\emph{"+label+r"}} " + "\\\\")
        for cut in config.SSFR_QUENCHED_VARIANTS:
            r = robust[(robust.barmode == mode) & (robust.quench_def == f"ssfr<{cut}")].iloc[0]
            tab.append(f"${cut}$ & ${r.wmean_delta_q:+.4f} \\pm {r.wmean_delta_q_se:.4f}$ & "
                       f"${r.re_mean_delta_q:+.4f} \\pm {r.re_mean_delta_q_se:.4f}$ & "
                       f"{r.n_cells_sig_pos}/{r.n_cells_sig_neg} " + r"\\")
    tab += [r"\bottomrule\end{tabular}\end{table}"]
    (paper/"table_robustness.tex").write_text("\n".join(tab)+"\n", encoding="utf-8")
    print(f"Generated {len(values)} numerical macros and two tables.")


if __name__ == "__main__":
    main()
