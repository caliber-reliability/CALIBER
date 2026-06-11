# -*- coding: utf-8 -*-
"""Regenerates Figure 4 (cross-class summary) WITHOUT (a)/(b) panel labels,
per IJIES format. Left: model accuracy by class; right: novelty-error coupling."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
NAVY, ORANGE = "#27416b", "#e3852b"

# values are produced by table3_accuracy.py and table5_novelty.py
ACC = {"Mean": (-0.006, -0.046), "Linear": (0.593, 0.616), "SVR": (0.786, 0.837),
       "RandomForest": (0.849, 0.857), "GradBoost\n(core)": (0.855, 0.844)}
RHO = {"kNN\nnovelty": (0.277, 0.045, 0.115, 0.149),
       "Isolation\nforest": (0.313, -0.032, 0.095, 0.168)}  # (concrete, sfrc, sd_c, sd_s)

def main(path="../results/figure4.png"):
    plt.rcParams.update({"font.size": 15})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.55, 5.79), dpi=150)
    x = np.arange(len(ACC)); w = 0.38
    axA.bar(x-w/2, [v[0] for v in ACC.values()], w, label="Concrete (Class A)", color=NAVY)
    axA.bar(x+w/2, [v[1] for v in ACC.values()], w, label="SFRC (Class B)", color=ORANGE)
    axA.axhline(0, color="#333", lw=1); axA.set_xticks(x); axA.set_xticklabels(ACC.keys())
    axA.set_ylim(-0.1, 1.0); axA.set_ylabel("Coefficient of determination  R2")
    axA.set_title("Prediction accuracy by model and class", fontweight="bold", color=NAVY)
    axA.legend(loc="upper left"); axA.grid(axis="y", ls=":", alpha=0.5)
    xb = np.arange(len(RHO))
    axB.axhspan(-0.15, 0.15, color="#000", alpha=0.07)
    axB.bar(xb-w/2, [v[0] for v in RHO.values()], w, yerr=[v[2] for v in RHO.values()],
            capsize=5, label="Concrete (Class A)", color=NAVY, error_kw=dict(ecolor="#111", lw=1.6))
    axB.bar(xb+w/2, [v[1] for v in RHO.values()], w, yerr=[v[3] for v in RHO.values()],
            capsize=5, label="SFRC (Class B)", color=ORANGE, error_kw=dict(ecolor="#111", lw=1.6))
    axB.axhline(0, color="#333", lw=1.2); axB.set_xticks(xb); axB.set_xticklabels(RHO.keys())
    axB.set_ylim(-0.25, 0.45); axB.set_ylabel("Spearman corr.  (novelty score, |error|)")
    axB.set_title("Domain detector: works on concrete, fails on SFRC", fontweight="bold", color=NAVY)
    axB.text(1.42, 0.10, "no useful\nsignal", color="#666", ha="center", fontsize=13)
    axB.legend(loc="upper right"); axB.grid(axis="y", ls=":", alpha=0.5)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", path)

if __name__ == "__main__":
    import os; main(os.path.join(os.path.dirname(__file__), os.pardir, "results", "figure4.png"))
