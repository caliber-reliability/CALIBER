# -*- coding: utf-8 -*-
"""Figure 4 (cross-class summary) - full-width single figure, 10-pt unified font,
no (a)/(b) labels, per-panel legends placed above each plot, no internal titles
(described by the caption), no outer border. Left: model accuracy by class
(table3_accuracy.py). Right: novelty-error coupling (table5_novelty.py)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
NAVY, ORANGE = "#27416b", "#e3852b"

ACC = {"Mean": (-0.006, -0.046), "Linear": (0.593, 0.616), "SVR": (0.786, 0.837),
       "RandomForest": (0.849, 0.857), "GradBoost (core)": (0.855, 0.844)}
RHO = {"kNN\nnovelty": (0.277, 0.045, 0.115, 0.149),
       "Isolation\nforest": (0.313, -0.032, 0.095, 0.168)}   # (concrete, sfrc, sd_c, sd_s)
LEG = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
           fontsize=10, columnspacing=1.1, handlelength=1.0, handletextpad=0.3, borderpad=0.1)

def main(path):
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(ACC)); w = 0.38

    axA.bar(x-w/2, [v[0] for v in ACC.values()], w, label="Concrete (Class A)", color=NAVY)
    axA.bar(x+w/2, [v[1] for v in ACC.values()], w, label="SFRC (Class B)", color=ORANGE)
    axA.axhline(0, color="#333", lw=0.7); axA.set_xticks(x)
    axA.set_xticklabels(ACC.keys(), rotation=22, ha="right", rotation_mode="anchor")
    axA.set_ylim(-0.1, 1.0); axA.set_ylabel("Test R²")
    axA.legend(**LEG)
    axA.grid(axis="y", ls=":", lw=0.5, color="#d8dce3"); axA.set_axisbelow(True)

    xb = np.arange(len(RHO))
    axB.axhspan(-0.15, 0.15, color="#000", alpha=0.06)
    axB.bar(xb-w/2, [v[0] for v in RHO.values()], w, yerr=[v[2] for v in RHO.values()],
            capsize=2.5, label="Concrete (Class A)", color=NAVY, error_kw=dict(ecolor="#111", lw=0.9))
    axB.bar(xb+w/2, [v[1] for v in RHO.values()], w, yerr=[v[3] for v in RHO.values()],
            capsize=2.5, label="SFRC (Class B)", color=ORANGE, error_kw=dict(ecolor="#111", lw=0.9))
    axB.axhline(0, color="#333", lw=0.8); axB.set_xticks(xb); axB.set_xticklabels(RHO.keys())
    axB.set_xlim(-0.65, 1.65); axB.set_ylim(-0.25, 0.47)
    axB.set_ylabel("Novelty–error ρ")
    axB.legend(**LEG)
    axB.grid(axis="y", ls=":", lw=0.5, color="#d8dce3"); axB.set_axisbelow(True)

    for ax in (axA, axB):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(pad=0.5, w_pad=5.5)
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    print("wrote", path)

if __name__ == "__main__":
    import os; main(os.path.join(os.path.dirname(__file__), os.pardir, "results", "figure4.png"))
