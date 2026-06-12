# -*- coding: utf-8 -*-
"""Figure 2 - empirical vs nominal conformal coverage on both classes.
Reuses the exact Table 4 computation (table4_coverage.py), so the figure and the
table are guaranteed identical. Single-column width (84 mm), 10-pt font, no outer
border, per IJIES format. Run: python figure2_coverage.py"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import table4_coverage as T4

plt.rcParams.update({"font.size": 10, "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.8})

LV = [0.20, 0.10, 0.05]
NOM = [80, 90, 95]
s = T4.sfrc_coverage()
cc = T4.concrete_split_coverage()

def mn(v): return [100*np.mean(v[a]) for a in LV]
def sd(v): return [100*np.std(v[a]) for a in LV]

series = [
    ("Concrete (split)",   mn(cc),         sd(cc),         "#1f3864", "o"),
    ("SFRC (split)",       mn(s["split"]), sd(s["split"]), "#3b7fc4", "s"),
    ("SFRC (normalized)",  mn(s["norm"]),  sd(s["norm"]),  "#e08214", "^"),
    ("SFRC (Mondrian)",    mn(s["mond"]),  sd(s["mond"]),  "#4a9b5e", "D"),
]

fig, ax = plt.subplots(figsize=(85/25.4, 88/25.4))
fig.subplots_adjust(left=0.165, right=0.97, top=0.85, bottom=0.355)
ax.set_title("Conformal interval coverage\nacross both composite classes",
             fontsize=10, fontweight="bold", color="#1f3864", linespacing=1.12, pad=5)
ax.plot([76, 99], [76, 99], ls="--", lw=0.8, color="#9aa3b2", label="Ideal", zorder=1)
mk_labels = ["Concrete (split)", "SFRC (split)", "SFRC (normalized)", "SFRC (Mondrian)"]
DX = [-0.37, -0.12, 0.12, 0.37]
for (lbl0, m, e, c, mk), lbl, dx in zip(series, mk_labels, DX):
    xs = [n + dx for n in NOM]
    ax.errorbar(xs, m, yerr=e, marker=mk, ms=3.2, lw=0.85, elinewidth=0.7, capsize=1.6,
                color=c, label=lbl, zorder=3)

ax.set_xlabel("Nominal coverage (%)"); ax.set_ylabel("Empirical coverage (%)")
ax.set_xticks(NOM); ax.set_xlim(77, 98); ax.set_ylim(72.5, 100)
ax.grid(True, ls=":", lw=0.4, color="#dde1e8", zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=10,
           frameon=False, columnspacing=1.1, handlelength=1.5, handletextpad=0.4, labelspacing=0.5)

fig.savefig("figure2.png", dpi=400, facecolor="white")
print("wrote figure2.png  | concrete mean:", [round(v,1) for v in mn(cc)])
