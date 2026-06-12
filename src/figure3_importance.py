# -*- coding: utf-8 -*-
"""Figure 3 - permutation feature importance for the SFRC predictive core.
Importance = increase in test MAE (MPa) when a feature is shuffled, averaged over
the ten leakage-free splits (error bars = SD across splits). Single-column width
(84 mm), 10-pt font, no outer border, per IJIES format. Run: python figure3_importance.py"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor as GB
from sklearn.inspection import permutation_importance
import common as C

plt.rcParams.update({"font.size": 10, "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.8})

df, X, y, gid = C.load_sfrc()
per_split = []
for s in range(10):
    tr, ca, te = C.sfrc_split_masks(gid, s)
    fit = tr | ca                          # train+calibration (80%), same as Table 3 for SFRC
    core = GB(random_state=C.RNG).fit(X[fit], y[fit])
    r = permutation_importance(core, X[te], y[te], n_repeats=20,
                               random_state=C.RNG, scoring="neg_mean_absolute_error")
    per_split.append(r.importances_mean)
per_split = np.array(per_split)           # (10 splits, 12 features), units = MAE increase (MPa)
mean_imp = per_split.mean(0)
sd_imp = per_split.std(0)

order = np.argsort(mean_imp)              # ascending -> most important ends up on top in barh
feats = [C.SFRC_FEATURES[i] for i in order]
vals = mean_imp[order]; errs = sd_imp[order]
colors = ["#e08214" if f == "Temperature" else "#27416b" for f in feats]

fig, ax = plt.subplots(figsize=(84/25.4, 85/25.4))
fig.subplots_adjust(left=0.31, right=0.965, top=0.835, bottom=0.145)
fig.suptitle("Permutation feature importance\nfor the SFRC predictive core",
             fontsize=10, fontweight="bold", color="#1f3864", x=0.5, y=0.99, linespacing=1.12)
ax.barh(range(len(feats)), vals, color=colors, height=0.74,
        xerr=errs, error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444444"))
ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats)
ax.set_ylim(-0.7, len(feats) - 0.3)
ax.set_xlim(0, 27.5)
ax.set_xlabel("Increase in MAE (MPa)")
ax.grid(True, axis="x", ls=":", lw=0.4, color="#dde1e8", zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

fig.savefig("figure3.png", dpi=400, facecolor="white")
print("wrote figure3.png")
print("order (top->bottom):", list(reversed(feats)))
print("mean MAE increase:", [round(v, 2) for v in reversed(vals)])
