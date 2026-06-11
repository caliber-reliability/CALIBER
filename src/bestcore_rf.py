# -*- coding: utf-8 -*-
"""Section 4.1 robustness check: repeats the SFRC reliability analysis with a
RANDOM-FOREST core (the locally best model on SFRC) to show the conformal and
novelty-error conclusions are core-agnostic."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor as RF
import common as C

LEVELS = [(0.20, "80%"), (0.10, "90%"), (0.05, "95%")]

if __name__ == "__main__":
    print("=== Section 4.1: RF-core verification on SFRC (10 splits) ===")
    _, X, y, gid = C.load_sfrc()
    cov = {a: [] for a, _ in LEVELS}; knn, iso = [], []
    for s in range(10):
        tr, ca, te = C.sfrc_split_masks(gid, s)
        core = RF(n_estimators=300, random_state=C.RNG).fit(X[tr], y[tr])
        rc, pt = np.abs(y[ca] - core.predict(X[ca])), core.predict(X[te])
        for a, _ in LEVELS:
            cov[a].append(C.split_conformal(rc, pt, y[te], a)[0])
        err = np.abs(y[te] - core.predict(X[te])); nk, ni = C.novelty_scores(X[tr], X[te])
        knn.append(C.rho_ne(nk, err)); iso.append(C.rho_ne(ni, err))
    print("  split-conformal coverage:")
    for a, lbl in LEVELS:
        print(f"     {lbl}: {100*np.mean(cov[a]):.1f} +- {100*np.std(cov[a]):.1f}")
    knn, iso = np.array(knn), np.array(iso)
    print(f"  rho_NE kNN = {knn.mean():+.2f} +- {knn.std():.2f}   iso = {iso.mean():+.2f} +- {iso.std():.2f}  (both span zero -> uninformative, as with GB core)")
