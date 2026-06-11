# -*- coding: utf-8 -*-
"""Section 5.3 stress test on a third, NON-cementitious class (Class C):
fibre-reinforced polymer textile laminates (TPCM, Malashin et al. 2024, ref [32]).
420 specimens / 32 formulation groups; target = warp tensile strength (MPa).
Demonstrates the leakage gap, conformal under-coverage under extreme extrapolation,
and a positive-but-not-established novelty-error coupling."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as GB
from sklearn.metrics import r2_score, mean_absolute_error
import common as C

def rand_masks(n, seed):
    rng = np.random.RandomState(seed); idx = rng.permutation(n)
    m = [np.zeros(n, bool) for _ in range(3)]
    for k, sl in enumerate((idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):])):
        m[k][sl] = True
    return m

if __name__ == "__main__":
    df, X, y, groups, ug = C.load_polymer()
    print(f"=== Class C (polymer): {len(df)} specimens, {len(ug)} formulation groups, "
          f"{X.shape[1]} features; target mean {y.mean():.0f} MPa, SD {y.std():.0f} MPa ===")

    print("\n-- leakage: random vs grouped (GB core, 10 splits) --")
    for lbl, fn in [("grouped", lambda s: C.grouped_masks_from_series(groups, ug, s)),
                    ("random ", lambda s: rand_masks(len(y), s))]:
        r2 = []
        for s in range(10):
            tr, ca, te = fn(s); fit = tr | ca
            m = GB(random_state=C.RNG).fit(X[fit], y[fit]); r2.append(r2_score(y[te], m.predict(X[te])))
        print(f"   {lbl}: R2 = {np.mean(r2):+.3f} +- {np.std(r2):.3f}")

    print("\n-- split-conformal coverage under extrapolation (10 splits) --")
    for a, lbl in [(0.20, "80%"), (0.10, "90%"), (0.05, "95%")]:
        cov = []
        for s in range(10):
            tr, ca, te = C.grouped_masks_from_series(groups, ug, s)
            core = GB(random_state=C.RNG).fit(X[tr], y[tr]); rc = np.abs(y[ca] - core.predict(X[ca]))
            cov.append(C.split_conformal(rc, core.predict(X[te]), y[te], a)[0])
        print(f"   {lbl}: {100*np.mean(cov):.0f} +- {100*np.std(cov):.0f}")

    print("\n-- novelty-error coupling (deployed train+cal core, test set) --")
    knn, iso, near, far = [], [], [], []
    for s in range(10):
        tr, ca, te = C.grouped_masks_from_series(groups, ug, s); fit = tr | ca
        core = GB(random_state=C.RNG).fit(X[fit], y[fit]); err = np.abs(y[te] - core.predict(X[te]))
        nk, ni = C.novelty_scores(X[fit], X[te]); knn.append(C.rho_ne(nk, err)); iso.append(C.rho_ne(ni, err))
        med = np.median(nk); near.append(err[nk <= med].mean()); far.append(err[nk > med].mean())
    knn, iso, near, far = map(np.array, (knn, iso, near, far)); gap = far - near
    print(f"   rho_NE kNN = {knn.mean():+.2f} +- {knn.std():.2f}   iso = {iso.mean():+.2f} +- {iso.std():.2f}")
    print(f"   MAE near = {near.mean():.0f}   far = {far.mean():.0f} MPa   gap positive {int((gap>0).sum())}/10")
    print("   -> coupling positive but, on 32 groups, across-split interval spans zero: SUGGESTIVE, not established.")
