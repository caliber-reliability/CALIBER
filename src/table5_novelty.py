# -*- coding: utf-8 -*-
"""Regenerates Table 5 — novelty-error coupling criterion (rho_NE) and the
near/far MAE ablation on both classes.

rho_NE is the ten-split mean +/- SD with the Holm-significant vote (rule of
Sec. 3.5.3: a detector is informative iff positive and Holm-significant in
>= 6/10 splits). For the near/far MAE the table follows the manuscript's
reporting exactly: the concrete (Class A) figures are the primary split
(seed 42, train-only core), and the SFRC (Class B) figures are the ten-split
mean (train+calibration core); the error-gap sign is counted over the ten
splits for both classes."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as GB
import common as C

def rho_over_splits(X, y, masks_fn, core_on_traincal):
    knn, iso, holm, gap_pos = [], [], [], 0
    for s in range(10):
        tr, ca, te = masks_fn(s); fit = (tr | ca) if core_on_traincal else tr
        core = GB(random_state=C.RNG).fit(X[fit], y[fit]); err = np.abs(y[te] - core.predict(X[te]))
        nk, ni = C.novelty_scores(X[fit], X[te])
        rk, pk = C.rho_ne_onesided_p(nk, err); ri, pi = C.rho_ne_onesided_p(ni, err)
        sk, si = C.holm_two(pk, pi); knn.append(rk); iso.append(ri); holm.append([sk, si])
        med = np.median(nk); gap_pos += int(err[nk > med].mean() > err[nk <= med].mean())
    return np.array(knn), np.array(iso), np.array(holm), gap_pos

def near_far_primary(X, y, masks_fn, tc):
    tr, ca, te = masks_fn(42); fit = (tr | ca) if tc else tr
    core = GB(random_state=C.RNG).fit(X[fit], y[fit]); err = np.abs(y[te] - core.predict(X[te]))
    nk, _ = C.novelty_scores(X[fit], X[te]); med = np.median(nk)
    return err[nk <= med].mean(), err[nk > med].mean()

def near_far_mean(X, y, masks_fn, tc):
    n, f = [], []
    for s in range(10):
        tr, ca, te = masks_fn(s); fit = (tr | ca) if tc else tr
        core = GB(random_state=C.RNG).fit(X[fit], y[fit]); err = np.abs(y[te] - core.predict(X[te]))
        nk, _ = C.novelty_scores(X[fit], X[te]); med = np.median(nk)
        n.append(err[nk <= med].mean()); f.append(err[nk > med].mean())
    return np.mean(n), np.mean(f)

if __name__ == "__main__":
    print("=== Table 5: novelty-error coupling criterion ===")
    dfA, gA, ugA = C.load_concrete()
    XA = dfA[C.CONCRETE_FEAT].values.astype(float); yA = dfA['strength'].values.astype(float)
    _, XB, yB, gidB = C.load_sfrc()
    A_masks = lambda s: (lambda sp: (sp == 'train', sp == 'cal', sp == 'test'))(C.concrete_split_masks(gA, ugA, s))
    B_masks = lambda s: C.sfrc_split_masks(gidB, s)
    for cls, X, y, mfn, tc, nf in [
            ("Class A (concrete)", XA, yA, A_masks, False, "primary"),
            ("Class B (SFRC)",     XB, yB, B_masks, True,  "ten-split mean")]:
        k, i, H, gap = rho_over_splits(X, y, mfn, tc)
        near, far = (near_far_primary if nf == "primary" else near_far_mean)(X, y, mfn, tc)
        print(f"\n{cls}:")
        print(f"  rho_NE kNN  = {k.mean():+.3f} +- {k.std():.3f}   Holm-significant {H[:,0].sum()}/10")
        print(f"  rho_NE iso  = {i.mean():+.3f} +- {i.std():.3f}   Holm-significant {H[:,1].sum()}/10")
        print(f"  MAE near = {near:.2f}    MAE far = {far:.2f}  ({nf})    error gap positive {gap}/10")
        print(f"  criterion verdict: {'INFORMATIVE' if max(H[:,0].sum(), H[:,1].sum()) >= 6 else 'WITHHELD (coupling not demonstrated)'}")
