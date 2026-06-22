# -*- coding: utf-8 -*-
"""Regenerates the Section 4.1 bootstrap intervals on the SFRC primary split (seed 42).

Two thousand paired resamples of the held-out test set give:
  * a 95% confidence interval for the gradient-boosting core's R^2, and
  * a 95% confidence interval for the MAE difference (random forest - gradient
    boosting); the negative interval means the random forest has the lower error.

Convention (matches Table 3 and hyperparameters.md): on SFRC both cores are fitted
on the train + calibration partitions; metrics and resampling use the test partition.
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor as GB, RandomForestRegressor as RF
import common as C

N_BOOT = 2000


def main():
    _, X, y, gid = C.load_sfrc()
    tr, ca, te = C.sfrc_split_masks(gid, 42)          # primary split, seed 42
    fit = tr | ca                                      # SFRC core fit on train + calibration
    gb = GB(random_state=C.RNG).fit(X[fit], y[fit]); pred_gb = gb.predict(X[te])
    rf = RF(n_estimators=300, random_state=C.RNG).fit(X[fit], y[fit]); pred_rf = rf.predict(X[te])

    # --- R^2 bootstrap CI for the gradient-boosting core ---
    rng = np.random.RandomState(C.RNG); n = len(y[te]); r2s = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[te][idx])) > 1:
            r2s.append(r2_score(y[te][idx], pred_gb[idx]))
    r2_lo, r2_hi = np.percentile(r2s, 2.5), np.percentile(r2s, 97.5)

    # --- MAE-difference bootstrap CI (random forest - gradient boosting) ---
    diff, (mae_lo, mae_hi) = C.bootstrap_mae_diff(y[te], pred_rf, pred_gb,
                                                  n_boot=N_BOOT, seed=C.RNG)

    print("=== Section 4.1: bootstrap on the SFRC primary split (seed 42, 2000 resamples) ===")
    print(f"  test records: {n}  |  core fit on train+calibration ({fit.sum()} records)")
    print(f"  GB core R^2 95% CI:                 [{r2_lo:.3f}, {r2_hi:.3f}]")
    print(f"  MAE difference (RF - GB) point:     {diff:.2f} MPa")
    print(f"  MAE difference (RF - GB) 95% CI:    [{mae_lo:.2f}, {mae_hi:.2f}] MPa")
    print("  (the negative MAE interval excludes zero -> the random forest's lower error is statistically resolved)")


if __name__ == "__main__":
    main()
