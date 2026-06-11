# -*- coding: utf-8 -*-
"""Regenerates Table 3 — predictive accuracy of 8 models on both classes,
mean +/- SD over 10 leakage-free splits. Class A on the training partition
(conformal-core convention); Class B on train+calibration."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score, mean_absolute_error
import common as C

def evaluate():
    dfA, gA, ugA = C.load_concrete()
    XA = dfA[C.CONCRETE_FEAT].values.astype(float); yA = dfA['strength'].values.astype(float)
    _, XB, yB, gidB = C.load_sfrc()
    names = list(C.model_zoo().keys())
    A = {n: [] for n in names}; B = {n: [] for n in names}
    for s in range(10):
        spA = C.concrete_split_masks(gA, ugA, s); trA, teA = spA == 'train', spA == 'test'
        for n, m in C.model_zoo().items():
            m.fit(XA[trA], yA[trA]); p = m.predict(XA[teA])
            A[n].append([r2_score(yA[teA], p), mean_absolute_error(yA[teA], p)])
        a, b, c = C.sfrc_split_masks(gidB, s); fit = a | b
        for n, m in C.model_zoo().items():
            m.fit(XB[fit], yB[fit]); p = m.predict(XB[c])
            B[n].append([r2_score(yB[c], p), mean_absolute_error(yB[c], p)])
    print(f"{'Model':<26}{'Class A R2':>16}{'Class A MAE':>16}{'Class B R2':>16}{'Class B MAE':>16}")
    for n in names:
        a, b = np.array(A[n]), np.array(B[n])
        print(f"{n:<26}{a[:,0].mean():>8.3f} +-{a[:,0].std():<5.3f}{a[:,1].mean():>8.2f} +-{a[:,1].std():<5.2f}"
              f"{b[:,0].mean():>8.3f} +-{b[:,0].std():<5.3f}{b[:,1].mean():>8.2f} +-{b[:,1].std():<5.2f}")

if __name__ == "__main__":
    print("=== Table 3: predictive accuracy (10 splits, mean +/- SD) ===")
    evaluate()
