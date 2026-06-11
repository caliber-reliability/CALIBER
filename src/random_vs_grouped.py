# -*- coding: utf-8 -*-
"""Reproduces the Section 5.2 leakage demonstration: record-level random splitting
vs composition-grouped splitting, identical gradient-boosting core, 10 splits each."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as GB
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
import common as C

def rand_masks(n, seed):
    rng = np.random.RandomState(seed); idx = rng.permutation(n)
    tr = np.zeros(n, bool); tr[idx[:int(.6*n)]] = True
    te = np.zeros(n, bool); te[idx[int(.8*n):]] = True
    return tr, te

def run_concrete():
    df, g, ug = C.load_concrete(); X = df[C.CONCRETE_FEAT].values.astype(float); y = df['strength'].values.astype(float)
    R, Gp = [], []
    for s in range(10):
        tr, te = rand_masks(len(y), s); m = GB(random_state=C.RNG).fit(X[tr], y[tr])
        R.append([r2_score(y[te], m.predict(X[te])), mean_absolute_error(y[te], m.predict(X[te]))])
        sp = C.concrete_split_masks(g, ug, s); m = GB(random_state=C.RNG).fit(X[sp=='train'], y[sp=='train'])
        Gp.append([r2_score(y[sp=='test'], m.predict(X[sp=='test'])), mean_absolute_error(y[sp=='test'], m.predict(X[sp=='test']))])
    return np.array(R), np.array(Gp)

def run_sfrc():
    _, X, y, gid = C.load_sfrc(); R, Gp = [], []
    for s in range(10):
        tr, te = rand_masks(len(y), s); m = GB(random_state=C.RNG).fit(X[tr], y[tr]); R.append(r2_score(y[te], m.predict(X[te])))
        a, b, c = C.sfrc_split_masks(gid, s); m = GB(random_state=C.RNG).fit(X[a], y[a]); Gp.append(r2_score(y[c], m.predict(X[c])))
    return np.array(R), np.array(Gp)

if __name__ == "__main__":
    print("=== Section 5.2: random vs grouped (train-only core) ===")
    R, G = run_concrete()
    t, p = stats.ttest_ind(R[:,0], G[:,0], equal_var=False)
    print(f"CONCRETE  random R2 = {R[:,0].mean():.3f} +- {R[:,0].std():.3f}  MAE = {R[:,1].mean():.2f}")
    print(f"          grouped R2 = {G[:,0].mean():.3f} +- {G[:,0].std():.3f}  MAE = {G[:,1].mean():.2f}   (Welch p = {p:.2e})")
    R, G = run_sfrc()
    print(f"SFRC      random R2 = {R.mean():.3f} +- {R.std():.3f}   grouped R2 = {G.mean():.3f} +- {G.std():.3f}")
