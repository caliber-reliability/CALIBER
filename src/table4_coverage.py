# -*- coding: utf-8 -*-
"""Regenerates Table 4 — empirical conformal coverage. SFRC split/normalized/
Mondrian variants and concrete split-conformal, mean +/- SD over 10 splits."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as GB
import common as C

LEVELS = [(0.20, "80%"), (0.10, "90%"), (0.05, "95%")]

def sfrc_coverage():
    _, X, y, gid = C.load_sfrc()
    out = {v: {a: [] for a, _ in LEVELS} for v in ("split", "norm", "mond")}
    for s in range(10):
        tr, ca, te = C.sfrc_split_masks(gid, s)
        core = GB(random_state=C.RNG).fit(X[tr], y[tr])
        pc, pt = core.predict(X[ca]), core.predict(X[te]); rc = np.abs(y[ca] - pc)
        for a, _ in LEVELS:
            out["split"][a].append(C.split_conformal(rc, pt, y[te], a, radius_fn=C.quantile_radius)[0])
            out["norm"][a].append(C.normalized_conformal(core, X[tr], y[tr], X[ca], y[ca], X[te], y[te], pc, pt, a, radius_fn=C.quantile_radius)[0])
            out["mond"][a].append(C.mondrian_conformal(X[ca], y[ca], X[te], y[te], pc, pt, a, radius_fn=C.quantile_radius)[0])
    return out

def concrete_split_coverage():
    df, g, ug = C.load_concrete(); X = df[C.CONCRETE_FEAT].values.astype(float); y = df['strength'].values.astype(float)
    res = {a: [] for a, _ in LEVELS}
    for s in range(10):
        sp = C.concrete_split_masks(g, ug, s); tr, ca, te = sp=='train', sp=='cal', sp=='test'
        core = GB(random_state=C.RNG).fit(X[tr], y[tr]); rc = np.abs(y[ca] - core.predict(X[ca])); pt = core.predict(X[te])
        for a, _ in LEVELS:
            res[a].append(C.split_conformal(rc, pt, y[te], a)[0])
    return res

if __name__ == "__main__":
    print("=== Table 4: conformal coverage (10 splits, mean +/- SD) ===")
    s = sfrc_coverage(); cc = concrete_split_coverage()
    print(f"{'Nominal':<9}{'SFRC split':>16}{'SFRC norm':>16}{'SFRC Mondrian':>16}{'Concrete split':>16}")
    for a, lbl in LEVELS:
        f = lambda v: f"{100*np.mean(v):.1f} +-{100*np.std(v):.1f}"
        print(f"{lbl:<9}{f(s['split'][a]):>16}{f(s['norm'][a]):>16}{f(s['mond'][a]):>16}{f(cc[a]):>16}")
