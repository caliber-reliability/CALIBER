# -*- coding: utf-8 -*-
"""
CALIBER — shared loaders, splits, models and reliability primitives.

Every experiment script in this repository imports from here, so all reported
numbers come from one verified pipeline. All runs are deterministic:
the primary split uses seed 42 and the ten evaluation splits use seeds 0-9.

Datasets (see data/README.md for sources and licences):
  * Class A  concrete         data/Concrete_Data.csv        (UCI, 1030 -> 1005 rows)
  * Class B  SFRC             data/SFRC_Data_v1.xlsx        (Mendeley 10.17632/hjrfgys29n.1)
  * Class C  polymer (stress) data/Polymer_TPCM.xlsx        (TPCM, Malashin et al. 2024)
"""
import os, math
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              IsolationForest)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

RNG = 42
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, os.pardir, "data")
CONCRETE_CSV = os.path.join(_DATA, "Concrete_Data.csv")
SFRC_XLSX    = os.path.join(_DATA, "SFRC_Data_v1.xlsx")
POLYMER_XLSX = os.path.join(_DATA, "Polymer_TPCM.xlsx")


# ============================================================ MODELS (Table 3)
def model_zoo(seed=RNG):
    """Eight predictors with the exact hyperparameters used in the paper.
    Gradient boosting (HistGradientBoostingRegressor, defaults) is the CALIBER core."""
    return {
        "Mean":         DummyRegressor(strategy="mean"),
        "Linear":       make_pipeline(StandardScaler(), LinearRegression()),
        "SVR":          make_pipeline(StandardScaler(), SVR(C=100, gamma="scale")),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed),
        "GradBoost":    HistGradientBoostingRegressor(random_state=seed),         # <- core
        "XGBoost":      _xgb(seed),
        "LightGBM":     _lgbm(seed),
        "MLP":          make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000,
                                       early_stopping=True, random_state=seed)),
    }

def _xgb(seed):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                        subsample=0.9, colsample_bytree=0.9, random_state=seed,
                        verbosity=0)

def _lgbm(seed):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=31,
                         random_state=seed, verbose=-1)


# ========================================================= CLASS A : CONCRETE
CONCRETE_FEAT = ['cement', 'slag', 'fly_ash', 'water', 'superplast',
                 'coarse_agg', 'fine_agg', 'age']
CONCRETE_COMP = CONCRETE_FEAT[:-1]                       # composition = all but age

def load_concrete():
    df = pd.read_csv(CONCRETE_CSV)
    df.columns = CONCRETE_FEAT + ['strength']
    df = df.drop_duplicates().reset_index(drop=True)     # 1030 -> 1005 rows
    groups = df[CONCRETE_COMP].astype(str).agg('|'.join, axis=1)
    ug = list(dict.fromkeys(groups.tolist()))            # 428 compositions, stable order
    return df, groups, ug

def concrete_split_masks(groups, ug, seed):
    """Composition-grouped 60/20/20 split -> array of 'train'/'cal'/'test'."""
    rng = np.random.RandomState(seed)
    u = ug.copy(); rng.shuffle(u); n = len(u)
    g_tr = set(u[:int(.6 * n)]); g_ca = set(u[int(.6 * n):int(.8 * n)]); g_te = set(u[int(.8 * n):])
    return groups.map(lambda g: 'train' if g in g_tr else ('cal' if g in g_ca else 'test')).values


# ============================================================= CLASS B : SFRC
SFRC_FEATURES = ["Length", "Diameter", "Vf", "Temperature", "Heating-rate",
                 "SF", "FA", "MK", "Slag", "GGBFS", "Microsilica", "Nanosilica"]
SFRC_GROUPKEY = ["Length", "Diameter", "Vf", "SF", "FA", "MK", "Slag",
                 "GGBFS", "Microsilica", "Nanosilica"]                 # composition (no T, no rate)
SFRC_TARGET = "Fc"

def load_sfrc():
    df = pd.read_excel(SFRC_XLSX)
    df.columns = [c.strip() for c in df.columns]
    df = df[SFRC_FEATURES + [SFRC_TARGET]].copy()
    df["gid"] = df.groupby(SFRC_GROUPKEY, dropna=False).ngroup()       # 158 groups
    X = df[SFRC_FEATURES].values.astype(float)
    y = df[SFRC_TARGET].values.astype(float)
    return df, X, y, df["gid"].values

def sfrc_split_masks(gid, seed):
    """Composition-grouped 60/20/20 -> (train, cal, test) boolean masks."""
    g = np.unique(gid); rs = np.random.RandomState(seed); rs.shuffle(g)
    n1, n2 = int(.6 * len(g)), int(.8 * len(g))
    tr, ca, te = set(g[:n1]), set(g[n1:n2]), set(g[n2:])
    m = lambda S: np.array([x in S for x in gid])
    return m(tr), m(ca), m(te)


# ========================================================== CLASS C : POLYMER
def load_polymer():
    """TPCM textile-polymer laminates: 420 specimens, 32 formulation groups.
    Target = warp tensile strength (MPa). Returns df, X, y, groups(Series), ug(list)."""
    df = pd.read_excel(POLYMER_XLSX)
    TARGET = 'ОСНОВНЫЕ СВОЙСТВА ПКМ: Прочность на растяжение по основе, МПа'
    CAT = ['ПКМ', 'ТКАНЬ: Вид плетения', 'ТЕХНОЛОГИЯ ФОРМИРОВАНИЯ: Название', 'СВЯЗУЮЩЕЕ: Тип']
    NUM = {
        'fabric_tensile': 'ОСНОВНЫЕ СВОЙСТВА ТКАНИ: Прочность при растяжении по основе, МПа',
        'fabric_mod':    'ОСНОВНЫЕ СВОЙСТВА ТКАНИ: Модуль упругости при растяжении по основе, ГПа',
        'fabric_areal':  'ОСНОВНЫЕ СВОЙСТВА ТКАНИ: Поверхностная плотность, г/м2',
        'fabric_thick':  'ОСНОВНЫЕ СВОЙСТВА ТКАНИ: Толщина, мм',
        'fibre_tensile': 'ОСНОВНЫЕ СВОЙСТВА НИТИ (ОСНОВА): Прочность при растяжении волокна, МПа',
        'fibre_mod':     'ОСНОВНЫЕ СВОЙСТВА НИТИ (ОСНОВА): Модуль упругости волокна при растяжении, ГПа',
        'fibre_tex':     'ОСНОВНЫЕ СВОЙСТВА НИТИ (ОСНОВА): Плотность, текс',
        'fibre_fil_d':   'ОСНОВНЫЕ СВОЙСТВА НИТИ (ОСНОВА): Диаметр филаментов, мкм',
        'binder_tensile':'ОСНОВНЫЕ СВОЙСТВА СВЯЗУЮЩЕГО: Прочность на растяжение, МПа',
        'binder_mod':    'ОСНОВНЫЕ СВОЙСТВА СВЯЗУЮЩЕГО: Модуль упругости при растяжении, ГПа',
    }
    RATIO = 'ТЕХНОЛОГИЯ ФОРМИРОВАНИЯ: Соотношение связующего к армирующему наполнителю'
    def ratio(s):
        try:
            a, b = str(s).split('/'); return float(a) / (float(a) + float(b))
        except Exception:
            return np.nan
    feat = pd.DataFrame({k: pd.to_numeric(df[c], errors='coerce') for k, c in NUM.items()})
    feat['binder_frac'] = df[RATIO].map(ratio)
    feat = pd.concat([feat, pd.get_dummies(df[CAT].astype(str), prefix=CAT)], axis=1)
    y = pd.to_numeric(df[TARGET], errors='coerce').values
    X = SimpleImputer(strategy='median').fit_transform(feat.values.astype(float))
    keycols = ['ПКМ', 'ТКАНЬ: Название', 'ТЕХНОЛОГИЯ ФОРМИРОВАНИЯ: Название',
               'СМОЛА: Название', 'СВЯЗУЮЩЕЕ: Тип']
    gk = df[keycols].apply(lambda r: '|'.join('' if pd.isna(v) else str(v) for v in r), axis=1).values
    groups = pd.Series(gk); ug = list(dict.fromkeys(gk.tolist()))
    return df, X, y, groups, ug

def grouped_masks_from_series(groups, ug, seed):
    rng = np.random.RandomState(seed); u = ug.copy(); rng.shuffle(u); n = len(u)
    S = (set(u[:int(.6*n)]), set(u[int(.6*n):int(.8*n)]), set(u[int(.8*n):]))
    return tuple(groups.isin(g).values for g in S)


# ===================================================== CONFORMAL (order stat Eq.2)
def order_stat_radius(resid, alpha):
    """Split-conformal radius = sorted absolute residual at index ceil((n+1)(1-alpha))."""
    r = np.sort(resid); n = len(r)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return r[k - 1]

def quantile_radius(resid, alpha):
    """Split-conformal radius via linear-interpolation quantile (numpy default).
    Equivalent guarantee to order_stat_radius; used for Class B (SFRC) to match the
    manuscript, which computed the SFRC conformal radius this way."""
    n = len(resid)
    return float(np.quantile(resid, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)))

def split_conformal(resid_cal, pred_te, y_te, alpha, radius_fn=order_stat_radius):
    q = radius_fn(resid_cal, alpha)
    return float(np.mean(np.abs(y_te - pred_te) <= q)), float(2 * q), q

def normalized_conformal(core, Xtr, ytr, Xcal, ycal, Xte, yte, pred_cal, pred_te, alpha,
                         radius_fn=order_stat_radius):
    sc = StandardScaler().fit(Xtr)
    nn = NearestNeighbors(n_neighbors=min(10, len(Xcal))).fit(sc.transform(Xtr))
    tr_res = np.abs(ytr - core.predict(Xtr))
    sigma = lambda Xq: tr_res[nn.kneighbors(sc.transform(Xq))[1]].mean(1) + 1e-6
    s_cal, s_te = sigma(Xcal), sigma(Xte)
    qn = radius_fn(np.abs(ycal - pred_cal) / s_cal, alpha)
    return float(np.mean(np.abs(yte - pred_te) <= qn * s_te)), float(np.mean(2 * qn * s_te))

def mondrian_conformal(Xcal, ycal, Xte, yte, pred_cal, pred_te, alpha, temp_col=3,
                       radius_fn=order_stat_radius):
    """Temperature-regime conditional quantiles (bins <=200 / <=600 / >600 degC)."""
    resid = np.abs(ycal - pred_cal)
    q_glob = radius_fn(resid, alpha)
    tbin = lambda T: np.where(T <= 200, 0, np.where(T <= 600, 1, 2))
    bc, bt = tbin(Xcal[:, temp_col]), tbin(Xte[:, temp_col])
    cov_w, wid_w = [], []
    for k in range(3):
        rk = resid[bc == k]
        qk = q_glob if len(rk) < 3 else radius_fn(rk, alpha)
        msk = bt == k
        if msk.sum() > 0:
            cov_w.append(np.mean(np.abs(yte[msk] - pred_te[msk]) <= qk) * msk.sum())
            wid_w.append(2 * qk * msk.sum())
    return float(sum(cov_w) / len(yte)), float(sum(wid_w) / len(yte))


# ===================================================== NOVELTY / rho_NE criterion
def novelty_scores(Xtrain, Xquery, seed=RNG):
    """kNN-distance and isolation-forest novelty of each query point vs training set."""
    sc = StandardScaler().fit(Xtrain)
    nn = NearestNeighbors(n_neighbors=10).fit(sc.transform(Xtrain))
    nov_knn = nn.kneighbors(sc.transform(Xquery))[0].mean(1)
    iso = IsolationForest(random_state=seed).fit(sc.transform(Xtrain))
    nov_iso = -iso.score_samples(sc.transform(Xquery))
    return nov_knn, nov_iso

def rho_ne(nov, err):
    """Spearman rank correlation between novelty score and absolute error."""
    return stats.spearmanr(nov, err).correlation

def rho_ne_onesided_p(nov, err):
    """One-sided p-value for H0: rho_NE <= 0 (criterion of Sec. 3.5.3)."""
    rho, p2 = stats.spearmanr(nov, err)
    return rho, (p2 / 2 if rho > 0 else 1 - p2 / 2)

def holm_two(p_knn, p_iso, alpha=0.05):
    """Holm step-down for the two candidate detectors within one split.
    Returns (sig_knn, sig_iso) booleans."""
    ps = [p_knn, p_iso]; order = np.argsort(ps); adj = [0.0, 0.0]
    adj[order[0]] = min(1.0, 2 * ps[order[0]])
    adj[order[1]] = min(1.0, max(adj[order[0]], ps[order[1]]))
    return adj[0] < alpha, adj[1] < alpha


# =============================================================== BOOTSTRAP (Sec.4.1)
def bootstrap_mae_diff(y_true, pred_a, pred_b, n_boot=2000, seed=RNG):
    """Paired bootstrap CI for (MAE_a - MAE_b). Sign convention: negative => model a
    has the lower error. Used for the random-forest vs gradient-boosting comparison."""
    rng = np.random.RandomState(seed)
    ea, eb = np.abs(y_true - pred_a), np.abs(y_true - pred_b)
    n = len(y_true); diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[b] = ea[idx].mean() - eb[idx].mean()
    return float(ea.mean() - eb.mean()), (float(np.percentile(diffs, 2.5)),
                                          float(np.percentile(diffs, 97.5)))
