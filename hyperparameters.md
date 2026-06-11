# Hyperparameters, seeds and protocol

All settings are defined in code in `src/common.py` (`model_zoo()` and the split functions).
This file is the human-readable summary.

## Random seeds
| Purpose | Seed(s) |
|---|---|
| Primary split (single-split headline numbers) | 42 |
| Ten evaluation splits (mean ± SD) | 0, 1, 2, …, 9 |
| Model `random_state` | 42 |
| Bootstrap resampling | 42 (2000 resamples) |
| Isolation-forest novelty | 42 |

## Models (Table 3)
All use scikit-learn unless noted. No dataset-specific tuning is applied; fixed hyperparameters are used for both classes.

| Model | Library | Hyperparameters |
|---|---|---|
| Mean (baseline) | sklearn `DummyRegressor` | `strategy="mean"` |
| Linear regression | sklearn `LinearRegression` | defaults, on `StandardScaler` features |
| Support-vector regression | sklearn `SVR` | `C=100, gamma="scale"`, on `StandardScaler` features |
| Random forest | sklearn `RandomForestRegressor` | `n_estimators=300, random_state=42` |
| **Gradient boosting (CORE)** | sklearn `HistGradientBoostingRegressor` | defaults, `random_state=42` |
| XGBoost | `xgboost.XGBRegressor` | `n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42` |
| LightGBM | `lightgbm.LGBMRegressor` | `n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42` |
| Multilayer perceptron | sklearn `MLPRegressor` | `hidden_layer_sizes=(100,50), max_iter=2000, early_stopping=True, random_state=42`, on `StandardScaler` features |

## Splitting protocol
* **Ratio:** 60 / 20 / 20 (train / calibration / test).
* **Grouping unit:**
  * Class A (concrete): the 8 mix proportions (everything except curing age) → 428 compositions.
  * Class B (SFRC): fibre + mixture indicators excluding temperature and heating rate → 158 groups.
  * Class C (polymer): the formulation key (composite, fabric, technology, resin, binder type) → 32 groups.
* **Mechanism:** group IDs are shuffled with `numpy.random.RandomState(seed)` and assigned to partitions; group order is stabilised with `dict.fromkeys` so the split is fully determined by the seed.
* Exact membership for all 11 splits of Classes A and B is in `splits/*.json`.

## Fitting convention
| Quantity | Core fit on |
|---|---|
| Class A accuracy (Table 3, left) | training partition only |
| Class B accuracy (Table 3, right) | train + calibration |
| Conformal calibration (all classes) | core fit on training, residuals from calibration |
| ρNE criterion (Sec. 3.5.3) | core fit on training; correlation on held-out data |
| Deployed near/far ablation | core fit on train + calibration |

## Conformal prediction
* Radius = order statistic of the absolute calibration residuals at index ⌈(n+1)(1−α)⌉ (clipped to n) for Class A (concrete) and Class C (polymer); for Class B (SFRC) the numpy linear-interpolation quantile at the same level is used (equivalent finite-sample guarantee), matching the manuscript.
* Levels: α ∈ {0.20, 0.10, 0.05} → 80 / 90 / 95% nominal coverage.
* Variants: **split** (constant width), **normalized** (scaled by 10-NN mean training residual), **Mondrian** (temperature-regime conditional, bins ≤200 / ≤600 / >600 °C, SFRC only).

## Novelty–error coupling criterion (Sec. 3.5.3)
* Novelty scores: 10-NN mean distance in standardized feature space, and isolation-forest anomaly score.
* ρNE = Spearman rank correlation between novelty score and absolute error on held-out data.
* Significance: one-sided Spearman test of H₀: ρNE ≤ 0, per split.
* Multiple testing: Holm correction across the two candidate detectors within each split.
* Decision rule: a detector is **informative** iff its per-split correlation is positive and Holm-significant in **≥ 6 of the 10** splits and the across-split mean exceeds its standard deviation.
* Novelty flag threshold: 90th percentile of the calibration novelty scores.
* Minimum sample size: a Fisher z power analysis indicates ≈ 150 held-out points to resolve ρNE ≈ 0.25 at 80% power; below this a near-zero estimate is read as "coupling not demonstrated", not "proven absent".

## Anti-confounding criterion (Sec. 3.6)
The reliability signal must not be a disguised proxy for the predicted magnitude: |corr(reliability signal, strength)| < 0.15.
