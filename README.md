# CALIBER

**A Reliability-Aware Information-Analytical System for Composite-Material Strength Prediction with Cross-Dataset Validation**

Reference implementation for the paper submitted to the *International Journal of Intelligent Engineering and Systems* (IJIES), Paper ID 20263531.

CALIBER pairs a gradient-boosting strength predictor with a **reliability layer** — split-conformal prediction intervals (split / normalized / Mondrian) and a domain-of-applicability detector — and introduces the **novelty–error coupling criterion (ρNE)**: a pre-deployment test that decides, from calibration data alone, whether a domain detector will be informative for a given dataset.

This repository contains everything needed to reproduce every table, figure and reported interval in the paper: the preprocessing pipeline, the exact group membership of all eleven splits, all model hyperparameters and random seeds, and the conformal, novelty and bootstrap code.

---

## Repository layout

```
CALIBER/
├── README.md               # this file
├── REPRODUCE.md            # step-by-step: which script makes which table/number
├── hyperparameters.md      # every model's hyperparameters + seeds
├── requirements.txt        # pinned dependencies
├── LICENSE                 # MIT (code); datasets keep their own licences
├── data/
│   ├── README.md           # dataset sources, DOIs, licences, attribution
│   ├── Concrete_Data.csv   # Class A  (UCI concrete)
│   ├── SFRC_Data_v1.xlsx   # Class B  (Mendeley SFRC)
│   └── Polymer_TPCM.xlsx   # Class C  (TPCM polymer, stress test)
├── splits/                 # machine-readable group membership of all 11 splits
│   ├── concrete_splits.json
│   └── sfrc_splits.json
├── results/                # generated figures/tables land here
└── src/
    ├── common.py           # loaders, splits, models, conformal, novelty, bootstrap
    ├── make_splits.py      # regenerates splits/*.json
    ├── table3_accuracy.py  # Table 3  — accuracy of 8 models, both classes
    ├── table4_coverage.py  # Table 4  — conformal coverage
    ├── table5_novelty.py   # Table 5  — novelty–error coupling criterion
    ├── random_vs_grouped.py# Sec. 5.2 — leakage demonstration
    ├── bestcore_rf.py      # Sec. 4.1 — random-forest-core robustness check
    ├── classC_polymer.py   # Sec. 5.3 — polymer stress test (Class C)
    └── figure4.py          # Figure 4 — cross-class summary
```

---

## Installation

```bash
git clone https://github.com/caliber-reliability/CALIBER.git
cd CALIBER
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Python 3.11–3.12 recommended.

## Quick start

All scripts are run from the `src/` directory and print the corresponding table to the console:

```bash
cd src
python table3_accuracy.py      # Table 3
python table4_coverage.py      # Table 4
python table5_novelty.py       # Table 5  (the ρNE criterion + near/far ablation)
python random_vs_grouped.py    # Section 5.2 leakage demonstration
python bestcore_rf.py          # Section 4.1 RF-core robustness
python classC_polymer.py       # Section 5.3 polymer stress test
python figure4.py              # writes results/figure4.png
python make_splits.py          # (re)writes splits/*.json
```

See **REPRODUCE.md** for the exact expected output of each script and how it maps to the manuscript.

---

## Reproducibility notes

* **Determinism.** The primary split uses seed 42; the ten evaluation splits use seeds 0–9. Every model is seeded. Re-running yields identical numbers on the pinned dependency versions.
* **Leakage-free protocol.** All splits are made at the **composition / formulation level**, never at the record level, so replicate specimens of one mixture cannot appear in both training and test sets. `splits/*.json` lists the exact row indices in each partition for all eleven splits.
* **Conformal convention.** The split-conformal radius is the order statistic ⌈(n+1)(1−α)⌉ of the absolute calibration residuals for the concrete and polymer classes, and the numpy linear-interpolation quantile at the same level for SFRC (equivalent finite-sample guarantee); this matches the manuscript. See `hyperparameters.md`.
* **Fitting convention.** Class A (concrete) accuracy uses the training partition only (the conformal-core convention); Class B (SFRC) uses train + calibration. Both are stated in each script and in `hyperparameters.md`.

## Datasets

Three openly published datasets are included under `data/` for one-command reproducibility. **See `data/README.md` for full sources, DOIs and licences** before redistributing.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{caliber2025,
  title   = {CALIBER: A Reliability-Aware Information-Analytical System for
             Composite-Material Strength Prediction with Cross-Dataset Validation},
  author  = {Sarsenbay, Magzhan and Zhuzbayev, Serik and Shayea, Ibraheem and Bakytzhan, Askar},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2025},
  note    = {Paper ID 20263531}
}
```

## License

Code: MIT (see `LICENSE`). Datasets: redistributed under their original licences (see `data/README.md`).
