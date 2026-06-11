# Datasets

Three openly published datasets are bundled here so the experiments run with no manual download.
Each is redistributed from its original public source **with attribution under its own licence**.
If you fork this repository, please keep these citations and verify the licences for your use.

---

## Class A — Concrete (`Concrete_Data.csv`)
* **Content:** 1030 concrete-mixture records (1005 after removing 25 exact duplicates), 8 mix/age inputs, compressive strength (MPa).
* **Source:** I.-C. Yeh, "Concrete Compressive Strength", UCI Machine Learning Repository.
* **URL / DOI:** https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength · DOI: 10.24432/C5PK67
* **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0).
* **Preprocessing (in `common.load_concrete`)**: columns renamed; exact duplicate rows dropped; composition key = the 8 inputs excluding curing age.

## Class B — Steel-fibre-reinforced concrete (`SFRC_Data_v1.xlsx`)
* **Content:** 307 SFRC specimens (158 fibre-and-mixture groups), 12 features after cleaning, compressive strength `Fc` (MPa) at several exposure temperatures.
* **Source:** F. Bagehrzadeh and T. Shafighfard, "Compressive strength of steel fiber reinforced concrete structures at elevated temperatures", Mendeley Data, Version 1, 2022.
* **DOI:** 10.17632/hjrfgys29n.1
* **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0).
* **Preprocessing (in `common.load_sfrc`)**: a free-text raw-material column duplicating the binary indicators and a length-to-diameter column (an exact function of fibre length and diameter) are dropped; group key excludes temperature and heating rate.

## Class C — Polymer textile composites (`Polymer_TPCM.xlsx`)
* **Content:** 420 fibre-reinforced polymer textile-laminate specimens from 32 formulations; many properties; target used here = warp tensile strength (MPa). Used only as an out-of-family **stress test** (Section 5.3).
* **Source:** I. Malashin, V. Tynchenko, A. Gantimurov, V. Nelyub, and A. Borodulin, "A multi-objective optimization of neural networks for predicting the physical properties of textile polymer composite materials", *Polymers*, Vol. 16, No. 12, Art. 1752, 2024 (ref [32]). Data: TPCM repository, https://github.com/catauggie/TPCM
* **Licence:** as released in the TPCM repository / the cited article. Please verify before redistribution.
* **Preprocessing (in `common.load_polymer`)**: numeric fabric/fibre/binder properties + parsed binder-to-reinforcement ratio + one-hot categorical descriptors; median imputation; formulation group key = composite/fabric/technology/resin/binder type.

---

### Note
The bundled files are provided for reproducibility convenience. The authoritative versions are the
original repositories cited above. No modification has been made to the raw measurement values;
all cleaning happens at load time in `src/common.py`.
