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

#### Class C column dictionary (Russian source → English meaning)

The TPCM source file keeps the original **Russian** column headers. Of its ~200 columns,
`common.load_polymer` uses only the ones below; it selects each by its Russian name and renames
it to the English code name used in the analysis. The target and every feature are therefore fully
documented in English here and in `src/common.py`.

| Role | Source column (Russian) | English meaning | Code name |
|---|---|---|---|
| **Target** | ОСНОВНЫЕ СВОЙСТВА ПКМ: Прочность на растяжение по основе, МПа | Composite warp tensile strength, MPa | `y` |
| Feature | ОСНОВНЫЕ СВОЙСТВА ТКАНИ: Прочность при растяжении по основе, МПа | Fabric warp tensile strength, MPa | `fabric_tensile` |
| Feature | ОСНОВНЫЕ СВОЙСТВА ТКАНИ: Модуль упругости при растяжении по основе, ГПа | Fabric warp tensile modulus, GPa | `fabric_mod` |
| Feature | ОСНОВНЫЕ СВОЙСТВА ТКАНИ: Поверхностная плотность, г/м2 | Fabric areal density, g/m² | `fabric_areal` |
| Feature | ОСНОВНЫЕ СВОЙСТВА ТКАНИ: Толщина, мм | Fabric thickness, mm | `fabric_thick` |
| Feature | ОСНОВНЫЕ СВОЙСТВА НИТИ (ОСНОВА): Прочность при растяжении волокна, МПа | Warp-yarn fibre tensile strength, MPa | `fibre_tensile` |
| Feature | ОСНОВНЫЕ СВОЙСТВА НИТИ (ОСНОВА): Модуль упругости волокна при растяжении, ГПа | Warp-yarn fibre tensile modulus, GPa | `fibre_mod` |
| Feature | ОСНОВНЫЕ СВОЙСТВА НИТИ (ОСНОВА): Плотность, текс | Warp-yarn linear density, tex | `fibre_tex` |
| Feature | ОСНОВНЫЕ СВОЙСТВА НИТИ (ОСНОВА): Диаметр филаментов, мкм | Warp-yarn filament diameter, µm | `fibre_fil_d` |
| Feature | ОСНОВНЫЕ СВОЙСТВА СВЯЗУЮЩЕГО: Прочность на растяжение, МПа | Binder tensile strength, MPa | `binder_tensile` |
| Feature | ОСНОВНЫЕ СВОЙСТВА СВЯЗУЮЩЕГО: Модуль упругости при растяжении, ГПа | Binder tensile modulus, GPa | `binder_mod` |
| Feature (derived) | ТЕХНОЛОГИЯ ФОРМИРОВАНИЯ: Соотношение связующего к армирующему наполнителю | Binder-to-reinforcement ratio → binder fraction | `binder_frac` |
| Categorical (one-hot) | ПКМ | Composite (PCM) designation | — |
| Categorical (one-hot) | ТКАНЬ: Вид плетения | Fabric weave type | — |
| Categorical (one-hot) | ТЕХНОЛОГИЯ ФОРМИРОВАНИЯ: Название | Forming-technology name | — |
| Categorical (one-hot) | СВЯЗУЮЩЕЕ: Тип | Binder type | — |
| Group key | ПКМ · ТКАНЬ: Название · ТЕХНОЛОГИЯ ФОРМИРОВАНИЯ: Название · СМОЛА: Название · СВЯЗУЮЩЕЕ: Тип | composite · fabric · technology · resin · binder type | `groups` |

*Glossary:* ПКМ = polymer composite material; ТКАНЬ = fabric; НИТЬ (ОСНОВА) = warp yarn; СВЯЗУЮЩЕЕ = binder; СМОЛА = resin; «по основе» = warp direction; Прочность при растяжении = tensile strength; Модуль упругости = elastic modulus.

---

### Note
The bundled files are provided for reproducibility convenience. The authoritative versions are the
original repositories cited above. No modification has been made to the raw measurement values;
all cleaning happens at load time in `src/common.py`.
