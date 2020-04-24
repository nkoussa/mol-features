Generate molecular feature sets for ML models.

## Molecular Features
The code takes SMILES stings, canonicalizes, and generates multiple feature sets (a dataframe for each feature type). Each feature in a dataframe is prefixed with an appropriate string indicating the feature type.
- [x] Mordred descriptors
- [x] ECFP2
- [x] ECFP4
- [x] ECFP6
- [ ] Images

<img src="images/smi-desc-df.png" alt="drawing" height="220"/>

## Getting started
Clone the repo.
```shell
$ git clone https://github.com/adpartin/mol-features/
```

Inside project dir, create folder that will contain raw SMILES.
```shell
$ cd mol-features
$ mkdir -p data/raw
```

Copy SMILES from Box or Petrel to `./data/raw/` (e.g., from Box copy 2019-nCoV/drug-screening/Baseline-Screen-Datasets).
Run the main script to canoncalize the SMILES and compute the feature sets. You need to specify the full path to the smiles file (argument `--smiles_path`). The file must contain a column `smiles` which is used to compute the features. Unless you specify the output dir (argument `--outdir`), the computed features are dumpet into `./out`.
```shell
$ python ./src/gen_mol_features.py --smiles_path data/raw/Baseline-Screen-Datasets/BL2-current/BL2.smi --par_jobs 8
```

The `par_jobs` argument uses the `joblib` Python package to parallelize the process https://joblib.readthedocs.io.



<!-- The original Mordred descriptors are stored in Box `2019-nCoV/drug-screening/ena+db.desc.gz`. This file requires some pre-processing (duplicates, bad rows, NaNs, casting). This needs to be done only once. The clean version of the features (Enamine + DrugBank; 300K SMILES) can be found in Box `2019-nCoV/drug-screening/features/ena+db/ena+db.features.parquet`. If you need to generate the descriptors from the original file, follow the steps below. -->

<!-- - Clean and canonicalize smiles `ena+db.smi`. Use `src/ena+db/clean_smiles.py` (updated file is in `2019-nCoV/drug_screening/features/ena+db/ena+db.smi.can.csv`) 
- Clean descriptors `ena+db.desc`. Use `src/ena+db/clean_desc.py` (updated file is in `2019-nCoV/drug_screening/features/ena+db/ena+db.desc.parquet`)
<!-- - Merge smiles and descriptors using `src/ena+db/merge_smi_desc.py` (updated file is in `2019-nCoV/drug_screening/descriptors/ena+db/ena+db.smi.desc.parquet`)
 -->
<!-- - Merge smiles with descriptors and generate fingerprints from smiles (ECFP2, ECFP4, ECFP6). Use `src/ena+db/gen_fea_df.py` (updated file is in `2019-nCoV/drug_screening/features/ena+db/ena+db.features.parquet`)
