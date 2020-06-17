Main use cases of this repo:
1. Generate molecular feature sets for ML models `src/gen_mol_fea.py`
2. Aggregate molecular feature sets (generated on an HPC) `src/agg_fea_hpc`

## Generate molecular feature sets
The code takes SMILES stings, canonicalizes, and generates multiple feature sets stored in separate files.<br>
Mordred descriptors and fingerprints are stored in dataframes (e.g., parquet, csv).<br>
Images are stored in python dictionaries (pickle files).<br>
Each feature in a dataframe is prefixed with an appropriate string indicating the type.
- [x] Mordred descriptors (prefix: `dd_`)
- [x] ECFP2 (prefix: `ecfp2`)
- [x] ECFP4 (prefix: `ecfp4`)
- [x] ECFP6 (prefix: `ecfp6`)
- [x] Images

These datasets are then used to generate ML dataframes in `github.com/2019-ncovgroup/ML-docking-dataframe-generator` for each protein target. Alteratively, these feature sets can be used for inference with `github.com/brettin/ML-training-inferencing`.
<img src="README/dsc.df.png" alt="drawing" height="220"/>

## Aggregate molecular feature sets from HPC runs
Takes the output from `github.com/globus-labs/covid-analyses`, and aggregates files into a single dataframe. At this point, the code was tested only for Mordred descriptors. 

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
