Main use cases of this repo:
1. Generate molecular features for ML models `src/gen_mol_fea.py`
2. Aggregate molecular features generated on HPC (Theta, Frontera) `src/agg_fea_hpc.py`

## Calculate molecular features
`src/gen_mol_fea.py` takes SMILES, canonicalizes, and generates multiple feature sets stored in separate files.<br>
Mordred descriptors and fingerprints are stored in dataframes (e.g., parquet, csv).<br>
Images are stored in python dictionaries (pickle files).<br>
Each feature (column) name in a dataframe is prefixed with an appropriate string indicating the type.
- [x] Mordred descriptors (prefix: `dd_`)
- [x] ECFP2 (prefix: `ecfp2_`)
- [x] ECFP4 (prefix: `ecfp4_`)
- [x] ECFP6 (prefix: `ecfp6_`)
- [x] Images

First, clone the repo.
```shell
$ git clone https://github.com/adpartin/mol-features/
```

### If you are working on Covid
<!--- These datasets are then used to generate ML dataframes for each target protein `github.com/2019-ncovgroup/ML-docking-dataframe-generator`.
Alteratively, these feature sets can be used for inference with `github.com/brettin/ML-training-inferencing`.
<img src="README/dsc.df.png" alt="drawing" height="220"/>
 --->

Inside project dir, create folder that will contain raw SMILES.
```shell
$ cd mol-features
$ mkdir -p data/raw/
```
Get the data from from Box or Petrel (e.g., from Box copy 2019-nCoV/drug-screening/Baseline-Screen-Datasets) to `data/raw`.<br>
Then, launch a python script or use a bash script (you may need to change the bash script to specify your parameters).
```shell
$ python src/gen_mol_fea.py --smiles_path data/OZD-dock-2020-06-01/OZD.May29.unique.csv --id_name TITLE --fea_type descriptors fps --par_jobs 16 --ignore_3D
```
```shell
$ bash scripts/covid/gen_fea_OZD.bash
```

### If you are working on Pilot1 (cancer)
Clone the repo.
```shell
$ git clone https://github.com/adpartin/mol-features/
```
Inside project dir, create folder that will contain raw SMILES.
```shell
$ cd mol-features
$ mkdir -p data/raw/July2020
```
Get the data from `/vol/ml/mshukla/data_frames/Jul2020/drug_info` to `data/raw/July2020`.<br>
Then, launch a python script or use a bash script (you may need to change the bash script to specify your parameters).
```shell
$ python src/gen_mol_fea.py --smiles_path data/July2020/drug_info --id_name ID --fea_type descriptors fps --par_jobs 16 --ignore_3D
```
```shell
$ bash scripts/pilot1/gen_fea_July2020.bash
```

## Aggregate molecular feature from HPC runs
Instead of calculating features with `src/gen_mol_fea.py`, we can take features computed on HPC and aggregate those into dataframes. `src/agg_fea_hpc.py` takes the output from `github.com/globus-labs/covid-analyses` and aggregates files into a single dataframe. At this point, the code was tested only to gerenare dataframe with Mordred descriptors. 


<!-- The original Mordred descriptors are stored in Box `2019-nCoV/drug-screening/ena+db.desc.gz`. This file requires some pre-processing (duplicates, bad rows, NaNs, casting). This needs to be done only once. The clean version of the features (Enamine + DrugBank; 300K SMILES) can be found in Box `2019-nCoV/drug-screening/features/ena+db/ena+db.features.parquet`. If you need to generate the descriptors from the original file, follow the steps below. -->

<!-- - Clean and canonicalize smiles `ena+db.smi`. Use `src/ena+db/clean_smiles.py` (updated file is in `2019-nCoV/drug_screening/features/ena+db/ena+db.smi.can.csv`) 
- Clean descriptors `ena+db.desc`. Use `src/ena+db/clean_desc.py` (updated file is in `2019-nCoV/drug_screening/features/ena+db/ena+db.desc.parquet`)
<!-- - Merge smiles and descriptors using `src/ena+db/merge_smi_desc.py` (updated file is in `2019-nCoV/drug_screening/descriptors/ena+db/ena+db.smi.desc.parquet`)
 -->
<!-- - Merge smiles with descriptors and generate fingerprints from smiles (ECFP2, ECFP4, ECFP6). Use `src/ena+db/gen_fea_df.py` (updated file is in `2019-nCoV/drug_screening/features/ena+db/ena+db.features.parquet`)
