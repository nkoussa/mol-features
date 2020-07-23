#!/bin/bash

# That's the general workflow to generate molecular features for a relatively
# small set of smiles strings. For the larger set of smiles we use HPC (Kyle
# usually generates those with this repo github.com/globus-labs/covid-analyses).

# Generate molecular features for Covid.
par_jobs=64
fea_type="descriptors fps"

# smiles_path=data/raw/Baseline-Screen-Datasets/BL2-current/BL2.smi
gout=out.covid/images_and_dfs
smiles_path=data/OZD-dock-2020-06-01/OZD.May29.unique.csv

python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --id_name TITLE \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type
