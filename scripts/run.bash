#!/bin/bash

# That's the general workflow to generate molecular features for a relatively
# small set of smiles strings. For the larger set of smiles we use HPC (Kyle
# usually generates those with this repo github.com/globus-labs/covid-analyses).

# smiles_path=data/raw/Baseline-Screen-Datasets/BL2-current/BL2.smi
smiles_path=data/raw/OZD-dock-2020-06-01/OZD.May29.unique.csv

outdir=out/images_and_dfs
par_jobs=1
# python src/gen_mol_fea.py --smiles_path $smiles_path --outdir $outdir --par_jobs $par_jobs --i2 20000
python src/gen_mol_fea.py --smiles_path $smiles_path --outdir $outdir --par_jobs $par_jobs --i2 100
