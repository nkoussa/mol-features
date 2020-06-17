#!/bin/bash

# That's the general workflow to generate molecular features for relatively
# small set of smiles strings. For the larger set of smiles we use HPC (Kyle
# usually generates those with this repo github.com/globus-labs/covid-analyses).

# smiles_path=data/raw/Baseline-Screen-Datasets/BL2-current/BL2.smi
# outdir=out/images_and_others
# par_jobs=64
# python src/gen_mol_fea.py --smiles_path $smiles_path --outdir $outdir --par_jobs $par_jobs --i2 30000
