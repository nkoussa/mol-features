#!/bin/bash
smiles_path=data/raw/Baseline-Screen-Datasets/BL2-current/BL2.smi
outdir=out/images_and_others
par_jobs=64

python src/gen_mol_features.py --smiles_path $smiles_path --outdir $outdir --par_jobs $par_jobs --i2 30000
