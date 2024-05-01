#!/bin/bash

# That's the general workflow to generate molecular features for a relatively
# small set of smiles strings. For the larger set of smiles we can use Parsl.

par_jobs=16
fea_type="mordred"

gout=out.drugbank/all
smiles_path=data/drugbank/drugbank-all.smi

# gout=out.drugbank/all-filtered
# smiles_path=data/drugbank/drugbank-all-filtered.smi

python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --id_name ID \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type
