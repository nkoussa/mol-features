#!/bin/bash

# Generate molecular features for Pilot1.
par_jobs=64
fea_type="descriptors fps"

# Non-NSC drugs (CCLE, CTRP, gCSI, GDSC): ~800 smiles
gout=out.pilot1/non-nsc/
smiles_path=data/raw/non-nsc/drug_smiles
python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type

# NSC drugs (NCI60): ~52K smiles
gout=out.pilot1/nsc/
smiles_path=data/raw/nsc/NCI60_drugs_52k_smiles
python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type
