#!/bin/bash

# Generate molecular features for Pilot1.
par_jobs=64
fea_type="descriptors fps"
# fea_type="descriptors"

# -------------------------------------------------------
# 1800 drugs (CCLE, CTRP, gCSI, GDSC, NCI60): 2142 smiles
dname=July2020/drug_info
gout=out.pilot1/$dname
smiles_path=data/$dname

python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --id_name ID \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type

# -------------------------------------------------------
# NSC drugs (NCI60): 52642 smiles
dname=July2020/NCI60_drugs_52k_smiles
gout=out.pilot1/$dname
smiles_path=data/$dname

python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --id_name ID \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type
