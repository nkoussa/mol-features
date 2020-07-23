#!/bin/bash

# Generate molecular features for Pilot1.
par_jobs=64
fea_type="descriptors fps"

# --------------------------------------------------
# Non-NSC drugs (CCLE, CTRP, gCSI, GDSC): 804 smiles
# --------------------------------------------------
dname=PreJuly2020/non-nsc/drug_smiles

gout=out.pilot1/$dname
smiles_path=data/$dname

python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type


# -------------------------------------------------------
# 1800 drugs (CCLE, CTRP, gCSI, GDSC, NCI60): 1799 smiles
# -------------------------------------------------------
dname=PreJuly2020/drugs_1800

gout=out.pilot1/$dname
smiles_path=data/$dname

python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type


# -------------------------------
# NSC drugs (NCI60): 52642 smiles
# -------------------------------
dname=PreJuly2020/nsc/NCI60_drugs_52k_smiles

gout=out.pilot1/$dname
smiles_path=data/$dname

python src/gen_mol_fea.py \
    --smiles_path $smiles_path \
    --gout $gout \
    --par_jobs $par_jobs \
    --ignore_3D \
    --fea_type $fea_type
