#!/bin/bash

# Example:
# bash scripts/run_from_hpc.bash 

# That's the general workflow to generate molecular features for a relatively
# small set of smiles strings. For the larger set of smiles we use HPC (Kyle
# usually generates those with this repo github.com/globus-labs/covid-analyses).

# DRG_SET=OZD
DRG_SET=ORD

# FEA_TYPE="descriptors fps images"
# FEA_TYPE="descriptors fps"
FEA_TYPE="descriptors"
# FEA_TYPE="fps"

OUTDIR=out/fea_hpc/$DRG_SET
JOBS=32

echo "Generate dataframes."
python src/agg_fea_hpc.py \
    --drg_set $DRG_SET \
    --outdir $OUTDIR \
    --par_jobs $JOBS \
    --fea_type $FEA_TYPE
