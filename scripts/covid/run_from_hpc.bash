#!/bin/bash

# That's the general workflow to generate molecular features for a relatively
# small set of smiles strings. For the larger set of smiles we use HPC (Kyle
# usually generates those with this repo github.com/globus-labs/covid-analyses).

# smiles_path=data/raw/Baseline-Screen-Datasets/BL2-current/BL2.smi
drg_set=OZD

outdir=out.pivot/images_and_dfs_hpc
par_jobs=32

python src/agg_fea_hpc.py \
    --drg_set $drg_set \
    --outdir $outdir \
    --par_jobs $par_jobs \
    --fea_type descriptors fps images
