#!/bin/bash

# Generate descriptors for ORD (computed on 06/15/2020)
python src/agg_fea_hps.py --drg_set ORD --fea_type descriptors --par_jobs 1
