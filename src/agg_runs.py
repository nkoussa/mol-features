"""
Aggregate feature subsets from multiple runs from feature generation
code in gen_mol_fea.py. 

Example:
    python src/agg_runs.py --res_dir out/BL2/2020-4-24 --fea dd --prfx BL2
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
import argparse
from glob import glob

import numpy as np
import pandas as pd

filepath = Path(__file__).resolve().parent

# Utils
from utils.utils import load_data


def parse_args(args):
    parser = argparse.ArgumentParser(description='Aggregate feature subsets from multiple runs.')
    parser.add_argument('--res_dir', required=True, default=None, type=str,
                        help='Global dir where multiple runs were dumped (default: None).')
    parser.add_argument('--fea', required=True, default='dd', type=str,
                        help='Feature type (default: dd).')
    parser.add_argument('--prfx', default=None, type=str,
                        help='Name prefix for the output file (default: None).')
    args = parser.parse_args(args)
    return args


def run(args):
    res_dir = Path( args['res_dir'] ).resolve()
    fea_name = args['fea']
    prfx = args['prfx']
    
    run_dirs = glob( str(res_dir/'*ids*') )
    fea_dfs = []
    for i, r in enumerate(run_dirs):
        dpath = sorted(Path(r).glob(f'*{fea_name}*'))[0]
        if not dpath.exists():
            continue
        df = load_data( dpath )

        if 'smiles' in df.columns:
            df = df.rename(columns={'smiles': 'SMILES'})
        if 'name' in df.columns:
            df = df.rename(columns={'name': 'TITLE'})

        fea_dfs.append( df )
    fea_df = pd.concat( fea_dfs, axis=0 )
    del fea_dfs

    print(fea_df.shape)
    fea_df = fea_df.drop_duplicates(subset=['TITLE']).reset_index(drop=True)
    print(fea_df.shape)
    
    fea_df.to_parquet(f'{res_dir}/{prfx}.{fea_name}.parquet')
    print('Done.')

    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


