"""
This script aggregates subsets of features from a large HPC run.
For example, it takes descriptors generated on Theta using this repo:
github.com/globus-labs/covid-analyses/tree/master/features/descriptors
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
from datetime import datetime
import argparse
from pprint import pformat
import pickle

import numpy as np
import pandas as pd

filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
from utils.utils import load_data, get_print_func

FEA_MAIN_DIR = Path(filepath, '../data/raw/fea-subsets-hpc')
DRUG_SET = 'OZD'
FEA_TYPE = 'descriptors'
FEA_DIR = FEA_MAIN_DIR/DRUG_SET/FEA_TYPE

def parse_args(args):
    parser = argparse.ArgumentParser(description='Aggregate molecular feature sets.')
    parser.add_argument('--drg_set', default=DRUG_SET, type=str,
                        choices=['OZD', 'ORD'],
                        help=f'Drug set (default: {DRUG_SET}).')
    parser.add_argument('--fea_type', default=FEA_TYPE, type=str,
                        choices=['descriptors'],
                        help=f'Feature type (default: {FEA_TYPE}).')
    parser.add_argument('-od', '--outdir', default=None, type=str,
                        help=f'Output dir (default: None).')
    parser.add_argument('--par_jobs', default=1, type=int, 
                        help=f'Number of joblib parallel jobs (default: 1).')
    args, other_args = parser.parse_known_args(args)
    return args


def sizeof(data, verbose=True):
    sz = sys.getsizeof(data)/1e9
    if verbose: print(f'Size in GB: {sz:.3f}')
    return sz


def run(args):
    t0 = time()
    drg_set = args['drg_set']
    fea_type = args['fea_type']
    par_jobs = args['par_jobs']

    outdir = args['outdir']
    if outdir is None:
        outdir = Path(filepath, '../out', FEA_MAIN_DIR.name, drg_set, fea_type).resolve()
    os.makedirs(outdir, exist_ok=True)
    
    # Get file names
    files_path = Path(FEA_MAIN_DIR, drg_set, fea_type).resolve()
    fea_files = sorted( files_path.glob(f'{drg_set}-*.csv') )

    # Logger
    lg = Logger( outdir/'gen.fea.dfs.log' )
    print_fn = get_print_func( lg.logger )
    print_fn( f'File path: {filepath}' )
    print_fn( f'\n{pformat(args)}' )

    dd_prfx = 'dd'
    dd_sep = '_'
    dd_fea_names = pd.read_csv(FEA_MAIN_DIR/'dd_fea_names.csv').columns.tolist()
    dd_fea_names = [c.strip() for c in dd_fea_names] # clean col names
    dd_fea_names = [dd_prfx+dd_sep+str(c) for c in dd_fea_names] # prefix fea cols
    cols = ['CAT', 'TITLE', 'SMILES'] + dd_fea_names

    dfs = []
    for i, f in enumerate(fea_files):
        if (i+1)%100 == 0:
            print(f'Load {i+1} ... {f.name}')
        dd = pd.read_csv( Path(fea_files[i]), names=cols )
        dfs.append(dd)

    fea_df = pd.concat(dfs, axis=0)
    fea_df = fea_df.drop_duplicates(subset=['TITLE'])
    fea_df[dd_fea_names] = fea_df[dd_fea_names].fillna(0)
    fea_df = fea_df.reset_index(drop=True)
    print_fn('fea_df.shape {}'.format(fea_df.shape))

    # Save
    fea_df.to_parquet(outdir/f'dd_fea.parquet')
    del fea_df

    # --------------------------------------------------------
    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


