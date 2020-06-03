"""
This script aggregates subsets of features from HPC larger.
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

FEA_MAIN_DIR = Path(filepath, '../data/raw/hpc-fea-splitted')
FEA_TYPE = 'descriptors'
FEA_DIR = FEA_MAIN_DIR/FEA_TYPE

# Global outdir
OUTDIR = Path(filepath, '../out', FEA_MAIN_DIR.name, FEA_TYPE).resolve()


def parse_args(args):
    parser = argparse.ArgumentParser(description='Aggregate molecular feature sets.')
    parser.add_argument('--fea_dir', default=FEA_DIR, type=str,
                        help=f'Full path to the fea dir file (default: {FEA_DIR}).')
    parser.add_argument('-od', '--outdir', default=OUTDIR, type=str,
                        help=f'Output dir (default: {OUTDIR}).')
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
    fea_dir = args['fea_dir']
    par_jobs = args['par_jobs']
    outdir = args['outdir']
    os.makedirs(outdir, exist_ok=True)
    
    # Get file names
    fea_files = sorted( fea_dir.glob('OZD-*.csv') )

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

    
    # n_step = 200 # num of files to aggregate in a single output file
    # for i, ii in enumerate(range(0, len(fea_files), n_step)):
    #     dfs = []
    #     for jj, f in enumerate(fea_files[ii:ii+n_step]):
    #         if ( ii+jj+1 )%50==0:
    #             print(f'Load {ii+jj+1} ... {f.name}')
    #         dd = pd.read_csv( Path(fea_files[ii+jj]), names=cols )
    #         dfs.append(dd)

    #     fea_df = pd.concat(dfs, axis=0)
    #     fea_df = fea_df.drop_duplicates(subset=['TITLE'])
    #     fea_df = fea_df.reset_index(drop=True)
    #     # print_fn(fea_df.shape)
    #     # tmp = print_fn(sizeof(fea_df, verbose=False));

    #     # Save
    #     fea_df.to_parquet(outdir/f'dd_fea{i}.parquet')
    #     del fea_df

    # import ipdb; ipdb.set_trace(context=5)
    dfs = []
    for i, f in enumerate(fea_files):
        if ( i+1 )%100==0:
            print(f'Load {i+1} ... {f.name}')
        dd = pd.read_csv( Path(fea_files[i]), names=cols )
        dfs.append(dd)

    fea_df = pd.concat(dfs, axis=0)
    fea_df = fea_df.drop_duplicates(subset=['TITLE'])
    fea_df = fea_df.reset_index(drop=True)
    print_fn('fea_df.shape', fea_df.shape)

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


