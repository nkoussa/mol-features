"""
This script computes Mordred descriptors and saves the dataframe.
(each feature type is prefixed with an appropriate identifier):
    Mordred descriptors (prefix: .mod)
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

import numpy as np
import pandas as pd

filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
from utils.utils import load_data, get_print_func, drop_dup_rows, dropna
from utils.smiles import canon_smiles, smiles_to_mordred, smiles_to_fps
# sys.path.append( os.path.abspath(filepath/'../utils') )
# from classlogger import Logger
# from utils import load_data, get_print_func, drop_dup_rows, dropna
# from smiles import canon_smiles, smiles_to_mordred

# DATADIR
# DATADIR = Path(filepath, '../data/raw/UC-molecules')
# DATADIR = Path(filepath, '../data/raw/Baseline-Screen-Datasets/BL1(ena+db)')
DATADIR = Path(filepath, '../data/raw/Baseline-Screen-Datasets/BL2-current')

# OUTDIR
t = datetime.now()
t = [t.year, '-', t.month, '-', t.day]
date = ''.join( [str(i) for i in t] )
# OUTDIR = Path( filepath, '../data/processed/UC-molecules/', date )
# OUTDIR = Path( filepath, '../data/processed/BL1/', date )
##OUTDIR = Path( filepath, '../data/processed/BL2/', date )
OUTDIR = Path( filepath, '../out/BL2/', date )

# SMILES
# in_fname = 'UC.smi'
# in_fname = 'ena+db.smi'
in_fname = 'BL2.smi'
SMILES_PATH = str( DATADIR/in_fname )


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate molecular feature sets.')
    parser.add_argument('--smiles_path', default=SMILES_PATH, type=str,
                        help=f'Full path to the smiles file (default: {SMILES_PATH}).')
    parser.add_argument('-od', '--outdir', default=None, type=str,
                        help=f'Output dir (default: None).')
    parser.add_argument('--par_jobs', default=1, type=int, 
                        help=f'Number of joblib parallel jobs (default: 1).')
    parser.add_argument('--i1', default=0, type=int, 
                        help=f'Start index of a smiles sample (default: 0).')
    parser.add_argument('--i2', default=None, type=int, 
                        help=f'End index of smiles sample (default: None).')
    args, other_args = parser.parse_known_args(args)
    return args


# Prefix features
def add_fea_prfx(df, prfx:str, id0:int):
    return df.rename(columns={s: prfx+str(s) for s in df.columns[id0:]})


def run(args):
    t0 = time()
    par_jobs = args['par_jobs']
    
    print('\nLoad smiles ...')
    smiles_path = Path(args['smiles_path'])
    smi = pd.read_csv( smiles_path, sep='\t', names=['smiles', 'name'] )
    smi['smiles'] = smi['smiles'].map(lambda x: x.strip())
    n_smiles = smi.shape[0]
    fea_id0 = smi.shape[1] # this used as index where features begin

    # Create outdir
    i1, i2 = args['i1'], args['i2']
    ids_dir = 'smi.ids.{}-{}'.format(i1, i2)
    if i2 is None:
        i2 = n_smiles
    if args['outdir'] is not None:
        outdir = Path( args['outdir'] ).resolve()
    else:
        outdir = OUTDIR
    outdir = OUTDIR/ids_dir
    os.makedirs( outdir, exist_ok=True )

    # Logger
    lg = Logger( outdir/'gen.mord.df.log' )
    print_fn = get_print_func( lg.logger )
    print_fn( f'File path: {filepath}' )
    print_fn( f'\n{pformat(args)}' )

    print_fn('\nInput data dir  {}'.format( DATADIR ))
    print_fn('Output data dir {}'.format( outdir ))

    # Duplicates
    # dup = smi[ smi.duplicated(subset=['smiles'], keep=False) ].reset_index(drop=True)
    # print( dup['smiles'].value_counts() )

    # Remove duplicates
    print_fn('\nDrop duplicates ...')
    smi = drop_dup_rows(smi, print_fn)
    
    # Exract subset smiles
    smi = smi.iloc[i1:i2+1, :].reset_index(drop=True)

    print_fn('\nCanonicalize smiles ...')
    can_smi_vec = canon_smiles( smi['smiles'], par_jobs=par_jobs )
    can_smi_vec = pd.Series(can_smi_vec)

    # Drop bad SMILES (that were not canonicalized)
    nan_ids = can_smi_vec.isna()
    bad_smi = smi[ nan_ids ]
    smi['smiles'] = can_smi_vec
    smi = smi[ ~nan_ids ].reset_index(drop=True)
    if len(bad_smi)>0:
        bad_smi.to_csv(outdir/'smi_canon_err.csv', index=False)

    # ==========================
    # Generate fingerprints
    # --------------------------
    def gen_fps_and_save(smi, radius=1, par_jobs=par_jobs):
        file_format='parquet'
        ecfp = smiles_to_fps(smi, smi_name='smiles', radius=radius, par_jobs=par_jobs)
        ecfp = add_fea_prfx(ecfp, prfx=f'ecfp{2*radius}.', id0=fea_id0)
        ecfp.to_parquet( outdir/f'ecfp{2*radius}.ids.{i1}-{i2}.{file_format}' )
        del ecfp

    gen_fps_and_save(smi, radius=1, par_jobs=par_jobs)
    gen_fps_and_save(smi, radius=2, par_jobs=par_jobs)
    gen_fps_and_save(smi, radius=3, par_jobs=par_jobs)
    # ==========================

    # ==========================
    # Generate descriptors
    # --------------------------
    dsc = smiles_to_mordred(smi, smi_name='smiles', par_jobs=par_jobs)
    dsc = add_fea_prfx(dsc, prfx='mod.', id0=fea_id0)

    # Filter NaNs (step 1)
    # Drop rows where all values are NaNs
    print_fn('\nDrop rows where all values are NaN ...')
    print_fn('Shape: {}'.format( dsc.shape ))
    idx = ( dsc.isna().sum(axis=1) == dsc.shape[1] ).values
    dsc = dsc.iloc[~idx, :].reset_index(drop=True)
    # Drop cols where all values are NaNs
    # idx = ( dsc.isna().sum(axis=0) == dsc.shape[0] ).values
    # dsc = dsc.iloc[:, ~idx].reset_index(drop=True)
    print_fn('Shape: {}'.format( dsc.shape ))

    # Filter NaNs (step 2)
    # Drop rows and cols based on a thershold of NaN values
    # print(dsc.isna().sum(axis=1).sort_values(ascending=False))
    # p=dsc.isna().sum(axis=1).sort_values(ascending=False).hist(bins=100);
    th = 0.2
    print_fn('\nDrop rows with at least {} NaNs (at least {} out of {}).'.format(
        th, int(th * dsc.shape[1]), dsc.shape[1]))
    print_fn('Shape: {}'.format( dsc.shape ))
    dsc = dropna(dsc, axis=1, th=th)
    print_fn('Shape: {}'.format( dsc.shape ))

    # Cast features (descriptors)
    print_fn('\nCast descriptors to float ...')
    dsc = dsc.astype({c: np.float32 for c in dsc.columns[fea_id0:]})

    # Impute missing values
    print_fn('\nImpute NaNs ...')
    print_fn('Total NaNs: {}'.format( dsc.isna().values.flatten().sum() ))
    dsc = dsc.fillna(0.0)
    print_fn('Total NaNs: {}'.format( dsc.isna().values.flatten().sum() ))

    # Save
    print_fn('\nSave ...')
    dsc = dsc.reset_index(drop=True)
    file_format='parquet'
    dsc.to_parquet( outdir/'dsc.ids.{}-{}.{}'.format(i1, i2, file_format) )
    # dsc.to_csv( outdir/'dsc.ids.{}-{}.{}'.format(i1, i2, file_format), index=False )
    # ==========================

    # ==========================
    # Generate images
    # --------------------------
    # TODO
    pass
    # ==========================

    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


