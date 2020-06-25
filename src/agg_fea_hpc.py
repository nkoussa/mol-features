"""
This script aggregates subsets of features from a large HPC run.
For example, it aggregates descriptors that were generated on Theta using this
code:
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

# Features
FEA_DIR = Path(filepath, '../data/raw/fea-subsets-hpc')
DRG_SET = 'OZD'

def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Aggregate molecular feature sets.')

    parser.add_argument('--drg_set',
                        type=str,
                        default=DRG_SET,
                        choices=['OZD', 'ORD'],
                        help=f'Drug set (default: {DRG_SET}).')
    parser.add_argument('--fea_type',
                        type=str, 
                        default=FEA_TYPE,
                        nargs='+',
                        choices=['descriptors', 'images', 'fps'],
                        help=f'Feature type (default: descriptors).')
    parser.add_argument('-od', '--outdir',
                        type=str,
                        default=None,
                        help=f'Output dir (default: None).')
    parser.add_argument('--par_jobs',
                        type=int, 
                        default=1,
                        help=f'Number of joblib parallel jobs (default: 1).')

    # args, other_args = parser.parse_known_args(args)
    args = parser.parse_args( args )
    return args


def sizeof(data, verbose=True):
    sz = sys.getsizeof(data)/1e9
    if verbose: print(f'Size in GB: {sz:.3f}')
    return sz


def run(args):
    # import ipdb; ipdb.set_trace()
    t0 = time()
    drg_set = args['drg_set']
    fea_type = args['fea_type']

    par_jobs = int( args['par_jobs'] )
    assert par_jobs > 0, f"The arg 'par_jobs' must be int >1 (got {par_jobs})"

    outdir = Path(args['outdir'])
    if outdir is None:
        outdir = Path(filepath, '../out', FEA_DIR.name, drg_set).resolve()
    os.makedirs(outdir, exist_ok=True)
    
    # Logger
    lg = Logger( outdir/'gen.fea.dfs.log' )
    print_fn = get_print_func( lg.logger )
    print_fn( f'File path: {filepath}' )
    print_fn( f'\n{pformat(args)}' )


    # ========================================================
    # Generate descriptors
    # --------------------------
    if 'descriptors' in fea_type:
        fea_outpath = outdir/'descriptors'
        files_path = Path(FEA_DIR, drg_set, 'descriptors').resolve()
        fea_files = sorted( files_path.glob(f'{drg_set}-*.csv') )

        if len(fea_files) > 0:
            os.makedirs(fea_outpath, exist_ok=True)

            fea_prfx = 'dd'
            fea_sep = '_'
            fea_names = pd.read_csv(FEA_DIR/'dd_fea_names.csv').columns.tolist()
            fea_names = [c.strip() for c in fea_names] # clean names
            fea_names = [fea_prfx+fea_sep+str(c) for c in fea_names] # prefix fea names
            cols = ['CAT', 'TITLE', 'SMILES'] + fea_names

            dfs = []
            for i, f in enumerate(fea_files[:N]):
                if (i+1) % 100 == 0:
                    print(f'Load {i+1} ... {f.name}')
                df = pd.read_csv( Path(fea_files[i]), names=cols )
                dfs.append(df)

            ID = 'TITLE'
            data = pd.concat(dfs, axis=0)
            data = data.drop_duplicates( subset=[ID] )
            data = data[ data[ID].notna() ].reset_index(drop=True)
            data[ fea_names ] = data[ fea_names ].fillna(0)
            data = data.reset_index( drop=True )
            print_fn('data.shape {}'.format(data.shape))

            # Save
            data.to_parquet(fea_outpath/f'{fea_prfx}.parquet')
            del data


    # ========================================================
    # Generate fingerprints
    # --------------------------
    if 'fps' in fea_type:
        fea_outpath = outdir/'fps'
        files_path = Path(FEA_DIR, drg_set, 'fps').resolve()
        fea_files = sorted( files_path.glob(f'{drg_set}-*.csv') )

        if len(fea_files) > 0:
            os.makedirs(fea_outpath, exist_ok=True)

            fea_prfx = 'ecfp2'
            fea_sep = '_'
            cols = ['CAT', 'TITLE', 'SMILES', 'fps']

            dfs = []
            for i, f in enumerate(fea_files[:N]):
                if (i+1) % 100 == 0:
                    print(f'Load {i+1} ... {f.name}')
                df = pd.read_csv( Path(fea_files[i]), names=cols )
                dfs.append(df)

            ID = 'TITLE'
            data = pd.concat(dfs, axis=0)
            data = data.drop_duplicates( subset=[ID] )
            data = data[ data[ID].notna() ].reset_index(drop=True)
            data = data.reset_index(drop=True)
            print_fn('data.shape {}'.format(data.shape))

            # Convert fps strings (base64) to integers
            import base64
            from rdkit.Chem import DataStructs
            aa = []
            for ii in data['fps'].values:
                ii = DataStructs.ExplicitBitVect( base64.b64decode(ii) )
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(ii, arr)
                aa.append( arr )

            fea_names = [fea_prfx+fea_sep+str(i+1) for i in range(len(aa[0]))] # prefix fea names
            cols = ['CAT', 'TITLE', 'SMILES'] + fea_names
            fea  = pd.DataFrame( np.vstack(aa), columns=fea_names )
            # fea[dd_fea_names] = fea[dd_fea_names].fillna(0)
            meta = data.drop(columns='fps')
            data = pd.concat([meta, fea], axis=1)

            # Save
            data.to_parquet(fea_outpath/f'{fea_prfx}.parquet')
            del data


    # ========================================================
    # Generate images
    # --------------------------
    # if 'images' in fea_type:
    #     fea_outpath = outdir/'images'
    #     files_path = Path(FEA_DIR, drg_set, 'images').resolve()
    #     fea_files = sorted( files_path.glob(f'{drg_set}-*.pkl') )

    #     if len(fea_files) > 0:
    #         os.makedirs(fea_outpath, exist_ok=True)

    #         dfs = []
    #         for i, f in enumerate(fea_files[:N]):
    #             if (i+1) % 100 == 0:
    #                 print(f'Load {i+1} ... {f.name}')
    #             imgs = pickle.load(open(fea_files[i], 'rb'))

    #             # That's from get_image_dct(mol)
    #             # image = (255 * transforms.ToTensor()(Invert()(generateFeatures.smiles_to_image(mol))).numpy()).astype(np.uint8)
    #             image = Invert()(image)
    #             image = transforms.ToTensor()(image)
    #             image = image.numpy()
    #             image = 255 * image
    #             image = image.astype(np.uint8)

    #             # To dict
    #             def img_data_to_dict( aa ):
    #                 dct = {}
    #                 dct['drg_set'] = aa[0]
    #                 dct['TITLE'] = aa[1]
    #                 dct['SMILES'] = aa[2]
    #                 dct['img'] = aa[3]


    # ========================================================
    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


