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
import argparse
from pprint import pformat
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
from utils.utils import load_data, get_print_func
from ml.data import extract_subset_fea, extract_subset_fea_col_names

# Drug set
DRG_SET = 'OZD'

# Features
FEA_DIR = Path(filepath, '../data/fea-subsets-hpc')


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
                        default=['descriptors'],
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

    args = parser.parse_args( args )
    return args


def sizeof(data, verbose=True):
    sz = sys.getsizeof(data)/1e9
    if verbose: print(f'Size in GB: {sz:.2f}')
    return sz


def load_and_get_samples(f, cols, col_name, drug_names=None):
    """ Load a subset of features and retain samples of interest.
    Args:
        f : file name
        cols : cols to read in
        col_name : ID col name for drug names (e.g. TITLE)
        drug_names : list of drug to extract
    """
    df = pd.read_csv(f, names=cols)
    if drug_names is not None:
        df = df[ df[col_name].isin(drug_names) ]
    return df


def fps_to_nparr(x):
    """ Convert fps strings (base64) to integers. """
    import base64
    from rdkit.Chem import DataStructs
    x = DataStructs.ExplicitBitVect(base64.b64decode(x))
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(x, arr)
    return arr


def load_mordred_descriptors(drg_set, fea_name, col_name, drug_names=None,
                             fea_sep='_', n_jobs=64, N=None):
    """ Load Mordred descriptors files. The files contains subsets of
    descriptors generated on an HPC.
    """
    files_path = Path(FEA_DIR, drg_set, fea_name).resolve()
    fea_files = sorted(files_path.glob(f'{drg_set}-*.csv'))

    if len(fea_files) > 0:
        fea_prfx = 'dd'
        fea_names = pd.read_csv(FEA_DIR/'dd_fea_names.csv').columns.tolist()
        fea_names = [c.strip() for c in fea_names]  # clean names
        fea_names = [fea_prfx + fea_sep + str(c) for c in fea_names]  # prefix fea names
        cols = ['CAT', 'TITLE', 'SMILES'] + fea_names
        # cols = ['CAT', 'TITLE'] + fea_names

        t = time()
        dfs = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(load_and_get_samples)
            (f, cols, col_name=col_name, drug_names=drug_names) for f in fea_files[:N]
        )
        # dfs = []
        # for ii, f in enumerate(fea_files):
        #     if ii%10 == 0:
        #         print(ii)
        #     df = load_and_get_samples(f, cols, col_name=col_name, drug_names=drug_names)
        #     dfs.append(df)
        t = time() - t

        fea_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        del dfs

        # fea_df.drop(columns='SMILES', inplace=True)
        fea_df = fea_df[fea_df[col_name].notna()]
        fea_df = fea_df.drop_duplicates(subset=[col_name]).reset_index(drop=True)

        # Cast
        xdata = extract_subset_fea(fea_df, fea_list=['dd'], fea_sep=fea_sep)
        xdata = xdata.astype(np.float32)
        # xdata = xdata.fillna(0)
        meta = fea_df.drop(columns=xdata.columns.tolist())
        fea_df = pd.concat([meta, xdata], axis=1)
        fea_df = fea_df.reset_index(drop=True)
        del meta, xdata

        return fea_df
    else:
        return None


def load_fps(drg_set, fea_name, col_name, drug_names=None,
             fea_sep='_', n_jobs=64, N=None):
    """ Load fingerprints files. The files contains subsets of
    descriptors generated on an HPC.
    """
    files_path = Path(FEA_DIR, drg_set, fea_name).resolve()
    fea_files = sorted(files_path.glob(f'{drg_set}-*.csv'))

    if len(fea_files) > 0:
        fea_prfx = 'ecfp2'
        cols = ['CAT', 'TITLE', 'SMILES', 'fps']
        # cols = ['CAT', 'TITLE', 'fps']

        t = time()
        dfs = Parallel(n_jobs=32, verbose=10)(
            delayed(load_and_get_samples)
            (f, cols, col_name=col_name, drug_names=drug_names) for f in fea_files[:N]
        )
        # dfs = []
        # for ii, f in enumerate(fea_files):
        #     if ii%10 == 0:
        #         print(ii)
        #     df = load_and_get_samples(f, cols, col_name=col_name, drug_names=drug_names)
        #     dfs.append(df)
        t = time() - t

        fea_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        del dfs

        # fea_df.drop(columns='SMILES', inplace=True)
        fea_df = fea_df[fea_df[col_name].notna()]
        fea_df = fea_df.drop_duplicates(subset=[col_name]).reset_index(drop=True)

        aa = Parallel(n_jobs=32, verbose=10)(
            delayed(fps_to_nparr)(x) for x in fea_df['fps'].values
        )
        fea_names = [fea_prfx + fea_sep + str(i+1) for i in range(len(aa[0]))]  # prfx fea names
        cols = ['CAT', 'TITLE', 'SMILES'] + fea_names
        xdata = pd.DataFrame(np.vstack(aa), columns=fea_names, dtype=np.float32)
        del aa

        meta = fea_df.drop(columns='fps')
        fea_df = pd.concat([meta, xdata], axis=1)
        fea_df = fea_df.reset_index(drop=True)
        del xdata, meta

        return fea_df
    else:
        return None


def run(args):
    # import ipdb; ipdb.set_trace()
    t0 = time()
    ID = 'TITLE'
    fea_sep = '_'

    assert args.par_jobs > 0, f"The arg 'par_jobs' must be int >1 (got {args.par_jobs})"

    outdir = Path(args.outdir)
    if outdir is None:
        outdir = Path(filepath, '../out', FEA_DIR.name, drg_set).resolve()
    os.makedirs(outdir, exist_ok=True)

    # Logger
    lg = Logger(outdir/'gen.fea.dfs.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(vars(args))}')

    # ========================================================
    # Aggregate features from files
    # -----------------------------
    drug_names = None
    # N = 20
    N = None

    for fea_name in args.fea_type:
        if 'descriptors' == fea_name:

            dd_fea = load_mordred_descriptors(
                drg_set=args.drg_set,
                fea_name=fea_name,
                col_name=ID,
                drug_names=drug_names,
                fea_sep=fea_sep,
                n_jobs=args.par_jobs,
                N=N)

            print_fn('dd_fea.shape {}'.format(dd_fea.shape))
            dd_fea.to_parquet(outdir/'descriptors.mordred.parquet')
            del dd_fea
        else:
            dd_fea = None

        if 'fps' == fea_name:
            fps_fea = load_fps(
                drg_set=args.drg_set,
                fea_name=fea_name,
                col_name=ID,
                drug_names=drug_names,
                fea_sep=fea_sep,
                n_jobs=args.par_jobs,
                N=N)

            print_fn('fps_fea.shape {}'.format(fps_fea.shape))
            fps_fea.to_parquet(outdir/'fps.ecfp2.parquet')
            del fps_fea
        else:
            fps_fea = None

        if 'images' == fea_name:
            pass
        else:
            img_fea = None

    # --------------------------
    # Generate images
    # --------------------------
    # if 'images' in fea_type:
    #     files_path = Path(FEA_DIR, drg_set, 'images').resolve()
    #     fea_files = sorted( files_path.glob(f'{drg_set}-*.pkl') )

    #     if len(fea_files) > 0:
    #         fea_outpath = outdir/'images'
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
    print_fn('\nRuntime {:.1f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
