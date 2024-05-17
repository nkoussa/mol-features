"""
This script computes molecular features and saves in appropriate files.
The feature columns are prefixed with appropriate string:
    - Mordred descriptors (dd_)
    - ECFP2 (ecfp2_)
    - ECFP4 (ecfp4_)
    - ECFP6 (ecfp6_)
    - Images (stored in dict)
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

# Utils
# sys.path.append( os.path.abspath(filepath/'../utils') )
from utils.classlogger import Logger
from utils.utils import get_print_func, drop_dup_rows, dropna
from utils.smiles import canon_smiles, smiles_to_mordred, smiles_to_fps, smiles_to_images

filepath = Path(__file__).resolve().parent

# Date
# t = datetime.now()
# t = [t.year, '-', t.month, '-', t.day]
# date = ''.join([str(i) for i in t])

# SMILES_PATH
# SMILES_PATH = Path(filepath, '../data/raw/UC-molecules/UC.smi')
# SMILES_PATH = Path(filepath, '../data/raw/Baseline-Screen-Datasets/BL1(ena+db)/ena+db.smi')
# SMILES_PATH = Path(filepath, '../data/raw/Baseline-Screen-Datasets/BL2-current/BL2.smi').resolve()
SMILES_PATH = Path(filepath, '../sample_data/BL2.smi.sample').resolve()

# Global outdir
GOUT = Path(filepath, '../out').resolve()


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate molecular feature sets.')

    parser.add_argument('--smiles_path',
                        type=str,
                        default=SMILES_PATH,
                        help=f'Full path to the smiles file (default: {SMILES_PATH}).')
    parser.add_argument('--id_name',
                        type=str,
                        required=True,
                        help="Column name that serves as the drug ID (we usually use\
                        'TITLE' for covid and 'ID' for pilot1.")
    parser.add_argument('--gout',
                        type=str,
                        default=GOUT,
                        help=f'Output dir (default: {GOUT}).')
    parser.add_argument('--fea_type',
                        type=str,
                        default=['mordred'],
                        nargs='+',
                        choices=['mordred', 'images', 'fps', 'canon_smi', 'infomax'],
                        help='Feature type (default: descriptors).')
    parser.add_argument('--nbits',
                        type=int,
                        default=2024,
                        help='Number of bits in Morgan (circular) fingerprints (default: 2024).')
    parser.add_argument('--par_jobs',
                        type=int,
                        default=1,
                        help='Number of joblib parallel jobs (default: 1).')
    # parser.add_argument('--i1',
    #                     type=int,
    #                     default=0,
    #                     help='Start index of a smiles sample (default: 0).')
    # parser.add_argument('--i2',
    #                     type=int,
    #                     default=None,
    #                     help='End index of smiles sample (default: None).')
    parser.add_argument('--ignore_3D',
                        action='store_true',
                        help='Ignore 3-D Mordred descriptors (default: False).')
    parser.add_argument('--impute',
                        action='store_true',
                        help='Whether to keep NA values (default: False).')

    args = parser.parse_args(args)
    return args


def add_fea_prfx(df, prfx: str, id0: int):
    """ Add prefix feature columns. """
    return df.rename(columns={s: prfx+str(s) for s in df.columns[id0:]})


def get_image(mol):
    image = (255 * transforms.ToTensor()(Invert()(generateFeatures.smiles_to_image(mol))).numpy()).astype(np.uint8)
    return image

# def get_image(mol):
#     """ (AP) breakdown of the function. """
#     im = generateFeatures.smiles_to_image(mol)
#     im = Invert()(im)
#     im = transforms.ToTensor()(im)
#     im = im.numpy()
#     im = 255 * im
#     image = im.astype(np.uint8)
#     return image


def run(args):
    t0 = time()
    smiles_path = args.smiles_path
    id_name = args.id_name
    nbits = args.nbits
    par_jobs = args.par_jobs
    fea_type = args.fea_type

    print('\nLoad SMILES.')
    smiles_path = Path(args.smiles_path)
    # ERROR: TypeError: argument of type 'PosixPath' is not iterable
    # if "drugbank" in smiles_path:
    #     smi = pd.read_csv(smiles_path, sep='\t', usecols=[0])  # drugbank all
    # else:
    #     smi = pd.read_csv(smiles_path, sep='\t')
    smi = pd.read_csv(smiles_path, sep='\t')


    new_id_name = "DrugID"  # rename column drug id_name
    smi = smi.rename(columns={id_name: new_id_name})
    id_name = new_id_name

    # if "drugbank" in smiles_path:
    #     # for TOM's drugbank-all.smi file!!
    #     smi = smi.astype({'SMILES': str})
    #     smi['SMILES'] = smi['SMILES'].map(lambda x: x.strip())
    #     smi['SMILES'] = smi['SMILES'].map(lambda x: x.split()[0])
    #     fea_id0 = smi.shape[1]  # index of the first feature
    # else:
    #     smi = smi.astype({'SMILES': str, id_name: str})
    #     smi['SMILES'] = smi['SMILES'].map(lambda x: x.strip())
    #     smi[id_name] = smi[id_name].map(lambda x: x.strip())
    #     # n_smiles = smi.shape[0]
    #     fea_id0 = smi.shape[1]  # index of the first feature

    smi = smi.astype({'SMILES': str, id_name: str})
    smi['SMILES'] = smi['SMILES'].map(lambda x: x.strip())
    smi[id_name] = smi[id_name].map(lambda x: x.strip())
    fea_id0 = smi.shape[1]  # index of the first feature

    # Create Outdir
    # i1, i2 = args.i1, args.i2
    # ids_dir = 'smi.ids.{}-{}'.format(i1, i2)
    # if i2 is None:
    #     i2 = n_smiles
    # gout = Path(args.gout, ids_dir)
    gout = Path(args.gout)
    os.makedirs(gout, exist_ok=True)

    # Logger
    lg = Logger(gout/'gen.fea.dfs.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(vars(args))}')

    print_fn('\nInput data path  {}'.format(smiles_path))
    print_fn('Output data dir  {}'.format(gout))

    # Duplicates
    # dup = smi[ smi.duplicated(subset=['smiles'], keep=False) ].reset_index(drop=True)
    # print(dup['smiles'].value_counts())

    # Drop duplicates
    smi = drop_dup_rows(smi, print_fn)

    # Exract subset SMILES
    # smi = smi.iloc[i1:i2+1, :].reset_index(drop=True)

    print_fn('\nCanonicalize SMILES.')
    can_smi_vec = canon_smiles(smi['SMILES'], par_jobs=par_jobs)
    can_smi_vec = pd.Series(can_smi_vec)

    # Save bad SMILES to file (that were not canonicalized)
    nan_ids = can_smi_vec.isna()
    bad_smi = smi[nan_ids]
    if len(bad_smi) > 0:
        bad_smi.to_csv(gout/'smi_canon_err.csv', index=False)

    # Keep the good (canonicalized) SMILES
    smi['SMILES'] = can_smi_vec
    smi = smi[~nan_ids].reset_index(drop=True)

    # ========================================================
    # Save canonical smiles
    # ---------------
    if 'canon_smi' in fea_type:
        fname = "smiles_canonical"
        smi.to_csv(gout/(fname+'.tsv'), sep='\t', index=False)

    # ========================================================
    # Generate infomax fingerprints (see: https://github.com/NetPharMedGroup/publication_fingerprint)
    # ---------------
    if 'infomax' in fea_type:
         import torch
        from torch.utils.data import DataLoader
        torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
        import dgl
        from dgl.nn.pytorch.glob import AvgPooling
        from dgllife.model import load_pretrained
        from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
        from rdkit import Chem

        # load pretrained model
        model = load_pretrained('gin_supervised_infomax') # contextpred infomax edgepred masking
        model.to('cpu')
        model.eval()
    
        b_res = smi["SMILES"].tolist()
        graphs = []
        for smi in b_res:
            #print(smi)
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                g = mol_to_bigraph(mol, add_self_loop=True,
                                node_featurizer=PretrainAtomFeaturizer(),
                                edge_featurizer=PretrainBondFeaturizer(),
                                canonical_atom_order=True)
                graphs.append(g)

            except:
                continue

        def collate(gs):
            return dgl.batch(gs)

        data_loader = DataLoader(graphs, batch_size=256, collate_fn=collate, shuffle=False)

        readout = AvgPooling()


        mol_emb = []
        for batch_id, bg in enumerate(data_loader):
            bg = bg.to('cpu')
            nfeats = [bg.ndata.pop('atomic_number').to('cpu'),
                    bg.ndata.pop('chirality_type').to('cpu')]
            efeats = [bg.edata.pop('bond_type').to('cpu'),
                    bg.edata.pop('bond_direction_type').to('cpu')]
            with torch.no_grad():
                node_repr = model(bg, nfeats, efeats)
            mol_emb.append(readout(bg, node_repr))
        mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()


        infomax_df = pd.concat([smi, pd.DataFrame(mol_emb)], axis=1)
    
        fname = "infomax_fingerprints"
        infomax_df.to_csv(gout/(fname+'.tsv'), sep='\t', index=False)



    # ========================================================
    # Generate images
    # ---------------
    if 'images' in fea_type:
        images = smiles_to_images(smi, smi_col_name='SMILES', title_col_name=id_name,
                                  molSize=(128, 128), kekulize=True, par_jobs=par_jobs)
        # print(images[0].keys())
        # img_outpath = gout/f'images.ids.{i1}-{i2}.pkl'
        img_outpath = gout/'images.pkl'

        # Dump images to file (list of dicts)
        pickle.dump(images, open(img_outpath, 'wb'))
        # Load pkl
        # aa = pickle.load(open(img_outpath, 'rb'))
        # sum(images[0]['img'].reshape(-1,)-aa[0]['img'].reshape(-1,))

    # ========================================================
    # Generate fingerprints
    # ---------------------
    if 'fps' in fea_type:
        def gen_fps_and_save(smi, radius=1, nbits=2048, par_jobs=par_jobs):
            ecfp = smiles_to_fps(smi, smi_name='SMILES', radius=radius,
                                 nbits=nbits, par_jobs=par_jobs)
            ecfp = add_fea_prfx(ecfp, prfx=f'ecfp{2*radius}.', id0=fea_id0)
            # ecfp.to_parquet(gout/f'ecfp{2*radius}.ids.{i1}-{i2}.{file_format}')
            ecfp.to_parquet(gout/f'ecfp{2*radius}_nbits{nbits}.parquet')
            ecfp.to_csv(gout/f'ecfp{2*radius}_nbits{nbits}', sep='\t', index=False)
            del ecfp

        gen_fps_and_save(smi, radius=1, nbits=nbits, par_jobs=par_jobs)
        gen_fps_and_save(smi, radius=2, nbits=nbits, par_jobs=par_jobs)
        gen_fps_and_save(smi, radius=3, nbits=nbits, par_jobs=par_jobs)

    # ========================================================
    # Generate Mordred
    # ----------------
    if 'mordred' in fea_type:
        dd = smiles_to_mordred(smi, smi_name='SMILES',
                               ignore_3D=args.ignore_3D,
                               par_jobs=par_jobs)
        fea_sep = '.'
        # dd = add_fea_prfx(dd, prfx=f'dd{fea_sep}', id0=fea_id0)
        dd = add_fea_prfx(dd, prfx=f'mordred{fea_sep}', id0=fea_id0)

        # Filter NaNs (step 1)
        # Drop rows where all values are NaNs
        print_fn('\nDrop rows where all values are NaN.')
        print_fn('Shape: {}'.format(dd.shape))
        idx = ( dd.isna().sum(axis=1) == dd.shape[1] ).values
        dd = dd.iloc[~idx, :].reset_index(drop=True)
        # Drop cols where all values are NaNs
        # idx = ( dd.isna().sum(axis=0) == dd.shape[0] ).values
        # dd = dd.iloc[:, ~idx].reset_index(drop=True)
        print_fn('Shape: {}'.format(dd.shape))

        # Filter NaNs (step 2)
        # Drop rows based on a thershold of NaN values.
        # print(dd.isna().sum(axis=1).sort_values(ascending=False))
        # p=dd.isna().sum(axis=1).sort_values(ascending=False).hist(bins=100);
        th = 0.25
        print_fn('\nDrop rows with at least {} NaNs (at least {} out of {}).'.format(
            th, int(th * dd.shape[1]), dd.shape[1]))
        print_fn('Shape: {}'.format(dd.shape))
        dd = dropna(dd, axis=0, th=th)
        print_fn('Shape: {}'.format(dd.shape))

        # Cast features (descriptors)
        print_fn('\nCast descriptors to float.')
        dd = dd.astype({c: np.float32 for c in dd.columns[fea_id0:]})

        # Dump the count of NANs in each column
        aa = dd.isna().sum(axis=0).reset_index()
        aa = aa.rename(columns={'index': 'col', 0: 'count'})
        aa = aa.sort_values('count', ascending=False).reset_index(drop=True)
        aa.to_csv(gout/'nan_count_per_col.csv', index=False)

        # # Impute missing values
        # if args.impute:
        #     print_fn('\nImpute NaNs.')
        #     print_fn('Total NaNs: {}'.format( dd.isna().values.flatten().sum() ))
        #     dd.iloc[:, fea_id0:] = dd.iloc[:, fea_id0:].fillna(0.0)
        #     print_fn('Total NaNs: {}'.format( dd.isna().values.flatten().sum() ))

        dd = dd.reset_index(drop=True)

        # Save
        # fname = 'dd.mordred.{}'.format('' if args.impute else 'with.nans')
        print_fn('\nSave mordred with nans.')
        fname = 'dd.mordred.with.nans'
        dd.to_parquet(gout/(fname+'.parquet'))
        dd.to_csv(gout/(fname+'.tsv'), sep='\t', index=False)
        # dd.to_csv( gout/'dd.ids.{}-{}.{}'.format(i1, i2, file_format), index=False )

        # Impute missing values
        print_fn('\nImpute NaNs.')
        print_fn('Total NaNs: {}'.format( dd.iloc[:, fea_id0:].isna().values.flatten().sum() ))
        dd.iloc[:, fea_id0:] = dd.iloc[:, fea_id0:].fillna(0.0)
        print_fn('Total NaNs: {}'.format( dd.iloc[:, fea_id0:].isna().values.flatten().sum() ))

        # Save
        print_fn('\nSave mordred with nans imputed.')
        fname = 'dd.mordred'
        dd.to_parquet(gout/(fname+'.parquet'))
        dd.to_csv(gout/(fname+'.tsv'), sep='\t', index=False)
        # dd.to_csv( gout/'dd.ids.{}-{}.{}'.format(i1, i2, file_format), index=False )

    # ======================================================
    print_fn('\nRuntime {:.1f} mins'.format((time()-t0)/60))
    print_fn('Done.')
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
