import argparse
import multiprocessing

from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from torchvision import transforms
from tqdm import tqdm

from features import generateFeatures
from features.utils import Invert

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)

    parser.add_argument('-s', type=int, default=None, help='Process a subset of samples')
    parser.add_argument('--par_jobs', default=64, type=int, 
                        help=f'Number of joblib parallel jobs (default: 64).')
    return parser.parse_args()

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

if __name__ == '__main__':
    t0 = time()
    args = get_args()
    images = []

    # smiles = pd.read_csv(args.i, header=None)
    smiles = pd.read_csv(args.i, names=['SMILES', 'TITLE'], sep='\t')
    n = smiles.shape[0]
    # smiles = list(smiles.iloc[:, 0])
    smiles = list(smiles['SMILES'])

    if args.s is not None:
        n_samples = 1000
    else:
        n_samples = n
    smiles = smiles[:n_samples]  # (ap) take subset

    smiles = filter(lambda x: x is not None, map(lambda x: Chem.MolFromSmiles(x), smiles))

    # with multiprocessing.Pool(32) as pool:
    #     smiles = pool.imap(get_image, smiles)
    #     for im in tqdm(smiles, total=n):
    #         images.append(im)

    from joblib import Parallel, delayed
    par_jobs = args.par_jobs
    images = Parallel(n_jobs=par_jobs, verbose=20)(
            delayed(get_image)(mol=smi) for smi in smiles )

    if (time()-t0)//3600 > 0:
        print('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
    else:
        print('Runtime: {:.1f} min'.format( (time()-t0)/60) )
    
    # for smi in smiles:
    #     im = get_image(smi)
    #     images.append(im)

    images = np.stack(images).astype(np.uint8)
    # np.save(args.o, images)

    outpath = Path(args.o).with_suffix('').name
    outpath = Path(args.o)/( Path(args.i).with_suffix('').name+f'.{n_samples}.images.npy' )
    np.save(outpath, images)
    print('Done.')


