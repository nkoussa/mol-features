from time import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def canon_single_smile(smi):
    """ Canonicalize single SMILES string. """
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles( smi )
        can_smi = Chem.MolToSmiles(mol, canonical=True)
    except:
        # import ipdb; ipdb.set_trace()
        print(f'Error in smile: {smi}')
        can_smi = np.nan
    return can_smi


# def canon_df(df, smi_name='SMILES', par_jobs=16):
#     """ Canonicalize the SMILES sting in column name smi_name. """
#     smi_vec = []
#     t0 = time()
#     if par_jobs>1:
#         smi_vec = Parallel(n_jobs=par_jobs, verbose=1)(
#                 delayed(canon_single_smile)(smi) for smi in df[smi_name].tolist())
#     else:
#         for i, smi in enumerate(df[smi_name].values):
#             if i%100000==0:
#                 print('{}: {:.2f} mins'.format(i, (time()-t0)/60 ))
#             can_smi = canon_single_smile( smi )
#             smi_vec.append( can_smi ) # TODO: consider return this, instead of modifying df
#     df.loc[:, 'SMILES'] = smi_vec
#     return df


def canon_smiles(smiles, par_jobs=16):
    """ Canonicalize each smile in the input smiles array. """
    smi_vec = []
    t0 = time()
    if par_jobs>1:
        smi_vec = Parallel(n_jobs=par_jobs, verbose=1)(
                delayed(canon_single_smile)(smi) for smi in smiles)
    else:
        for i, smi in enumerate(smiles):
            if i%100000==0:
                print('{}: {:.2f} mins'.format(i, (time()-t0)/60 ))
            can_smi = canon_single_smile( smi )
            smi_vec.append( can_smi )
    return smi_vec


def fps_single_smile(smi, radius=2, nbits=2048):
    """ Convert single smiles into Morgan fingerprints.
    From www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints:
    When comparing the ECFP/FCFP FPs and the Morgan FPs generated by the RDKit, remember that the
    4 in ECFP4 corresponds to the diameter of the atom environments considered, while the Morgan FPs
    take a radius parameter. So when radius=2, this is roughly equivalent to ECFP4 and FCFP4.

    https://www.researchgate.net/post/How_to_choose_bits_and_radius_during_circular_fingerprint_calculation_in_RDKit
    The Morgan fingerprint is basically a reimplementation of the extended conectivity
    fingerprint (ECFP). There is a paper describing it if you want more details but in
    essence you go through each atom of the molecule and obtain all possible paths through
    this atom with a specific radius. Then each unique path is hashed into a number with a
    maximum based on bit number. The higher the radius, the bigger fragments are encoded.
    So a Morgan radius 2 has all paths found in Morgan radius 1 and then some additional
    ones. In general, people use radius 2 (similar to ECFP4) and 3 (similar to ECFP6). As
    for number of bits it depends on your dataset. The higher bit number the more
    discriminative your fingerprint can be. If you have a large and diverse dataset but
    only have 32 bits, it will not be good. I would start at 1024 bits but also check
    higher numbers and see if you are losing too much information.
    """
    # stackoverflow.com/questions/54809506/
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    # smi=pybel.readstring("smi", row["smiles"]).write("can").strip()
    mol = Chem.MolFromSmiles( smi )
    fp = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=nbits)
    fp_arr = np.array(fp) # .tolist()
    # res = {'SMILES': smi, 'fps': fp_arr}
    # return res
    return fp_arr


def smiles_to_fps(df, smi_name='SMILES', radius=2, nbits=2048, par_jobs=8):
    """ Generate dataframe of fingerprints from SMILES. """
    df = df.reset_index(drop=True)
    smiles = df[smi_name].values
    res = Parallel(n_jobs=par_jobs, verbose=1)(
            delayed(fps_single_smile)(smi, radius=radius, nbits=nbits) for smi in smiles)
    # fps_list = [dct['fps'] for dct in res]
    # smi_list = [dct['SMILES'] for dct in res]
    # fps_arr = np.vstack( fps_list )
    # fps = pd.DataFrame( fps_arr )
    # fps.insert(loc=0, column='SMILES', value=smi_list)
    fps_arr = np.vstack( res )
    fps = pd.DataFrame( fps_arr, dtype=np.int8 )
    fps = pd.concat([df, fps], axis=1)
    return fps


def smiles_to_mordred(df, smi_name='SMILES', ignore_3D=True, par_jobs=8):
    """ Generate dataframe of Mordred descriptors from SMILES. """
    from rdkit import Chem
    from mordred import Calculator, descriptors
    df = df.reset_index(drop=True)

    # Convert SMILES to mols
    smiles = df[smi_name].values
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]

    # Create Mordred calculator and compute descriptors from molecules 
    # mordred-descriptor.github.io/documentation/master/_modules/mordred/_base/calculator.html#Calculator.pandas
    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    dd = calc.pandas( mols, nproc=par_jobs, nmols=None, quiet=False, ipynb=False )
    dd = pd.concat([df, dd], axis=1)
    return dd

# =================================================================================
# class SMILES_TO_IMAGES():
# TODO

def smiles_to_images(df, smi_col_name='SMILES', title_col_name='TITLE',
                     molSize=(128, 128), kekulize=True, par_jobs=8):
    """ Convert SMILES to images of molecules.
    Args:
        df : dataframe with smiles and id columns of the molecules
        smi_col_name : col name that stores SMILES string
        title_col_name : col name that is used as molecule ID (e.g., TITLE)
    """
    assert smi_col_name in df.columns, f'{smi_col_name} is not in df.columns'
    assert title_col_name in df.columns, f'{title_col_name} is not in df.columns'
    kwargs = {'molSize': (128, 128), 'kekulize': True}
    df = df.reset_index(drop=True)
    smiles = df[smi_col_name].tolist()
    # import pdb; pdb.set_trace()

    if par_jobs>1:
        res = Parallel(n_jobs=par_jobs, verbose=1)(
                delayed(calc_img_item)(
                    df.loc[i, smi_col_name], df.loc[i, title_col_name],
                    **kwargs) for i in range(df.shape[0]))
    else:
        res = []
        for i in range( df.shape[0] ):
            mol_smi = df.loc[i, smi_col_name]
            mol_title = df.loc[i, title_col_name]
            item = calc_img_item( mol_smi, mol_title, **kwargs )
            res.append( item )
    return res


def calc_img_item(mol_smi, mol_title, molSize=(128, 128), kekulize=True):
    """ Call for a func that generates the mol image and put it
    in a dict with appropriate metadata.
    """
    kwargs = {'molSize': molSize, 'kekulize': kekulize}
    ret = smile_to_image(mol_smi, **kwargs)
    ret.update( {'TITLE': mol_title} )
    ret = {k: ret[k] for k in ['SMILES', 'TITLE', 'img']} # re-org dict
    return ret


def smile_to_image(smi, molSize=(128, 128), kekulize=True, mol_computed=True):
    """ Covenrt SMILES to image.
    This is from:
    github.com/2019-ncovgroup/MolecularAttention/blob/covid-images/features/generateFeatures.py
    """
    # (ap)
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    import io
    import cairosvg
    from PIL import Image
    from images.features.utils import Invert
    from torchvision import transforms
    # (ap)
    
    # if not mol_computed:
    #     # If mol is already converted from SMILES
    #     mol = Chem.MolFromSmiles(mol)
    mol = Chem.MolFromSmiles( smi )

    mc = Chem.Mol( mol.ToBinary() )

    if kekulize:
        try:
            Chem.Kekulize( mc )
        except:
            mc = Chem.Mol( mol.ToBinary() )

    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords( mc )

    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule( mc )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    image = Image.open(io.BytesIO(cairosvg.svg2png(
        bytestring=svg, parent_width=100, parent_height=100, scale=1)))
    image.convert('RGB')
    image = Invert()(image)

    # That's from get_image_dct(mol)
    # image = (255 * transforms.ToTensor()(Invert()(generateFeatures.smiles_to_image(mol))).numpy()).astype(np.uint8)
    image = Invert()(image) # TODO is this redundant??
    image = transforms.ToTensor()(image)
    image = image.numpy()
    image = 255 * image
    image = image.astype(np.uint8)

    ret = {'SMILES': smi, 'img': image}
    return ret

