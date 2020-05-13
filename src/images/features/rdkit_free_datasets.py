import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDatasetPreLoaded(Dataset):
    def __init__(self, smiles, descs, imputer_pickle=None, property_func=None, cache=True, values=1, rot=0,
                 images=None):
        self.smiles = smiles
        self.images = images
        self.descs = descs
        self.property_func = property_func
        self.imputer = None
        self.scaler = None
        if imputer_pickle is not None:
            with open(imputer_pickle, 'rb') as f:
                dd = pickle.load(f)
                self.imputer, self.scaler = dd['imputer'], dd['scaler']
        self.cache = cache
        self.values = values
        self.data_cache = {}
        self.transform = transforms.Compose([transforms.RandomRotation(degrees=(0, rot)), transforms.ToTensor()])

    def __getitem__(self, item):
        if self.images is not None:
            image = transforms.ToPILImage()(torch.from_numpy(self.images[item].astype(np.float32) / 255.0))
            image = self.transform(image)

            if self.imputer is not None:
                vec = self.scaler.transform(self.imputer.transform(self.descs[item].reshape(1, -1))).flatten()
            else:
                vec = self.descs[item].flatten()
            vec = torch.from_numpy(np.nan_to_num(vec, nan=0, posinf=0, neginf=0)).float()
            return image, vec
        else:
            assert (False)

    def __len__(self):
        return len(self.smiles)
