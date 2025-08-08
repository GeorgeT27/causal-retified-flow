import gzip
import os
import random
import struct
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from hps import Hparams
from utils import log_standardize, normalize

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.
    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.
    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.
    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith(".gz") else open
    with open_fcn(path, "rb") as f:
        return _load_uint8(f)

def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    metrics_filename = prefix + "-morpho.csv"
    images_path = os.path.join(root_dir, images_filename)
    labels_path = os.path.join(root_dir, labels_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)
    return images_path, labels_path, metrics_path

def load_morphomnist_like(root_dir, train: bool = True, columns=None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    image_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images=load_idx(image_path)
    labels = load_idx(labels_path)
    if columns is not None and 'index' not in columns:
        usecols=['index'] + list(columns)
    else:
        usecols=columns
    metrics = pd.read_csv(metrics_path, usecols=usecols, index_col='index')
    return images, labels, metrics

class MorphoMNIST(Dataset):
    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: Optional[torchvision.transforms.Compose] = None,
        columns: Optional[List[str]] = None,
        norm: Optional[str] = None,
        concat_pa: bool = True,
    ):
        self.train = train
        self.transform = transform
        self.columns = columns
        self.concat_pa = concat_pa
        self.norm = norm

        cols_not_digit = [c for c in self.columns if c != "digit"]
        images, labels, metrics_df = load_morphomnist_like(
            root_dir, train, cols_not_digit
        )
        self.images = torch.from_numpy(np.array(images)).unsqueeze(1)
        self.labels = torch.from_numpy(np.array(labels)).long()

        if self.columns is None:
            self.columns = metrics_df.columns
        self.samples = {k: torch.tensor(metrics_df[k]) for k in cols_not_digit}

        self.min_max = {
            "thickness": [0.87598526, 6.255515],
            "intensity": [66.601204, 254.90317],
        }

        for k, v in self.samples.items():  # optional preprocessing
            print(f"{k} normalization: {norm}")
            if norm == "[-1,1]":
                self.samples[k] = normalize(
                    v, x_min=self.min_max[k][0], x_max=self.min_max[k][1]
                )
            elif norm == "[0,1]":
                self.samples[k] = normalize(
                    v, x_min=self.min_max[k][0], x_max=self.min_max[k][1], zero_one=True
                )
            elif norm == None:
                pass
            else:
                NotImplementedError(f"{norm} not implemented.")
        print(f"#samples: {len(metrics_df)}\n")

        self.samples.update({"digit": self.labels})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = {}
        sample["x"] = self.images[idx]

        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])

        if self.concat_pa:
            sample["pa"] = torch.cat(
                [
                    v[idx] if k == "digit" else torch.tensor([v[idx]])
                    for k, v in self.samples.items()
                ],
                dim=0,
            )
        else:
            sample.update({k: v[idx] for k, v in self.samples.items()})
        return sample


def morphomnist(args: Hparams) -> Dict[str, MorphoMNIST]:
    # Load data
    if not args.data_dir:
        args.data_dir = "../morphomnist/"

    augmentation = {
        "train": TF.Compose(
            [
                TF.RandomCrop((args.input_res, args.input_res), padding=args.pad),
            ]
        ),
        "eval": TF.Compose(
            [
                TF.Pad(padding=2),  # (32, 32)
            ]
        ),
    }

    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = MorphoMNIST(
            root_dir=args.data_dir,
            train=(split == "train"),  # test set is valid set
            transform=augmentation[("eval" if split != "train" else split)],
            columns=args.parents_x,
            norm=args.context_norm,
            concat_pa=args.concat_pa,
        )
    return datasets

        