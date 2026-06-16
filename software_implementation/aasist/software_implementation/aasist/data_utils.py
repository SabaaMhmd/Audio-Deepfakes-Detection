import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(csv_file, is_eval=False):

    df = pd.read_csv(csv_file)

    file_list = df["path"].tolist()

    if is_eval:
        return file_list

    labels = dict(
        zip(
            df["path"],
            df["label"]
        )
    )

    return labels, file_list

def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 96000):
    x_len = x.shape[0]

    if x_len > max_len:
        stt = np.random.randint(0, x_len - max_len + 1)
        return x[stt:stt + max_len]

    elif x_len == max_len:
        return x

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 96000  # take ~6 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(key)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):

    def __init__(self, list_IDs, labels):

        self.list_IDs = list_IDs
        self.labels = labels
        self.cut = 96000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        key = self.list_IDs[index]

        X, _ = sf.read(key)

        X_pad = pad(X, self.cut)

        x_inp = Tensor(X_pad)

        y = self.labels[key]

        return x_inp, y