import numpy as np
import torch

from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, ftype='train', transform=None):
        fname = 'datasets/context-%s.txt' % ftype
        items = []
        with open(fname) as f:
            for ln in f:
                if len(ln.strip()) > 0:
                    elms = eval(ln)
                    if len(elms) == 12:
                        items.append(np.array(elms, dtype=np.int64))
        self.data = torch.LongTensor(np.array(items, dtype=np.int64))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
       return self.data[idx]
