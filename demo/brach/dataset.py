import numpy as np

from torch.utils.data import Dataset


class ValidDataset(Dataset):

    def __len__(self):
        return 100

    def __getitem__(self, index):
        #rng = 10 * np.random.rand(1)
        #return np.float32(rng), 0
        return np.array((2, 1), dtype=np.float32), 0


class TestDataset(Dataset):

    def __len__(self):
        return 100

    def __getitem__(self, index):
        #rng = 10 * np.random.rand(1)
        #return np.float32(rng), 0
        return np.array((2, 1), dtype=np.float32), 0


class TrainDataset(Dataset):

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        #rng = 10 * np.random.rand(1)
        #return np.float32(rng), 0
        return np.array((2, 1), dtype=np.float32), 0
