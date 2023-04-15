import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError
