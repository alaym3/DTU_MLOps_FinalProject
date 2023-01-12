import torch
from torch.utils.data import Dataset


class RottenTomatoes(Dataset):
    def __init__(self, path):

        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return {"data": data}
