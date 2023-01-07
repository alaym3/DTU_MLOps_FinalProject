import os
from torch.utils.data import Dataset, DataLoader
from src.data.dataloader import RottenTomatoes
import torch

training = RottenTomatoes(os.path.join(os.getcwd(),"data/processed/train.pt"))
print(training[0])