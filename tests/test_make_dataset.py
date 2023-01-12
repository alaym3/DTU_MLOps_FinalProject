# do this to load the dataset
import os
from torch.utils.data import Dataset, DataLoader
from src.data.dataloader import RottenTomatoes
import pytest


def load_data():
    training = RottenTomatoes(os.path.join(os.getcwd(), "data/processed/train.pt"))
    testing = RottenTomatoes(os.path.join(os.getcwd(), "data/processed/test.pt"))
    validation = RottenTomatoes(os.path.join(os.getcwd(), "data/processed/validation.pt"))

    return training, testing, validation


def test_data_sample_sizes():
    training, testing, validation = load_data()
    assert (
        len(training) == 8530
    ), "Training dataset did not have the correct number of samples"
    assert (
        len(testing) == 1066
    ), "Testing dataset did not have the correct number of samples"
    assert (
        len(validation) == 1066
    ), "Validation dataset did not have the correct number of samples"
