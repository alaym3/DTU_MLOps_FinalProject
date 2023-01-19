from datasets import load_from_disk


def load_raw_data():
    # Load train and validation sets
    training = load_from_disk("./data/raw/train")
    testing = load_from_disk("./data/raw/test")
    validation = load_from_disk("./data/raw/validation")

    return training, testing, validation


def test_raw_data_sample_sizes():
    training, testing, validation = load_raw_data()
    assert (
        len(training) == 8530
    ), "Training dataset did not have the correct number of samples"
    assert (
        len(testing) == 1066
    ), "Testing dataset did not have the correct number of samples"
    assert (
        len(validation) == 1066
    ), "Validation dataset did not have the correct number of samples"
