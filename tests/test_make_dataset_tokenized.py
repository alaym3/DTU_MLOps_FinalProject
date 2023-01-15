from datasets import load_from_disk


def load_tokenized_data():
    # Load train and validation sets
    training = load_from_disk("data/processed/tokenized_train")
    testing = load_from_disk("data/processed/tokenized_test")
    validation = load_from_disk("data/processed/tokenized_validation")
    
    return training, testing, validation


def test_tokenized_data():
    training, testing, validation = load_tokenized_data()
    assert (
        training.shape == (8530, 5)
    ), "Training dataset did not have the correct shape of (8530, 5) after tokenization"
    assert (
        testing.shape == (1066, 5)
    ), "Testing dataset did not have the correct shape of (1066, 5) after tokenization"
    assert (
        validation.shape == (1066, 5)
    ), "Validation dataset did not have the correct shape of (1066, 5) after tokenization"

