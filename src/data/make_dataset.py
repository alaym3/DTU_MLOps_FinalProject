# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from transformers import AutoTokenizer
import hydra


def get_tokenizer():
    '''Get the tokenizer
    '''
    tokenizer_type = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, use_fast=False)
    return tokenizer


def preprocess_function(examples):
    '''Prepare the text inputs for the model.
    '''
    tokenizer = get_tokenizer()
    return tokenizer(examples["text"], truncation=True)


@hydra.main(config_path="../../config", config_name="config_default.yaml")
def main(cfg):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Downloading raw data from huggingface
    dataset_name = cfg.data.dataset_name # "rotten_tomatoes"
    train = load_dataset(dataset_name, split="train")
    test = load_dataset(dataset_name, split="test")
    validation = load_dataset(dataset_name, split="validation")

    # Saving raw data to data/raw/
    train.save_to_disk(os.path.join(cfg.data.input_filepath, "train"))
    test.save_to_disk(os.path.join(cfg.data.input_filepath, "test"))
    validation.save_to_disk(os.path.join(cfg.data.input_filepath, "validation"))

    # #Tokenizing raw data
    tokenized_train = train.map(preprocess_function, batched=True)
    tokenized_test = test.map(preprocess_function, batched=True)
    tokenized_validation = validation.map(preprocess_function, batched=True)

    # Saving tokenized data to data/processed/
    tokenized_train.save_to_disk(os.path.join(cfg.data.output_filepath, "tokenized_train"))
    tokenized_test.save_to_disk(os.path.join(cfg.data.output_filepath, "tokenized_test"))
    tokenized_validation.save_to_disk(
        os.path.join(cfg.data.output_filepath, "tokenized_validation")
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
