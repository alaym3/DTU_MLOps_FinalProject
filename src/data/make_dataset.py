# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datasets import load_dataset
from torchvision import transforms
import torch
import os



@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    dataset_name = "rotten_tomatoes"
    train = load_dataset(dataset_name, split="train")
    test = load_dataset(dataset_name, split="test")
    validation = load_dataset(dataset_name, split="validation")

    torch.save(train, os.path.join(output_filepath, 'train.pt'))
    torch.save(test, os.path.join(output_filepath, 'test.pt'))
    torch.save(validation, os.path.join(output_filepath, 'validation.pt'))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()