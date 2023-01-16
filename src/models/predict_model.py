from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Load test dataset 
#test_dataset = load_from_disk("data/processed/tokenized_test")
#text = test_dataset[0]['text'] # or text = "I love this movie"


@click.command()
@click.argument('text')
def main(text):
    '''Takes a text (str) and returns the probabilities of being a negative or positive text.
           Parameters:
               text (str): input text
           Returns probability of being a negative text and probability of being a negative
    '''

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("models")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Get prediction for a new phrase
    inputs = tokenizer(text, return_tensors="pt") # , padding = True, truncation = True, return_tensors='pt').to('cuda')
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()

    # Store the positive and negative predictions
    neg_prediction = round(predictions[0][0]*100, 2)
    pos_prediction = round(predictions[0][1]*100, 2)

    # Print results
    print('Predictions for the phrase "' + text + '" : ', predictions)
    print(f'Probability of the phrase being negative: {neg_prediction}%')
    print(f'Probability of the phrase being positive: {pos_prediction}%')

    return neg_prediction, pos_prediction

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()