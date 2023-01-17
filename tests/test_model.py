import os

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("models")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    return model, tokenizer


def between(predictions):
    in_between = 1
    for class_pred in predictions:
        if (class_pred < 0) | (class_pred > 1):
            in_between = 0
    return in_between


def test_output_shape():
    model, tokenizer = load_model()

    # Prediction for a new phrase
    text = "I love this movie"
    inputs = tokenizer(text, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)
    outputs = model(**inputs, labels=labels)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()

    assert len(predictions[0]) == 2, "There isn't a prediction for every class"
    assert (
        between(predictions[0]) == 1
    ), "The predictions aren't probabilities between 0 and 1"
