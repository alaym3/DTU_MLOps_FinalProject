from http import HTTPStatus

import torch
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def predict_class(data):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("../models")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Prediction for a new phrase
    inputs = tokenizer(data, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()

    # store the positive and negative predictions
    neg_prediction = round(predictions[0][0] * 100, 2)
    pos_prediction = round(predictions[0][1] * 100, 2)

    if neg_prediction > pos_prediction:
        predicted_class = "negative"
    else:
        predicted_class = "positive"

    return predicted_class, neg_prediction, pos_prediction


@app.get("/input_text/")
def classify_text(data: str):
    predclass, neg, pos = predict_class(data)
    response = {
        "Your text": data,
        "Probability of positive sentiment (%)": pos,
        "Probability of negative sentiment (%)": neg,
        "Is it positive or negative": predclass,
    }
    return response
