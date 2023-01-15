from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import torch

test_dataset = load_from_disk("data/processed/tokenized_test")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("models")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Prediction for a new phrase
text = test_dataset[0]['text'] # or text = "I love this movie"
inputs = tokenizer(text, return_tensors="pt") # , padding = True, truncation = True, return_tensors='pt').to('cuda')
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()
print('Predictions for the phrase "' + text + '" : ', predictions)
