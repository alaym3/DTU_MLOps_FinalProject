import argparse
# turn off wandb so that this can run in docker
import os
import sys

import click
import numpy as np
import sklearn
import torch
from datasets import load_from_disk, load_metric
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          DistilBertForSequenceClassification, Trainer,
                          TrainingArguments)

# import wandb
# wandb.login()
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="rotten_tomatoes")
    


# Turn off wandb so that this can run in docker
import os

# os.environ["WANDB_DISABLED"] = "true"

# Load train and validation sets
train_dataset = load_from_disk("data/processed/tokenized_train")
val_dataset = load_from_disk("data/processed/tokenized_validation")

# Define the evaluation metrics
def compute_metrics(eval_pred):
    '''Defines the evaluation metrics.
           Parameters:
               eval_pred
       Returns a dictionary string to metric values.
    '''
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    # wandb.log({"accuracy": accuracy, "f1": f1})
    return {"accuracy": accuracy, "f1": f1}

# Load BERT-base-uncased model
model_name = "distilbert-base-uncased" # "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Set DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="models/",                   # output directory where model predictions and checkpoints will be written
    learning_rate=2e-5,                     # learning rate (float)
    per_device_train_batch_size=16,         # batch size per GPU/TPU core/CPU for training
    per_device_eval_batch_size=16,          # batch size per GPU/TPU core/CPU for evaluation
    num_train_epochs=2,                     # total number of training epochs to perform
    weight_decay=0.01,                      # weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    save_strategy="epoch")                  # checkpoint save strategy to adopt during training

trainer = Trainer(
    model=model,                            # the instantiated 🤗 Transformers model to be trained
    args=training_args,                     # training arguments, defined above
    train_dataset=train_dataset,            # training dataset
    eval_dataset=val_dataset,               # evaluation dataset
    tokenizer=tokenizer,                    # tokenizer, defined above
    data_collator=data_collator,            # function to use to form a batch from a list of elements of train_dataset or eval_dataset
    compute_metrics=compute_metrics,        # function that will be used to compute metrics at evaluation
)

# Train the model
trainer.train() # only uncomment if you want to re-train the model

# Evaluation of model
trainer.evaluate()

# Save the model into models/
trainer.save_model("models/")
