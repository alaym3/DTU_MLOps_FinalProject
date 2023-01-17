import os
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
import sys
import click
import torch
from torch import nn, optim
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding, AutoModel, DistilBertForSequenceClassification
import numpy as np
from datasets import load_metric
from datasets import load_from_disk
import hydra
​
@hydra.main(config_path="config", config_name="model_config.yaml")
def main(cfg):
    # Load train and validation sets
    dataset_path = os.path.join(hydra.utils.get_original_cwd(), 'data\processed')
    train_dataset = load_from_disk(os.path.join(dataset_path, "tokenized_train"))
    val_dataset = load_from_disk(os.path.join(dataset_path,"tokenized_validation"))
​
    # Define the evaluation metrics
    def compute_metrics(eval_pred):
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}
​
    # Load BERT-base-uncased model
    model_name = cfg.model # "distilbert-base-uncased" # "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
​
    # Set DistilBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
​
    # Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
​
    training_args = TrainingArguments(
        output_dir="models/",
        learning_rate=cfg.lr, # 2e-5,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs, # 2,
        weight_decay=cfg.weight_decay, # 0.01,
        save_strategy="epoch",
        no_cuda=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
​
    # Save the model
    trainer.save_model("models/")
​
​
if __name__ == "__main__":
    main()