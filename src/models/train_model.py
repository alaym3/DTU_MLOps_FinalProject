import cProfile
import os
import pstats

import hydra
import numpy as np
from datasets import load_from_disk, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger

import wandb

wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="rotten_tomatoes"
)


@hydra.main(config_path="../../config", config_name="config_default.yaml")
def main(cfg):
    model_dir = "models"
    output_dir = os.path.join(model_dir, cfg.train.experiment_name)  # .replace("/","\\")  #("models/")#join("models/", cfg.train.experiment_name)
    print(output_dir)
    # Load train and validation sets
    dataset_path = os.path.join(hydra.utils.get_original_cwd(), "data/processed/")
    train_dataset = load_from_disk(os.path.join(dataset_path, "tokenized_train"))
    val_dataset = load_from_disk(os.path.join(dataset_path, "tokenized_validation"))

    validation_inputs = val_dataset.remove_columns(
        ["label", "attention_mask", "input_ids"]
    )
    validation_targets = [
        val_dataset.features["label"].int2str(x) for x in val_dataset["label"]
    ]

    validation_logger = ValidationDataLogger(
        inputs=validation_inputs[:], targets=validation_targets
    )

    accuracy_metric = load_metric("accuracy")

    # Define the evaluation metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # convert predictions from class (0, 1, 2…) to label (Health, Science…)
        prediction_labels = [
            val_dataset.features["label"].int2str(x.item()) for x in predictions
        ]

        # log predictions
        validation_logger.log_predictions(prediction_labels)

        # metrics from the datasets library have a compute method
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # Load BERT-base-uncased model
    model_name = cfg.train.model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Set DistilBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        report_to="wandb",  # enable logging to W&B
        output_dir=os.path.join(model_dir, cfg.train.experiment_name),  # set output directory
        overwrite_output_dir=True,
        evaluation_strategy="steps",  # check evaluation metrics on a given # of steps
        learning_rate=cfg.train.lr,  # we can customize learning rate
        max_steps=cfg.train.max_steps,
        logging_steps=cfg.train.logging_steps,  # we will log every 100 steps
        eval_steps=cfg.train.eval_steps,  # we will perform evaluation every 1000 steps
        eval_accumulation_steps=cfg.train.eval_accumulation_steps,  # report evaluation results after each step
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        # dataloader_num_workers=8,  # number of workers for distributed data loading
        metric_for_best_model="accuracy",
        run_name="my_training_run",  # name of the W&B run
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        tokenizer=tokenizer,  # tokenizer, defined above
        data_collator=data_collator,  # function to use to form a batch from a list of elements of train_dataset or eval_dataset
        compute_metrics=compute_metrics,  # function that will be used to compute metrics at evaluation
    )

    # Train the model
    trainer.train()

    # Evaluation of model
    trainer.evaluate()

    # Save the model into models/
    trainer.save_model(os.path.join(model_dir, cfg.train.experiment_name))


if __name__ == "__main__":
    main()

    # Profiling: creates profile.dat, profile_time.txt and profile_calls.txt with profiling info
    cProfile.run("main()", "profile.dat")

    with open("profile_time.txt", "w") as f:
        p = pstats.Stats("profile.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("profile_calls.txt", "w") as f:
        p = pstats.Stats("profile.dat", stream=f)
        p.sort_stats("calls").print_stats()
