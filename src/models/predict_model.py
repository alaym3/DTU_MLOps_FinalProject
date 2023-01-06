

# It won't work
# from train_model import Trainer
# trainer = Trainer.trainer

trainer.evaluate(test_dataset)

pred = trainer.predict(test_dataset)