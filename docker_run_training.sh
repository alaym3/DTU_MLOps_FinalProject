set -e
dvc pull
python3.9 -u src/models/train_model.py