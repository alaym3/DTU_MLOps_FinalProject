MLOps project description: sentiment classification of Rotten Tomaties movie reviews
==============================

### Overall goal of the project
The goal of the project is to use natural language processing in order to perform sentiment classification on text, in order to predict whether a certain movie review from [Rotten Tomatoes](https://www.rottentomatoes.com/) is positive or negative.

### What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
We will use the [Transformers](https://huggingface.co/) framework since we are working with Natural Language Processing, specifically for sentiment classification of text.

### How to you intend to include the framework into your project
We will work on sentiment classification of text. The Transformers framework is highly flexible and allows many customizations. Many pretrained models for various types of Natural Language Processing tasks exist. They also provide datasets that can be combined with the pretrained models they offer, which makes the framework perfect for our task.

### What data are you going to run on (initially, may change)
We plan to use datasets provided by [HuggingFace](https://huggingface.co/datasets) - we will use the [Rotten Tomatoes review dataset](https://huggingface.co/datasets/rotten_tomatoes). The dataset includes two columns: the text from Rotten Tomatoes reviews for movies, along with a column indicating if the review is positive or negative. [Rotten Tomatoes](https://www.rottentomatoes.com/) is a platform where movie reviews are submitted by expert audiences and regular people.

We may look into other datasets from [HuggingFace](https://huggingface.co/datasets) or [Kaggle](https://www.kaggle.com/datasets) related to reviews of content or services, as we continue.

### What deep learning models do you expect to use
We expect to start by using the pre-trained transformer [bert-base-uncased](https://huggingface.co/bert-base-uncased) since it is the top used model for performing Natural Language Processing tasks on English text, including classification and question-answering. BERT consists of a bidirectional transformer that looks back and forward when analysing the tokens to learn the context of words. Since we want to perform sentiment classification on movie reviews, BERT is a natural model to begin with.


### Checklist
[View our checklist](CHECKLIST.md)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
