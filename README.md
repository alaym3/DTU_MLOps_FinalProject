MLOps project description: sentiment classification of Rotten Tomatoes movie reviews
==============================
# [Application link!](https://streamlit-pqpw5ljsba-ew.a.run.app)

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

### Last update on the model we used
When we were starting to work on our project, we first considered using the pre-trained transformer [bert-base-uncased](https://huggingface.co/bert-base-uncased) as explained previously. However, we finally used [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) which is a distilled version of the Bert base model, this means that it has been trained to mimic the behavior of BERT-base-uncased while being smaller and faster. We made this choice because DistilBERT-base-uncased is faster and more memory-efficient, making it more suitable for use on devices with limited resources.


# How does our app work?
Our sentiment classification application uses [Streamlit](https://streamlit.io/) and is deployed on Google Cloud via [this link](https://streamlit-pqpw5ljsba-ew.a.run.app). The Streamlit app is containerized and deployed via Cloud Run. Our custom trained huggingface transformers model is downloaded from our Google Cloud Bucket and users are able to type in any text input they want, and view the probability of the phrase being positive and negative. It uses the streamlit.dockerfile in the docker folder.

### Do you want to run the image locally?
- clone our repo `git clone https://github.com/alaym3/DTU_MLOps_FinalProject.git`
- have [Docker](https://www.docker.com/) installed and running
- run these:
   - `docker build -f docker/streamlit.dockerfile . -t streamlit:latest`
   - `docker run -p 8080:8080 streamlit:latest`

### Do you want to rebuild this all on your own and deploy your own app via Streamlit on Cloud Run??
- clone our repo `git clone https://github.com/alaym3/DTU_MLOps_FinalProject.git`
- have [Docker](https://www.docker.com/) installed and running
- create a project in [Google Cloud](https://console.cloud.google.com/)
- make sure you have money in your billing account since costs are incurred by the container!! ????????????
- ensure you are [authenticated with google cloud auth](https://cloud.google.com/container-registry/docs/advanced-authentication) - check the `gcloud auth activate-service-account` command specifically. [This article](https://cloud.google.com/sdk/gcloud/reference/auth/activate-service-account) may also help.
- add a creds folder and a creds.json inside of it (pertaining to the config.json file auto created by the above steps), connected to your project in Google Cloud
- add folders in root called `models/`, `data/raw/`, and `data/processed/`
- run `make data` to create the datasets
- run `make train` to train the model and save the model files to the `models/` folder
- run the streamlit.dockerfile found in the docker folder, tag it, push the image to your project, then run a command to auto deploy via Cloud Run. Example below:
   - `docker build --platform linux/amd64 -f docker/streamlit.dockerfile . -t streamlit:latest`
   - `docker tag streamlit:latest gcr.io/<project-id>/streamlit`
   - `docker push gcr.io/<project-id>/streamlit`
   - `gcloud run deploy streamlit --image gcr.io/<project-id>/streamlit`

Note: this was originally built from a Macbook with an M1 chip, which cannot run/deploy docker containers. That is why we added the `--platform linux/amd64` command.


### Checklist
[View our checklist](CHECKLIST.md)

### WandB.ai report
[Check out our training report](https://wandb.ai/dtu_ml_ops/rotten_tomatoes/reports/Evaluation-of-model-training--VmlldzozMzczMDU5)


