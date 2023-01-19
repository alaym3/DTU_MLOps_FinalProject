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


# How does our app work?
Our sentiment classification application uses [Streamlit](https://streamlit.io/) and is deployed on Google Cloud via [this link](https://streamlit-pqpw5ljsba-ew.a.run.app). The Streamlit app is containerized and deployed via Cloud Run. Our custom trained huggingface transformers model is downloaded from our Google Cloud Bucket and users are able to type in any text input they want, and view the probability of the phrase being positive and negative. It uses the streamlit.dockerfile in the docker folder.

### Do you want to run the image locally?
- run `docker build -f docker/streamlit.dockerfile . -t streamlit:latest` then `docker run -p 8080:8080 streamlit:latest`

### Do you want to deploy this app via Streamlit on Cloud Run??
- clone our repo
- create a project in Google Cloud
- make sure you have money in your billing account since costs are incurred by the container
- add a creds folder and a creds.json inside of it, pertaining to your google cloud authentication credentials created from your project.
- run the streamlit.dockerfile found in the docker folder, tag it, push the image to your project, then run a command to auto deploy via Cloud Run. Example below:
   - `docker build --platform linux/amd64 -f docker/streamlit.dockerfile . -t streamlit:latest`
   - `docker tag streamlit:latest gcr.io/<project-id>/streamlit`
   - `docker push gcr.io/<project-id>/streamlit`
   - `gcloud run deploy streamlit --image gcr.io/<project-id>/streamlit`



### Checklist
[View our checklist](CHECKLIST.md)




