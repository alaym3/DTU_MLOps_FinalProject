import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 1st step always: make layout wider
st.set_page_config(layout="wide", page_title="text-classification")
st.title("Sentiment classification app")
st.markdown(
    "### This app allows you to type in a phrase and see how much the phrase is classified to be positive or negative. \
    Try it out on your own!"
)


# @st.cache
def load_model():
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./models/")
    # model = AutoModelForSequenceClassification.from_pretrained("gs://dtu_mlops_final_model")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer


model, tokenizer = load_model()

text = st.text_area(
    "Write some text to perform sentiment analysis!",
    "This is a very happy, positive-sounding text sample! Type your own.",
)

inputs = tokenizer(
    text, return_tensors="pt"
)  # , padding = True, truncation = True, return_tensors='pt').to('cuda')
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()

# store the positive and negative predictions
neg_prediction = predictions[0][0] * 100
pos_prediction = predictions[0][1] * 100

# print results
st.markdown(f"**Probability of the phrase being negative: {neg_prediction}%**")
st.markdown(f"**Probability of the phrase being positive: {pos_prediction}%**")
st.markdown("#### About the model:")
st.markdown(
    "- We started with the pre-trained transformer [bert-base-uncased](https://huggingface.co/bert-base-uncased) since it is the top used model \
    for performing Natural Language Processing tasks on English text, including classification and \
        question-answering. BERT consists of a bidirectional transformer that looks back and forward when \
            analysing the tokens to learn the context of words."
)
st.markdown("#### About the training:")
st.markdown(
    "- The model has been trained on a [Rotten Tomatoes review dataset](https://huggingface.co/datasets/rotten_tomatoes). The dataset includes two columns: the text from Rotten Tomatoes \
    reviews for movies, along with a column indicating if the review is positive or negative. [Rotten Tomatoes](https://www.rottentomatoes.com/) is a platform \
        where movie reviews are submitted by expert audiences and regular people."
)
