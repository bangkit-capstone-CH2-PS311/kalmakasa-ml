from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

#preprocessing lib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
import re
import numpy as np
import json
import random
from bs4 import BeautifulSoup

# Download necessary NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
nltk.download('stopwords')
print("path",nltk.data.path)

from nltk.corpus import stopwords


kamus_singkatan = pd.read_csv('./kamus_singkatan.csv', header=None, names=['slang', 'terjemahan'], delimiter=';')

# Create a mapping dictionary for slang words
mapping_dict = dict(zip(kamus_singkatan['slang'], kamus_singkatan['terjemahan']))

# Define cleaning and preprocessing functions
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('indonesian'))

chatbot_model = load_model('model_chatbot.h5', compile=False)
twitter_model = load_model('model_twitter.h5', compile=False)
reddit_model = load_model('model_reddit.h5', compile=False)

CHATBOT_MAX_SEQUENCE_LENGTH = 120
TWITTER_MAX_SEQUENCE_LENGTH = 100
REDDIT_MAX_SEQUENCE_LENGTH = 1000
VOCAB_SIZE = 1000
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"

# Loading tokenizer
with open('tokenizer_chatbot.json') as f:
    chatbot_tokenizer_config = json.load(f)
    chatbot_tokenizer = tokenizer_from_json(chatbot_tokenizer_config)

with open('tokenizer_twitter.json') as f:
    twitter_tokenizer_config = json.load(f)
    twitter_tokenizer = tokenizer_from_json(twitter_tokenizer_config)

with open('tokenizer_reddit.json') as f:
    reddit_tokenizer_config = json.load(f)
    reddit_tokenizer = tokenizer_from_json(reddit_tokenizer_config)

with open('label_chatbot.json', 'r') as f:
    label_mapping = json.load(f)
# load response file
with open('label_chatbot_res.json', 'r') as f:
    response_mapping = json.load(f)

def clean_text(text):
    if isinstance(text, str):
        text = BeautifulSoup(text, "html.parser").get_text()  # HTML decoding
        text = text.lower()  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwords from text
        return text
    else:
        return ""
    
def ganti_slang(teks):
    kata_kunci = teks.split()
    teks_baru = ' '.join([mapping_dict.get(kata, kata) for kata in kata_kunci])
    return teks_baru

# predict chatbot model
def predict_chatbot(text_input):
    teks_p = []

    prediction_input = text_input.lower()

    # Remove non-alphanumeric characters, URLs, and subreddit mentions
    prediction_input = re.sub(r'/r/|[^\w\s]|https?://\S+|www\.\S+|\#\w+', '', prediction_input)
    # words = word_tokenize(prediction_input)

    teks_p.append(prediction_input)

    prediction_input = chatbot_tokenizer.texts_to_sequences(teks_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], CHATBOT_MAX_SEQUENCE_LENGTH)
    # predict chatbot model
    result = chatbot_model.predict(prediction_input)

    # get label with highest probability
    predicted_labels = [label for label, index in label_mapping.items()]
    predicted_label_index = np.argmax(result)
    predicted_label = predicted_labels[predicted_label_index]

    # get random response from predicted label
    predicted_response = random.choice(response_mapping[predicted_label])
    
    return predicted_response


def predict_twitter(text_input):
    # Clean and replace slang
    cleaned_text = clean_text(text_input)
    final_text = ganti_slang(cleaned_text)

    print('Final text=',final_text)

    input_sequences = twitter_tokenizer.texts_to_sequences([final_text])

    print("Input Sequences:", input_sequences)

    input_padded = pad_sequences(
        input_sequences,
        maxlen=TWITTER_MAX_SEQUENCE_LENGTH,
        dtype="int32",
        padding=PADDING_TYPE,
        truncating=TRUNC_TYPE
    )

    print("Input Padded:", input_padded)
    # predict chatbot model
    result = twitter_model.predict(input_padded)
    print("Result:", result)

    predict_label=['anger', 'fear', 'happy', 'love', 'sadness']
    return predict_label[np.argmax(result)]

def predict_reddit(text_input):
     # lowercase text
    text_input = text_input.lower()

    cleaned_text = clean_text(text_input)

    # tokenize text input
    input_sequences = reddit_tokenizer.texts_to_sequences([cleaned_text])
    input_padded = pad_sequences(input_sequences, REDDIT_MAX_SEQUENCE_LENGTH)
    # predict chatbot model
    result = reddit_model.predict(input_padded)
    print("Result:", result)

    print("Input Sequences:", input_sequences)

    # load label file
    with open('label_reddit.json', 'r') as f:
        label_mapping = json.load(f)
    
    # get label with highest probability
    predicted_labels = [label for label, index in label_mapping.items()]
    predicted_label_index = np.argmax(result)
    predicted_label = predicted_labels[predicted_label_index]
    
    return predicted_label


