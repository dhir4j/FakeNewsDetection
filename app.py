import re
import pickle
import numpy as np
import pandas as pd
import torch
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from urllib.parse import urlparse
from newsplease import NewsPlease
import news_channels

# Load saved CNN model and tokenizer (adjust file paths accordingly)
model = load_model('/home/dhir4j/code/flask/FND/Flask FND/data/fake_news_detector_CNN.keras')
with open('/home/dhir4j/code/flask/FND/Flask FND/data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    parsed_url = urlparse(data)
    print(data)
    if parsed_url.scheme and parsed_url.netloc:
        print('IN url')
        try: 
            domain = parsed_url.netloc
            print(domain)
            channelVerify = news_channels.verify_url(domain)
            if channelVerify is not None:
                result = "real" if channelVerify else "fake"
                return jsonify({"result": result})
        except Exception as e:
            print(e)
            return jsonify({"result": "Unable to Fetch Text"})
        else:
            print('ML')
            article = NewsPlease.from_url(data)
            if article:
                try: 
                    text = article.maintext
                    # Preprocess text using the loaded tokenizer
                    processed_text = tokenizer.texts_to_sequences([text])
                    padded_sequence = pad_sequences(processed_text, maxlen=100)  # Adjust maxlen if needed

                    # Make prediction using the CNN model
                    prediction = model.predict(padded_sequence)[0][0]  # Extract scalar prediction
                    print(text)
                    result = "real" if prediction > 0.5 else "fake"
                    return jsonify({"result": result})
                except Exception as e:
                    print(e)
                    return jsonify({"result": "Unable to Fetch Text"})
        
    # elif 'text' in data:
    else:
        print('IN Text')
        # Process text input
        text = data
        if len(text) >= 300:
            # Preprocess text using the loaded tokenizer
            processed_text = tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(processed_text, maxlen=100)  # Adjust maxlen if needed

            # Make prediction using the CNN model
            prediction = model.predict(padded_sequence)[0][0]  # Extract scalar prediction
            result = "real" if prediction > 0.5 else "fake"
            return jsonify({"result": result})
        else:
            print("lowtextsize")
            return jsonify({"result": "lowtextsize"})
        
    # else:
    #     print("IN invalid")
    #     result = "Invalid input format"
    
    # return jsonify({"result": result})


if __name__ == '__main__':
    app.run()
