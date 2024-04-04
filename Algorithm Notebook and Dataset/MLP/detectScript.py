import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.tokenize import word_tokenize

# Define tokenize_text function
def tokenize_text(text):
    return word_tokenize(text)

# Load the saved model
model = joblib.load('/home/dhir4j/code/flask/FND/testing/MLP/fake_news_detector_MLP.pkl')

# Load tokenizer function using pickle
with open('/home/dhir4j/code/flask/FND/testing/MLP/tokenizer.pkl', 'rb') as f:
    tokenizer_function = pickle.load(f)

# Load TF-IDF vectorizer using pickle
with open('/home/dhir4j/code/flask/FND/testing/MLP/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define preprocess_text function using custom tokenizer and TF-IDF vectorizer
def preprocess_text(text):
    # Tokenize text using the loaded tokenizer function
    tokens = tokenizer_function(text)
    
    # Join tokens into a string
    processed_text = ' '.join(tokens)
    
    # Transform the processed text using the loaded TF-IDF vectorizer
    processed_vector = vectorizer.transform([processed_text])
    
    return processed_vector

# Get user input
text = input("Enter some news text: ")

# Preprocess the text
processed_text = preprocess_text(text)

# Make prediction
prediction = model.predict(processed_text)[0]
result = "Real News" if prediction > 0.5 else "Fake News"

# Print the result
print(f"The news article is most likely: {result}")
