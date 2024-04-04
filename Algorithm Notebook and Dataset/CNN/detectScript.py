from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import pickle

# Load the saved model and tokenizer (with adjusted file paths)
model = load_model('/home/dhir4j/code/flask/FND/Final/Algorithm/CNN/fake_news_detector_CNN.keras')
with open('/home/dhir4j/code/flask/FND/Final/Algorithm/CNN/tokenizer.pkl', 'rb') as f:  # Assuming tokenizer.pkl is in the same directory
    tokenizer = pickle.load(f)

# Function to preprocess text (no changes here)
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)  # Adjust maxlen if needed
    return padded_sequence

# Get user input (no changes here)
text = input("Enter some news text: ")

# Preprocess the text (no changes here)
processed_text = preprocess_text(text)

# Make prediction (no changes here)
prediction = model.predict(processed_text)[0]
result = "Real News" if prediction > 0.5 else "Fake News"

# Print the result (no changes here)
print(f"The news article is most likely: {result}")
