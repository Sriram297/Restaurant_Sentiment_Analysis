from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('logistic_regression.pkl', 'rb') as file:
    model_lr = pickle.load(file)
with open('naive_bayes.pkl', 'rb') as file:
    model_nb = pickle.load(file)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    wl = WordNetLemmatizer()
    words = text.lower().split()
    words = [wl.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        processed_review = preprocess_text(review)
        review_vector = vectorizer.transform([processed_review])
        
        # Get predictions
        pred_lr = model_lr.predict(review_vector)[0]
        pred_nb = model_nb.predict(review_vector.toarray())[0]
        
        # Majority voting for final sentiment
        final_prediction = "Positive" if (pred_lr + pred_nb) >= 1 else "Negative"
        
        return render_template('index.html', review=review, sentiment=final_prediction)

if __name__ == '__main__':
    app.run(debug=True)
