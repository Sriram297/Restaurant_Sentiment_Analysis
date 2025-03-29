# Restaurant Sentiment Analysis

## Project Overview

This project performs sentiment analysis on restaurant reviews using Natural Language Processing (NLP) techniques. The goal is to classify customer reviews into positive, neutral, or negative sentiments based on their textual content. The analysis helps restaurants understand customer feedback better and make data-driven improvements.

## Features

- Preprocessing of textual data (stopword removal, lemmatization).
- Vectorization using CountVectorizer.
- Training machine learning models for sentiment classification.
- Evaluation of model performance.

## Installation

To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the following manually:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

## Dataset

The dataset consists of restaurant reviews labeled with sentiment categories. It includes:

- **Text Reviews**: Raw customer feedback.
- **Sentiment Labels**: Categorized as Positive or  Negative

## Usage

Run the Jupyter Notebook to perform sentiment analysis:

```bash
jupyter notebook Restaurant_Sentiment_Analysis.ipynb
```

### Steps:

1. Load the dataset.
2. Preprocess the text data.
3. Convert text to numerical representations using CountVectorizer.
4. Train multiple machine learning models.
5. Evaluate model performance using accuracy and classification reports.

## Machine Learning Models Used

- Logistic Regression
- Naive Bayes

## Results

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

Results are visualized using confusion matrices and classification reports.

## Contributing

Feel free to contribute by improving the preprocessing techniques, testing deep learning models, or optimizing model performance.

---

###

