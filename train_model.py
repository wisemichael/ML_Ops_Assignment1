import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
df = pd.read_csv("IMDB_Dataset/IMDB Dataset.csv")

# Features and labels
x = df["review"]
y = df["sentiment"]

# Create the pipeline (TF-IDF vectorizer + Naive Bayes)
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model on full data (not split)
model.fit(x, y)

# Save model to disk
joblib.dump(model, "sentiment_model.pkl")
print("✅ Model trained and saved as sentiment_model.pkl")

