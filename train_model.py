import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Map labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})

# Features and labels
X = df['message']
y = df['label']

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(clf, "NB_spam_model.pkl")
joblib.dump(cv, "count_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
