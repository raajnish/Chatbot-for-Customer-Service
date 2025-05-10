import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset (assuming CSV with 'text' and 'label' columns)
df = pd.read_csv("spam_dataset.csv")

# Preprocessing
X = df["text"]
y = df["label"].map({"ham": 0, "spam": 1})  # Convert labels

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

# Train SVM Model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# Display results
print(f"Naive Bayes Accuracy: {nb_acc:.2f}")
print(f"SVM Accuracy: {svm_acc:.2f}")
