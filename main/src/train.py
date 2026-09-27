import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score, recall_score, f1_score
)

# Allow importing preprocessing.py
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from preprocessing import prepare_dataframe


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "tickets.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

df = prepare_dataframe(df)

print(
    "Empty cleaned descriptions:",
    (df["Cleaned Description"] == "").sum()
)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df["Cleaned Description"]

y = df["Ticket Category"]


# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print(
    "Training TF-IDF shape:",
    X_train_tfidf.shape
)

print(
    "Testing TF-IDF shape:",
    X_test_tfidf.shape
)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    C=10000,
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(
    X_test_tfidf
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

y_train_pred = model.predict(X_train_tfidf)
y_test_pred = model.predict(X_test_tfidf)

print("\nTraining Performance")
print("--------------------")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred, average="macro", zero_division=0))
print("Recall:", recall_score(y_train, y_train_pred, average="macro", zero_division=0))
print("F1 Score:", f1_score(y_train, y_train_pred, average="macro", zero_division=0))

print("\nTesting Performance")
print("-------------------")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, average="macro", zero_division=0))
print("Recall:", recall_score(y_test, y_test_pred, average="macro", zero_division=0))
print("F1 Score:", f1_score(y_test, y_test_pred, average="macro", zero_division=0))

print("\n" + "=" * 60)
print("MODEL RESULT")
print("=" * 60)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy percentage: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

model_path = os.path.join(
    MODEL_DIR,
    "ticket_classifier.pkl"
)

joblib.dump(
    tfidf,
    vectorizer_path
)

joblib.dump(
    model,
    model_path
)


# --------------------------------------------------
# Save test predictions
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

prediction_path = os.path.join(
    MODEL_DIR,
    "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    "Vectorizer:",
    vectorizer_path
)

print(
    "Model:",
    model_path
)

print(
    "Predictions:",
    prediction_path
)