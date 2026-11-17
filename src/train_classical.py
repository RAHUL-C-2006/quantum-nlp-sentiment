
import pandas as pd
import joblib
import json
import time

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")
dev_df = pd.read_csv(DATA_DIR / "dev.csv")

X_train = train_df["sentence"]
y_train = train_df["label"]

X_dev = dev_df["sentence"]
y_dev = dev_df["label"]

# Build the classical ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Train the model
print("Training classical TF-IDF + Logistic Regression model...")

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# Predict on development data
start_time = time.time()
predictions = model.predict(X_dev)
prediction_time = time.time() - start_time

# Calculate evaluation metrics
metrics = {
    "accuracy": accuracy_score(y_dev, predictions),
    "precision": precision_score(y_dev, predictions, zero_division=0),
    "recall": recall_score(y_dev, predictions, zero_division=0),
    "f1_score": f1_score(y_dev, predictions, zero_division=0),
    "training_time_seconds": training_time,
    "prediction_time_seconds": prediction_time
}

# Display results
print("\nCLASSICAL MODEL RESULTS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(y_dev, predictions, zero_division=0))

# Save trained model
model_path = MODEL_DIR / "classical_tfidf_logreg.joblib"
joblib.dump(model, model_path)

# Save evaluation metrics
metrics_path = OUTPUT_DIR / "classical_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved to:", model_path)
print("Metrics saved to:", metrics_path)
print("\nClassical baseline training completed!")
