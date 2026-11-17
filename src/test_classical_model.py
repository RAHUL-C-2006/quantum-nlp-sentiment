
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "classical_tfidf_logreg.joblib"

model = joblib.load(MODEL_PATH)

sentences = [
    "This movie is absolutely wonderful.",
    "This movie is boring and terrible."
]

predictions = model.predict(sentences)

print("Model loaded successfully!")

for sentence, prediction in zip(sentences, predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"\nSentence: {sentence}")
    print(f"Prediction: {sentiment}")
