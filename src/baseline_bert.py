import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

val_df = pd.read_csv("data_val.csv")

name = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
model.eval()

preds = []
with torch.no_grad():
    for s in val_df["sentence"]:
        inputs = tokenizer(s, return_tensors="pt", truncation=True)
        preds.append(model(**inputs).logits.argmax(dim=1).item())

acc = (pd.Series(preds) == val_df["label"]).mean()
print("BERT val acc:", round(acc, 3))