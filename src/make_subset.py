from datasets import load_dataset
import pandas as pd

dataset = load_dataset("nyu-mll/glue", "sst2")

def make_subset(split, n_per_class, max_words, seed=42):
    df = split.to_pandas()
    df["sentence"] = df["sentence"].str.strip()
    df["n_words"] = df["sentence"].str.split().str.len()
    df = df[(df["n_words"] >= 3) & (df["n_words"] <= max_words)]
    parts = []
    for lab in (0, 1):
        sub = df[df["label"] == lab]
        parts.append(sub.sample(n=min(n_per_class, len(sub)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

train_df = make_subset(dataset["train"], n_per_class=150, max_words=6)
val_df = make_subset(dataset["validation"], n_per_class=50, max_words=10)

train_df.to_csv("data_train.csv", index=False)
val_df.to_csv("data_val.csv", index=False)

print("Train subset:", len(train_df), "| label counts:", train_df["label"].value_counts().to_dict())
print("Val subset:  ", len(val_df), "| label counts:", val_df["label"].value_counts().to_dict())
print()
print(train_df[["label", "sentence"]].head(8).to_string())