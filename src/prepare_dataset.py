
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

RANDOM_SEED = 42

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Downloading SST-2 dataset...")

# Load the SST-2 dataset
dataset = load_dataset("nyu-mll/glue", "sst2")

# Convert dataset splits into DataFrames
train_df = pd.DataFrame(dataset["train"])[["sentence", "label"]]
validation_df = pd.DataFrame(dataset["validation"])[["sentence", "label"]]

# Select a small balanced training sample
train_pool, _ = train_test_split(
    train_df,
    train_size=160,
    stratify=train_df["label"],
    random_state=RANDOM_SEED
)

# Split into training and development sets
train_data, dev_data = train_test_split(
    train_pool,
    test_size=0.25,
    stratify=train_pool["label"],
    random_state=RANDOM_SEED
)

# Select a separate held-out evaluation sample
test_data, _ = train_test_split(
    validation_df,
    train_size=200,
    stratify=validation_df["label"],
    random_state=RANDOM_SEED
)

# Save CSV files
train_data.to_csv(DATA_DIR / "train.csv", index=False)
dev_data.to_csv(DATA_DIR / "dev.csv", index=False)
test_data.to_csv(DATA_DIR / "test.csv", index=False)

print("\nDataset preparation completed!")
print("Training samples:", len(train_data))
print("Development samples:", len(dev_data))
print("Held-out test samples:", len(test_data))
print("\nFiles saved in:", DATA_DIR)
