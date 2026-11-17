
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

for filename in ["train.csv", "dev.csv", "test.csv"]:
    df = pd.read_csv(DATA_DIR / filename)

    print(f"\n--- {filename} ---")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())
    print("Sample sentence:", df.iloc[0]["sentence"])

    assert list(df.columns) == ["sentence", "label"]
    assert df["label"].isin([0, 1]).all()
    assert df["sentence"].notna().all()

print("\nAll dataset files verified successfully!")
