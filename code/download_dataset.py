from datasets import load_dataset
from pathlib import Path
import math

import model_params as Params

OUTPUT_DIR = Path(Params.DATA_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PERCENT = 10.0

SEED = Params.SEED

def main():
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    ds = ds.shuffle(seed=SEED)

    def take_percentage(data, percentage):
        if percentage >= 100:
            return data
        n = math.floor(len(data) * percentage / 100.0)
        return data.select(range(n))

    train_ds = take_percentage(ds, TRAIN_PERCENT)

    def clean(example):
        example["text"] = example["text"].strip().replace("\n", " ")
        return example

    train_ds = train_ds.map(clean, num_proc=4)

    print(f"Train samples: {len(train_ds)}")

    train_ds.to_parquet(OUTPUT_DIR / "train.parquet")

    print("Saved Parquet file.")

if __name__ == "__main__":
    main()
