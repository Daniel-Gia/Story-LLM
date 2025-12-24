from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset
from pathlib import Path

import model_params as Params

DATA_DIR = Path(Params.DATA_DIR)
TOKENIZER_DIR = Path("tokenizer")
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = Params.VOCAB_SIZE
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def batch_iterator(batch_size=1000):
    ds = load_dataset("parquet", data_files={
        "train": str(DATA_DIR / "train.parquet"),
    })

    for split in ["train"]:
        for i in range(0, len(ds[split]), batch_size):
            yield ds[split][i:i+batch_size]["text"]

def main():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer)

    tokenizer.save(str(TOKENIZER_DIR / "tokenizer.json"))
    print("Tokenizer saved.")

    #Verify tokenizer
    print("\n=== Tokenizer Test ===")

    ds = load_dataset("parquet", data_files={
        "train": str(DATA_DIR / "train.parquet"),
    })

    example_text = ds["train"][0]["text"]

    encoded = tokenizer.encode(example_text)
    decoded = tokenizer.decode(encoded.ids)

    print("\n[Original]")
    print(example_text)

    print("\n[Encoded IDs]")
    print(encoded.ids)
    print(len(encoded.ids), "tokens")

    print("\n[Decoded]")
    print(decoded)

    print("\n[Match]")
    print("✅ OK" if example_text.lstrip() == decoded.lstrip() else "❌ WRONG")

if __name__ == "__main__":
    main()
