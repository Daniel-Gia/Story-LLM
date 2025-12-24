import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from tokenizers import Tokenizer

from model import LLM_Model

import model_params as Params

import os

# Paths
DATA_DIR = Params.DATA_DIR
TOKENIZER_PATH = Params.TOKENIZER_PATH

# Training
BATCH_SIZE = Params.BATCH_SIZE
MAX_MODEL_CONTEXT = Params.MAX_MODEL_CONTEXT
EPOCHS = Params.EPOCHS
LR = Params.LEARNING_RATE
WEIGHT_DECAY = Params.WEIGHT_DECAY
GRAD_CLIP = Params.GRAD_CLIP

DEVICE = Params.DEVICE if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

PAD_ID = tokenizer.token_to_id("<pad>")
BOS_ID = tokenizer.token_to_id("<bos>")
EOS_ID = tokenizer.token_to_id("<eos>")

os.makedirs(Params.CHECKPOINT_DIR, exist_ok=True)

class TinyStoriesDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_model_context):
        self.ds = load_dataset("parquet", data_files=parquet_path)["train"]
        self.tokenizer = tokenizer
        self.max_model_context = max_model_context

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["text"]

        enc = self.tokenizer.encode(text)
        ids = enc.ids

        ids = [BOS_ID] + ids[: self.max_model_context - 2] + [EOS_ID]

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        return input_ids, target_ids

# To make each batch have tensors of the same length
def collate_fn(batch):
    inputs, targets = zip(*batch)

    max_len = max(x.size(0) for x in inputs)

    padded_inputs = torch.full((len(inputs), max_len), PAD_ID)
    padded_targets = torch.full((len(inputs), max_len), PAD_ID)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, : inp.size(0)] = inp
        padded_targets[i, : tgt.size(0)] = tgt

    return padded_inputs, padded_targets

train_dataset = TinyStoriesDataset(
    f"{DATA_DIR}/train.parquet",
    tokenizer,
    MAX_MODEL_CONTEXT,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
)

model = LLM_Model(
    vocab_size=tokenizer.get_vocab_size(),
    max_context_len=MAX_MODEL_CONTEXT,
    embed_dim=Params.EMBED_DIM,
    num_layers=Params.NUM_LAYERS,
    num_heads=Params.NUM_HEADS,
    dropout=Params.DROPOUT,
).to(DEVICE)


start_checkpoint_path = os.path.join(Params.CHECKPOINT_DIR, "model-start.pt")

if os.path.exists(start_checkpoint_path):
    print(f"Loading checkpoint from {start_checkpoint_path} to resume training...")
    checkpoint = torch.load(start_checkpoint_path, map_location=DEVICE)
    
    checkpoint_config = checkpoint.get("model_config", None)
    current_config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "max_context_len": Params.MAX_MODEL_CONTEXT,
        "embed_dim": Params.EMBED_DIM,
        "num_layers": Params.NUM_LAYERS,
        "num_heads": Params.NUM_HEADS,
        "mlp_ratio": Params.MLP_RATIO,
        "dropout": Params.DROPOUT,
    }
    
    if checkpoint_config != current_config:
        raise ValueError(
            f"Checkpoint config does not match current model config!\n"
            f"Checkpoint config: {checkpoint_config}\n"
            f"Current config: {current_config}"
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded.")
else:
    print("No checkpoint found. Starting training from scratch. (To start from checkpoint place 'model-start.pt' in the checkpoint directory)")

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

loss_function = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def train():
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for step, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            logits = model(input_ids)

            loss = loss_function(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP
            )

            optimizer.step()

            total_loss += loss.item()

            if step % Params.LOG_INTERVAL == 0:
                print(
                    f"Epoch {epoch+1} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )
        avg_loss = total_loss / len(train_loader)
        ppl = math.exp(avg_loss)
        print(
            f"\nEpoch {epoch+1} completed | "
            f"Avg Loss {avg_loss:.4f} | PPL {ppl:.2f}\n"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": tokenizer.get_vocab_size(),
                "max_context_len": Params.MAX_MODEL_CONTEXT,
                "embed_dim": Params.EMBED_DIM,
                "num_layers": Params.NUM_LAYERS,
                "num_heads": Params.NUM_HEADS,
                "mlp_ratio": Params.MLP_RATIO,
                "dropout": Params.DROPOUT,
            }
        }
        torch.save(checkpoint, f"{Params.CHECKPOINT_DIR}/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    print(f"Training on {DEVICE}")
    train()
