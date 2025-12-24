# Story-LLM (TinyStories Transformer)

A small GPT-style Transformer language model trained to generate short stories.  
It uses a ByteLevel BPE tokenizer and a causal self-attention model implemented in PyTorch.

- **Pretrained weights + tokenizer (trained by me - 29M parameters):** https://huggingface.co/DanielG9/Story-LLM  
- **Dataset used:** https://huggingface.co/datasets/roneneldan/TinyStories

---

## 1) What this project is

This repo contains everything needed to:

- download and preprocess a TinyStories subset,
- train a small Transformer language model,
- download my pretrained checkpoint + tokenizer from Hugging Face.

---

## 2) How to run the model

### Install dependencies
```sh
pip install -r requirements.txt
```

### Option A: Download my pretrained model + tokenizer (recommended)
```sh
python code/download_model.py
```

This downloads:
- `checkpoints/model.pt`
- `tokenizer/tokenizer.json`

### Run interactive generation
```sh
python code/run.py
```

You’ll get a prompt:
- type a prompt and press Enter to generate text.

---

## 3) How to train (or continue training)

### Step 1: Download the dataset (TinyStories)
```sh
python code/download_dataset.py
```

This saves:
- `data/train.parquet`

### Step 2: Train a tokenizer (required if you are not using the pretrained tokenizer)
```sh
python code/tokenizer.py
```

This saves:
- `tokenizer/tokenizer.json`

### Step 3: Train the model
```sh
python code/train.py
```

Checkpoints are saved to:
- `checkpoints/model_epoch_*.pt`

### Continue training from a checkpoint
Training resumes automatically **only** if you place a checkpoint at:

- `checkpoints/model-start.pt`

Then run:
```sh
python code/train.py
```

> Note: training will raise an error if the checkpoint’s saved `model_config` does not match your current settings in `code/model_params.py`.

---

## 4) Project structure

```txt
.
├─ code/
│  ├─ download_dataset.py    # Downloads TinyStories and writes data/train.parquet
│  ├─ download_model.py      # Downloads pretrained weights + tokenizer from HF
│  ├─ model.py               # Transformer model
│  ├─ model_params.py        # All hyperparameters and paths
│  ├─ run.py                 # Run the model
│  ├─ tokenizer.py           # Trains and saves ByteLevel BPE tokenizer
│  └─ train.py               # Training loop + checkpoint saving
├─ checkpoints/              # Saved model checkpoints (ignored by git)
├─ data/                     # Dataset parquet files (ignored by git)
├─ tokenizer/                # tokenizer.json (ignored by git)
└─ requirements.txt
```

---

## Configuration

All key settings live in:
- `code/model_params.py`

Including:
- tokenizer vocab size (`VOCAB_SIZE`)
- model size (`EMBED_DIM`, `NUM_LAYERS`, `NUM_HEADS`, `MAX_MODEL_CONTEXT`)
- training params (`BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, etc.)