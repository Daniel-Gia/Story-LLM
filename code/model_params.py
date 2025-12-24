# =====================
# Paths
# =====================
DATA_DIR = "data"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
CHECKPOINT_DIR = "checkpoints"

# =====================
# Tokenizer
# =====================
VOCAB_SIZE = 8000

# =====================
# Model architecture
# =====================
MAX_MODEL_CONTEXT = 256
EMBED_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
MLP_RATIO = 4.0
DROPOUT = 0.1

# =====================
# Training
# =====================
BATCH_SIZE = 30 #192
EPOCHS = 30

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# =====================
# Logging / saving
# =====================
LOG_INTERVAL = 100
SAVE_EVERY_EPOCH = True

# =====================
# Other
# =====================
SEED = 42
DEVICE = "cuda"