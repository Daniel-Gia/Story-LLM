from huggingface_hub import hf_hub_download
from pathlib import Path
import model_params as Params
import shutil

REPO_ID = "DanielG9/Story-LLM"

CHECKPOINT_DIR = Path(Params.CHECKPOINT_DIR)
TOKENIZER_DIR = Path(Params.TOKENIZER_PATH).parent
TOKENIZER_FILE = Path(Params.TOKENIZER_PATH).name

CHECKPOINT_DIR.mkdir(exist_ok=True)
TOKENIZER_DIR.mkdir(exist_ok=True)

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="model.pt",
)

tokenizer_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=TOKENIZER_FILE,
)

shutil.move(model_path, CHECKPOINT_DIR / "model.pt")
shutil.move(tokenizer_path, TOKENIZER_DIR / TOKENIZER_FILE)

print("✅ Model and tokenizer downloaded successfully")
