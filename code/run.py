import torch
import torch.nn.functional as F

from tokenizers import Tokenizer

from model import LLM_Model
import model_params as Params

DEVICE = (
    "cuda"
    if torch.cuda.is_available() and Params.DEVICE == "cuda"
    else "cpu"
)

tokenizer = Tokenizer.from_file(Params.TOKENIZER_PATH)

BOS_ID = tokenizer.token_to_id("<bos>")
EOS_ID = tokenizer.token_to_id("<eos>")

checkpoint_path = f"{Params.CHECKPOINT_DIR}/model.pt"

checkpoint = torch.load(checkpoint_path, map_location="cpu")

config = checkpoint["model_config"]

model = LLM_Model(
    vocab_size=config["vocab_size"],
    max_context_len=config["max_context_len"],
    embed_dim=config["embed_dim"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    mlp_ratio=config["mlp_ratio"],
    dropout=0.0,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

def sample_next_token(logits, temperature=1.0, top_k=None):
    logits = logits / temperature

    if top_k is not None:
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    enc = tokenizer.encode(prompt)
    ids = [BOS_ID] + enc.ids

    input_ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        input_slice = input_ids[:, -Params.MAX_MODEL_CONTEXT :]

        logits = model(input_slice)
        next_logits = logits[0, -1]

        next_id = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
        )

        if next_id.item() == EOS_ID:
            break

        input_ids = torch.cat(
            [input_ids, next_id.view(1, 1)],
            dim=1,
        )

    return tokenizer.decode(input_ids[0].tolist())


def main():
    print("Type a prompt\n")

    while True:
        prompt = input("> ")

        text = generate(
            prompt,
            max_new_tokens=200,
            temperature=0.8,
            top_k=40,
        )

        print("\n" + text + "\n")


if __name__ == "__main__":
    main()
