from pprint import pprint
from pathlib import Path
import json

from scripts.evaluate import (
    generate_text_with_temperature_topk,
    text_to_token_ids,
    token_ids_to_text,
)
from configs.model_configs import get_gpt_configs, GPT_configs
from src.data.pretrain_dataset.dataset import create_dataloader, ensure_bin_dataset
from src.training.train import train_model
from src.models.gpt_model import GPT_model
from src.utils.save_model_hf import save_model_hf, ensure_huggingface_login
from src.utils.load_model import load_model_if_exists

import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchinfo import summary


import sys


def main():
    # --- Parse command-line args (simple, no argparse overhead) ---
    args = sys.argv[1:]  # Skip script name

    print("Welcome to the gpt2 model pipeline! loading stuff please wait ...")
    cfg = get_gpt_configs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT_model(cfg).to(device=device)
    print(f"Current device is {device}.")

    if cfg.repo_id != "INVALID":
        ensure_huggingface_login(cfg=cfg)

    if "-t" in args:
        print("start train process ...")
        torch.manual_seed(42)
        train(model, cfg)
        print("training process finished, Now you can use model.")
        return 0

    if "-g" in args:
        # Check if a prompt was passed with -p, else ask interactively
        if "-p" in args:
            prompt = args[args.index("-p") + 1]
        else:
            prompt = input("Enter your prompt: ")

        if load_model_if_exists(cfg, model, device):
            generate(model, cfg, prompt, 42)
        else:
            print("Model not found, try to train it first.")
        return 0

    if "-i" in args:
        model_info(model, cfg)
        return 0

    # --- Original Interactive Loop ---
    while True:
        print(
            "Chose what to do: (q = quit, t = train, g = generate, i = model and configs info)"
        )
        inp = input().lower()

        if inp == "q":
            print("Bye!")
            return 0

        elif inp == "t":
            print("start train process ...")
            torch.manual_seed(42)
            train(model, cfg)
            print("training process finished, Now you can use model.")
            return 0

        elif inp == "g":
            if load_model_if_exists(cfg, model, device):
                prompt = input("Enter your prompt: ")
                generate(model, cfg, prompt, 42)
                return 0
            print("Model not found, try to train it first.")

        elif inp == "i":
            model_info(model, cfg)

        else:
            print("unknown input, try again.")


def model_info(model, cfg):
    print("Model's configuration:")
    pprint(cfg.__dict__)
    print("=================================================================")
    print("Model's real tranable parameters count:")
    print(sum(p.numel() for p in model.parameters()))
    if model.tok_emb.weight is model.out_head.weight:
        print(
            "[INFO] You correctly tied token embedding matrix (wte) and the output projection (lm_head). "
            "So your model will be same size as the original gpt2 and ignore what torchinfo say about `Trainable params`."
        )
    else:
        print(
            "[WARNING] You didn't tie token embedding matrix (wte) and the output projection (lm_head). "
            "So your model will be bigger than original gpt2."
        )
    print("Model's architecture (torchinfo summary):")
    summary(model)


def generate(model, cfg: GPT_configs, prompt: str, seed=None):
    model.eval()
    tokenizer = tiktoken.get_encoding(cfg.ticktoken_tokenizer)
    generated_ids = generate_text_with_temperature_topk(
        model=model,
        idx=text_to_token_ids(text=prompt, tokenizer=tokenizer),
        context_size=cfg.context_length,
        max_new_tokens=25,
        top_k=90,
        temperature=0.7,
    )

    print(token_ids_to_text(generated_ids, tokenizer))


def train(model: GPT_model, cfg: GPT_configs):
    # ==== TOKENIZE ONCE (CACHED) AND BUILD TRAIN/VAL LOADERS FROM ONE BIN FILE ====
    bin_path = ensure_bin_dataset(cfg)
    train_ratio = 0.80

    tokenizer = tiktoken.get_encoding(cfg.ticktoken_tokenizer)
    train_loader = create_dataloader(
        bin_path,
        batch_size=cfg.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=4,
        start_frac=0.0,
        end_frac=train_ratio,
    )
    val_loader = create_dataloader(
        bin_path,
        batch_size=cfg.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        start_frac=train_ratio,
        end_frac=1.0,
    )

    # ==== SETUP EPOCHS DEVICE AND OPTIMIZER ====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)
    optim = torch.optim.AdamW(params=model.parameters(), lr=4e-4, weight_decay=0.1)

    # ==== TRAIN MODEL ====
    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        cfg.epochs,
        optim,
        device,
        eval_freq=5,
        eval_iter=cfg.batch_size,
        lr_schedule_step=cfg.lr_schedule_step,
        start_context="Hello I am",
        tokenizer=tokenizer,
        create_checkpoints=cfg.create_checkpoints,
        use_checkpoints=cfg.use_checkpoints,
        checkpoint_freq=cfg.checkpoint_freq,
        checkpoint_path=cfg.checkpoints_path,
    )

    # ==== SAVE MODEL WITH CONFIGS ====
    folder_name = Path(cfg.save_model_path)
    folder_name.mkdir(exist_ok=True)
    torch.save(model.state_dict(), (folder_name / "pytorch_model.bin"))

    with open(folder_name / "configs.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=4)

    save_model_hf(cfg)

    # ==== PLOT MODEL LOSSES ====
    plot_losses(cfg.epochs, tokens_seen, train_losses, val_losses)


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()

    fig.savefig("train-val-loss.png")


main()
