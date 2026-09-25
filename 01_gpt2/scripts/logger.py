import os
from datetime import datetime
import json

import torch

from scripts.evaluation import calc_perplexity


def save_print_log(
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    device="cuda",
):
    """
    ## Logger helepr

    This function will print a log and also write that log into a file.

    ---
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not torch.cuda.is_available():
        allocated = 0.0
        reserved = 0.0
        total = 0.0
    else:
        # Getting some data about GPU so I may decide to use another approach
        # and all these data is saving as GB.
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    log_data = {
        "timestamp": timestamp,
        "epoch": epoch + 1,
        "step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_ppl": calc_perplexity(train_loss),
        "val_ppl": calc_perplexity(val_loss),
        "lr": lr,
        "GPU_allocated_VRAM": allocated,
        "GPU_reserved_VRAM": reserved,
        "GPU_total_VRAM": total,
    }
    print(
        f"[{timestamp}] "
        f"Epoch {epoch+1:03d} (Step {global_step:08d}): "
        f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
        f"PPL {calc_perplexity(train_loss):.2f}/{calc_perplexity(val_loss):.2f} | "
        f"LR {lr:.6e} | ",
        flush=True,
    )

    with open("./training-process/logs.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")
