import os
import torch

from configs.model_configs import GPT_configs


def load_model_if_exists(cfg: GPT_configs, model: torch.nn.Module, device="cpu"):
    """
    ## Load model weights from a local file, falling back to the latest checkpoint.

    This function first checks for the final saved PyTorch model weights file
    (`pytorch_model.bin`) inside `cfg.save_model_path`. If it exists, it's loaded
    and the function returns `True`.

    If no final weights file exists, it falls back to looking for a training
    checkpoint (`latest.pt`) inside `cfg.checkpoints_path`. If found, only the
    `model_state_dict` portion of the checkpoint is loaded into `model` (the
    optimizer state, epoch, step, etc. are ignored here since this helper has
    no optimizer to restore them into — see `src.utils.ckeckpoints.load_checkpoint`
    if you need a full training-resume load).

    If neither file exists, prints a message and returns `False`.

    Args:
        cfg (GPT_configs): Configuration object containing:
            - save_model_path (str): Directory where the final model weights file is stored.
            - checkpoints_path (str): Directory where training checkpoints are stored.
        model (torch.nn.Module): The model instance into which the weights will be loaded.
        device (str, optional): Device to map the loaded state dict to (e.g., "cpu", "cuda"). Default is "cpu".

    Returns:
        bool: `True` if weights were successfully loaded (from either source), `False` otherwise.
    """
    model_weights_path = os.path.join(cfg.save_model_path, "pytorch_model.bin")

    # --- 1. Prefer the final, fully-trained weights if they exist ---
    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_weights_path}")
        return True

    print(f"No model weights found at {model_weights_path}")

    # --- 2. Fall back to the latest training checkpoint, if any ---
    checkpoint_path = os.path.join(cfg.checkpoints_path, "latest.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded model weights from checkpoint {checkpoint_path} "
            f"(epoch {checkpoint.get('epoch')}, step {checkpoint.get('global_step')})"
        )
        return True

    print(f"No checkpoint found at {checkpoint_path} either")
    return False
