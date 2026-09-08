import os
import functools

import torch

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

from configs.model_configs import get_gpt_configs

HF_CHECKPOINTS_SUBFOLDER = "check-points"


# Repos we've already confirmed exist on the Hub, so we don't call
# create_repo() again on every single checkpoint save.
@functools.lru_cache(maxsize=None)
def _ensure_repo(repo_id: str) -> None:
    create_repo(repo_id=repo_id, exist_ok=True, private=False)


@functools.lru_cache(maxsize=1)
def _repo_id() -> str:
    return get_gpt_configs().repo_id


def _push_checkpoint_to_hub(path, repo_id, epoch, global_step):
    """Upload a single checkpoint file to the `check-points` folder of the HF repo.

    Never raises: a Hub/network hiccup should not crash a training run. Errors are
    printed as warnings instead.
    """

    filename = os.path.basename(path)
    path_in_repo = f"{HF_CHECKPOINTS_SUBFOLDER}/{filename}"

    try:
        _ensure_repo(repo_id)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=f"Add checkpoint (epoch {epoch}, step {global_step})",
        )
        print(
            f"Checkpoint pushed to https://huggingface.co/{repo_id}/blob/main/{path_in_repo}"
        )
    except HfHubHTTPError as exc:
        print(f"Warning: Hugging Face Hub rejected the checkpoint upload: {exc}")
    except (
        Exception
    ) as exc:  # noqa: BLE001 - deliberately broad, this must never kill training
        print(f"Warning: failed to push checkpoint to Hugging Face Hub: {exc}")


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    tokens_seen,
    train_losses,
    val_losses,
    track_tokens_seen,
    total_steps,
):
    """Save everything needed to exactly resume training, and mirror it to the
    `check-points` folder of the configured Hugging Face repo (`REPO_ID`).

    If `repo_id` is unset/invalid ("INVALID"), the Hugging Face push is skipped
    and only the local save happens, same as before.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen,
            "total_steps": total_steps,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")

    repo_id = _repo_id()
    if repo_id and repo_id != "INVALID":
        _push_checkpoint_to_hub(path, repo_id, epoch, global_step)
    else:
        print("REPO_ID not set (INVALID) — skipping Hugging Face push.")


def load_checkpoint(path, model, optimizer, device):
    """Load a checkpoint and return the state needed to resume training."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"Checkpoint loaded from {path} (resuming at step {checkpoint['global_step']})"
    )
    return checkpoint
