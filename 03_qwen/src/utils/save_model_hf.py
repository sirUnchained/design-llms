from huggingface_hub import HfApi, create_repo


import os
import getpass

from configs.model_configs import Qwen_Configs

from huggingface_hub import HfApi, create_repo
from huggingface_hub import login, whoami
from huggingface_hub.errors import HfHubHTTPError


def ensure_huggingface_login(cfg):
    """
    Logs the user into Hugging Face if the condition is met and they are not already logged in.
    If the token is missing from environment variables, prompts the user to enter it via terminal.
    """
    # Check the condition: skip everything if repo_id is invalid
    if cfg.repo_id == "INVALID":
        return

    # Verify whether the user is already authenticated
    try:
        user_info = whoami()
        # Already logged in → do nothing (just "pass")
        print(f"Already logged in as: {user_info.get('name', 'Unknown')}")
        return
    except HfHubHTTPError as e:
        # Not authenticated (typically 401 Unauthorized) → proceed to login
        print("Not logged in. Proceeding to login...")
    except Exception as e:
        # Fallback for network errors or older versions
        print(f"Could not verify login status ({e}). Attempting login anyway...")

    # 1. Try to read the token from environment variable
    token = os.getenv("HF_TOKEN")

    # 2. If not set in env, ask the user interactively in the terminal
    if not token:
        print("HF_TOKEN environment variable is not set.")
        token = getpass.getpass("Please paste your Hugging Face access token: ")

    # 3. Safety guard – ensure we actually have a token
    if not token:
        raise ValueError("Hugging Face token is required to proceed.")

    # 4. Perform the login using the latest method
    #    add_to_git_credential=True also stores it locally for `git` commands
    login(token=token, add_to_git_credential=True)
    print("Successfully logged in to Hugging Face Hub.")


# --------------------------------------------------------------------------
# Updated save_model_hf function
# --------------------------------------------------------------------------


def save_model_hf(cfg: Qwen_Configs):
    """
    Upload a trained model and its configuration to the Hugging Face Hub.

    Uses the new login mechanism that checks the condition and handles
    environment variables / interactive token input.

    Args:
        cfg (Qwen_Configs): Configuration object containing:
            - save_model_path (str): Local directory where model files are stored.
            - checkpoints_path (str): Local directory for checkpoints (optional).
            - repo_id (str): Hugging Face Hub repository ID (e.g., "username/model-name").
            - create_checkpoints (bool): Whether to upload checkpoints folder.

    Returns:
        None
    """

    # --- 1. Ensure login (with condition check inside) ---
    ensure_huggingface_login(cfg)

    # --- 2. Build file paths ---
    folder_checkpoints_name = cfg.checkpoints_path
    folder_weights_name = cfg.save_model_path
    model_weights_name = "pytorch_model.bin"
    config_file_name = "configs.json"

    model_weights_path = os.path.join(folder_weights_name, model_weights_name)
    config_file_path = os.path.join(folder_weights_name, config_file_name)

    # --- 3. Verify local files exist ---
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found at {config_file_path}")

    # --- 4. Prepare the API and repository ---
    repo_id = cfg.repo_id
    api = HfApi()

    print(f"Attempting to create/get repository: {repo_id}")
    create_repo(repo_id=repo_id, exist_ok=True, private=False)
    print(f"Repository {repo_id} created or already exists.")

    # --- 5. Upload model weights ---
    print(f"Uploading model weights ({model_weights_name})...")
    api.upload_file(
        path_or_fileobj=model_weights_path,
        path_in_repo=model_weights_name,
        repo_id=repo_id,
        commit_message="Add PyTorch model weights",
    )

    # --- 6. Upload configuration ---
    print(f"Uploading model configuration ({config_file_name})...")
    api.upload_file(
        path_or_fileobj=config_file_path,
        path_in_repo=config_file_name,
        repo_id=repo_id,
        commit_message="Add model configuration",
    )

    # --- 7. Optionally upload checkpoints ---
    if cfg.create_checkpoints:
        print(f"Uploading model checkpoints ({folder_checkpoints_name})...")
        api.upload_folder(
            folder_path=folder_checkpoints_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model checkpoints",
        )

    print(
        f"Successfully pushed model to Hugging Face Hub: https://huggingface.co/{repo_id}"
    )
