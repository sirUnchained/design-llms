import torch
from dotenv import load_dotenv, find_dotenv

import os


class Qwen_Configs:
    epochs = 5
    batch_size = 2

    vocab_size = 50257
    context_length = 1024
    emb_dim = 768
    n_heads = 12
    n_layers = 12
    head_dim = 128
    hidden_dim = 3072
    n_kv_groups = 6
    rope_base = 1000000.0
    qk_norm = False
    dtype = torch.float32

    seed = 0

    lr_schedule_step = 0

    tokenizer_name = "Qwen/Qwen3-0.6B"
    data_path = "./data/llm_dataset.txt"
    save_model_path = "model-weights/"
    repo_id = "INVALID"

    checkpoint_freq = 1000
    create_checkpoints = True
    use_checkpoints = True
    checkpoints_path = "check-points/"


def get_Qwen_Configs():
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    config = Qwen_Configs()

    config.epochs = int(os.getenv("EPOCHS", config.epochs))
    config.batch_size = int(os.getenv("BATCH_SIZE", config.batch_size))

    config.vocab_size = int(os.getenv("VOCAB_SIZE", config.vocab_size))
    config.context_length = int(os.getenv("CONTEXT_LENGTH", config.context_length))
    config.emb_dim = int(os.getenv("EMB_DIM", config.emb_dim))
    config.n_heads = int(os.getenv("N_HEADS", config.n_heads))
    config.n_layers = int(os.getenv("N_LAYERS", config.n_layers))
    config.head_dim = int(os.getenv("HEAD_DIM", config.head_dim))
    config.hidden_dim = int(os.getenv("HIDDEN_DIM", config.hidden_dim))
    config.n_kv_groups = int(os.getenv("N_KV_GROUPS", config.n_kv_groups))
    config.rope_base = float(os.getenv("ROPE_BASE", config.rope_base))
    config.qk_norm = os.getenv("QK_NORM", str(config.qk_norm)).lower() in (
        "true",
        "1",
        "yes",
    )

    config.seed = int(os.getenv("SEED", config.seed))

    config.lr_schedule_step = int(
        os.getenv("LR_SCHEDULE_STEP", config.lr_schedule_step)
    )

    config.tokenizer_name = os.getenv("TOKENIZER_NAME", config.tokenizer_name)
    config.data_path = os.getenv("DATA_PATH", config.data_path)
    config.save_model_path = os.getenv("SAVE_MODEL_PATH", config.save_model_path)
    config.repo_id = os.getenv("REPO_ID", config.repo_id)

    config.checkpoint_freq = int(os.getenv("CHECKPOINT_FREQ", config.checkpoint_freq))
    config.create_checkpoints = os.getenv(
        "CREAET_CHECKPOINTS", str(config.create_checkpoints)
    ).lower() in (
        "true",
        "1",
        "yes",
    )
    config.use_checkpoints = os.getenv(
        "USE_CHECKPOINTS", str(config.use_checkpoints)
    ).lower() in (
        "true",
        "1",
        "yes",
    )
    config.checkpoints_path = os.getenv("CHECKPOINTS_PATH", config.save_model_path)

    return config
