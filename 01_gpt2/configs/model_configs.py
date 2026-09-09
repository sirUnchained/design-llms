from dotenv import load_dotenv, find_dotenv

import os


class GPT_configs:
    epochs = 5
    batch_size = 2

    vocab_size = 50257
    context_length = 1024
    emb_dim = 768
    n_heads = 12
    n_layers = 12
    drop_rate = 0.1
    qkv_bias = False

    lr_schedule_step = 0

    ticktoken_tokenizer = "gpt2"
    data_path = "./data/llm_dataset.txt"
    save_model_path = "model-weights/"
    repo_id = "INVALID"

    checkpoint_freq = 1000
    create_checkpoints = True
    use_checkpoints = True
    checkpoints_path = "check-points/"


def get_gpt_configs():
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    config = GPT_configs()

    config.epochs = int(os.getenv("EPOCHS", config.epochs))
    config.batch_size = int(os.getenv("BATCH_SIZE", config.batch_size))

    config.vocab_size = int(os.getenv("VOCAB_SIZE", config.vocab_size))
    config.context_length = int(os.getenv("CONTEXT_LENGTH", config.context_length))
    config.emb_dim = int(os.getenv("EMB_DIM", config.emb_dim))
    config.n_heads = int(os.getenv("N_HEADS", config.n_heads))
    config.n_layers = int(os.getenv("N_LAYERS", config.n_layers))
    config.drop_rate = float(os.getenv("DROP_RATE", config.drop_rate))
    config.qkv_bias = os.getenv("QKV_BIAS", str(config.qkv_bias)).lower() in (
        "true",
        "1",
        "yes",
    )

    config.lr_schedule_step = int(
        os.getenv("LR_SCHEDULE_STEP", "0"), config.lr_schedule_step
    )

    config.ticktoken_tokenizer = os.getenv("TOKENIZER", config.ticktoken_tokenizer)
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
