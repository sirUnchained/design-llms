# GPT-2 From Scratch

A from-scratch PyTorch implementation of a GPT-2 style decoder-only transformer, including training, checkpointing, Hugging Face Hub sync, and text generation.

> **Note**: This version only uses a single GPU so if you have more than 1 GPU, this model only uses one of them.

## Features

- Custom `GPT_model` built from token/positional embeddings, stacked transformer blocks, layer norm, and an output head
- Multi-head causal self-attention implemented manually (no `nn.MultiheadAttention`)
- Configurable via `.env` (model size, training hyperparameters, tokenizer, paths)
- Memory-mapped dataset pipeline: JSONL/text corpora are tokenized once to a `uint16` binary file, then streamed via `numpy.memmap`, so RAM usage stays flat regardless of dataset size (multi-GB+ corpora included)
- Training loop with periodic evaluation, perplexity tracking, and loss plotting
- Resumable checkpoints, mirrored automatically to a Hugging Face Hub repo
- Greedy and temperature/top-k sampling for text generation
- Simple CLI (`-t` train, `-g` generate, `-i` info) plus an interactive fallback menu

## Architecture

```mermaid
flowchart TD
    A["Input token ids (batch, seq_len)"] --> B["Token Embedding"]
    A --> C["Positional Embedding"]
    B --> D["Add"]
    C --> D
    D --> E["Dropout"]
    E --> F["Transformer Block x N layers"]
    F --> G["Final LayerNorm"]
    G --> H["Linear Output Head"]
    H --> I["Logits (batch, seq_len, vocab_size)"]
```

### Transformer Block

```mermaid
flowchart TD
    X["Input x"] --> N1["LayerNorm"]
    N1 --> MHA["Multi-Head Attention"]
    MHA --> D1["Dropout"]
    D1 --> S1["Add (residual)"]
    X --> S1
    S1 --> N2["LayerNorm"]
    N2 --> FF["Feed Forward (Linear -> GELU -> Linear)"]
    FF --> D2["Dropout"]
    D2 --> S2["Add (residual)"]
    S1 --> S2
    S2 --> OUT["Output"]
```

### Multi-Head Attention

```mermaid
flowchart LR
    X["Input x"] --> Q["Linear: Query"]
    X --> K["Linear: Key"]
    X --> V["Linear: Value"]
    Q --> SPLIT1["Split into heads"]
    K --> SPLIT2["Split into heads"]
    V --> SPLIT3["Split into heads"]
    SPLIT1 --> ATT["Scaled Dot-Product Attention + Causal Mask"]
    SPLIT2 --> ATT
    ATT --> SOFT["Softmax + Dropout"]
    SOFT --> CTX["Weighted Sum with Values"]
    SPLIT3 --> CTX
    CTX --> MERGE["Concatenate heads"]
    MERGE --> PROJ["Linear: Output Projection"]
    PROJ --> OUT["Context vectors"]
```

### Data Pipeline

```mermaid
flowchart TD
    A["Raw corpus (.txt / .jsonl)"] --> B{".bin already exists?"}
    B -- no --> C["tokenize_to_bin: stream-encode in chunks"]
    C --> D["Write uint16 tokens to .bin via memmap"]
    B -- yes --> E["Reuse existing .bin"]
    D --> F["MemmapGPTDataset: lazy random-access reads"]
    E --> F
    F --> G["DataLoader (num_workers, pin_memory)"]
    G --> H["Train / Val split by token-position range"]
```

## Project Structure

```
.
├── main.py                     # CLI entry point (train / generate / info)
├── configs/
│   └── model_configs.py        # GPT_configs dataclass + .env loader
├── scripts/
│   └── evaluate.py             # Tokenization helpers + text generation
├── src/
│   ├── data/
│   │   │
│   │   pretrain_dataset/
│   │       ├── tokenize.py         # One-time corpus -> uint16 .bin tokenization
│   │       └── dataset.py          # Memmap Dataset, DataLoader, .bin caching
│   ├── models/
│   │   ├── gpt_model.py        # Top-level GPT_model
│   │   ├── transformer_block.py
│   │   ├── multi_head_attention.py
│   │   ├── feed_forward.py
│   │   ├── activations.py      # GELU
│   │   └── normalizers.py      # LayerNorm
│   ├── training/
│   │   ├── train.py            # Training loop, checkpoint resume, LR schedule
│   │   └── loss.py             # Cross-entropy loss + perplexity
│   └── utils/
│       ├── load_model.py       # Load local weights if present
│       ├── save_model_hf.py    # Push model + config to HF Hub
│       └── ckeckpoints.py      # Save/load training checkpoints (local + HF)
├── requirements.txt
└── .env                        # Runtime configuration (not committed with real secrets)
```

## Setup

```bash
pip install -r requirements.txt
```

Configure your run in `.env`:

```env
EPOCHS=30
BATCH_SIZE=2
VOCAB_SIZE=50257
CONTEXT_LENGTH=1024
EMB_DIM=768
N_HEADS=12
N_LAYERS=12
DROP_RATE=0.2
QKV_BIAS=True
TIKTOKEN_TOKENIZER=gpt2

DATA_PATH=./data/llm_dataset.jsonl
SAVE_MODEL_PATH=model-weights/
REPO_ID=your-username/your-repo

CHECKPOINT_FREQ=1000
CREAET_CHECKPOINTS=true
USE_CHECKPOINTS=true
CHECKPOINTS_PATH=check-points/
```

> Data can be a plain `.txt` file or a `.jsonl` file where each line has a `"text"` field.

### Dataset tokenization (automatic, cached)

On the first training run, the corpus at `DATA_PATH` is tokenized once and written to a `uint16` binary file next to it (e.g. `./data/llm_dataset.jsonl` $\to$ `./data/llm_dataset.bin`). Every subsequent run reuses that `.bin` file directly via `numpy.memmap` instead of re-tokenizing, and only the token windows actually being read are paged into memory — so a 7GB+ corpus trains with roughly the same RAM footprint as a tiny one. Delete the `.bin` file (or pass `force=True` to `prepare_bin_dataset`) if you change the source data and need to re-tokenize.

If `REPO_ID` is a valid Hugging Face repo id, the app will prompt for login (or read `HF_TOKEN`) and automatically push weights, config, and checkpoints to the Hub.

## Usage

### Train

```bash
python main.py -t
```

Skips training and loads existing weights if `model-weights/pytorch_model.bin` is already present.

### Generate text

```bash
python main.py -g -p "Once upon a time"
```

Or omit `-p` to be prompted interactively.

### Inspect model & config

```bash
python main.py -i
```

### Interactive mode

Run with no flags for a `q` (quit) / `t` (train) / `g` (generate) / `i` (info) menu.

## Training Flow

```mermaid
flowchart TD
    S["Ensure tokenized .bin exists (tokenize once, cached)"] --> DL["Build train/val DataLoaders (memmap, split by token position)"]
    DL --> CK{"Resume checkpoint?"}
    CK -- yes --> LOAD["Load latest.pt"]
    CK -- no --> INIT["Start fresh"]
    LOAD --> EPOCHLOOP["For each epoch"]
    INIT --> EPOCHLOOP
    EPOCHLOOP --> BATCHLOOP["For each batch"]
    BATCHLOOP --> FWD["Forward pass + cross-entropy loss"]
    FWD --> BWD["Backward pass + optimizer step"]
    BWD --> EVALCHECK{"global_step % eval_freq == 0?"}
    EVALCHECK -- yes --> EVAL["Evaluate on train/val + log perplexity"]
    EVALCHECK -- no --> CKPTCHECK
    EVAL --> CKPTCHECK{"global_step % checkpoint_freq == 0?"}
    CKPTCHECK -- yes --> SAVE["Save checkpoint (local + HF Hub)"]
    CKPTCHECK -- no --> BATCHLOOP
    SAVE --> BATCHLOOP
    BATCHLOOP -- epoch done --> SAMPLE["Generate sample text"]
    SAMPLE --> EPOCHCKPT["Save end-of-epoch checkpoint"]
    EPOCHCKPT --> EPOCHLOOP
    EPOCHLOOP -- all epochs done --> SAVEFINAL["Save final weights + config"]
    SAVEFINAL --> PUSH["Push to Hugging Face Hub"]
    PUSH --> PLOT["Plot train/val loss curve"]
```

## Sources

I used [Build a Large Language Model (From Scratch)](https://www.oreilly.com/library/view/build-a-large/9781633437166/) + [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) + [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the first part of [Hands-On Large Language Models: Language Understanding and Generation](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/) so I could understand deeply what is going on inside LLMs.

## License

For this repo we use MIT License so feel free to change it or help me to improve.
