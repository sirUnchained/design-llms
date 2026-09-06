import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from configs.model_configs import GPT_configs
from src.data.pretrain_dataset.tokenize import tokenize_to_bin


class MemmapGPTDataset(Dataset):
    """
    ## Memory-Mapped GPT Dataset Class

    This class makes your pre-tokenized binary corpus ready for the model to
    learn on it, without ever loading the full token sequence into RAM.

    Unlike a dataset that tokenizes raw text up front and pre-materializes
    every training chunk as a tensor, this class reads token ids lazily from
    a `uint16` binary file on disk using `numpy.memmap`. Only the slice of
    tokens needed for the requested chunk is paged into memory, so RAM usage
    stays flat no matter how large the underlying dataset is.

    A `[start_idx, end_idx)` token range can be given so a single binary
    file can be shared between a train split and a val split without
    duplicating it on disk.

    The memmap handle is opened lazily inside `__getitem__` (not `__init__`)
    so that each DataLoader worker process opens its own file handle, which
    is required for correctness when `num_workers > 0`.

    ---

    Args:
        bin_path (str):
            Path to the binary file of `uint16` token ids, as produced by
            `tokenize_to_bin`.
        context_length (int):
            Maximum sequence length (number of tokens per sample).
        stride (int):
            Stride per sample, in tokens. Use `stride == context_length` for
            non-overlapping chunks.
        start_idx (int, optional):
            First token index (inclusive) this dataset is allowed to read
            from. Default is 0.
        end_idx (int, optional):
            Last token index (exclusive) this dataset is allowed to read up
            to. Default is `None`, meaning the end of the file.
    """

    def __init__(
        self, bin_path, context_length, stride, start_idx=0, end_idx=None
    ) -> None:
        self.bin_path = bin_path
        self.context_length = context_length
        self.stride = stride
        self.start_idx = start_idx

        # Opened here only to read the token count cheaply; not kept around,
        # since the real per-worker handle is opened lazily in __getitem__.
        data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        n_tokens = data.shape[0]
        self.end_idx = n_tokens if end_idx is None else min(end_idx, n_tokens)

        usable_tokens = self.end_idx - self.start_idx
        self.length = max(0, (usable_tokens - context_length) // stride)

        self._data = None

    def _ensure_open(self):
        """
        ## Lazily open the memmap handle for the current process.

        Ensures each DataLoader worker process (when `num_workers > 0`) opens
        its own independent memmap handle rather than sharing one created in
        the main process, which numpy's memmap does not support safely across
        forked/spawned workers.

        ---

        Args:
            None

        Returns:
            None: Sets `self._data` in place.
        """
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        ## Fetch one (input, target) chunk pair by index.

        Reads a single window of `context_length + 1` tokens starting at
        `start_idx + idx * stride` from the memmap, then splits it into an
        input chunk (all but the last token) and a target chunk (all but
        the first token) — the standard next-token-prediction shift.

        ---

        Args:
            idx (int): Index of the sample to fetch, relative to this
                      dataset's `[start_idx, end_idx)` range.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(input_ids, target_ids)`,
                each of shape `(context_length,)` and dtype `int64`.
        """
        self._ensure_open()
        start = self.start_idx + idx * self.stride
        chunk = self._data[start : start + self.context_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def get_bin_path(data_path: str) -> str:
    """
    ## Derive the tokenized binary path for a given raw data path.

    Keeps the tokenized cache next to the source corpus, same name, `.bin`
    extension. So `./data/llm_dataset.jsonl` maps to `./data/llm_dataset.bin`.

    ---

    Args:
        data_path (str): Path to the raw `.txt` or `.jsonl` corpus.

    Returns:
        str: Path to the corresponding `.bin` tokenized file.
    """
    root, _ = os.path.splitext(data_path)
    return root + ".bin"


def prepare_bin_dataset(cfg: GPT_configs, force: bool = False) -> str:
    """
    ## Ensure a tokenized binary exists for the configured dataset, building it if needed.

    Checks for a `.bin` file next to `cfg.data_path` (see `get_bin_path`). If
    it's missing (or `force=True`), tokenizes the raw corpus once via
    `tokenize_to_bin`. On every subsequent run, the existing `.bin` is reused
    and tokenization is skipped entirely.

    ---

    Args:
        cfg (GPT_configs): Your GPT config class. Uses `cfg.data_path` as the
                           source corpus and `cfg.ticktoken_tokenizer` as the
                           tokenizer name.
        force (bool, optional): If `True`, re-tokenize even if a `.bin` file
                                already exists. Default is `False`.

    Returns:
        str: Path to the ready-to-use tokenized `.bin` file.
    """
    bin_path = get_bin_path(cfg.data_path)

    if force or not os.path.exists(bin_path):
        print(
            f"No tokenized binary found at {bin_path}, tokenizing {cfg.data_path} ..."
        )
        tokenize_to_bin(cfg.data_path, bin_path, tokenizer_name=cfg.ticktoken_tokenizer)
    else:
        print(f"Found existing tokenized binary at {bin_path}, skipping tokenization.")

    return bin_path


def create_dataloader(
    bin_path,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
    start_frac=0.0,
    end_frac=1.0,
):
    """
    ## Dataloader Creator

    This function will create a pytorch dataloader backed by `MemmapGPTDataset`,
    which is essential for training on datasets too large to tokenize and hold
    in RAM all at once.

    `start_frac`/`end_frac` let you carve a train/val split (or any subset)
    out of a single tokenized `.bin` file by token position, so you don't need
    to tokenize or store separate files per split.

    ---

    Args:
        bin_path (str):
            Path to the tokenized `.bin` file, as returned by `prepare_bin_dataset`.
        batch_size (int):
            The batch size.
        max_length (int):
            Maximum sequence length.
        stride (int):
            Stride per sample.
        shuffle (bool):
            Send True if you need to shuffle input text.
        drop_last (bool):
            This will drop last batch if it is shorter than the specified `batch_size`.
        num_workers (int):
            How many subprocesses to use for data loading. Safe to set > 0
            here since each worker opens its own memmap handle.
        pin_memory (bool):
            If True, copies tensors into pinned memory before returning them,
            which speeds up the host-to-GPU transfer. Default is True.
        start_frac (float, optional):
            Fraction (0.0-1.0) of the token stream where this dataloader's
            range begins. Default is 0.0.
        end_frac (float, optional):
            Fraction (0.0-1.0) of the token stream where this dataloader's
            range ends. Default is 1.0.

    Returns:
        torch.utils.data.Dataloader: Our needed dataloader
    """
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_tokens = data.shape[0]

    start_idx = int(start_frac * n_tokens)
    end_idx = int(end_frac * n_tokens)

    dataset = MemmapGPTDataset(
        bin_path,
        context_length=max_length,
        stride=stride,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return dataloader


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        txt_path = os.path.join(tmp_dir, "sample.txt")
        bin_path = os.path.join(tmp_dir, "sample.bin")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Some hello world text? I think it should be, But let's see ...")

        tokenize_to_bin(txt_path, bin_path)

        dataloader = create_dataloader(
            bin_path,
            batch_size=2,
            max_length=3,
            stride=2,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        input_batch, target_batch = next(iter(dataloader))
        print("input batch:", input_batch)
        print("target batch:", target_batch)
