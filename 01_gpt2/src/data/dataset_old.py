import os
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tiktoken

from configs.model_configs import GPT_configs


class GPT2DatasetV1(Dataset):
    """
    ## GPT Dataset Class

    This class makes you're text ready for model to learn on it.

    ---

    Args:

        txt(str) :
            The input text data
        tokenizer (object):
            The tokenizer object
        max_length (int):
            Maximum sequence length
        stride (int):
            Stride per sample
    """

    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    tokenizer_name="gpt2",
):
    """
    ## Dataloader Creator

    This function will create pytorch dataloader for model, which is essential for machine learning tasks with pytorch.

    ---

    Args:
        txt(str):
            The input text data
        batch_size(int):
            The batch size
        max_length(int):
            Maximum sequence length
        stride(int):
            Stride per sample
        shuffle (bool):
            Send True if you need to shuffle input text
        drop_last (bool):
            This will drop last batch if it is shorter than the specified `batch_size`
        num_workers(int):
            how many subprocesses to use for data loading

    Returns:
        torch.utils.data.Dataloader: Our needed dataloader

    """

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    dataset = GPT2DatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
        num_workers=num_workers,  # The number of CPU processes to use for preprocessing
    )

    return dataloader


def load_text_data(cfg: GPT_configs) -> str:
    """
    Load and return the entire content of a text file as a string.

    Args:
        cfg (GPT_configs): Youre GPT config class.

    Returns:
        str: The complete text content read from the file.
    """

    file_path = cfg.data_path
    _, ext = os.path.splitext(file_path)

    # Handle JSONL files: read each line, parse JSON, extract the "text" field
    if ext.lower() == ".jsonl":
        text_parts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                data = json.loads(line)
                text_parts.append(data["text"])  # Explicitly raise KeyError if missing
        return "\n".join(text_parts)

    # Default: treat as plain text file
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    dataloader = create_dataloader(
        txt="Some hello world text? I think it should be, But let's see ...",
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
