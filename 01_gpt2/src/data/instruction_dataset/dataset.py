import torch
from torch.utils.data import Dataset
import tiktoken

from typing import Optional
import json
from pydantic.dataclasses import dataclass


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        self.data = data
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\\n### Response:\\n{entry.output}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def get_data(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\\n### Instruction:\\n{entry.instruction}"
    )

    input_text = f"\n\\n### Input:\\n{entry.input}" if entry.input else ""
    return instruction_text + input_text


def custom_collate_draft(
    batch: list[list[int]],
    pad_token_id=50256,
    ignore_index=-100,  # why we chosed -100? because by default `torch.nn.CrossEntropyLoss` uses `-100` to ignore.
    allowed_max_length=0,
    device="cpu",
):
    """
    Collate variable-length token sequences into padded input and target tensors for causal language model training.
    Each sequence is appended with a padding/EOF token and padded to the maximum sequence length in the batch.
    The resulting sequence is then shifted by one position to create the input-target pair required for next-token prediction.

    Padding tokens in the target tensor are replaced with `ignore_index` so that they are excluded from the loss calculation
    by PyTorch loss functions such as `torch.nn.CrossEntropyLoss`.

    If `allowed_max_length` is greater than zero, both inputs and targets are truncated to that maximum length after padding.

    ---

    Args:
        batch (int): A batch of tokenized sequences. Each sequence is represented as a list of integer token IDs and may have a different length.
        pad_token_id (int): Token ID used for padding sequences and for the sequence-ending token appended before creating the input-target pairs.
            Defaults to ``50256``.
        ignore_index (int): Value used to mark target positions that should be ignored when computing the loss. PyTorch's ``CrossEntropyLoss``
            uses ``-100`` by default. Defaults to ``-100``.
        allowed_max_length (int): Maximum number of tokens allowed in the resulting input and target sequences. A value of ``0``
            disables truncation. Defaults to ``0``.
        device (str): Device to which the resulting tensors are moved, such as ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:

            - ``inputs_tensor``: Tensor of shape ``(batch_size, sequence_length)`` containing input token IDs.
            - ``targets_tensor``: Tensor of shape ``(batch_size, sequence_length)`` containing next-token prediction targets.
                Padding positions are replaced with ``ignore_index``.
    """

    # find the largest item in batch then get it's size and add 1 in it
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # this is how we pad the text, prompt + (eof_token * total_prompt_len - prompt_len)
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # but we truncate last token for input text
        targets = torch.tensor(padded[1:])  # we also ignore first token for target text

        # we are now replacing all padded token (except first one) with `ignore_index`
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # this part is optional, we jst truncate texts to be as size as `allowed_max_length`
        if allowed_max_length > 0:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


if __name__ == "__main__":

    @dataclass
    class instruct_dtype:
        instruction: str
        input: str
        output: str

    data = [
        instruct_dtype(
            instruction="Edit the following sentence for grammar.",
            input="He go to the park every day.",
            output="He goes to the park every day.",
        ),
        instruct_dtype(
            instruction="Convert 45 kilometers to meters.",
            input="",
            output="45 kilometers is 45000 meters.",
        ),
    ]
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = InstructionDataset(data=data, tokenizer=tokenizer)
    print("MAIN TEXT:")
    print(dataset.get_data(1))
    print("=" * 100)
    print("TOKENIZED TEXT:")
    print(dataset[1])
    print("=" * 100)
    print("PADDED TOKEN:")
    padded_text_input, padded_text_target = custom_collate_draft(
        batch=[dataset[0], dataset[1]], allowed_max_length=1024
    )
    print("input:", padded_text_input[1])
    print("target:", padded_text_target[1])
    print("=" * 100)
