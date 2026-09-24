import torch
from torch.utils.data import Dataset
import tiktoken

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


def custom_collate_draft(batch: list[list[int]], pad_token_id=50256, device="cpu"):
    # find the largest item in batch then get it's size and add 1 in it
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # this is how we pad the text, prompt + (eof_token * total_prompt_len - prompt_len)
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
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
        batch=[dataset[0], dataset[1]]
    )
    print("input:", padded_text_input[1])
    print("target:", padded_text_target[1])
    print("=" * 100)
