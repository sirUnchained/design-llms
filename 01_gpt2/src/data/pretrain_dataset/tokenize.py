# src/data/tokenize.py
import json

import numpy as np
import tiktoken


def _iter_text_chunks(src_path, chunk_chars):
    """
    ## Yield bounded-size text chunks from a corpus file.

    Used by `tokenize_to_bin` so only one chunk of raw text is held in
    memory at a time, regardless of total corpus size.

    ---

    Args:
        src_path (str): Path to a `.txt` or `.jsonl` corpus file.
        chunk_chars (int): Number of characters to accumulate per yielded chunk.

    Yields:
        str: Successive chunks of raw text, each up to ~`chunk_chars` long.
    """
    is_jsonl = src_path.endswith(".jsonl")
    buf = ""

    with open(src_path, "r", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                buf += json.loads(line)["text"] + "\n"
                if len(buf) >= chunk_chars:
                    yield buf
                    buf = ""
        else:
            while True:
                piece = f.read(chunk_chars)
                if not piece:
                    break
                buf += piece
                yield buf
                buf = ""
    if buf:
        yield buf


def tokenize_to_bin(src_path, out_path, tokenizer_name="gpt2", chunk_chars=50_000_000):
    """
    ## Tokenize a large text/JSONL corpus into a binary token file, in a single pass.

    Reads the corpus in bounded-size chunks, encodes each chunk with
    tiktoken, and appends the resulting `uint16` token ids straight onto the
    end of `out_path` as raw bytes. There's no need to know the total token
    count up front: a plain file handle in append-binary mode grows on disk
    as we write, so peak RAM stays bounded by `chunk_chars` regardless of
    corpus size, and the corpus is only tokenized once.

    The resulting file has the same on-disk layout a `numpy.memmap` of
    dtype `uint16` would produce, so it can still be opened for reading with
    `np.memmap(out_path, dtype=np.uint16, mode="r")` (see `MemmapGPTDataset`).

    `uint16` is safe for the GPT-2 tokenizer since `vocab_size` (50257) fits
    under 65536, and it halves storage compared to `int64`.

    This is meant to be run once as a preprocessing step, not during training.
    See `prepare_bin_dataset` in `src/data/dataset.py` for the cached wrapper
    that calls this automatically when needed.

    ---

    Args:
        src_path (str):
            Path to the source corpus. Supports plain `.txt` files and
            `.jsonl` files (one JSON object per line with a `"text"` field).
        out_path (str):
            Path where the resulting binary token file will be written.
        tokenizer_name (str, optional):
            Name of the tiktoken encoding to use. Default is `"gpt2"`.
        chunk_chars (int, optional):
            Number of characters to accumulate before encoding and writing a
            chunk. Controls the memory/throughput tradeoff: larger chunks
            mean fewer tiktoken calls (faster) but more RAM held per chunk.
            Default is 50,000,000 (~50MB of text per chunk).

    Returns:
        int: Total number of tokens written to `out_path`.
    """
    enc = tiktoken.get_encoding(tokenizer_name)
    total_len = 0

    with open(out_path, "wb") as out_f:
        for chunk in _iter_text_chunks(src_path, chunk_chars):
            ids = enc.encode_ordinary(chunk)
            arr = np.array(ids, dtype=np.uint16)
            out_f.write(arr.tobytes())
            total_len += arr.size

    print(f"Wrote {total_len:,} tokens to {out_path}")
    return total_len
