import json

import numpy as np
import tiktoken


def tokenize_to_bin(src_path, out_path, tokenizer_name="gpt2", chunk_chars=50_000_000):
    """
    ## Tokenize a large text/JSONL corpus into a binary token file.

    Instead of loading the entire corpus into memory as a Python string and
    encoding it in one shot (which explodes RAM on multi-GB datasets), this
    function reads the source file in bounded-size character chunks, encodes
    each chunk independently with tiktoken, and appends the resulting token
    ids into a single `uint16` binary file on disk via `numpy.memmap`.

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
            Number of characters to accumulate before encoding and flushing
            a chunk. Controls the memory/throughput tradeoff during
            preprocessing. Default is 50,000,000 (~50MB of text per chunk).

    Returns:
        int: Total number of tokens written to `out_path`.
    """
    enc = tiktoken.get_encoding(tokenizer_name)
    is_jsonl = src_path.endswith(".jsonl")

    arrs = []
    buf = ""

    def flush(buf):
        """
        ## Encode a text buffer into a uint16 numpy array of token ids.

        Small helper used by `tokenize_to_bin` to convert an accumulated
        chunk of raw text into token ids without special-token handling,
        since a raw training corpus shouldn't contain control tokens.

        ---

        Args:
            buf (str): Accumulated text chunk to encode.

        Returns:
            np.ndarray: Array of token ids with dtype `uint16`. Empty array
                        if `buf` is empty.
        """
        if not buf:
            return np.empty(0, dtype=np.uint16)
        ids = enc.encode_ordinary(buf)
        return np.array(ids, dtype=np.uint16)

    with open(src_path, "r", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                buf += json.loads(line)["text"] + "\n"
                if len(buf) >= chunk_chars:
                    arrs.append(flush(buf))
                    buf = ""
        else:
            while True:
                piece = f.read(chunk_chars)
                if not piece:
                    break
                buf += piece
                arrs.append(flush(buf))
                buf = ""
        if buf:
            arrs.append(flush(buf))

    total_len = sum(a.size for a in arrs)
    mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_len,))
    offset = 0
    for a in arrs:
        mm[offset : offset + a.size] = a
        offset += a.size
    mm.flush()
    print(f"Wrote {total_len:,} tokens to {out_path}")

    return total_len
