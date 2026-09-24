import torch
import tiktoken


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding):
    """
    ## Convert a tensor of token IDs back into a human-readable text string.

    The tensor is first squeezed to remove any singleton dimensions, then converted
    to a list of integers before being decoded by the tokenizer.

    Args:
        token_ids (torch.Tensor): Tensor containing token IDs (can be multi-dimensional).
        tokenizer (tiktoken.Encoding): Tokenizer instance used for decoding.

    Returns:
        str: The decoded text string.
    """
    flated_token_ids = token_ids.squeeze().tolist()
    return tokenizer.decode(flated_token_ids)


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding):
    """
    ## Convert a text string into a tensor of token IDs.

    The tokenizer encodes the input text (allowing the special token "<|endoftext|>").
    The resulting token list is then converted to a PyTorch tensor with an added batch dimension.

    Args:
        text (str): The input text to be tokenized.
        tokenizer (tiktoken.Encoding): Tokenizer instance used for encoding.

    Returns:
        torch.Tensor: A 2D tensor of shape (1, seq_len) containing the token IDs.
    """
    encoded_text = tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded_text).unsqueeze(dim=0)
    return encoded_tensor


def generate_text(model, idx, max_new_tokens, content_size):
    """
    ## Autoregressively generate new tokens given an initial sequence.

    The function truncates the input context to the last `content_size` tokens, then repeatedly feeds the model to obtain logits for the next token. The next token
    is selected using greedy decoding (argmax of softmax probabilities). The generatedtoken is appended to the sequence, and the process repeats until `max_new_tokens`
    have been produced.

    Args:
        model (torch.nn.Module): The language model used for generation.
        idx (torch.Tensor): Initial token indices (shape: (batch_size, seq_len)).
        max_new_tokens (int): Number of new tokens to generate.
        content_size (int): Maximum number of past tokens to consider as context.
                            (Note: this likely refers to the model's context window size.)

    Returns:
        torch.Tensor: The extended token sequence including the original tokens
                      and the newly generated ones (shape: (batch_size, seq_len + max_new_tokens)).
    """
    for _ in range(max_new_tokens):
        limited_input_data = idx[:, -content_size:]

        with torch.no_grad():
            logits = model(limited_input_data)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text_with_temperature_topk(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=0, eos_id=None
):
    """
    ## Generate new tokens using temperature scaling and optional top-k sampling.

    This function autoregressively generates text by repeatedly feeding the model the last `context_size` tokens. It supports:
    - Greedy decoding (temperature=0.0)
    - Stochastic sampling with temperature > 0.0
    - Top-k filtering to sample only from the k most likely tokens
    - Early stopping when an end-of-sequence token (eos_id) is generated.

    Args:
        model (torch.nn.Module): The language model.
        idx (torch.Tensor): Initial token indices of shape (batch_size, seq_len).
        max_new_tokens (int): Maximum number of new tokens to generate.
        context_size (int): Number of past tokens the model can attend to (context window).
        temperature (float, optional): Sampling temperature. Values > 0.0 enable
            probabilistic sampling. Default is 0.0 (greedy / argmax).
        top_k (int, optional): If provided, only the `top_k` tokens with the highest
            logits are considered; others are masked to -inf. Default is 0 (no filtering).
        eos_id (int, optional): Token ID that marks the end of a sequence. If provided,
            generation stops when this token is produced. Default is None (no early stopping).

    Returns:
        torch.Tensor: The extended token sequence of shape (batch_size, seq_len + generated_tokens).
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # apply top‑k filtering: keep only the k largest logits
        if top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # Sample or greedily pick the next token
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # stop if the generated token is the EOS token
        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
