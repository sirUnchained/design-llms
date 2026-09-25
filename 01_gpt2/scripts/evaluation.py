import math

import torch

from scripts.generation import text_to_token_ids, generate_text, token_ids_to_text


def evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter):
    """
    ## Evaluate the model on the training and validation dataloaders.

    This function computes the average loss over all batches in the training and validation sets using `calc_loader_cost`.
    The model is temporarily set to evaluation mode, and then restored to training mode.

    ---

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device on which the tensors are allocated.
        eval_iter (int): Number of batches to use for evaluation.

    Returns:
        tuple: (train_loss, val_loss) where each is a scalar tensor representing
               the average cross-entropy loss over the respective dataloader.
    """
    # Evaluate model
    model.eval()
    with torch.no_grad():
        train_loss = calc_loader_cost(train_dataloader, model, device, eval_iter)
        val_loss = calc_loader_cost(val_dataloader, model, device, eval_iter)
    model.train()

    # Turn the tensores into numbers
    train_loss = (
        train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
    )
    val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    ## Generate a text sample from the model and print it.

    Useful for monitoring training progress: after each epoch, this function generates a fixed number of
    tokens (10) conditioned on `start_context` and prints the resulting text on a single line.

    ---

    Args:
        model (torch.nn.Module): The language model used for generation.
        tokenizer: Tokenizer object that converts between text and token ids.
        device (torch.device): Device where the model and tensors reside.
        start_context (str): Initial prompt string to condition the generation.

    Returns:
        None
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded_text = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model, encoded_text, 10, context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Print sample as a single line
    model.train()


def calc_perplexity(loss) -> float:
    """
    ## Convert a cross-entropy loss value into perplexity.

    Perplexity = exp(cross-entropy loss). It's the standard human-readable evaluation metric for language models:
    roughly "how many tokens, on average, was the model choosing between" at each step. Lower is better.

    > Note: if you're using `z_loss_coeff` > 0 in `calc_batch_cost`, pass the plain cross-entropy component here
    (not the combined loss), since the z-loss term isn't part of the probabilistic interpretation perplexity relies on.

    Args:
        loss (torch.Tensor | float): A (mean) cross-entropy loss value.

    Returns:
        float: The corresponding perplexity. `inf` if the loss overflows exp().
    """
    loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")


def calc_batch_cost(inp_batch, target_batch, model, device):
    """
    ## Calculate the cross-entropy loss for a single batch.

    The input and target batches are moved to the specified device, then passed through the model to obtain logits.
    The loss is computed using `torch.nn.functional.cross_entropy` after flattening the logits and targets to shape (batch_size * seq_len, num_classes).

    Args:
        inp_batch (torch.Tensor): Input token IDs for the batch (shape: (batch_size, seq_len)).
        target_batch (torch.Tensor): Target token IDs for the batch (same shape as `inp_batch`).
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): Device where tensors should be placed.

    Returns:
        torch.Tensor: A scalar tensor containing the average cross-entropy loss for the batch.
    """
    inp_batch = inp_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    logits = model(inp_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loader_cost(data_loader, model, device, num_batches=None):
    """
    ## Compute the average loss over a subset of batches from a DataLoader.

    This function iterates over the DataLoader and uses `calc_batch_cost` to compute the loss for each batch.
    It accumulates the losses over the first `num_batches` batches (or all batches if `num_batches` is None) and returns the average.

    Args:
        data_loader (DataLoader): PyTorch DataLoader yielding (input_batch, target_batch).
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): Device for computations.
        num_batches (int, optional): Number of batches to process. If None, all batches are used.
                                     If specified value exceeds the total number of batches,
                                     it is clipped to the DataLoader length.

    Returns:
        float: The average loss over the processed batches. Returns NaN if the DataLoader is empty.
    """
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (inp_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_cost(inp_batch, target_batch, model, device)
            total_loss += loss
        else:
            break

    return total_loss / num_batches
