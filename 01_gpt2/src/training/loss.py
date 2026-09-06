import torch
import math


def calc_perplexity(loss) -> float:
    """
    ## Convert a cross-entropy loss value into perplexity.

    Perplexity = exp(cross-entropy loss). It's the standard human-readable
    evaluation metric for language models: roughly "how many tokens, on
    average, was the model choosing between" at each step. Lower is better.

    Note: if you're using `z_loss_coeff` > 0 in `calc_batch_cost`, pass the
    plain cross-entropy component here (not the combined loss), since the
    z-loss term isn't part of the probabilistic interpretation perplexity relies on.

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

    The input and target batches are moved to the specified device, then passed through
    the model to obtain logits. The loss is computed using `torch.nn.functional.cross_entropy`
    after flattening the logits and targets to shape (batch_size * seq_len, num_classes).

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

    This function iterates over the DataLoader and uses `calc_batch_cost` to compute the
    loss for each batch. It accumulates the losses over the first `num_batches` batches
    (or all batches if `num_batches` is None) and returns the average.

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
