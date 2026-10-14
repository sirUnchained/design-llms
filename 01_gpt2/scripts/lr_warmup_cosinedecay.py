import torch

import math


def learning_rate_change(
    global_step: int,
    total_training_steps: int,
    warmup_percent: float,
    optimizer: torch.optim.Optimizer,
    initial_lr=4e-4,
    peak_lr=4e-4,
) -> float:
    """
    ## Update the optimizer's learning rate using a warmup then cosine decay schedule.

    During the warmup phase (first `warmup_percent` fraction of total steps), the learning rate increases linearly from `initial_lr` to `peak_lr`.
    After warmup, the learning rate decays following a cosine curve from `peak_lr` down to `min_lr = 0.1 * initial_lr` over the remaining steps.

    Args:
        global_step (int): Current training step (0‑based).
        total_training_steps (int): Total number of training steps.
        warmup_percent (float): Fraction of total steps used for warmup (e.g., 0.1 for 10%).
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be updated.
        initial_lr (float, optional): Starting learning rate before warmup. Default 1e-6.
        peak_lr (float, optional): Maximum learning rate reached at the end of warmup. Default 3e-4.

    Returns:
        float: The updated learning rate value (from the first parameter group).
    """

    warmup_steps = int(warmup_percent * total_training_steps)
    lr_increment = (peak_lr - initial_lr) / warmup_steps if warmup_steps > 0 else 0
    min_lr = 0.1 * peak_lr

    if global_step < warmup_steps:
        # Linear increase
        progress = global_step / max(warmup_steps, 1)
        lr = initial_lr + progress * lr_increment
    else:
        # Cosine decay
        decay_steps = total_training_steps - warmup_steps
        progress = (global_step - warmup_steps) / decay_steps

        progress = min(progress, 1.0)  # guard against overshoot

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (peak_lr - min_lr) * cosine_decay

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
