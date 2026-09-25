import os

import torch
import tiktoken

from scripts.logger import save_print_log
from scripts.lr_increase_decay import learning_rate_change
from scripts.evaluation import (
    generate_and_print_sample,
    evaluate_model,
    calc_batch_cost,
)
from configs.model_configs import GPT_configs
from src.data.pretrain_dataset.dataset import (
    create_pretrain_dataloader,
    ensure_bin_dataset,
)
from src.models.gpt_model import GPT_model
from src.utils.ckeckpoints import load_checkpoint, save_checkpoint


def pretrain_model(
    model,
    train_loader_fn,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    device,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    lr_schedule_step: int,
    checkpoint_path: str,
    checkpoint_freq: int = 1000,
    use_checkpoints=False,
    create_checkpoints=False,
):
    """
    Trains a language model over multiple epochs with periodic evaluation and checkpointing.

    The training loop iterates over batches from `train_dataloader`, computes the loss via `calc_batch_cost`, backpropagates, and updates the model weights.
    At intervals defined by `eval_freq` (global steps), the model is evaluated on the `eval_iter` batch size of training and validationdatasets, and the losses are recorded.
    After each complete epoch, a text sample is generated using `start_context` to monitor qualitative performance.

    Checkpointing is managed by the boolean parameters `use_checkpoints` and `create_checkpoints`. When resuming (`use_checkpoints=True`), the function looks for
    `latest.pt` in `checkpoint_path`. When saving (`create_checkpoints=True`), it writes periodic snapshots (`latest.pt` and epoch-named files) to the same directory.

    We save the train process in the `training-process` folder, so you can monitor it.

    Args:
        model (torch.nn.Module): The language model to be trained.
        train_loader_fn (function): A dynamic function which creates train dataloader for us.
        train_dataloader (DataLoader): DataLoader yielding training batches.
        val_dataloader (DataLoader): DataLoader yielding validation batches.
        num_epochs (int): Number of complete passes over the training data.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient-based updates.
        device (torch.device): Device (CPU or CUDA) on which to perform computations.
        eval_freq (int): Evaluate the model every N global steps.
        eval_iter (int): Placeholder for the number of batches to use during evaluation. Currently, `evaluate_model` ignores this value.
        start_context (str): Initial text prompt used for sample generation after each epoch.
        tokenizer: Tokenizer instance for converting text to token IDs and vice versa.
        checkpoint_path (str): Directory where checkpoint files are read from and written to.
        checkpoint_freq (int, optional): If `create_checkpoints` is True, save a checkpoint every N global steps. Defaults to 1000.
        use_checkpoints (bool, optional): If True, attempt to resume training from`{checkpoint_path}/latest.pt`. Defaults to False.
        create_checkpoints (bool, optional): If True, save periodic and end-of-epoch checkpoints to `checkpoint_path`. Defaults to False.

    Returns:
        tuple: A 3-element tuple containing:
            - train_losses (list): Recorded average training losses at each evaluation step.
            - val_losses (list): Recorded average validation losses at each evaluation step.
            - track_tokens_seen (list): Cumulative number of tokens processed at each evaluation step, used for plotting loss vs. tokens.
    """

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    total_steps = len(train_dataloader) * num_epochs
    start_epoch = 0
    start_index = 0

    latest_ckpt_path = os.path.join(checkpoint_path, "latest.pt")

    # Resume from checkpoint if requested and available
    if use_checkpoints and os.path.exists(latest_ckpt_path):
        checkpoint = load_checkpoint(latest_ckpt_path, model, optimizer, device)
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        tokens_seen = checkpoint["tokens_seen"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        track_tokens_seen = checkpoint["track_tokens_seen"]
        start_index = checkpoint.get("start_index", start_index)
        total_steps = checkpoint.get("total_steps", total_steps)
    elif use_checkpoints:
        print(
            f"`use_checkpoints` is true but no checkpoint found at {latest_ckpt_path}, starting fresh."
        )

    total_steps = None

    for epoch in range(start_epoch, num_epochs):
        model.train()

        # We refresh dataloader after every epoch by adding epoch number with seed
        this_start_index = start_index if epoch == start_epoch else 0
        train_dataloader = train_loader_fn(epoch, this_start_index)
        print("=" * 100)
        print(f"the starting index is now {this_start_index} ")
        print(f"total remain steps is {len(train_dataloader) * num_epochs} ")
        print("=" * 100)

        if total_steps is None:
            # we load a sample dataloader and getting smaples count.
            # So by multiply it to epochs we have total steps count.
            full_loader = train_loader_fn(epoch, 0)
            total_steps = len(full_loader) * num_epochs

        samples_done_this_epoch = this_start_index

        for step, (input_batch, target_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = calc_batch_cost(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            samples_done_this_epoch += input_batch.shape[0]

            lr = learning_rate_change(
                global_step - lr_schedule_step,
                total_steps,
                0.0,
                optimizer,
                initial_lr=2e-4,
                peak_lr=2e-4,
            )

            if global_step % eval_freq == 0:
                # Evaluate model and it the returned
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                save_print_log(
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=lr,
                )

            # Periodic checkpoint save
            if (
                create_checkpoints
                and global_step % checkpoint_freq == 0
                and global_step > 0
            ):
                save_checkpoint(
                    path=latest_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    track_tokens_seen=track_tokens_seen,
                    total_steps=total_steps,
                    start_index=samples_done_this_epoch,
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

        # End-of-epoch checkpoint save
        if create_checkpoints:
            epoch_ckpt_path = os.path.join(checkpoint_path, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                path=epoch_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                tokens_seen=tokens_seen,
                train_losses=train_losses,
                val_losses=val_losses,
                track_tokens_seen=track_tokens_seen,
                total_steps=total_steps,
                start_index=0,
            )
            # also update "latest" so resuming picks up here
            save_checkpoint(
                path=latest_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                start_index=0,
                global_step=global_step,
                tokens_seen=tokens_seen,
                train_losses=train_losses,
                val_losses=val_losses,
                track_tokens_seen=track_tokens_seen,
                total_steps=total_steps,
            )

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    cfg = GPT_configs()

    bin_path = ensure_bin_dataset(cfg)
    train_ratio = 0.80

    def train_loader_fn(epoch, start_index):
        return create_pretrain_dataloader(
            bin_path,
            start_frac=0.0,
            end_frac=train_ratio,
            batch_size=cfg.batch_size,
            max_length=cfg.context_length,
            stride=cfg.context_length,
            drop_last=True,
            shuffle=True,
            num_workers=4,
            seed=cfg.seed,
            epoch=epoch,
            start_index=start_index,
        )

    tokenizer = tiktoken.get_encoding(cfg.ticktoken_tokenizer)
    train_loader = create_pretrain_dataloader(
        bin_path,
        start_frac=0.0,
        end_frac=train_ratio,
        batch_size=cfg.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
    )
    val_loader = create_pretrain_dataloader(
        bin_path,
        start_frac=train_ratio,
        end_frac=1.0,
        shuffle=False,
        batch_size=cfg.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
    )
    epochs = 60
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(123)
    model = GPT_model(cfg)
    model.to(device=device)
    optim = torch.optim.AdamW(params=model.parameters(), lr=5e-4, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = pretrain_model(
        model,
        train_loader_fn,
        train_loader,
        val_loader,
        epochs,
        optim,
        device,
        eval_freq=5,
        eval_iter=5,
        lr_schedule_step=cfg.lr_schedule_step,
        start_context="Hello I am ",
        tokenizer=tokenizer,
        checkpoint_path=cfg.checkpoints_path,
    )
