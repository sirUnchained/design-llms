import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    ## FeedForward

    This is an impelimentation of `SwiGLU` variant which uses **SiLU** (also called Swish) activation function + 3 fully connected networks.

    ---

    Args
        cfg (Qwen_configs): Qwen LLM config class.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)

        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)

        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, dtype=cfg.dtype, bias=False)

    def forward(self, x: torch.Tensor):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)

        x = nn.functional.silu(x_fc1) * x_fc2

        return self.fc3(x)
