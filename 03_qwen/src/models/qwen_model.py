import torch

from src.models.transformer_block import TransformerBlock
from src.models.normalizers import RMSNorm
from src.models.rope import compute_rope_params


class Qwen3Model(torch.nn.Module):
    """
    ## An impelimentation of Qwen language model.

    This implementation includes positional encoding with RoPE (on Queries and Keys),
    a stack of transformer blocks, RMSNorm normalization, and an output
    linear head that projects to vocabulary size.

    ---

    Args:
        cfg (Qwen_configs): Configuration object containing model hyperparameters:
            - emb_dim (int): Embedding dimension.
            - num_kv_groups (int): Number of keys and values groups.
            - qk_norm (bool): To do query and key normalization or not.
            - num_heads (int): Number of heads.
    """

    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = torch.nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        self.trf_blocks = torch.nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = torch.nn.Linear(
            cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

        # Reusable utilities
        if cfg.head_dim is None:
            head_dim = cfg.emb_dim // cfg.n_heads
        else:
            head_dim = cfg.head_dim

        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
        )

        ## We cannot convert normal tensors device to other devices, so we use register_buffer insted
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.cfg = cfg

        # I dont remember qwen3 uses weight tying but any way i dont have also enaugh hardware for this
        # So I'll use it
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]

        pos_start = 0  # Not strictly necessary but helps torch.compile
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        mask = mask[None, None, :, :]

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin, start_pos=pos_start)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg.dtype))
        return logits
