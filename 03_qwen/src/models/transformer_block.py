from torch import nn

from src.models.grouped_query_attention import GroupedQueryAttention
from src.models.feed_forward import FeedForward
from src.models.normalizers import RMSNorm


class TransformerBlock(nn.Module):
    """
    ## Transformer Block

    This is the core concept of LLMs. For the first part of this block we are going to use Grouped Query Attention like this:

    1. We first get backup from input data for skip connections.
    2. Now the data will be passed from a layer normalizer.
    3. Data will passed into Grouped Query Attention module.
    4. And finally we just caluclate sum of the bakuped data and current data.

    In second part of this block, we use FeedForward module:
    1. We first get backup from input data for skip connections.
    2. Now the data will be passed from a layer normalizer.
    3. Data will passed into Feed Forward module.
    4. And finally we just caluclate sum of the bakuped data and current data.

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

        self.group_attn = GroupedQueryAttention(
            d_in=cfg.emb_dim,
            num_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            num_kv_groups=cfg.n_kv_groups,
            qk_norm=cfg.qk_norm,
            dtype=cfg.dtype,
        )

        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.emb_dim, eps=1e-6)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0):
        shortcut = x
        x = self.norm1(x)
        x = self.group_attn(x, mask, cos, sin, start_pos=start_pos)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x
