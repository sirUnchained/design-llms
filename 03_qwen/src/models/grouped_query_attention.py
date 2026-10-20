import torch
from torch import nn

from src.models.rope import apply_rope
from src.models.normalizers import RMSNorm


class GroupedQueryAttention(nn.Module):
    """
    ## Grouped Query Attention

    We receive an input matrix X and pass it through linear layers for `query`, `key`, and `value` projections, resulting in three matrices (Q, K, V).
    We then split each matrix along the feature dimension into `num_heads` heads. For example, if the matrix dimension is 10 and `num_heads` is 2, each head gets a dimension of 5.

    Next if we are allowed to do normalization on Queries and Keys, then we normalize them using `RMSNorm`.
    For positional encoding we are going to apply RoPE on Queries and Keys as it shall be applyed on them.
    Now after encoding we'll grouping keys and values together as much as `num_kv_groups` allow us.

    Next, we compute attention scores by taking the dot product of query and key vectors.
    We apply a causal mask so this stops a token to see future tokens (if we dont do this, we'll have Data Leakage), this helps the model learn better.

    Then we'll normalize attention scores by dividing it to root of keys dimension, after this step we pass normalized attention scores to softmax function which will pass us a probability distribution.
    Finally, we compute the context vector by multiplying the attention weights with the value matrix.

    Since we transformed the input batch into multiple heads, we need to reshape (concatenate) the heads back to the original dimension.
    After that, we pass the result through a final projection linear layer and return it.

    ---

    Args:

    d_in(int):
        Input feature dimension.
    num_heads(int):
        Number of attention heads.
    num_kv_groups(int):
        Number of keys and values group.
    head_dim(int):
        Dimension of each head (default is None)
    qk_norm(bool):
        Apply normalization on keys and queries or not (default is False)
    dtype(torch.dtype):
        Parameters data type (default is None)
    """

    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ) -> None:
        super().__init__()

        assert (
            num_heads % num_kv_groups == 0
        ), f"Number of heads ({num_heads}) must be divisible by number of KV groups ({num_kv_groups})."

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert (
                d_in % num_heads == 0
            ), f"input dimension ({d_in}) must be divisible by number of heads ({num_heads})."
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = head_dim * num_heads

        self.W_query = nn.Linear(
            in_features=d_in, out_features=self.d_out, bias=False, dtype=dtype
        )
        self.W_key = nn.Linear(
            in_features=d_in,
            out_features=num_kv_groups * head_dim,
            bias=False,
            dtype=dtype,
        )
        self.W_value = nn.Linear(
            in_features=d_in,
            out_features=num_kv_groups * head_dim,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            in_features=self.d_out, out_features=d_in, bias=False, dtype=dtype
        )

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos=0,
    ) -> torch.Tensor:
        """
        `GroupedQueryAttention` forward pass.

        Args:
            x (torch.Tensor):
                Input tensor.
            mask (torch.Tensor):
                Masking tensor for masking.
            cos (torch.Tensor):
                Cosine tensor which used for RoPE.
            sin (torch.Tensor):
                Sine tensor which used for RoPE.
            start_pos (int):
                Start position of RoPE (default is 0).
        """

        batches, num_tokens, _ = x.shape

        # ==== CALCULATING QUERY, KEY, VALUE MATRICES ====
        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)

        # ==== SPLIT MATRICES INTO HEADS & GROUPING KV ====
        queries = queries.view(
            batches, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batches, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batches, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # ==== NORMALIZING QUERIES AND KEYS IF AVAILABLE ====
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # ==== APPLY ROPE ====
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys, cos, sin, offset=start_pos)

        # ==== GROUPING KV MATRICES ====
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # ==== CALCULATE ATTENTION SCORES WITH MASKING ====
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # ==== NORMALIZE ATTENTION SCORES ====
        attn_scores = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # ==== CALCULATE CONTEXT VECTORS ====
        context_vec = attn_scores @ values

        # ==== TURN BACK HEADS INTO MATRIX ====
        context = context_vec.transpose(1, 2).reshape(batches, num_tokens, self.d_out)

        # ==== CALCULATE FINAL MATRIX WITH PROJ WEIGHTS AND RETURN IT ====
        return self.out_proj(context)
