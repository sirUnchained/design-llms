from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    """
    ## Multi Head Attention

    We receive an input matrix X and pass it through linear layers for `query`, `key`, and `value` projections, resulting in three matrices (Q, K, V).
    We then split each matrix along the feature dimension into `n_heads` heads. For example, if the matrix dimension is 10 and `n_heads` is 2, each head gets a dimension of 5.

    Next, we compute attention scores by taking the dot product of query and key vectors.
    We apply a causal mask (if needed) to prevent the model from attending to future tokens, which helps the model learn better.
    The attention scores are then normalized using the softmax function, and dropout is applied to the normalized scores to produce the final attention weights.

    Finally, we compute the context vector by multiplying the attention weights with the value matrix.

    Since we transformed the input batch into multiple heads, we need to reshape (concatenate) the heads back to the original dimension.
    After that, we pass the result through a final projection linear layer and return it.

    ---

    Args:
        d_in (int):
            Input feature dimension.
        d_out (int):
            Output feature dimension.
        context_length (int):
            The maximum number of tokens the model can attend to (context window size).
        dropout (float):
            Dropout rate applied after softmax.
        n_heads (int):
            Number of attention heads.
        qkv_bias (bool):
            If True, adds a learnable bias to the query, key, and value linear layers. Default is False.

    Returns:
        torch.Tensor: The output tensor after multi-head attention and final projection.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        n_heads: int,
        qkv_bias=False,
    ) -> None:
        super().__init__()

        assert d_out % n_heads == 0, "output dim is not divisible by number of heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.heads_dim = d_out // n_heads
        self.w_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.w_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.w_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        self.w_proj = nn.Linear(in_features=d_out, out_features=d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert (
            len(x.shape) == 3
        ), f"The input data dimention is not 3  it is {len(x.shape)} please check you're data"

        batches, num_tokens, token_dims = x.shape

        # ==== CALCULATING QUERY, KEY, VECTOR MATRIXES ====
        queries: torch.Tensor = self.w_query(x)
        keys: torch.Tensor = self.w_key(x)
        values: torch.Tensor = self.w_value(x)

        # ==== SPLIT MATRIXES INTO HEADS ====
        queries = queries.view(batches, num_tokens, self.n_heads, self.heads_dim)
        keys = keys.view(batches, num_tokens, self.n_heads, self.heads_dim)
        values = values.view(batches, num_tokens, self.n_heads, self.heads_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # ==== ATTENTION SCORES WITH MASKING ====
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # ==== NORMALIZE ATTENTION SCORES WITH DROPOUT ====
        attn_norm = torch.softmax(
            attn_scores / (torch.sqrt(torch.tensor(keys.shape[-1]))), dim=-1
        )
        attn_norm = self.dropout(attn_norm)

        # ==== CALCULATE CONTEXT VECTORES ====
        ctx_vec = attn_norm @ values

        # ==== TURN BACK HEADS INTO MATRIX ====
        ctx_vec = ctx_vec.transpose(1, 2)
        ctx_vec = ctx_vec.contiguous().view(batches, num_tokens, self.d_out)

        # ==== CALCULATE FINAL MATRIX WITH PROJ WEIGTHS AND RETURN IT ====
        ctx_vec = self.w_proj(ctx_vec)
        return ctx_vec


if __name__ == "__main__":
    inp = torch.tensor(
        [
            [
                [0.7270, 0.3990, 0.2379, 0.9915, 0.7636, 0.4507],
                [0.9949, 0.8612, 0.9033, 0.9265, 0.1857, 0.4383],
                [0.7735, 0.5923, 0.7120, 0.4057, 0.9538, 0.5655],
                [0.2860, 0.2608, 0.9973, 0.0295, 0.3294, 0.7030],
                [0.6911, 0.9787, 0.6241, 0.1684, 0.0759, 0.4814],
                [0.9448, 0.6401, 0.6554, 0.2093, 0.8803, 0.5675],
                [0.1347, 0.7257, 0.6005, 0.9069, 0.5066, 0.5869],
                [0.4809, 0.7310, 0.5968, 0.3790, 0.5758, 0.1221],
                [0.2617, 0.6890, 0.3275, 0.7216, 0.7093, 0.8472],
            ],
            [
                [0.4759, 0.3368, 0.4943, 0.1596, 0.9671, 0.0694],
                [0.6195, 0.1539, 0.7912, 0.0565, 0.9365, 0.0819],
                [0.1328, 0.0304, 0.9196, 0.0793, 0.8204, 0.5018],
                [0.5039, 0.1384, 0.7975, 0.7329, 0.4971, 0.5675],
                [0.8634, 0.8833, 0.9917, 0.9412, 0.3009, 0.8806],
                [0.0661, 0.4826, 0.4121, 0.5254, 0.1280, 0.0554],
                [0.2619, 0.3421, 0.7092, 0.0851, 0.6465, 0.5629],
                [0.0724, 0.1846, 0.0766, 0.8108, 0.2997, 0.2209],
                [0.3933, 0.8057, 0.7022, 0.6170, 0.8729, 0.9253],
            ],
            [
                [0.2158, 0.6011, 0.5714, 0.9913, 0.4278, 0.9706],
                [0.2396, 0.4917, 0.4233, 0.5869, 0.9952, 0.9553],
                [0.3150, 0.4534, 0.2172, 0.5958, 0.8819, 0.1767],
                [0.8788, 0.3910, 0.8415, 0.7266, 0.6024, 0.3362],
                [0.4571, 0.9515, 0.1001, 0.8459, 0.8435, 0.9129],
                [0.6346, 0.6501, 0.8478, 0.5069, 0.0469, 0.6340],
                [0.8114, 0.5475, 0.4792, 0.4817, 0.0899, 0.2053],
                [0.6352, 0.2238, 0.9391, 0.6520, 0.1547, 0.3119],
                [0.6519, 0.9865, 0.2958, 0.3414, 0.1299, 0.3648],
            ],
        ]
    )

    # inp = torch.rand((3, 9, 6))

    mha = MultiHeadAttention(
        d_in=6, d_out=6, context_length=10, dropout=0.1, n_heads=2, qkv_bias=True
    )

    print(mha(inp))
