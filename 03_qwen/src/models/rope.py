import torch


def compute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32
):
    r"""
    ## Compute RoPE Params

    This function will first calculate Angular frequency using this formula:
    $$
    \omega_i=\frac{1}{\theta_{\text{base}}^{2i/d}}
    $$

    > **Note** that we split the head_dims in half.

    Then we multiply each frequency with the positions which are build as `context_length`, this
    Will gave us the angles and we concatinate these angles for making it as size as `head_dim`
    because as you remember we had split it.

    finally we calculate cosine and sine and return them.

    ---

    Args:
        head_dim (int): The dimension of you're heads.
        theta_base (int): theta base for the angular frequency formula (default is 10_000).
        context_length (int): you're model context length (default is 4096).
        dtype (torch.dtype): model's dtype (default is torch.float32).
    """

    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # define the Angular frequency using $\omega_i=\frac{1}{\theta_{\text{base}}^{2i/d}}$ formula
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )

    # Simply create some positions from 0 to m
    positions = torch.arange(context_length, dtype=dtype)

    # Remember: `positions[:, None]` will temporary reshape the pytorch array so if array shape
    # was (4,) now it is (4, 1). If you write `inv_freq[None, :]` the shape (3,) will be (1, 3).
    # The matrix multipication of shape (4, 1) and (1, 3) will be (4, 3).
    angles = positions[:, None] * inv_freq[None, :]

    # Why we duplicate angles 2 times? Look at `inv_freq`, it has been calculated as half of `head_dim`
    # so we are going to duplicate angles size, and we concatinate them as **column index (dim=1)** so
    # the `head_dim` will be as same as before
    angles = torch.cat([angles, angles], dim=1)

    # calculate cosine and sine and return them
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset=0):
    r"""
    ## Apply Rope

    This functino is depended on `Compute RoPE Params` functino which you should already impeliment.

    This function uses **Two-halves form (split-half)** approach for impeliment the RoPE, so we first split `x` tensor (input)
    in 2 tensors (named `x1` and `x2`), we then unsqueeze input `cos` and `sin` because if we dont do this the shape of them is (seq_len, head_dims) while
    `x` shape is (batch_size, num_heads, seq_len, head_dim) and by doing it the `cos` and `sin` shape will be (1, 1, seq_len, head_dims).

    we then negine the `x2` and concat it with `x1` after that we apply rotation formula:

    $$
    \boxed{\mathbf{x}'=\mathbf{x}\odot\cos(\boldsymbol{\theta})+\operatorname{rotate}(\mathbf{x}\odot\sin(\boldsymbol{\theta})}
    $$

    and setting model precision to what we define, why? we may use AMP (BF16) but we need it to calculate as float32

    ---

    Args:
        x (torch.Tensor): The input tensor of model.
        cos (torch.Tensor): The cosine matrix.
        sin (torch.Tensor): The sine matrix.
        offset (int): the offset number (default is 0)
    """

    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # simply just split x tensor into two halfs
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # cosine and sine has shape like (seq_len, head_dim), while this is not same as
    # x shape which is (batch_size, num_heads, seq_len, head_dim), to do this we call
    # `.unsqueeze(0)` 2 times so it adds 2 other dimensinos so it looks like (1, 1, seq_len, head_dim).
    cos = cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # In here we turning second half to negative, and then
    # creating 2 pairs (like what I talk about for **Two-halves form**)
    rotated = torch.cat((-x2, x1), dim=-1)

    # This is now the dot product of 2 vectors will be calculated.
    # This is the core idea of this approach. You may ask
    # **This is not same as what we have talked above, why?** Because the book
    # impelimented **Two-halves form** version of RoPE which we talked about.
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)
