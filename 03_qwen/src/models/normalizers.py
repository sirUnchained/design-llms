import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    r"""
    ## RMSNorm

    This class is impelimentaion of RMSNorm variant, Normally RMSNorm uses this formula to normalize:

    $$
    \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
    $$

    where $\gamma$ is a learnable scale.

    So the main variant dose not have a `shift` parameter (more performance) but we put an option so we can include it as bias:

    $$
    \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma + \beta
    $$

    where $\beta$ is a learnable shift.

    ---

    Args
        embedding_dims (int): The embedding dimention of model.
        eps (float): Very small value to prevent division by zero.
        bias (bool): Include shift parameter or not (defalut False).
        qwen3_compatible (bool): Use `float32` precision for calculation or the model default (defalut is True).
    """

    def __init__(
        self,
        embedding_dims,
        eps=1e-6,
        bias=False,
        qwen3_compatible=True,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(embedding_dims))
        self.shift = nn.Parameter(torch.zeros(embedding_dims)) if bias else None

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        # calculate RMS
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
