import torch
from torch import nn


class Gelu(nn.Module):
    r"""
    ## GELU Activation Function

    GELU (Gaussian Error Linear Unit) is defined by the approximation:

    $$
    \text{GELU}(x) \approx 0.5 \, x \left( 1 + \tanh\left[ \sqrt{\frac{2}{\pi}} \left( x + 0.044715\, x^3 \right) \right] \right)
    $$

    We choose GELU over ReLU because it has no sharp corners, providing smoother gradients and often better performance in deep learning models.

    ---

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output after applying GELU activation.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
