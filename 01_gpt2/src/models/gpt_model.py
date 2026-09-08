import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchinfo import summary

from configs.model_configs import GPT_configs
from .transformer_block import TransformerBlock
from .normalizers import LayerNormalizer


class GPT_model(nn.Module):
    """
    ## A Generative Pre-trained Transformer (GPT) language model.

    This implementation includes token and positional embeddings, dropout,
    a stack of transformer blocks, final layer normalization, and an output
    linear head that projects to vocabulary size.

    Args:
        cfg (GPT_configs): Configuration object containing model hyperparameters:
            - vocab_size (int): Size of the vocabulary.
            - emb_dim (int): Embedding dimension.
            - context_length (int): Maximum sequence length (context window).
            - drop_rate (float): Dropout probability.
            - n_layers (int): Number of transformer blocks.
            - qkv_bias (bool): Whether to use bias in attention linear layers.
        use_checkpointing (bool): If set True then we use checkpoint the transformer
            Blocks so it can help us to save GPU memory but with slower training so
            we set default to False.
    """

    def __init__(self, cfg: GPT_configs, use_checkpointing: bool = False) -> None:
        super().__init__()

        self.use_checkpointing = use_checkpointing

        self.tok_emb = nn.Embedding(
            num_embeddings=cfg.vocab_size, embedding_dim=cfg.emb_dim
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=cfg.context_length, embedding_dim=cfg.emb_dim
        )

        self.dropout = nn.Dropout(p=cfg.drop_rate)

        # ModuleList instead of Sequential — we need to loop manually to
        # conditionally checkpoint each block.
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNormalizer(emb_dim=cfg.emb_dim)

        self.out_head = nn.Linear(
            in_features=cfg.emb_dim, out_features=cfg.vocab_size, bias=False
        )

        # Important: Real GPT-2 ties the token embedding matrix (wte) and the output projection
        # (lm_head), they literally share the same weight tensor.
        # So if you don't uncomment code below, your model will be larger than what it was in the main paper.
        self.out_head.weight = self.tok_emb.weight

    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT model.

        Args:
            input_sequences (torch.Tensor): Token indices of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        batches, seq_len = input_sequences.shape

        # ==== GIVE THE EMBEDDING LAYER OUR SEQUENCES ====
        tok_embeddings = self.tok_emb(input_sequences)

        # ==== ADD POSITIONAL EMBEDDINGS ====
        pos_embeddings = self.pos_emb(
            torch.arange(seq_len, device=input_sequences.device)
        )
        x = tok_embeddings + pos_embeddings

        # ==== APPLY DROPOUT ====
        x = self.dropout(x)

        # ==== GIVE OUR EMBEDDINGS TO TRANSFORMER BLOCKS ====
        # Note that here we are using checkpointing
        for block in self.transformer_blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # ==== APPLY FINAL LAYER NORMALIZER ====
        x = self.final_norm(x)

        # ==== APPLY FINAL LINEAR LAYER ====
        logits = self.out_head(x)

        return logits


if __name__ == "__main__":
    cfg = GPT_configs()

    gpt = GPT_model(cfg)
    data = torch.rand((3, 12)).type(torch.long)

    print(f"model generated data in this shape: {gpt(data).shape}")
    print("model architecture summary:")
    summary(gpt)
