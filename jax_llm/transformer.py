from typing import NamedTuple

from flax import nnx
from jax import Array
from jax.typing import ArrayLike

from jax_llm.embedding import TokenAndPositionEmbedding
from jax_llm.modules import Block


class TransformerConfig(NamedTuple):
    """Configuration for the GPT model."""

    vocab_size: int
    seq_length: int
    embed_dim: int
    head_dim: int
    num_heads: int
    num_layers: int
    ff_hidden_dim: int


class Transformer(nnx.Module):
    """GPT-style decoder-only transformer model."""

    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.embedder = TokenAndPositionEmbedding(
            config.vocab_size, config.embed_dim, config.seq_length, rngs=rngs
        )
        self.layers = [
            Block(
                embed_dim=config.embed_dim,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                ff_hidden_dim=config.ff_hidden_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        self.final_layer_norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)
        self.output_proj = nnx.Linear(
            config.embed_dim, config.vocab_size, use_bias=False, rngs=rngs
        )

    def __call__(self, input_ids: ArrayLike) -> Array:
        x = self.embedder(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_layer_norm(x)
        return self.output_proj(x)
