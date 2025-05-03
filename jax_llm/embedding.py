import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike


class TokenAndPositionEmbedding(nnx.Module):
    """Encode tokens and positions into a single embedding."""

    def __init__(
        self, vocab_size: int, embed_dim: int, seq_length: int, rngs: nnx.Rngs
    ):
        self.token_embedding = nnx.Embed(
            vocab_size, embed_dim, dtype=jnp.bfloat16, rngs=rngs
        )
        # TODO: Use rotary positional embeddings.
        self.position_embedding = nnx.Embed(
            seq_length, embed_dim, dtype=jnp.bfloat16, rngs=rngs
        )

    def __call__(self, tokens: ArrayLike) -> Array:
        token_embeddings = self.token_embedding(tokens)
        positions = jnp.arange(tokens.shape[-1])[None]
        position_embeddings = self.position_embedding(positions)
        return token_embeddings + position_embeddings
