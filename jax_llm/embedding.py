import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike


class TokenAndPositionEmbedding(nnx.Module):
    """Encode tokens and positions into a single embedding."""

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, rngs: nnx.Rngs):
        self.token_embedding = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        # TODO: Use rotary positional embeddings.
        self.position_embedding = nnx.Embed(max_len, embed_dim, rngs=rngs)

    def __call__(self, tokens: ArrayLike):
        token_embeddings = self.token_embedding(tokens)
        positions = jnp.arange(tokens.shape[-1])[None]
        position_embeddings = self.position_embedding(positions)
        return token_embeddings + position_embeddings
