from flax import nnx
from jax import numpy as jnp
from jax import random

from jax_llm.embedding import TokenAndPositionEmbedding


def test_token_and_position_embedding():
    # Initialize parameters
    vocab_size = 1000
    embed_dim = 256
    seq_length = 1024
    dtype = jnp.bfloat16
    batch_size = 2
    seq_len = 32

    # Create a random key
    key = random.key(0)

    # Initialize the embedding module
    embedding_module = TokenAndPositionEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_length=seq_length,
        dtype=dtype,
        rngs=nnx.Rngs(params=key),
    )

    # Create some random tokens
    tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Get embeddings
    embeddings = embedding_module(tokens)

    # Check the output shape
    assert embeddings.shape == (batch_size, seq_len, embed_dim)
