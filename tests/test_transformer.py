import jax
import pytest
from jax import numpy as jnp
from jax import random

from jax_llm.transformer import dot_product_attention


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_dot_product_attention(seed):
    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_dim = 16

    key = random.key(seed)
    key_q, key_k, key_v = random.split(key, 3)
    queries = random.normal(key_q, (batch_size, seq_len, num_heads, head_dim))
    keys = random.normal(key_k, (batch_size, seq_len, num_heads, head_dim))
    values = random.normal(key_v, (batch_size, seq_len, num_heads, head_dim))

    our_output = dot_product_attention(queries, keys, values)
    jax_output = jax.nn.dot_product_attention(
        query=queries, key=keys, value=values, is_causal=True
    )

    assert our_output.shape == jax_output.shape
    assert jnp.allclose(our_output, jax_output, atol=1e-5), (
        f"Max difference for seed {seed}: {jnp.abs(our_output - jax_output).max()}"
    )
