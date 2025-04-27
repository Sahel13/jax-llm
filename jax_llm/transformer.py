import jax
import jax.numpy as jnp
from jax import Array


def dot_product_attention(queries: Array, keys: Array, values: Array) -> Array:
    """Computes dot product causal attention."""

    def large_negative_number(dtype):
        """From the jax.nn implementation."""
        dtype_max = jnp.finfo(dtype).max
        return jnp.asarray(-0.7 * dtype_max, dtype=dtype)

    logits = jnp.einsum("...QHD,...KHD->...HQK", queries, keys)
    logits /= jnp.sqrt(queries.shape[-1])

    Q = queries.shape[1]
    mask = jnp.tril(jnp.ones((Q, Q), dtype=jnp.bool_))
    logits = jnp.where(mask, logits, large_negative_number(logits.dtype))
    probs = jax.nn.softmax(logits)

    return jnp.einsum("...HQK,...KHD->...QHD", probs, values)
