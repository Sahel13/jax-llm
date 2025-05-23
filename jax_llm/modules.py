import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike


class FeedForward(nnx.Module):
    """Feedforward neural network with GeGLU activation."""

    def __init__(
        self, features: int, hidden_dim: int, *, dtype: jnp.dtype, rngs: nnx.Rngs
    ):
        # TODO: Check which initialization to use.
        self.gate_proj = nnx.Linear(features, hidden_dim, dtype=dtype, rngs=rngs)
        self.up_proj = nnx.Linear(features, hidden_dim, dtype=dtype, rngs=rngs)
        self.down_proj = nnx.Linear(hidden_dim, features, dtype=dtype, rngs=rngs)

    def __call__(self, x: ArrayLike) -> Array:
        ff_gate = self.gate_proj(x)
        gate_value = nnx.gelu(ff_gate)

        ff_out = self.up_proj(x)
        activations = gate_value * ff_out

        return self.down_proj(activations)


def dot_product_attention(queries: Array, keys: Array, values: Array) -> Array:
    """Computes dot product causal attention."""

    def large_negative_number(dtype):
        """From the jax.nn implementation."""
        dtype_max = jnp.finfo(dtype).max
        return jnp.asarray(-0.7 * dtype_max, dtype=dtype)

    dtype = queries.dtype

    # Dot product and softmax are done in float32 to avoid overflow.
    # Reference: https://arxiv.org/abs/2312.02696
    logits = jnp.einsum(
        "...QHD,...KHD->...HQK", queries.astype(jnp.float32), keys.astype(jnp.float32)
    )
    logits /= jnp.sqrt(queries.shape[-1])

    Q = queries.shape[1]
    mask = jnp.tril(jnp.ones((Q, Q), dtype=jnp.bool_))
    logits = jnp.where(mask, logits, large_negative_number(logits.dtype))
    probs = jax.nn.softmax(logits).astype(dtype)

    return jnp.einsum("...HQK,...KHD->...QHD", probs, values)


class CausalSelfAttention(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        num_heads: int,
        *,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.qkv_proj = nnx.LinearGeneral(
            embed_dim, (num_heads, 3 * head_dim), dtype=dtype, rngs=rngs
        )
        self.output_proj = nnx.LinearGeneral(
            (num_heads, head_dim),
            embed_dim,
            axis=(-2, -1),
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> Array:
        queries, keys, values = jnp.split(self.qkv_proj(x), 3, axis=-1)
        outputs = dot_product_attention(queries, keys, values)
        return self.output_proj(outputs)


class Block(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        *,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.attention = CausalSelfAttention(
            embed_dim, head_dim, num_heads, dtype=dtype, rngs=rngs
        )
        self.ff = FeedForward(embed_dim, ff_hidden_dim, dtype=dtype, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(embed_dim, dtype=dtype, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(embed_dim, dtype=dtype, rngs=rngs)

    def __call__(self, x: ArrayLike) -> Array:
        x += self.attention(self.layer_norm1(x))
        x += self.ff(self.layer_norm2(x))
        return x
