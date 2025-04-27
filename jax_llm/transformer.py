from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike

from jax_llm.embedding import TokenAndPositionEmbedding


class FeedForward(nnx.Module):
    """Feedforward neural network with GeGLU activation."""

    def __init__(self, features: int, hidden_dim: int, rngs: nnx.Rngs):
        # TODO: Check which initialization to use.
        self.gate_proj = nnx.Linear(features, hidden_dim, rngs=rngs)
        self.up_proj = nnx.Linear(features, hidden_dim, rngs=rngs)
        self.down_proj = nnx.Linear(hidden_dim, features, rngs=rngs)

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

    logits = jnp.einsum("...QHD,...KHD->...HQK", queries, keys)
    logits /= jnp.sqrt(queries.shape[-1])

    Q = queries.shape[1]
    mask = jnp.tril(jnp.ones((Q, Q), dtype=jnp.bool_))
    logits = jnp.where(mask, logits, large_negative_number(logits.dtype))
    probs = jax.nn.softmax(logits)

    return jnp.einsum("...HQK,...KHD->...QHD", probs, values)


class CausalSelfAttention(nnx.Module):
    def __init__(self, embed_dim: int, head_dim: int, num_heads: int, rngs: nnx.Rngs):
        self.qkv_proj = nnx.LinearGeneral(
            in_features=embed_dim, out_features=(num_heads, 3 * head_dim), rngs=rngs
        )
        self.output_proj = nnx.LinearGeneral(
            in_features=(num_heads, head_dim),
            out_features=embed_dim,
            axis=(-2, -1),
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> Array:
        queries, keys, values = self.qkv_proj(x).split(3, axis=-1)
        outputs = dot_product_attention(queries, keys, values)
        return self.output_proj(outputs)


class Block(nnx.Module):
    """Transformer block with attention and feedforward layers."""

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        self.attention = CausalSelfAttention(embed_dim, head_dim, num_heads, rngs)
        self.ff = FeedForward(embed_dim, ff_hidden_dim, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(embed_dim)
        self.layer_norm2 = nnx.LayerNorm(embed_dim)

    def __call__(self, x: ArrayLike) -> Array:
        x += self.attention(self.layer_norm1(x))
        x += self.ff(self.layer_norm2(x))
        return x


class TransformerConfig(NamedTuple):
    """Configuration for the GPT model."""

    vocab_size: int
    max_length: int
    embed_dim: int
    head_dim: int
    num_heads: int
    num_layers: int
    ff_hidden_dim: int


class Transformer(nnx.Module):
    """GPT-style decoder-only transformer model."""

    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.embedder = TokenAndPositionEmbedding(
            config.vocab_size, config.embed_dim, config.max_length, rngs=rngs
        )
        self.blocks = [
            Block(
                embed_dim=config.embed_dim,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                ff_hidden_dim=config.ff_hidden_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        self.final_layer_norm = nnx.LayerNorm(config.embed_dim)
        self.output_proj = nnx.Linear(
            config.embed_dim, config.vocab_size, use_bias=False, rngs=rngs
        )

    def __call__(self, input_ids: ArrayLike) -> Array:
        x = self.embedder(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_layer_norm(x)
        return self.output_proj(x)
