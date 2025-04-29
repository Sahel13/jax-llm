import jax
import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from jax_llm.modules import Block, CausalSelfAttention, dot_product_attention


@pytest.fixture
def params():
    """Common model parameters for tests."""
    return {
        "batch_size": 2,
        "seq_len": 10,
        "num_heads": 4,
        "head_dim": 16,
        "embed_dim": 32,
        "ff_hidden_dim": 128,
    }


@pytest.fixture
def model_inputs(params):
    """Generate input tensors for model testing."""
    key = random.key(0)
    x = random.normal(
        key, (params["batch_size"], params["seq_len"], params["embed_dim"])
    )
    return x


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_dot_product_attention(seed, params):
    qkvs = random.normal(
        random.key(seed),
        (
            params["batch_size"],
            params["seq_len"],
            params["num_heads"],
            3 * params["head_dim"],
        ),
    )
    queries, keys, values = jnp.split(qkvs, 3, axis=-1)

    output = dot_product_attention(queries, keys, values)
    ref_output = jax.nn.dot_product_attention(
        query=queries, key=keys, value=values, is_causal=True
    )

    assert output.shape == ref_output.shape
    assert jnp.allclose(output, ref_output)


def test_causal_self_attention(params, model_inputs):
    output = CausalSelfAttention(
        params["embed_dim"], params["head_dim"], params["num_heads"], rngs=nnx.Rngs(0)
    )(model_inputs)

    assert output.shape == model_inputs.shape


def test_block(params, model_inputs):
    output = Block(
        params["embed_dim"],
        params["head_dim"],
        params["num_heads"],
        params["ff_hidden_dim"],
        rngs=nnx.Rngs(0),
    )(model_inputs)

    assert output.shape == model_inputs.shape
