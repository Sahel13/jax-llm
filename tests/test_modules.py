import os

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import pytest
from jax import numpy as jnp
from jax import random
from jax.sharding import NamedSharding, PartitionSpec as P

from jax_llm.modules import Block, CausalSelfAttention, dot_product_attention
from jax_llm.utils import initialize_sharded_model_factory

# Skip tests if 4 devices are not available
pytestmark = pytest.mark.skipif(
    jax.device_count() != 4,
    reason="4 devices are not available",
)


@pytest.fixture
def params():
    """Common model parameters for tests."""
    mesh = jax.make_mesh((2, 2), ("fsdp", "tp"))
    input_sharding = NamedSharding(mesh, P("fsdp", None, "tp"))
    return {
        "batch_size": 8,
        "seq_len": 10,
        "num_heads": 4,
        "head_dim": 16,
        "embed_dim": 8,
        "ff_hidden_dim": 32,
        "mesh": mesh,
        "input_sharding": input_sharding,
        "dtype": jnp.bfloat16,
    }


@pytest.fixture
def model_inputs(params):
    """Generate input tensors for model testing."""
    key = random.key(0)
    x = random.normal(
        key, (params["batch_size"], params["seq_len"], params["embed_dim"])
    )
    return jax.device_put(x, params["input_sharding"])


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
    assert jnp.allclose(output, ref_output, rtol=1e-5, atol=1e-8)


def test_causal_self_attention(params, model_inputs):
    initializer = initialize_sharded_model_factory(
        CausalSelfAttention,
        params["embed_dim"],
        params["head_dim"],
        params["num_heads"],
        dtype=params["dtype"],
    )
    with params["mesh"]:
        model = initializer()
        output = model(model_inputs)

    assert output.shape == model_inputs.shape


def test_block(params, model_inputs):
    initializer = initialize_sharded_model_factory(
        Block,
        params["embed_dim"],
        params["head_dim"],
        params["num_heads"],
        params["ff_hidden_dim"],
        dtype=params["dtype"],
    )
    with params["mesh"]:
        model = initializer()
        output = model(model_inputs)

    assert output.shape == model_inputs.shape
    assert output.sharding == params["input_sharding"]
