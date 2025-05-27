from collections.abc import Callable

import jax
from flax import nnx


def initialize_sharded_model_factory(
    model_cls: type[nnx.Module], *args, **kwargs
) -> Callable:
    """Returns an initializer for a sharded Flax model.

    Reference:
        https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html#initialize-a-sharded-model
    """

    @nnx.jit
    def initializer():
        # TODO: Support passing a seed for the rngs.
        model = model_cls(*args, rngs=nnx.Rngs(0), **kwargs)
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)
        return model

    return initializer
