import jax
from flax import nnx
from functools import partial


@partial(nnx.jit, static_argnums=(0, 1, 2))
def initialize_sharded_model(
    model_cls: type[nnx.Module], *args, **kwargs
) -> nnx.Module:
    """Initializer a Flax model with the weights sharded across devices.

    Reference:
        https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html#initialize-a-sharded-model
    """
    model = model_cls(*args, rngs=nnx.Rngs(0), **kwargs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model
