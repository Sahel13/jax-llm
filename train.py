import os

# See https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md
# for details on these environment variables.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true"
)

import time
from dataclasses import dataclass

import grain.python as pygrain
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import tiktoken
from flax import nnx
from jax import Array
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jax_llm.transformer import Transformer, TransformerConfig
from jax_llm.utils import initialize_sharded_model_factory


@dataclass
class TextDataset:
    data: list
    maxlen: int
    tokenizer: tiktoken.Encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer.encode(
            self.data[idx], allowed_special={"<|endoftext|>"}
        )[: self.maxlen]
        return encoding + [0] * (self.maxlen - len(encoding))  # Pad to maxlen


def load_and_preprocess_data(
    file_path: str,
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    maxlen: int,
    num_epochs: int,
) -> pygrain.DataLoader:
    with open(file_path, "r") as f:
        text = f.read()

    stories = text.split("<|endoftext|>")
    stories = [story + "<|endoftext|>" for story in stories if story.strip()]
    df = pd.DataFrame({"text": stories})
    data = df["text"].dropna().tolist()
    dataset = TextDataset(data, maxlen, tokenizer)

    sampler = pygrain.IndexSampler(
        len(dataset),
        shuffle=False,
        seed=42,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )

    return pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
    )


def loss_fn(model: nnx.Module, batch: tuple[Array, Array]) -> tuple[Array, Array]:
    logits = model(batch[0]).astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, batch: tuple[Array, Array]
) -> Array:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss


if __name__ == "__main__":
    # ------------- Hyperparameters ---------- #

    # Set the mesh for sharding
    mesh = jax.make_mesh((4, 1), ("fsdp", "tp"))

    model_config = TransformerConfig(
        # From the MiniGPT example: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining
        vocab_size=50257,
        seq_length=256,
        embed_dim=256,
        head_dim=64,
        num_heads=8,
        num_layers=8,
        ff_hidden_dim=256,
    )

    batch_size = 1024
    num_epochs = 1

    # ------------- Initialize model and optimizer ---------- #
    initializer = initialize_sharded_model_factory(Transformer, model_config)
    with mesh:
        model = initializer()

        # # Print model summary
        # print(
        #     nnx.tabulate(
        #         model, jnp.ones((batch_size, model_config.seq_length), dtype=jnp.int32)
        #     )
        # )

    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    # ------------- Load data ---------- #
    tokenizer = tiktoken.get_encoding("gpt2")
    text_dl = load_and_preprocess_data(
        "data/TinyStoriesV2-GPT4-train.txt",
        tokenizer,
        batch_size,
        model_config.seq_length,
        num_epochs,
    )

    prep_target_batch = jax.jit(
        jax.vmap(lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0]))))
    )

    model_params = nnx.state(model, nnx.Param)
    model_size = sum(x.size for x in jax.tree.leaves(model_params))
    print(f"Model size: {model_size / 1e6:.2f}M parameters")

    # Approximate FLOPs in a training step.
    # This ignores the self-attention layer.
    batch_size_in_tokens = batch_size * model_config.seq_length
    approx_flops = 6 * model_size * batch_size_in_tokens
    flops_per_device = approx_flops / jax.device_count()

    print_every_n_steps = 10
    # jax.profiler.start_trace("tmp/tensorboard")

    # ------------- Training loop ---------- #
    step = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch in text_dl:
            if len(batch) % jax.device_count() != 0:
                continue  # skip the remaining elements

            input_batch = jnp.array(batch).T
            input_batch = jax.device_put(
                input_batch, NamedSharding(mesh, P("fsdp", None))
            )
            target_batch = prep_target_batch(input_batch)

            with mesh:
                loss = train_step(model, optimizer, (input_batch, target_batch))

            if (step + 1) % print_every_n_steps == 0:
                elapsed_time = time.time() - start_time

                # Print performance metrics
                num_tokens_processed = print_every_n_steps * batch_size_in_tokens
                tokens_per_second = num_tokens_processed / elapsed_time

                print(
                    f"Step {step + 1}, Loss: {loss.item():6.3f}, "
                    f"Tokens/sec: {tokens_per_second:12.2f}, "
                    f"TFLOPS/device: {flops_per_device * print_every_n_steps / (1e12 * elapsed_time):6.2f}"
                )
                start_time = time.time()

            step += 1

            # if step == 100:
            #     jax.profiler.stop_trace()
            #     break
