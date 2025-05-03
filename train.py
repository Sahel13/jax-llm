import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
from dataclasses import dataclass

import grain.python as pygrain
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import tiktoken
from flax import nnx

from jax_llm.transformer import Transformer, TransformerConfig


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


def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.astype(jnp.float32), labels=batch[1]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(grads)


if __name__ == "__main__":
    # ------------- Hyperparameters ---------- #

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

    batch_size = 256
    num_epochs = 1

    # ------------- Initialize model and optimizer ---------- #

    model = Transformer(model_config, nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    # # ------------- Print model summary ---------- #
    # print(
    #     nnx.tabulate(
    #         model, jnp.ones((batch_size, model_config.seq_length), dtype=jnp.int32)
    #     )
    # )

    # ------------- Load data ---------- #
    tokenizer = tiktoken.get_encoding("gpt2")
    text_dl = load_and_preprocess_data(
        "data/TinyStoriesV2-GPT4-train.txt",
        tokenizer,
        batch_size,
        model_config.seq_length,
        num_epochs,
    )

    # start_prompt = "Once upon a time"
    # start_tokens = tokenizer.encode(start_prompt)[: model_config.max_length]
    # generated_text = model.generate_text(model_config.max_length, start_tokens)
    # print(f"Initial generated text:\n{generated_text}\n")

    metrics_history = {
        "train_loss": [],
    }

    prep_target_batch = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
    )

    batch_size_in_tokens = batch_size * model_config.seq_length
    model_params = nnx.state(model, nnx.Param)
    model_size = sum(x.size for x in jax.tree.leaves(model_params))
    print(f"Model size: {model_size / 1e6:.2f}M parameters")

    # Approximate FLOPs in a training step.
    # This ignore the self-attention layer.
    approx_flops = 6 * model_size * batch_size_in_tokens
    flops_per_device = approx_flops / jax.device_count()

    # ------------- Training loop ---------- #
    step = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch in text_dl:
            if len(batch) % len(jax.devices()) != 0:
                continue  # skip the remaining elements
            input_batch = jnp.array(batch).T
            target_batch = prep_target_batch(input_batch)
            train_step(model, optimizer, metrics, (input_batch, target_batch))

            if (step + 1) % 200 == 0:
                elapsed_time = time.time() - start_time

                for metric, value in metrics.compute().items():
                    metrics_history[f"train_{metric}"].append(value)
                metrics.reset()

                # Print performance metrics
                num_tokens_processed = 200 * batch_size_in_tokens
                tokens_per_second = num_tokens_processed / elapsed_time

                print(
                    f"Step {step + 1}, Loss: {metrics_history['train_loss'][-1]:6.3f}, "
                    f"Tokens/sec: {tokens_per_second:12.2f}, "
                    f"TFLOPS/device: {flops_per_device * 200 / (1e12 * elapsed_time):6.2f}"
                )
                start_time = time.time()

                # generated_text = model.generate_text(model_config.max_length, start_tokens)
                # print(f"Generated text:\n{generated_text}\n")
            step += 1

    # # Final text generation
    # generated_text = model.generate_text(model_config.max_length, start_tokens)
    # print(f"Final generated text:\n{generated_text}")
