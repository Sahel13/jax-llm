# jax-llm

A minimal implementation of a language model in JAX.

## Installation

Create a Python virtual environment with Python 3.12. Then install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) with the required hardware accelerator support (CPU, GPU, TPU) and install the package with

```bash
pip install -e .
```

## Details of the Architecture

The architecture of the model is based on [Gemma 3](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/) from Google Deepmind, specifically the 2B version as implemented in the [Flax library](https://github.com/google/flax/tree/main/examples/gemma).
However, the model is not identical, with most changes arising out of the need to have a small model (<100M params). Some of the key differences are:

- I use the GPT-2 tokenizer from [_tiktoken_](https://github.com/openai/tiktoken) with a vocab size of 50257, while Gemma uses a tokenizer from [_SentencePiece_](https://github.com/google/sentencepiece) with a vocab size of 256k.
- I use multi-head attention, while Gemma uses multi-query attention for the smaller versions.

## TODO

- FLOPS calculation in `train.py` appears to be wrong.
