"""Model/tokenizer loading and chat formatting, shared by direction, clas and evaluate."""

from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_model() -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load Llama-3.1-8B-Instruct in bf16 on one GPU, frozen and in eval mode."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID, dtype=getattr(torch, config.DTYPE), device_map=config.DEVICE
        )
    except Exception as exc:  # gated repo / missing token / no cache
        raise RuntimeError(
            f"Could not load {config.MODEL_ID}. It is gated on Hugging Face -- check that "
            f"HF_TOKEN is set and the licence is accepted. Original error: {exc}"
        ) from exc

    tokenizer.pad_token = tokenizer.eos_token
    # Left padding is required: the direction is read from h[:, -1], which is the true
    # last token only under left padding. Right padding would average pad activations.
    tokenizer.padding_side = "left"

    model.eval()
    model.requires_grad_(False)

    assert model.config.num_hidden_layers == 32, model.config.num_hidden_layers
    assert model.config.hidden_size == 4096, model.config.hidden_size
    assert max(config.STEER_LAYERS) < model.config.num_hidden_layers
    return model, tokenizer


def format_prompt(tokenizer, instruction: str) -> str:
    """Apply the model's own chat template, with the assistant turn opened."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_prompts(tokenizer, instructions: List[str], device: str = config.DEVICE):
    assert tokenizer.padding_side == "left", "left padding is required"
    texts = [format_prompt(tokenizer, i) for i in instructions]
    return tokenizer(texts, return_tensors="pt", padding=True,
                     add_special_tokens=False).to(device)
