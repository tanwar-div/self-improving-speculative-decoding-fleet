"""The frozen target model.

Speculative decoding never changes what the target would have produced; the
target is loaded once per replica and shared across every draft version that
gets hot-swapped underneath it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fleet.config import settings

log = logging.getLogger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name != "auto":
        return _DTYPES[name]
    # fp16 on CUDA; fp32 on CPU because fp16 CPU matmul is dramatically slower
    # and would make the speculative speedup unmeasurable for the wrong reason.
    return torch.float16 if device.startswith("cuda") else torch.float32


@dataclass
class TargetModel:
    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def load(
        cls,
        name: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        load_in_4bit: bool | None = None,
    ) -> TargetModel:
        name = name or settings.target_model
        device = device or settings.device
        torch_dtype = resolve_dtype(dtype or settings.dtype, device)
        four_bit = settings.load_in_4bit if load_in_4bit is None else load_in_4bit

        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        kwargs: dict = {"dtype": torch_dtype}
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": device}

        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if not four_bit:
            model = model.to(device)
        model.eval()
        model.config.use_cache = True

        log.info(
            "loaded target %s on %s dtype=%s 4bit=%s params=%.0fM",
            name,
            device,
            torch_dtype,
            four_bit,
            sum(p.numel() for p in model.parameters()) / 1e6,
        )
        return cls(
            name=name,
            model=model,
            tokenizer=tok,
            device=torch.device(device),
            dtype=torch_dtype,
        )

    @property
    def eos_ids(self) -> set[int]:
        ids: set[int] = set()
        cfg_eos = getattr(self.model.generation_config, "eos_token_id", None)
        for candidate in (self.tokenizer.eos_token_id, cfg_eos):
            if isinstance(candidate, int):
                ids.add(candidate)
            elif isinstance(candidate, (list, tuple)):
                ids.update(int(x) for x in candidate)
        return ids

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def encode_chat(self, prompt: str) -> list[int]:
        """Render a user prompt through the model's chat template when it has one."""
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(text, add_special_tokens=False).input_ids
        return self.tokenizer(prompt).input_ids
