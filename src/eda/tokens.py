"""Token-length report over the full training sequence (system + macro + text + answer)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .prompts import build_messages

# Thinking off for every model: Qwen3 and SmolLM3 read this kwarg, Gemma 3 ignores it.
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}


def adapt_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """SmolLM3's template otherwise injects metadata with today's date into the system turn."""
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "smollm3" in name and messages and messages[0]["role"] == "system":
        messages = [{**messages[0], "content": messages[0]["content"] + "\n/system_override"}, *messages[1:]]
    return messages


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        adapt_messages(tokenizer, messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **CHAT_TEMPLATE_KWARGS,
    )


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """The rendered template already contains BOS where needed (e.g. Gemma): don't add it twice."""
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def token_length_report(
    tokenizer: Any,
    df: pd.DataFrame,
    thresholds: tuple[int, ...] = (1024, 2048, 3072, 4096, 6144, 8192),
) -> tuple[pd.Series, pd.DataFrame]:
    """Sequence length per row and coverage per threshold; df needs macro_category, sub_category, text."""
    lengths = pd.Series(
        [
            len(
                encode_prompt(
                    tokenizer,
                    render_prompt(tokenizer, build_messages(m, t, mi), add_generation_prompt=False),
                )
            )
            for m, t, mi in zip(df["macro_category"], df["text"], df["sub_category"])
        ],
        index=df.index,
        name="seq_tokens",
    )
    coverage = pd.DataFrame(
        {
            "max_seq_len": list(thresholds),
            "pct_covered": [round(float((lengths <= t).mean()) * 100, 2) for t in thresholds],
            "n_truncated": [int((lengths > t).sum()) for t in thresholds],
        }
    )
    return lengths, coverage


def recommend_max_seq_len(lengths: dict[str, pd.Series], headroom: float = 0.20, multiple: int = 1024) -> int:
    """Longest sequence across tokenizers + headroom, rounded up to `multiple`."""
    longest = max(int(s.max()) for s in lengths.values())
    target = longest * (1 + headroom)
    return int(-(-target // multiple) * multiple)
