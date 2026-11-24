"""Minimal prompt (v1) used only to size the token budget in O2.3.

The enriched prompt variants live in the training repo; the EDA needs only the
minimal template because max_seq_length is dominated by the segment text.
"""

from __future__ import annotations

from .taxonomy import MICRO_PER_MACRO

PROMPT_VERSION = "v1"

BASE_SYSTEM_PROMPT = (
    "Sei un classificatore di segmenti di documenti assicurativi. "
    "Ti viene indicata la macrocategoria (macro) e il testo del segmento. "
    "Rispondi solo con il nome della microcategoria (micro), "
    "senza spiegazioni, senza JSON e senza altro testo."
)


def system_prompt_for_macro(macro: str) -> str:
    allowed = ", ".join(MICRO_PER_MACRO[macro])
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Per la macro «{macro}», le uniche micro ammesse sono: {allowed}. "
        f"Rispondi con una sola di queste micro, scritta esattamente come nell'elenco."
    )


def user_content(macro: str, text: str) -> str:
    return f"Macro: {macro}\n\n{text}"


def build_messages(macro: str, text: str, micro: str | None = None) -> list[dict[str, str]]:
    """Chat messages; pass `micro` to measure the full training conversation."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt_for_macro(macro)},
        {"role": "user", "content": user_content(macro, text)},
    ]
    if micro is not None:
        messages.append({"role": "assistant", "content": micro})
    return messages
