"""OCR surrogate text normalization and hashing."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np


def normalize_text(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).replace("\x00", "").replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(re.sub(r"[ \t\f\v]+", " ", line) for line in s.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def text_hash(text: str) -> str:
    """Stable identity of normalized text (exact-duplicate detection)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
