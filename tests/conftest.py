"""Synthetic segment fixtures (no real AXA data in the repo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

MACRO = "documentazione_medica"
SUBS = [
    "prescrizione_medica",
    "referto_del_pronto_soccorso",
    "certificato_avvenuta_guarigione",
    "lettera_di_dimissioni",
    "documentazione_medica_altro",
]


def make_segments(n_per_sub: dict[str, int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for sub, n in n_per_sub.items():
        for _ in range(n):
            doc = f"doc_{i // 2}"  # ~2 segmenti per documento
            rows.append(
                {
                    "record_key": f"rk_{i}",
                    "document_id": doc,
                    "segment_id": f"seg_{i}",
                    "job_id": "job_1",
                    "created_at": pd.Timestamp("2026-10-01", tz="UTC")
                    + pd.Timedelta(days=int(rng.integers(0, 40))),
                    "macro_category": MACRO,
                    "sub_category": sub,
                    "pages_list": [1],
                    "ocr_confidence": float(rng.uniform(0.7, 1.0)),
                    "ocr_surrogates": f"testo {sub} id {i} segnale {rng.integers(0, 1000)} parole di prova "
                    * int(rng.integers(3, 10)),
                    "ocr_num_words": 30,
                    "ocr_num_words_handwritten": 0,
                    "ocr_handwritten_rate": 0.0,
                }
            )
            i += 1
    df = pd.DataFrame(rows)
    df["created_date"] = df["created_at"].dt.normalize()
    df["text"] = df["ocr_surrogates"]
    return df


@pytest.fixture
def segments_df() -> pd.DataFrame:
    return make_segments({s: 40 for s in SUBS})
