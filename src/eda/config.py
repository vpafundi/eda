"""Run configuration: dataset path, temporal window, thresholds, seed."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

EDA_COLUMNS = [
    "record_key",
    "document_id",
    "segment_id",
    "job_id",
    "created_at",
    "macro_category",
    "sub_category",
    "pages_list",
    "ocr_confidence",
    "ocr_surrogates",
    "ocr_num_words",
    "ocr_num_words_handwritten",
    "ocr_handwritten_rate",
]

REQUIRED_COLUMNS = [
    "record_key",
    "document_id",
    "created_at",
    "macro_category",
    "sub_category",
    "ocr_surrogates",
]


@dataclass
class Thresholds:
    """Soglie condivise tra notebook e CLI (vedi docs/eda_strategy.md)."""

    min_documents_eda: int = 20
    min_segments_train: int = 100
    target_segments_train: int = 300
    min_words: int = 20
    low_confidence: float = 0.80
    high_handwritten: float = 0.50
    quasi_dup_threshold: float = 0.98
    quasi_dup_sample_per_macro: int = 2000
    # O1.5 test sizing
    expected_recall: float = 0.95
    confidence_level: float = 0.98
    test_margin: float = 0.05
    # O4 probe
    tfidf_full_data: bool = False
    tfidf_max_per_sub: int = 600
    tfidf_max_per_macro: int = 5000
    bge_max_per_sub: int = 800
    bge_model: str = "BAAI/bge-m3"
    bge_max_chars: int = 4000
    probe_folds: int = 5
    # O2.3 token scan
    token_scan_max_rows: int = 8000
    candidate_max_seq_lengths: tuple[int, ...] = (4096, 8192, 16384, 32768)


@dataclass
class EdaConfig:
    """Config of one EDA run; serialized into the run store manifest."""

    dataset_path: str = field(default_factory=lambda: os.environ.get("EDA_DATA_PATH", ""))
    text_column: str = "ocr_surrogates"
    output_dir: str = field(default_factory=lambda: os.environ.get("EDA_OUTPUT_DIR", "data"))
    tag: str = "eda"
    seed: int = 42
    # Finestra temporale sui dati (ISO date, inclusive); None = tutto il parquet.
    window_start: str | None = None
    window_end: str | None = None
    # Limita l'analisi a una sola macro (None = tutte quelle in perimetro).
    macro_focus: str | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["thresholds"]["candidate_max_seq_lengths"] = list(
            self.thresholds.candidate_max_seq_lengths
        )
        return d
