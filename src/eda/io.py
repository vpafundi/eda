"""Load the segment parquet (local path or s3://), apply window/perimeter filters."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from .config import EDA_COLUMNS, REQUIRED_COLUMNS, EdaConfig
from .taxonomy import is_in_perimeter
from .text import normalize_text


def dataset_sha256(path: str) -> str | None:
    """Hash of the local parquet for the run manifest (None for s3:// or missing file)."""
    p = Path(path)
    if str(path).startswith("s3://") or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_segments(cfg: EdaConfig) -> pd.DataFrame:
    """Read the parquet (s3:// needs s3fs), check schema, normalize text, add created_date.

    Returns every row (including out-of-perimeter / unlabeled): the split into
    `labeled_df` happens in `prepare_labeled`.
    """
    if not cfg.dataset_path:
        raise ValueError("dataset_path vuoto: passa --data o imposta EDA_DATA_PATH")
    path = cfg.dataset_path
    if not str(path).startswith("s3://") and not Path(path).is_file():
        raise FileNotFoundError(f"Dataset non trovato: {path}")

    try:
        df = pd.read_parquet(path, columns=EDA_COLUMNS)
    except Exception:
        df = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nel parquet: {sorted(missing)}")
    duplicate_rows = int(df["record_key"].duplicated().sum())
    if duplicate_rows:
        # Stesso segmento re-ingestito con OCR/surrogati diversi (etichette in genere uguali).
        logger.warning(
            "record_key duplicati: %s righe; tiene l'ultima per key (vedi manifest ingestione)",
            duplicate_rows,
        )
        df = df.sort_values(["record_key", "created_at"]).drop_duplicates(
            subset="record_key", keep="last"
        )
        df = df.reset_index(drop=True)

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["created_date"] = df["created_at"].dt.normalize()

    if cfg.window_start:
        df = df[df["created_at"] >= pd.Timestamp(cfg.window_start, tz="UTC")]
    if cfg.window_end:
        df = df[df["created_at"] <= pd.Timestamp(cfg.window_end, tz="UTC") + pd.Timedelta(days=1)]
    df = df.copy()

    for col in ("macro_category", "sub_category"):
        df[col] = df[col].astype(str).str.strip()
    df["text"] = df[cfg.text_column].map(normalize_text)
    return df


def prepare_labeled(df: pd.DataFrame, cfg: EdaConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split into (labeled_df in perimeter, out_of_taxonomy, stats).

    labeled_df keeps empty-text rows (flagged downstream in O2) but drops rows
    without macro/sub or outside the allowed taxonomy.
    """
    missing_labels = (df["macro_category"].replace("nan", "") == "") | (
        df["sub_category"].replace("nan", "") == ""
    )
    labeled = df[~missing_labels].copy()

    in_perimeter = np.array(
        [is_in_perimeter(m, s) for m, s in zip(labeled["macro_category"], labeled["sub_category"])],
        dtype=bool,
    )
    out_of_taxonomy = labeled[~in_perimeter].copy()
    labeled = labeled[in_perimeter].copy()

    if cfg.macro_focus:
        labeled = labeled[labeled["macro_category"] == cfg.macro_focus].copy()
        if labeled.empty:
            raise ValueError(f"nessun segmento per macro_focus={cfg.macro_focus!r}")

    stats = {
        "segments_total": int(len(df)),
        "segments_without_labels": int(missing_labels.sum()),
        "segments_out_of_taxonomy": int(len(out_of_taxonomy)),
        "segments_in_perimeter": int(len(labeled)),
        "documents": int(labeled["document_id"].nunique()),
        "empty_text": int((labeled["text"].str.len() == 0).sum()),
        "window_first_day": str(labeled["created_date"].min().date()) if len(labeled) else None,
        "window_last_day": str(labeled["created_date"].max().date()) if len(labeled) else None,
        "days_with_data": int(labeled["created_date"].nunique()),
    }
    return labeled, out_of_taxonomy, stats
