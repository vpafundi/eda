"""O6 - Leakage, exact and quasi duplicates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Thresholds
from ..leakage import check_no_group_leakage
from ..text import text_hash


def leakage_stats(labeled_df: pd.DataFrame, splits_path: Path | None = None) -> dict:
    """Document grouping stats, optional saved-splits check, exact text duplicates."""
    segs_per_doc = labeled_df.groupby("document_id").size()
    macros_per_doc = labeled_df.groupby("document_id")["macro_category"].nunique()

    splits_rows = None
    splits_leakage_ok = None
    if splits_path is not None and Path(splits_path).is_file():
        splits = pd.read_parquet(splits_path)
        splits_rows = int(len(splits))
        try:
            check_no_group_leakage(splits)
            splits_leakage_ok = True
        except AssertionError:
            splits_leakage_ok = False

    df = labeled_df
    if "text_hash" not in df.columns:
        df = df.copy()
        df["text_hash"] = df["text"].map(lambda t: text_hash(t) if t else "")
    hash_dupes = df[df["text_hash"] != ""].groupby("text_hash").filter(lambda g: len(g) > 1)
    cross_doc_exact = hash_dupes.groupby("text_hash")["document_id"].nunique() if len(hash_dupes) else pd.Series(dtype=int)
    label_conflict_exact = (
        hash_dupes.groupby("text_hash").apply(
            lambda g: g.groupby(["macro_category", "sub_category"]).ngroups > 1,
            include_groups=False,
        )
        if len(hash_dupes)
        else pd.Series(dtype=bool)
    )
    return {
        "segments_per_doc_median": float(segs_per_doc.median()),
        "segments_per_doc_max": int(segs_per_doc.max()),
        "docs_with_multiple_macros": int((macros_per_doc > 1).sum()),
        "splits_path": str(Path(splits_path).resolve()) if splits_path is not None else None,
        "splits_rows": splits_rows,
        "splits_leakage_ok": splits_leakage_ok,
        "exact_duplicate_segments": int(len(hash_dupes)),
        "exact_duplicate_hashes": int(hash_dupes["text_hash"].nunique()) if len(hash_dupes) else 0,
        "duplicate_hashes_cross_document": int((cross_doc_exact > 1).sum()) if len(hash_dupes) else 0,
        "exact_dupes_conflicting_labels": int(label_conflict_exact.sum()) if len(label_conflict_exact) else 0,
    }


def quasi_dup_by_macro(
    labeled_df: pd.DataFrame,
    th: Thresholds,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Sampled quasi-duplicate pairs (TF-IDF char n-grams, cosine >= threshold) per macro."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    pool = labeled_df[labeled_df["text"].str.len() > 0]
    quasi_pairs = []
    for macro, mdf in pool.groupby("macro_category"):
        if len(mdf) > th.quasi_dup_sample_per_macro:
            mdf = mdf.sample(th.quasi_dup_sample_per_macro, random_state=seed)
        if len(mdf) < 2:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50_000)
        x = vec.fit_transform(mdf["text"].str.slice(0, 4000))
        sim = cosine_similarity(x)
        np.fill_diagonal(sim, 0)
        i_idx, j_idx = np.where(sim >= th.quasi_dup_threshold)
        for i, j in zip(i_idx, j_idx):
            if i >= j:
                continue
            ri, rj = mdf.iloc[i], mdf.iloc[j]
            quasi_pairs.append(
                {
                    "macro_category": macro,
                    "sim": sim[i, j],
                    "same_label": (ri["sub_category"] == rj["sub_category"]),
                    "same_document": ri["document_id"] == rj["document_id"],
                }
            )

    quasi_df = pd.DataFrame(quasi_pairs)
    extra: dict = {"quasi_dup_threshold": th.quasi_dup_threshold}
    if not len(quasi_df):
        return pd.DataFrame(), extra
    extra["quasi_dup_pairs_sampled"] = int(len(quasi_df))
    extra["quasi_dup_label_conflicts"] = int((~quasi_df["same_label"]).sum())
    by_macro = quasi_df.groupby("macro_category").agg(
        pairs=("sim", "size"),
        conflict_rate=("same_label", lambda s: round(1 - s.mean(), 4)),
    )
    return by_macro, extra
