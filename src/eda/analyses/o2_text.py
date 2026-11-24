"""O2 - Input text quality: length, OCR confidence, handwriting, token budget."""

from __future__ import annotations

import pandas as pd

from ..config import Thresholds


def add_quality_flags(labeled_df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Adds n_words, n_pages, is_short, empty_text, low_confidence, mostly_handwritten (in place copy)."""
    out = labeled_df.copy()
    out["n_words"] = out["text"].str.split().str.len()
    out["n_pages"] = out["pages_list"].str.len() if "pages_list" in out.columns else pd.NA
    out["is_short"] = out["n_words"] < th.min_words
    out["empty_text"] = out["text"].str.len() == 0

    confidence_scale = 100 if out["ocr_confidence"].max() > 1 else 1
    out["low_confidence"] = out["ocr_confidence"] < th.low_confidence * confidence_scale
    out["mostly_handwritten"] = out["ocr_handwritten_rate"] > th.high_handwritten
    return out


def length_summary(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """O2.1 - per (macro, sub): empty share, pages, words, short share."""
    return (
        flagged_df.groupby(["macro_category", "sub_category"])
        .agg(
            segments=("record_key", "size"),
            empty_share=("empty_text", "mean"),
            pages_p50=("n_pages", "median"),
            pages_max=("n_pages", "max"),
            words_p50=("n_words", "median"),
            words_p95=("n_words", lambda s: s.quantile(0.95)),
            short_share=("is_short", "mean"),
        )
        .round(3)
    )


def quality_summary(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """O2.2 - OCR confidence and handwriting per (macro, sub)."""
    return (
        flagged_df.groupby(["macro_category", "sub_category"])
        .agg(
            segments=("record_key", "size"),
            missing_ocr_stats=("ocr_confidence", lambda s: s.isna().mean()),
            confidence_p50=("ocr_confidence", "median"),
            low_confidence_share=("low_confidence", "mean"),
            handwritten_p50=("ocr_handwritten_rate", "median"),
            mostly_handwritten_share=("mostly_handwritten", "mean"),
        )
        .round(3)
    )


def token_scan_frame(flagged_df: pd.DataFrame, th: Thresholds, seed: int = 42) -> pd.DataFrame:
    """O2.3 - stratified-per-macro sample of non-empty rows for the tokenizer scan."""
    eda_df = flagged_df[~flagged_df["empty_text"]].copy()
    if len(eda_df) > th.token_scan_max_rows:
        n_macros = eda_df["macro_category"].nunique()
        per_macro = max(1, th.token_scan_max_rows // n_macros)
        parts = [
            g.sample(n=min(len(g), per_macro), random_state=seed)
            for _, g in eda_df.groupby("macro_category", sort=False)
        ]
        eda_df = pd.concat(parts)
    return eda_df


def flag_impact(flagged_df: pd.DataFrame, quality_flags: list[str] | None = None) -> pd.DataFrame:
    """O2.4 - aggregated share of flagged segments per (macro, sub)."""
    flags = quality_flags or ["is_short", "low_confidence", "mostly_handwritten", "empty_text"]
    flags = [f for f in flags if f in flagged_df.columns]
    out = flagged_df.copy()
    out["any_flag"] = out[flags].any(axis=1)
    summary = out.groupby(["macro_category", "sub_category"])[flags + ["any_flag"]].mean().round(3)
    summary.insert(0, "segments", out.groupby(["macro_category", "sub_category"]).size())
    return summary
