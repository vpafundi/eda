"""Dataset curation (pre-split) and proportional train subsampling (post-split).

Rules v1 — applied to the whole corpus BEFORE any split:
  A1 taxonomy perimeter        -> drop
  A2 empty normalized text     -> drop
  A3 exact dup, conflicting labels (macro::sub) -> drop the whole group
  A4 exact dup, same label     -> keep one record per text hash
  A5 cleanlab suspected mislabels -> drop (lista in o5_suspected_mislabels per revisione)

`proportional_train_subset` belongs AFTER the split, on the train rows only:
it rescales every sub with the same factor so class priors stay unchanged
(distillation-friendly), targeting `min_class_target` rows on the smallest sub.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .taxonomy import is_in_perimeter
from .text import normalize_text, text_hash


def apply_curation_v1(
    df: pd.DataFrame,
    *,
    text_col: str = "ocr_surrogates",
    cleanlab_issues: pd.DataFrame | None = None,
    macro_focus: str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Returns (curated_df train-ready, stats per rule)."""
    stats: dict = {"n_in": int(len(df)), "seed": seed}
    out = df.copy()

    # A1 perimetro
    mask = np.array(
        [is_in_perimeter(m, s) for m, s in zip(out["macro_category"], out["sub_category"])],
        dtype=bool,
    )
    out = out.loc[mask].copy()
    stats["n_after_a1_perimeter"] = int(len(out))

    if macro_focus:
        out = out[out["macro_category"] == macro_focus].copy()
        stats["n_after_macro_focus"] = int(len(out))

    # A2 testo vuoto (su testo normalizzato)
    if "text" in out.columns:
        out["_text_norm"] = out["text"]
    else:
        out["_text_norm"] = out[text_col].map(normalize_text)
    empty = out["_text_norm"].str.len() == 0
    stats["n_dropped_a2_empty_text"] = int(empty.sum())
    out = out.loc[~empty].copy()

    out["_text_hash"] = out["_text_norm"].map(text_hash)

    # A3 stesso testo, etichette (macro::sub) in conflitto -> drop gruppo
    out["_label_key"] = out["macro_category"] + "::" + out["sub_category"]
    labels_per_hash = out.groupby("_text_hash")["_label_key"].nunique()
    bad_hashes = set(labels_per_hash[labels_per_hash > 1].index)
    a3_mask = out["_text_hash"].isin(bad_hashes)
    stats["n_dropped_a3_exact_dup_conflicting_label"] = int(a3_mask.sum())
    out = out.loc[~a3_mask].copy()

    # A4 stesso testo, stessa etichetta -> tieni 1 record per hash (min record_key)
    out = out.sort_values("record_key")
    before_a4 = len(out)
    out = out.drop_duplicates(subset=["_text_hash"], keep="first")
    stats["n_dropped_a4_exact_dup_same_label"] = int(before_a4 - len(out))

    # A5 cleanlab: drop quando il probe OOF contraddice l'etichetta silver
    stats["n_dropped_a5_cleanlab"] = 0
    if cleanlab_issues is not None and len(cleanlab_issues):
        iss = cleanlab_issues.copy()
        if "oof_pred" in iss.columns:
            iss = iss[iss["sub_category"].astype(str) != iss["oof_pred"].astype(str)]
        bad_keys = set(iss["record_key"].astype(str))
        a5_mask = out["record_key"].astype(str).isin(bad_keys)
        stats["n_dropped_a5_cleanlab"] = int(a5_mask.sum())
        out = out.loc[~a5_mask].copy()

    curated = out[[c for c in out.columns if not c.startswith("_")]].copy()
    stats["n_out_curated"] = int(len(curated))
    stats["by_sub"] = {
        f"{m}::{s}": int(n)
        for (m, s), n in curated.groupby(["macro_category", "sub_category"]).size().items()
    }
    return curated, stats


def proportional_train_subset(
    train_df: pd.DataFrame,
    *,
    min_class_target: int = 1250,
    seed: int = 42,
    label_col: str = "sub_category",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale every sub by the same factor: n_sample(sub) = round(n(sub) * target / n_min).

    Class priors stay unchanged (distillation). When the smallest sub already has
    fewer rows than `min_class_target`, the factor is > 1 and sampling uses
    replacement (proportional oversample).

    Returns (sampled train_df shuffled, summary per sub).
    """
    if train_df.empty:
        return train_df, pd.DataFrame()

    rng = np.random.default_rng(seed)
    counts = train_df.groupby(label_col, dropna=False).size()
    n_min = int(counts.min())
    if n_min <= 0:
        raise ValueError("classe vuota nel train")
    scale = min_class_target / n_min

    parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for sub, n_avail in counts.items():
        n_target = max(1, int(round(n_avail * scale)))
        g = train_df[train_df[label_col] == sub]
        replace = n_target > len(g)
        take = g.sample(n=n_target, replace=replace, random_state=int(rng.integers(0, 2**31 - 1)))
        summary_rows.append(
            {
                label_col: sub,
                "n_available": int(n_avail),
                "n_sampled": int(len(take)),
                "replace": bool(replace),
                "frac_of_available": round(len(take) / len(g), 4) if len(g) else 0.0,
            }
        )
        parts.append(take)

    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out, pd.DataFrame(summary_rows)
