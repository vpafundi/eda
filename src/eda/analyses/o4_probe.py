"""O4 - Linear probes per macro: TF-IDF (char n-grams) and bge-m3 embeddings.

Probes measure feasibility (CV grouped by document); official baseline numbers
belong to the training repo.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import Thresholds
from ..taxonomy import MICRO_PER_MACRO

logger = logging.getLogger(__name__)


def docs_for_probe(frame: pd.DataFrame, max_chars: int) -> pd.Series:
    return "macro_" + frame["macro_category"] + " " + frame["text"].str.slice(0, max_chars)


def macro_probe_frame(mdf: pd.DataFrame, th: Thresholds, seed: int = 42) -> pd.DataFrame:
    """Stratified cap per sub + cap per macro (skipped when tfidf_full_data)."""
    if th.tfidf_full_data:
        return mdf
    parts = [
        g.sample(n=min(len(g), th.tfidf_max_per_sub), random_state=seed)
        for _, g in mdf.groupby("sub_category", sort=False)
    ]
    out = pd.concat(parts)
    if len(out) > th.tfidf_max_per_macro:
        out = out.sample(n=th.tfidf_max_per_macro, random_state=seed)
    return out


def linear_probe_macro(
    macro: str,
    frame: pd.DataFrame,
    x_rep: Any,
    n_splits: int = 5,
    probe_name: str = "probe",
    seed: int = 42,
) -> dict:
    """Out-of-fold logistic regression grouped by document; returns preds, proba, report."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    y = frame["sub_category"].to_numpy()
    groups = frame["document_id"].to_numpy()
    # Piccoli pool (smoke / macro rare): riduci i fold al massimo fattibile.
    min_class = int(pd.Series(y).value_counts().min())
    n_groups = int(pd.Series(groups).nunique())
    effective_splits = min(n_splits, min_class, n_groups, len(frame))
    if effective_splits < 2:
        raise ValueError(
            f"{macro}: pool troppo piccolo per il probe (min_class={min_class}, groups={n_groups})"
        )
    if effective_splits < n_splits:
        logger.warning("O4 %s | %s: fold ridotti %s -> %s", probe_name, macro, n_splits, effective_splits)
        n_splits = effective_splits
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.empty(len(frame), dtype=object)
    classes = np.unique(y)
    class_to_col = {c: i for i, c in enumerate(classes)}
    oof_proba = np.zeros((len(frame), len(classes)))

    t_macro = time.perf_counter()
    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(frame, y, groups), start=1):
        t_fold = time.perf_counter()
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed)
        clf.fit(x_rep[train_idx], y[train_idx])
        pred = clf.predict(x_rep[test_idx])
        oof_pred[test_idx] = pred
        proba = clf.predict_proba(x_rep[test_idx])
        for k, cls in enumerate(clf.classes_):
            oof_proba[test_idx, class_to_col[cls]] = proba[:, k]
        logger.info(
            "O4 %s | %s fold %s/%s in %.1fs (train=%s test=%s)",
            probe_name, macro, fold_i, n_splits,
            time.perf_counter() - t_fold, len(train_idx), len(test_idx),
        )

    macro_f1 = f1_score(y, oof_pred, average="macro")
    logger.info(
        "O4 %s | %s done macro-F1=%.3f total %.1fs (n=%s)",
        probe_name, macro, macro_f1, time.perf_counter() - t_macro, len(frame),
    )
    return {
        "macro_f1": macro_f1,
        "report": classification_report(y, oof_pred, zero_division=0),
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "classes": classes,
        "frame": frame,
    }


def run_tfidf_probes(
    flagged_df: pd.DataFrame,
    th: Thresholds,
    seed: int = 42,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """TF-IDF probe for every macro with >= 2 subs. Returns (results per macro, summary)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    probe_rows = []
    results: dict[str, dict] = {}
    for macro in MICRO_PER_MACRO:
        mdf = flagged_df[(flagged_df["macro_category"] == macro) & (~flagged_df["empty_text"])].copy()
        if mdf.empty or mdf["sub_category"].nunique() < 2:
            logger.info("O4 TF-IDF | skip %s (missing or single sub)", macro)
            continue
        mdf_probe = macro_probe_frame(mdf, th, seed=seed)
        logger.info(
            "O4 TF-IDF | start %s pool=%s probe=%s (full=%s)",
            macro, len(mdf), len(mdf_probe), th.tfidf_full_data,
        )
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=200_000
        )
        x = vec.fit_transform(docs_for_probe(mdf_probe, th.bge_max_chars))
        try:
            res = linear_probe_macro(
                macro, mdf_probe, x, n_splits=th.probe_folds, probe_name="TF-IDF", seed=seed
            )
        except ValueError as exc:
            logger.warning("O4 TF-IDF | skip %s: %s", macro, exc)
            continue
        results[macro] = res
        probe_rows.append(
            {
                "macro_category": macro,
                "representation": "tfidf",
                "macro_f1": res["macro_f1"],
                "n_pool": len(mdf),
                "n_probe": len(mdf_probe),
            }
        )
    return results, pd.DataFrame(probe_rows).round(3)


def run_bge_probes(
    flagged_df: pd.DataFrame,
    th: Thresholds,
    seed: int = 42,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """bge-m3 probe on a stratified subsample per sub. Heavy: needs sentence-transformers + torch."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("O4 bge-m3 | device=%s | loading model once…", device)
    model = SentenceTransformer(th.bge_model, device=device)
    encode_batch = 64 if device == "cuda" else 16

    rows = []
    results: dict[str, dict] = {}
    for macro in MICRO_PER_MACRO:
        pool = flagged_df[(flagged_df["macro_category"] == macro) & (~flagged_df["empty_text"])].copy()
        if pool.empty or pool["sub_category"].nunique() < 2:
            continue
        parts = [
            g.sample(n=min(len(g), th.bge_max_per_sub), random_state=seed)
            for _, g in pool.groupby("sub_category", sort=False)
        ]
        mdf = pd.concat(parts, ignore_index=True)
        logger.info("O4 bge-m3 | start %s n=%s", macro, len(mdf))
        emb = model.encode(
            docs_for_probe(mdf, th.bge_max_chars).tolist(),
            batch_size=encode_batch,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        res = linear_probe_macro(macro, mdf, emb, n_splits=th.probe_folds, probe_name="bge-m3", seed=seed)
        results[macro] = {**res, "emb": emb}
        rows.append(
            {
                "macro_category": macro,
                "representation": "bge-m3",
                "macro_f1": res["macro_f1"],
                "n_pool": len(pool),
                "n_probe": len(mdf),
            }
        )
    return results, pd.DataFrame(rows).round(3)


def estimate_tfidf_runtime(flagged_df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Rough per-macro minutes before launching the TF-IDF probe."""
    rows = []
    for macro in MICRO_PER_MACRO:
        mdf = flagged_df[(flagged_df["macro_category"] == macro) & (~flagged_df["empty_text"])]
        if mdf.empty or mdf["sub_category"].nunique() < 2:
            continue
        n_probe = len(macro_probe_frame(mdf, th))
        vec_sec = 15.0 + 0.0025 * n_probe
        cv_sec = th.probe_folds * (12.0 + 0.012 * n_probe)
        rows.append(
            {
                "macro_category": macro,
                "segments_pool": len(mdf),
                "probe_segments": n_probe,
                "est_minutes": round((vec_sec + cv_sec) / 60.0, 1),
            }
        )
    return pd.DataFrame(rows)
