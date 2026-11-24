"""O5 - Silver label consistency via cleanlab on OOF probe probabilities; gold set sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cleanlab_label_issues(tfidf_results: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """find_label_issues per macro on the TF-IDF OOF probabilities.

    Returns (summary per macro, flagged rows with oof_pred and issue_rank).
    """
    from cleanlab.filter import find_label_issues

    rows = []
    issue_frames = []
    for macro, res in tfidf_results.items():
        frame = res["frame"]
        classes = res["classes"]
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        y = np.array([label_to_idx[label] for label in frame["sub_category"].to_numpy()], dtype=int)
        issues = find_label_issues(
            y, res["oof_proba"], return_indices_ranked_by="self_confidence", n_jobs=1
        )
        rows.append({"macro_category": macro, "n_issues": len(issues), "n": len(frame)})
        if len(issues):
            flagged = frame.iloc[list(issues)][["record_key", "document_id", "sub_category"]].copy()
            flagged["macro_category"] = macro
            flagged["issue_rank"] = np.arange(len(flagged))
            flagged["oof_pred"] = res["oof_pred"][issues]
            issue_frames.append(flagged)

    summary = pd.DataFrame(rows)
    issues_df = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame()
    return summary, issues_df


def gold_sketch(sub_summary: pd.DataFrame, gold_min: int = 200, gold_max: int = 500) -> dict:
    """Sizing of the manually-verified gold set across (macro, sub) pairs."""
    n_pairs = max(len(sub_summary), 1)
    per_pair = max(1, gold_min // n_pairs)
    return {
        "GOLD_MIN": gold_min,
        "GOLD_MAX": gold_max,
        "n_macro_sub_pairs": int(n_pairs),
        "segments_per_pair": int(per_pair),
        "estimated_total": int(min(per_pair * n_pairs, gold_max)),
    }
