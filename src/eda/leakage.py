"""Split integrity checks (document-level grouping)."""

from __future__ import annotations

import pandas as pd


def check_no_group_leakage(splits: pd.DataFrame, group_col: str = "document_id") -> None:
    """Raise AssertionError when one group (document) spans more than one split."""
    if "split" not in splits.columns:
        return
    per_group = splits.groupby(group_col)["split"].nunique()
    leaked = per_group[per_group > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} documenti in più split")
