"""O7 - Light temporal drift: weekly volumes and era share around a cutoff date."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_volumes(labeled_df: pd.DataFrame) -> pd.Series:
    df = labeled_df.copy()
    df["week"] = df["created_at"].dt.tz_localize(None).dt.to_period("W").astype(str)
    return df.groupby("week").size().rename("segments")


def era_share(labeled_df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Share of each (macro, sub) collected from `cutoff` (ISO date, UTC) onwards."""
    df = labeled_df.copy()
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    before, after = f"before_{cutoff}", f"from_{cutoff}"
    df["era"] = np.where(df["created_date"] < cutoff_ts, before, after)
    out = df.groupby(["macro_category", "sub_category", "era"]).size().unstack(fill_value=0)
    share_col = f"share_{after}"
    out[share_col] = (out.get(after, 0) / out.sum(axis=1).replace(0, np.nan)).round(3)
    return out.sort_values(share_col, ascending=False)
