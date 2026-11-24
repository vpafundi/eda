"""O1 - Sub coverage and balance per macro, daily volumes, test sizing."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from ..config import Thresholds


def label_consistency(labeled_df: pd.DataFrame) -> dict[str, pd.Series]:
    """O1.1 - spelling variants and subs shared across macros."""
    macro_variants = labeled_df.groupby(
        labeled_df["macro_category"].str.strip().str.casefold().rename("normalized_macro")
    )["macro_category"].unique()
    macro_variants = macro_variants[macro_variants.map(len) > 1]

    sub_variants = labeled_df.groupby(
        ["macro_category", labeled_df["sub_category"].str.strip().str.casefold().rename("normalized_sub")]
    )["sub_category"].unique()
    sub_variants = sub_variants[sub_variants.map(len) > 1]

    macros_per_sub = labeled_df.groupby("sub_category")["macro_category"].unique()
    shared_subs = macros_per_sub[macros_per_sub.map(len) > 1]
    return {"macro_variants": macro_variants, "sub_variants": sub_variants, "shared_subs": shared_subs}


def macro_summary(labeled_df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """O1.2 - volumes per macro."""
    segments_per_pair = labeled_df.groupby(["macro_category", "sub_category"]).size()
    out = (
        labeled_df.groupby("macro_category")
        .agg(
            segments=("record_key", "size"),
            documents=("document_id", "nunique"),
            sub_categories=("sub_category", "nunique"),
        )
        .sort_values("segments", ascending=False)
    )
    out["share"] = (out["segments"] / out["segments"].sum()).round(3)
    out["segments_per_day"] = (out["segments"] / max(n_days, 1)).round(1)
    out["max_min_ratio"] = (
        segments_per_pair.groupby(level="macro_category").agg(lambda s: s.max() / s.min()).round(1)
    )
    out["single_sub"] = out["sub_categories"] == 1
    return out


def sub_summary(labeled_df: pd.DataFrame, n_days: int, th: Thresholds) -> pd.DataFrame:
    """O1.3 - sub coverage against training thresholds."""
    macro = macro_summary(labeled_df, n_days)
    single_sub_macros = macro.index[macro["single_sub"]]

    out = (
        labeled_df.groupby(["macro_category", "sub_category"])
        .agg(
            segments=("record_key", "size"),
            documents=("document_id", "nunique"),
            days_seen=("created_date", "nunique"),
        )
        .reset_index()
    )
    out["share_in_macro"] = (
        out["segments"] / out.groupby("macro_category")["segments"].transform("sum")
    ).round(3)

    daily_rate = out["segments"] / max(n_days, 1)
    out["segments_per_day"] = daily_rate.round(2)
    out["days_to_target"] = np.ceil(
        (th.target_segments_train - out["segments"]).clip(lower=0) / daily_rate.replace(0, np.nan)
    ).astype("Int64")

    out["status"] = np.select(
        [
            out["macro_category"].isin(single_sub_macros),
            out["documents"] < th.min_documents_eda,
            out["segments"] < th.min_segments_train,
            out["segments"] < th.target_segments_train,
        ],
        ["macro con una sola sub", "insufficiente", "solo EDA", "training (minimo)"],
        default="training (target)",
    )
    return out.sort_values(["macro_category", "segments"], ascending=[True, False]).reset_index(drop=True)


def daily_volumes(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """O1.4 - segments per day by macro (0 = missing run), full date range."""
    daily = (
        labeled_df.groupby(["created_date", "macro_category"]).size().unstack(fill_value=0).sort_index()
    )
    expected_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(expected_days, fill_value=0)
    daily.index = daily.index.date
    return daily


def test_sizing(labeled_df: pd.DataFrame, n_days: int, th: Thresholds) -> tuple[pd.DataFrame, int]:
    """O1.5 - independent test documents per sub at expected recall (collection plan)."""
    z_value = NormalDist().inv_cdf((1 + th.confidence_level) / 2)
    test_documents = int(
        np.ceil(z_value**2 * th.expected_recall * (1 - th.expected_recall) / th.test_margin**2)
    )

    plan = labeled_df.groupby(["macro_category", "sub_category"]).agg(
        segments=("record_key", "size"),
        documents=("document_id", "nunique"),
    )
    spd = plan["segments"] / plan["documents"]
    plan["segments_per_document"] = spd.round(2)
    validation_segments = 50
    plan["segments_needed"] = (
        th.target_segments_train + validation_segments + np.ceil(test_documents * spd)
    ).astype(int)
    plan["missing_segments"] = (plan["segments_needed"] - plan["segments"]).clip(lower=0)
    plan["surplus_segments"] = (plan["segments"] - plan["segments_needed"]).clip(lower=0)

    rate = (
        labeled_df.groupby(["macro_category", "sub_category"]).size().reindex(plan.index, fill_value=0)
        / max(n_days, 1)
    )
    plan["segments_per_day"] = rate.round(1)
    days = np.ceil(plan["missing_segments"] / rate.replace(0, np.nan)).replace(np.inf, np.nan)
    plan["days_to_ready"] = days.where(plan["missing_segments"] > 0, 0)
    return plan.sort_values(["macro_category", "days_to_ready"], ascending=[True, False]), test_documents
