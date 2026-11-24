import pandas as pd

from eda.curation import apply_curation_v1, proportional_train_subset
from tests.conftest import SUBS, make_segments


def test_curation_drops_out_of_perimeter(segments_df: pd.DataFrame) -> None:
    bad = segments_df.iloc[[0]].copy()
    bad["record_key"] = "rk_bad"
    bad["sub_category"] = "fattura"  # non ammessa sotto documentazione_medica
    df = pd.concat([segments_df, bad], ignore_index=True)

    curated, stats = apply_curation_v1(df)
    assert stats["n_in"] == len(df)
    assert "rk_bad" not in set(curated["record_key"])
    assert stats["n_after_a1_perimeter"] == len(df) - 1


def test_curation_a3_conflicting_dup_dropped(segments_df: pd.DataFrame) -> None:
    dup = segments_df.iloc[[0]].copy()
    dup["record_key"] = "rk_dup_conflict"
    dup["sub_category"] = SUBS[1]  # stesso testo, etichetta diversa
    df = pd.concat([segments_df, dup], ignore_index=True)

    curated, stats = apply_curation_v1(df)
    assert stats["n_dropped_a3_exact_dup_conflicting_label"] == 2
    assert "rk_dup_conflict" not in set(curated["record_key"])
    assert segments_df.iloc[0]["record_key"] not in set(curated["record_key"])


def test_curation_a4_same_label_dup_keeps_one(segments_df: pd.DataFrame) -> None:
    dup = segments_df.iloc[[0]].copy()
    dup["record_key"] = "zz_dup_same"  # ordina dopo: viene droppato lui
    df = pd.concat([segments_df, dup], ignore_index=True)

    curated, stats = apply_curation_v1(df)
    assert stats["n_dropped_a4_exact_dup_same_label"] == 1
    assert segments_df.iloc[0]["record_key"] in set(curated["record_key"])
    assert "zz_dup_same" not in set(curated["record_key"])


def test_curation_a5_flags_only_disagreeing_issues(segments_df: pd.DataFrame) -> None:
    issues = pd.DataFrame(
        {
            "record_key": [segments_df.iloc[0]["record_key"], segments_df.iloc[1]["record_key"]],
            "sub_category": [segments_df.iloc[0]["sub_category"], segments_df.iloc[1]["sub_category"]],
            "oof_pred": [SUBS[1], segments_df.iloc[1]["sub_category"]],  # solo il primo contraddice
        }
    )
    curated, stats = apply_curation_v1(segments_df, cleanlab_issues=issues)
    flagged = set(curated.loc[curated["exclude_from_train"] == 1, "record_key"])
    assert flagged == {segments_df.iloc[0]["record_key"]}
    assert stats["n_flagged_a5_exclude_from_train"] == 1
    # le righe flaggate restano nel curated (servono al test silver)
    assert segments_df.iloc[0]["record_key"] in set(curated["record_key"])


def test_proportional_subset_preserves_priors() -> None:
    df = make_segments({SUBS[0]: 400, SUBS[1]: 100, SUBS[4]: 1000})
    out, summary = proportional_train_subset(df, min_class_target=50, seed=42)

    p_before = df.groupby("sub_category").size() / len(df)
    p_after = out.groupby("sub_category").size() / len(out)
    for sub in p_before.index:
        assert abs(p_before[sub] - p_after[sub]) < 0.02
    # classe più piccola ~ target
    assert abs(out.groupby("sub_category").size()[SUBS[1]] - 50) <= 1
    assert not summary["replace"].any()


def test_proportional_subset_oversamples_when_below_target() -> None:
    df = make_segments({SUBS[0]: 30, SUBS[4]: 90})
    out, summary = proportional_train_subset(df, min_class_target=60, seed=42)
    sizes = out.groupby("sub_category").size()
    assert sizes[SUBS[0]] == 60
    assert sizes[SUBS[4]] == 180
    assert summary.set_index("sub_category").loc[SUBS[0], "replace"]
