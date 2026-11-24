"""End-to-end smoke of the headless pipeline on synthetic data (no bge, no token scan)."""

from __future__ import annotations

import pandas as pd

from eda.config import EdaConfig, Thresholds
from eda.results import EdaRunStore
from eda.run import run_pipeline
from tests.conftest import SUBS, make_segments


def test_run_pipeline_smoke(tmp_path) -> None:
    df = make_segments({s: 60 for s in SUBS}, seed=1)
    data_path = tmp_path / "segments.parquet"
    df.drop(columns=["text", "created_date"]).to_parquet(data_path, index=False)

    cfg = EdaConfig(
        dataset_path=str(data_path),
        output_dir=str(tmp_path / "out"),
        tag="smoke",
        thresholds=Thresholds(probe_folds=2, tfidf_max_per_sub=60, quasi_dup_sample_per_macro=100),
    )
    store = run_pipeline(cfg, skip_bge=True, drift_cutoff="2026-10-20")

    loaded = EdaRunStore.load(store.root)
    arts = set(loaded["manifest"]["artifacts"])
    for expected in (
        "o1_sub_summary.csv",
        "o2_flag_impact.csv",
        "o6_leakage_duplicates.json",
        "o4/probe_summary.csv",
        "o5_cleanlab_by_macro.csv",
        "o7_weekly_segments.csv",
        "o7_era_share.csv",
        "decisions.json",
        "curation/df_segments_curated_v1.parquet",
        "curation/curation_manifest.json",
    ):
        assert expected in arts, f"missing artifact {expected}"

    curated = loaded["curation/df_segments_curated_v1.parquet"]
    assert "exclude_from_train" not in curated.columns
    assert len(curated) > 0
    assert isinstance(loaded["decisions.json"]["tfidf_probe_macro_f1"], dict)

    stats = loaded["manifest"]["dataset_stats"]
    assert stats["segments_in_perimeter"] == len(df)
    assert pd.notna(stats["window_first_day"])
