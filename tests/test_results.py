import json

import pandas as pd

from eda.config import EdaConfig
from eda.results import EdaRunStore


def _cfg(tmp_path) -> EdaConfig:
    data = tmp_path / "segments.parquet"
    pd.DataFrame({"record_key": ["a"], "x": [1]}).to_parquet(data)
    return EdaConfig(
        dataset_path=str(data),
        output_dir=str(tmp_path / "out"),
        tag="test",
        window_start="2026-10-01",
        window_end="2026-11-10",
    )


def test_round_trip(tmp_path) -> None:
    store = EdaRunStore(_cfg(tmp_path))
    store.save_json("decisions.json", {"k": 1})
    store.save_df("o1_sub_summary.csv", pd.DataFrame({"a": [1, 2]}))
    store.save_df("curation/df_segments_curated_v1.parquet", pd.DataFrame({"record_key": ["x"]}))
    store.update_manifest(dataset_stats={"segments_in_perimeter": 1})

    loaded = EdaRunStore.load(store.root)
    m = loaded["manifest"]
    assert m["run_id"] == store.run_id
    assert m["window_start"] == "2026-10-01"
    assert m["dataset_sha256"] is not None
    assert m["dataset_stats"]["segments_in_perimeter"] == 1
    assert set(m["artifacts"]) == {
        "decisions.json",
        "o1_sub_summary.csv",
        "curation/df_segments_curated_v1.parquet",
    }
    assert loaded["decisions.json"] == {"k": 1}
    assert len(loaded["o1_sub_summary.csv"]) == 2
    assert list(loaded["curation/df_segments_curated_v1.parquet"]["record_key"]) == ["x"]


def test_list_runs_and_export(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    store = EdaRunStore(cfg)
    store.save_json("decisions.json", {})

    runs = EdaRunStore.list_runs(cfg.output_dir)
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == store.run_id

    archive = store.export_archive(tmp_path)
    assert archive.is_file() and archive.suffix == ".zip"


def test_manifest_is_valid_json(tmp_path) -> None:
    store = EdaRunStore(_cfg(tmp_path))
    manifest = json.loads((store.root / "manifest.json").read_text())
    assert manifest["package_version"]
