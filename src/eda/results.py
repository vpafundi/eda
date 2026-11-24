"""Versioned run store: data/runs/<run_id>/ with manifest, per-objective artifacts, export.

The run_id ties one EDA execution (dataset snapshot + temporal window + config)
to downstream training experiments: cite it in the experiment manifest.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .config import EdaConfig
from .io import dataset_sha256


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


class EdaRunStore:
    """One directory per run under <output_dir>/runs/."""

    def __init__(self, cfg: EdaConfig, run_id: str | None = None) -> None:
        self.cfg = cfg
        created = datetime.now(UTC)
        self.run_id = run_id or f"{created.strftime('%Y%m%dT%H%M%SZ')}__{cfg.tag}"
        self.root = Path(cfg.output_dir) / "runs" / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at_utc": created.isoformat(),
            "package_version": __version__,
            "dataset_path": cfg.dataset_path,
            "dataset_sha256": dataset_sha256(cfg.dataset_path),
            "window_start": cfg.window_start,
            "window_end": cfg.window_end,
            "macro_focus": cfg.macro_focus,
            "config": cfg.to_dict(),
            "artifacts": [],
        }
        self._write_manifest()

    @property
    def artifacts(self) -> list[str]:
        return list(self._manifest["artifacts"])

    # ---- write ----

    def _write_manifest(self) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

    def _register(self, rel_path: str) -> None:
        if rel_path not in self._manifest["artifacts"]:
            self._manifest["artifacts"].append(rel_path)
        self._write_manifest()

    def update_manifest(self, **fields: Any) -> None:
        """Attach extra metadata (e.g. dataset stats, observed window) to the manifest."""
        self._manifest.update(fields)
        self._write_manifest()

    def save_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        self._register(str(path.relative_to(self.root)))
        return path

    def save_df(self, name: str, df: pd.DataFrame, *, index: bool = False) -> Path:
        """CSV or parquet depending on the extension in `name`."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            df.to_parquet(path, index=index)
        else:
            df.to_csv(path, index=index)
        self._register(str(path.relative_to(self.root)))
        return path

    # ---- read / export ----

    @classmethod
    def load(cls, run_dir: str | Path) -> dict[str, Any]:
        """Read one run back: manifest + artifacts (DataFrame for csv/parquet, dict for json)."""
        root = Path(run_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No EDA run at {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        out: dict[str, Any] = {"manifest": manifest}
        for rel in manifest.get("artifacts", []):
            path = root / rel
            if not path.is_file():
                continue
            if path.suffix == ".json":
                out[rel] = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".parquet":
                out[rel] = pd.read_parquet(path)
            elif path.suffix == ".csv":
                out[rel] = pd.read_csv(path)
        return out

    @classmethod
    def list_runs(cls, output_dir: str | Path = "data") -> pd.DataFrame:
        """Summary of every run under <output_dir>/runs/, newest first."""
        runs_dir = Path(output_dir) / "runs"
        rows = []
        if runs_dir.is_dir():
            for mpath in sorted(runs_dir.glob("*/manifest.json")):
                m = json.loads(mpath.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "run_id": m.get("run_id"),
                        "created_at_utc": m.get("created_at_utc"),
                        "dataset_path": m.get("dataset_path"),
                        "window_start": m.get("window_start"),
                        "window_end": m.get("window_end"),
                        "macro_focus": m.get("macro_focus"),
                        "n_artifacts": len(m.get("artifacts", [])),
                        "path": str(mpath.parent),
                    }
                )
        df = pd.DataFrame(rows)
        return df.sort_values("created_at_utc", ascending=False).reset_index(drop=True) if len(df) else df

    def export_archive(self, dest_dir: str | Path | None = None) -> Path:
        """Zip the run folder (e.g. to attach it to an experiment or share it)."""
        dest_dir = Path(dest_dir) if dest_dir else self.root.parent
        archive = shutil.make_archive(str(dest_dir / self.run_id), "zip", root_dir=self.root)
        return Path(archive)
