"""Generator for 00_eda.ipynb. Run from project root: uv run python notebooks/_generate_00_eda.py"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "00_eda.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        """# 00 - EDA del dataset AIC anonimizzato

Strategia e obiettivi: [docs/eda_strategy.md](../docs/eda_strategy.md).

Target: `sub_category` data la `macro_category` (già nota in input). Tutte le analisi sono **per macro**.
Le etichette sono **silver** (previsioni del teacher). Testo modello: `ocr_surrogates` normalizzato in `text`.

Tutta la logica sta nel package `eda` (`src/eda/`); questo notebook orchestra e visualizza.
Ordine: **Setup → O1 → O2 → O3 → O6 → O4 → O5 → O7 → Decisioni → Curation (A) → Sampling train (B, preview)**.

## Setup
"""
    ),
    code(
        """import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)
logger = logging.getLogger("eda-notebook")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)

from eda.config import EdaConfig, Thresholds
from eda.io import load_segments, prepare_labeled
from eda.results import EdaRunStore

# --- parametri della run (adatta qui) ---
DATA_PATH = os.environ.get("EDA_DATA_PATH", "data/df_sample_all.parquet")
RUN_TAG = "notebook"
WINDOW_START = None   # es. "2026-10-01"
WINDOW_END = None     # es. "2026-11-20"
MACRO_FOCUS = None    # es. "documentazione_medica"
DRIFT_CUTOFF = "2026-10-15"
SPLITS_PATH = None    # parquet split esistente da verificare in O6 (opzionale)

cfg = EdaConfig(
    dataset_path=DATA_PATH,
    tag=RUN_TAG,
    window_start=WINDOW_START,
    window_end=WINDOW_END,
    macro_focus=MACRO_FOCUS,
)
th = cfg.thresholds
store = EdaRunStore(cfg)
print("run_id:", store.run_id)
print("output:", store.root)
"""
    ),
    md(
        """## Caricamento e perimetro

Parquet locale o `s3://…`; righe senza macro/sub o fuori tassonomia escluse da `labeled_df`.
"""
    ),
    code(
        """df = load_segments(cfg)
labeled_df, out_of_taxonomy, stats = prepare_labeled(df, cfg)
n_days = max(stats["days_with_data"], 1)
store.update_manifest(dataset_stats=stats)

display(pd.Series(stats, name="value").to_frame())
if len(out_of_taxonomy):
    display(
        out_of_taxonomy.groupby(["macro_category", "sub_category"])
        .size().rename("segments").to_frame().head(20)
    )
"""
    ),
    md(
        """## O1 - Copertura e bilanciamento delle sub per macro

Conteggi, soglie train, volumi giornalieri, dimensionamento test (O1.5).
"""
    ),
    code(
        """from eda.analyses import o1_volumes

consistency = o1_volumes.label_consistency(labeled_df)
for name, series in consistency.items():
    if len(series):
        print(f"--- {name} ---")
        display(series.to_frame("variants"))

macro_summary = o1_volumes.macro_summary(labeled_df, n_days)
display(macro_summary)
store.save_df("o1_macro_summary.csv", macro_summary, index=True)
"""
    ),
    code(
        """sub_summary = o1_volumes.sub_summary(labeled_df, n_days, th)
display(sub_summary)
display(pd.crosstab(sub_summary["macro_category"], sub_summary["status"], margins=True))
store.save_df("o1_sub_summary.csv", sub_summary)

for macro, group in sub_summary.groupby("macro_category"):
    group = group.sort_values("segments")
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(group) + 1.5))
    ax.barh(group["sub_category"], group["segments"], color="tab:blue")
    ax.axvline(th.min_segments_train, color="tab:orange", linestyle="--", label=f"min training ({th.min_segments_train})")
    ax.axvline(th.target_segments_train, color="tab:green", linestyle="--", label=f"target ({th.target_segments_train})")
    ax.set_xscale("log")
    ax.set_xlabel("segments (log scale)")
    ax.set_title(f"{macro} - segments per sub ({group['segments'].sum()} total)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        """daily = o1_volumes.daily_volumes(labeled_df)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
daily.plot.bar(stacked=True, ax=axes[0], width=0.9)
axes[0].set_title("Segments per day by macro (0 = missing run)")
daily.cumsum().plot(ax=axes[1], drawstyle="steps-post")
axes[1].set_title("Cumulative segments by macro")
for ax in axes:
    ax.set_xlabel("created_at (day, UTC)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """collection_plan, test_documents = o1_volumes.test_sizing(labeled_df, n_days, th)
logger.info("Test documents per sub: %s (recall %s, margin ±%s)", test_documents, th.expected_recall, th.test_margin)
display(collection_plan)
store.save_df("o1_collection_plan.csv", collection_plan, index=True)
"""
    ),
    md(
        """## O2 - Qualità del testo in ingresso

Lunghezza, OCR confidence, scrittura a mano; flag riusati da O3/O4/O6 e dalla curation.
"""
    ),
    code(
        """from eda.analyses import o2_text

labeled_df = o2_text.add_quality_flags(labeled_df, th)

length_summary = o2_text.length_summary(labeled_df)
display(length_summary.head(30))
store.save_df("o2_length_summary.csv", length_summary, index=True)

quality_summary = o2_text.quality_summary(labeled_df)
display(quality_summary.head(20))
store.save_df("o2_quality_summary.csv", quality_summary, index=True)
"""
    ),
    md(
        """### O2.3 - Lunghezza in token (prompt v1)

Campione stratificato per macro; richiede i tokenizer HF (`HF_TOKEN` per Gemma).
Metti `RUN_TOKEN_SCAN = False` per saltare (es. senza rete).
"""
    ),
    code(
        """RUN_TOKEN_SCAN = True
MODEL_IDS = ["Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM3-3B", "google/gemma-3-4b-it"]

selected_max_seq_length = None
if RUN_TOKEN_SCAN:
    from transformers import AutoTokenizer

    from eda.tokens import recommend_max_seq_len, token_length_report

    scan_df = o2_text.token_scan_frame(labeled_df, th, seed=cfg.seed)
    logger.info("Token scan on %s segments", len(scan_df))

    lengths_by_model = {}
    coverage_frames = []
    for model_id in MODEL_IDS:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            logger.warning("Skip tokenizer %s: %s", model_id, exc)
            continue
        lengths, cov = token_length_report(tok, scan_df, thresholds=th.candidate_max_seq_lengths)
        lengths_by_model[model_id] = lengths
        cov["model"] = model_id
        coverage_frames.append(cov)
        display(cov)

    if lengths_by_model:
        selected_max_seq_length = recommend_max_seq_len(lengths_by_model, headroom=0.20, multiple=1024)
        logger.info("Recommended max_seq_length (sample): %s", selected_max_seq_length)
        store.save_df("o2_token_coverage.csv", pd.concat(coverage_frames, ignore_index=True))
"""
    ),
    code(
        """flag_impact = o2_text.flag_impact(labeled_df)
display(flag_impact.head(25))
store.save_df("o2_flag_impact.csv", flag_impact, index=True)
print(f"Empty text to exclude from eval: {int(labeled_df['empty_text'].sum())}")
"""
    ),
    md(
        """## O3 - Segnali discriminanti nei surrogati

Pattern lessicali per sub: l'anonimizzazione conserva segnale utile?
"""
    ),
    code(
        """from eda.analyses import o3_signals

signal_df = o3_signals.signal_table(labeled_df)
if len(signal_df):
    display(signal_df.sort_values(["macro_category", "sub_category"]))
    store.save_df("o3_signal_patterns.csv", signal_df)
"""
    ),
    md(
        """## O6 - Leakage, duplicati, split

Duplicati esatti (hash del testo), quasi-duplicati (TF-IDF cosine ≥ soglia), verifica split esistente.
"""
    ),
    code(
        """from eda.analyses import o6_dups

o6_stats = o6_dups.leakage_stats(labeled_df, SPLITS_PATH)
quasi_by_macro, quasi_extra = o6_dups.quasi_dup_by_macro(labeled_df, th, seed=cfg.seed)
o6_stats = {**o6_stats, **quasi_extra}

print(json.dumps({k: v for k, v in o6_stats.items() if k != "splits_path"}, indent=2))
if len(quasi_by_macro):
    display(quasi_by_macro)
    store.save_df("o6_quasi_dup_by_macro.csv", quasi_by_macro, index=True)
store.save_json("o6_leakage_duplicates.json", o6_stats)
"""
    ),
    md(
        """## O4 - Linear probe (TF-IDF + bge-m3)

Fattibilità del task per macro (CV raggruppata per documento). Su **Mac** lascia
`th.tfidf_full_data = False`; su **SageMaker** puoi metterlo a `True`.
`RUN_BGE = False` per saltare il probe con embedding (lento senza GPU).
"""
    ),
    code(
        """from eda.analyses import o4_probe

display(o4_probe.estimate_tfidf_runtime(labeled_df, th))

tfidf_results, probe_summary = o4_probe.run_tfidf_probes(labeled_df, th, seed=cfg.seed)
display(probe_summary)
"""
    ),
    code(
        """RUN_BGE = False

bge_results = {}
if RUN_BGE:
    bge_results, bge_summary = o4_probe.run_bge_probes(labeled_df, th, seed=cfg.seed)
    probe_summary = pd.concat([probe_summary, bge_summary], ignore_index=True)
    display(probe_summary)

store.save_df("o4/probe_summary.csv", probe_summary)

oof_frames = []
for macro, res in tfidf_results.items():
    oof = res["frame"][["record_key", "document_id", "sub_category"]].copy()
    oof["macro_category"] = macro
    oof["oof_pred"] = res["oof_pred"]
    oof_frames.append(oof)
if oof_frames:
    store.save_df("o4/tfidf_oof_predictions.parquet", pd.concat(oof_frames, ignore_index=True))
"""
    ),
    code(
        """RUN_UMAP = False  # solo esplorazione visiva, richiede RUN_BGE = True

if RUN_UMAP and bge_results:
    import umap

    for macro, res in bge_results.items():
        y = res["frame"]["sub_category"].to_numpy()
        xy = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(res["emb"])
        fig, ax = plt.subplots(figsize=(8, 6))
        for sub in np.unique(y):
            m = y == sub
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.5, label=sub)
        ax.set_title(f"UMAP bge-m3 — {macro} (subsample)")
        ax.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
"""
    ),
    md(
        """## O5 - Coerenza etichette silver (cleanlab) e set gold

cleanlab sulle probabilità out-of-fold del probe TF-IDF; i flagged alimentano la regola A5 della curation.
"""
    ),
    code(
        """from eda.analyses import o5_labels

cleanlab_summary, cleanlab_issues = o5_labels.cleanlab_label_issues(tfidf_results)
display(cleanlab_summary)
if len(cleanlab_issues):
    print("Top sospetti (no testo):")
    display(cleanlab_issues.head(10))
    store.save_df("o5_suspected_mislabels.parquet", cleanlab_issues)
store.save_df("o5_cleanlab_by_macro.csv", cleanlab_summary)

gold = o5_labels.gold_sketch(sub_summary)
print(json.dumps(gold, indent=2))
store.save_json("o5_gold_sketch.json", gold)
"""
    ),
    md("## O7 - Drift temporale leggero"),
    code(
        """from eda.analyses import o7_drift

weekly = o7_drift.weekly_volumes(labeled_df)
fig, ax = plt.subplots(figsize=(12, 4))
weekly.plot(ax=ax, marker="o")
ax.set_title("Segments per week (all macros)")
plt.tight_layout()
plt.show()
store.save_df("o7_weekly_segments.csv", weekly.to_frame(), index=True)

era = o7_drift.era_share(labeled_df, DRIFT_CUTOFF)
display(era.head(20))
store.save_df("o7_era_share.csv", era, index=True)
print("Teacher version not in parquet — interpret drift as collection mix, not model change.")
"""
    ),
    md(
        """## Decisioni finali

Riepilogo per gli esperimenti a valle; salvato come `decisions.json` nel run store.
"""
    ),
    code(
        """decisions = {
    "text_column": cfg.text_column,
    "segments_in_perimeter": int(len(labeled_df)),
    "empty_text_exclude_eval": int(labeled_df["empty_text"].sum()),
    "out_of_taxonomy_excluded": int(len(out_of_taxonomy)),
    "test_documents_per_sub": int(test_documents),
    "eda_max_seq_length_sample": int(selected_max_seq_length) if selected_max_seq_length else None,
    "tfidf_probe_macro_f1": (
        probe_summary[probe_summary["representation"] == "tfidf"]
        .set_index("macro_category")["macro_f1"].to_dict()
        if len(probe_summary) else {}
    ),
    "rare_subs_below_min_train": sub_summary[sub_summary["segments"] < th.min_segments_train][
        ["macro_category", "sub_category", "segments", "status"]
    ].to_dict(orient="records"),
}
print(json.dumps(decisions, ensure_ascii=False, indent=2))
store.save_json("decisions.json", decisions)
"""
    ),
    md(
        """## Curation (A) — dataset curated pre-split

Regole v1 (vedi `eda.curation`):
- **A1** perimetro tassonomia → drop
- **A2** testo vuoto → drop
- **A3** duplicato esatto con etichette in conflitto → drop del gruppo
- **A4** duplicato esatto, stessa etichetta → tieni 1 record per hash
- **A5** cleanlab (probe contraddice l'etichetta) → drop dal curated (`o5_suspected_mislabels` per revisione)

Output: `curation/df_segments_curated_v1.parquet` nel run store → input della pipeline di split/training.
"""
    ),
    code(
        """from eda.curation import apply_curation_v1

curated, cur_stats = apply_curation_v1(
    labeled_df,
    text_col=cfg.text_column,
    cleanlab_issues=cleanlab_issues if len(cleanlab_issues) else None,
    seed=cfg.seed,
)

flag_cols = ["n_words", "n_pages", "is_short", "empty_text", "low_confidence", "mostly_handwritten"]
curated_export = curated.drop(columns=[c for c in flag_cols if c in curated.columns])

print(json.dumps({k: v for k, v in cur_stats.items() if k != "by_sub"}, indent=2))
display(pd.Series(cur_stats["by_sub"]).rename("segments").sort_values(ascending=False).head(20))

store.save_df("curation/df_segments_curated_v1.parquet", curated_export)
store.save_json("curation/curation_manifest.json", cur_stats)
print(
    f"Curated: {len(curated_export)} righe "
    f"(A5 dropped={cur_stats.get('n_dropped_a5_cleanlab', 0)})"
)
"""
    ),
    md(
        """## Sampling train (B) — preview proporzionale

Lo split train/val/test vive nella pipeline di training (raggruppato per documento).
Qui solo una **preview** della policy: scala tutte le sub con lo stesso fattore
(priors invariati, distillazione) portando la classe più piccola a `MIN_CLASS_TARGET`.
"""
    ),
    code(
        """from eda.curation import proportional_train_subset

MIN_CLASS_TARGET = 1250

train_pool = curated_export
preview, sampling_summary = proportional_train_subset(
    train_pool, min_class_target=MIN_CLASS_TARGET, seed=cfg.seed
)
display(sampling_summary)

p_before = train_pool.groupby("sub_category").size() / len(train_pool)
p_after = preview.groupby("sub_category").size() / len(preview)
display(pd.concat([p_before.rename("p_before"), p_after.rename("p_after")], axis=1).round(4))
store.save_df("curation/train_sampling_preview.csv", sampling_summary)
"""
    ),
    md(
        """## Chiusura run

Il `run_id` è l'aggancio per associare questa EDA a un esperimento di training.
`EdaRunStore.list_runs()` elenca le run; `export_archive()` crea uno zip condivisibile.
"""
    ),
    code(
        """manifest = json.loads((store.root / "manifest.json").read_text())
print("run_id:", manifest["run_id"])
print("window:", manifest.get("window_start"), "→", manifest.get("window_end"))
display(pd.Series(manifest["artifacts"], name="artifacts"))
display(EdaRunStore.list_runs(cfg.output_dir))
# store.export_archive()  # zip della run per condividerla / allegarla a un esperimento
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python (eda-aic)", "language": "python", "name": "eda-aic"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
