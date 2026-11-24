"""Headless EDA pipeline: O1 -> O2 -> O3 -> O6 -> O4 -> O5 -> O7 -> decisions -> curation.

Every artifact lands in the run store (data/runs/<run_id>/); the notebook covers
the same steps interactively with plots.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .analyses import o1_volumes, o2_text, o3_signals, o5_labels, o6_dups, o7_drift
from .config import EdaConfig, Thresholds
from .curation import apply_curation_v1
from .io import load_segments, prepare_labeled
from .results import EdaRunStore

logger = logging.getLogger("eda")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eda-run", description="EDA AIC headless (O1-O7 + curation)")
    p.add_argument("--data", default=None, help="Parquet segmenti (default: env EDA_DATA_PATH)")
    p.add_argument("--output", default=None, help="Output dir (default: env EDA_OUTPUT_DIR o data/)")
    p.add_argument("--tag", default="eda", help="Suffisso del run_id (es. nov2026)")
    p.add_argument("--window-start", default=None, help="Filtro created_at >= (ISO date)")
    p.add_argument("--window-end", default=None, help="Filtro created_at <= (ISO date)")
    p.add_argument("--macro", default=None, help="Limita a una macro (es. documentazione_medica)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tfidf-full", action="store_true", help="O4 TF-IDF su tutto il pool (lento)")
    p.add_argument("--skip-bge", action="store_true", help="Salta il probe bge-m3 (O4)")
    p.add_argument("--skip-probe", action="store_true", help="Salta O4/O5 (niente TF-IDF/cleanlab)")
    p.add_argument("--token-scan", action="store_true", help="O2.3: scan token con i tokenizer HF")
    p.add_argument("--drift-cutoff", default=None, help="Data di cutoff per O7 era share (ISO)")
    p.add_argument("--splits", default=None, help="Parquet split esistente da verificare in O6")
    return p.parse_args(argv)


def run_pipeline(cfg: EdaConfig, *, skip_bge: bool = True, skip_probe: bool = False,
                 token_scan: bool = False, drift_cutoff: str | None = None,
                 splits_path: str | None = None) -> EdaRunStore:
    th = cfg.thresholds
    store = EdaRunStore(cfg)
    logger.info("run_id=%s output=%s", store.run_id, store.root)

    # Load + perimetro
    df = load_segments(cfg)
    labeled_df, out_of_taxonomy, stats = prepare_labeled(df, cfg)
    n_days = max(stats["days_with_data"], 1)
    store.update_manifest(dataset_stats=stats)
    logger.info("segments in perimeter: %s (days=%s)", len(labeled_df), n_days)

    # O1
    sub_sum = o1_volumes.sub_summary(labeled_df, n_days, th)
    store.save_df("o1_sub_summary.csv", sub_sum)
    store.save_df("o1_macro_summary.csv", o1_volumes.macro_summary(labeled_df, n_days), index=True)
    plan, test_docs = o1_volumes.test_sizing(labeled_df, n_days, th)
    store.save_df("o1_collection_plan.csv", plan, index=True)

    # O2
    flagged = o2_text.add_quality_flags(labeled_df, th)
    store.save_df("o2_length_summary.csv", o2_text.length_summary(flagged), index=True)
    store.save_df("o2_quality_summary.csv", o2_text.quality_summary(flagged), index=True)
    store.save_df("o2_flag_impact.csv", o2_text.flag_impact(flagged), index=True)

    selected_max_seq_len = None
    if token_scan:
        from transformers import AutoTokenizer

        from .tokens import recommend_max_seq_len, token_length_report

        scan_df = o2_text.token_scan_frame(flagged, th, seed=cfg.seed)
        lengths_by_model: dict[str, pd.Series] = {}
        coverage_frames = []
        for model_id in ("Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM3-3B", "google/gemma-3-4b-it"):
            try:
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            except Exception as exc:  # gated models need HF_TOKEN
                logger.warning("skip tokenizer %s: %s", model_id, exc)
                continue
            lengths, cov = token_length_report(tok, scan_df, thresholds=th.candidate_max_seq_lengths)
            lengths_by_model[model_id] = lengths
            cov["model"] = model_id
            coverage_frames.append(cov)
        if lengths_by_model:
            selected_max_seq_len = recommend_max_seq_len(lengths_by_model, headroom=0.20, multiple=1024)
            store.save_df("o2_token_coverage.csv", pd.concat(coverage_frames, ignore_index=True))
            logger.info("recommended max_seq_len (sample): %s", selected_max_seq_len)

    # O3
    signals = o3_signals.signal_table(flagged)
    if len(signals):
        store.save_df("o3_signal_patterns.csv", signals)

    # O6
    o6_stats = o6_dups.leakage_stats(flagged, splits_path)
    quasi_by_macro, quasi_extra = o6_dups.quasi_dup_by_macro(flagged, th, seed=cfg.seed)
    o6_stats = {**o6_stats, **quasi_extra}
    store.save_json("o6_leakage_duplicates.json", o6_stats)
    if len(quasi_by_macro):
        store.save_df("o6_quasi_dup_by_macro.csv", quasi_by_macro, index=True)

    # O4 + O5
    cleanlab_issues = pd.DataFrame()
    cleanlab_summary = pd.DataFrame()
    probe_summary = pd.DataFrame()
    if not skip_probe:
        from .analyses import o4_probe

        tfidf_results, probe_summary = o4_probe.run_tfidf_probes(flagged, th, seed=cfg.seed)
        if not skip_bge:
            _, bge_summary = o4_probe.run_bge_probes(flagged, th, seed=cfg.seed)
            probe_summary = pd.concat([probe_summary, bge_summary], ignore_index=True)
        if len(probe_summary):
            store.save_df("o4/probe_summary.csv", probe_summary)
        if tfidf_results:
            oof_frames = []
            for macro, res in tfidf_results.items():
                oof = res["frame"][["record_key", "document_id", "sub_category"]].copy()
                oof["macro_category"] = macro
                oof["oof_pred"] = res["oof_pred"]
                oof_frames.append(oof)
            store.save_df("o4/tfidf_oof_predictions.parquet", pd.concat(oof_frames, ignore_index=True))

            cleanlab_summary, cleanlab_issues = o5_labels.cleanlab_label_issues(tfidf_results)
            store.save_df("o5_cleanlab_by_macro.csv", cleanlab_summary)
            if len(cleanlab_issues):
                store.save_df("o5_suspected_mislabels.parquet", cleanlab_issues)
            store.save_json("o5_gold_sketch.json", o5_labels.gold_sketch(sub_sum))

    # O7
    store.save_df("o7_weekly_segments.csv", o7_drift.weekly_volumes(flagged).to_frame(), index=True)
    if drift_cutoff:
        store.save_df("o7_era_share.csv", o7_drift.era_share(flagged, drift_cutoff), index=True)

    # Decisioni
    decisions = {
        "text_column": cfg.text_column,
        "segments_in_perimeter": int(len(labeled_df)),
        "empty_text_exclude_eval": int(flagged["empty_text"].sum()),
        "out_of_taxonomy_excluded": int(len(out_of_taxonomy)),
        "test_documents_per_sub": int(test_docs),
        "eda_max_seq_length_sample": selected_max_seq_len,
        "tfidf_probe_macro_f1": (
            probe_summary[probe_summary["representation"] == "tfidf"]
            .set_index("macro_category")["macro_f1"].to_dict()
            if len(probe_summary)
            else {}
        ),
        "rare_subs_below_min_train": sub_sum[sub_sum["segments"] < th.min_segments_train][
            ["macro_category", "sub_category", "segments", "status"]
        ].to_dict(orient="records"),
    }
    store.save_json("decisions.json", decisions)

    # Curation (A1-A5) sul corpus in perimetro
    curated, cur_stats = apply_curation_v1(
        flagged,
        text_col=cfg.text_column,
        cleanlab_issues=cleanlab_issues if len(cleanlab_issues) else None,
        macro_focus=None,  # perimetro/macro_focus già applicati in prepare_labeled
        seed=cfg.seed,
    )
    drop_cols = [
        c
        for c in (
            "n_words", "n_pages", "is_short", "empty_text",
            "low_confidence", "mostly_handwritten", "text_hash",
        )
        if c in curated.columns
    ]
    curated_export = curated.drop(columns=drop_cols)
    store.save_df("curation/df_segments_curated_v1.parquet", curated_export)
    store.save_json("curation/curation_manifest.json", cur_stats)
    excluded = curated_export[curated_export["exclude_from_train"] == 1]
    if len(excluded):
        store.save_df("curation/exclude_from_train.parquet", excluded)

    logger.info("done: %s artifacts under %s", len(store.artifacts), store.root)
    return store


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args(argv or sys.argv[1:])
    th = Thresholds(tfidf_full_data=args.tfidf_full)
    cfg = EdaConfig(
        dataset_path=args.data or EdaConfig().dataset_path,
        output_dir=args.output or EdaConfig().output_dir,
        tag=args.tag,
        seed=args.seed,
        window_start=args.window_start,
        window_end=args.window_end,
        macro_focus=args.macro,
        thresholds=th,
    )
    store = run_pipeline(
        cfg,
        skip_bge=args.skip_bge,
        skip_probe=args.skip_probe,
        token_scan=args.token_scan,
        drift_cutoff=args.drift_cutoff,
        splits_path=args.splits,
    )
    print(f"run_id: {store.run_id}")
    print(f"output: {store.root}")


if __name__ == "__main__":
    main()
