# EDA AIC — segmenti documentali assicurativi

Progetto standalone di EDA per i segmenti AIC anonimizzati (etichette **silver** dal teacher,
tassonomia macro → micro). Produce analisi O1–O7 secondo [docs/eda_strategy.md](docs/eda_strategy.md),
un dataset **curated** (regole A1–A5) e un **run store** versionato associabile agli esperimenti
di training a valle.

## Setup

Richiede [uv](https://docs.astral.sh/uv/) e Python ≥ 3.12.

```bash
make install        # core + ml + notebook
make install-s3     # in più s3fs (SageMaker / parquet su S3)
make kernel         # registra il kernel Jupyter "Python (eda-aic)"
```

## Dati in ingresso

Un parquet di segmenti (locale o `s3://…`) con almeno le colonne:
`record_key`, `document_id`, `segment_id`, `created_at`, `macro_category`,
`sub_category`, `pages_list`, `ocr_confidence`, `ocr_surrogates`,
`ocr_num_words`, `ocr_num_words_handwritten`, `ocr_handwritten_rate`.

Il path si passa via CLI (`--data`), via env var `EDA_DATA_PATH`, o nel notebook.
I dati **non vanno mai committati** (cartella `data/` gitignored).

## Uso

### Notebook (interattivo)

```bash
make notebook   # apre notebooks/00_eda.ipynb
```

Il notebook è lineare e segue l'ordine della strategia: O1 → O2 → O3 → O6 → O4 → O5 → O7
→ Decisioni → Curation. Tutta la logica sta nel package `eda`; il notebook orchestra e mostra.

### CLI (headless, automatizzabile)

```bash
uv run eda-run --data /path/to/segments.parquet --tag nov2026
# oppure: make run DATA=/path/to/segments.parquet TAG=nov2026
```

Opzioni principali: `--window-start/--window-end` (filtro su `created_at`),
`--macro` (una sola macro), `--skip-bge`, `--token-scan`, `--tfidf-full`,
`--output` (default `data/`). `eda-run --help` per l'elenco completo.

## Risultati (run store)

Ogni esecuzione scrive `data/runs/<run_id>/` con:

- `manifest.json` — run id, finestra temporale dei dati, path + sha256 del parquet,
  config completa, versione del package;
- artefatti per obiettivo (`o1_sub_summary.csv`, `o4/probe_summary.csv`,
  `o5_suspected_mislabels.parquet`, `o6_leakage_duplicates.json`, …);
- `decisions.json` — decisioni finali;
- `curation/` — `df_segments_curated_v1.parquet`, `curation_manifest.json`,
  `exclude_from_train.parquet`.

Il `run_id` è l'aggancio per associare l'EDA a un esperimento di training
(es. citarlo nel manifest dell'esperimento).

## Struttura

```
src/eda/
  config.py      # EdaConfig (path, finestra temporale, soglie, seed)
  taxonomy.py    # MICRO_PER_MACRO, perimetro
  text.py        # normalize_text, text_hash
  io.py          # load parquet locale/S3 + filtro finestra
  leakage.py     # check_no_group_leakage
  tokens.py      # token_length_report, recommend_max_seq_len
  prompts.py     # prompt minimale per il budget token
  analyses/      # O1..O7 come funzioni pure (DataFrame in → DataFrame/dict out)
  curation.py    # apply_curation_v1 (A1–A5) + proportional_train_subset
  results.py     # EdaRunStore
  run.py         # CLI headless
notebooks/00_eda.ipynb
docs/eda_strategy.md
tests/
```

## Sviluppo

```bash
make test   # pytest su dati sintetici
make lint   # ruff
```
