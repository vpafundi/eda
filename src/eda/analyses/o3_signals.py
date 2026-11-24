"""O3 - Lexical signal patterns surviving anonymization, per sub."""

from __future__ import annotations

import pandas as pd

SIGNAL_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "documento_di_spesa": [
        ("iva", r"\biva\b|imposta"),
        ("importo_euro", r"€|eur\b|totale"),
        ("scontrino_kw", r"scontrino|corrispettiv"),
        ("fattura_kw", r"fattura|fatt\.?\s*n"),
        ("ricevuta_kw", r"ricevuta|pagamento"),
        ("preventivo_kw", r"preventivo|stima"),
    ],
    "documento_di_spesa_medica": [
        ("ticket_ssn", r"ticket|ssn|servizio sanitario|tesser"),
        ("fattura_kw", r"fattura"),
        ("importo_euro", r"€|eur\b"),
    ],
    "documentazione_medica": [
        ("pronto_soccorso", r"pronto\s*soccorso|ps\b|emergenz"),
        ("dimissioni", r"dimission|reparto"),
        ("prescrizione", r"prescriz|farmaco|ricetta"),
        ("certificato", r"certificat|guarigion"),
    ],
    "denuncia": [
        ("cai", r"\bcai\b|constatazione|modulo\s*blu"),
        ("danni", r"dann|risarciment|sinistro"),
    ],
}


def signal_table(flagged_df: pd.DataFrame, patterns: dict[str, list[tuple[str, str]]] | None = None) -> pd.DataFrame:
    """Hit rate of each pattern per (macro, sub) on non-empty text."""
    patterns = patterns or SIGNAL_PATTERNS
    rows = []
    for macro, pats in patterns.items():
        sub_df = flagged_df[(flagged_df["macro_category"] == macro) & (~flagged_df["empty_text"])]
        if sub_df.empty:
            continue
        for sub, grp in sub_df.groupby("sub_category"):
            text_l = grp["text"].str.lower()
            row = {"macro_category": macro, "sub_category": sub, "segments": len(grp)}
            for name, pat in pats:
                row[f"hit_{name}"] = text_l.str.contains(pat, regex=True, na=False).mean()
            rows.append(row)
    return pd.DataFrame(rows).round(3)
