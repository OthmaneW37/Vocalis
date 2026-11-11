"""Évaluation : Word Error Rate (WER)."""

import re
import string
import unicodedata

import jiwer

_PONCTUATION = string.punctuation + "«»“”’…–—"


def normaliser(texte: str) -> str:
    """Minuscules, apostrophes unifiées, ponctuation retirée, espaces simplifiés."""
    texte = unicodedata.normalize("NFC", texte).lower().replace("’", "'")
    texte = texte.replace("'", "' ")  # « l'ordinateur » → « l' ordinateur »
    texte = texte.translate(str.maketrans({c: " " for c in _PONCTUATION if c != "'"}))
    return re.sub(r"\s+", " ", texte).strip()


def wer(reference: str, hypothese: str) -> dict:
    """WER = (S + D + I) / N, avec le détail des erreurs."""
    ref, hyp = normaliser(reference), normaliser(hypothese)
    r = jiwer.process_words(ref, hyp)
    n = r.hits + r.substitutions + r.deletions
    return {
        "wer": r.wer,
        "substitutions": r.substitutions,
        "suppressions": r.deletions,
        "insertions": r.insertions,
        "mots_reference": n,
    }
