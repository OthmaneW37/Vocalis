"""Emplacement des données de l'app (réglages, historique, modèle maison)."""

import os
from pathlib import Path


def dossier_donnees() -> Path:
    """`%APPDATA%\\Vocalis`, ou le dossier donné par `VOCALIS_DOSSIER` (tests)."""
    force = os.environ.get("VOCALIS_DOSSIER")
    if force:
        dossier = Path(force)
    else:
        dossier = Path(os.environ.get("APPDATA") or Path.home()) / "Vocalis"
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier
