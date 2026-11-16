"""Moteur Maison : le modèle Conformer + CTC entraîné de zéro dans `modele/`."""

import time
from pathlib import Path

import numpy as np

from stt.audio_base import SR
from stt.resultat import Resultat

from ..chemins import dossier_donnees


class ModeleMaisonAbsent(Exception):
    """Aucun checkpoint entraîné n'est disponible."""


def chemin_checkpoint() -> Path:
    return dossier_donnees() / "modele_maison.pt"


class MoteurMaison:
    nom = "Modèle maison"

    def __init__(self, checkpoint: Path | None = None):
        checkpoint = Path(checkpoint or chemin_checkpoint())
        if not checkpoint.is_file():
            raise ModeleMaisonAbsent(str(checkpoint))
        # PyTorch n'est importé qu'ici : l'app reste légère avec le moteur Pro.
        from modele.inference import Transcripteur

        self._transcripteur = Transcripteur(checkpoint)

    def transcrire(self, audio: np.ndarray, langue: str | None = None) -> Resultat:
        if len(audio) == 0:
            return Resultat("", langue="fr")
        t0 = time.perf_counter()
        segments = self._transcripteur.transcrire(audio)
        return Resultat(
            texte=" ".join(t for _, _, t in segments),
            segments=segments, langue="fr",
            duree_audio=len(audio) / SR,
            temps_calcul=time.perf_counter() - t0)
