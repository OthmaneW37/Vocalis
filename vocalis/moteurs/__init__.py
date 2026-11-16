"""Moteurs de transcription interchangeables."""

from .base import Moteur
from .maison import ModeleMaisonAbsent, MoteurMaison, chemin_checkpoint
from .pro import MoteurPro

__all__ = ["Moteur", "MoteurPro", "MoteurMaison", "ModeleMaisonAbsent",
           "chemin_checkpoint", "creer_moteur"]

AVERTISSEMENT_REPLI = ("Le modèle maison n'est pas encore entraîné : "
                       "le moteur Whisper est utilisé à la place.")


def creer_moteur(reglages) -> tuple[Moteur, str | None]:
    """Construit le moteur demandé par les réglages.

    Rend `(moteur, avertissement)` ; l'avertissement n'est pas `None` quand le
    moteur Maison a été demandé sans checkpoint et qu'on s'est replié sur Pro.
    """
    if reglages.moteur == "maison":
        try:
            return MoteurMaison(), None
        except ModeleMaisonAbsent:
            return MoteurPro(reglages.modele_pro), AVERTISSEMENT_REPLI
    return MoteurPro(reglages.modele_pro), None
