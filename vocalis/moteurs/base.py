"""Interface commune des moteurs de transcription."""

from typing import Protocol

import numpy as np

from stt.resultat import Resultat


class Moteur(Protocol):
    nom: str

    def transcrire(self, audio: np.ndarray, langue: str | None = None) -> Resultat:
        """Transcrit un signal mono 16 kHz (float32 dans [-1, 1])."""
        ...
