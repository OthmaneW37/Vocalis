"""Moteur Pro : Whisper via faster-whisper (CTranslate2), sur CPU en int8."""

import time

import numpy as np

from stt.audio_base import SR
from stt.resultat import Resultat


class MoteurPro:
    def __init__(self, taille: str = "small"):
        from faster_whisper import WhisperModel

        self.nom = f"Whisper {taille}"
        options = dict(device="cpu", compute_type="int8")
        try:
            # Modèle déjà en cache : démarrage immédiat, sans accès réseau.
            self._modele = WhisperModel(taille, local_files_only=True, **options)
        except Exception:
            self._modele = WhisperModel(taille, **options)

    def transcrire(self, audio: np.ndarray, langue: str | None = None) -> Resultat:
        duree = len(audio) / SR
        if len(audio) == 0:
            return Resultat("", duree_audio=0.0)
        t0 = time.perf_counter()
        morceaux, info = self._modele.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=langue, beam_size=5, vad_filter=True,
            condition_on_previous_text=False)
        segments = [(s.start, s.end, s.text.strip()) for s in morceaux]
        segments = [s for s in segments if s[2]]
        return Resultat(
            texte=" ".join(t for _, _, t in segments),
            segments=segments,
            langue=info.language or (langue or ""),
            probas_langues=[(info.language, float(info.language_probability))],
            duree_audio=duree,
            temps_calcul=time.perf_counter() - t0)
