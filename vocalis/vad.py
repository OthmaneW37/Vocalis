"""Détection de voix par énergie : coupe les silences de début et de fin."""

import numpy as np

from stt.audio_base import SR

TRAME_S = 0.03          # trames de 30 ms
SEUIL_MIN = 0.005       # en dessous, c'est du silence quel que soit le bruit de fond
SEUIL_MAX = 0.05        # au-dessus, c'est de la voix même si on parle sans pause
TRAMES_MIN = 3          # au moins 90 ms actives, sinon c'est un clic


def rogner_silences(audio: np.ndarray, sr: int = SR, marge_s: float = 0.2) -> np.ndarray:
    """Rend la partie parlée du signal, ou un tableau vide s'il n'y a pas de voix.

    Le seuil vaut 4 fois le plancher de bruit (10ᵉ centile de l'énergie des
    trames), borné entre SEUIL_MIN et SEUIL_MAX.
    """
    vide = np.zeros(0, dtype=np.float32)
    n = int(TRAME_S * sr)
    nb = len(audio) // n
    if nb == 0:
        return vide
    trames = np.asarray(audio[: nb * n], dtype=np.float32).reshape(nb, n)
    rms = np.sqrt((trames ** 2).mean(axis=1))
    seuil = float(np.clip(4 * np.percentile(rms, 10), SEUIL_MIN, SEUIL_MAX))
    actives = np.flatnonzero(rms > seuil)
    if len(actives) < TRAMES_MIN:
        return vide
    marge = int(marge_s * sr)
    debut = max(0, actives[0] * n - marge)
    fin = min(len(audio), (actives[-1] + 1) * n + marge)
    return np.asarray(audio[debut:fin], dtype=np.float32)
