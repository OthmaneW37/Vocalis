"""Étapes 1 et 2 : lecture du signal numérique et prétraitement."""

import librosa
import numpy as np
import soundfile as sf

from .audio_base import SR  # 16 kHz mono, constante partagée avec l'app


def infos_fichier(chemin: str) -> dict:
    """Caractéristiques du fichier AVANT prétraitement."""
    try:
        info = sf.info(chemin)
        return {
            "frequence": info.samplerate,
            "canaux": info.channels,
            "duree": info.duration,
            "format": f"{info.format} / {info.subtype}",
        }
    except Exception:
        # Formats que soundfile ne lit pas : on passe par librosa.
        y, sr = librosa.load(chemin, sr=None, mono=False)
        canaux = 1 if y.ndim == 1 else y.shape[0]
        return {"frequence": sr, "canaux": canaux,
                "duree": y.shape[-1] / sr, "format": "inconnu"}


def charger_audio(chemin: str, sr: int = SR) -> np.ndarray:
    """Lit un fichier audio en mono, rééchantillonné à `sr`, valeurs dans [-1, 1]."""
    audio, _ = librosa.load(chemin, sr=sr, mono=True)
    return audio.astype(np.float32)
