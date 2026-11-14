"""Capture du micro en mono 16 kHz."""

import threading
from typing import Callable

import numpy as np

from stt.audio_base import SR


class MicroIndisponible(Exception):
    """Le micro n'a pas pu être ouvert (absent, débranché, refusé par Windows)."""


def lister_micros() -> list[tuple[int, str]]:
    """[(index, nom)] des périphériques d'entrée de l'API audio par défaut."""
    import sounddevice as sd

    try:
        api_defaut = sd.default.hostapi
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0 and d["hostapi"] == api_defaut]
    except Exception:
        return []


def _flux_sounddevice(**options):
    import sounddevice as sd

    return sd.InputStream(**options)


class Enregistreur:
    """Accumule les blocs audio entre `demarrer()` et `arreter()`."""

    def __init__(self, peripherique: int | None = None,
                 sur_niveau: Callable[[float], None] | None = None,
                 fabrique_flux: Callable = _flux_sounddevice):
        self.peripherique = peripherique
        self.sur_niveau = sur_niveau
        self._fabrique = fabrique_flux
        self._flux = None
        self._blocs: list[np.ndarray] = []
        self._verrou = threading.Lock()

    def _rappel(self, donnees, nb_trames, temps, statut):
        bloc = np.asarray(donnees, dtype=np.float32).reshape(len(donnees), -1)[:, 0].copy()
        with self._verrou:
            self._blocs.append(bloc)
        if self.sur_niveau is not None and len(bloc):
            self.sur_niveau(float(np.sqrt((bloc ** 2).mean())))

    def demarrer(self) -> None:
        self._fermer()
        with self._verrou:
            self._blocs = []
        try:
            self._flux = self._fabrique(samplerate=SR, channels=1, dtype="float32",
                                        device=self.peripherique, callback=self._rappel)
            self._flux.start()
        except Exception as e:
            self._flux = None
            raise MicroIndisponible(f"Micro indisponible : {e}") from e

    def arreter(self) -> np.ndarray:
        self._fermer()
        with self._verrou:
            blocs, self._blocs = self._blocs, []
        if not blocs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocs)

    def _fermer(self) -> None:
        flux, self._flux = self._flux, None
        if flux is None:
            return
        try:
            flux.stop()
            flux.close()
        except Exception:
            pass   # micro débranché pendant l'enregistrement : on garde l'audio reçu
