"""Résultat d'une transcription et export en sous-titres SRT.

Module volontairement sans dépendance lourde : il est partagé par la démo
(Whisper via Transformers) et par l'app de dictée, qui ne charge pas PyTorch.
"""

from dataclasses import dataclass, field


@dataclass
class Resultat:
    texte: str
    segments: list = field(default_factory=list)   # [(début, fin, texte)]
    langue: str = ""
    probas_langues: list = field(default_factory=list)
    duree_audio: float = 0.0
    temps_calcul: float = 0.0

    @property
    def facteur_temps_reel(self) -> float:
        return self.temps_calcul / self.duree_audio if self.duree_audio else 0.0


def _horodatage_srt(s: float) -> str:
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"


def vers_srt(segments, duree_audio: float) -> str:
    """Segments horodatés → sous-titres au format SRT."""
    blocs = []
    for i, (debut, fin, texte) in enumerate(segments, 1):
        fin = duree_audio if fin is None else fin
        blocs.append(f"{i}\n{_horodatage_srt(debut or 0)} --> "
                     f"{_horodatage_srt(fin)}\n{texte}\n")
    return "\n".join(blocs)
