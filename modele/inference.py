"""Chargement d'un checkpoint et transcription d'un signal avec le modèle maison."""

from pathlib import Path

import numpy as np
import torch

from .conformer import ConfigModele, Conformer
from .decodage import decoder_glouton
from .features import HOP, SR, LogMel
from .tokenizer import TokenizerBPE

TRANCHE_MAX_S = 20.0        # durée maximale passée au modèle en une fois
RECHERCHE_S = 5.0           # on cherche le creux d'énergie dans les 5 dernières secondes


def sauver_checkpoint(chemin, modele: Conformer, tokenizer: TokenizerBPE, **infos) -> None:
    """Écrit un checkpoint autonome : configuration, vocabulaire et poids."""
    torch.save({"config": modele.config.vers_dict(),
                "fusions": [list(f) for f in tokenizer.fusions],
                "poids": modele.state_dict(), **infos}, str(chemin))


def charger_checkpoint(chemin, appareil="cpu"):
    """→ (modele en mode eval, tokenizer, dictionnaire brut du checkpoint)."""
    donnees = torch.load(str(chemin), map_location="cpu", weights_only=True)
    tokenizer = TokenizerBPE(donnees["fusions"])
    modele = Conformer(ConfigModele(**donnees["config"]))
    modele.load_state_dict(donnees["poids"])
    return modele.to(appareil).eval(), tokenizer, donnees


def decouper(audio: np.ndarray, sr: int = SR) -> list[tuple[int, int]]:
    """Découpe en tranches d'au plus TRANCHE_MAX_S, en coupant dans les creux d'énergie."""
    tranches, debut = [], 0
    maxi, recherche, trame = int(TRANCHE_MAX_S * sr), int(RECHERCHE_S * sr), int(0.03 * sr)
    while len(audio) - debut > maxi:
        zone = audio[debut + maxi - recherche: debut + maxi]
        nb = len(zone) // trame
        energie = (zone[: nb * trame].reshape(nb, trame) ** 2).mean(axis=1)
        coupe = debut + maxi - recherche + (int(energie.argmin()) * trame + trame // 2)
        tranches.append((debut, coupe))
        debut = coupe
    if len(audio) - debut > 0:
        tranches.append((debut, len(audio)))
    return tranches


class Transcripteur:
    def __init__(self, checkpoint: Path):
        self.modele, self.tokenizer, self.infos = charger_checkpoint(checkpoint)
        self.log_mel = LogMel()

    @torch.no_grad()
    def transcrire(self, audio: np.ndarray) -> list[tuple[float, float, str]]:
        """Signal mono 16 kHz → [(début, fin, texte)], une entrée par tranche non vide."""
        segments = []
        for debut, fin in decouper(np.asarray(audio, dtype=np.float32)):
            if fin - debut < 4 * HOP * 4:        # trop court pour le sous-échantillonnage
                continue
            mel = self.log_mel(torch.from_numpy(np.ascontiguousarray(audio[debut:fin]))[None])
            log_probas, longueurs = self.modele(mel, torch.tensor([mel.shape[2]]))
            texte = decoder_glouton(log_probas, longueurs, self.tokenizer)[0]
            if texte:
                segments.append((debut / SR, fin / SR, texte))
        return segments
