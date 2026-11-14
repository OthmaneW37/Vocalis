"""Jeu de données audio → (log-Mel, tokens), lots par durée et SpecAugment."""

import csv
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .features import SR, LogMel


@dataclass
class Exemple:
    chemin: Path
    duree: float
    texte: str


def lire_manifeste(chemin) -> list[Exemple]:
    """Lit un TSV `chemin, duree, texte` ; les chemins sont relatifs au manifeste.

    Les lignes dont le fichier audio manque sont ignorées (avec un avertissement).
    """
    chemin = Path(chemin)
    exemples, manquants = [], 0
    with chemin.open(encoding="utf-8", newline="") as f:
        for ligne in csv.DictReader(f, delimiter="\t"):
            audio = Path(ligne["chemin"])
            if not audio.is_absolute():
                audio = chemin.parent / audio
            if not audio.is_file():
                manquants += 1
                continue
            exemples.append(Exemple(audio, float(ligne["duree"]), ligne["texte"]))
    if manquants:
        warnings.warn(f"{chemin.name} : {manquants} fichier(s) audio introuvable(s), ignoré(s)")
    return exemples


def ecrire_manifeste(chemin, exemples: list[Exemple]) -> None:
    chemin = Path(chemin)
    with chemin.open("w", encoding="utf-8", newline="") as f:
        sortie = csv.writer(f, delimiter="\t")
        sortie.writerow(["chemin", "duree", "texte"])
        for e in exemples:
            try:
                relatif = Path(e.chemin).relative_to(chemin.parent).as_posix()
            except ValueError:
                relatif = str(e.chemin)
            sortie.writerow([relatif, f"{e.duree:.2f}", e.texte])


def charger_signal(chemin) -> np.ndarray:
    """Fichier audio → mono 16 kHz float32."""
    import soundfile as sf

    signal, sr = sf.read(str(chemin), dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        diviseur = gcd(SR, sr)
        signal = resample_poly(signal, SR // diviseur, sr // diviseur).astype(np.float32)
    return signal


class JeuAudio(Dataset):
    def __init__(self, exemples: list[Exemple], tokenizer):
        self.exemples = exemples
        self.tokenizer = tokenizer
        self.log_mel = LogMel()

    def __len__(self) -> int:
        return len(self.exemples)

    def __getitem__(self, i: int):
        exemple = self.exemples[i]
        signal = torch.from_numpy(charger_signal(exemple.chemin))
        mel = self.log_mel(signal[None])[0]                         # [80, T]
        cible = torch.tensor(self.tokenizer.encoder(exemple.texte), dtype=torch.long)
        return mel, cible


def assembler(lot):
    """Complète avec des zéros → (mels [B, 80, T], longueurs, cibles [B, L], longueurs)."""
    mels, cibles = zip(*lot)
    longueurs_mel = torch.tensor([m.shape[1] for m in mels])
    longueurs_cible = torch.tensor([len(c) for c in cibles])
    lot_mel = torch.zeros(len(mels), mels[0].shape[0], int(longueurs_mel.max()))
    lot_cible = torch.zeros(len(cibles), max(1, int(longueurs_cible.max())), dtype=torch.long)
    for i, (mel, cible) in enumerate(zip(mels, cibles)):
        lot_mel[i, :, : mel.shape[1]] = mel
        lot_cible[i, : len(cible)] = cible
    return lot_mel, longueurs_mel, lot_cible, longueurs_cible


class EchantillonneurParDuree:
    """Regroupe des énoncés de durées voisines ; un lot « pèse » au plus
    `secondes_par_lot` une fois complété à la durée du plus long."""

    def __init__(self, durees: list[float], secondes_par_lot: float,
                 melanger: bool = True, graine: int = 0):
        self.melanger = melanger
        self._hasard = random.Random(graine)
        self.lots: list[list[int]] = []
        courant: list[int] = []
        for i in sorted(range(len(durees)), key=lambda i: durees[i]):
            # Trié par durée croissante : le dernier ajouté est le plus long du lot.
            if courant and durees[i] * (len(courant) + 1) > secondes_par_lot:
                self.lots.append(courant)
                courant = []
            courant.append(i)
        if courant:
            self.lots.append(courant)

    def __len__(self) -> int:
        return len(self.lots)

    def __iter__(self):
        lots = list(self.lots)
        if self.melanger:
            self._hasard.shuffle(lots)
        return iter(lots)


class SpecAugment(nn.Module):
    """Masque des bandes de fréquence et des tranches de temps (entraînement seulement)."""

    def __init__(self, masques_freq: int = 2, largeur_freq: int = 27,
                 masques_temps: int = 5, part_temps: float = 0.05):
        super().__init__()
        self.masques_freq, self.largeur_freq = masques_freq, largeur_freq
        self.masques_temps, self.part_temps = masques_temps, part_temps

    def forward(self, mel: torch.Tensor, longueurs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mel
        mel = mel.clone()
        nb_bandes = mel.shape[1]
        for b in range(mel.shape[0]):
            for _ in range(self.masques_freq):
                largeur = random.randint(0, self.largeur_freq)
                debut = random.randint(0, max(0, nb_bandes - largeur))
                mel[b, debut: debut + largeur, :] = 0.0
            duree = int(longueurs[b])
            largeur_max = max(1, int(self.part_temps * duree))
            for _ in range(self.masques_temps):
                largeur = random.randint(0, largeur_max)
                debut = random.randint(0, max(0, duree - largeur))
                mel[b, :, debut: debut + largeur] = 0.0
        return mel
