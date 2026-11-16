"""Encodeur Conformer + tête CTC, écrit à partir des briques de base de PyTorch.

Référence : Gulati et al., « Conformer: Convolution-augmented Transformer for
Speech Recognition » (2020). L'attention utilise des positions relatives par
rotation (RoPE) plutôt que le décalage relatif de Transformer-XL.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

N_MELS = 80


@dataclass
class ConfigModele:
    d: int = 256                # largeur du modèle
    couches: int = 16
    tetes: int = 4
    noyau_conv: int = 15
    dropout: float = 0.1
    taille_vocab: int = 500

    @classmethod
    def moyen(cls, **k) -> "ConfigModele":      # ~26 M de paramètres
        return cls(**k)

    @classmethod
    def petit(cls, **k) -> "ConfigModele":      # ~8 M de paramètres (entraînement CPU)
        return cls(d=144, **k)

    def vers_dict(self) -> dict:
        return asdict(self)


def longueurs_sorties(longueurs: torch.Tensor) -> torch.Tensor:
    """Nombre de trames après le sous-échantillonnage ×4 (deux convolutions de pas 2)."""
    apres_1 = torch.div(longueurs - 1, 2, rounding_mode="floor")
    return torch.div(apres_1 - 1, 2, rounding_mode="floor").clamp_min(0)


class SousEchantillonnage(nn.Module):
    """[B, 80, T] → [B, T/4, d] : deux convolutions 3×3 de pas 2, puis une projection."""

    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d, 3, stride=2), nn.ReLU(),
            nn.Conv2d(d, d, 3, stride=2), nn.ReLU())
        bandes = ((N_MELS - 1) // 2 - 1) // 2
        self.projection = nn.Linear(d * bandes, d)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv(mel.transpose(1, 2).unsqueeze(1))        # [B, d, T', bandes]
        b, d, t, f = x.shape
        return self.projection(x.permute(0, 2, 1, 3).reshape(b, t, d * f))


def _rotation(x: torch.Tensor) -> torch.Tensor:
    """Applique la rotation positionnelle (RoPE) à x de forme [B, têtes, T, dim]."""
    dim = x.shape[-1]
    positions = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)
    frequences = torch.exp(torch.arange(0, dim, 2, device=x.device, dtype=x.dtype)
                           * (-math.log(10000.0) / dim))
    angles = positions[:, None] * frequences[None, :]           # [T, dim/2]
    cos, sin = angles.cos(), angles.sin()
    pair, impair = x[..., 0::2], x[..., 1::2]
    return torch.stack((pair * cos - impair * sin, pair * sin + impair * cos),
                       dim=-1).flatten(-2)


class Attention(nn.Module):
    """Auto-attention multi-têtes ; le masque écarte les trames de remplissage."""

    def __init__(self, d: int, tetes: int, dropout: float):
        super().__init__()
        assert d % tetes == 0 and (d // tetes) % 2 == 0
        self.tetes = tetes
        self.qkv = nn.Linear(d, 3 * d)
        self.sortie = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (m.view(b, t, self.tetes, d // self.tetes).transpose(1, 2)
                   for m in self.qkv(x).chunk(3, dim=-1))
        q, k = _rotation(q), _rotation(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.tetes)
        scores = scores.masked_fill(~masque[:, None, None, :], float("-inf"))
        poids = self.dropout(scores.softmax(dim=-1))
        return self.sortie((poids @ v).transpose(1, 2).reshape(b, t, d))


class ModuleConvolution(nn.Module):
    """pointwise → GLU → depthwise → BatchNorm → Swish → pointwise."""

    def __init__(self, d: int, noyau: int, dropout: float):
        super().__init__()
        self.norme = nn.LayerNorm(d)
        self.entree = nn.Conv1d(d, 2 * d, 1)
        self.profondeur = nn.Conv1d(d, d, noyau, padding=noyau // 2, groups=d)
        self.norme_lot = nn.BatchNorm1d(d)
        self.activation = nn.SiLU()
        self.sortie = nn.Conv1d(d, d, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:
        y = self.entree(self.norme(x).transpose(1, 2))          # [B, 2d, T]
        a, porte = y.chunk(2, dim=1)
        y = a * porte.sigmoid()                                 # GLU
        y = y.masked_fill(~masque[:, None, :], 0.0)             # le remplissage reste nul
        y = self.sortie(self.activation(self.norme_lot(self.profondeur(y))))
        return self.dropout(y.transpose(1, 2))


class FeedForward(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reseau(x)


class BlocConformer(nn.Module):
    def __init__(self, c: ConfigModele):
        super().__init__()
        self.ff1 = FeedForward(c.d, c.dropout)
        self.norme_attention = nn.LayerNorm(c.d)
        self.attention = Attention(c.d, c.tetes, c.dropout)
        self.dropout = nn.Dropout(c.dropout)
        self.convolution = ModuleConvolution(c.d, c.noyau_conv, c.dropout)
        self.ff2 = FeedForward(c.d, c.dropout)
        self.norme = nn.LayerNorm(c.d)

    def forward(self, x: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.dropout(self.attention(self.norme_attention(x), masque))
        x = x + self.convolution(x, masque)
        x = x + 0.5 * self.ff2(x)
        return self.norme(x)


class Conformer(nn.Module):
    def __init__(self, config: ConfigModele):
        super().__init__()
        self.config = config
        self.entree = SousEchantillonnage(config.d)
        self.dropout = nn.Dropout(config.dropout)
        self.blocs = nn.ModuleList(BlocConformer(config) for _ in range(config.couches))
        self.tete = nn.Linear(config.d, config.taille_vocab)

    def forward(self, mel: torch.Tensor, longueurs: torch.Tensor, intermediaire: bool = False):
        """mel [B, 80, T], longueurs [B] (en trames) → (log-probas [B, T/4, vocab], longueurs).

        Avec `intermediaire=True`, rend aussi les log-probas lues à mi-profondeur
        par la même tête : une perte CTC auxiliaire à cet endroit (« InterCTC »)
        aide les encodeurs profonds à sortir du palier de début d'entraînement.
        """
        x = self.dropout(self.entree(mel))
        longueurs = longueurs_sorties(longueurs).clamp_max(x.shape[1])
        masque = torch.arange(x.shape[1], device=x.device)[None, :] < longueurs[:, None]
        milieu = None
        for i, bloc in enumerate(self.blocs, 1):
            x = bloc(x, masque)
            if intermediaire and i == len(self.blocs) // 2:
                milieu = self.tete(x).log_softmax(dim=-1)
        finales = self.tete(x).log_softmax(dim=-1)
        if intermediaire:
            return finales, longueurs, (milieu if milieu is not None else finales)
        return finales, longueurs

    def nb_parametres(self) -> int:
        return sum(p.numel() for p in self.parameters())
