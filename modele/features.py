"""Spectrogramme log-Mel en PyTorch, identique à `stt.features.log_mel`.

Le banc de filtres Mel (échelle de Slaney, filtres triangulaires normalisés
en aire) est recalculé ici à la main, sans librosa.
"""

import math

import numpy as np
import torch
from torch import nn

SR = 16000
N_FFT = 400        # fenêtre de 25 ms
HOP = 160          # pas de 10 ms → 100 trames par seconde
N_MELS = 80


def _hz_vers_mel(f):
    f = np.asarray(f, dtype=np.float64)
    pas = 200.0 / 3                                  # partie linéaire sous 1 kHz
    mel = f / pas
    coude_mel, pas_log = 1000.0 / pas, math.log(6.4) / 27.0
    haut = f >= 1000.0
    return np.where(haut, coude_mel + np.log(np.maximum(f, 1e-10) / 1000.0) / pas_log, mel)


def _mel_vers_hz(m):
    m = np.asarray(m, dtype=np.float64)
    pas = 200.0 / 3
    coude_mel, pas_log = 1000.0 / pas, math.log(6.4) / 27.0
    return np.where(m >= coude_mel, 1000.0 * np.exp(pas_log * (m - coude_mel)), pas * m)


def banc_mel(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Matrice (n_mels, n_fft // 2 + 1) de filtres triangulaires."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    centres = _mel_vers_hz(np.linspace(_hz_vers_mel(0.0), _hz_vers_mel(sr / 2), n_mels + 2))
    ecarts = np.diff(centres)
    rampes = centres[:, None] - freqs[None, :]
    filtres = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        montee = -rampes[i] / ecarts[i]
        descente = rampes[i + 2] / ecarts[i + 1]
        filtres[i] = np.maximum(0, np.minimum(montee, descente))
    filtres *= (2.0 / (centres[2:] - centres[:-2]))[:, None]     # aire constante
    return filtres.astype(np.float32)


class LogMel(nn.Module):
    """audio [B, T] → log-Mel [B, 80, T // 160], mis à l'échelle comme Whisper."""

    def __init__(self):
        super().__init__()
        self.register_buffer("fenetre", torch.hann_window(N_FFT, periodic=True),
                             persistent=False)
        self.register_buffer("filtres", torch.from_numpy(banc_mel()), persistent=False)

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(audio, N_FFT, HOP, window=self.fenetre, center=True,
                          pad_mode="reflect", return_complex=True)
        puissance = stft[..., :-1].abs() ** 2
        mel = self.filtres @ puissance
        log = torch.log10(mel.clamp_min(1e-10))
        plancher = log.amax(dim=(-2, -1), keepdim=True) - 8.0
        return (torch.maximum(log, plancher) + 4.0) / 4.0
