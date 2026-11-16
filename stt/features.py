"""Étape 3 : extraction de caractéristiques (spectrogramme log-Mel) et figures."""

import librosa
import numpy as np

from .audio import SR

N_FFT = 400        # fenêtre de 25 ms à 16 kHz
HOP = 160          # décalage de 10 ms → 100 trames par seconde
N_MELS = 80        # 80 bandes Mel (128 pour large-v3 / turbo)


def log_mel(audio: np.ndarray, sr: int = SR, n_mels: int = N_MELS) -> np.ndarray:
    """Spectrogramme log-Mel calculé exactement comme Whisper.

    1. STFT : fenêtres de Hann de 25 ms, pas de 10 ms
    2. Spectre de puissance |X|²
    3. Banc de filtres Mel (80 filtres triangulaires)
    4. log10, plancher à (max - 8), puis mise à l'échelle (x + 4) / 4
    """
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, window="hann",
                        center=True, pad_mode="reflect")
    puissance = np.abs(stft[:, :-1]) ** 2
    filtres = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=n_mels)
    mel = filtres @ puissance
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def figure_signal(audio: np.ndarray, sr: int = SR, titre: str = ""):
    """Forme d'onde + spectrogramme log-Mel, alignés sur le même axe du temps."""
    from matplotlib.figure import Figure   # import local : log_mel reste utilisable sans matplotlib

    duree = len(audio) / sr
    spec = log_mel(audio, sr)
    t = np.arange(len(audio)) / sr

    # Figure créée sans pyplot : sûr dans les threads du serveur Gradio
    # et sans effet sur le backend d'affichage du notebook.
    fig = Figure(figsize=(10, 5.5))
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    ax1.plot(t, audio, linewidth=0.5, color="#2b6cb0")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1, 1)
    ax1.set_title(titre or "Forme d'onde (signal numérique, 16 kHz)")

    im = ax2.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, duree, 0, spec.shape[0]])
    ax2.set_ylabel("Bande Mel")
    ax2.set_xlabel("Temps (s)")
    ax2.set_title(f"Spectrogramme log-Mel ({spec.shape[0]} × {spec.shape[1]}) "
                  "— l'« image » que lit Whisper")
    fig.colorbar(im, ax=[ax1, ax2], shrink=0.6, label="log-énergie")
    return fig
