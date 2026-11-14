"""Constantes audio partagées, sans dépendance (importable par l'app légère)."""

# Whisper attend du mono à 16 kHz : la parole utile est sous 8 kHz (Nyquist).
SR = 16000
