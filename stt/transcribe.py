"""Étapes 4 à 7 : encodeur, décodeur, recherche et post-traitement avec Whisper."""

import time
from functools import lru_cache

import numpy as np
import torch
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          pipeline)

from .audio_base import SR
from .resultat import Resultat, vers_srt  # noqa: F401 (réexportés)

MODELES = {
    "tiny (39 M)": "openai/whisper-tiny",
    "base (74 M)": "openai/whisper-base",
    "small (244 M)": "openai/whisper-small",
}
MODELE_DEFAUT = "small (244 M)"

NOMS_LANGUES = {"fr": "français", "en": "anglais", "ar": "arabe", "es": "espagnol",
                "de": "allemand", "it": "italien", "pt": "portugais", "nl": "néerlandais",
                "ja": "japonais", "zh": "chinois", "ru": "russe", "la": "latin",
                "br": "breton", "cy": "gallois", "ca": "catalan", "tr": "turc"}


@lru_cache(maxsize=3)
def charger_modele(nom: str):
    """Charge (une seule fois) le processeur, le modèle et le pipeline associé."""
    ident = MODELES.get(nom, nom)
    processeur = WhisperProcessor.from_pretrained(ident)
    modele = WhisperForConditionalGeneration.from_pretrained(
        ident, dtype=torch.float32).eval()
    asr = pipeline("automatic-speech-recognition", model=modele,
                   tokenizer=processeur.tokenizer,
                   feature_extractor=processeur.feature_extractor,
                   device="cpu")
    return processeur, modele, asr


def tokens_langues(modele) -> dict:
    """{code langue: id du token}, ex. {'fr': 50265, ...} (99 langues)."""
    return {jeton[2:-2]: i
            for jeton, i in modele.generation_config.lang_to_id.items()}


@torch.no_grad()
def detecter_langue(processeur, modele, audio: np.ndarray, top: int = 5):
    """Détection de langue : on donne <|startoftranscript|> au décodeur et on lit
    la probabilité de chaque token de langue pour le token suivant."""
    feats = processeur(audio[: 30 * SR], sampling_rate=SR,
                       return_tensors="pt").input_features
    sot = processeur.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    logits = modele(input_features=feats,
                    decoder_input_ids=torch.tensor([[sot]])).logits[0, -1]
    langues = tokens_langues(modele)
    codes = list(langues)
    probas = torch.softmax(logits[[langues[c] for c in codes]], dim=-1)
    ordre = torch.argsort(probas, descending=True)[:top]
    return [(codes[i], float(probas[i])) for i in ordre]


def transcrire(audio: np.ndarray, nom_modele: str = MODELE_DEFAUT,
               langue: str | None = None, tache: str = "transcribe") -> Resultat:
    """Transcrit un signal mono 16 kHz. `langue=None` → détection automatique."""
    processeur, modele, asr = charger_modele(nom_modele)
    t0 = time.perf_counter()

    probas = detecter_langue(processeur, modele, audio)
    langue_utilisee = langue or probas[0][0]

    sortie = asr({"raw": audio, "sampling_rate": SR},
                 chunk_length_s=30, batch_size=1, return_timestamps=True,
                 generate_kwargs={"language": langue_utilisee, "task": tache,
                                  "num_beams": 1})
    segments = [(c["timestamp"][0], c["timestamp"][1], c["text"].strip())
                for c in sortie.get("chunks", [])]
    return Resultat(texte=sortie["text"].strip(), segments=segments,
                    langue=langue_utilisee, probas_langues=probas,
                    duree_audio=len(audio) / SR,
                    temps_calcul=time.perf_counter() - t0)
