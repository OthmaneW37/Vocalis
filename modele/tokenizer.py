"""Tokenizer BPE (Byte-Pair Encoding) écrit à la main, pour le français.

L'id 0 est réservé au symbole « blanc » de la CTC. Le début de chaque mot est
marqué par « ▁ », ce qui permet de reconstruire les espaces au décodage.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

MARQUE = "▁"
ALPHABET = "abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüÿœæ'-"
_HORS_ALPHABET = re.compile(f"[^{re.escape(ALPHABET)} ]")


def normaliser_texte(texte: str) -> str:
    """Minuscules, apostrophes unifiées, tout caractère hors alphabet → espace."""
    texte = unicodedata.normalize("NFC", texte).lower()
    texte = texte.replace("’", "'").replace("ʼ", "'")
    texte = _HORS_ALPHABET.sub(" ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def _fusionner(symboles: tuple, paire: tuple) -> tuple:
    a, b = paire
    sortie, i = [], 0
    while i < len(symboles):
        if i + 1 < len(symboles) and symboles[i] == a and symboles[i + 1] == b:
            sortie.append(a + b)
            i += 2
        else:
            sortie.append(symboles[i])
            i += 1
    return tuple(sortie)


class TokenizerBPE:
    def __init__(self, fusions: list[tuple[str, str]]):
        self.fusions = [tuple(f) for f in fusions]
        self._rang = {f: i for i, f in enumerate(self.fusions)}
        symboles = [MARQUE] + list(ALPHABET) + [a + b for a, b in self.fusions]
        self.symboles = ["<blanc>"] + symboles          # l'indice dans la liste est l'id
        self._id = {s: i for i, s in enumerate(self.symboles)}
        self._cache: dict[str, list[int]] = {}

    @property
    def taille(self) -> int:
        return len(self.symboles)

    @classmethod
    def entrainer(cls, textes, taille_vocab: int = 500) -> "TokenizerBPE":
        """Apprend les fusions les plus fréquentes jusqu'à `taille_vocab` symboles."""
        frequences = Counter(mot for t in textes for mot in normaliser_texte(t).split())
        mots = {(MARQUE,) + tuple(mot): n for mot, n in frequences.items()}
        nb_fusions = max(0, taille_vocab - (2 + len(ALPHABET)))
        fusions = []
        for _ in range(nb_fusions):
            paires = Counter()
            for symboles, n in mots.items():
                for paire in zip(symboles, symboles[1:]):
                    paires[paire] += n
            if not paires:
                break
            # À fréquence égale, l'ordre alphabétique rend l'apprentissage déterministe.
            meilleure = min(paires, key=lambda p: (-paires[p], p))
            if paires[meilleure] < 2:
                break
            fusions.append(meilleure)
            nouveaux = {}
            for symboles, n in mots.items():
                cle = _fusionner(symboles, meilleure)
                nouveaux[cle] = nouveaux.get(cle, 0) + n
            mots = nouveaux
        return cls(fusions)

    def _encoder_mot(self, mot: str) -> list[int]:
        if mot not in self._cache:
            symboles = (MARQUE,) + tuple(mot)
            while len(symboles) > 1:
                paires = set(zip(symboles, symboles[1:]))
                candidate = min(paires, key=lambda p: self._rang.get(p, float("inf")))
                if candidate not in self._rang:
                    break
                symboles = _fusionner(symboles, candidate)
            self._cache[mot] = [self._id[s] for s in symboles]
        return self._cache[mot]

    def encoder(self, texte: str) -> list[int]:
        return [i for mot in normaliser_texte(texte).split() for i in self._encoder_mot(mot)]

    def decoder(self, ids) -> str:
        texte = "".join(self.symboles[i] for i in ids if 0 < i < len(self.symboles))
        return texte.replace(MARQUE, " ").strip()

    def enregistrer(self, chemin) -> None:
        Path(chemin).write_text(json.dumps({"fusions": self.fusions}, ensure_ascii=False),
                                encoding="utf-8")

    @classmethod
    def charger(cls, chemin) -> "TokenizerBPE":
        return cls(json.loads(Path(chemin).read_text(encoding="utf-8"))["fusions"])
