"""Réglages de l'app, enregistrés en JSON dans le dossier de données."""

import json
from dataclasses import asdict, dataclass, fields

from .chemins import dossier_donnees

MODES = ("maintenir", "basculer")
MOTEURS = ("pro", "maison")
MODELES_PRO = ("base", "small", "large-v3-turbo")


@dataclass
class Reglages:
    raccourci: str = "ctrl+space"
    mode: str = "maintenir"            # "maintenir" | "basculer"
    moteur: str = "pro"                # "pro" | "maison"
    modele_pro: str = "small"
    langue: str | None = "fr"          # None = détection automatique
    micro: int | None = None           # None = micro par défaut de Windows
    coller: bool = True                # coller dans la fenêtre active
    demarrage_auto: bool = False

    @staticmethod
    def chemin():
        return dossier_donnees() / "reglages.json"

    @classmethod
    def charger(cls) -> "Reglages":
        """Lit le fichier ; toute valeur absente, inconnue ou invalide retombe
        sur sa valeur par défaut, sans jamais lever."""
        try:
            brut = json.loads(cls.chemin().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(brut, dict):
            return cls()
        connus = {f.name for f in fields(cls)}
        reglages = cls(**{k: v for k, v in brut.items() if k in connus})
        return reglages._valide()

    def _valide(self) -> "Reglages":
        defaut = Reglages()
        if self.mode not in MODES:
            self.mode = defaut.mode
        if self.moteur not in MOTEURS:
            self.moteur = defaut.moteur
        if self.modele_pro not in MODELES_PRO:
            self.modele_pro = defaut.modele_pro
        if not isinstance(self.raccourci, str) or not self.raccourci.strip():
            self.raccourci = defaut.raccourci
        if not (self.langue is None or isinstance(self.langue, str)):
            self.langue = defaut.langue
        if not (self.micro is None or isinstance(self.micro, int)):
            self.micro = None
        self.coller = bool(self.coller)
        self.demarrage_auto = bool(self.demarrage_auto)
        return self

    def enregistrer(self) -> None:
        self.chemin().write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
