"""Historique des dictées, dans une base SQLite locale."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .chemins import dossier_donnees


@dataclass
class Entree:
    id: int
    date: str            # ISO 8601, heure locale
    texte: str
    moteur: str
    duree_audio: float
    temps_calcul: float


class Historique:
    def __init__(self, chemin: Path | None = None):
        self.chemin = Path(chemin or dossier_donnees() / "historique.db")
        with closing(self._connexion()) as c, c:
            c.execute("""CREATE TABLE IF NOT EXISTS dictees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL, texte TEXT NOT NULL, moteur TEXT NOT NULL,
                duree_audio REAL NOT NULL, temps_calcul REAL NOT NULL)""")

    def _connexion(self) -> sqlite3.Connection:
        # Une connexion par appel : l'historique est écrit depuis le fil de transcription.
        return sqlite3.connect(self.chemin)

    def ajouter(self, texte: str, moteur: str, duree_audio: float,
                temps_calcul: float) -> int:
        with closing(self._connexion()) as c, c:
            curseur = c.execute(
                "INSERT INTO dictees (date, texte, moteur, duree_audio, temps_calcul) "
                "VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), texte, moteur,
                 duree_audio, temps_calcul))
            return curseur.lastrowid

    def lister(self, limite: int = 200) -> list[Entree]:
        with closing(self._connexion()) as c:
            lignes = c.execute(
                "SELECT id, date, texte, moteur, duree_audio, temps_calcul "
                "FROM dictees ORDER BY id DESC LIMIT ?", (limite,)).fetchall()
        return [Entree(*ligne) for ligne in lignes]

    def supprimer(self, id: int) -> None:
        with closing(self._connexion()) as c, c:
            c.execute("DELETE FROM dictees WHERE id = ?", (id,))

    def vider(self) -> None:
        with closing(self._connexion()) as c, c:
            c.execute("DELETE FROM dictees")
