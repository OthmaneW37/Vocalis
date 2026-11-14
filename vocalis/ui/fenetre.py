"""Fenêtre principale : historique, transcription de fichiers, réglages."""

import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from stt.resultat import vers_srt

from ..config import MODELES_PRO, Reglages
from ..micro import lister_micros
from ..moteurs import chemin_checkpoint
from ..raccourci import analyser
from .style import FEUILLE, icone

LANGUES = [("Français", "fr"), ("Anglais", "en"), ("Arabe", "ar"),
           ("Espagnol", "es"), ("Détection automatique", None)]
MODES = [("Maintenir pour parler", "maintenir"),
         ("Appuyer pour démarrer, appuyer pour arrêter", "basculer")]
MOTEURS = [("Whisper (moteur Pro)", "pro"), ("Modèle maison (entraîné de zéro)", "maison")]


def _liste(choix, valeur) -> QComboBox:
    combo = QComboBox()
    for libelle, donnee in choix:
        combo.addItem(libelle, donnee)
    index = combo.findData(valeur)
    combo.setCurrentIndex(index if index >= 0 else 0)
    return combo


class OngletReglages(QWidget):
    enregistres = Signal(object)        # Reglages

    def __init__(self, reglages: Reglages):
        super().__init__()
        self._base = reglages
        self.raccourci = QLineEdit(reglages.raccourci)
        self.raccourci.setPlaceholderText("ex. ctrl+space, f9, ctrl+shift+d")
        self.mode = _liste(MODES, reglages.mode)
        self.moteur = _liste(MOTEURS, reglages.moteur)
        self.modele_pro = _liste([(m, m) for m in MODELES_PRO], reglages.modele_pro)
        self.langue = _liste(LANGUES, reglages.langue)
        self.micro = _liste([("Micro par défaut de Windows", None)] + [
            (nom, i) for i, nom in lister_micros()], reglages.micro)
        self.coller = QCheckBox("Coller le texte dans la fenêtre active")
        self.coller.setChecked(reglages.coller)
        self.demarrage = QCheckBox("Lancer Vocalis au démarrage de Windows")
        self.demarrage.setChecked(reglages.demarrage_auto)
        self.message = QLabel("")
        self.note_maison = QLabel("")
        self.note_maison.setObjectName("doux")
        self.note_maison.setWordWrap(True)
        self.moteur.currentIndexChanged.connect(self._maj_note)

        formulaire = self._formulaire = QFormLayout()
        formulaire.setVerticalSpacing(12)
        formulaire.addRow("Raccourci", self.raccourci)
        formulaire.addRow("Déclenchement", self.mode)
        formulaire.addRow("Moteur", self.moteur)
        formulaire.addRow("", self.note_maison)
        formulaire.addRow("Modèle Whisper", self.modele_pro)
        formulaire.addRow("Langue", self.langue)
        formulaire.addRow("Micro", self.micro)
        formulaire.addRow("", self.coller)
        formulaire.addRow("", self.demarrage)

        bouton = QPushButton("Enregistrer")
        bouton.setObjectName("principal")
        bouton.clicked.connect(self.valider)
        bas = QHBoxLayout()
        bas.addWidget(self.message, 1)
        bas.addWidget(bouton)

        page = QVBoxLayout(self)
        page.setContentsMargins(20, 20, 20, 20)
        page.addLayout(formulaire)
        page.addStretch(1)
        page.addLayout(bas)
        self._maj_note()

    def _maj_note(self) -> None:
        if self.moteur.currentData() == "maison" and not chemin_checkpoint().is_file():
            self.note_maison.setText(
                "Le modèle maison n'est pas encore entraîné : Whisper sera utilisé "
                "tant que le fichier modele_maison.pt n'existe pas.")
        else:
            self.note_maison.setText("")
        self._formulaire.setRowVisible(self.note_maison, bool(self.note_maison.text()))

    def lire(self) -> Reglages:
        """Rend les réglages saisis ; lève `ValueError` si le raccourci est invalide."""
        raccourci = self.raccourci.text().strip().lower()
        analyser(raccourci)
        return replace(self._base, raccourci=raccourci,
                       mode=self.mode.currentData(), moteur=self.moteur.currentData(),
                       modele_pro=self.modele_pro.currentData(),
                       langue=self.langue.currentData(), micro=self.micro.currentData(),
                       coller=self.coller.isChecked(),
                       demarrage_auto=self.demarrage.isChecked())

    def valider(self) -> None:
        try:
            reglages = self.lire()
        except ValueError as e:
            self.message.setObjectName("erreur")
            self.message.setText(str(e))
        else:
            self._base = reglages
            self.message.setObjectName("doux")
            self.message.setText("Réglages enregistrés.")
            self.enregistres.emit(reglages)
        self.message.style().unpolish(self.message)
        self.message.style().polish(self.message)


class OngletHistorique(QWidget):
    def __init__(self, historique):
        super().__init__()
        self.historique = historique
        self.liste = QListWidget()
        self.liste.setWordWrap(True)
        self.liste.itemDoubleClicked.connect(lambda _: self.copier())
        self.vide = QLabel("Aucune dictée pour l'instant.\nTes transcriptions apparaîtront ici.")
        self.vide.setObjectName("doux")
        self.vide.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.b_copier = QPushButton("Copier")
        self.b_supprimer = QPushButton("Supprimer")
        self.b_vider = QPushButton("Tout effacer")
        self.b_copier.clicked.connect(self.copier)
        self.b_supprimer.clicked.connect(self.supprimer)
        self.b_vider.clicked.connect(self.vider)
        boutons = QHBoxLayout()
        boutons.addWidget(self.b_copier)
        boutons.addWidget(self.b_supprimer)
        boutons.addStretch(1)
        boutons.addWidget(self.b_vider)

        page = QVBoxLayout(self)
        page.setContentsMargins(20, 20, 20, 20)
        page.addWidget(self.liste, 1)
        page.addWidget(self.vide, 1)
        page.addLayout(boutons)
        self.rafraichir()

    def rafraichir(self) -> None:
        self.liste.clear()
        entrees = self.historique.lister()
        for e in entrees:
            try:
                date = datetime.fromisoformat(e.date).strftime("%d/%m %H:%M")
            except ValueError:
                date = e.date
            item = QListWidgetItem(f"{e.texte}\n{date} · {e.moteur} · {e.duree_audio:.1f} s")
            item.setData(Qt.ItemDataRole.UserRole, (e.id, e.texte))
            self.liste.addItem(item)
        self.liste.setVisible(bool(entrees))
        self.vide.setVisible(not entrees)
        for bouton in (self.b_copier, self.b_supprimer, self.b_vider):
            bouton.setEnabled(bool(entrees))
        if entrees:
            self.liste.setCurrentRow(0)

    def _selection(self):
        item = self.liste.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def copier(self) -> None:
        if (choix := self._selection()):
            QGuiApplication.clipboard().setText(choix[1])

    def supprimer(self) -> None:
        if (choix := self._selection()):
            self.historique.supprimer(choix[0])
            self.rafraichir()

    def vider(self) -> None:
        self.historique.vider()
        self.rafraichir()


class OngletFichier(QWidget):
    _termine = Signal(object, object)       # (Resultat | None, message d'erreur | None)

    def __init__(self, obtenir_moteur: Callable, obtenir_langue: Callable):
        super().__init__()
        self._obtenir_moteur = obtenir_moteur
        self._obtenir_langue = obtenir_langue
        self.resultat = None
        self.chemin: Path | None = None
        self.b_choisir = QPushButton("Choisir un fichier audio…")
        self.b_choisir.setObjectName("principal")
        self.b_srt = QPushButton("Exporter en .srt")
        self.b_copier = QPushButton("Copier le texte")
        self.etat = QLabel("Formats lus : wav, mp3, flac, ogg.")
        self.etat.setObjectName("doux")
        self.texte = QPlainTextEdit()
        self.texte.setReadOnly(True)
        self.texte.setPlaceholderText("La transcription du fichier s'affichera ici.")
        self.b_choisir.clicked.connect(self.choisir)
        self.b_srt.clicked.connect(self.exporter)
        self.b_copier.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.texte.toPlainText()))
        self._termine.connect(self._sur_termine)
        self._maj_boutons()

        haut = QHBoxLayout()
        haut.addWidget(self.b_choisir)
        haut.addWidget(self.etat, 1)
        bas = QHBoxLayout()
        bas.addWidget(self.b_copier)
        bas.addWidget(self.b_srt)
        bas.addStretch(1)
        page = QVBoxLayout(self)
        page.setContentsMargins(20, 20, 20, 20)
        page.addLayout(haut)
        page.addWidget(self.texte, 1)
        page.addLayout(bas)

    def _maj_boutons(self) -> None:
        pret = self.resultat is not None and bool(self.resultat.texte)
        self.b_srt.setEnabled(pret)
        self.b_copier.setEnabled(pret)

    def choisir(self) -> None:
        chemin, _ = QFileDialog.getOpenFileName(
            self, "Choisir un fichier audio", "",
            "Audio (*.wav *.mp3 *.flac *.ogg);;Tous les fichiers (*)")
        if chemin:
            self.transcrire(Path(chemin))

    def transcrire(self, chemin: Path) -> None:
        moteur = self._obtenir_moteur()
        if moteur is None:
            self.etat.setText("Le moteur est encore en chargement, réessaie dans un instant.")
            return
        self.chemin, self.resultat = chemin, None
        self.texte.clear()
        self.b_choisir.setEnabled(False)
        self._maj_boutons()
        self.etat.setText(f"Transcription de {chemin.name} avec {moteur.nom}…")
        threading.Thread(target=self._travail,
                         args=(moteur, chemin, self._obtenir_langue()), daemon=True).start()

    def _travail(self, moteur, chemin, langue) -> None:
        try:
            from stt.audio import charger_audio     # librosa : importé à la demande
            self._termine.emit(moteur.transcrire(charger_audio(str(chemin)), langue), None)
        except Exception as e:
            self._termine.emit(None, f"Lecture ou transcription impossible : {e}")

    def _sur_termine(self, resultat, erreur) -> None:
        self.b_choisir.setEnabled(True)
        if erreur:
            self.etat.setText(erreur)
        else:
            self.resultat = resultat
            self.texte.setPlainText(resultat.texte)
            self.etat.setText(
                f"{self.chemin.name} · {resultat.duree_audio:.1f} s d'audio · "
                f"transcrit en {resultat.temps_calcul:.1f} s"
                if resultat.texte else f"{self.chemin.name} : aucune parole reconnue.")
        self._maj_boutons()

    def exporter(self) -> None:
        defaut = str(self.chemin.with_suffix(".srt")) if self.chemin else "transcription.srt"
        chemin, _ = QFileDialog.getSaveFileName(self, "Exporter les sous-titres", defaut,
                                                "Sous-titres (*.srt)")
        if chemin:
            Path(chemin).write_text(
                vers_srt(self.resultat.segments, self.resultat.duree_audio), encoding="utf-8")
            self.etat.setText(f"Sous-titres enregistrés : {Path(chemin).name}")


class FenetrePrincipale(QWidget):
    def __init__(self, reglages: Reglages, historique, obtenir_moteur: Callable):
        super().__init__()
        self.setWindowTitle("Vocalis")
        self.setWindowIcon(icone())
        self.setStyleSheet(FEUILLE)
        self.resize(620, 560)
        self._reglages = reglages

        titre = QLabel("Vocalis")
        titre.setObjectName("titre")
        self.statut = QLabel("")
        self.statut.setObjectName("doux")
        self.statut.setWordWrap(True)

        self.historique = OngletHistorique(historique)
        self.reglages = OngletReglages(reglages)
        self.fichier = OngletFichier(obtenir_moteur, lambda: self._reglages.langue)
        self.reglages.enregistres.connect(self._garder)
        self.onglets = QTabWidget()
        self.onglets.addTab(self.historique, "Historique")
        self.onglets.addTab(self.fichier, "Fichier audio")
        self.onglets.addTab(self.reglages, "Réglages")

        entete = QVBoxLayout()
        entete.setContentsMargins(20, 18, 20, 6)
        entete.addWidget(titre)
        entete.addWidget(self.statut)
        page = QVBoxLayout(self)
        page.setContentsMargins(0, 0, 0, 0)
        page.addLayout(entete)
        page.addWidget(self.onglets, 1)

    def _garder(self, reglages: Reglages) -> None:
        self._reglages = reglages

    def afficher_statut(self, texte: str) -> None:
        self.statut.setText(texte)

    def montrer(self, onglet: int | None = None) -> None:
        if onglet is not None:
            self.onglets.setCurrentIndex(onglet)
        self.historique.rafraichir()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, evenement):
        evenement.ignore()      # fermer la fenêtre ne quitte pas l'app (zone de notification)
        self.hide()
