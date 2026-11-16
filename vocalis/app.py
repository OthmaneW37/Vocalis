"""Assemblage de l'app : zone de notification, raccourci, contrôleur, fenêtres."""

import sys
import threading

from PySide6.QtCore import QLockFile, QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from . import demarrage
from .chemins import dossier_donnees
from .config import Reglages
from .controleur import ECOUTE, Controleur
from .historique import Historique
from .micro import Enregistreur
from .moteurs import creer_moteur
from .raccourci import DetecteurRaccourci, EcouteurClavier, analyser
from .saisie import coller_texte
from .ui.fenetre import FenetrePrincipale
from .ui.pastille import Pastille
from .ui.style import ROUGE, icone


class Vocalis(QObject):
    _moteur_pret = Signal(object, object, int)      # moteur, avertissement, n° de demande
    _moteur_echec = Signal(str, int)
    _touche_debut = Signal()        # émis depuis le fil du crochet clavier
    _touche_fin = Signal()

    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.reglages = Reglages.charger()
        self.historique = Historique()
        self.controleur = Controleur(None, Enregistreur(self.reglages.micro),
                                     self.historique, self.reglages)
        self.pastille = Pastille()
        self.fenetre = FenetrePrincipale(self.reglages, self.historique,
                                         lambda: self.controleur.moteur)
        self._demande = 0
        self._ecouteur: EcouteurClavier | None = None

        c = self.controleur
        c.etat_change.connect(self.pastille.afficher_etat)
        c.etat_change.connect(self._sur_etat)
        c.niveau.connect(self.pastille.afficher_niveau)
        c.texte_pret.connect(self._sur_texte)
        c.info.connect(self.pastille.afficher_message)
        c.erreur.connect(lambda m: self.pastille.afficher_message(m, 4000))
        self._touche_debut.connect(c.debut_dictee)
        self._touche_fin.connect(c.fin_dictee)
        self._moteur_pret.connect(self._sur_moteur_pret)
        self._moteur_echec.connect(self._sur_moteur_echec)
        self.fenetre.reglages.enregistres.connect(self._sur_reglages)

        self._creer_icone()
        self._brancher_raccourci()
        self._charger_moteur()

    # --- Zone de notification ------------------------------------------------

    def _creer_icone(self) -> None:
        self.icone_repos, self.icone_ecoute = icone(), icone(ROUGE)
        self.tray = QSystemTrayIcon(self.icone_repos, self)
        menu = QMenu()
        for libelle, onglet in (("Historique", 0), ("Transcrire un fichier…", 1),
                                ("Réglages", 2)):
            action = QAction(libelle, menu)
            action.triggered.connect(lambda _=False, o=onglet: self.fenetre.montrer(o))
            menu.addAction(action)
        menu.addSeparator()
        quitter = QAction("Quitter Vocalis", menu)
        quitter.triggered.connect(self.quitter)
        menu.addAction(quitter)
        self._menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda raison: raison == QSystemTrayIcon.ActivationReason.Trigger
            and self.fenetre.montrer())
        self.tray.show()

    def _statut(self, texte: str) -> None:
        self.tray.setToolTip(f"Vocalis — {texte}")
        self.fenetre.afficher_statut(texte)

    def _sur_etat(self, etat: str) -> None:
        self.tray.setIcon(self.icone_ecoute if etat == ECOUTE else self.icone_repos)

    # --- Raccourci -----------------------------------------------------------

    def _brancher_raccourci(self) -> None:
        if self._ecouteur is not None:
            self._ecouteur.arreter()
        detecteur = DetecteurRaccourci(analyser(self.reglages.raccourci), self.reglages.mode,
                                       self._touche_debut.emit, self._touche_fin.emit)
        self._ecouteur = EcouteurClavier(detecteur)
        self._ecouteur.demarrer()

    def _consigne(self) -> str:
        geste = "maintiens" if self.reglages.mode == "maintenir" else "appuie sur"
        return f"{geste} {self.reglages.raccourci.upper()} pour dicter"

    # --- Moteur --------------------------------------------------------------

    def _charger_moteur(self) -> None:
        self._demande += 1
        self.controleur.changer_moteur(None)
        self._statut("chargement du moteur…")
        threading.Thread(target=self._travail_moteur,
                         args=(self.reglages, self._demande), daemon=True).start()

    def _travail_moteur(self, reglages, demande) -> None:
        try:
            moteur, avertissement = creer_moteur(reglages)
            self._moteur_pret.emit(moteur, avertissement, demande)
        except Exception as e:
            self._moteur_echec.emit(str(e), demande)

    def _sur_moteur_pret(self, moteur, avertissement, demande) -> None:
        if demande != self._demande:
            return              # un réglage plus récent a relancé un chargement
        self.controleur.changer_moteur(moteur)
        self._statut(f"{moteur.nom} prêt · {self._consigne()}")
        self.pastille.afficher_message(avertissement or f"Prêt · {self._consigne()}", 3500)

    def _sur_moteur_echec(self, message, demande) -> None:
        if demande != self._demande:
            return
        self._statut(f"moteur indisponible : {message}")
        self.pastille.afficher_message("Le moteur n'a pas pu être chargé", 4000)

    # --- Événements ----------------------------------------------------------

    def _sur_texte(self, texte: str) -> None:
        presse_papiers = self.app.clipboard()
        if self.reglages.coller:
            coller_texte(texte, presse_papiers)
        else:
            presse_papiers.setText(texte)
            self.pastille.afficher_message("Texte copié")
        if self.fenetre.isVisible():
            self.fenetre.historique.rafraichir()

    def _sur_reglages(self, nouveaux: Reglages) -> None:
        anciens, self.reglages = self.reglages, nouveaux
        nouveaux.enregistrer()
        self.controleur.reglages = nouveaux
        self.controleur.enregistreur.peripherique = nouveaux.micro
        self._brancher_raccourci()
        if nouveaux.demarrage_auto != anciens.demarrage_auto:
            try:
                demarrage.activer(nouveaux.demarrage_auto)
            except OSError as e:
                self.pastille.afficher_message(f"Démarrage automatique impossible : {e}", 4000)
        if (nouveaux.moteur, nouveaux.modele_pro) != (anciens.moteur, anciens.modele_pro):
            self._charger_moteur()
        elif self.controleur.moteur is not None:
            self._statut(f"{self.controleur.moteur.nom} prêt · {self._consigne()}")

    def quitter(self) -> None:
        if self._ecouteur is not None:
            self._ecouteur.arreter()
        self.tray.hide()
        self.app.quit()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Vocalis")
    app.setQuitOnLastWindowClosed(False)

    verrou = QLockFile(str(dossier_donnees() / "vocalis.lock"))
    verrou.setStaleLockTime(0)
    if not verrou.tryLock(100):
        QMessageBox.information(None, "Vocalis", "Vocalis est déjà lancé : "
                                "cherche son icône dans la zone de notification.")
        return 0

    vocalis = Vocalis(app)
    if not Reglages.chemin().exists():      # premier lancement : on se présente
        vocalis.reglages.enregistrer()
        vocalis.fenetre.montrer(2)
    code = app.exec()
    verrou.unlock()
    return code
