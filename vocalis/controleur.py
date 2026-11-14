"""Machine à états de la dictée : REPOS → ECOUTE → TRANSCRIPTION → REPOS."""

import threading

from PySide6.QtCore import QObject, QTimer, Signal

from stt.audio_base import SR

from .micro import MicroIndisponible
from .vad import rogner_silences

REPOS, ECOUTE, TRANSCRIPTION = "repos", "ecoute", "transcription"
DUREE_MIN_S = 0.3           # en dessous, c'est un appui accidentel
DUREE_MAX_S = 300           # arrêt automatique d'un enregistrement oublié


class Controleur(QObject):
    etat_change = Signal(str)
    niveau = Signal(float)          # niveau du micro pendant l'écoute
    texte_pret = Signal(str)
    info = Signal(str)              # message non bloquant (rien entendu…)
    erreur = Signal(str)
    _termine = Signal(object, object)   # (Resultat | None, Exception | None)

    def __init__(self, moteur, enregistreur, historique, reglages, parent=None):
        super().__init__(parent)
        self.moteur = moteur            # None tant que le modèle se charge
        self.enregistreur = enregistreur
        self.historique = historique
        self.reglages = reglages
        self.etat = REPOS
        enregistreur.sur_niveau = self.niveau.emit
        self._termine.connect(self._sur_termine)
        self._garde_fou = QTimer(self)
        self._garde_fou.setSingleShot(True)
        self._garde_fou.setInterval(DUREE_MAX_S * 1000)
        self._garde_fou.timeout.connect(self.fin_dictee)

    def _passer(self, etat: str) -> None:
        self.etat = etat
        self.etat_change.emit(etat)

    def changer_moteur(self, moteur) -> None:
        self.moteur = moteur

    def debut_dictee(self) -> None:
        if self.etat != REPOS:
            return                  # déjà en écoute, ou transcription en cours
        if self.moteur is None:
            self.erreur.emit("Le moteur de transcription est encore en chargement.")
            return
        try:
            self.enregistreur.demarrer()
        except MicroIndisponible as e:
            self.erreur.emit(str(e))
            return
        self._garde_fou.start()
        self._passer(ECOUTE)

    def fin_dictee(self) -> None:
        if self.etat != ECOUTE:
            return
        self._garde_fou.stop()
        audio = self.enregistreur.arreter()
        if len(audio) / SR < DUREE_MIN_S:
            self._passer(REPOS)
            return
        voix = rogner_silences(audio)
        if len(voix) == 0:
            self._passer(REPOS)
            self.info.emit("Aucune voix détectée")
            return
        self._passer(TRANSCRIPTION)
        moteur, langue = self.moteur, self.reglages.langue
        threading.Thread(target=self._travail, args=(moteur, voix, langue),
                         daemon=True).start()

    def _travail(self, moteur, voix, langue) -> None:
        try:
            self._termine.emit((moteur, moteur.transcrire(voix, langue)), None)
        except Exception as e:      # le fil ne doit jamais mourir en silence
            self._termine.emit(None, e)

    def _sur_termine(self, retour, exception) -> None:
        self._passer(REPOS)
        if exception is not None:
            self.erreur.emit(f"Échec de la transcription : {exception}")
            return
        moteur, resultat = retour
        texte = resultat.texte.strip()
        if not texte:
            self.info.emit("Rien n'a été reconnu")
            return
        self.historique.ajouter(texte, moteur.nom, resultat.duree_audio,
                                resultat.temps_calcul)
        self.texte_pret.emit(texte)
