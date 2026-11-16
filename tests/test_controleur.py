import threading

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer

from stt.resultat import Resultat
from vocalis.config import Reglages
from vocalis.controleur import ECOUTE, REPOS, TRANSCRIPTION, Controleur
from vocalis.historique import Historique
from vocalis.micro import MicroIndisponible

SR = 16000


def _voix(duree=1.0):
    t = np.arange(int(duree * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FauxEnregistreur:
    def __init__(self, audio, panne=False):
        self.audio, self.panne, self.sur_niveau = audio, panne, None
        self.demarrages = 0

    def demarrer(self):
        if self.panne:
            raise MicroIndisponible("Micro indisponible : aucun périphérique")
        self.demarrages += 1

    def arreter(self):
        return self.audio


class FauxMoteur:
    nom = "Faux"

    def __init__(self, texte="bonjour", exception=None):
        self.texte, self.exception = texte, exception
        self.appels = []
        self.liberer = threading.Event()
        self.liberer.set()

    def transcrire(self, audio, langue=None):
        self.appels.append((len(audio), langue))
        self.liberer.wait(5)
        if self.exception:
            raise self.exception
        return Resultat(self.texte, duree_audio=len(audio) / SR, temps_calcul=0.01)


def _monter(qapp, audio, moteur=None, panne=False):
    moteur = moteur or FauxMoteur()
    c = Controleur(moteur, FauxEnregistreur(audio, panne), Historique(), Reglages())
    journal = {"etats": [], "textes": [], "erreurs": [], "infos": []}
    c.etat_change.connect(journal["etats"].append)
    c.texte_pret.connect(journal["textes"].append)
    c.erreur.connect(journal["erreurs"].append)
    c.info.connect(journal["infos"].append)
    return c, moteur, journal


def _attendre(condition, delai_ms=3000):
    boucle = QEventLoop()
    sonde = QTimer()
    sonde.timeout.connect(lambda: condition() and boucle.quit())
    sonde.start(5)
    QTimer.singleShot(delai_ms, boucle.quit)
    boucle.exec()
    assert condition(), "délai dépassé"


def test_cycle_complet(qapp):
    c, moteur, j = _monter(qapp, _voix())
    c.debut_dictee()
    assert c.etat == ECOUTE
    c.fin_dictee()
    _attendre(lambda: j["textes"])
    assert j["etats"] == [ECOUTE, TRANSCRIPTION, REPOS]
    assert j["textes"] == ["bonjour"] and j["erreurs"] == []
    assert moteur.appels[0][1] == "fr"
    entrees = c.historique.lister()
    assert [(e.texte, e.moteur) for e in entrees] == [("bonjour", "Faux")]


@pytest.mark.parametrize("audio", [
    np.zeros(2 * SR, dtype=np.float32),      # silence
    _voix(0.1),                              # appui trop bref
    np.zeros(0, dtype=np.float32),           # rien reçu du micro
])
def test_sans_voix_le_moteur_n_est_pas_appele(qapp, audio):
    c, moteur, j = _monter(qapp, audio)
    c.debut_dictee()
    c.fin_dictee()
    assert c.etat == REPOS and moteur.appels == [] and j["textes"] == []
    assert c.historique.lister() == []


def test_raccourci_ignore_pendant_la_transcription(qapp):
    moteur = FauxMoteur()
    moteur.liberer.clear()                   # la transcription reste en cours
    c, _, j = _monter(qapp, _voix(), moteur)
    c.debut_dictee()
    c.fin_dictee()
    assert c.etat == TRANSCRIPTION
    c.debut_dictee()
    c.fin_dictee()
    assert c.etat == TRANSCRIPTION and c.enregistreur.demarrages == 1
    moteur.liberer.set()
    _attendre(lambda: j["textes"])
    assert len(moteur.appels) == 1 and c.etat == REPOS


def test_micro_indisponible(qapp):
    c, moteur, j = _monter(qapp, _voix(), panne=True)
    c.debut_dictee()
    assert c.etat == REPOS and j["etats"] == []
    assert "Micro indisponible" in j["erreurs"][0]


def test_exception_du_moteur(qapp):
    c, _, j = _monter(qapp, _voix(), FauxMoteur(exception=RuntimeError("boum")))
    c.debut_dictee()
    c.fin_dictee()
    _attendre(lambda: j["erreurs"])
    assert c.etat == REPOS and "boum" in j["erreurs"][0] and j["textes"] == []
    c.debut_dictee()                         # l'app reste utilisable après l'erreur
    assert c.etat == ECOUTE


def test_texte_vide_n_est_ni_colle_ni_enregistre(qapp):
    c, _, j = _monter(qapp, _voix(), FauxMoteur(texte="  "))
    c.debut_dictee()
    c.fin_dictee()
    _attendre(lambda: j["infos"])
    assert j["textes"] == [] and c.historique.lister() == []


def test_moteur_pas_encore_charge(qapp):
    c, _, j = _monter(qapp, _voix())
    c.changer_moteur(None)
    c.debut_dictee()
    assert c.etat == REPOS and "chargement" in j["erreurs"][0]
