import pytest

from vocalis import demarrage
from vocalis.config import Reglages
from vocalis.controleur import ECOUTE, REPOS, TRANSCRIPTION
from vocalis.historique import Historique


def test_pastille_etats(qapp):
    from PySide6.QtCore import Qt
    from vocalis.ui.pastille import Pastille
    p = Pastille()
    assert p.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert p.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    p.afficher_etat(ECOUTE)
    for rms in (0.0, 0.05, 2.0, -1.0):
        p.afficher_niveau(rms)
    assert p.isVisible() and p.texte == "Écoute…"
    assert all(0.0 <= n <= 1.0 for n in p._niveaux)
    p.afficher_message("ne doit pas remplacer l'écoute")
    assert p.texte == "Écoute…"
    p.afficher_etat(TRANSCRIPTION)
    assert p.isVisible() and p.texte == "Transcription…"
    p.grab()                                # le dessin ne lève pas
    p.afficher_etat(REPOS)
    assert not p.isVisible()
    p.afficher_message("Aucune voix détectée", 10)
    assert p.isVisible() and p.texte == "Aucune voix détectée"


def test_reglages_lus_depuis_le_formulaire(qapp):
    from vocalis.ui.fenetre import OngletReglages
    onglet = OngletReglages(Reglages())
    recus = []
    onglet.enregistres.connect(recus.append)
    onglet.raccourci.setText("Ctrl+Shift+D")
    onglet.mode.setCurrentIndex(onglet.mode.findData("basculer"))
    onglet.langue.setCurrentIndex(onglet.langue.findData(None))
    onglet.modele_pro.setCurrentIndex(onglet.modele_pro.findData("base"))
    onglet.coller.setChecked(False)
    onglet.valider()
    assert recus == [Reglages(raccourci="ctrl+shift+d", mode="basculer", langue=None,
                              modele_pro="base", coller=False)]


def test_reglages_refuse_un_raccourci_invalide(qapp):
    from vocalis.ui.fenetre import OngletReglages
    onglet = OngletReglages(Reglages())
    recus = []
    onglet.enregistres.connect(recus.append)
    onglet.raccourci.setText("ctrl+shift")
    onglet.valider()
    assert recus == [] and "raccourci" in onglet.message.text().lower()


def test_note_quand_le_modele_maison_manque(qapp):
    from vocalis.ui.fenetre import OngletReglages
    onglet = OngletReglages(Reglages(moteur="maison"))
    assert "pas encore entraîné" in onglet.note_maison.text()


def test_fenetre_historique(qapp):
    from vocalis.ui.fenetre import FenetrePrincipale
    h = Historique()
    fenetre = FenetrePrincipale(Reglages(), h, lambda: None)
    onglet = fenetre.historique
    assert onglet.liste.count() == 0 and not onglet.b_copier.isEnabled()
    h.ajouter("première dictée", "Whisper small", 2.0, 0.4)
    h.ajouter("seconde dictée", "Whisper small", 1.0, 0.3)
    onglet.rafraichir()
    assert onglet.liste.count() == 2
    assert onglet.liste.item(0).text().startswith("seconde dictée")
    onglet.copier()
    assert qapp.clipboard().text() == "seconde dictée"
    onglet.supprimer()
    assert [e.texte for e in h.lister()] == ["première dictée"]
    onglet.vider()
    assert onglet.liste.count() == 0


def test_fichier_sans_moteur(qapp, tmp_path):
    from vocalis.ui.fenetre import OngletFichier
    onglet = OngletFichier(lambda: None, lambda: "fr")
    onglet.transcrire(tmp_path / "a.wav")
    assert "chargement" in onglet.etat.text() and not onglet.b_srt.isEnabled()


class FauxRegistre:
    HKEY_CURRENT_USER, KEY_SET_VALUE, REG_SZ = "HKCU", 2, 1

    def __init__(self):
        self.valeurs = {}

    def OpenKey(self, racine, cle, reserve=0, acces=0):
        assert (racine, cle) == ("HKCU", demarrage.CLE)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def QueryValueEx(self, cle, nom):
        if nom not in self.valeurs:
            raise FileNotFoundError(nom)
        return self.valeurs[nom], 1

    def SetValueEx(self, cle, nom, reserve, type_, valeur):
        self.valeurs[nom] = valeur

    def DeleteValue(self, cle, nom):
        if nom not in self.valeurs:
            raise FileNotFoundError(nom)
        del self.valeurs[nom]


def test_demarrage_automatique():
    registre = FauxRegistre()
    assert not demarrage.est_actif(registre)
    demarrage.activer(True, registre)
    assert demarrage.est_actif(registre)
    assert "lancer.pyw" in registre.valeurs["Vocalis"]
    assert "pythonw.exe" in registre.valeurs["Vocalis"]
    demarrage.activer(False, registre)
    demarrage.activer(False, registre)      # désactiver deux fois ne lève pas
    assert not demarrage.est_actif(registre)
