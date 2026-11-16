"""VAD, micro, raccourci, collage, historique."""

import numpy as np
import pytest

from vocalis.historique import Historique
from vocalis.micro import Enregistreur, MicroIndisponible
from vocalis.raccourci import DetecteurRaccourci, analyser, nom_touche
from vocalis.vad import rogner_silences

SR = 16000


# --- VAD -------------------------------------------------------------------

def _sinus(duree, amplitude=0.3):
    t = np.arange(int(duree * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_vad_rogne_les_silences():
    silence = np.zeros(SR, dtype=np.float32)
    rendu = rogner_silences(np.concatenate([silence, _sinus(1.0), silence]))
    assert 1.0 <= len(rendu) / SR <= 1.6


@pytest.mark.parametrize("audio", [
    np.zeros(2 * SR, dtype=np.float32),
    np.zeros(0, dtype=np.float32),
    (0.001 * np.random.default_rng(0).standard_normal(2 * SR)).astype(np.float32),
])
def test_vad_sans_voix_rend_vide(audio):
    assert len(rogner_silences(audio)) == 0


def test_vad_parole_sans_pause_est_gardee():
    assert len(rogner_silences(_sinus(2.0))) == 2 * SR


def test_vad_ignore_un_clic():
    audio = np.zeros(SR, dtype=np.float32)
    audio[8000:8100] = 0.9
    assert len(rogner_silences(audio)) == 0


# --- Micro -----------------------------------------------------------------

class FauxFlux:
    def __init__(self, **options):
        self.options = options
        self.ouvert = False

    def start(self):
        self.ouvert = True

    def stop(self):
        self.ouvert = False

    def close(self):
        pass


def test_micro_concatene_les_blocs_dans_l_ordre():
    flux = []
    niveaux = []
    e = Enregistreur(sur_niveau=niveaux.append,
                     fabrique_flux=lambda **o: flux.append(FauxFlux(**o)) or flux[-1])
    e.demarrer()
    rappel = flux[0].options["callback"]
    rappel(np.full((4, 1), 0.1, dtype=np.float32), 4, None, None)
    rappel(np.full((2, 1), 0.2, dtype=np.float32), 2, None, None)
    audio = e.arreter()
    assert flux[0].options["samplerate"] == SR and flux[0].options["channels"] == 1
    np.testing.assert_allclose(audio, [0.1] * 4 + [0.2] * 2)
    assert audio.dtype == np.float32 and len(niveaux) == 2
    assert not flux[0].ouvert


def test_micro_arreter_sans_demarrer():
    assert len(Enregistreur(fabrique_flux=FauxFlux).arreter()) == 0


def test_micro_indisponible():
    def fabrique(**options):
        raise RuntimeError("Error querying device -1")

    e = Enregistreur(fabrique_flux=fabrique)
    with pytest.raises(MicroIndisponible):
        e.demarrer()
    assert len(e.arreter()) == 0


def test_micro_debranche_pendant_l_enregistrement_garde_l_audio():
    class FluxCasse(FauxFlux):
        def stop(self):
            raise RuntimeError("device unplugged")

    flux = []
    e = Enregistreur(fabrique_flux=lambda **o: flux.append(FluxCasse(**o)) or flux[-1])
    e.demarrer()
    flux[0].options["callback"](np.ones((3, 1), dtype=np.float32), 3, None, None)
    assert len(e.arreter()) == 3


# --- Raccourci -------------------------------------------------------------

def test_analyser():
    assert analyser("Ctrl+Space") == {"ctrl", "space"}
    assert analyser(" control + maj + F9 ") == {"ctrl", "shift", "f9"}
    assert analyser("f9") == {"f9"}


@pytest.mark.parametrize("texte", ["", "ctrl", "ctrl+shift", "a+b", None])
def test_analyser_refuse(texte):
    with pytest.raises(ValueError):
        analyser(texte)


def _detecteur(mode, combinaison="ctrl+space"):
    journal = []
    d = DetecteurRaccourci(analyser(combinaison), mode,
                           lambda: journal.append("debut"), lambda: journal.append("fin"))
    return d, journal


def test_maintenir():
    d, journal = _detecteur("maintenir")
    d.touche_enfoncee("ctrl")
    assert journal == []
    d.touche_enfoncee("space")
    d.touche_enfoncee("space")      # répétition automatique du clavier
    d.touche_enfoncee("space")
    assert journal == ["debut"]
    d.touche_relachee("ctrl")       # relâcher n'importe quelle touche arrête
    assert journal == ["debut", "fin"]
    d.touche_relachee("space")
    assert journal == ["debut", "fin"]


def test_basculer():
    d, journal = _detecteur("basculer")
    for _ in range(2):
        d.touche_enfoncee("ctrl")
        d.touche_enfoncee("space")
        d.touche_enfoncee("space")
        d.touche_relachee("space")
        d.touche_relachee("ctrl")
    assert journal == ["debut", "fin"]


def test_touche_en_trop_ne_declenche_rien():
    d, journal = _detecteur("maintenir")
    for touche in ("ctrl", "shift", "space"):
        d.touche_enfoncee(touche)
    assert journal == []
    assert not d.modificateurs_tenus()


def test_touche_seule():
    d, journal = _detecteur("maintenir", "f9")
    d.touche_enfoncee("f9")
    d.touche_relachee("f9")
    assert journal == ["debut", "fin"]


def test_synchroniser_debloque_un_modificateur_fantome():
    d, journal = _detecteur("maintenir")
    d.touche_enfoncee("shift")          # relâchement de shift jamais reçu
    d.touche_enfoncee("ctrl")
    d.synchroniser({"ctrl"})
    d.touche_enfoncee("space")
    assert journal == ["debut"]


def test_nom_touche():
    assert [nom_touche(v) for v in (0xA2, 0x20, 0x78, 0x41, 0x35)] == [
        "ctrl", "space", "f9", "a", "5"]


# --- Collage ---------------------------------------------------------------

def _attendre(ms):
    from PySide6.QtCore import QEventLoop, QTimer
    boucle = QEventLoop()
    QTimer.singleShot(ms, boucle.quit)
    boucle.exec()


def test_coller_restaure_le_texte(qapp):
    from vocalis.saisie import coller_texte
    pp = qapp.clipboard()
    pp.setText("ancien contenu")
    vus = []
    coller_texte("bonjour le monde", pp, lambda: vus.append(pp.text()), delai_ms=20)
    assert vus == ["bonjour le monde"]      # Ctrl+V envoyé une fois, avec le bon texte
    _attendre(80)
    assert pp.text() == "ancien contenu"


def test_coller_restaure_une_image(qapp):
    from PySide6.QtGui import QColor, QImage
    from vocalis.saisie import coller_texte
    pp = qapp.clipboard()
    image = QImage(8, 6, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    pp.setImage(image)
    coller_texte("texte dicté", pp, lambda: None, delai_ms=20)
    _attendre(80)
    assert pp.mimeData().hasImage()
    rendue = pp.image()
    assert (rendue.width(), rendue.height()) == (8, 6)
    assert rendue.pixelColor(2, 2) == QColor("red")


def test_coller_texte_vide_ne_touche_a_rien(qapp):
    from vocalis.saisie import coller_texte
    pp = qapp.clipboard()
    pp.setText("intact")
    appels = []
    coller_texte("", pp, lambda: appels.append(1), delai_ms=20)
    _attendre(50)
    assert appels == [] and pp.text() == "intact"


def test_coller_ne_restaure_pas_si_l_utilisateur_a_copie_entre_temps(qapp):
    from vocalis.saisie import coller_texte
    pp = qapp.clipboard()
    pp.setText("ancien")
    coller_texte("dicté", pp, lambda: None, delai_ms=20)
    pp.setText("copie de l'utilisateur")
    _attendre(80)
    assert pp.text() == "copie de l'utilisateur"


# --- Historique ------------------------------------------------------------

def test_historique():
    h = Historique()
    a = h.ajouter("première", "Whisper small", 2.0, 0.5)
    b = h.ajouter("Ça marche, l'été 🎉", "Modèle maison", 3.5, 0.2)
    entrees = h.lister()
    assert [e.id for e in entrees] == [b, a]
    assert entrees[0].texte == "Ça marche, l'été 🎉"
    assert entrees[0].moteur == "Modèle maison" and entrees[0].duree_audio == 3.5
    h.supprimer(b)
    assert [e.id for e in h.lister()] == [a]
    h.vider()
    assert h.lister() == []
    assert Historique().lister() == []      # la base se rouvre sans erreur
