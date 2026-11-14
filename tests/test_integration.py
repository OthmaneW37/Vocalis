"""App assemblée de bout en bout : raccourci → micro → Whisper → collage → historique.

Le micro est remplacé par un échantillon du repo et le clavier par des appels
directs ; tout le reste (moteur, contrôleur, pastille, presse-papiers) est réel.
"""

import json

from PySide6.QtCore import QEventLoop, QTimer

from stt.evaluate import wer

from conftest import SAMPLES


def _attendre(condition, delai_ms):
    boucle = QEventLoop()
    sonde = QTimer()
    sonde.timeout.connect(lambda: condition() and boucle.quit())
    sonde.start(20)
    QTimer.singleShot(delai_ms, boucle.quit)
    boucle.exec()
    assert condition(), "délai dépassé"


def test_dictee_de_bout_en_bout(qapp, monkeypatch):
    import vocalis.app as module
    from stt.audio import charger_audio
    from vocalis.config import Reglages
    from vocalis.raccourci import EcouteurClavier

    audio = charger_audio(str(SAMPLES / "exemple_fr.wav"))
    colles = []
    monkeypatch.setattr(EcouteurClavier, "demarrer", lambda self: None)
    monkeypatch.setattr(module, "coller_texte",
                        lambda texte, presse_papiers: colles.append(texte))
    Reglages(modele_pro="base").enregistrer()

    vocalis = module.Vocalis(qapp)
    try:
        assert vocalis.controleur.moteur is None            # chargement en arrière-plan
        _attendre(lambda: vocalis.controleur.moteur is not None, 60_000)
        assert "Whisper base prêt" in vocalis.fenetre.statut.text()

        enregistreur = vocalis.controleur.enregistreur
        monkeypatch.setattr(enregistreur, "demarrer", lambda: None)
        monkeypatch.setattr(enregistreur, "arreter", lambda: audio)

        detecteur = vocalis._ecouteur.detecteur             # comme si on tenait Ctrl+Espace
        detecteur.touche_enfoncee("ctrl")
        detecteur.touche_enfoncee("space")
        _attendre(lambda: vocalis.controleur.etat == "ecoute", 2000)
        assert vocalis.pastille.isVisible()
        detecteur.touche_relachee("space")
        _attendre(lambda: colles, 60_000)

        reference = json.loads((SAMPLES / "references.json").read_text(encoding="utf-8"))
        assert wer(reference["exemple_fr.wav"], colles[0])["wer"] < 0.15
        assert vocalis.controleur.etat == "repos" and not vocalis.pastille.isVisible()
        assert [e.texte for e in vocalis.historique.lister()] == colles
    finally:
        vocalis.tray.hide()
