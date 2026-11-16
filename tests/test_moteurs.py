import json

import numpy as np
import pytest

from stt.evaluate import wer
from vocalis.config import Reglages
from vocalis.moteurs import (ModeleMaisonAbsent, MoteurMaison, MoteurPro,
                             creer_moteur)

from conftest import SAMPLES


@pytest.fixture(scope="module")
def pro():
    return MoteurPro("base")


def test_pro_transcrit_le_francais(pro):
    from stt.audio import charger_audio
    reference = json.loads((SAMPLES / "references.json").read_text(encoding="utf-8"))
    res = pro.transcrire(charger_audio(str(SAMPLES / "exemple_fr.wav")), "fr")
    assert wer(reference["exemple_fr.wav"], res.texte)["wer"] < 0.15
    assert res.langue == "fr" and res.segments and res.duree_audio > 5
    assert res.temps_calcul > 0


def test_pro_silence_rend_un_texte_vide(pro):
    assert pro.transcrire(np.zeros(2 * 16000, dtype=np.float32), "fr").texte == ""
    assert pro.transcrire(np.zeros(0, dtype=np.float32), "fr").texte == ""


def test_maison_sans_checkpoint():
    with pytest.raises(ModeleMaisonAbsent):
        MoteurMaison()


def test_creer_moteur_repli_sur_pro(monkeypatch):
    import vocalis.moteurs as moteurs
    monkeypatch.setattr(moteurs, "MoteurPro", lambda taille: ("pro", taille))
    moteur, avertissement = creer_moteur(Reglages(moteur="maison", modele_pro="base"))
    assert moteur == ("pro", "base") and avertissement
    moteur, avertissement = creer_moteur(Reglages(moteur="pro"))
    assert moteur == ("pro", "small") and avertissement is None
