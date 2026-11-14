from stt.resultat import Resultat, vers_srt
from vocalis.chemins import dossier_donnees
from vocalis.config import Reglages


def test_vers_srt_horodatage():
    srt = vers_srt([(0.0, 1.5, "a"), (1.5, None, "b")], 2.0)
    assert "1\n00:00:00,000 --> 00:00:01,500\na" in srt
    assert "2\n00:00:01,500 --> 00:00:02,000\nb" in srt


def test_facteur_temps_reel():
    assert Resultat("x", duree_audio=4.0, temps_calcul=1.0).facteur_temps_reel == 0.25
    assert Resultat("x").facteur_temps_reel == 0.0


def test_dossier_donnees_isole(dossier_isole):
    assert dossier_donnees() == dossier_isole
    assert dossier_isole.is_dir()


def test_reglages_defaut_sans_fichier():
    r = Reglages.charger()
    assert (r.raccourci, r.mode, r.moteur, r.modele_pro, r.langue) == (
        "ctrl+space", "maintenir", "pro", "small", "fr")
    assert r.micro is None and r.coller is True and r.demarrage_auto is False


def test_reglages_aller_retour():
    Reglages(raccourci="f9", mode="basculer", langue=None, micro=3).enregistrer()
    r = Reglages.charger()
    assert (r.raccourci, r.mode, r.langue, r.micro) == ("f9", "basculer", None, 3)


def test_reglages_json_corrompu():
    Reglages.chemin().write_text("{pas du json", encoding="utf-8")
    assert Reglages.charger() == Reglages()


def test_reglages_cle_inconnue_et_valeur_invalide():
    Reglages.chemin().write_text(
        '{"inconnue": 1, "mode": "nimporte", "raccourci": "f8", "moteur": 12}',
        encoding="utf-8")
    r = Reglages.charger()
    assert r.raccourci == "f8"          # valeur valide conservée
    assert r.mode == "maintenir"        # valeur invalide → défaut
    assert r.moteur == "pro"


def test_reglages_json_pas_un_objet():
    Reglages.chemin().write_text("[1, 2]", encoding="utf-8")
    assert Reglages.charger() == Reglages()
