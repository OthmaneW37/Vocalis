"""Modèle maison : tokenizer, log-Mel, Conformer, CTC, données, entraînement, inférence."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from modele.conformer import ConfigModele, Conformer, longueurs_sorties  # noqa: E402
from modele.decodage import (decoder_glouton, distance_edition, reduire_ctc,  # noqa: E402
                             taux_erreur)
from modele.tokenizer import TokenizerBPE, normaliser_texte  # noqa: E402

from conftest import SAMPLES  # noqa: E402

SR = 16000
CORPUS = ["le chat dort sur le tapis", "la chatte mange le poisson",
          "aujourd'hui l'été est là", "le chien court dans le jardin",
          "peut-être que ça marche", "où est le garçon"] * 5


# --- Tokenizer -------------------------------------------------------------

def test_normaliser_texte():
    assert normaliser_texte("  L’Été,  c'est   BEAU ! 123 ") == "l'été c'est beau"


def test_bpe_aller_retour_et_blanc_reserve(tmp_path):
    t = TokenizerBPE.entrainer(CORPUS, taille_vocab=120)
    assert t.taille <= 120 and t.symboles[0] == "<blanc>"
    for texte in ["Le chat dort !", "Aujourd'hui, l'été", "zèbre œuvre 你好 wagon",
                  "mot-jamais-vu", ""]:
        ids = t.encoder(texte)
        assert 0 not in ids
        assert t.decoder(ids) == normaliser_texte(texte)
    assert len(t.encoder("le chat")) < len("le chat")     # les fusions compressent
    t.enregistrer(tmp_path / "vocab.json")
    recharge = TokenizerBPE.charger(tmp_path / "vocab.json")
    assert recharge.encoder("le chat dort") == t.encoder("le chat dort")
    assert recharge.taille == t.taille


def test_bpe_apprentissage_deterministe():
    assert (TokenizerBPE.entrainer(CORPUS, 100).fusions
            == TokenizerBPE.entrainer(list(reversed(CORPUS)), 100).fusions)


# --- Log-Mel ---------------------------------------------------------------

def test_log_mel_identique_a_la_reference_librosa():
    from modele.features import LogMel
    from stt.audio import charger_audio
    from stt.features import log_mel
    audio = charger_audio(str(SAMPLES / "exemple_fr.wav"))
    reference = log_mel(audio)
    obtenu = LogMel()(torch.from_numpy(audio)[None])[0].numpy()
    assert obtenu.shape == reference.shape == (80, len(audio) // 160)
    assert np.abs(obtenu - reference).max() < 1e-3


# --- Décodage et métriques -------------------------------------------------

def test_reduire_ctc():
    assert reduire_ctc([0, 5, 5, 0, 5, 7, 7, 0, 0]) == [5, 5, 7]
    assert reduire_ctc([]) == [] and reduire_ctc([0, 0]) == []


def test_distance_et_taux_erreur():
    assert distance_edition(list("chat"), list("chats")) == 1
    assert distance_edition([], list("ab")) == 2
    assert taux_erreur(["le chat dort"], ["le chien dort"]) == pytest.approx(1 / 3)
    assert taux_erreur(["Le chat !"], ["le chat"]) == 0.0
    assert taux_erreur(["abc"], ["abd"], "caractere") == pytest.approx(1 / 3)
    assert taux_erreur([], []) == 0.0


# --- Conformer -------------------------------------------------------------

MINI = ConfigModele(d=64, couches=2, tetes=4, dropout=0.0, taille_vocab=30)


def test_formes_et_longueurs():
    modele = Conformer(MINI).eval()
    longueurs = torch.tensor([200, 123])
    log_probas, sorties = modele(torch.randn(2, 80, 200), longueurs)
    assert log_probas.shape == (2, 49, 30)
    assert sorties.tolist() == longueurs_sorties(longueurs).tolist() == [49, 30]
    assert torch.allclose(log_probas.exp().sum(-1), torch.ones(2, 49), atol=1e-4)


def test_nombre_de_parametres():
    assert 20e6 < Conformer(ConfigModele.moyen()).nb_parametres() < 35e6
    assert 6e6 < Conformer(ConfigModele.petit()).nb_parametres() < 12e6


def test_le_remplissage_ne_change_pas_la_sortie():
    torch.manual_seed(0)
    modele = Conformer(MINI).eval()
    court, long_ = torch.randn(80, 120), torch.randn(80, 260)
    seul, n = modele(court[None], torch.tensor([120]))
    lot = torch.zeros(2, 80, 260)
    lot[0, :, :120], lot[1] = court, long_
    ensemble, longueurs = modele(lot, torch.tensor([120, 260]))
    assert int(longueurs[0]) == int(n[0])
    assert (ensemble[0, : int(n[0])] - seul[0]).abs().max() < 1e-4


def test_sur_apprentissage_de_quatre_enonces():
    """Le modèle et la perte CTC savent apprendre : 4 énoncés mémorisés à 100 %."""
    torch.manual_seed(0)
    modele = Conformer(MINI).train()
    longueurs = torch.tensor([160, 140, 120, 100])
    mels = torch.randn(4, 80, 160)
    cibles = [[3, 7, 7, 12], [5, 9, 2], [20, 4, 4, 4, 11], [8, 15]]
    lot = torch.zeros(4, 5, dtype=torch.long)
    for i, c in enumerate(cibles):
        lot[i, : len(c)] = torch.tensor(c)
    tailles = torch.tensor([len(c) for c in cibles])
    optimiseur = torch.optim.Adam(modele.parameters(), lr=2e-3)
    for _ in range(400):
        log_probas, sorties = modele(mels, longueurs)
        perte = torch.nn.functional.ctc_loss(log_probas.transpose(0, 1), lot, sorties, tailles)
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        if perte.item() < 0.05:
            break
    assert perte.item() < 0.1
    with torch.no_grad():       # même mode que l'apprentissage (statistiques du lot)
        log_probas, sorties = modele(mels, longueurs)
    decodes = [reduire_ctc(log_probas[i, : int(sorties[i])].argmax(-1).tolist())
               for i in range(4)]
    assert decodes == cibles


# --- Données ---------------------------------------------------------------

MOTS = {"le": 220, "chat": 330, "dort": 440, "ici": 550, "la": 660, "nuit": 770}


def _phrase_synthetique(dossier, i, mots):
    """Une « phrase » = une suite de notes, une par mot (0,25 s chacune)."""
    import soundfile as sf
    t = np.arange(int(0.25 * SR)) / SR
    signal = np.concatenate([0.3 * np.sin(2 * np.pi * MOTS[m] * t) for m in mots])
    chemin = dossier / f"p{i}.wav"
    sf.write(chemin, signal.astype(np.float32), SR)
    return chemin, len(signal) / SR, " ".join(mots)


@pytest.fixture
def mini_corpus(tmp_path):
    from modele.donnees import Exemple, ecrire_manifeste
    phrases = [["le", "chat", "dort"], ["la", "nuit"], ["le", "chat", "dort", "ici"],
               ["ici", "la", "nuit"], ["chat", "dort"], ["la", "nuit", "le", "chat"],
               ["dort", "ici"], ["le", "chat"]]
    dossier = tmp_path / "donnees"
    dossier.mkdir()
    exemples = [Exemple(*_phrase_synthetique(dossier, i, p)) for i, p in enumerate(phrases)]
    for partie in ("train", "dev"):
        ecrire_manifeste(dossier / f"{partie}.tsv", exemples)
    return dossier


def test_manifeste_et_fichier_manquant(mini_corpus):
    from modele.donnees import lire_manifeste
    exemples = lire_manifeste(mini_corpus / "train.tsv")
    assert len(exemples) == 8 and exemples[0].texte == "le chat dort"
    assert exemples[0].duree == pytest.approx(0.75)
    (mini_corpus / "p3.wav").unlink()
    with pytest.warns(UserWarning, match="introuvable"):
        assert len(lire_manifeste(mini_corpus / "train.tsv")) == 7


def test_jeu_et_assemblage(mini_corpus):
    from modele.donnees import JeuAudio, assembler, lire_manifeste
    exemples = lire_manifeste(mini_corpus / "train.tsv")
    tokenizer = TokenizerBPE.entrainer([e.texte for e in exemples], 60)
    jeu = JeuAudio(exemples, tokenizer)
    mels, longueurs, cibles, tailles = assembler([jeu[0], jeu[1]])
    assert mels.shape == (2, 80, 75) and longueurs.tolist() == [75, 50]
    assert (mels[1, :, 50:] == 0).all()
    assert cibles.shape[0] == 2 and tailles.tolist() == [
        len(tokenizer.encoder("le chat dort")), len(tokenizer.encoder("la nuit"))]


def test_echantillonneur_par_duree():
    from modele.donnees import EchantillonneurParDuree
    durees = [1.0, 9.0, 2.0, 8.0, 3.0, 2.5, 15.0]
    e = EchantillonneurParDuree(durees, secondes_par_lot=10.0, melanger=False)
    assert sorted(i for lot in e for i in lot) == list(range(7))    # chaque énoncé une fois
    for lot in e:
        assert max(durees[i] for i in lot) * len(lot) <= 10.0 or len(lot) == 1
    assert len(e) == len(e.lots)


def test_spec_augment():
    from modele.donnees import SpecAugment
    aug = SpecAugment()
    mel, longueurs = torch.ones(2, 80, 300), torch.tensor([300, 200])
    assert torch.equal(aug.eval()(mel, longueurs), mel)
    masque = aug.train()(mel, longueurs)
    assert (masque == 0).any() and (masque == 1).any() and (mel == 1).all()


# --- Entraînement, évaluation, inférence -----------------------------------

def test_entrainement_reprise_evaluation_et_moteur(mini_corpus, tmp_path):
    from modele.entrainer import entrainer, taux_apprentissage
    from modele.evaluer import main as evaluer
    from vocalis.moteurs import MoteurMaison

    assert taux_apprentissage(50, 1e-3, 100) == pytest.approx(5e-4)
    assert taux_apprentissage(400, 1e-3, 100) == pytest.approx(5e-4)

    sortie = tmp_path / "run"
    options = dict(donnees=mini_corpus, sortie=sortie, secondes_par_lot=4.0, lr=2e-3,
                   montee=20, eval_tous_les=60, travailleurs=0, taille_vocab=60,
                   appareil="cpu", journal=lambda *_: None,
                   config=ConfigModele(d=64, couches=2, dropout=0.0, taille_vocab=60))
    etat = entrainer(pas_max=120, epoques=1000, **options)
    assert etat["pas"] == 120
    lignes = (sortie / "journal.csv").read_text(encoding="utf-8").splitlines()
    pertes = [float(ligne.split(",")[2]) for ligne in lignes[1:]]
    assert len(pertes) >= 2 and pertes[-1] < pertes[0]          # la perte baisse
    assert (sortie / "dernier.pt").exists() and (sortie / "meilleur.pt").exists()

    etat = entrainer(pas_max=140, epoques=1000, reprendre=True, **options)
    assert etat["pas"] == 140                                    # repart de 120, pas de 0

    resultats = evaluer(["--checkpoint", str(sortie / "meilleur.pt"),
                         "--manifeste", str(mini_corpus / "dev.tsv")])
    assert 0.0 <= resultats["maison"]["wer"] <= 1.5 and resultats["maison"]["n"] == 8

    moteur = MoteurMaison(sortie / "meilleur.pt")
    from modele.donnees import charger_signal
    resultat = moteur.transcrire(charger_signal(mini_corpus / "p0.wav"))
    assert resultat.langue == "fr" and resultat.duree_audio == pytest.approx(0.75)
    assert isinstance(resultat.texte, str)
    assert moteur.transcrire(np.zeros(0, dtype=np.float32)).texte == ""
    assert moteur.transcrire(np.zeros(100, dtype=np.float32)).texte == ""


def test_decoupe_des_longs_enregistrements():
    from modele.inference import decouper
    audio = np.random.default_rng(0).standard_normal(50 * SR).astype(np.float32)
    audio[18 * SR: 18 * SR + 8000] = 0.0            # un silence à 18 s
    tranches = decouper(audio)
    assert tranches[0][0] == 0 and tranches[-1][1] == len(audio)
    assert all(a[1] == b[0] for a, b in zip(tranches, tranches[1:]))      # sans trou
    assert all(fin - debut <= 20 * SR for debut, fin in tranches)
    assert 18 * SR <= tranches[0][1] <= 18.5 * SR   # la coupe tombe dans le silence
    assert decouper(np.zeros(0, dtype=np.float32)) == []
