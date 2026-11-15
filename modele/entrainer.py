"""Entraînement du modèle maison (Conformer + CTC), de zéro.

    python -m modele.entrainer --donnees donnees --sortie entrainements/v1 --taille moyen

Reprise après interruption : relancer la même commande avec --reprendre.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .conformer import ConfigModele, Conformer
from .donnees import (EchantillonneurParDuree, JeuAudio, SpecAugment, assembler,
                      lire_manifeste)
from .evaluer import evaluer_modele
from .inference import sauver_checkpoint
from .tokenizer import TokenizerBPE


def choisir_appareil(demande: str | None = None):
    """GPU si PyTorch en voit un (CUDA / ROCm, puis DirectML), sinon CPU."""
    if demande and demande != "auto":
        if demande == "directml":
            import torch_directml
            return torch_directml.device()
        return torch.device(demande)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def taux_apprentissage(pas: int, crete: float, montee: int) -> float:
    """Montée linéaire pendant `montee` pas, puis décroissance en 1/√pas."""
    pas = max(pas, 1)
    return crete * min(pas / montee, (montee / pas) ** 0.5)


def perte_ctc(log_probas, longueurs, cibles, longueurs_cibles) -> torch.Tensor:
    # Hors CUDA, la CTC est calculée sur CPU : elle n'existe pas sur DirectML.
    ou = log_probas.device if log_probas.device.type == "cuda" else torch.device("cpu")
    return F.ctc_loss(log_probas.float().to(ou).transpose(0, 1), cibles.to(ou),
                      longueurs.to(ou), longueurs_cibles.to(ou),
                      blank=0, zero_infinity=True)


def entrainer(donnees, sortie, taille: str = "moyen", epoques: int = 60,
              secondes_par_lot: float = 160.0, lr: float = 1e-3, montee: int = 10_000,
              eval_tous_les: int = 2000, sauver_tous_les: int = 250,
              pas_max: int | None = None,
              reprendre: bool = False, appareil: str | None = None,
              travailleurs: int = 4, taille_vocab: int = 500, limite_dev: int = 400,
              config: ConfigModele | None = None, augmenter: bool = True,
              debut_augmentation: int = 0, poids_intermediaire: float = 0.3,
              amorce_enonces: int = 0, amorce_pas: int = 0,
              journal=print) -> dict:
    donnees, sortie = Path(donnees), Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    appareil = choisir_appareil(appareil)
    exemples = lire_manifeste(donnees / "train.tsv")
    dev = lire_manifeste(donnees / "dev.tsv")[:limite_dev]
    if not exemples:
        raise ValueError(f"Aucun énoncé utilisable dans {donnees / 'train.tsv'}")

    chemin_vocab = sortie / "tokenizer.json"
    if chemin_vocab.exists():
        tokenizer = TokenizerBPE.charger(chemin_vocab)
    else:
        tokenizer = TokenizerBPE.entrainer((e.texte for e in exemples), taille_vocab)
        tokenizer.enregistrer(chemin_vocab)

    if config is None:
        config = getattr(ConfigModele, taille)(taille_vocab=tokenizer.taille)
    modele = Conformer(config).to(appareil)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=lr, betas=(0.9, 0.98),
                                   weight_decay=1e-2)
    augmentation = SpecAugment()
    etat = {"pas": 0, "epoque": 0, "meilleur_wer": float("inf")}

    dernier = sortie / "dernier.pt"
    if reprendre and dernier.exists():
        sauvegarde = torch.load(str(dernier), map_location="cpu", weights_only=True)
        modele.load_state_dict(sauvegarde["poids"])
        optimiseur.load_state_dict(sauvegarde["optimiseur"])
        etat = {c: sauvegarde[c] for c in etat}
        journal(f"Reprise au pas {etat['pas']} (époque {etat['epoque']})")

    heures = sum(e.duree for e in exemples) / 3600
    journal(f"{len(exemples)} énoncés ({heures:.1f} h) · {modele.nb_parametres() / 1e6:.1f} M "
            f"de paramètres · vocabulaire {tokenizer.taille} · appareil {appareil}")

    precision_mixte = getattr(appareil, "type", "") == "cuda"
    echelle = torch.amp.GradScaler("cuda", enabled=precision_mixte)
    chemin_journal = sortie / "journal.csv"
    if not chemin_journal.exists():
        chemin_journal.write_text("pas,epoque,perte,lr,wer_dev,cer_dev\n", encoding="utf-8")

    def sauver(chemin):
        sauver_checkpoint(chemin, modele, tokenizer, optimiseur=optimiseur.state_dict(), **etat)

    def evaluer_et_sauver(perte_moyenne):
        mesures = evaluer_modele(modele, tokenizer, dev, appareil) if dev else {
            "wer": float("nan"), "cer": float("nan")}
        with chemin_journal.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([etat["pas"], etat["epoque"], f"{perte_moyenne:.4f}",
                                    f"{optimiseur.param_groups[0]['lr']:.2e}",
                                    f"{mesures['wer']:.4f}", f"{mesures['cer']:.4f}"])
        journal(f"pas {etat['pas']:>7} · époque {etat['epoque']:>3} · perte "
                f"{perte_moyenne:.3f} · WER dev {mesures['wer']:.1%} · CER dev "
                f"{mesures['cer']:.1%}")
        if mesures["wer"] < etat["meilleur_wer"]:
            etat["meilleur_wer"] = mesures["wer"]
            sauver(sortie / "meilleur.pt")
        sauver(dernier)

    def chargeur_de(liste, graine):
        echantillonneur = EchantillonneurParDuree([e.duree for e in liste], secondes_par_lot,
                                                  graine=graine)
        return DataLoader(JeuAudio(liste, tokenizer), batch_sampler=echantillonneur,
                          collate_fn=assembler, num_workers=travailleurs,
                          persistent_workers=travailleurs > 0)

    def lots():
        """Enchaîne la phase d'amorçage puis les passages sur toutes les données."""
        if amorce_enonces and etat["pas"] < amorce_pas:
            # Amorçage : un petit sous-ensemble vu en boucle. Une CTC entraînée de zéro
            # reste sinon bloquée sur un palier où elle ignore l'audio ; mémoriser
            # quelques centaines d'énoncés lui apprend d'abord à aligner son et texte.
            import random
            petit = random.Random(0).sample(exemples, min(amorce_enonces, len(exemples)))
            journal(f"Amorçage sur {len(petit)} énoncés jusqu'au pas {amorce_pas}")
            chargeur = chargeur_de(petit, 0)
            while etat["pas"] < amorce_pas:
                for lot in chargeur:
                    yield lot
                    if etat["pas"] >= amorce_pas:
                        break
            del chargeur
        # Graine liée au pas : après une reprise, l'ordre des lots change, ce qui évite
        # de retomber indéfiniment sur le même lot si l'un d'eux fait planter le GPU.
        chargeur = chargeur_de(exemples, etat["pas"])
        while etat["epoque"] < epoques:
            yield from chargeur
            etat["epoque"] += 1

    cumul, nb, debut = 0.0, 0, time.perf_counter()
    modele.train()
    augmentation.train()
    for mels, longueurs, cibles, longueurs_cibles in lots():
        etat["pas"] += 1
        for groupe in optimiseur.param_groups:
            groupe["lr"] = taux_apprentissage(etat["pas"], lr, montee)
        if augmenter and etat["pas"] > debut_augmentation:
            mels = augmentation(mels, longueurs)
        mels = mels.to(appareil)
        with torch.autocast("cuda", enabled=precision_mixte):
            log_probas, longueurs_sortie, milieu = modele(mels, longueurs.to(appareil),
                                                          intermediaire=True)
        perte = perte_ctc(log_probas, longueurs_sortie, cibles, longueurs_cibles)
        if poids_intermediaire:
            perte = ((1 - poids_intermediaire) * perte + poids_intermediaire
                     * perte_ctc(milieu, longueurs_sortie, cibles, longueurs_cibles))
        optimiseur.zero_grad(set_to_none=True)
        echelle.scale(perte).backward()
        echelle.unscale_(optimiseur)
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 5.0)
        echelle.step(optimiseur)
        echelle.update()
        cumul, nb = cumul + perte.item(), nb + 1
        if etat["pas"] % 100 == 0:
            journal(f"pas {etat['pas']:>7} · perte {cumul / nb:.3f} · "
                    f"{(time.perf_counter() - debut) / nb:.2f} s/pas")
        if etat["pas"] % sauver_tous_les == 0:
            sauver(dernier)
        if etat["pas"] % eval_tous_les == 0:
            evaluer_et_sauver(cumul / nb)
            cumul, nb, debut = 0.0, 0, time.perf_counter()
        if pas_max is not None and etat["pas"] >= pas_max:
            break
    if nb:                      # des pas ont eu lieu depuis la dernière évaluation
        evaluer_et_sauver(cumul / nb)
    return {**etat, "sortie": sortie, "parametres": modele.nb_parametres()}


def main(arguments=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--donnees", type=Path, default=Path("donnees"))
    p.add_argument("--sortie", type=Path, default=Path("entrainements/v1"))
    p.add_argument("--taille", choices=["petit", "moyen"], default="moyen")
    p.add_argument("--epoques", type=int, default=60)
    p.add_argument("--secondes-par-lot", type=float, default=160.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--montee", type=int, default=10_000)
    p.add_argument("--eval-tous-les", type=int, default=2000)
    p.add_argument("--sauver-tous-les", type=int, default=250)
    p.add_argument("--debut-augmentation", type=int, default=0,
                   help="n'appliquer SpecAugment qu'après ce nombre de pas")
    p.add_argument("--poids-intermediaire", type=float, default=0.3)
    p.add_argument("--amorce-enonces", type=int, default=0,
                   help="taille du sous-ensemble vu en boucle au début (0 = pas d'amorçage)")
    p.add_argument("--amorce-pas", type=int, default=0, help="durée de l'amorçage, en pas")
    p.add_argument("--pas-max", type=int)
    p.add_argument("--travailleurs", type=int, default=4)
    p.add_argument("--appareil", default="auto", help="auto, cpu, cuda ou directml")
    p.add_argument("--reprendre", action="store_true")
    entrainer(**vars(p.parse_args(arguments)))


if __name__ == "__main__":
    main()
