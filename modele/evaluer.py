"""Évaluation du modèle maison : WER et CER sur un manifeste, et comparaison avec Whisper.

    python -m modele.evaluer --checkpoint entrainements/v1/meilleur.pt \
        --manifeste donnees/test.tsv --comparer-whisper small
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .decodage import decoder_glouton, taux_erreur
from .donnees import EchantillonneurParDuree, JeuAudio, assembler, lire_manifeste
from .inference import charger_checkpoint


@torch.no_grad()
def evaluer_modele(modele, tokenizer, exemples, appareil="cpu",
                   secondes_par_lot: float = 120.0) -> dict:
    """Décodage glouton de tous les exemples → {"wer", "cer", "n", "exemples"}."""
    etait_en_entrainement = modele.training
    modele.eval()
    echantillonneur = EchantillonneurParDuree([e.duree for e in exemples],
                                              secondes_par_lot, melanger=False)
    chargeur = DataLoader(JeuAudio(exemples, tokenizer), batch_sampler=echantillonneur,
                          collate_fn=assembler)
    references, hypotheses = [], []
    for indices, (mels, longueurs, _, _) in zip(echantillonneur.lots, chargeur):
        log_probas, longueurs_sortie = modele(mels.to(appareil), longueurs.to(appareil))
        hypotheses += decoder_glouton(log_probas.cpu(), longueurs_sortie.cpu(), tokenizer)
        references += [exemples[i].texte for i in indices]
    modele.train(etait_en_entrainement)
    return {"wer": taux_erreur(references, hypotheses, "mot"),
            "cer": taux_erreur(references, hypotheses, "caractere"),
            "n": len(references),
            "exemples": list(zip(references, hypotheses))[:10]}


def evaluer_whisper(exemples, taille: str) -> dict:
    """Même mesure avec le moteur Pro de l'app, sur les mêmes fichiers."""
    from vocalis.moteurs import MoteurPro

    from .donnees import charger_signal

    moteur = MoteurPro(taille)
    references = [e.texte for e in exemples]
    hypotheses = [moteur.transcrire(charger_signal(e.chemin), "fr").texte for e in exemples]
    return {"wer": taux_erreur(references, hypotheses, "mot"),
            "cer": taux_erreur(references, hypotheses, "caractere"), "n": len(exemples)}


def main(arguments=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifeste", type=Path, required=True)
    p.add_argument("--limite", type=int, help="n'évaluer que les N premiers énoncés")
    p.add_argument("--comparer-whisper", metavar="TAILLE",
                   help="mesurer aussi Whisper (base, small…) sur les mêmes fichiers")
    a = p.parse_args(arguments)

    exemples = lire_manifeste(a.manifeste)[: a.limite]
    modele, tokenizer, infos = charger_checkpoint(a.checkpoint)
    t0 = time.perf_counter()
    maison = evaluer_modele(modele, tokenizer, exemples)
    duree_audio = sum(e.duree for e in exemples)
    maison["facteur_temps_reel"] = (time.perf_counter() - t0) / max(duree_audio, 1e-9)
    resultats = {"manifeste": str(a.manifeste), "heures": duree_audio / 3600,
                 "pas": infos.get("pas"), "parametres": modele.nb_parametres(),
                 "maison": maison}
    print(f"Modèle maison  : WER {maison['wer']:.1%} · CER {maison['cer']:.1%} "
          f"sur {maison['n']} énoncés")
    for reference, hypothese in maison["exemples"][:5]:
        print(f"   réf : {reference}\n   hyp : {hypothese}")
    if a.comparer_whisper:
        whisper = evaluer_whisper(exemples, a.comparer_whisper)
        resultats[f"whisper_{a.comparer_whisper}"] = whisper
        print(f"Whisper {a.comparer_whisper:<7}: WER {whisper['wer']:.1%} · "
              f"CER {whisper['cer']:.1%}")
    sortie = a.checkpoint.with_name(f"resultats_{a.manifeste.stem}.json")
    sortie.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats écrits dans {sortie}")
    return resultats


if __name__ == "__main__":
    main()
