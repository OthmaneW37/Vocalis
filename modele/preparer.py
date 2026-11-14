"""Préparation des données d'entraînement → train.tsv, dev.tsv, test.tsv.

Deux sources :

  --source mls        Multilingual LibriSpeech français (livres audio lus, licence
                      CC BY 4.0), téléchargé en flux depuis Hugging Face : seules
                      les heures demandées sont écrites sur le disque.
  --source dossier    Un dossier Common Voice déjà extrait (train.tsv, dev.tsv,
                      test.tsv et clips/), passé avec --dossier.

    python -m modele.preparer --source mls --heures 100 --sortie donnees
"""

import argparse
import csv
import io
from pathlib import Path

from .donnees import Exemple, ecrire_manifeste
from .tokenizer import normaliser_texte

DUREE_MIN_S, DUREE_MAX_S = 1.0, 20.0


def _garder(duree: float, texte: str) -> bool:
    return DUREE_MIN_S <= duree <= DUREE_MAX_S and bool(normaliser_texte(texte))


def depuis_mls(sortie: Path, heures: dict[str, float], journal=print) -> None:
    import soundfile as sf
    from datasets import Audio, load_dataset

    for partie, nom in (("train", "train"), ("dev", "dev"), ("test", "test")):
        flux = load_dataset("facebook/multilingual_librispeech", "french",
                            split=nom, streaming=True)
        flux = flux.cast_column("audio", Audio(decode=False))
        if partie == "train":           # mélange des livres : plus de locuteurs différents
            flux = flux.shuffle(seed=0, buffer_size=2000)
        dossier = sortie / "audio" / partie
        dossier.mkdir(parents=True, exist_ok=True)
        exemples, total, cible = [], 0.0, heures[partie] * 3600
        for i, ligne in enumerate(flux):
            octets = ligne["audio"]["bytes"]
            texte = ligne.get("transcript") or ligne.get("text") or ""
            duree = ligne.get("audio_duration")
            if duree is None:
                infos = sf.info(io.BytesIO(octets))
                duree = infos.frames / infos.samplerate
            if not _garder(float(duree), texte):
                continue
            extension = Path(ligne["audio"].get("path") or "x.opus").suffix or ".opus"
            fichier = dossier / f"{partie}_{i:07d}{extension}"
            fichier.write_bytes(octets)
            exemples.append(Exemple(fichier, float(duree), normaliser_texte(texte)))
            total += float(duree)
            if len(exemples) % 500 == 0:
                journal(f"{partie} : {total / 3600:.1f} h / {heures[partie]:.1f} h")
            if total >= cible:
                break
        ecrire_manifeste(sortie / f"{partie}.tsv", exemples)
        journal(f"{partie}.tsv : {len(exemples)} énoncés, {total / 3600:.2f} h")


def depuis_common_voice(dossier: Path, sortie: Path, heures: dict[str, float],
                        journal=print) -> None:
    import soundfile as sf

    sortie.mkdir(parents=True, exist_ok=True)
    for partie in ("train", "dev", "test"):
        exemples, total, cible = [], 0.0, heures[partie] * 3600
        with (dossier / f"{partie}.tsv").open(encoding="utf-8", newline="") as f:
            for ligne in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                clip = dossier / "clips" / ligne["path"]
                try:
                    infos = sf.info(str(clip))
                except Exception:
                    continue                    # clip absent ou illisible
                duree = infos.frames / infos.samplerate
                if not _garder(duree, ligne["sentence"]):
                    continue
                exemples.append(Exemple(clip.resolve(), duree,
                                        normaliser_texte(ligne["sentence"])))
                total += duree
                if total >= cible:
                    break
        ecrire_manifeste(sortie / f"{partie}.tsv", exemples)
        journal(f"{partie}.tsv : {len(exemples)} énoncés, {total / 3600:.2f} h")


def main(arguments=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=["mls", "dossier"], default="mls")
    p.add_argument("--dossier", type=Path, help="dossier Common Voice extrait")
    p.add_argument("--sortie", type=Path, default=Path("donnees"))
    p.add_argument("--heures", type=float, default=100.0, help="heures d'entraînement")
    p.add_argument("--heures-dev", type=float, default=1.0)
    p.add_argument("--heures-test", type=float, default=2.0)
    a = p.parse_args(arguments)
    heures = {"train": a.heures, "dev": a.heures_dev, "test": a.heures_test}
    a.sortie.mkdir(parents=True, exist_ok=True)
    if a.source == "mls":
        depuis_mls(a.sortie, heures)
    else:
        if a.dossier is None:
            p.error("--dossier est requis avec --source dossier")
        depuis_common_voice(a.dossier, a.sortie, heures)


if __name__ == "__main__":
    main()
