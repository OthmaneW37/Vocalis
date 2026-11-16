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


HEURES_MLS = {"train": 1076.0, "dev": 10.0, "test": 10.0}    # tailles du corpus français


def _fragments_mls(partie: str) -> list[str]:
    from huggingface_hub import HfApi

    fichiers = HfApi().list_repo_files("facebook/multilingual_librispeech",
                                       repo_type="dataset")
    return sorted(f for f in fichiers if f.startswith(f"french/{partie}-")
                  and f.endswith(".parquet"))


def _extraire_fragment(fragment: str, numero: int, dossier: Path, pas: int,
                       quota_s: float) -> list[Exemple]:
    """Parcourt un fichier du corpus en flux et garde un énoncé sur `pas`.

    Les énoncés sont rangés locuteur par locuteur : en piocher un sur `pas` dans
    tout le fichier donne beaucoup plus de voix différentes que d'en prendre le début.
    """
    import soundfile as sf
    from datasets import Audio, load_dataset

    flux = load_dataset("facebook/multilingual_librispeech",
                        data_files={"x": [fragment]}, split="x", streaming=True)
    flux = flux.cast_column("audio", Audio(decode=False))
    exemples, total = [], 0.0
    for i, ligne in enumerate(flux):
        if i % pas:
            continue
        octets = ligne["audio"]["bytes"]
        texte = ligne.get("transcript") or ligne.get("text") or ""
        duree = ligne.get("audio_duration")
        if duree is None:
            infos = sf.info(io.BytesIO(octets))
            duree = infos.frames / infos.samplerate
        if not _garder(float(duree), texte):
            continue
        extension = Path(ligne["audio"].get("path") or "x.opus").suffix or ".opus"
        fichier = dossier / f"f{numero:02d}_{i:06d}{extension}"
        fichier.write_bytes(octets)
        exemples.append(Exemple(fichier, float(duree), normaliser_texte(texte)))
        total += float(duree)
        if total >= quota_s:
            break
    return exemples


def depuis_mls(sortie: Path, heures: dict[str, float], journal=print,
               paralleles: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    for partie in ("dev", "test", "train"):
        fragments = _fragments_mls(partie)
        dossier = sortie / "audio" / partie
        dossier.mkdir(parents=True, exist_ok=True)
        pas = max(1, int(HEURES_MLS[partie] / heures[partie]))
        quota_s = heures[partie] * 3600 / len(fragments)
        exemples = []
        with ThreadPoolExecutor(paralleles) as groupe:
            taches = [groupe.submit(_extraire_fragment, f, n, dossier, pas, quota_s)
                      for n, f in enumerate(fragments)]
            for faites, tache in enumerate(as_completed(taches), 1):
                exemples += tache.result()
                ecrire_manifeste(sortie / f"{partie}.tsv", sorted(exemples, key=lambda e: e.chemin))
                journal(f"{partie} : fichier {faites}/{len(fragments)} · "
                        f"{sum(e.duree for e in exemples) / 3600:.1f} h", flush=True)
        journal(f"{partie}.tsv : {len(exemples)} énoncés, "
                f"{sum(e.duree for e in exemples) / 3600:.2f} h", flush=True)


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
