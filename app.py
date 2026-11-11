"""Démonstration Speech-to-Text — Projet 25 (Audio).

Lancement :  .venv\\Scripts\\python app.py   puis ouvrir http://127.0.0.1:7860
"""

import json
import tempfile
from pathlib import Path

import gradio as gr
from transformers.utils import logging as hf_logging

from stt.audio import charger_audio, infos_fichier
from stt.evaluate import wer
from stt.features import figure_signal
from stt.transcribe import (MODELE_DEFAUT, MODELES, NOMS_LANGUES, charger_modele,
                            transcrire, vers_srt)

hf_logging.set_verbosity_error()

ICI = Path(__file__).parent
SAMPLES = ICI / "samples"
REFERENCES = json.loads((SAMPLES / "references.json").read_text(encoding="utf-8"))

LANGUES = {"Détection automatique": None, "Français": "fr", "Anglais": "en",
           "Arabe": "ar", "Espagnol": "es"}
TACHES = {"Transcrire": "transcribe", "Traduire en anglais": "translate"}


def nom_langue(code: str) -> str:
    return NOMS_LANGUES.get(code, code)


def traiter(chemin, nom_modele, choix_langue, choix_tache, reference):
    if not chemin:
        raise gr.Error("Aucun audio reçu : importe un fichier ou enregistre-toi au micro "
                       "(après avoir choisi un exemple, attends qu'il apparaisse dans le lecteur).")

    # Étapes 1-2 : lecture + prétraitement (mono, 16 kHz)
    infos = infos_fichier(chemin)
    audio = charger_audio(chemin)
    figure = figure_signal(audio, titre=f"Forme d'onde — {Path(chemin).name}")

    # Étapes 3-7 : log-Mel → encodeur → décodeur → texte
    res = transcrire(audio, nom_modele, LANGUES[choix_langue], TACHES[choix_tache])

    probas = " · ".join(f"{nom_langue(c)} {p:.1%}" for c, p in res.probas_langues[:3])
    details = (
        f"**Langue utilisée :** {nom_langue(res.langue)}  \n"
        f"**Probabilités estimées par le modèle :** {probas}  \n"
        f"**Fichier d'origine :** {infos['frequence']} Hz, {infos['canaux']} canal(aux), "
        f"{infos['duree']:.1f} s → converti en mono 16 kHz "
        f"({len(audio):_} échantillons)  \n".replace("_", " ") +
        f"**Temps de calcul :** {res.temps_calcul:.1f} s "
        f"(facteur temps réel {res.facteur_temps_reel:.2f})"
    )

    score = ""
    if reference and reference.strip():
        w = wer(reference, res.texte)
        score = (f"### WER : {w['wer']:.1%}\n"
                 f"{w['substitutions']} substitution(s), {w['suppressions']} "
                 f"suppression(s), {w['insertions']} insertion(s) "
                 f"sur {w['mots_reference']} mots de référence.")

    segments = [[f"{d or 0:.2f}", f"{(f if f is not None else res.duree_audio):.2f}", t]
                for d, f, t in res.segments]

    srt = Path(tempfile.gettempdir()) / f"{Path(chemin).stem}.srt"
    srt.write_text(vers_srt(res.segments, res.duree_audio), encoding="utf-8")

    return res.texte, details, score, segments, str(srt), figure


def reference_exemple(chemin):
    """Remplit automatiquement la référence quand on choisit un exemple fourni."""
    return REFERENCES.get(Path(chemin).name, "") if chemin else ""


with gr.Blocks(title="Reconnaissance vocale — Whisper") as demo:
    gr.Markdown(
        "# Reconnaissance vocale (Speech-to-Text) avec Whisper\n"
        "Importe un enregistrement ou parle dans le micro : le signal est converti "
        "en mono 16 kHz, transformé en spectrogramme log-Mel, puis Whisper "
        "(encodeur-décodeur Transformer) génère le texte."
    )
    with gr.Row():
        with gr.Column(scale=1):
            entree = gr.Audio(sources=["upload", "microphone"], type="filepath",
                              label="Enregistrement")
            modele = gr.Dropdown(list(MODELES), value=MODELE_DEFAUT, label="Modèle Whisper")
            langue = gr.Dropdown(list(LANGUES), value="Détection automatique",
                                 label="Langue")
            tache = gr.Radio(list(TACHES), value="Transcrire", label="Tâche")
            reference = gr.Textbox(lines=3, label="Texte de référence (optionnel, pour le WER)")
            bouton = gr.Button("Transcrire", variant="primary")
        with gr.Column(scale=1):
            texte = gr.Textbox(lines=6, label="Transcription", buttons=["copy"])
            details = gr.Markdown()
            score = gr.Markdown()
            segments = gr.Dataframe(headers=["Début (s)", "Fin (s)", "Texte"],
                                    label="Segments horodatés", wrap=True)
            fichier_srt = gr.File(label="Sous-titres (.srt)")

    figure = gr.Plot(label="Signal et spectrogramme")

    gr.Examples(
        examples=[[str(SAMPLES / f)] for f in REFERENCES if (SAMPLES / f).exists()],
        inputs=[entree], label="Enregistrements de test",
    )
    entree.change(reference_exemple, inputs=entree, outputs=reference)
    bouton.click(traiter, inputs=[entree, modele, langue, tache, reference],
                 outputs=[texte, details, score, segments, fichier_srt, figure])

if __name__ == "__main__":
    charger_modele(MODELE_DEFAUT)  # chargé une fois au démarrage, pas au premier clic
    demo.launch()
