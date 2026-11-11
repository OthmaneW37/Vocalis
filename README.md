# Projet 25 – Reconnaissance vocale (Speech-to-Text)

Démonstration de transcription automatique avec **Whisper** (OpenAI), **Hugging Face Transformers** et **Librosa**.

| Fichier | Rôle |
|---|---|
| `app.py` | Application Gradio : importer un audio (ou parler au micro) → transcription |
| `demo_etapes.ipynb` | Notebook qui déroule les 7 étapes signal → texte, avec figures, + évaluation WER |
| `stt/audio.py` | Étapes 1-2 : lecture, mono, rééchantillonnage 16 kHz |
| `stt/features.py` | Étape 3 : spectrogramme log-Mel (ré-implémenté à la main) + figures |
| `stt/transcribe.py` | Étapes 4-7 : modèle Whisper, détection de langue, transcription, SRT |
| `stt/evaluate.py` | WER (Word Error Rate) |
| `samples/` | Enregistrements de test + `references.json` (textes exacts) |
| `RECHERCHE_Speech_Recognition.md` | Dossier de recherche (théorie, choix, plan de présentation) |

## Installation (une fois)

```bash
python -m venv --system-site-packages .venv
```
```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

Les modèles `tiny`, `base` et `small` sont téléchargés depuis Hugging Face au premier usage (~1,5 Go au total), puis tout fonctionne **hors ligne**.

## Lancer l'application

```bash
.venv\Scripts\python app.py
```

Puis ouvrir http://127.0.0.1:7860. Choisir un enregistrement de test (la référence se remplit seule et le WER est calculé) ou enregistrer sa voix au micro.

## Ouvrir le notebook

Dans VS Code, ouvrir `demo_etapes.ipynb` et choisir le noyau **.venv** (ou « Python (projet STT) »). Il est déjà exécuté : toutes les sorties et figures sont visibles sans le relancer.

## Remplacer les enregistrements de test

Les fichiers fournis sont générés par la synthèse vocale de Windows (voix trop « propre »). Pour la soutenance, enregistre ta propre voix en `.wav` ou `.mp3` dans `samples/` et ajoute le texte exact dans `samples/references.json`. Le format `.m4a` n'est pas lu (ffmpeg absent) : convertis-le d'abord en `.wav`.
