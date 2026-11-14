# Vocalis — dictée vocale pour Windows

Maintiens un raccourci, parle, relâche : le texte s'écrit dans la fenêtre active.
Tout tourne en local, hors ligne.

Vocalis embarque deux moteurs de reconnaissance vocale :

| Moteur | Ce que c'est | Usage |
|---|---|---|
| **Pro** | Whisper (OpenAI) via faster-whisper, sur CPU | dictée de tous les jours |
| **Maison** | un Conformer + CTC écrit et entraîné de zéro dans ce repo (`modele/`) | projet pédagogique, sélectionnable dans l'app une fois entraîné |

Le projet est né comme démonstration « Projet 25 – Reconnaissance vocale » ; la démo
Gradio et le notebook des 7 étapes sont toujours là (voir plus bas).

## Installation

```bash
py -3.12 -m venv .venv
```
```bash
.venv\Scripts\python -m pip install -r requirements-app.txt
```

Au premier lancement, le modèle Whisper choisi est téléchargé depuis Hugging Face
(`small` ≈ 500 Mo), puis tout fonctionne hors ligne.

## Lancer l'app

```bash
.venv\Scripts\python -m vocalis
```

Ou double-clic sur `lancer.pyw` (sans console). L'icône apparaît dans la zone de
notification ; un clic ouvre la fenêtre.

- **Dicter** : maintiens `Ctrl+Espace`, parle, relâche. Une pastille en bas de l'écran
  montre l'écoute puis la transcription ; le texte est collé là où se trouve le curseur.
- **Historique** : toutes les dictées, avec copie en un clic.
- **Fichier audio** : transcrit un fichier (wav, mp3, flac, ogg) et l'exporte en `.srt`.
- **Réglages** : raccourci, mode (maintenir / appuyer pour démarrer-arrêter), moteur,
  taille du modèle Whisper, langue, micro, collage automatique, démarrage avec Windows.

Les réglages et l'historique sont dans `%APPDATA%\Vocalis`. Le presse-papiers est
restauré après chaque collage.

## Le modèle maison

Tout est écrit à la main avec les briques de base de PyTorch, sans poids pré-entraînés :

| Fichier | Rôle |
|---|---|
| `modele/tokenizer.py` | tokenizer BPE (sous-mots) |
| `modele/features.py` | spectrogramme log-Mel, banc de filtres Mel recalculé à la main |
| `modele/conformer.py` | encodeur Conformer (attention à positions relatives + convolutions) et tête CTC |
| `modele/decodage.py` | décodage CTC glouton, WER et CER |
| `modele/donnees.py` | jeu de données, lots par durée, SpecAugment |
| `modele/preparer.py` | téléchargement des données d'entraînement |
| `modele/entrainer.py` | boucle d'entraînement avec reprise |
| `modele/evaluer.py` | évaluation, comparaison avec Whisper |
| `modele/inference.py` | chargement d'un checkpoint et transcription |

```bash
.venv\Scripts\python -m pip install -r requirements-entrainement.txt
```
```bash
.venv\Scripts\python -m modele.preparer --source mls --heures 100 --sortie donnees
```
```bash
.venv\Scripts\python -m modele.entrainer --donnees donnees --sortie entrainements/v1 --taille petit
```
```bash
.venv\Scripts\python -m modele.evaluer --checkpoint entrainements/v1/meilleur.pt --manifeste donnees/test.tsv --comparer-whisper small
```

Les données sont Multilingual LibriSpeech français (livres audio lus, licence CC BY 4.0),
téléchargées en flux : 100 h occupent environ 1,7 Go. Un entraînement interrompu se
reprend avec `--reprendre`.

Pour utiliser le modèle dans l'app, copie `meilleur.pt` vers
`%APPDATA%\Vocalis\modele_maison.pt`, puis choisis « Modèle maison » dans les réglages.
Tant que ce fichier n'existe pas, l'app se replie sur Whisper et le dit.

**État actuel : le modèle maison n'a pas encore été entraîné sur de vraies données.**
Le code est vérifié par des tests (mémorisation d'un petit corpus, parité du log-Mel avec
librosa, mini-entraînement avec reprise), mais aucun WER réel n'a été mesuré. Les tailles
disponibles : `petit` (8,4 M de paramètres) et `moyen` (26,3 M).

## Tests

```bash
.venv\Scripts\python -m pytest tests -q
```

## La démo d'origine (Gradio + notebook)

| Fichier | Rôle |
|---|---|
| `app.py` | application Gradio : importer un audio (ou parler au micro) → transcription Whisper |
| `demo_etapes.ipynb` | notebook qui déroule les 7 étapes signal → texte, avec figures et WER |
| `stt/` | lecture audio, log-Mel, Whisper via Transformers, WER, SRT |
| `samples/` | enregistrements de test + `references.json` |
| `RECHERCHE_Speech_Recognition.md` | dossier de recherche |

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```
```bash
.venv\Scripts\python app.py
```

Puis ouvrir http://127.0.0.1:7860.
